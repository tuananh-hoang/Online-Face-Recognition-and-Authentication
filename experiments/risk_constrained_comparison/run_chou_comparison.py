"""Run Chou et al. online-protocol comparisons with risk-constrained methods."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.risk_constrained_comparison.methods import build_methods
from experiments.risk_constrained_comparison.metrics import (
    SUMMARY_COLUMNS,
    summarize_by_far_budget,
    summarize_by_method,
)


TRIAL_COLUMNS = [
    "trial_id",
    "repeat_id",
    "split",
    "person_id_a",
    "person_id_b",
    "image_a",
    "image_b",
    "is_genuine",
    "identity_exists",
    "embedding_a_path",
    "embedding_b_path",
    "similarity",
    "chou_threshold",
    "registration_id",
    "query_id",
    "identity",
    "session_id",
    "matched_index",
    "source_csv",
    "max_compare_num",
]


PREDICTION_COLUMNS = [
    "trial_id",
    "repeat_id",
    "split",
    "method_name",
    "score",
    "is_genuine",
    "identity_exists",
    "decision",
    "accepted",
    "deferred",
    "threshold",
    "threshold_accept",
    "threshold_reject",
    "far_budget",
    "defer_margin",
    "person_id_a",
    "person_id_b",
    "image_a",
    "image_b",
]


class OnlineGallery:
    """Minimal clone of the original Database logic without the tabulate dependency."""

    def __init__(
        self,
        data_num: int,
        embedding_dim: int,
        compare_num: int,
        rng: random.Random,
    ) -> None:
        self.embs = np.zeros((data_num, embedding_dim), dtype=float)
        self.labels: List[str] = []
        self.image_paths: List[str] = []
        self.embedding_refs: List[str] = []
        self.thresholds: List[float] = []
        self.indices = 0
        self.compare_num = compare_num
        self.class_dict: Dict[str, List[int]] = {}
        self.rng = rng

    def __len__(self) -> int:
        return self.indices

    def contains(self, label: str) -> bool:
        return label in self.labels

    def insert(self, label: str, emb: np.ndarray, image_path: str, embedding_ref: str) -> None:
        self.embs[self.indices] = emb
        self.labels.append(label)
        self.image_paths.append(image_path)
        self.embedding_refs.append(embedding_ref)
        self.thresholds.append(0.0)
        self.class_dict.setdefault(label, []).append(self.indices)
        self._update_thresholds(emb, label)
        self.indices += 1

    def most_similar(self, emb: np.ndarray) -> tuple[int, float]:
        similarities = self.embs[: self.indices].dot(emb)
        max_id = int(np.argmax(similarities))
        return max_id, float(similarities[max_id])

    def _sample(self, values: Iterable[int], count: int) -> List[int]:
        values = list(values)
        if count <= 0:
            return []
        if count >= len(values):
            return values
        return self.rng.sample(values, count)

    def _update_thresholds(self, emb_test: np.ndarray, label_test: str) -> None:
        max_threshold = -1.0
        all_classes = list(self.class_dict.keys())
        class_num = len(all_classes)

        if self.indices == 0:
            compare_indices: Iterable[int] = []
        elif class_num <= self.compare_num and self.indices <= self.compare_num:
            compare_indices = range(self.indices)
        elif class_num <= self.compare_num and self.indices > self.compare_num:
            per_class = int(np.floor(float(self.compare_num / class_num)))
            compare_indices_list: List[int] = []
            leftovers: List[int] = []
            selected_count = 0
            for class_name in all_classes:
                class_indices = self.class_dict[class_name]
                if len(class_indices) >= per_class:
                    selected = self._sample(class_indices, per_class)
                    compare_indices_list.extend(selected)
                    leftovers.extend(
                        index for index in class_indices if index not in selected
                    )
                    selected_count += per_class
                else:
                    compare_indices_list.extend(class_indices)
                    selected_count += len(class_indices)
            compare_indices_list.extend(
                self._sample(leftovers, self.compare_num - selected_count)
            )
            compare_indices = compare_indices_list
        else:
            compare_classes = self.rng.sample(all_classes, self.compare_num)
            compare_indices = [
                self.rng.choice(self.class_dict[class_name])
                for class_name in compare_classes
            ]

        for index in compare_indices:
            if self.labels[index] == label_test:
                continue
            new_threshold = float(np.sum(emb_test * self.embs[index]))
            if new_threshold > self.thresholds[index]:
                self.thresholds[index] = new_threshold
            if new_threshold > max_threshold:
                max_threshold = new_threshold

        if max_threshold > -1:
            self.thresholds[self.indices] = max_threshold


def parse_embedding(raw: str) -> np.ndarray:
    cleaned = raw.replace("[", " ").replace("]", " ").replace("\n", " ")
    vector = np.fromstring(cleaned, dtype=float, sep=" ")
    if vector.size == 0:
        raise ValueError("Empty embedding vector in feature CSV.")
    return vector


def read_feature_csv(filepath: Path) -> List[dict]:
    rows = []
    with filepath.open(newline="") as handle:
        reader = csv.DictReader(
            handle,
            fieldnames=["name", "features", "threshold", "filepath"],
        )
        for row_index, row in enumerate(reader):
            label = str(row["name"])
            image_path = row.get("filepath") or f"{label}/{row_index}"
            rows.append(
                {
                    "label": label,
                    "embedding": parse_embedding(row["features"]),
                    "image_path": image_path,
                    "row_index": row_index,
                    "embedding_ref": f"{filepath.as_posix()}#row={row_index}",
                }
            )
    return rows


def dataset_dir_for(dataset: str, data_dir: Optional[str]) -> Path:
    if data_dir:
        return Path(data_dir)
    return Path("data") / dataset


def feature_file_for_repeat(dataset_dir: Path, dataset: str, repeat_id: int) -> Path:
    expected = dataset_dir / f"features_{dataset}_v{repeat_id}.csv"
    if expected.exists():
        return expected
    matches = sorted(dataset_dir.glob(f"features_*_v{repeat_id}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No feature CSV for repeat {repeat_id} under {dataset_dir}"
        )
    return matches[0]


def format_vector(vector: np.ndarray) -> str:
    return " ".join(f"{value:.9g}" for value in vector)


def build_trials_for_repeat(
    feature_csv: Path,
    repeat_id: int,
    max_compare_num: int,
    max_trials: Optional[int],
    random_seed: int,
    include_embedding_vectors: bool,
) -> List[dict]:
    feature_rows = read_feature_csv(feature_csv)
    if not feature_rows:
        return []

    embedding_dim = int(feature_rows[0]["embedding"].shape[0])
    compare_num = len(feature_rows) if max_compare_num < 1 else max_compare_num
    gallery = OnlineGallery(
        data_num=len(feature_rows),
        embedding_dim=embedding_dim,
        compare_num=compare_num,
        rng=random.Random(random_seed + repeat_id),
    )
    trials = []

    for query_index, row in enumerate(feature_rows):
        if len(gallery) > 0:
            matched_index, similarity = gallery.most_similar(row["embedding"])
            matched_label = gallery.labels[matched_index]
            identity_exists = gallery.contains(row["label"])
            is_genuine = row["label"] == matched_label
            trial = {
                "trial_id": f"r{repeat_id}_t{query_index:06d}",
                "repeat_id": repeat_id,
                "split": "",
                "person_id_a": row["label"],
                "person_id_b": matched_label,
                "image_a": row["image_path"],
                "image_b": gallery.image_paths[matched_index],
                "is_genuine": is_genuine,
                "identity_exists": identity_exists,
                "embedding_a_path": row["embedding_ref"],
                "embedding_b_path": gallery.embedding_refs[matched_index],
                "similarity": similarity,
                "chou_threshold": float(gallery.thresholds[matched_index]),
                "registration_id": matched_index,
                "query_id": query_index,
                "identity": row["label"],
                "session_id": f"repeat_{repeat_id}/query_{query_index}",
                "matched_index": matched_index,
                "source_csv": feature_csv.as_posix(),
                "max_compare_num": compare_num,
            }
            if include_embedding_vectors:
                trial["embedding_a_vector"] = format_vector(row["embedding"])
                trial["embedding_b_vector"] = format_vector(gallery.embs[matched_index])
            trials.append(trial)
            if max_trials is not None and len(trials) >= max_trials:
                break

        gallery.insert(
            row["label"],
            row["embedding"],
            row["image_path"],
            row["embedding_ref"],
        )

    return trials


def assign_splits(trials: List[dict], calibration_fraction: float) -> None:
    by_repeat: Dict[int, List[dict]] = defaultdict(list)
    for trial in trials:
        by_repeat[int(trial["repeat_id"])].append(trial)

    for repeat_trials in by_repeat.values():
        repeat_trials.sort(key=lambda trial: int(trial["query_id"]))
        if len(repeat_trials) <= 1:
            calibration_count = 0
        else:
            raw_count = int(len(repeat_trials) * calibration_fraction)
            calibration_count = min(max(1, raw_count), len(repeat_trials) - 1)
        for index, trial in enumerate(repeat_trials):
            trial["split"] = "calibration" if index < calibration_count else "test"


def parse_far_budgets(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def serialize_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(filepath: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(row.get(key)) for key in fieldnames})


def metric_value(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if key.startswith("n_"):
        return str(int(number))
    return f"{number:.4f}"


def markdown_table(rows: Sequence[dict], columns: Sequence[str], limit: int = 20) -> str:
    shown = list(rows[:limit])
    if not shown:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(metric_value(row, column) for column in columns) + " |"
        for row in shown
    ]
    if len(rows) > limit:
        body.append(f"| ... | {' | '.join('' for _ in columns[1:])} |")
    return "\n".join([header, separator] + body)


def conclusion(summary_rows: Sequence[dict]) -> str:
    chou_rows = [
        row
        for row in summary_rows
        if row.get("method_name") == "B1_chou_adaptive_threshold"
    ]
    m6_rows = [
        row
        for row in summary_rows
        if row.get("method_name") == "Ours_M6_risk_constrained_defer"
    ]
    if not chou_rows or not m6_rows:
        return "The run did not produce both Chou adaptive-threshold and M6 rows, so no direct conclusion is reported."

    chou_accuracy = float(chou_rows[0]["accuracy"])
    best_m6 = max(m6_rows, key=lambda row: float(row["accuracy"]))
    best_m6_accuracy = float(best_m6["accuracy"])
    best_m6_far = float(best_m6["FAR"])
    chou_far = float(chou_rows[0]["FAR"])

    if best_m6_accuracy > chou_accuracy:
        return (
            "On this reproduced held-out split, M6 improves Chou-protocol "
            "accuracy over the adaptive-threshold baseline while also reporting "
            "FAR, FRR, and defer rate."
        )
    if best_m6_far < chou_far:
        return (
            "M6 does not improve raw Chou-protocol accuracy in this run, but it "
            "achieves a safer operating point with lower pairwise FAR and exposes "
            "uncertain cases through deferral."
        )
    return (
        "This run does not show M6 as universally better under the Chou protocol. "
        "It provides a different operating trade-off: explicit FAR control and "
        "defer reporting."
    )


def original_smoke_status(repo_root: Path) -> str:
    error_file = repo_root / "outputs/chou_reproduction/original_smoke_error.md"
    log_file = repo_root / "outputs/chou_reproduction/original_smoke.log"
    if error_file.exists():
        return (
            "The original smoke test was attempted and failed. The captured error "
            "is in `outputs/chou_reproduction/original_smoke_error.md`; in this "
            "environment the first blocker was `ModuleNotFoundError: tabulate`."
        )
    if log_file.exists():
        return (
            "The original smoke test completed; see "
            "`outputs/chou_reproduction/original_smoke.log`. On Windows, the "
            "original script first needed its expected nested result directory "
            "pre-created because `util.create_output_path()` builds an output "
            "path with a backslash but only creates the top-level `result/` "
            "directory."
        )
    return "The original smoke test was not run by this wrapper."


def write_report(
    output_dir: Path,
    summary_rows: Sequence[dict],
    args: argparse.Namespace,
    repo_root: Path,
) -> None:
    accuracy_columns = [
        "method_name",
        "far_budget",
        "accuracy",
        "accuracy_active",
        "accuracy_with_defer_as_failure",
        "verification_accuracy",
    ]
    risk_columns = [
        "method_name",
        "far_budget",
        "FAR",
        "FAR_active",
        "FRR",
        "FRR_active",
        "FRR_with_defer_as_failure_for_genuine",
        "TAR",
        "defer_rate",
        "automation_rate",
        "n_trials",
    ]
    report = f"""# Chou Protocol Risk-Constrained Comparison

## 1. Original Protocol Summary

Chou et al. evaluate an online open-set registration protocol. The gallery starts empty. For each shuffled face image, the system compares the query embedding against the current gallery, accepts it as the nearest identity if the similarity passes the method threshold, otherwise rejects it as an intruder, and then inserts the query into the gallery. The original repo uses precomputed FaceNet embeddings and inner-product similarity.

This wrapper keeps the same shuffled feature CSVs and the same online trial construction, then adds a calibration/test split so thresholds for fixed, M5, and M6 are selected without using test decisions.

## 2. Reproduction Status

{original_smoke_status(repo_root)}

The new wrapper uses the checked-in precomputed embeddings, so it does not regenerate FaceNet features.

## 3. Compared Methods

- `B0_fixed_threshold`: one global threshold selected on calibration accuracy unless `--fixed-threshold` is supplied.
- `B1_chou_adaptive_threshold`: per-gallery-record adaptive threshold mirrored from the original simulator.
- `Ours_M5_risk_constrained`: global threshold selected on calibration trials so calibration pairwise FAR is within the requested budget.
- `Ours_M6_risk_constrained_defer`: M5 accept threshold plus a defer band with `defer_margin={args.defer_margin}`.

## 4. Accuracy

`accuracy` is Chou-protocol online accuracy: true accept plus true reject over all evaluated test sessions. `verification_accuracy` is the pairwise accept/reject accuracy against the nearest gallery record.

{markdown_table(summary_rows, accuracy_columns)}

## 5. FAR / FRR / Defer

FAR and FRR are computed against whether the nearest gallery record has the same identity as the query. This makes identification errors visible as impostor accepts. The FAR budget is enforced on calibration trials; the table below reports held-out test FAR, which can exceed the nominal budget if the calibration split is not representative.

{markdown_table(summary_rows, risk_columns)}

## 6. Scope-Correct Conclusion

{conclusion(summary_rows)}

## 7. Limitations

- The fixed-threshold baseline in this wrapper is calibrated from online calibration trials, not from the paper's randomly generated 6,000 verification pairs with 10-fold CV.
- The calibration/test split is an extension required for fair M5/M6 threshold selection; it is not a byte-for-byte reproduction of the paper's all-session accuracy table.
- The reported FAR budget is a calibration constraint. Held-out test FAR should be interpreted as an empirical generalization result, not a guaranteed bound.
- The comparison is direct only when all rows use the same `--embedding-backend`, dataset CSVs, repeat IDs, and online trial construction.
- Regenerating embeddings was not attempted because this environment has Python {sys.version.split()[0]} and the original embedding path depends on old TensorFlow/OpenCV tooling.
"""
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")


def copy_reproduction_notes(output_dir: Path, repo_root: Path) -> None:
    source = repo_root / "docs/reproduction_notes.md"
    destination = output_dir / "reproduction_notes.md"
    if source.exists():
        shutil.copyfile(source, destination)


def run(args: argparse.Namespace) -> None:
    repo_root = Path.cwd()
    if args.embedding_backend not in {"facenet", "precomputed"}:
        raise ValueError(
            "Only the checked-in FaceNet/precomputed CSV embeddings are supported."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    far_budgets = parse_far_budgets(args.far_budgets)
    data_dir = dataset_dir_for(args.dataset, args.data_dir)
    all_trials: List[dict] = []
    for repeat_id in range(args.repeat):
        feature_csv = feature_file_for_repeat(data_dir, args.dataset, repeat_id)
        all_trials.extend(
            build_trials_for_repeat(
                feature_csv=feature_csv,
                repeat_id=repeat_id,
                max_compare_num=args.max_compare_num,
                max_trials=args.max_trials,
                random_seed=args.random_seed,
                include_embedding_vectors=args.include_embedding_vectors,
            )
        )

    assign_splits(all_trials, args.calibration_fraction)
    calibration_trials = [trial for trial in all_trials if trial["split"] == "calibration"]
    test_trials = [trial for trial in all_trials if trial["split"] == "test"]
    if not test_trials:
        raise ValueError("No test trials were produced; increase --max-trials or --repeat.")

    methods = build_methods(
        far_budgets=far_budgets,
        defer_margin=args.defer_margin,
        fixed_threshold=args.fixed_threshold,
    )
    predictions = []
    for method in methods:
        method.fit(calibration_trials)
        predictions.extend(method.predict(test_trials))

    trial_columns = list(TRIAL_COLUMNS)
    if args.include_embedding_vectors:
        trial_columns.extend(["embedding_a_vector", "embedding_b_vector"])
    write_csv(output_dir / "pairs_or_trials.csv", all_trials, trial_columns)
    write_csv(output_dir / "predictions_by_method.csv", predictions, PREDICTION_COLUMNS)

    summary_rows = summarize_by_method(predictions)
    budget_rows = summarize_by_far_budget(predictions)
    write_csv(output_dir / "summary_by_method.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "summary_by_far_budget.csv", budget_rows, SUMMARY_COLUMNS)
    copy_reproduction_notes(output_dir, repo_root)

    run_config = {
        "argv": sys.argv,
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "embedding_backend": args.embedding_backend,
        "repeat": args.repeat,
        "far_budgets": far_budgets,
        "defer_margin": args.defer_margin,
        "output_dir": str(output_dir),
        "max_trials": args.max_trials,
        "max_compare_num": args.max_compare_num,
        "calibration_fraction": args.calibration_fraction,
        "fixed_threshold": args.fixed_threshold,
        "random_seed": args.random_seed,
        "n_trials_total": len(all_trials),
        "n_calibration_trials": len(calibration_trials),
        "n_test_trials": len(test_trials),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, summary_rows, args, repo_root)

    print(f"Wrote {len(all_trials)} trials and {len(predictions)} predictions to {output_dir}")
    print(output_dir / "summary_by_method.csv")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lfw", help="Dataset name under data/.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional directory containing features_<dataset>_v*.csv files.",
    )
    parser.add_argument(
        "--embedding-backend",
        default="facenet",
        help="Currently supports the checked-in facenet/precomputed CSV embeddings.",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--far-budgets", default="0.01,0.02,0.03,0.05")
    parser.add_argument("--defer-margin", type=float, default=0.03)
    parser.add_argument("--output-dir", default="outputs/chou_comparison")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-compare-num", type=int, default=100)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--include-embedding-vectors",
        action="store_true",
        help="Write full embedding vectors into pairs_or_trials.csv.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
