"""Metrics for online open-set and risk-constrained decision outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


Prediction = dict


SUMMARY_COLUMNS = [
    "method_name",
    "far_budget",
    "accuracy",
    "accuracy_active",
    "accuracy_with_defer_as_failure",
    "FAR",
    "FAR_active",
    "FRR",
    "FRR_active",
    "FRR_with_defer_as_failure_for_genuine",
    "TAR",
    "defer_rate",
    "automation_rate",
    "n_trials",
    "n_genuine",
    "n_impostor",
    "verification_accuracy",
    "identification_error_rate",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _is_accept(prediction: Prediction) -> bool:
    return str(prediction["decision"]) == "accept"


def _is_reject(prediction: Prediction) -> bool:
    return str(prediction["decision"]) == "reject"


def _is_defer(prediction: Prediction) -> bool:
    return str(prediction["decision"]) == "defer"


def _chou_correct(prediction: Prediction) -> bool:
    if _is_defer(prediction):
        return False
    if _is_accept(prediction):
        return bool(prediction["is_genuine"])
    return not bool(prediction["identity_exists"])


def _verification_correct(prediction: Prediction) -> bool:
    if _is_defer(prediction):
        return False
    if _is_accept(prediction):
        return bool(prediction["is_genuine"])
    return not bool(prediction["is_genuine"])


def summarize_predictions(predictions: Iterable[Prediction]) -> Dict[str, float]:
    rows = list(predictions)
    n_trials = len(rows)
    genuine = [row for row in rows if bool(row["is_genuine"])]
    impostor = [row for row in rows if not bool(row["is_genuine"])]
    non_deferred = [row for row in rows if not _is_defer(row)]
    active_genuine = [row for row in genuine if not _is_defer(row)]
    active_impostor = [row for row in impostor if not _is_defer(row)]

    accepted_impostor = sum(1 for row in impostor if _is_accept(row))
    active_accepted_impostor = sum(1 for row in active_impostor if _is_accept(row))
    rejected_genuine = sum(1 for row in genuine if _is_reject(row))
    active_rejected_genuine = sum(1 for row in active_genuine if _is_reject(row))
    deferred_genuine = sum(1 for row in genuine if _is_defer(row))
    deferred = sum(1 for row in rows if _is_defer(row))

    correct = sum(1 for row in rows if _chou_correct(row))
    active_correct = sum(1 for row in non_deferred if _chou_correct(row))
    verification_correct = sum(1 for row in rows if _verification_correct(row))
    identification_errors = sum(
        1
        for row in rows
        if _is_accept(row)
        and bool(row["identity_exists"])
        and not bool(row["is_genuine"])
    )

    frr = safe_divide(rejected_genuine, len(genuine))
    return {
        "accuracy": safe_divide(correct, n_trials),
        "accuracy_active": safe_divide(active_correct, len(non_deferred)),
        "accuracy_with_defer_as_failure": safe_divide(active_correct, n_trials),
        "FAR": safe_divide(accepted_impostor, len(impostor)),
        "FAR_active": safe_divide(active_accepted_impostor, len(active_impostor)),
        "FRR": frr,
        "FRR_active": safe_divide(active_rejected_genuine, len(active_genuine)),
        "FRR_with_defer_as_failure_for_genuine": safe_divide(
            rejected_genuine + deferred_genuine,
            len(genuine),
        ),
        "TAR": 1.0 - frr,
        "defer_rate": safe_divide(deferred, n_trials),
        "automation_rate": 1.0 - safe_divide(deferred, n_trials),
        "n_trials": n_trials,
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "verification_accuracy": safe_divide(verification_correct, n_trials),
        "identification_error_rate": safe_divide(identification_errors, n_trials),
    }


def summarize_by_method(predictions: Iterable[Prediction]) -> List[Dict[str, float]]:
    groups: Dict[Tuple[str, str], List[Prediction]] = defaultdict(list)
    for prediction in predictions:
        far_budget = prediction.get("far_budget")
        far_key = "" if far_budget is None else str(far_budget)
        groups[(str(prediction["method_name"]), far_key)].append(prediction)

    summaries = []
    for (method_name, far_budget), rows in sorted(groups.items()):
        summary = summarize_predictions(rows)
        summary["method_name"] = method_name
        summary["far_budget"] = far_budget
        summaries.append(summary)
    return summaries


def summarize_by_far_budget(predictions: Iterable[Prediction]) -> List[Dict[str, float]]:
    groups: Dict[Tuple[str, str], List[Prediction]] = defaultdict(list)
    for prediction in predictions:
        far_budget = prediction.get("far_budget")
        if far_budget is not None:
            groups[(str(far_budget), str(prediction["method_name"]))].append(prediction)

    summaries = []
    for (far_budget, method_name), rows in sorted(
        groups.items(),
        key=lambda item: (float(item[0][0]), item[0][1]),
    ):
        summary = summarize_predictions(rows)
        summary["method_name"] = method_name
        summary["far_budget"] = far_budget
        summaries.append(summary)
    return summaries
