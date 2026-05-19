"""Decision policies for the Chou et al. online-registration protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


Trial = dict
Prediction = dict


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _chou_correct(trial: Trial, accepted: bool) -> bool:
    """Correctness used by the original online open-set protocol."""

    if accepted:
        return bool(trial["is_genuine"])
    return not bool(trial["identity_exists"])


def select_threshold_for_accuracy(trials: Sequence[Trial]) -> float:
    """Select one global threshold by calibration accuracy."""

    if not trials:
        return 0.0

    sorted_trials = sorted(trials, key=lambda trial: float(trial["similarity"]), reverse=True)
    epsilon = 1e-12
    threshold = float(sorted_trials[0]["similarity"]) + epsilon
    correct = sum(1 for trial in sorted_trials if not bool(trial["identity_exists"]))
    best_key = (_safe_divide(correct, len(sorted_trials)), -threshold)
    best_threshold = threshold

    index = 0
    while index < len(sorted_trials):
        score = float(sorted_trials[index]["similarity"])
        while index < len(sorted_trials) and float(sorted_trials[index]["similarity"]) == score:
            trial = sorted_trials[index]
            reject_correct = not bool(trial["identity_exists"])
            accept_correct = bool(trial["is_genuine"])
            correct += int(accept_correct) - int(reject_correct)
            index += 1
        key = (_safe_divide(correct, len(sorted_trials)), -score)
        if key > best_key:
            best_key = key
            best_threshold = score

    all_accept_threshold = float(sorted_trials[-1]["similarity"]) - epsilon
    key = (_safe_divide(correct, len(sorted_trials)), -all_accept_threshold)
    if key > best_key:
        best_threshold = all_accept_threshold
    return best_threshold


def select_threshold_for_far_budget(trials: Sequence[Trial], far_budget: float) -> float:
    """Select the lowest-risk threshold that maximizes TAR under a FAR budget."""

    if not trials:
        return 0.0

    sorted_trials = sorted(trials, key=lambda trial: float(trial["similarity"]), reverse=True)
    epsilon = 1e-12
    total_genuine = sum(1 for trial in sorted_trials if bool(trial["is_genuine"]))
    total_impostor = len(sorted_trials) - total_genuine

    threshold = float(sorted_trials[0]["similarity"]) + epsilon
    correct = sum(1 for trial in sorted_trials if not bool(trial["identity_exists"]))
    accepted_genuine = 0
    accepted_impostor = 0
    best_key = (
        _safe_divide(accepted_genuine, total_genuine),
        _safe_divide(correct, len(sorted_trials)),
        -threshold,
    )
    best_threshold = threshold

    index = 0
    while index < len(sorted_trials):
        score = float(sorted_trials[index]["similarity"])
        while index < len(sorted_trials) and float(sorted_trials[index]["similarity"]) == score:
            trial = sorted_trials[index]
            if bool(trial["is_genuine"]):
                accepted_genuine += 1
            else:
                accepted_impostor += 1
            reject_correct = not bool(trial["identity_exists"])
            accept_correct = bool(trial["is_genuine"])
            correct += int(accept_correct) - int(reject_correct)
            index += 1

        far = _safe_divide(accepted_impostor, total_impostor)
        if far <= far_budget:
            key = (
                _safe_divide(accepted_genuine, total_genuine),
                _safe_divide(correct, len(sorted_trials)),
                -score,
            )
            if key > best_key:
                best_key = key
                best_threshold = score

    all_accept_threshold = float(sorted_trials[-1]["similarity"]) - epsilon
    far = _safe_divide(accepted_impostor, total_impostor)
    if far <= far_budget:
        key = (
            _safe_divide(accepted_genuine, total_genuine),
            _safe_divide(correct, len(sorted_trials)),
            -all_accept_threshold,
        )
        if key > best_key:
            best_threshold = all_accept_threshold
    return best_threshold


@dataclass
class DecisionMethod:
    """Base interface used by the experiment runner."""

    name: str
    far_budget: Optional[float] = None
    defer_margin: Optional[float] = None

    def fit(self, calibration_df: Sequence[Trial]) -> "DecisionMethod":
        return self

    def predict(self, test_df: Sequence[Trial]) -> List[Prediction]:
        raise NotImplementedError

    def _prediction(
        self,
        trial: Trial,
        decision: str,
        threshold: Optional[float],
        threshold_accept: Optional[float],
        threshold_reject: Optional[float],
    ) -> Prediction:
        accepted = decision == "accept"
        deferred = decision == "defer"
        return {
            "trial_id": trial["trial_id"],
            "repeat_id": trial["repeat_id"],
            "split": trial["split"],
            "method_name": self.name,
            "score": float(trial["similarity"]),
            "is_genuine": bool(trial["is_genuine"]),
            "identity_exists": bool(trial["identity_exists"]),
            "decision": decision,
            "accepted": accepted,
            "deferred": deferred,
            "threshold": threshold,
            "threshold_accept": threshold_accept,
            "threshold_reject": threshold_reject,
            "far_budget": self.far_budget,
            "defer_margin": self.defer_margin,
            "person_id_a": trial["person_id_a"],
            "person_id_b": trial["person_id_b"],
            "image_a": trial["image_a"],
            "image_b": trial["image_b"],
        }


class FixedThresholdMethod(DecisionMethod):
    """One global threshold for every trial."""

    def __init__(self, threshold: Optional[float] = None) -> None:
        super().__init__(name="B0_fixed_threshold")
        self.threshold = threshold

    def fit(self, calibration_df: Sequence[Trial]) -> "FixedThresholdMethod":
        if self.threshold is None:
            self.threshold = select_threshold_for_accuracy(calibration_df)
        return self

    def predict(self, test_df: Sequence[Trial]) -> List[Prediction]:
        threshold = 0.0 if self.threshold is None else float(self.threshold)
        predictions = []
        for trial in test_df:
            decision = "accept" if float(trial["similarity"]) >= threshold else "reject"
            predictions.append(
                self._prediction(trial, decision, threshold, threshold, None)
            )
        return predictions


class ChouAdaptiveThresholdMethod(DecisionMethod):
    """Per-record adaptive threshold reproduced from the original simulator."""

    def __init__(self) -> None:
        super().__init__(name="B1_chou_adaptive_threshold")

    def predict(self, test_df: Sequence[Trial]) -> List[Prediction]:
        predictions = []
        for trial in test_df:
            threshold = float(trial["chou_threshold"])
            decision = "accept" if float(trial["similarity"]) >= threshold else "reject"
            predictions.append(
                self._prediction(trial, decision, threshold, threshold, None)
            )
        return predictions


class RiskConstrainedThresholdMethod(DecisionMethod):
    """M5: one calibrated threshold constrained by a target FAR."""

    def __init__(self, far_budget: float) -> None:
        super().__init__(
            name="Ours_M5_risk_constrained",
            far_budget=float(far_budget),
        )
        self.threshold: Optional[float] = None

    def fit(self, calibration_df: Sequence[Trial]) -> "RiskConstrainedThresholdMethod":
        self.threshold = select_threshold_for_far_budget(
            calibration_df,
            float(self.far_budget or 0.0),
        )
        return self

    def predict(self, test_df: Sequence[Trial]) -> List[Prediction]:
        threshold = 0.0 if self.threshold is None else float(self.threshold)
        predictions = []
        for trial in test_df:
            decision = "accept" if float(trial["similarity"]) >= threshold else "reject"
            predictions.append(
                self._prediction(trial, decision, threshold, threshold, None)
            )
        return predictions


class RiskConstrainedDeferMethod(DecisionMethod):
    """M6: risk-constrained accept threshold plus an uncertainty/defer zone."""

    def __init__(self, far_budget: float, defer_margin: float = 0.03) -> None:
        super().__init__(
            name="Ours_M6_risk_constrained_defer",
            far_budget=float(far_budget),
            defer_margin=float(defer_margin),
        )
        self.threshold_accept: Optional[float] = None
        self.threshold_reject: Optional[float] = None

    def fit(self, calibration_df: Sequence[Trial]) -> "RiskConstrainedDeferMethod":
        self.threshold_accept = select_threshold_for_far_budget(
            calibration_df,
            float(self.far_budget or 0.0),
        )
        self.threshold_reject = self.threshold_accept - float(self.defer_margin or 0.0)
        return self

    def predict(self, test_df: Sequence[Trial]) -> List[Prediction]:
        threshold_accept = (
            0.0 if self.threshold_accept is None else float(self.threshold_accept)
        )
        threshold_reject = (
            threshold_accept - float(self.defer_margin or 0.0)
            if self.threshold_reject is None
            else float(self.threshold_reject)
        )

        predictions = []
        for trial in test_df:
            score = float(trial["similarity"])
            if score >= threshold_accept:
                decision = "accept"
            elif score < threshold_reject:
                decision = "reject"
            else:
                decision = "defer"
            predictions.append(
                self._prediction(
                    trial,
                    decision,
                    None,
                    threshold_accept,
                    threshold_reject,
                )
            )
        return predictions


def build_methods(
    far_budgets: Iterable[float],
    defer_margin: float,
    fixed_threshold: Optional[float] = None,
) -> List[DecisionMethod]:
    methods: List[DecisionMethod] = [
        FixedThresholdMethod(threshold=fixed_threshold),
        ChouAdaptiveThresholdMethod(),
    ]
    for far_budget in far_budgets:
        methods.append(RiskConstrainedThresholdMethod(far_budget))
        methods.append(RiskConstrainedDeferMethod(far_budget, defer_margin))
    return methods
