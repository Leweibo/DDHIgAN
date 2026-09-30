from dataclasses import dataclass

import numpy as np

try:
    from python.utils.probabilities import project_probability_roundoff
except ModuleNotFoundError:  # pragma: no cover - direct script import path
    from utils.probabilities import project_probability_roundoff  # type: ignore[no-redef]


@dataclass(frozen=True)
class StepSurvival:
    """Right-continuous Kaplan-Meier step function."""

    times: np.ndarray
    survival: np.ndarray

    def at(self, query: float | np.ndarray) -> float | np.ndarray:
        query_arr = np.asarray(query, dtype=float)
        indices = np.searchsorted(self.times, query_arr, side="right") - 1
        values = np.ones(query_arr.shape, dtype=float)
        valid = indices >= 0
        values[valid] = self.survival[indices[valid]]
        return float(values) if values.ndim == 0 else values

    def before(self, query: float | np.ndarray) -> float | np.ndarray:
        query_arr = np.asarray(query, dtype=float)
        indices = np.searchsorted(self.times, query_arr, side="left") - 1
        values = np.ones(query_arr.shape, dtype=float)
        valid = indices >= 0
        values[valid] = self.survival[indices[valid]]
        return float(values) if values.ndim == 0 else values


def _validate_survival_arrays(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.ndim != 1 or event.ndim != 1 or len(time) != len(event):
        raise ValueError("time and event must be one-dimensional arrays of equal length")
    if len(time) == 0:
        raise ValueError("survival arrays must not be empty")
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("time must contain finite non-negative values")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must contain only 0 and 1")
    return time, event


def _kaplan_meier(time: np.ndarray, event: np.ndarray) -> StepSurvival:
    time, event = _validate_survival_arrays(time, event)
    event_times = np.unique(time[event == 1])
    survival = []
    current = 1.0
    for t in event_times:
        at_risk = np.sum(time >= t)
        events = np.sum((time == t) & (event == 1))
        current *= 1.0 - events / at_risk
        survival.append(current)
    return StepSurvival(event_times.astype(float), np.asarray(survival, dtype=float))


def estimate_censoring_distribution(time: np.ndarray, event: np.ndarray) -> StepSurvival:
    """Estimate G(t)=P(C>t) by treating censoring as the KM event."""
    time, event = _validate_survival_arrays(time, event)
    return _kaplan_meier(time, 1 - event)


def _positive_censoring_probability(values: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 1e-12):
        raise ValueError(f"censoring survival is zero for {context}; evaluation is unsupported")
    return values


def ipcw_brier_scores(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    predicted_survival: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Graf-style IPCW Brier scores using training-fold censoring weights."""
    train_time, train_event = _validate_survival_arrays(train_time, train_event)
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    eval_times = np.asarray(eval_times, dtype=float)
    predicted_survival = np.asarray(predicted_survival, dtype=float)
    if predicted_survival.shape != (len(test_time), len(eval_times)):
        raise ValueError("predicted_survival must have shape (n_test, n_eval_times)")
    predicted_survival, _ = project_probability_roundoff(predicted_survival)

    censoring = estimate_censoring_distribution(train_time, train_event)
    scores = []
    for column, t in enumerate(eval_times):
        pred = predicted_survival[:, column]
        event_before = (test_time <= t) & (test_event == 1)
        event_free = test_time > t
        weights = np.zeros(len(test_time), dtype=float)
        if event_before.any():
            g_event = _positive_censoring_probability(
                censoring.before(test_time[event_before]), "event times"
            )
            weights[event_before] = 1.0 / g_event
        if event_free.any():
            g_t = _positive_censoring_probability(
                np.asarray(censoring.at(t)), f"evaluation time {t}"
            )
            weights[event_free] = 1.0 / float(g_t)
        observed_survival = event_free.astype(float)
        scores.append(np.mean(weights * (observed_survival - pred) ** 2))
    return np.asarray(scores)


def uno_c_index(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk_scores: np.ndarray,
    tau: float | None = None,
) -> float:
    """Uno's IPCW concordance index, with higher scores indicating higher risk."""
    train_time, train_event = _validate_survival_arrays(train_time, train_event)
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if risk_scores.shape != test_time.shape:
        raise ValueError("risk_scores must match test_time")
    if tau is None:
        tau = float(np.max(test_time))

    censoring = estimate_censoring_distribution(train_time, train_event)
    concordant = 0.0
    comparable = 0.0
    for i in np.flatnonzero((test_event == 1) & (test_time <= tau)):
        later = test_time > test_time[i]
        if not later.any():
            continue
        g = _positive_censoring_probability(
            np.asarray(censoring.before(test_time[i])), f"event time {test_time[i]}"
        )
        weight = 1.0 / float(g) ** 2
        differences = risk_scores[i] - risk_scores[later]
        concordant += weight * (
            np.sum(differences > 0) + 0.5 * np.sum(differences == 0)
        )
        comparable += weight * np.sum(later)
    if comparable == 0:
        raise ValueError("no comparable pairs for Uno C-index")
    return float(concordant / comparable)


def cumulative_dynamic_auc(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk_scores: np.ndarray,
    eval_time: float,
) -> float:
    """IPCW cumulative/dynamic AUC at a fixed residual time."""
    train_time, train_event = _validate_survival_arrays(train_time, train_event)
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    risk_scores = np.asarray(risk_scores, dtype=float)
    if risk_scores.shape != test_time.shape:
        raise ValueError("risk_scores must match test_time")

    cases = np.flatnonzero((test_time <= eval_time) & (test_event == 1))
    controls = np.flatnonzero(test_time > eval_time)
    if len(cases) == 0 or len(controls) == 0:
        raise ValueError("dynamic AUC requires at least one case and one control")

    censoring = estimate_censoring_distribution(train_time, train_event)
    case_g = _positive_censoring_probability(
        censoring.before(test_time[cases]), "case event times"
    )
    case_weights = 1.0 / case_g
    numerator = 0.0
    denominator = 0.0
    control_risk = risk_scores[controls]
    for case, weight in zip(cases, case_weights):
        differences = risk_scores[case] - control_risk
        numerator += weight * (
            np.sum(differences > 0) + 0.5 * np.sum(differences == 0)
        )
        denominator += weight * len(controls)
    return float(numerator / denominator)


def calibration_groups(
    test_time: np.ndarray,
    test_event: np.ndarray,
    predicted_risk: np.ndarray,
    eval_time: float,
    n_groups: int = 5,
) -> list[dict[str, float | int]]:
    """Quantile-group calibration with Kaplan-Meier observed event risk."""
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    predicted_risk = np.asarray(predicted_risk, dtype=float)
    if predicted_risk.shape != test_time.shape:
        raise ValueError("predicted_risk must match test_time")
    if n_groups < 2 or n_groups > len(test_time):
        raise ValueError("n_groups must be between 2 and the test sample size")

    ordered = np.argsort(predicted_risk, kind="stable")
    groups = []
    for group_number, indices in enumerate(np.array_split(ordered, n_groups), start=1):
        event_km = _kaplan_meier(test_time[indices], test_event[indices])
        groups.append({
            "group": group_number,
            "n": int(len(indices)),
            "mean_predicted_risk": float(np.mean(predicted_risk[indices])),
            "observed_risk": float(1.0 - event_km.at(eval_time)),
        })
    return groups


def ipcw_calibration_parameters(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    predicted_risk: np.ndarray,
    eval_time: float,
) -> tuple[float, float]:
    """IPCW logistic calibration intercept and slope at one horizon."""
    train_time, train_event = _validate_survival_arrays(train_time, train_event)
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    risk = np.clip(np.asarray(predicted_risk, dtype=float), 1e-6, 1 - 1e-6)
    if risk.shape != test_time.shape:
        raise ValueError("predicted_risk must match test_time")
    cases = (test_time <= eval_time) & (test_event == 1)
    controls = test_time > eval_time
    usable = cases | controls
    if not cases.any() or not controls.any():
        raise ValueError("calibration requires at least one case and one control")
    censoring = estimate_censoring_distribution(train_time, train_event)
    weights = np.zeros(len(test_time), dtype=float)
    weights[cases] = 1.0 / _positive_censoring_probability(
        censoring.before(test_time[cases]), "calibration event times"
    )
    weights[controls] = 1.0 / float(_positive_censoring_probability(
        np.asarray(censoring.at(eval_time)), f"calibration time {eval_time}"
    ))
    y = cases[usable].astype(float)
    logit = np.log(risk[usable] / (1.0 - risk[usable]))
    design = np.column_stack([np.ones(len(logit)), logit])
    weights = weights[usable]
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        eta = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        score = design.T @ (weights * (y - fitted))
        information = design.T @ ((weights * fitted * (1.0 - fitted))[:, None] * design)
        step = np.linalg.pinv(information) @ score
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    if not np.isfinite(beta).all():
        raise ValueError("calibration model did not produce finite parameters")
    return float(beta[0]), float(beta[1])


def ipcw_weight_diagnostics(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    eval_time: float,
) -> dict[str, float | int]:
    """Summarize nonzero horizon-specific IPCW weights."""
    train_time, train_event = _validate_survival_arrays(train_time, train_event)
    test_time, test_event = _validate_survival_arrays(test_time, test_event)
    censoring = estimate_censoring_distribution(train_time, train_event)
    cases = (test_time <= eval_time) & (test_event == 1)
    controls = test_time > eval_time
    weights = np.zeros(len(test_time), dtype=float)
    if cases.any():
        weights[cases] = 1.0 / _positive_censoring_probability(
            censoring.before(test_time[cases]), "diagnostic event times"
        )
    if controls.any():
        weights[controls] = 1.0 / float(_positive_censoring_probability(
            np.asarray(censoring.at(eval_time)), f"diagnostic time {eval_time}"
        ))
    nonzero = weights[weights > 0]
    if not len(nonzero):
        raise ValueError("no nonzero IPCW weights at the evaluation horizon")
    return {
        "n_nonzero": int(len(nonzero)),
        "max": float(np.max(nonzero)),
        "p99": float(np.quantile(nonzero, 0.99)),
        "effective_sample_size": float(np.sum(nonzero) ** 2 / np.sum(nonzero ** 2)),
    }


def calc_c_index(time: np.ndarray, event: np.ndarray, risk_scores: np.ndarray) -> float:
    """Calculate Harrell's C-index. risk_scores: higher = higher risk."""
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    risk_scores = np.asarray(risk_scores)

    n = len(time)
    concordant = 0
    permissible = 0

    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] > time[i] or (time[j] == time[i] and event[j]):
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    if permissible == 0:
        return 0.5
    return float(concordant / permissible)


def calc_ibs(time: np.ndarray, event: np.ndarray, surv_probs: np.ndarray,
             times_grid: np.ndarray) -> float:
    """Calculate Integrated Brier Score (approximation via trapezoid rule).
    surv_probs: array (n_samples, n_times) with survival probabilities.
    """
    time = np.asarray(time)
    event = np.asarray(event).astype(bool)
    surv_probs = np.asarray(surv_probs)
    times_grid = np.asarray(times_grid)

    n = len(time)
    bs_values = []

    for t_idx, t in enumerate(times_grid):
        S_pred = surv_probs[:, t_idx]
        # observed status at time t: 1 if event happened before t, 0 otherwise
        # but we need to handle censoring
        y = np.zeros(n)
        for i in range(n):
            if time[i] <= t:
                if event[i]:
                    y[i] = 1.0
                else:
                    # censored before t: use 1 - S_pred as approximate
                    y[i] = 1.0 - S_pred[i]
            else:
                y[i] = 0.0

        bs = np.mean((y - (1.0 - S_pred)) ** 2)
        bs_values.append(bs)

    bs_values = np.array(bs_values)
    # Trapezoid integration over time
    if len(times_grid) < 2:
        return float(bs_values[0]) if len(bs_values) > 0 else 0.0

    # numpy >= 2.0 uses trapezoid instead of trapz
    try:
        ibs = np.trapezoid(bs_values, times_grid) / (times_grid[-1] - times_grid[0])
    except AttributeError:
        ibs = np.trapz(bs_values, times_grid) / (times_grid[-1] - times_grid[0])
    return float(ibs)


def calc_calibration_slope(predicted_probs: np.ndarray, observed: np.ndarray) -> float:
    """Simple calibration slope via linear regression on logit scale.
    predicted_probs: predicted event probability at a fixed time point.
    observed: binary observed outcome at that time point.
    """
    p = np.clip(predicted_probs, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    x = logit_p.reshape(-1, 1)
    y = np.asarray(observed, dtype=float)
    # Add intercept column
    X = np.hstack([np.ones((len(x), 1)), x])
    # Ordinary least squares: beta = (X^T X)^{-1} X^T y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1])
