from __future__ import annotations

import numpy as np


def recalibrate(values: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("expected risks at years 1 through 10")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("raw risk curve is invalid")
    hazard = -np.log1p(-np.clip(values, 1e-7, 1 - 1e-7))
    result = -np.expm1(-np.exp(np.clip(alpha, -30, 30)) * hazard ** beta)
    result = np.clip(result, 0, 1)
    if np.any(np.diff(result, axis=1) < -1e-7):
        raise ValueError("calibrated curve is not monotonic")
    return result


def calibration_uncertainty_interval(
    values: np.ndarray,
    draws: list[dict],
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise interval from frozen-risk calibration-parameter draws only."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("expected risks at years 1 through 10")
    alpha = np.asarray([row["alpha"] for row in draws], dtype=float)
    beta = np.asarray([row["beta"] for row in draws], dtype=float)
    if len(draws) != 2000 or not np.isfinite(alpha).all() or not np.isfinite(beta).all():
        raise ValueError("calibration uncertainty draws are incomplete or invalid")
    if np.any(beta <= 0) or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("calibration uncertainty contract is invalid")
    hazard = -np.log1p(-np.clip(values, 1e-7, 1 - 1e-7))
    sampled = -np.expm1(
        -np.exp(np.clip(alpha[:, None, None], -30, 30))
        * hazard[None, :, :] ** beta[:, None, None]
    )
    lower, upper = np.quantile(np.clip(sampled, 0, 1), quantiles, axis=0)
    if np.any(np.diff(lower, axis=1) < -1e-7) or np.any(np.diff(upper, axis=1) < -1e-7):
        raise ValueError("calibration uncertainty interval is not monotonic")
    return lower, upper
