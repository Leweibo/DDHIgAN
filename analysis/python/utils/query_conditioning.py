"""Query survival conditioning on a finite residual grid (no extrapolation)."""
from __future__ import annotations

import numpy as np


def log_survival_from_logits(logits, num_event_bins):
    """S(0)=1; S(t_k) is the sum of all masses strictly after bin k.

    Reverse log-sums avoid cancellation in 1-CIF, including tiny tail masses.
    Computation is float64 and never floors a denominator or invents tail times.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != num_event_bins + 1:
        raise ValueError("logits must contain event bins plus one tail mass")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    logits = logits - logits.max(axis=1, keepdims=True)
    reverse = np.logaddexp.accumulate(logits[:, ::-1], axis=1)[:, ::-1]
    return np.column_stack([np.zeros(len(logits)), reverse[:, 1:] - reverse[:, :1]])


def conditional_log_survival(log_survival, boundaries, delay, horizons):
    """Return log S(delay+h) - log S(delay), linear within log-survival bins."""
    values = np.asarray(log_survival, dtype=np.float64)
    grid = np.asarray(boundaries, dtype=np.float64)
    delay = np.asarray(delay, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("boundaries must strictly increase from zero")
    if values.ndim != 2 or values.shape[1] != len(grid):
        raise ValueError("log survival shape does not match boundaries")
    if not np.isfinite(values).all() or np.any(values > 0) or np.any(values[:, 0] != 0) or np.any(np.diff(values, axis=1) > 0):
        raise ValueError("log survival must be finite, nonpositive, monotone, and start at zero")
    if delay.shape != (len(values),) or not np.isfinite(delay).all() or np.any(delay < 0):
        raise ValueError("prediction delay must be finite and nonnegative for every patient")
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or np.any(horizons < 0):
        raise ValueError("horizons must be finite and nonnegative")
    times = delay[:, None] + horizons[None, :]
    if np.any(delay > grid[-1]) or np.any(times > grid[-1]):
        raise ValueError("query plus horizon exceeds finite model support")
    def interpolate(t):
        index = np.searchsorted(grid, t, side="right") - 1
        index = np.minimum(index, len(grid) - 2)
        weight = (t - grid[index]) / (grid[index + 1] - grid[index])
        row = np.arange(len(values))[:, None]
        return (1 - weight) * values[row, index] + weight * values[row, index + 1]
    return interpolate(times) - interpolate(delay[:, None])
