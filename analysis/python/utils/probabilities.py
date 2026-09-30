"""Numerical boundary handling for stored survival probabilities."""

from __future__ import annotations

import numpy as np


# Softmax/cumulative-sum output is normally float32.  This permits only a
# bounded number of ULP-scale excursions before projecting to the closed unit
# interval; it is not a general clipping policy for invalid model output.
FLOAT32_PROBABILITY_ROUNDOFF_TOLERANCE = float(32 * np.finfo(np.float32).eps)


def project_probability_roundoff(values: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project finite float32-scale endpoint roundoff onto ``[0, 1]``.

    Material probability violations remain fail-closed.  The returned audit is
    aggregate-only and can be recorded by callers without retaining rows.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("probabilities must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("probabilities contain non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    tolerance = FLOAT32_PROBABILITY_ROUNDOFF_TOLERANCE
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError("probabilities exceed the locked float32 roundoff tolerance")
    projected = np.clip(array, 0.0, 1.0)
    changed = projected != array
    row_changed = changed.any(axis=1) if changed.ndim > 1 else changed
    return projected, {
        "tolerance": tolerance,
        "adjusted_values": int(changed.sum()),
        "adjusted_rows": int(row_changed.sum()),
        "maximum_absolute_adjustment": float(np.abs(projected - array).max(initial=0.0)),
        "pre_projection_minimum": minimum,
        "pre_projection_maximum": maximum,
    }
