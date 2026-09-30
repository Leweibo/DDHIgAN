"""Evaluation-only record recency contract for the 12-year support experiment."""
import numpy as np


def eligible_queries(query, last, max_gap):
    query, last = np.broadcast_arrays(np.asarray(query, float), np.asarray(last, float))
    if max_gap not in (1, 2):
        raise ValueError('max_gap must be 1 or 2 years')
    if not np.isfinite(query).all() or not np.isfinite(last).all():
        raise ValueError('query and last observation must be finite')
    if np.any(last < 0) or np.any(last > query):
        raise ValueError('history must be at or before query, after biopsy')
    # Inclusive threshold: the user excludes strictly MORE than 1/2 years.
    return query - last <= max_gap
