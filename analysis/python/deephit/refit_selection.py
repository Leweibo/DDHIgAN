"""Pure helpers for the DDHIgAN refit-plus-selection bootstrap."""
from __future__ import annotations

from collections import Counter
from typing import Mapping


WEIGHTS = {
    "rank000": 0.0,
    "rank0003": 0.003,
    "rank001": 0.01,
    "rank003": 0.03,
    "rank010": 0.1,
}


def candidate_tasks(replicates: int = 5, folds: int = 5):
    if replicates <= 0 or folds <= 0:
        raise ValueError("replicates and folds must be positive")
    return [(rep, fold, tag) for rep in range(replicates) for fold in range(folds) for tag in WEIGHTS]


def select_candidate(scores: Mapping[str, float], tolerance: float = 1e-12) -> str:
    if set(scores) != set(WEIGHTS):
        raise ValueError("one finite score is required for every registered candidate")
    if any(not isinstance(value, (int, float)) or value != value for value in scores.values()):
        raise ValueError("candidate scores must be finite numbers")
    best = min(float(value) for value in scores.values())
    eligible = [tag for tag, value in scores.items() if float(value) <= best + tolerance]
    return max(eligible, key=lambda tag: WEIGHTS[tag])


def selection_frequency(choices):
    counts = Counter(choices)
    unknown = set(counts) - set(WEIGHTS)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    return {tag: int(counts.get(tag, 0)) for tag in WEIGHTS}
