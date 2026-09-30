from __future__ import annotations

import numpy as np


def administratively_censor(
    time: np.ndarray,
    event: np.ndarray,
    max_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.shape != event.shape:
        raise ValueError("time and event must have matching shapes")
    if max_time <= 0:
        raise ValueError("max_time must be positive")
    censored_time = np.minimum(time, max_time)
    censored_event = event.copy()
    censored_event[time > max_time] = 0
    return censored_time, censored_event


def split_patient_ids(
    patient_ids: list[str],
    validation_fraction: float,
    seed: int,
    strata: list[object] | None = None,
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between 0 and 1")
    ids = np.asarray(list(map(str, patient_ids)), dtype=object)
    if len(ids) < 2:
        raise ValueError("at least two patients are required")
    rng = np.random.default_rng(seed)
    if strata is None:
        permutation = rng.permutation(len(ids))
        validation_count = max(1, int(round(len(ids) * validation_fraction)))
        validation_indices = permutation[:validation_count]
    else:
        labels = np.asarray(list(strata), dtype=object)
        if labels.shape != ids.shape:
            raise ValueError("strata must have one value per patient_id")
        chunks = []
        for label in sorted(set(labels.tolist()), key=str):
            indices = np.flatnonzero(labels == label)
            if len(indices) < 2:
                continue
            count = max(1, int(round(len(indices) * validation_fraction)))
            count = min(count, len(indices) - 1)
            chunks.append(rng.permutation(indices)[:count])
        validation_indices = np.concatenate(chunks) if chunks else np.asarray([rng.integers(len(ids))])
    training_indices = np.setdiff1d(np.arange(len(ids)), validation_indices, assume_unique=False)
    return ids[training_indices].tolist(), ids[validation_indices].tolist()
