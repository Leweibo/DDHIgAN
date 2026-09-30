"""Pure helpers for the locked patient-cluster bootstrap-refit contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:  # Supports both ``python.*`` imports in tests and script-local imports.
    from python.utils.probabilities import (
        FLOAT32_PROBABILITY_ROUNDOFF_TOLERANCE,
        project_probability_roundoff,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct scripts
    from utils.probabilities import (  # type: ignore[no-redef]
        FLOAT32_PROBABILITY_ROUNDOFF_TOLERANCE,
        project_probability_roundoff,
    )


def generate_rosters(
    patient_ids: list[str], replicates: int = 1000, seed: int = 316,
):
    ids = tuple(sorted(map(str, patient_ids)))
    if len(ids) != 9948 or len(set(ids)) != len(ids):
        raise ValueError("bootstrap population must contain exactly 9,948 unique patients")
    if replicates != 1000 or seed != 316:
        raise ValueError("formal bootstrap is locked to 1,000 replicates and seed 316")
    rng = np.random.default_rng(seed)
    population = np.asarray(ids, dtype=object)
    for replicate in range(replicates):
        sampled = rng.choice(population, size=len(population), replace=True)
        yield replicate, dict(Counter(map(str, sampled)))


def validate_roster(
    multiplicity: dict[str, int], population_ids: list[str], replicate: int,
) -> dict:
    population = set(map(str, population_ids))
    clean = {str(key): int(value) for key, value in multiplicity.items()}
    if not clean or not set(clean).issubset(population):
        raise ValueError("bootstrap roster contains unknown or no patients")
    if any(value <= 0 for value in clean.values()):
        raise ValueError("bootstrap roster counts must be positive integers")
    if sum(clean.values()) != len(population):
        raise ValueError("bootstrap roster must contain exactly 9,948 patient slots")
    oob = sorted(population.difference(clean))
    if not oob:
        raise ValueError("bootstrap roster has no OOB patients")
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return {
        "replicate": int(replicate),
        "slots": int(sum(clean.values())),
        "unique_inbag_patients": len(clean),
        "oob_patients": len(oob),
        "max_multiplicity": max(clean.values()),
        "roster_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def load_roster(path: str | Path, population_ids: list[str], replicate: int):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("replicate") != int(replicate):
        raise ValueError("bootstrap roster replicate mismatch")
    multiplicity = payload.get("multiplicity")
    if not isinstance(multiplicity, dict):
        raise ValueError("bootstrap roster lacks multiplicity mapping")
    summary = validate_roster(multiplicity, population_ids, replicate)
    return {str(key): int(value) for key, value in multiplicity.items()}, summary
