"""Patient bootstrap inside fixed development partitions; no model selection."""
from collections import Counter
import hashlib
import json
import numpy as np


def patient_counts(patient_ids, *, replicate, fold, partition):
    ids = sorted(map(str, patient_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('population must contain unique patients')
    if replicate < 0 or fold not in range(5) or partition not in (0, 1):
        raise ValueError('invalid replicate, fold or development partition')
    rng = np.random.default_rng(np.random.SeedSequence([316, replicate, fold, partition]))
    return dict(sorted(Counter(map(str, rng.choice(ids, size=len(ids), replace=True))).items()))


def audit_partitions(train_population, validation_population, test_ids, train_counts, validation_counts):
    tr, va, te = map(lambda x: set(map(str, x)), (train_population, validation_population, test_ids))
    if tr & va or tr & te or va & te:
        raise ValueError('patient leakage across development/test partitions')
    for population, counts in ((tr, train_counts), (va, validation_counts)):
        if not counts or not set(counts) <= population:
            raise ValueError('bootstrap contains an ineligible patient')
        if any(type(v) is not int or v <= 0 for v in counts.values()):
            raise ValueError('multiplicities must be positive integers')
        if sum(counts.values()) != len(population):
            raise ValueError('bootstrap slot count differs from original partition size')
    return {
        'training_slots': sum(train_counts.values()),
        'validation_slots': sum(validation_counts.values()),
        'training_unique': len(train_counts), 'validation_unique': len(validation_counts),
        'test_patients': len(te), 'partitions_disjoint': True,
        'roster_sha256': hashlib.sha256(json.dumps(
            [train_counts, validation_counts], sort_keys=True).encode()).hexdigest(),
    }


def validation_weights(patient_ids, prefix_weights, multiplicity):
    ids = list(map(str, patient_ids))
    if set(ids) != set(multiplicity):
        raise ValueError('validation dataset and roster differ')
    old = np.asarray(prefix_weights, dtype=float)
    if old.shape != (len(ids),) or not np.isfinite(old).all() or (old <= 0).any():
        raise ValueError('invalid original prefix weights')
    weights = old * np.asarray([multiplicity[x] for x in ids], dtype=float)
    for pid in set(ids):
        ix = np.asarray([x == pid for x in ids])
        if not np.isclose(old[ix].sum(), 1., atol=1e-5):
            raise ValueError('original validation weights are not patient-balanced')
    return weights
