"""Sampling and signed optimism for a fixed five-fold learning procedure."""
from collections import Counter
import hashlib
import json
import numpy as np


def draw_counts(folds, replicate, seed=316):
    if len(folds) != 5 or replicate < 0:
        raise ValueError('five folds and a nonnegative replicate are required')
    seen = set()
    result = {}
    for fold, population in enumerate(folds):
        ids = sorted(map(str, population))
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('outer-fold patients must be unique and disjoint')
        seen.update(ids)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 9173, replicate, fold]))
        result.update(Counter(map(str, rng.choice(ids, len(ids), replace=True))))
    return dict(sorted(result.items()))


def audit(train, validation, test, tc, vc):
    tr, va, te = (set(map(str, x)) for x in (train, validation, test))
    if tr & va or tr & te or va & te:
        raise ValueError('patient overlap across partitions')
    for population, counts in ((tr, tc), (va, vc)):
        if not counts or not set(counts) <= population:
            raise ValueError('sampled patient outside partition')
        if any(type(v) is not int or v <= 0 for v in counts.values()):
            raise ValueError('invalid multiplicity')
    return dict(training_slots=sum(tc.values()), validation_slots=sum(vc.values()),
                training_unique=len(tc), validation_unique=len(vc), test_patients=len(te),
                partitions_disjoint=True,
                roster_sha256=hashlib.sha256(json.dumps([tc, vc], sort_keys=True).encode()).hexdigest())


def signed_optimism(apparent, original, reference):
    a, o, r = map(lambda x: np.asarray(x, dtype=float), (apparent, original, reference))
    if a.shape != o.shape or a.ndim < 1 or a.shape[0] < 2 or a.shape[1:] != r.shape:
        raise ValueError('paired replicates and matching reference are required')
    if not all(np.isfinite(x).all() for x in (a, o, r)):
        raise ValueError('nonfinite metrics')
    difference = a-o
    mean = difference.mean(axis=0)
    return dict(signed_optimism=mean, corrected=r-mean,
                monte_carlo_se=difference.std(axis=0, ddof=1)/np.sqrt(len(a)))
