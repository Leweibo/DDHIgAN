"""Reconstruct and hash actual-visit query rosters before any model fitting."""
import hashlib
import json
import numpy as np


def query_roster_sha256(patient_ids, query_times):
    if len(patient_ids)!=len(query_times):
        raise ValueError('query roster lengths differ')
    digest=hashlib.sha256()
    for patient_id, time in zip(patient_ids,query_times):
        if not np.isfinite(time): raise ValueError('nonfinite query time')
        digest.update(json.dumps([str(patient_id),float(time).hex()],separators=(',',':')).encode()+b'\n')
    return digest.hexdigest()


def reconstruct_query_rosters(longitudinal, outcomes, patient_ids, *, tolerance):
    from .data_utils import build_dynamic_prefix_inputs
    from .time_grid import HalfYearTimeGrid
    args=dict(long_df=longitudinal,outcomes_df=outcomes,patient_ids=patient_ids,
              static_cols=[],long_cols=[],max_visits=1,norm_stats={},
              evaluation_landmarks=[],include_actual_queries=True,max_query_time=5.,
              min_history_time=0.,anchor_nearest_t0=True,input_time_scale=10.)
    legacy=build_dynamic_prefix_inputs(**args,time_grid=HalfYearTimeGrid(10))
    candidate=build_dynamic_prefix_inputs(**args,time_grid=HalfYearTimeGrid(15),
                                         preserve_source_time=True,actual_query_end_tolerance=tolerance)
    # Prove equality of every patient/query slot, not only aggregate counts.
    if candidate['patient_id']!=legacy['patient_id']:
        raise ValueError('candidate and reference patient/query slots differ before fitting')
    if not np.array_equal(candidate['query_time'].astype(np.float32),legacy['query_time']):
        raise ValueError('candidate and reference actual query roster differs before fitting')
    return legacy,candidate
