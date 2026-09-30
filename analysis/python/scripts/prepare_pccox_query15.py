"""Prepare query-aligned PCCox evaluation inputs; retain actual measurement times."""
from pathlib import Path
import argparse
import json
import pandas as pd
from python.scripts.prepare_pccox_inputs import FEATURES
from python.scripts.run_arbitrary_visit_reanalysis_references import (
    _anchored_longitudinal, _base_frame, _fill_dynamic, get_clinical_variable_set,
    build_pccox_actual_visit_training_frame, impute_from_training,
)
from python.utils.data_utils import get_fold_split


def query_frame(baseline, longitudinal, outcomes, ids, query, *, anchored=None):
    if not 0 <= query <= 5:
        raise ValueError('query outside [0,5]')
    variables=get_clinical_variable_set('minimal_core')
    base=_base_frame(baseline,outcomes,ids,'ESKD')
    base=base[base.eskd_time>query].copy()
    history=_anchored_longitudinal(longitudinal) if anchored is None else anchored
    history=history[history.visit_time<=query].copy()
    for col in variables.dynamic_cols:
        history[col]=pd.to_numeric(history[col],errors='coerce')
        history[col]=history.groupby('patient_id',sort=False)[col].ffill()
    latest=history.groupby('patient_id',as_index=False,sort=False).tail(1)
    frame=base.merge(latest[['patient_id','visit_time',*variables.dynamic_cols]],on='patient_id',how='left',validate='one_to_one')
    if frame.visit_time.isna().any():
        raise ValueError('patient lacks observed history; do not fabricate a visit')
    frame=_fill_dynamic(frame,variables.dynamic_cols,variables.baseline_fallback)
    frame['query_time']=float(query)
    frame['duration']=frame.eskd_time.astype(float)-query
    frame['event']=frame.eskd_status.astype(int)
    frame['last_observation_time']=frame.visit_time
    frame['measurement_time']=frame.last_observation_time
    frame['prediction_delay']=query-frame.last_observation_time
    if (frame.prediction_delay<0).any():raise ValueError('future measurement')
    return frame[['patient_id','duration','event','query_time',*variables.cox_static_cols,*variables.dynamic_cols,
                  'last_observation_time','measurement_time','prediction_delay']].reset_index(drop=True)


def prepare(data,splits,parent_prepared,output):
    if output.exists():raise FileExistsError(output)
    baseline=pd.read_csv(data/'igan_baseline.csv',dtype={'patient_id':str})
    longitudinal=pd.read_csv(data/'igan_longitudinal.csv',dtype={'patient_id':str})
    outcomes=pd.read_csv(data/'igan_outcomes.csv',dtype={'patient_id':str})
    output.mkdir(parents=True)
    reports=[]
    anchored=_anchored_longitudinal(longitudinal)
    raw_all=build_pccox_actual_visit_training_frame(baseline,longitudinal,outcomes,outcomes.patient_id.tolist(),'ESKD',5.)
    for fold in range(5):
        train=get_fold_split(splits,fold,'train');test=get_fold_split(splits,fold,'test')
        raw=raw_all[raw_all.patient_id.isin(train)].reset_index(drop=True)
        fitted,_=impute_from_training(raw,raw,FEATURES)
        parent=pd.read_csv(parent_prepared/f'fold{fold}/fit.csv',dtype={'patient_id':str})
        pd.testing.assert_frame_equal(fitted.reset_index(drop=True),parent,check_dtype=False,rtol=1e-12,atol=1e-12)
        directory=output/f'fold{fold}';directory.mkdir()
        cells=[]
        for q in range(6):
            new=query_frame(baseline,longitudinal,outcomes,test,q,anchored=anchored)
            _,new=impute_from_training(fitted,new,FEATURES)
            old=pd.read_csv(parent_prepared/f'fold{fold}/test_landmark{q}.csv',dtype={'patient_id':str})
            pd.testing.assert_frame_equal(new[['patient_id',*FEATURES]],old[['patient_id',*FEATURES]],check_dtype=False,rtol=1e-12,atol=1e-12)
            new.to_csv(directory/f'test_landmark{q}.csv',index=False)
            cells.append(dict(query=q,n=len(new),delayed=int((new.prediction_delay>0).sum())))
        reports.append(dict(fold=fold,cells=cells,training_and_imputation_unchanged=True))
    (output/'manifest.json').write_text(json.dumps(dict(status='PREPARED_NO_FIT',folds=reports),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['data','splits','parent-prepared','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.splits,a.parent_prepared,a.output)
