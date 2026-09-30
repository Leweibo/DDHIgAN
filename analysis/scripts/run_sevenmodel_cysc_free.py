"""Five CysC-free fits; current rank-zero core configuration and fixed records.

Run from an immutable copy of the validated base release. Private outputs stay RIS.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python.evaluation.query12_recency import eligible_queries

NAME = 'physv15_rankzero_sevenmodel_20260911'
ROOT = Path('results') / NAME
BRANCH = None
CONFIG = None
PREFIX = 'ddhigan_' + NAME
V = 'physv15_outcome_recheck_9947_20260908'
BASE_SHA = 'ee8db6f7317260389777b8ca4633d5535d02bd56'
RELEASE = 'd2ddd4b402568e4371fda34e23b5135ce7e5a47185ccf72fed5b13e8feb2f20b'
REF = Path('results/physv15_query15_boundaryfix_20260908')
REF_PREFIX = 'ddhigan_physv15_query15_boundaryfix_20260908'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def select(branch):
    global BRANCH,CONFIG,PREFIX
    BRANCH=branch; CONFIG=Path('config')/NAME/(branch+'.yaml'); PREFIX='ddhigan_'+branch+'_'+NAME

def config():
    c=yaml.safe_load(CONFIG.read_text())
    p=yaml.safe_load(Path('config/physv15_rankzero_standard_20260911/core.yaml').read_text())
    expected=json.loads(json.dumps(p))
    expected['data']['static_cols'].remove('baseline_CystatinC')
    expected['data']['long_cols'].remove('CystatinC')
    expected['model']['name']=PREFIX
    expected['output']={k:str(ROOT/'ddhigan_cysc_free'/k.removesuffix('_dir')) for k in p['output']}
    assert c==expected
    assert len(c['data']['static_cols'])==5 and len(c['data']['long_cols'])==3
    assert sha(Path(c['data']['processed_dir'])/'release_manifest.json')==RELEASE
    for manifest in ('CODE_RELEASE.json','RANKZERO_CODE.json','SEVENMODEL_CODE.json'):
        m=json.loads(Path(manifest).read_text())
        if manifest=='CODE_RELEASE.json': assert m['git_commit']==BASE_SHA
        for path,h in m['files'].items(): assert sha(path)==h,path
    return c

def reference(fold,q):
    p=Path('results/physv15_query11_recency1_20260910/ddhigan/prediction')/f'ddhigan_physv15_query11_recency1_20260910_ESKD_fold{fold}_test_landmark{q}.csv'
    return p,pd.read_csv(p,dtype={'patient_id':str}).sort_values('patient_id').reset_index(drop=True)

def resources():
    assert platform.node()=='gpu'
    import psutil,torch
    busy=[psutil.cpu_percent(interval=1) for _ in range(3)]
    assert max(busy)<55 and psutil.virtual_memory().available>64*2**30
    devices=[]
    for i in range(4):
        free,total=torch.cuda.mem_get_info(i); assert free>20*2**30
        devices.append(dict(index=i,free_bytes=free,total_bytes=total))
    return dict(host=platform.node(),cpu_busy_samples=busy,devices=devices,python=sys.executable,torch=torch.__version__)

def preflight():
    resource=resources()
    from python.utils.data_utils import get_fold_split,load_longitudinal_data,compute_dynamic_norm_stats
    from python.utils.survival_data import split_patient_ids
    from python.utils.reproducibility import effective_fold_seed
    prior=json.loads((REF/'run/preflight.json').read_text())
    report=dict(status='VALID',resources=resource,branches={},core_reused=True,new_fit_count=5)
    for branch in ('cysc_free',):
        select(branch);c=config()
        manifest=json.loads((Path(c['data']['processed_dir'])/'release_manifest.json').read_text())
        for n in ('igan_baseline.csv','igan_longitudinal.csv','igan_outcomes.csv'):
            assert sha(Path(c['data']['processed_dir'])/n)==manifest['artifact_sha256'][n],n
        for n,h in manifest['fold_sha256'].items():
            assert sha(Path(c['data']['splits_dir'])/n)==h,n
        long,out=load_longitudinal_data(c['data']['processed_dir'])
        formal=out[['patient_id','eskd_time','eskd_status']].rename(columns={'eskd_time':'event_time','eskd_status':'event_status'})
        formal.patient_id=formal.patient_id.astype(str)
        evidence=dict(config_sha256=sha(CONFIG),folds=[])
        parent=Path('results/physv15_ranking_ablation_20260910/rank000/seed_316')
        pp='ddhigan_rank000_seed316_physv15_ranking_ablation_20260910'
        for fold in range(5):
            ids=get_fold_split(c['data']['splits_dir'],fold,'train')
            strata=formal.set_index('patient_id').event_status.reindex(list(map(str,ids))).astype(int).tolist()
            tr,va=split_patient_ids(ids,validation_fraction=.2,seed=effective_fold_seed(316,fold),strata=strata)
            digest=lambda a: hashlib.sha256('\n'.join(sorted(map(str,a))).encode()).hexdigest()
            assert digest(tr)==prior['splits'][fold]['train_patient_set_sha256']
            assert digest(va)==prior['splits'][fold]['validation_patient_set_sha256']
            stats=compute_dynamic_norm_stats(long,formal,c['data']['static_cols'],c['data']['long_cols'],tr,min_visit_time=0.,anchor_nearest_t0=True,max_visit_time=5.)
            ns=parent/'norm_stats'/f'fold_{fold}_{pp}_ESKD_norm_stats.json'
            parent_stats=json.loads(ns.read_text())
            assert stats=={k:v for k,v in parent_stats.items() if k not in ('CystatinC','baseline_CystatinC')}
            reduced_sha=hashlib.sha256(json.dumps(stats,indent=2).encode()).hexdigest()
            evidence['folds'].append(dict(fold=fold,norm_stats_sha256=reduced_sha,parent_norm_stats_sha256=sha(ns),train_patient_set_sha256=digest(tr),validation_patient_set_sha256=digest(va)))
        report['branches'][branch]=evidence
    core_inv=json.loads(Path('results/physv15_query11_recency1_20260910/evaluation_attempt2/aggregate_review/PREPARED.json').read_text())['prediction_inventory']
    for row in core_inv:
        p,x=reference(row['fold'],row['query']); assert sha(p)==row['sha256']
        assert eligible_queries(x.query_time,x.last_observation_time,1).all()
    write(ROOT/'run/preflight.json',report)
    print('PREFLIGHT_VALID',flush=True)

def predict(fold):
    c=config()
    import torch
    from torch.utils.data import DataLoader
    from python.deephit.predict import predict_landmark
    from python.deephit.data_loader import DynamicDeepHitDataset, collate_fn
    from python.deephit.train import build_model
    from python.utils.data_utils import load_norm_stats
    from python.utils.time_grid import HalfYearTimeGrid
    checkpoint=Path(c['output']['checkpoint_dir'])/f'{PREFIX}_ESKD_fold{fold}.pt'
    ck=torch.load(checkpoint,map_location='cuda')
    assert ck['config']==c
    model=build_model(c).cuda(); model.load_state_dict(ck['model_state_dict'])
    stats=Path(c['output']['norm_stats_dir'])/f'fold_{fold}_{PREFIX}_ESKD_norm_stats.json'
    pre=json.loads((ROOT/'run/preflight.json').read_text())
    assert sha(stats)==pre['branches'][BRANCH]['folds'][fold]['norm_stats_sha256']
    grid=HalfYearTimeGrid(max_time=11.,interval_width=.5)
    for q in range(6):
        _, ref=reference(fold,q)
        ref=ref[eligible_queries(ref.query_time,ref.last_observation_time,1)]
        ds=DynamicDeepHitDataset(processed_dir=c['data']['processed_dir'],patient_ids=ref.patient_id.tolist(),
            static_cols=c['data']['static_cols'],long_cols=c['data']['long_cols'],outcome='ESKD',max_visits=30,time_grid=grid,
            multimodal_embedding_path=c['data'].get('multimodal_embedding_path'),multimodal_cols=c['data'].get('multimodal_cols'),norm_stats=load_norm_stats(stats),evaluation_landmarks=[q],include_actual_queries=False,min_history_time=0.,
            anchor_nearest_t0=True,append_missing_query_row=False,input_time_scale=10.,preserve_source_time=True,actual_query_end_tolerance=1e-10)
        assert ds.x_values.shape[-1]==9 and ds.x_missing.shape[-1]==3
        assert model.longitudinal_dim==3
        assert eligible_queries(ds.query_time.numpy(),ds.last_observation_time.numpy(),1).all()
        pred=predict_landmark(model,DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate_fn),q,[3,5,10],grid,torch.device('cuda'),
                              residual_grid=list(range(1,11)),conditioning='query_survival').sort_values('patient_id')
        assert pred.patient_id.tolist()==ref.patient_id.tolist()
        np.testing.assert_allclose(pred.prediction_delay,ref.prediction_delay,rtol=0,atol=1e-12)
        np.testing.assert_allclose(pred.residual_time,ref.residual_time,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(pred.true_event,ref.true_event)
        p=ROOT/('ddhigan_'+BRANCH)/'prediction'/f'{PREFIX}_ESKD_fold{fold}_test_landmark{q}.csv'
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as handle: pred.to_csv(handle,index=False)
    write(ROOT/'run'/f'{BRANCH}_fold{fold}_validated.json',dict(status='VALID',fold=fold,checkpoint_sha256=sha(checkpoint),norm_sha256=sha(stats)))

def train():
    resources()
    assert json.loads((ROOT/'run/preflight.json').read_text())['status']=='VALID'
    (ROOT/'run/training_started').mkdir()
    tasks=[(branch,f) for branch in ('cysc_free',) for f in range(5)]
    write(ROOT/'run/runner.json',dict(pid=os.getpid(),host=platform.node(),started=time.time(),cpu_threads_per_fold=4,tasks=tasks))
    def lane(gpu,items):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for branch,fold in items:
            cfg=str(Path('config')/NAME/(branch+'.yaml'))
            with (ROOT/'run'/f'{branch}_fold{fold}.log').open('x') as log:
                subprocess.run([sys.executable,'python/deephit/train.py','--config',cfg,'--fold',str(fold),'--outcome','ESKD'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=23*3600)
                subprocess.run([sys.executable,__file__,'predict','--branch',branch,'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(lane,g,tasks[g::4]) for g in range(4)]
        for job in jobs:job.result()
    write(ROOT/'run/TRAINING_COMPLETE.json',dict(status='FIVE_FOLDS_VALIDATED',fit_count=5,core_refit=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train','predict']);p.add_argument('--branch',choices=['cysc_free']);p.add_argument('--fold',type=int)
    args=p.parse_args()
    if args.stage=='predict':select(args.branch);predict(args.fold)
    elif args.stage=='preflight':preflight()
    else:train()
