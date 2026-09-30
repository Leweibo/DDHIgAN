"""Isolated fixed-partition patient-bootstrap refits; exploratory, no selection."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, os, platform, pickle, subprocess, sys, time
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from python.deephit.refit_pilot import patient_counts, audit_partitions, validation_weights

NAME='physv15_bootstrap_refit_pilot_20260915'
ROOT=Path('results')/NAME
SPEC=Path('config')/NAME/'run_spec.json'
BASE=Path('config/physv15_rankzero_standard_20260911/core.yaml')
REFERENCE=Path('results/physv15_rankzero_standard_20260911')
RELEASE='d2ddd4b402568e4371fda34e23b5135ce7e5a47185ccf72fed5b13e8feb2f20b'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def taskdir(rep,fold):return ROOT/'tasks'/f'rep{rep:02d}_fold{fold}'
def codecheck():
    manifest=json.loads(Path('PILOT_CODE.json').read_text())
    for name,digest in manifest['files'].items():assert sha(name)==digest,name
    spec=json.loads(SPEC.read_text())
    assert spec['replicates']==5 and spec['new_fits']==25
    c=yaml.safe_load(BASE.read_text())
    assert c['model']['loss']['ranking']==0 and c['model']['max_residual_time']==11
    assert c['training']['seed']==316 and c['training']['patience']==20
    assert sha(c['data']['release_manifest_path'])==RELEASE
    return c

def resources():
    import psutil,torch
    assert platform.node()=='gpu'
    busy=[psutil.cpu_percent(interval=1) for _ in range(3)]
    assert max(busy)<55 and psutil.virtual_memory().available>64*2**30
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not active,'GPU has existing jobs'
    devices=[]
    for g in range(4):
        free,total=torch.cuda.mem_get_info(g);assert free>20*2**30
        devices.append(dict(index=g,free_bytes=free,total_bytes=total))
    return dict(host=platform.node(),cpu_busy_percent=busy,memory_available=psutil.virtual_memory().available,
                devices=devices,python=sys.executable,torch=torch.__version__,numpy=np.__version__)

def cache():
    meta=json.loads((REFERENCE/'aggregate_review/PREPARED.json').read_text())
    p=REFERENCE/'evaluation_cache.pkl';assert sha(p)==meta['cache_sha256']
    with p.open('rb') as f:obj=pickle.load(f)
    assert len(obj['ids'])==9947 and obj['ids'].is_unique
    return obj

def preflight():
    c=codecheck();res=resources()
    from python.utils.data_utils import get_fold_split,load_longitudinal_data
    from python.utils.survival_data import split_patient_ids
    from python.deephit.train import enforce_external_confirmations
    enforce_external_confirmations(c)
    _,out=load_longitudinal_data(c['data']['processed_dir'])
    statuses=out.assign(patient_id=out.patient_id.astype(str)).set_index('patient_id').eskd_status
    ref=cache();assert set(statuses.index)==set(ref['ids'])
    records=[]
    for fold in range(5):
        outer=list(map(str,get_fold_split(c['data']['splits_dir'],fold,'train')))
        test=list(map(str,get_fold_split(c['data']['splits_dir'],fold,'test')))
        assert len(outer)+len(test)==9947 and not set(outer)&set(test)
        tr,va=split_patient_ids(outer,validation_fraction=.2,seed=316+fold,strata=statuses.reindex(outer).astype(int).tolist())
        for rep in range(5):
            dest=taskdir(rep,fold);dest.mkdir(parents=True,exist_ok=False)
            tc=patient_counts(tr,replicate=rep,fold=fold,partition=0)
            vc=patient_counts(va,replicate=rep,fold=fold,partition=1)
            audit=audit_partitions(tr,va,test,tc,vc)
            assert statuses.reindex(list(tc)).nunique()==2 and statuses.reindex(list(vc)).nunique()==2
            roster=dict(replicate=rep,fold=fold,train_counts=tc,validation_counts=vc,audit=audit)
            write(dest/'private_roster.json',roster)
            cfg=json.loads(json.dumps(c));cfg['model']['name']=f'ddhigan_refit_pilot_rep{rep:02d}'
            cfg['data'].pop('query_roster_preflight_path',None)
            for key,sub in [('norm_stats_dir','norm_stats'),('checkpoint_dir','checkpoint'),('prediction_dir','prediction'),('training_dir','training'),('tensorboard_dir','tensorboard')]:
                cfg['output'][key]=str(dest/sub)
            (dest/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            records.append(dict(replicate=rep,fold=fold,**audit,config_sha256=sha(dest/'config.yaml'),roster_file_sha256=sha(dest/'private_roster.json')))
    write(ROOT/'run/preflight.json',dict(status='VALID',resources=res,tasks=records,reference_cache_sha256=sha(REFERENCE/'evaluation_cache.pkl'),base_config_sha256=sha(BASE),release_manifest_sha256=RELEASE))
    print('PREFLIGHT_VALID: 25 isolated fit tasks; exact original test folds',flush=True)

def loadtask(rep,fold):
    codecheck();dest=taskdir(rep,fold)
    pre=json.loads((ROOT/'run/preflight.json').read_text())
    record=next(t for t in pre['tasks'] if t['replicate']==rep and t['fold']==fold)
    assert sha(dest/'private_roster.json')==record['roster_file_sha256']
    assert sha(dest/'config.yaml')==record['config_sha256']
    return dest,yaml.safe_load((dest/'config.yaml').read_text()),json.loads((dest/'private_roster.json').read_text())

def install_training_hooks(rep,fold,*,dry=False):
    import torch
    from python.deephit import train as T
    from python.deephit.data_loader import PatientMultiplicityPrefixBatchSampler
    from python.utils.query_roster import query_roster_sha256
    dest,c,roster=loadtask(rep,fold);tc=roster['train_counts'];vc=roster['validation_counts']
    original_split=T.split_patient_ids;original_norm=T.compute_dynamic_norm_stats;original_dataset=T.DynamicDeepHitDataset
    audits={}
    def split(ids,validation_fraction,seed,strata):
        assert seed==316+fold and validation_fraction==.2
        tr,va=original_split(ids,validation_fraction,seed,strata)
        test=T.get_fold_split(c['data']['splits_dir'],fold,'test')
        assert audit_partitions(tr,va,test,tc,vc)==roster['audit']
        return sorted(tc),sorted(vc)
    def norm(*args,**kwargs):
        assert set(map(str,args[4]))==set(tc)
        assert kwargs.get('patient_multiplicity') is None
        kwargs['patient_multiplicity']=tc
        return original_norm(*args,**kwargs)
    class Dataset(original_dataset):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            ids=set(map(str,self.patient_ids))
            if ids==set(tc):label='train'
            elif ids==set(vc):
                label='validation'
                self.sample_weight=torch.as_tensor(validation_weights(self.patient_ids,self.sample_weight.numpy(),vc),dtype=self.sample_weight.dtype)
            else:raise ValueError('dataset does not match a sampled partition')
            audits[label]=dict(prefixes=len(self),patient_set_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),query_roster_sha256=query_roster_sha256(self.patient_ids,self.query_time.tolist()))
    def sampler(ids,times,batch_size,seed,selection):
        assert seed==316+fold and selection=='cycle'
        obj=PatientMultiplicityPrefixBatchSampler(ids,times,tc,batch_size,seed)
        assert obj.total_slots==roster['audit']['training_slots']
        return obj
    T.split_patient_ids=split;T.compute_dynamic_norm_stats=norm;T.DynamicDeepHitDataset=Dataset;T.PatientUniquePrefixBatchSampler=sampler
    return T,dest,c,roster,audits

def deep_preflight():
    import torch
    from torch.utils.data import DataLoader
    from python.deephit.data_loader import collate_fn
    from python.utils.time_grid import HalfYearTimeGrid
    T,dest,c,roster,audits=install_training_hooks(0,0,dry=True)
    long,out=T.load_longitudinal_data(c['data']['processed_dir'])
    formal=out[['patient_id','eskd_time','eskd_status']].rename(columns={'eskd_time':'event_time','eskd_status':'event_status'})
    outer=list(map(str,T.get_fold_split(c['data']['splits_dir'],0,'train')))
    strata=formal.assign(patient_id=formal.patient_id.astype(str)).set_index('patient_id').event_status.reindex(outer).astype(int).tolist()
    tr,va=T.split_patient_ids(outer,.2,316,strata)
    stats=T.compute_dynamic_norm_stats(long,formal,c['data']['static_cols'],c['data']['long_cols'],tr,min_visit_time=0.,anchor_nearest_t0=True,max_visit_time=5.,patient_multiplicity=None)
    assert all(np.isfinite([v['mean'],v['std']]).all() for v in stats.values())
    kwargs=dict(processed_dir=c['data']['processed_dir'],static_cols=c['data']['static_cols'],long_cols=c['data']['long_cols'],outcome='ESKD',max_visits=30,time_grid=HalfYearTimeGrid(max_time=11.,interval_width=.5),norm_stats=stats,evaluation_landmarks=[],include_actual_queries=True,max_query_time=5.,input_time_scale=10.,preserve_source_time=True,actual_query_end_tolerance=1e-10,min_history_time=0.,anchor_nearest_t0=True,include_time_delta=False)
    train=T.DynamicDeepHitDataset(patient_ids=tr,**kwargs);val=T.DynamicDeepHitDataset(patient_ids=va,**kwargs)
    sampler=T.PatientUniquePrefixBatchSampler(train.patient_ids,train.query_time,32,316,selection='cycle')
    from collections import Counter
    slots=Counter()
    for batch in sampler:
        ids=[str(train.patient_ids[i]) for i in batch]
        assert len(ids)==len(set(ids));slots.update(ids)
    assert dict(slots)==roster['train_counts']
    assert np.isclose(float(val.sample_weight.sum()),sum(roster['validation_counts'].values()),atol=.01)
    model=T.build_model(c).cuda()
    batch=next(iter(DataLoader(val,batch_size=32,collate_fn=collate_fn)))
    metrics=T.run_epoch(model,[batch],torch.device('cuda'),c,likelihood_only=True)
    assert np.isfinite(metrics['likelihood'])
    write(ROOT/'run/DEEP_PREFLIGHT.json',dict(status='VALID',task='rep0_fold0',audit=audits,training_slots=sum(slots.values()),validation_weight_sum=float(val.sample_weight.sum()),finite_forward_likelihood=True,training_steps=0))
    print('DEEP_PREFLIGHT_VALID: real data, weighted normalization/sampling/validation, finite forward pass; no fitting',flush=True)

def fit(rep,fold):
    T,dest,c,roster,audits=install_training_hooks(rep,fold)
    assert not list((dest/'checkpoint').glob('*.pt')),'existing checkpoint: no automatic overwrite'
    (dest/'fit_started').mkdir()
    sys.argv=['train','--config',str(dest/'config.yaml'),'--fold',str(fold),'--outcome','ESKD']
    T.main()
    assert set(audits)=={'train','validation'}
    prefix=c['model']['name'];summary=json.loads((dest/'training'/f'{prefix}_ESKD_fold{fold}.json').read_text())
    assert summary['train_patients']==len(roster['train_counts']) and summary['validation_patients']==len(roster['validation_counts'])
    assert summary['early_stopping'] and summary['effective_seed']==316+fold
    for label in audits:assert summary[label+'_patient_set_sha256']==audits[label]['patient_set_sha256']
    write(dest/'FIT_VALIDATED.json',dict(status='VALID',audit=audits,roster=roster['audit'],summary=summary))

def predict(rep,fold):
    import torch
    from torch.utils.data import DataLoader
    from python.deephit.predict import predict_landmark
    from python.deephit.data_loader import DynamicDeepHitDataset,collate_fn
    from python.deephit.train import build_model
    from python.utils.data_utils import load_norm_stats
    from python.utils.time_grid import HalfYearTimeGrid
    from scripts import evaluate_query12_recency as E
    dest,c,roster=loadtask(rep,fold)
    assert json.loads((dest/'FIT_VALIDATED.json').read_text())['status']=='VALID'
    prefix=c['model']['name'];checkpoint=dest/'checkpoint'/f'{prefix}_ESKD_fold{fold}.pt'
    stats=dest/'norm_stats'/f'fold_{fold}_{prefix}_ESKD_norm_stats.json'
    ck=torch.load(checkpoint,map_location='cuda');assert ck['config']==c
    model=build_model(c).cuda();model.load_state_dict(ck['model_state_dict'])
    grid=HalfYearTimeGrid(max_time=11.,interval_width=.5);obj=cache();rows=[];risk=[];inventory=[];groups=[]
    for q in range(6):
        reference=obj['cells'][0,fold,q];ids=obj['ids'][reference['indices']].tolist()
        assert not set(ids)&(set(roster['train_counts'])|set(roster['validation_counts']))
        ds=DynamicDeepHitDataset(processed_dir=c['data']['processed_dir'],patient_ids=ids,static_cols=c['data']['static_cols'],long_cols=c['data']['long_cols'],outcome='ESKD',max_visits=30,time_grid=grid,norm_stats=load_norm_stats(stats),evaluation_landmarks=[q],include_actual_queries=False,min_history_time=0.,anchor_nearest_t0=True,append_missing_query_row=False,input_time_scale=10.,preserve_source_time=True,actual_query_end_tolerance=1e-10)
        pred=predict_landmark(model,DataLoader(ds,batch_size=32,shuffle=False,collate_fn=collate_fn),q,[3,5,10],grid,torch.device('cuda'),residual_grid=list(range(1,11)),conditioning='query_survival').sort_values('patient_id')
        assert pred.patient_id.tolist()==ids
        for h in (3,5,10):
            np.testing.assert_allclose(pred[f'cond_surv_{h}y'],pred[f'resid_surv_{h:.1f}y'],rtol=0,atol=1e-12)
            pred[f'cond_surv_{h}y']=pred[f'resid_surv_{h:.1f}y']
        assert E.eligible_queries(pred.query_time,pred.last_observation_time,1).all()
        cell=E.build_cell(pred,reference['censoring'],obj['ids'],obj['outcomes'],q)
        counts=np.ones(9947,int);values,gs=E.metric(cell,counts,True);baseline,_=E.metric(reference,counts)
        assert np.isfinite(values).all()
        rows.append(dict(replicate=rep,fold=fold,query=q,**dict(zip(E.METRICS,values.tolist()))))
        groups.extend(dict(replicate=rep,fold=fold,query=q,**g) for g in gs)
        differences=reference['survival'][:,-1]-cell['survival'][:,-1]
        risk.append(dict(replicate=rep,fold=fold,query=q,n=len(ids),mean_signed_risk_difference=float(differences.mean()),mean_absolute_risk_difference=float(np.abs(differences).mean()),p95_absolute_risk_difference=float(np.quantile(np.abs(differences),.95)),maximum_absolute_risk_difference=float(np.abs(differences).max()),reference_metrics=dict(zip(E.METRICS,baseline.tolist()))))
        p=dest/'prediction'/f'query{q}.csv';p.parent.mkdir(exist_ok=True)
        with p.open('x') as f:pred.to_csv(f,index=False)
        inventory.append(dict(query=q,n=len(pred),sha256=sha(p),path=str(p)))
    write(dest/'EVALUATED.json',dict(status='VALID',replicate=rep,fold=fold,metrics=rows,groups=groups,risk=risk,inventory=inventory,checkpoint_sha256=sha(checkpoint),norm_sha256=sha(stats)))
    print(f'EVALUATED rep={rep} fold={fold}',flush=True)

def summarize():
    from scripts import evaluate_query12_recency as E
    allrows=[];risks=[];timings=[]
    for rep in range(5):
        for fold in range(5):
            dest,c,_=loadtask(rep,fold);j=json.loads((dest/'EVALUATED.json').read_text())
            assert j['status']=='VALID'
            for row in j['inventory']:assert sha(row['path'])==row['sha256']
            assert sha(dest/'checkpoint'/f"{c['model']['name']}_ESKD_fold{fold}.pt")==j['checkpoint_sha256']
            allrows+=j['metrics'];risks+=j['risk']
            s=json.loads((dest/'FIT_VALIDATED.json').read_text())['summary']
            timings.append(dict(replicate=rep,fold=fold,best_epoch=s['best_epoch'],stopped_after_epoch=s['stopped_after_epoch'],elapsed_seconds=s['elapsed_seconds']))
    out=ROOT/'aggregate_review';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(allrows);assert len(frame)==150
    selected=['auc_10y','ipcw_ibs_1_10y','calibration_mae_10y']
    query=frame.groupby(['replicate','query'])[selected].mean().reset_index()
    mean=query.groupby('replicate')[selected].mean()
    frame[['replicate','fold','query']+selected].to_csv(out/'evaluation_cells.csv',index=False)
    query.to_csv(out/'per_query_metrics.csv',index=False);mean.to_csv(out/'per_replicate_mean.csv')
    spread=mean.agg(['mean','std','min','max']);spread.to_csv(out/'descriptive_variability.csv')
    for row in risks:row.pop('reference_metrics')
    pd.DataFrame(risks).to_csv(out/'prediction_change_summary.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'training_timing.csv',index=False)
    write(out/'COMPLETE.json',dict(status='COMPLETE_VALIDATED',fits=25,prediction_cells=150,replicates=5,scope='fixed partitions and fixed configuration; descriptive pilot only',formal_ci=False,model_selection_repeated=False,manuscript_updated=False,summary=mean.reset_index().to_dict('records')))
    print('COMPLETE_VALIDATED: exploratory 5-replicate/25-fit pilot',flush=True)

def train():
    codecheck();res=resources()
    assert json.loads((ROOT/'run/DEEP_PREFLIGHT.json').read_text())['status']=='VALID'
    (ROOT/'run/training_started').mkdir()
    write(ROOT/'run/runner.json',dict(pid=os.getpid(),host=platform.node(),resources=res,started=time.time(),fit_budget=25))
    tasks=[(r,f) for r in range(5) for f in range(5)]
    def lane(gpu,items):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for rep,fold in items:
            with (taskdir(rep,fold)/'task.log').open('x') as log:
                for stage in ('fit','predict'):
                    subprocess.run([sys.executable,__file__,stage,'--replicate',str(rep),'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3*3600)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(lane,g,tasks[g::4]) for g in range(4)]
            for f in futures:f.result()
        summarize()
    except Exception as e:
        write(ROOT/'run/FAILED.json',dict(status='FAILED',error=repr(e)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','deep_preflight','train','fit','predict','summarize']);p.add_argument('--replicate',type=int);p.add_argument('--fold',type=int);a=p.parse_args()
    if a.stage in ('fit','predict'):globals()[a.stage](a.replicate,a.fold)
    else:globals()[a.stage]()
