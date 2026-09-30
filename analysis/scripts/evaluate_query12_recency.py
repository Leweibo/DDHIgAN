"""Aggregate comparison with one shared patient bootstrap; private cache stays RIS."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from python.evaluation.query12_recency import eligible_queries
from python.evaluation.uno_ibs_cluster_bootstrap import _auc, _uno_c_index, _censoring_step, _step_query, _ibs
from python.utils.metrics import calibration_groups

NAME='physv15_query12_recency_20260910'
ROOT=Path('results')/NAME
OUT=ROOT/'aggregate_review'
V='physv15_outcome_recheck_9947_20260908'
DATA=Path('data/processed')/V
SPLITS=Path('data/splits')/V
NODES=np.array([1.,2.,3.,5.,7.,10.])
METRICS=('uno_c_index','ipcw_ibs_1_10y','auc_3y','auc_5y','auc_10y','calibration_mae_3y','calibration_mae_5y','calibration_mae_10y')
VARIANTS=('15y_all','15y_gap1','15y_gap2','12y_gap1','12y_gap2')
_CELLS=None
_DRAWS=None


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def km(t,e,h):
    values,inverse,n=np.unique(t,return_inverse=True,return_counts=True)
    d=np.bincount(inverse,weights=e,minlength=len(values));risk=np.cumsum(n[::-1])[::-1]
    return float(1-np.prod(1-d[values<=h]/risk[values<=h]))


def metric(cell,counts,collect=False):
    idx=np.repeat(np.arange(len(cell['indices'])),counts[cell['indices']])
    t,e,s=cell['time'][idx],cell['event'][idx],cell['survival'][idx]
    if len(t)<5:raise ValueError('fewer than five evaluation rows')
    y,w=cell['alive'][idx],cell['weights'][idx]
    r=1-s;g=cell['censoring']
    uno=_uno_c_index(t,e,r[:,-1],g,10.)
    auc=[_auc(t,e,r[:,j],g,NODES[j]) for j in (2,3,5)]
    ibs=float(np.mean(w*(y-s)**2,axis=0)@np.array([.5,1,1.5,2,2.5,1.5])/9)
    maes=[];groups=[]
    for j in (2,3,5):
        errors=[]
        for group,ix in enumerate(np.array_split(np.argsort(r[:,j],kind='stable'),5),1):
            observed=km(t[ix],e[ix],NODES[j]);pred=float(r[ix,j].mean())
            errors.append(abs(pred-observed))
            if collect:groups.append(dict(horizon=int(NODES[j]),group=group,n=len(ix),predicted=pred,observed=observed))
        maes.append(np.mean(errors))
    return np.array([uno,ibs,*auc,*maes]),groups


def build_cell(x,censor,ids,outcomes,q):
    x=x.sort_values('patient_id');t=outcomes.loc[x.patient_id,'eskd_time'].to_numpy(float)-q
    e=outcomes.loc[x.patient_id,'eskd_status'].to_numpy(int)
    np.testing.assert_allclose(x.residual_time,t,atol=1e-9,rtol=0)
    np.testing.assert_array_equal(x.true_event,e)
    s=x[[f'resid_surv_{h:.1f}y' for h in NODES]].to_numpy(float)
    assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all() and (np.diff(s,axis=1)<=1e-12).all()
    for h in (3,5,10):np.testing.assert_array_equal(x[f'cond_surv_{h}y'],x[f'resid_surv_{h:.1f}y'])
    y=t[:,None]>NODES;cases=(t[:,None]<=NODES)&(e[:,None]==1)
    event_g=np.ones(len(t));use=cases.any(axis=1)
    event_g[use]=_step_query(*censor,t[use],before=True)
    hg=_step_query(*censor,NODES)
    assert (event_g>0).all() and (hg>0).all()
    w=cases/event_g[:,None]+y/hg
    return dict(indices=ids.get_indexer(x.patient_id),time=t,event=e,survival=s,alive=y.astype(float),weights=w,censoring=censor)


def reference():
    assert platform.node() in ('node2','node3','fat2')
    OUT.mkdir(parents=True,exist_ok=True)
    assert sha(DATA/'release_manifest.json')=='d2ddd4b402568e4371fda34e23b5135ce7e5a47185ccf72fed5b13e8feb2f20b'
    outcomes=pd.read_csv(DATA/'igan_outcomes.csv',dtype={'patient_id':str}).set_index('patient_id').sort_index()
    ids=outcomes.index;assert len(ids)==9947 and ids.is_unique
    inv=json.loads(Path('results/physv15_query15_boundaryfix_20260908/aggregate_review/output_validation.json').read_text())
    hashes={(x['fold'],x['landmark']):x['sha256'] for x in inv['inventory'] if x['branch']=='prediction'}
    cells={};rows=[];inventory=[];rosters={}
    for f in range(5):
        tr=pd.read_csv(SPLITS/f'fold_{f}_train.csv',dtype={'patient_id':str}).patient_id
        te=pd.read_csv(SPLITS/f'fold_{f}_test.csv',dtype={'patient_id':str}).patient_id
        assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(ids)
        for q in range(6):
            train=outcomes.loc[tr];train=train[train.eskd_time>q]
            g=_censoring_step(train.eskd_time.to_numpy(float)-q,train.eskd_status.to_numpy(int))
            p=Path('results/physv15_query15_boundaryfix_20260908/ddhigan/prediction')/f'ddhigan_physv15_query15_boundaryfix_20260908_ESKD_fold{f}_test_landmark{q}.csv'
            assert sha(p)==hashes[f,q]
            x=pd.read_csv(p,dtype={'patient_id':str}).sort_values('patient_id')
            expected=outcomes.loc[te];expected=expected[expected.eskd_time>q]
            assert set(x.patient_id)==set(expected.index) and x.patient_id.is_unique
            inventory.append(dict(fold=f,query=q,sha256=sha(p),n=len(x)))
            for vi,gap in enumerate((None,1,2)):
                use=np.ones(len(x),bool) if gap is None else eligible_queries(x.query_time,x.last_observation_time,gap)
                z=x[use];cells[vi,f,q]=build_cell(z,g,ids,outcomes,q)
                rosters[vi,f,q]=z.patient_id.tolist()
                rows.append(dict(variant=VARIANTS[vi],fold=f,query=q,original_n=len(x),n=len(z),excluded_n=len(x)-len(z),events=int(z.true_event.sum()),
                                 events_by_10y=int(((z.residual_time<=10)&(z.true_event==1)).sum()),followed_beyond_10y=int((z.residual_time>10).sum()),min_G_1_10=float(_step_query(*g,NODES).min())))
    with (ROOT/'reference_cache.pkl').open('xb') as f:pickle.dump(dict(cells=cells,ids=ids,rosters=rosters,outcomes=outcomes),f)
    # One global draw table, reused unchanged across both model supports and gaps.
    rng=np.random.default_rng(316)
    draws=np.asarray([rng.multinomial(len(ids),np.full(len(ids),1/len(ids))) for _ in range(2000)],dtype=np.uint16)
    with (ROOT/'patient_draws.npy').open('xb') as f:np.save(f,draws)
    pd.DataFrame(rows).to_csv(OUT/'risk_sets.csv',index=False)
    write(OUT/'REFERENCE_READY.json',dict(status='VALID',reference_inventory=inventory,draw_sha256=sha(ROOT/'patient_draws.npy'),draws=2000,seed=316,
                                        private_cache_sha256=sha(ROOT/'reference_cache.pkl'),outcomes_sha256=sha(DATA/'igan_outcomes.csv')))
    print('REFERENCE_AND_UNIQUE_DRAWS_READY',flush=True)


def prepare():
    assert json.loads((ROOT/'run/TRAINING_COMPLETE.json').read_text())['status']=='FIVE_FOLDS_VALIDATED'
    with (ROOT/'reference_cache.pkl').open('rb') as f:cache=pickle.load(f)
    cells,ids=cache['cells'],cache['ids'];inventory=[];checks=[]
    for f in range(5):
        summary=json.loads((ROOT/'ddhigan/training'/f'ddhigan_{NAME}_ESKD_fold{f}.json').read_text())
        prior=json.loads(Path(f'results/physv15_query15_boundaryfix_20260908/ddhigan/training/ddhigan_physv15_query15_boundaryfix_20260908_ESKD_fold{f}.json').read_text())
        for key in ('train_patient_set_sha256','validation_patient_set_sha256','train_query_roster_sha256','validation_query_roster_sha256','train_patients','validation_patients','train_prefixes','validation_prefixes'):
            assert summary[key]==prior[key],key
        checks.append(dict(fold=f,split_roster_match=True))
        for q in range(6):
            p=ROOT/'ddhigan/prediction'/f'ddhigan_{NAME}_ESKD_fold{f}_test_landmark{q}.csv'
            x=pd.read_csv(p,dtype={'patient_id':str}).sort_values('patient_id')
            assert x.patient_id.tolist()==cache['rosters'][2,f,q]
            assert eligible_queries(x.query_time,x.last_observation_time,2).all()
            inventory.append(dict(fold=f,query=q,sha256=sha(p),n=len(x)))
            for vi,gap,ref in ((3,1,1),(4,2,2)):
                z=x[eligible_queries(x.query_time,x.last_observation_time,gap)]
                assert z.patient_id.tolist()==cache['rosters'][ref,f,q]
                cells[vi,f,q]=build_cell(z,cells[ref,f,q]['censoring'],ids,cache['outcomes'],q)
                np.testing.assert_array_equal(cells[vi,f,q]['indices'],cells[ref,f,q]['indices'])
    cache['cells']=cells
    with (ROOT/'evaluation_cache.pkl').open('xb') as f:pickle.dump(cache,f)
    global _CELLS;_CELLS=cells
    observed,unit,groups=evaluate(np.ones(len(ids),int),collect=True)
    np.save(OUT/'observed.npy',observed)
    pd.DataFrame(unit).to_csv(OUT/'evaluation_cells.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'calibration_groups.csv',index=False)
    # Independent existing metric implementations on all 150 observed cells.
    maximum=0.
    for key,cell in cells.items():
        vals,_=metric(cell,np.ones(len(ids),int))
        expected=_ibs(cell['time'],cell['event'],cell['survival'],cell['censoring'])
        maximum=max(maximum,abs(vals[1]-expected))
        for pos,j in enumerate((2,3,5)):
            gs=calibration_groups(cell['time'],cell['event'],1-cell['survival'][:,j],NODES[j],n_groups=5)
            m=np.mean([abs(g['mean_predicted_risk']-g['observed_risk']) for g in gs])
            maximum=max(maximum,abs(vals[5+pos]-m))
    assert maximum<1e-12
    # q=0 has unchanged eligibility under both thresholds.
    np.testing.assert_array_equal(observed[0,0],observed[1,0]);np.testing.assert_array_equal(observed[1,0],observed[2,0])
    np.testing.assert_array_equal(observed[3,0],observed[4,0])
    write(OUT/'PREPARED.json',dict(status='VALID',evaluation_cells=150,split_checks=checks,prediction_inventory=inventory,independent_metric_max_diff=maximum,cache_sha256=sha(ROOT/'evaluation_cache.pkl')))


def evaluate(counts,collect=False):
    values=np.empty((5,5,6,8));units=[];groups=[]
    for (vi,f,q),cell in _CELLS.items():
        values[vi,f,q],gs=metric(cell,counts,collect)
        if collect:
            units.append(dict(variant=VARIANTS[vi],fold=f,query=q,**dict(zip(METRICS,values[vi,f,q].tolist()))))
            groups.extend(dict(variant=VARIANTS[vi],fold=f,query=q,**g) for g in gs)
    mean=values.mean(axis=1)
    result=np.concatenate((mean,mean.mean(axis=1,keepdims=True)),axis=1)
    return result,units,groups


def worker(i):
    return i,evaluate(_DRAWS[i])[0]


def bootstrap(args):
    import psutil
    assert platform.node()==args.host and args.host in ('fat2','node2','node3')
    physical=psutil.cpu_count(logical=False);reserve=8 if args.host=='fat2' else 4
    busy=[psutil.cpu_percent(interval=1) for _ in range(3)]
    assert args.jobs<=physical-reserve and physical*(1-max(busy)/100)>=args.jobs+reserve
    assert psutil.virtual_memory().available>(4+args.jobs)*2**30
    prepared=json.loads((OUT/'PREPARED.json').read_text());assert prepared['status']=='VALID'
    assert sha(ROOT/'evaluation_cache.pkl')==prepared['cache_sha256']
    assert sha(ROOT/'patient_draws.npy')==json.loads((OUT/'REFERENCE_READY.json').read_text())['draw_sha256']
    global _CELLS,_DRAWS
    with (ROOT/'evaluation_cache.pkl').open('rb') as f:_CELLS=pickle.load(f)['cells']
    _DRAWS=np.load(ROOT/'patient_draws.npy',mmap_mode='r')
    assert _DRAWS.shape==(2000,9947) and np.all(_DRAWS.sum(axis=1)==9947)
    assert 0<=args.start<args.stop<=2000
    task=ROOT/'bootstrap_shards'/f'{args.start:04d}_{args.stop:04d}';task.mkdir(parents=True)
    write(task/'RUNNING.json',dict(host=args.host,pid=os.getpid(),jobs=args.jobs,start=args.start,stop=args.stop,draw_sha256=sha(ROOT/'patient_draws.npy')))
    begin=time.monotonic();rows=[];rep=[]
    with mp.get_context('fork').Pool(args.jobs) as pool:
        for i,result in pool.imap_unordered(worker,range(args.start,args.stop),chunksize=2):
            rep.append(i);rows.append(result)
    order=np.argsort(rep);np.savez_compressed(task/'metrics.npz',replicate_ids=np.array(rep)[order],metrics=np.array(rows)[order])
    write(task/'COMPLETE.json',dict(status='VALID',count=len(rep),elapsed_seconds=time.monotonic()-begin,metrics_sha256=sha(task/'metrics.npz')))


def merge():
    reps=[];metrics=[]
    for task in sorted((ROOT/'bootstrap_shards').iterdir()):
        done=json.loads((task/'COMPLETE.json').read_text());assert sha(task/'metrics.npz')==done['metrics_sha256']
        a=np.load(task/'metrics.npz');reps.extend(a['replicate_ids']);metrics.extend(a['metrics'])
    order=np.argsort(reps);reps=np.array(reps)[order];draws=np.array(metrics)[order]
    np.testing.assert_array_equal(reps,np.arange(2000));assert np.isfinite(draws).all()
    observed=np.load(OUT/'observed.npy');rows=[]
    contrasts=[('12y_minus_15y_gap1',3,1),('12y_minus_15y_gap2',4,2),('15y_gap1_minus_all',1,0),('15y_gap2_minus_all',2,0)]
    comparisons=[(v,observed[i],draws[:,i]) for i,v in enumerate(VARIANTS)]
    comparisons += [(n,observed[a]-observed[b],draws[:,a]-draws[:,b]) for n,a,b in contrasts]
    for name,estimate,bs in comparisons:
        for q in range(7):
            for j,m in enumerate(METRICS):
                lo,hi=np.quantile(bs[:,q,j],[.025,.975])
                rows.append(dict(comparison=name,query=str(q) if q<6 else 'mean',metric=m,estimate=estimate[q,j],ci_lower=lo,ci_upper=hi,bootstrap_mean=bs[:,q,j].mean()))
    pd.DataFrame(rows).to_csv(OUT/'comparisons.csv',index=False)
    np.savez_compressed(OUT/'bootstrap_aggregate.npz',replicate_ids=reps,metrics=draws)
    write(OUT/'COMPLETE.json',dict(status='COMPLETE_VALIDATED',replicates=2000,comparison_rows=len(rows),variants=list(VARIANTS),metrics=list(METRICS),base_model_fits=5,recalibration=False,
                                bootstrap_refits=False,draw_sha256=sha(ROOT/'patient_draws.npy')))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','prepare','bootstrap','merge']);p.add_argument('--host');p.add_argument('--jobs',type=int,default=16);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=2000);args=p.parse_args()
    if args.stage=='reference':reference()
    elif args.stage=='prepare':prepare()
    elif args.stage=='bootstrap':bootstrap(args)
    else:merge()


if __name__=='__main__':main()
