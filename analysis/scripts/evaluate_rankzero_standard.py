"""Six fixed models, common recent-record risk sets and original patient draws."""
import argparse,json,pickle,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_query12_recency as E
NAME='physv15_rankzero_standard_20260911'
E.ROOT=Path('results')/NAME;E.OUT=E.ROOT/'aggregate_review'
E.VARIANTS=('DDHIgAN','Baseline Cox','PCCox','DynForest','DDHIgAN-Pathology','DDHIgAN-Expanded')
CORE=Path('results/physv15_inner_ranking_20260910/evaluation_attempt2')
DRAW=Path('results/physv15_query12_recency_20260910/patient_draws.npy')
DRAW_SHA='88222ce77f8cd1ad9afc0ccca3bc3135cb5cb77bb701205b25e1e9666f11e89a'

ORIGINAL_BUILD=E.build_cell
PRECISION=[]
def build_cell(x,censor,ids,outcomes,q):
    x=x.copy()
    for h in (3,5,10):
        canonical=x[f'resid_surv_{h:.1f}y'].to_numpy()
        diff=np.abs(x[f'cond_surv_{h}y'].to_numpy()-canonical)
        assert np.isfinite(diff).all() and diff.max()<=1e-12
        PRECISION.append(float(diff.max()))
        x[f'cond_surv_{h}y']=canonical
    return ORIGINAL_BUILD(x,censor,ids,outcomes,q)
E.build_cell=build_cell

def evaluate(counts,collect=False):
    values=np.full((6,5,6,8),np.nan);units=[];groups=[]
    for (vi,f,q),cell in E._CELLS.items():
        values[vi,f,q],gs=E.metric(cell,counts,collect)
        if collect:
            units.append(dict(variant=E.VARIANTS[vi],fold=f,query=q,**dict(zip(E.METRICS,values[vi,f,q].tolist()))))
            groups.extend(dict(variant=E.VARIANTS[vi],fold=f,query=q,**g) for g in gs)
    assert np.isfinite(values).all()
    mean=values.mean(axis=1)
    return np.concatenate((mean,mean.mean(axis=1,keepdims=True)),axis=1),units,groups
E.evaluate=evaluate

def reference():
    assert platform.node() in ('fat2','node2','node3')
    E.OUT.mkdir(parents=True,exist_ok=True)
    pre=json.loads((CORE/'aggregate_review/PREPARED.json').read_text())
    assert E.sha(CORE/'evaluation_cache.pkl')==pre['cache_sha256'] and E.sha(DRAW)==DRAW_SHA
    assert set(json.loads((CORE/'selection/LOCKED.json').read_text())['choices'].values())=={'rank000'}
    old=Path('results/physv15_standard11_recency1_20260910')
    previous=json.loads((old/'aggregate_review/PREPARED.json').read_text())
    assert E.sha(old/'evaluation_cache.pkl')==previous['cache_sha256']
    with (old/'evaluation_cache.pkl').open('rb') as f:cache=pickle.load(f)
    with (CORE/'evaluation_cache.pkl').open('rb') as f:selected=pickle.load(f)
    np.testing.assert_array_equal(cache['ids'],selected['ids'])
    cells={(vi,f,q):cache['cells'][vi,f,q] for vi in range(4) for f in range(5) for q in range(6)}
    for f in range(5):
        for q in range(6):
            np.testing.assert_array_equal(cells[0,f,q]['indices'],selected['cells'][1,f,q]['indices'])
            cells[0,f,q]=selected['cells'][1,f,q]
    inventory=[r for r in previous['inventory'] if r['model'] in E.VARIANTS[1:4]]
    assert len(inventory)==90
    cache['cells']=cells
    with (E.ROOT/'reference_cache.pkl').open('xb') as f:pickle.dump(cache,f)
    (E.ROOT/'patient_draws.npy').symlink_to(DRAW.resolve())
    E.write(E.OUT/'REFERENCE_READY.json',dict(status='VALID',draw_sha256=DRAW_SHA,draws=2000,seed=316,inventory=inventory,core_cache_sha256=pre['cache_sha256'],cache_sha256=E.sha(E.ROOT/'reference_cache.pkl')))
    print('REFERENCE_VALID: reused core and 90 clinical predictions',flush=True)

def prepare():
    assert json.loads((E.ROOT/'run/TRAINING_COMPLETE.json').read_text())['status']=='TEN_FOLDS_VALIDATED'
    ref=json.loads((E.OUT/'REFERENCE_READY.json').read_text());assert E.sha(E.ROOT/'reference_cache.pkl')==ref['cache_sha256']
    with (E.ROOT/'reference_cache.pkl').open('rb') as f:cache=pickle.load(f)
    cells=cache['cells'];inventory=list(ref['inventory'])
    prior=json.loads(Path('results/physv15_query15_boundaryfix_20260908/run/preflight.json').read_text())
    training=[]
    for vi,branch in [(4,'pathology'),(5,'expanded')]:
        prefix='ddhigan_'+branch+'_'+NAME;root=E.ROOT/('ddhigan_'+branch)
        for f in range(5):
            valid=json.loads((E.ROOT/'run'/f'{branch}_fold{f}_validated.json').read_text())
            assert E.sha(root/'checkpoint'/f'{prefix}_ESKD_fold{f}.pt')==valid['checkpoint_sha256']
            summary=json.loads((root/'training'/f'{prefix}_ESKD_fold{f}.json').read_text())
            for k in ('train_patient_set_sha256','validation_patient_set_sha256','train_query_roster_sha256','validation_query_roster_sha256','train_prefixes','validation_prefixes'):
                assert summary[k]==prior['splits'][f][k],k
            training.append(dict(branch=branch,**valid,training_summary=summary))
            for q in range(6):
                p=root/'prediction'/f'{prefix}_ESKD_fold{f}_test_landmark{q}.csv'
                x=pd.read_csv(p,dtype={'patient_id':str}).sort_values('patient_id')
                assert x.patient_id.is_unique and x.patient_id.tolist()==cache['ids'][cells[0,f,q]['indices']].tolist()
                assert E.eligible_queries(x.query_time,x.last_observation_time,1).all() and (x.prediction_delay+10<=11).all()
                cells[vi,f,q]=E.build_cell(x,cells[0,f,q]['censoring'],cache['ids'],cache['outcomes'],q)
                inventory.append(dict(model=E.VARIANTS[vi],fold=f,query=q,sha256=E.sha(p),n=len(x)))
    assert len(cells)==180
    E._CELLS=cells
    observed,units,groups=evaluate(np.ones(9947,int),True)
    np.testing.assert_allclose(observed[0],np.load(CORE/'aggregate_review/observed.npy')[1],rtol=0,atol=1e-12)
    with (E.ROOT/'evaluation_cache.pkl').open('xb') as f:pickle.dump(cache,f)
    np.save(E.OUT/'observed.npy',observed)
    pd.DataFrame(units).to_csv(E.OUT/'evaluation_cells.csv',index=False)
    pd.DataFrame(groups).to_csv(E.OUT/'calibration_groups.csv',index=False)
    diagnostics=[];risksets=[]
    for q in range(6):
        risksets.append(dict(query=q,n=sum(len(cells[0,f,q]['indices']) for f in range(5))))
        for j in (2,3,5):
            w=np.concatenate([cells[0,f,q]['weights'][:,j] for f in range(5)]);w=w[w>0]
            diagnostics.append(dict(landmark_years=q,horizon_years=int(E.NODES[j]),n_nonzero=len(w),maximum_weight=float(w.max()),p99_weight=float(np.quantile(w,.99)),effective_sample_size=float(w.sum()**2/(w*w).sum())))
    E.write(E.OUT/'ipcw_diagnostics.json',diagnostics);E.write(E.OUT/'risk_sets.json',risksets)
    E.write(E.OUT/'training_validation.json',training)
    E.write(E.OUT/'PREPARED.json',dict(status='VALID',evaluation_cells=180,inventory=inventory,core_reused=True,cache_sha256=E.sha(E.ROOT/'evaluation_cache.pkl')))
    E.write(E.OUT/'precision_validation.json',dict(max_abs=max(PRECISION),tolerance=1e-12,canonical_changed=False));print('PREPARED_VALID: 180 aligned cells',flush=True)

def merge():
    reps=[];metrics=[]
    for task in sorted((E.ROOT/'bootstrap_shards').iterdir()):
        done=json.loads((task/'COMPLETE.json').read_text());assert E.sha(task/'metrics.npz')==done['metrics_sha256']
        a=np.load(task/'metrics.npz');reps.extend(a['replicate_ids']);metrics.extend(a['metrics'])
    order=np.argsort(reps);reps=np.array(reps)[order];draws=np.array(metrics)[order]
    np.testing.assert_array_equal(reps,np.arange(2000));assert draws.shape==(2000,6,7,8) and np.isfinite(draws).all()
    old=np.load(CORE/'aggregate_review/bootstrap_aggregate.npz')['metrics'][:,1]
    np.testing.assert_allclose(draws[:,0],old,rtol=0,atol=1e-12)
    obs=np.load(E.OUT/'observed.npy');e=dict(status='VALID',cohort=E.V,models=list(E.VARIANTS),n_patients=9947,folds=5,landmarks=list(range(6)),absolute={},paired_vs_ddhigan={},calibration_curves=[],bootstrap=dict(replicates=2000,seed=316,draw_sha256=DRAW_SHA),comparison_scope='11-year DDHIgAN family; latest record within one year for all six models; ranking weight zero in all DDHIgAN variants; support and ranking selected after internal exploration')
    rows=[]
    def summaries(point,bs,delta=False):
        result={}
        for q in range(7):
            label=f'landmark_{q}' if q<6 else 'mean';result[label]={}
            for j,m in enumerate(E.METRICS):
                lo,hi=np.quantile(bs[:,q,j],[.025,.975]);result[label][('delta_' if delta else '')+m]=dict(estimate=float(point[q,j]),ci_lower=float(lo),ci_upper=float(hi),bootstrap_mean=float(bs[:,q,j].mean()))
        return result
    for vi,model in enumerate(E.VARIANTS):
        e['absolute'][model]=summaries(obs[vi],draws[:,vi])
        if vi:e['paired_vs_ddhigan'][model]=summaries(obs[vi]-obs[0],draws[:,vi]-draws[:,0],True)
    gs=pd.read_csv(E.OUT/'calibration_groups.csv')
    for keys,g in gs.groupby(['variant','query','horizon','group']):
        model,q,h,group=keys;assert len(g)==5
        e['calibration_curves'].append(dict(model=model,landmark_years=int(q),horizon_years=int(h),group=int(group),mean_predicted_risk=float(g.predicted.mean()),observed_risk=float(g.observed.mean())))
    E.write(E.OUT/'six_model_evidence.json',e)
    np.savez_compressed(E.OUT/'bootstrap_aggregate.npz',replicate_ids=reps,metrics=draws)
    E.write(E.OUT/'COMPLETE.json',dict(status='COMPLETE_VALIDATED',replicates=2000,cells=180,core_bootstrap_reproduced=True,new_fits=10,core_refit=False,draw_sha256=DRAW_SHA))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','prepare','bootstrap','merge']);p.add_argument('--host');p.add_argument('--jobs',type=int,default=100);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=2000);a=p.parse_args()
    if a.stage=='bootstrap':E.bootstrap(a)
    else:globals()[a.stage]()
