"""Add only CysC-free metrics; reuse six-model values, censoring and draws verbatim."""
import argparse, json, pickle, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_rankzero_standard as S
E=S.E
NAME='physv15_rankzero_sevenmodel_20260911'
OLD=Path('results/physv15_rankzero_standard_20260911')
E.ROOT=Path('results')/NAME; E.OUT=E.ROOT/'aggregate_review'
E.VARIANTS=('DDHIgAN-CysC-free',)
MODELS=('DDHIgAN','Baseline Cox','PCCox','DynForest','DDHIgAN-Pathology','DDHIgAN-Expanded','DDHIgAN-CysC-free')

def evaluate(counts,collect=False):
    values=np.full((1,5,6,8),np.nan);units=[];groups=[]
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
    E.OUT.mkdir(parents=True,exist_ok=True)
    pre=json.loads((OLD/'aggregate_review/PREPARED.json').read_text())
    assert E.sha(OLD/'evaluation_cache.pkl')==pre['cache_sha256']
    assert E.sha(S.DRAW)==S.DRAW_SHA
    manifest=json.loads((OLD/'aggregate_review/DELIVERY_MANIFEST.json').read_text())
    for n,d in manifest['files'].items():assert E.sha(OLD/'aggregate_review'/n)==d['sha256'],n
    with (OLD/'evaluation_cache.pkl').open('rb') as h:cache=pickle.load(h)
    assert len(cache['cells'])==180 and len(cache['ids'])==9947
    # The old cache and old aggregate stay read-only; only a reference digest is written.
    (E.ROOT/'patient_draws.npy').symlink_to(S.DRAW.resolve())
    E.write(E.OUT/'REFERENCE_READY.json',dict(status='VALID',cache_sha256=pre['cache_sha256'],draw_sha256=S.DRAW_SHA,old_manifest=manifest))
    print('REFERENCE_VALID',flush=True)

def prepare():
    assert json.loads((E.ROOT/'run/TRAINING_COMPLETE.json').read_text())['status']=='FIVE_FOLDS_VALIDATED'
    ref=json.loads((E.OUT/'REFERENCE_READY.json').read_text())
    assert E.sha(OLD/'evaluation_cache.pkl')==ref['cache_sha256']
    with (OLD/'evaluation_cache.pkl').open('rb') as h:cache=pickle.load(h)
    cells={};inventory=[];training=[]
    prior=json.loads(Path('results/physv15_query15_boundaryfix_20260908/run/preflight.json').read_text())
    prefix='ddhigan_cysc_free_'+NAME;root=E.ROOT/'ddhigan_cysc_free'
    for f in range(5):
        valid=json.loads((E.ROOT/'run'/f'cysc_free_fold{f}_validated.json').read_text())
        assert E.sha(root/'checkpoint'/f'{prefix}_ESKD_fold{f}.pt')==valid['checkpoint_sha256']
        summary=json.loads((root/'training'/f'{prefix}_ESKD_fold{f}.json').read_text())
        for k in ('train_patient_set_sha256','validation_patient_set_sha256','train_query_roster_sha256','validation_query_roster_sha256','train_prefixes','validation_prefixes'):
            assert summary[k]==prior['splits'][f][k],k
        training.append(dict(fold=f,checkpoint_sha256=valid['checkpoint_sha256'],rosters_match=True))
        for q in range(6):
            p=root/'prediction'/f'{prefix}_ESKD_fold{f}_test_landmark{q}.csv'
            x=pd.read_csv(p,dtype={'patient_id':str}).sort_values('patient_id')
            core=cache['cells'][0,f,q]
            assert x.patient_id.is_unique and x.patient_id.tolist()==cache['ids'][core['indices']].tolist()
            assert E.eligible_queries(x.query_time,x.last_observation_time,1).all()
            assert (x.prediction_delay+10<=11+1e-12).all()
            cell=E.build_cell(x,core['censoring'],cache['ids'],cache['outcomes'],q)
            for key in ('indices','time','event','alive','weights'):np.testing.assert_array_equal(cell[key],core[key])
            cells[0,f,q]=cell
            inventory.append(dict(model=MODELS[-1],fold=f,query=q,n=len(x),sha256=E.sha(p)))
    assert len(cells)==30
    E._CELLS=cells
    observed,units,groups=evaluate(np.ones(9947,int),True)
    cache['cells']=cells
    with (E.ROOT/'evaluation_cache.pkl').open('xb') as h:pickle.dump(cache,h)
    np.save(E.OUT/'observed_new.npy',observed)
    # Preserve all existing six-model aggregate rows and numbers.
    for name,new in [('evaluation_cells.csv',units),('calibration_groups.csv',groups)]:
        old=pd.read_csv(OLD/'aggregate_review'/name)
        pd.concat([old,pd.DataFrame(new)],ignore_index=True).to_csv(E.OUT/name,index=False)
    for n in ('risk_sets.json','ipcw_diagnostics.json'):shutil.copyfile(OLD/'aggregate_review'/n,E.OUT/n)
    E.write(E.OUT/'PREPARED.json',dict(status='VALID',evaluation_cells=210,new_cells=30,reused_cells=180,inventory=inventory,training_validation=training,cache_sha256=E.sha(E.ROOT/'evaluation_cache.pkl')))
    print('PREPARED_VALID_210',flush=True)

def merge():
    reps=[];metrics=[]
    for task in sorted((E.ROOT/'bootstrap_shards').iterdir()):
        done=json.loads((task/'COMPLETE.json').read_text());assert E.sha(task/'metrics.npz')==done['metrics_sha256']
        a=np.load(task/'metrics.npz');reps.extend(a['replicate_ids']);metrics.extend(a['metrics'])
    order=np.argsort(reps);reps=np.array(reps)[order];new=np.array(metrics)[order]
    np.testing.assert_array_equal(reps,np.arange(2000));assert new.shape==(2000,1,7,8) and np.isfinite(new).all()
    old=np.load(OLD/'aggregate_review/bootstrap_aggregate.npz');np.testing.assert_array_equal(reps,old['replicate_ids'])
    draws=np.concatenate([old['metrics'],new],axis=1)
    obs=np.concatenate([np.load(OLD/'aggregate_review/observed.npy'),np.load(E.OUT/'observed_new.npy')])
    np.testing.assert_array_equal(draws[:,:6],old['metrics'])
    e=json.loads((OLD/'aggregate_review/six_model_evidence.json').read_text())
    assert e['models']==list(MODELS[:6]);e['models']=list(MODELS)
    def summary(point,bs,delta=False):
        result={}
        for q in range(7):
            result[f'landmark_{q}' if q<6 else 'mean']={}
            for j,m in enumerate(E.METRICS):
                lo,hi=np.quantile(bs[:,q,j],[.025,.975])
                result[f'landmark_{q}' if q<6 else 'mean'][('delta_' if delta else '')+m]=dict(estimate=float(point[q,j]),ci_lower=float(lo),ci_upper=float(hi),bootstrap_mean=float(bs[:,q,j].mean()))
        return result
    e['absolute'][MODELS[-1]]=summary(obs[6],draws[:,6])
    e['paired_vs_ddhigan'][MODELS[-1]]=summary(obs[6]-obs[0],draws[:,6]-draws[:,0],True)
    gs=pd.read_csv(E.OUT/'calibration_groups.csv');gs=gs[gs.variant==MODELS[-1]]
    for (q,h,g),z in gs.groupby(['query','horizon','group']):
        assert len(z)==5
        e['calibration_curves'].append(dict(model=MODELS[-1],landmark_years=int(q),horizon_years=int(h),group=int(g),mean_predicted_risk=float(z.predicted.mean()),observed_risk=float(z.observed.mean())))
    e['comparison_scope']='Seven models, four rank-zero DDHIgAN versions, 11-year support, common <=1-year record recency; variant minus DDHIgAN'
    e['status']='RESULTS_AWAITING_HUMAN_CONFIRMATION'
    E.write(E.OUT/'seven_model_evidence.json',e)
    np.save(E.OUT/'observed.npy',obs);np.savez_compressed(E.OUT/'bootstrap_aggregate.npz',replicate_ids=reps,metrics=draws)
    ref=json.loads((E.OUT/'REFERENCE_READY.json').read_text())
    for n,d in ref['old_manifest']['files'].items():assert E.sha(OLD/'aggregate_review'/n)==d['sha256']
    assert E.sha(OLD/'evaluation_cache.pkl')==ref['cache_sha256']
    E.write(E.OUT/'COMPLETE.json',dict(status='COMPLETE_VALIDATED',replicates=2000,cells=210,new_fits=5,reused_fits=True,old_six_values_unchanged=True,draw_sha256=S.DRAW_SHA,replicate_ids_equal=True))
    print('SEVEN_MODEL_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','prepare','bootstrap','merge']);p.add_argument('--host');p.add_argument('--jobs',type=int,default=16);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=2000);a=p.parse_args()
    if a.stage=='bootstrap':E.bootstrap(a)
    else:globals()[a.stage]()
