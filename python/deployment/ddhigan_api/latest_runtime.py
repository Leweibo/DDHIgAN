"""Frozen physv15 inference: explicit model choice, no inherited recalibration."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from python.deephit.model import FormalDynamicDeepHit
from python.utils.query_conditioning import log_survival_from_logits,conditional_log_survival

STATIC=('age_at_biopsy','gender','baseline_CREA','baseline_CystatinC','baseline_ALB','baseline_log_PRO24H')
LONG=('CREA','CystatinC','ALB','log_PRO24H')

def prepare(request,stats,model_id):
    reduced=model_id=='DDHIgAN-CysC-free'
    static_cols=[c for c in STATIC if not(reduced and c=='baseline_CystatinC')]
    long_cols=[c for c in LONG if not(reduced and c=='CystatinC')]
    def protein(x):return None if x is None or x==0 else float(np.log(x))
    def labs(x):return dict(CREA=x.creatinine_mg_dl,CystatinC=x.cystatin_c_mg_l,ALB=x.albumin_g_l,log_PRO24H=protein(x.proteinuria_g_24h))
    raw=dict(age_at_biopsy=request.static.age_at_biopsy_years,gender=float(request.static.sex=='female'),**{'baseline_'+k:v for k,v in labs(request.static).items()})
    missing=[]
    def norm(v,c,where):
        if v is None:missing.append(where+'.'+c);return 0.
        return ((np.asarray([v],dtype=np.float32)-stats[c]['mean'])/stats[c]['std']).item()
    static=[norm(raw[c],c,'static') for c in static_cols]
    visits=request.visits;start=max(0,len(visits)-30);n=len(visits)-start
    values=np.zeros((1,30,1+len(static_cols)+len(long_cols)),np.float32)
    masks=np.zeros((1,30,len(long_cols)),np.float32);seq=np.zeros((1,30),np.float32)
    for i,v in enumerate(visits[start:]):
        j=i+start;values[0,i,0]=0. if i==0 else np.float32(v.time_years-visits[j-1].time_years)/np.float32(10)
        values[0,i,1:1+len(static_cols)]=static;lv=labs(v)
        for k,c in enumerate(long_cols):
            values[0,i,1+len(static_cols)+k]=norm(lv[c],c,f'visits[{j}]');masks[0,i,k]=float(lv[c] is None)
    seq[0,:n]=1
    return dict(x_values=values,x_missing=masks,sequence_mask=seq,query_time=np.array([request.query_time_years],np.float32),missing_fields=sorted(set(missing)),history_processing=dict(received_visits=len(visits),encoded_visits=n,omitted_visits=start))

class LatestRuntime:
    def __init__(self,bundle):
        self.bundle=Path(bundle);self.api_release=self.bundle.resolve().parent.name
        checks={}
        for line in (self.bundle/'SHA256SUMS').read_text().splitlines():
            digest,name=line.split(maxsplit=1);name=name.strip();path=self.bundle/name
            if path.parent != self.bundle or not path.is_file():raise ValueError('invalid bundle member')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('bundle checksum mismatch')
            checks[name]=digest
        if set(checks)!={'latest_manifest.json','core_weights.pt','cysc_free_weights.pt'}:raise ValueError('incomplete bundle')
        self.manifest=json.loads((self.bundle/'latest_manifest.json').read_text())
        assert self.manifest['format']=='ddhigan_physv15_selectable_v1'
        assert self.manifest['cohort']=='physv15_outcome_recheck_9947_20260908'
        assert self.manifest['aggregation']=='five_fold_mean'
        self.provenance={'model_version':self.manifest['version']}
        self.models={};torch.set_num_threads(2)
        for name,item in self.manifest['models'].items():
            assert name in ('DDHIgAN','DDHIgAN-CysC-free')
            path=self.bundle/item['weights'];assert path.parent==self.bundle
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            states=torch.load(path,map_location='cpu',weights_only=True)
            assert len(states)==len(item['normalizations'])==5
            cfg=item['architecture'];self.models[name]=[]
            assert cfg['num_event_bins']==22 and cfg['longitudinal_dim']==(3 if name.endswith('free') else 4)
            for state,stats in zip(states,item['normalizations']):
                assert set(stats)==set(c for c in (*STATIC,*LONG) if not(name.endswith('free') and 'CystatinC' in c))
                assert all(np.isfinite(s['mean']) and np.isfinite(s['std']) and s['std']>0 for s in stats.values())
                model=FormalDynamicDeepHit(**cfg);model.load_state_dict(state,strict=True);model.eval();self.models[name].append((model,stats))
        assert set(self.models)=={'DDHIgAN','DDHIgAN-CysC-free'}
    def predict(self,request):
        name=request.model_id;curves=[];h=np.arange(1,11,dtype=float)
        with torch.inference_mode():
            for model,stats in self.models[name]:
                p=prepare(request,stats,name)
                output=model(*(torch.from_numpy(p[k]) for k in ['x_values','x_missing','sequence_mask']),query_time=torch.from_numpy(p['query_time']))
                ls=log_survival_from_logits(output['logits'].cpu().numpy(),22)
                # The API query is the last real visit, hence conditioning delay is zero.
                curves.append(-np.expm1(conditional_log_survival(ls,np.arange(23)*.5,np.zeros(1),h))[0])
        risk=np.mean(curves,axis=0);assert np.isfinite(risk).all() and (risk>=0).all() and (risk<=1).all() and (np.diff(risk)>=-1e-12).all()
        late=request.query_time_years>5
        return dict(schema_version='1.0',model_id=name,model_version=self.provenance['model_version'],api_release=self.api_release,aggregation='five_fold_mean',calibrator_version=None,predictive_uncertainty=None,future_risk_curve=[dict(years=int(t),risk=float(v)) for t,v in zip(h,risk)],risk_3y=float(risk[2]),risk_5y=float(risk[4]),risk_10y=float(risk[9]),history_processing=p['history_processing'],warnings=dict(missing=p['missing_fields'],distribution=[],out_of_distribution=late,late_followup_extrapolation='Query exceeds the internally validated 0–5-year range.' if late else None))
    def model_info(self):
        return dict(model_version=self.provenance['model_version'],api_release=self.api_release,models=list(self.models),aggregation='five_fold_mean',internal_support_years=11,calibration='none',individual_confidence_interval=False,validation_scope='Internal validation used each patient\'s held-out fold; the deployment mean is not that held-out predictor.')
