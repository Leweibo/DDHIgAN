import os,unittest,copy
import numpy as np
from pydantic import ValidationError
from python.deployment.ddhigan_api.schemas import PredictionRequest
from python.deployment.ddhigan_api.latest_runtime import prepare,STATIC,LONG,LatestRuntime

def payload(model='DDHIgAN',n=2):
 b=dict(creatinine_mg_dl=1.,albumin_g_l=40.,proteinuria_g_24h=1.)
 if model=='DDHIgAN':b['cystatin_c_mg_l']=1.
 return dict(schema_version='1.0',model_id=model,query_time_years=(n-1)/10,kidney_failure_free_at_query=True,static=dict(age_at_biopsy_years=40,sex='female',**b),visits=[dict(time_years=i/10,**b) for i in range(n)])
class ContractTests(unittest.TestCase):
 def test_cysc_rejected_not_ignored(self):
  d=payload('DDHIgAN-CysC-free');d['static']['cystatin_c_mg_l']=1
  with self.assertRaises(ValidationError):PredictionRequest(**d)
 def test_unknown_model_and_identifier_rejected(self):
  for key,value in [('model_id','unknown'),('patient_id','synthetic-id')]:
   d=payload();d[key]=value
   with self.assertRaises(ValidationError):PredictionRequest(**d)
 def test_future_and_duplicate_records_rejected(self):
  for t in [-1,0,1]:
   d=payload();d['visits'][-1]['time_years']=t
   with self.assertRaises(ValidationError):PredictionRequest(**d)
 def test_strict_log_zero_missing_and_truncation_boundary(self):
  d=payload(n=40);d['static']['proteinuria_g_24h']=0;d['visits'][-1]['proteinuria_g_24h']=np.e
  stats={c:dict(mean=0.,std=1.) for c in (*STATIC,*LONG)}
  p=prepare(PredictionRequest(**d),stats,'DDHIgAN')
  self.assertEqual(p['history_processing']['omitted_visits'],10)
  self.assertEqual(p['x_values'][0,0,0],0.)
  self.assertAlmostEqual(p['x_values'][0,1,0],.01)
  self.assertEqual(p['x_values'][0,0,6],0)
  self.assertAlmostEqual(p['x_values'][0,29,10],1.)
  self.assertIn('static.baseline_log_PRO24H',p['missing_fields'])
 def test_matches_training_encoder(self):
  try:
   import pandas as pd
   from python.utils.data_utils import build_dynamic_prefix_inputs
  except ImportError:
   self.skipTest("training encoder available only in research checkout")
  from python.utils.time_grid import HalfYearTimeGrid
  for name in ['DDHIgAN','DDHIgAN-CysC-free']:
   for count in [1,2,40]:
    d=payload(name,count);req=PredictionRequest(**d)
    sc=[c for c in STATIC if not(name.endswith('free') and 'CystatinC' in c)]
    lc=[c for c in LONG if not(name.endswith('free') and 'CystatinC' in c)]
    stats={c:dict(mean=.1234567,std=1.234567) for c in sc+lc}
    static=dict(age_at_biopsy=40.,gender=1.,baseline_CREA=1.,baseline_CystatinC=1.,baseline_ALB=40.,baseline_log_PRO24H=0.)
    rows=[dict(patient_id='synthetic',visit_time=v['time_years'],CREA=1.,CystatinC=1.,ALB=40.,log_PRO24H=0.,**static) for v in d['visits']]
    reference=build_dynamic_prefix_inputs(pd.DataFrame(rows),pd.DataFrame([dict(patient_id='synthetic',event_time=20.,event_status=0)]),['synthetic'],static_cols=sc,long_cols=lc,max_visits=30,time_grid=HalfYearTimeGrid(max_time=11.,interval_width=.5),evaluation_landmarks=[req.query_time_years],norm_stats=stats,include_actual_queries=False,anchor_nearest_t0=True,append_missing_query_row=False,preserve_source_time=True,input_time_scale=10.)
    actual=prepare(req,stats,name)
    for key in ['x_values','x_missing','sequence_mask']:np.testing.assert_array_equal(actual[key],reference[key])
 def test_reduced_channels(self):
  cols=[c for c in (*STATIC,*LONG) if 'CystatinC' not in c]
  p=prepare(PredictionRequest(**payload('DDHIgAN-CysC-free')),{c:dict(mean=0.,std=1.) for c in cols},'DDHIgAN-CysC-free')
  self.assertEqual(p['x_values'].shape,(1,30,9));self.assertEqual(p['x_missing'].shape,(1,30,3))
 @unittest.skipUnless(os.environ.get('DDHIGAN_BUNDLE_DIR'),'bundle integration test')
 def test_real_bundles_auth_model_routing_and_monotonicity(self):
  os.environ['DDHIGAN_API_KEY']='synthetic-test-only-key'
  from python.deployment.ddhigan_api.app import app
  from fastapi.testclient import TestClient
  c=TestClient(app);self.assertEqual(c.get('/health/ready').status_code,200)
  self.assertEqual(c.post('/ddhigan/v1/predict',json=payload()).status_code,401)
  risks=[]
  for model in ['DDHIgAN','DDHIgAN-CysC-free']:
   for n in [1,2,40,256]:
    r=c.post('/ddhigan/v1/predict',json=payload(model,n),headers={'X-API-Key':os.environ['DDHIGAN_API_KEY']});self.assertEqual(r.status_code,200,r.text)
    out=r.json();self.assertEqual(out['model_id'],model);self.assertIsNone(out['predictive_uncertainty']);self.assertIsNone(out['calibrator_version'])
    y=[v['risk'] for v in out['future_risk_curve']];self.assertTrue(np.all(np.diff(y)>=0));self.assertTrue(all(0<=v<=1 for v in y));self.assertEqual(out['warnings']['out_of_distribution'],n==256)
    if n==2:risks.append(y)
  self.assertFalse(np.allclose(*risks))
if __name__=='__main__':unittest.main()
