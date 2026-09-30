"""Scientific invariants for reducing predictors without deleting visits."""
from pathlib import Path
import copy
import unittest
import numpy as np
import pandas as pd
import yaml
from python.utils.data_utils import build_dynamic_prefix_inputs
from python.utils.time_grid import HalfYearTimeGrid

class ReducedInputs(unittest.TestCase):
    def test_config_and_missing_only_visit(self):
        core=yaml.safe_load(Path('config/physv15_rankzero_standard_20260911/core.yaml').read_text())
        reduced=yaml.safe_load(Path('config/physv15_rankzero_sevenmodel_20260911/cysc_free.yaml').read_text())
        self.assertEqual(core['training'],reduced['training'])
        self.assertEqual(core['prediction'],reduced['prediction'])
        self.assertEqual(reduced['data']['static_cols'],[c for c in core['data']['static_cols'] if c!='baseline_CystatinC'])
        self.assertEqual(reduced['data']['long_cols'],[c for c in core['data']['long_cols'] if c!='CystatinC'])
        rows=[]
        for t in [0.,.5,1.,2.]:
            row=dict(patient_id='synthetic',visit_time=t)
            row.update({c:1. for c in core['data']['static_cols']+core['data']['long_cols']})
            if t==.5:
                for c in ['CREA','ALB','log_PRO24H']:row[c]=np.nan
            rows.append(row)
        df=pd.DataFrame(rows);out=pd.DataFrame([dict(patient_id='synthetic',event_time=12.,event_status=1)])
        def build(c):
            return build_dynamic_prefix_inputs(df,out,['synthetic'],c['data']['static_cols'],c['data']['long_cols'],30,HalfYearTimeGrid(max_time=11.,interval_width=.5),evaluation_landmarks=list(range(6)),include_actual_queries=True,max_query_time=5.,min_history_time=0.,anchor_nearest_t0=True,append_missing_query_row=False,input_time_scale=10.,preserve_source_time=True,actual_query_end_tolerance=1e-10)
        a,b=build(core),build(reduced)
        for key in ['patient_id','query_time','last_observation_time','sequence_mask','target_time','target_event','evaluation_time','evaluation_event']:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        self.assertIn(.5,b['query_time'])
        self.assertEqual(b['x_values'].shape[-1],9)
        self.assertEqual(b['x_missing'].shape[-1],3)
        np.testing.assert_array_equal(a['x_values'][...,[0,1,2,3,5,6,7,9,10]],b['x_values'])
        np.testing.assert_array_equal(a['x_missing'][...,[0,2,3]],b['x_missing'])

    def test_auxiliary_head(self):
        from python.deephit.train import build_model
        c=yaml.safe_load(Path('config/physv15_rankzero_sevenmodel_20260911/cysc_free.yaml').read_text())
        model=build_model(c)
        self.assertEqual(model.longitudinal_head.out_features,3)
        self.assertEqual(model.missing_dim,3)
        self.assertEqual(model.value_dim,9)

if __name__=='__main__':unittest.main()
