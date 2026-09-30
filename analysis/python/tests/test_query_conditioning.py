import copy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from python.utils.query_conditioning import conditional_log_survival, log_survival_from_logits
from python.utils.time_grid import HalfYearTimeGrid
from python.utils.data_utils import build_dynamic_prefix_inputs
from python.deephit.data_loader import collate_fn
from python.deephit.predict import predict_landmark, prediction_landmark_label


class QueryConditioningTests(unittest.TestCase):
    def setUp(self):
        self.grid = HalfYearTimeGrid(15)
        self.t = self.grid.boundaries
        self.log_s = (-.03 * self.t ** 2)[None, :]

    def test_formula_non_grid_queries_and_zero_delay(self):
        out = conditional_log_survival(self.log_s, self.t, [1.13], [0, 3, 10])
        expected = np.interp(np.array([1.13, 4.13, 11.13]), self.t, self.log_s[0]) - np.interp(1.13, self.t, self.log_s[0])
        np.testing.assert_allclose(out[0], expected)
        np.testing.assert_array_equal(conditional_log_survival(self.log_s, self.t, [0], self.t), self.log_s)

    def test_exponential_memorylessness(self):
        logs = np.repeat((-.2 * self.t)[None, :], 3, axis=0)
        actual = conditional_log_survival(logs, self.t, [0, 1.13, 5], [3, 10])
        np.testing.assert_allclose(actual, np.tile([-.6, -2], (3, 1)))

    def test_sequential_conditioning(self):
        shifted = np.r_[0., self.t[self.t > 1.13] - 1.13]
        first = conditional_log_survival(self.log_s, self.t, [1.13], shifted)
        twice = conditional_log_survival(first, shifted, [2.54], [1, 3, 10])
        direct = conditional_log_survival(self.log_s, self.t, [3.67], [1, 3, 10])
        np.testing.assert_allclose(twice, direct)

    def test_tail_stability_endpoint_and_monotonicity(self):
        logits = np.full((1, 31), -900.)
        logits[0, 0] = 0
        logs = log_survival_from_logits(logits, 30)
        out = conditional_log_survival(logs, self.t, [5], [0, 1, 3, 10])
        self.assertTrue(np.isfinite(out).all())
        self.assertAlmostEqual(np.exp(out[0, -1]), 1 / 21)
        risk = -np.expm1(out)
        self.assertTrue(np.all((risk >= 0) & (risk <= 1)))
        self.assertTrue(np.all(np.diff(risk) >= 0))
        np.testing.assert_equal(conditional_log_survival(logs, self.t, [15], [0]), [[0]])

    def test_invalid_inputs_fail(self):
        for delay, horizons in [([-1],[1]), ([np.nan],[1]), ([5],[10.01]), ([0],[-1]), ([0],[np.inf])]:
            with self.subTest(delay=delay,horizons=horizons), self.assertRaises(ValueError):
                conditional_log_survival(self.log_s, self.t, delay, horizons)
        for bad in [self.log_s * -1, np.full_like(self.log_s,np.nan), np.full_like(self.log_s,-np.inf)]:
            with self.assertRaises(ValueError):
                conditional_log_survival(bad,self.t,[0],[1])
        for logits in [np.zeros((1,30)),np.full((1,31),np.nan)]:
            with self.assertRaises(ValueError): log_survival_from_logits(logits,30)

    def build(self, rows, query, support=15, modern=True):
        return build_dynamic_prefix_inputs(
            rows, pd.DataFrame({'patient_id':['synthetic'], 'event_time':[20.], 'event_status':[1]}),
            ['synthetic'], [], ['marker'], 30, HalfYearTimeGrid(support),
            evaluation_landmarks=[query], norm_stats={'marker':{'mean':0.,'std':1.}},
            include_actual_queries=False, append_missing_query_row=False,
            input_time_scale=10., preserve_source_time=modern)

    def test_history_isolation_precision_and_new_observation(self):
        q=1.123456789
        rows=pd.DataFrame({'patient_id':['synthetic']*3,'visit_time':[0.,q,q+1e-9],'marker':[1.,2.,999.]})
        data=self.build(rows,q)
        self.assertEqual(data['query_time'][0],q)
        self.assertEqual(data['last_observation_time'][0],q)
        self.assertEqual(data['sequence_mask'].sum(),2)
        changed=rows.copy(); changed.loc[2,'marker']=-999.
        np.testing.assert_array_equal(data['x_values'],self.build(changed,q)['x_values'])
        before=self.build(rows,q-1e-9)
        self.assertEqual(before['sequence_mask'].sum(),1)
        self.assertNotEqual(before['last_observation_time'][0],q)

    def test_fixed_input_scale_and_administrative_target(self):
        rows=pd.DataFrame({'patient_id':['synthetic']*2,'visit_time':[0.,1.],'marker':[1.,2.]})
        old=self.build(rows,1,support=10,modern=False)
        new=self.build(rows,1)
        np.testing.assert_array_equal(old['x_values'],new['x_values'])
        self.assertEqual(old['target_time'][0],10)
        self.assertEqual(new['target_time'][0],15)
        self.assertEqual(new['target_event'][0],0)
        self.assertEqual(HalfYearTimeGrid(15).administrative_target([11.],[1])[1][0],1)
        self.assertEqual(HalfYearTimeGrid(10).administrative_target([11.],[1])[1][0],0)

    def test_no_synthetic_history(self):
        rows=pd.DataFrame({'patient_id':['synthetic'],'visit_time':[1.],'marker':[2.]})
        with self.assertRaises(ValueError): self.build(rows,0)

    def prediction(self, q, mode):
        rows=pd.DataFrame({'patient_id':['synthetic']*2,'visit_time':[0.,1.],'marker':[1.,2.]})
        data=self.build(rows,q)
        items=[{k: v[0] if k=='patient_id' else torch.from_numpy(v)[0] for k,v in data.items()}]
        mass=np.r_[1-np.exp(self.log_s[0,1]), np.exp(self.log_s[0,1:-1])-np.exp(self.log_s[0,2:]), np.exp(self.log_s[0,-1])]
        class Fixed(torch.nn.Module):
            def forward(self,*args,**kwargs):
                return {'logits':torch.tensor(np.log(mass)[None,:]),'survival':torch.tensor(np.exp(self_outer.log_s[:,1:])), 'cif':torch.tensor(1-np.exp(self_outer.log_s[:,1:]))}
        self_outer=self
        return predict_landmark(Fixed(),DataLoader(items,collate_fn=collate_fn),q,[3,5,10],self.grid,'cpu',residual_grid=list(range(1,11)),conditioning=mode)

    def test_original_bug_and_candidate_outputs(self):
        a=self.prediction(1,'legacy'); b=self.prediction(2,'legacy')
        self.assertEqual(a.risk_score[0],b.risk_score[0])
        a=self.prediction(1,'query_survival'); b=self.prediction(2,'query_survival')
        self.assertAlmostEqual(a.cond_surv_10y[0],np.exp(-3))
        self.assertAlmostEqual(b.cond_surv_10y[0],np.exp(-.03*(11**2-1)))
        self.assertAlmostEqual(b.risk_score[0],1-b.cond_surv_10y[0])
        diagnostic=self.prediction(2,'unconditioned_diagnostic')
        self.assertAlmostEqual(a.risk_score[0],diagnostic.risk_score[0])
        cols=[f'resid_surv_{h:.1f}y' for h in range(1,11)]
        self.assertTrue(np.all(np.diff(b[cols].to_numpy(),axis=1)<=0))

    def test_external_candidate_horizon_contract(self):
        for horizons, curve in [([.5],[1]),([10],[.5]),([2.5],[1])]:
            with self.assertRaises(ValueError):
                predict_landmark(torch.nn.Identity(), [], 0, horizons, self.grid, "cpu", curve, "query_survival")

    def test_non_grid_query_output_names_do_not_collide(self):
        queries=[0.,.5,1.,1.12345671,1.12345672,5.]
        labels=[prediction_landmark_label(q,"query_survival") for q in queries]
        self.assertEqual(len(set(labels)),len(queries))
        self.assertEqual(list(map(float,labels)),queries)
        self.assertEqual(prediction_landmark_label(1.,"query_survival"),"1")
        self.assertEqual(prediction_landmark_label(.5,"legacy"),"0")


if __name__=='__main__': unittest.main()
