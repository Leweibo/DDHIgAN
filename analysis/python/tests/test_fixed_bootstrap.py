import unittest
import numpy as np
from python.deephit.fixed_bootstrap import draw_counts, audit, signed_optimism


class FixedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.folds = [[f'{f}_{i}' for i in range(60)] for f in range(5)]

    def test_shared_patient_draw_and_fixed_outer_sizes(self):
        a = draw_counts(self.folds, 0)
        self.assertEqual(a, draw_counts([list(reversed(f)) for f in self.folds], 0))
        self.assertNotEqual(a, draw_counts(self.folds, 1))
        self.assertEqual(sum(a.values()), 300)
        for f in self.folds:
            self.assertEqual(sum(a.get(p, 0) for p in f), 60)
        # The same patient has exactly the same multiplicity in every role/fold.
        for outer in range(5):
            development = sum((f for j, f in enumerate(self.folds) if j != outer), [])
            self.assertEqual(sum(a.get(p, 0) for p in development), 240)

    def test_partition_leakage_rejected(self):
        with self.assertRaises(ValueError):
            audit(['a'], ['a'], ['c'], {'a': 2}, {'a': 1})
        with self.assertRaises(ValueError):
            draw_counts([['a']]*5, 0)
        value = audit(['a','b'], ['c'], ['d'], {'a': 3}, {'c': 1})
        self.assertEqual(value['training_slots'], 3)

    def test_optimism_direction_for_auc_and_loss(self):
        # AUC decreases and loss increases after correction of optimistic apparent values.
        a = [[.92, .03], [.94, .02]]
        o = [[.90, .04], [.90, .05]]
        result = signed_optimism(a, o, [.93, .025])
        np.testing.assert_allclose(result['corrected'], [.90, .045])
        np.testing.assert_allclose(result['monte_carlo_se'], [.01, .01])

    def test_zero_and_negative_optimism_not_clipped(self):
        a = [[.8], [.8]]
        result = signed_optimism(a, [[.81], [.83]], [.8])
        np.testing.assert_allclose(result['corrected'], [.82])
        with self.assertRaises(ValueError):
            signed_optimism([[float('nan')], [.8]], a, [.8])


if __name__ == '__main__':
    unittest.main()
