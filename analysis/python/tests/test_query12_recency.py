import unittest
import numpy as np
from python.evaluation.query12_recency import eligible_queries
from python.utils.query_conditioning import conditional_log_survival


class RecencyTests(unittest.TestCase):
    def test_inclusive_boundary_and_nested_subsets(self):
        q = np.array([0., 1., 2., 2.00000001, 5., 5.])
        v = np.array([0., 0., 0., 0., 3., 4.])
        one, two = [eligible_queries(q, v, gap) for gap in (1, 2)]
        np.testing.assert_array_equal(one, [1, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(two, [1, 1, 1, 0, 1, 1])
        self.assertTrue(np.all(~one | two))

    def test_invalid_history_rejected(self):
        for q, v in [(1, 2), (1, np.nan), (0, -1)]:
            with self.assertRaises(ValueError): eligible_queries(q, v, 2)

    def test_twelve_year_support_covers_exact_two_year_gap(self):
        grid = np.arange(0, 12.5, .5)
        s = np.tile(-.1 * grid, (3, 1))
        delay = np.array([0., 1., 2.])
        actual = conditional_log_survival(s, grid, delay, [1, 3, 5, 10])
        np.testing.assert_allclose(actual, np.tile(-.1*np.array([1, 3, 5, 10]), (3, 1)))
        with self.assertRaises(ValueError):
            conditional_log_survival(s[:1], grid, [2.00000001], [10])

    def test_baseline_conditioning_is_identity(self):
        grid = np.arange(0, 12.5, .5)
        s = (-.03 * grid ** 1.5)[None, :]
        actual = conditional_log_survival(s, grid, [0.], [1, 5, 10])
        np.testing.assert_array_equal(actual[0], s[0, [2, 10, 20]])


if __name__ == '__main__': unittest.main()
