# Code style: AI-code. AI-generated statistical controls for shape closure.
"""Check source reuse, donor normalization and invariance to repeated rows."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from shift_mixing_shape_closure import OBSERVABLES, prepare_rows, shape_closure


def events(count):
    return [dict(event_key=f"campaign:qcd_0to1:1:1:{index}",
                 stratum="qcd_0to1", partition="train", detector_weight=1.,
                 activity={"nCSCSegments": 3}) for index in range(count)]


def row(first, second, weight=1.):
    values = dict(mass=1.5 if first % 2 else 3., pt=1.5 if first % 2 else 3.,
                  deltaPhi=.75, openingAngle=.02, deltaEta=.75, ptAsymmetry=.3)
    return dict(anchor_event=first, donor_event=second, weight=weight,
                kind="same_os" if first == second else "event_os", pair=values)


class ShapeClosureTest(unittest.TestCase):
    def test_exact_factorized_graph_preserves_shared_shape_in_every_replica(self):
        # Every anchor supplies one shape; sum over all its eligible donors.
        # With the same source multiplier and resampled donor normalization,
        # every bootstrap real and mixed shape must agree, despite source reuse.
        source = events(20)
        same = [row(index, index) for index in range(20)]
        mixed = [row(first, second, 1. / 19.) for first in range(20)
                 for second in range(20) if first != second]
        result = shape_closure(same, mixed, source, 50, 123)
        self.assertAlmostEqual(result["mixed_model_over_real_rate"], 1.)
        self.assertEqual(result["real_effective_originals"], 20.)
        for variable in OBSERVABLES:
            record = result["observables"][variable]
            np.testing.assert_allclose(record["difference"], 0., atol=1.e-14)
            np.testing.assert_allclose(record["difference_std"], 0., atol=1.e-14)

    def test_repeated_combinations_do_not_create_statistical_precision(self):
        source = events(8)
        same = [row(index, index) for index in range(8)]
        mixed = [row(index, (index + 1) % 8) for index in range(8)]
        repeated = [dict(item, weight=item["weight"] / 10.) for item in mixed for _ in range(10)]
        first = shape_closure(same, mixed, source, 60, 45)
        second = shape_closure(same, repeated, source, 60, 45)
        for variable in OBSERVABLES:
            for name in ("real_shape", "mixed_shape", "difference_std", "difference_covariance"):
                np.testing.assert_allclose(first["observables"][variable][name],
                                           second["observables"][variable][name], atol=2.e-15)

    def test_mixed_self_pairs_and_uncovered_domains_are_rejected(self):
        invalid = dict(row(0, 0), kind="event_os")
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepare_rows([invalid], 2)
        outside = row(0, 1)
        outside["pair"]["mass"] = 500.
        with self.assertRaisesRegex(ValueError, "domain"):
            prepare_rows([outside], 2)

    def test_extra_matching_pools_must_keep_each_original_once(self):
        source = events(4)
        same = [row(index, index) for index in range(4)]
        mixed = [row(index, (index + 1) % 4) for index in range(4)]
        with self.assertRaisesRegex(ValueError, "partition all"):
            shape_closure(same, mixed, source, 5, 1,
                          donor_pools={"first": [0, 1, 2], "second": [2, 3]})


if __name__ == "__main__":
    unittest.main()
