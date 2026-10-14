# Code style: AI-code. AI-generated independent parent-accounting tests.
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "vertex_efficiency", Path(__file__).parents[1] / "utils/shift_vertex_efficiency.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ids(events):
    return np.array([[1, 10, event] for event in events], dtype=np.uint64).reshape(-1, 3)


class VertexEfficiencyTests(unittest.TestCase):
    def summarize(self, parents=(1, 2, 3, 4), weights=(1, 1, 1, 1),
                  pairs=(1, 1, 1, 2, 3), labels=(1, 1, 0, 1, -1),
                  scores=(0.4, 0.8, 0.9, 0.3, 0.99), cut=0.5):
        return module.summarize_vertex_efficiency(
            ids(parents), np.asarray(weights, dtype=float), ids(pairs),
            np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=float), cut
        )

    def test_multiple_candidates_count_a_parent_once(self):
        report = self.summarize()
        self.assertEqual(report["counts"]["GEN_parents"], 4)
        self.assertEqual(report["counts"]["parents_without_candidates"], 1)
        self.assertEqual(report["full_parent_baseline"]["empirical"]["passed"], 2)
        self.assertEqual(report["full_parent_selected"]["empirical"]["efficiency"], 0.25)
        self.assertEqual(report["conditional_event_retention"]["empirical"]["efficiency"], 0.5)
        self.assertEqual(report["genuine_pair_retention"]["empirical"]["efficiency"], 1 / 3)
        self.assertEqual(report["classes"]["fake"]["selected_pairs"], 1)
        self.assertEqual(report["classes"]["unknown"]["selected_pairs"], 1)
        json.dumps(report, allow_nan=False)

    def test_duplicate_candidates_are_preserved_not_deduplicated(self):
        report = self.summarize(parents=(1, 2), weights=(1, 1), pairs=(1, 1),
                                labels=(1, 1), scores=(0.5, 0.5))
        self.assertEqual(report["classes"]["genuine"]["selected_pairs"], 2)
        self.assertEqual(report["classes"]["genuine"]["selected_parents"], 1)
        self.assertEqual(report["full_parent_selected"]["empirical"]["efficiency"], 0.5)
        self.assertEqual(report["genuine_pair_retention"]["empirical"]["total"], 2)
        self.assertIsNone(report["genuine_pair_retention"]["empirical"]["wilson95"])
        self.assertEqual(
            report["genuine_pair_retention"]["empirical"]["uncertainty_reason"],
            "multiple_genuine_candidates_share_GEN_parent",
        )
        self.assertIsNotNone(report["full_parent_selected"]["empirical"]["wilson95"])

    def test_unknown_or_fake_only_parent_is_not_genuine_selected(self):
        report = self.summarize(parents=(1, 2), weights=(1, 1), pairs=(1, 2),
                                labels=(0, -1), scores=(0.99, 0.99))
        self.assertEqual(report["full_parent_selected"]["empirical"]["passed"], 0)
        self.assertIsNone(report["conditional_event_retention"]["empirical"]["efficiency"])

    def test_signal_qualification_preserves_non_signal_genuine_labels(self):
        report = module.summarize_vertex_efficiency(
            ids((1, 2, 3)), np.ones(3), ids((1, 2)), np.array([1, 1]),
            np.array([0.9, 0.9]), 0.5, qualified_genuine_mask=np.array([True, False]),
        )
        self.assertEqual(report["classes"]["genuine"]["pairs"], 2)
        self.assertEqual(report["classes"]["genuine"]["selected_parents"], 2)
        self.assertEqual(report["classes"]["qualified_genuine"]["pairs"], 1)
        self.assertEqual(report["classes"]["unknown"]["pairs"], 0)
        self.assertEqual(report["full_parent_selected"]["empirical"]["passed"], 1)
        self.assertEqual(report["full_parent_selected"]["empirical"]["total"], 3)
        self.assertEqual(report["conditional_event_retention"]["empirical"]["total"], 1)
        self.assertEqual(report["genuine_pair_retention"]["empirical"]["total"], 1)

    def test_invalid_qualification_masks_rejected(self):
        for mask in (np.array([True]), np.array([1, 0]), np.array([False, True])):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "qualified_genuine_mask"):
                module.summarize_vertex_efficiency(
                    ids((1, 2)), np.ones(2), ids((1, 2)), np.array([1, -1]),
                    np.array([0.9, 0.9]), 0.5, qualified_genuine_mask=mask,
                )

    def test_weighted_event_and_pair_ratios_and_parent_influences(self):
        # Parent1 has two genuine candidates, one selected, weight2.
        # Parent2 has one unselected candidate, weight1; parent3 has none, weight3.
        report = self.summarize(parents=(1, 2, 3), weights=(2, 1, 3),
                                pairs=(1, 1, 2), labels=(1, 1, 1), scores=(0.8, 0.2, 0.1))
        selected = report["full_parent_selected"]["weighted"]
        self.assertAlmostEqual(selected["efficiency"], 1 / 3)
        self.assertEqual(selected["denominator_sumw"], 6)
        self.assertEqual(selected["denominator_sumw2"], 14)
        self.assertAlmostEqual(selected["n_eff"], 36 / 14)
        # Influences are (4/3,-1/3,-1), divided by denominator6.
        self.assertAlmostEqual(selected["standard_error"], np.sqrt(26) / 18)
        conditional = report["conditional_event_retention"]["weighted"]
        self.assertAlmostEqual(conditional["efficiency"], 2 / 3)
        self.assertAlmostEqual(conditional["standard_error"], np.sqrt(8) / 9)
        pair = report["genuine_pair_retention"]["weighted"]
        self.assertAlmostEqual(pair["efficiency"], 2 / 5)
        # Parent-level numerator/denominator are(1,2),(0,1): influences .4,-.4.
        self.assertAlmostEqual(pair["standard_error"], np.sqrt(0.32) / 5)
        self.assertAlmostEqual(pair["n_eff"], 25 / 17)
        self.assertNotIn("wilson95", pair)

    def test_common_weight_scaling_preserves_ratio_uncertainty(self):
        first = self.summarize(weights=(2, 1, 3, 4))
        second = self.summarize(weights=(14, 7, 21, 28))
        for name in ("full_parent_baseline", "full_parent_selected",
                     "conditional_event_retention", "genuine_pair_retention"):
            for field in ("efficiency", "n_eff", "standard_error"):
                self.assertAlmostEqual(first[name]["weighted"][field], second[name]["weighted"][field])

    def test_duplicate_candidate_replication_does_not_fake_independent_support(self):
        original = self.summarize(parents=(1, 2), weights=(2, 1),
                                  pairs=(1, 2), labels=(1, 1), scores=(0.8, 0.2))
        duplicated = self.summarize(parents=(1, 2), weights=(2, 1),
                                    pairs=(1, 1, 2, 2), labels=(1, 1, 1, 1),
                                    scores=(0.8, 0.8, 0.2, 0.2))
        for name in ("full_parent_selected", "genuine_pair_retention"):
            for field in ("efficiency", "n_eff", "standard_error"):
                self.assertAlmostEqual(original[name]["weighted"][field], duplicated[name]["weighted"][field])
        self.assertEqual(duplicated["genuine_pair_retention"]["empirical"]["total"], 4)
        self.assertIsNotNone(original["genuine_pair_retention"]["empirical"]["wilson95"])
        self.assertIsNone(duplicated["genuine_pair_retention"]["empirical"]["wilson95"])

    def test_identity_matching_does_not_depend_on_parent_order(self):
        first = self.summarize(parents=(1, 2, 3, 4), weights=(2, 1, 3, 4))
        reordered = self.summarize(parents=(4, 2, 1, 3), weights=(4, 1, 2, 3))
        self.assertEqual(first["counts"], reordered["counts"])
        for name in ("full_parent_baseline", "full_parent_selected",
                     "conditional_event_retention", "genuine_pair_retention"):
            for field in ("efficiency", "n_eff", "standard_error"):
                self.assertAlmostEqual(first[name]["weighted"][field], reordered[name]["weighted"][field])

    def test_zero_candidate_parents_are_kept_with_defined_zero_efficiency(self):
        report = self.summarize(parents=(1, 2, 3), weights=(1, 2, 3),
                                pairs=(), labels=(), scores=())
        self.assertEqual(report["counts"]["parents_without_candidates"], 3)
        self.assertEqual(report["full_parent_selected"]["weighted"]["efficiency"], 0)
        self.assertGreater(report["full_parent_selected"]["empirical"]["wilson95"][1], 0)
        self.assertIsNone(report["conditional_event_retention"]["weighted"]["efficiency"])
        self.assertIsNone(report["genuine_pair_retention"]["empirical"]["efficiency"])

    def test_empty_parent_population_and_zero_weight_population_are_explicit(self):
        empty = self.summarize(parents=(), weights=(), pairs=(), labels=(), scores=())
        self.assertIsNone(empty["full_parent_selected"]["empirical"]["wilson95"])
        self.assertEqual(empty["full_parent_selected"]["weighted"]["undefined_reason"], "empty_population")
        zero = self.summarize(weights=(0, 0, 0, 0))
        self.assertEqual(zero["full_parent_selected"]["empirical"]["efficiency"], 0.25)
        self.assertEqual(zero["full_parent_selected"]["weighted"]["undefined_reason"], "zero_total_weight")
        self.assertIsNone(zero["full_parent_selected"]["weighted"]["standard_error"])
        self.assertEqual(zero["full_parent_selected"]["weighted"]["n_eff"], 0)
        json.dumps(empty, allow_nan=False)
        json.dumps(zero, allow_nan=False)

    def test_uint64_ids_above_float_precision_remain_distinct(self):
        large = 2 ** 63 + 1
        report = self.summarize(parents=(large, large + 1), weights=(1, 1),
                                pairs=(large + 1,), labels=(1,), scores=(0.9,))
        self.assertEqual(report["counts"]["parents_without_candidates"], 1)
        self.assertEqual(report["full_parent_selected"]["empirical"]["passed"], 1)

    def test_duplicate_parents_and_orphan_candidates_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate GEN parent"):
            self.summarize(parents=(1, 1, 3, 4))
        with self.assertRaisesRegex(ValueError, "Orphan candidate"):
            self.summarize(pairs=(1, 1, 1, 2, 9))

    def test_corrupt_values_and_cuts_rejected(self):
        for invalid in ((-1, 1, 1, 1), (np.nan, 1, 1, 1), (np.inf, 1, 1, 1)):
            with self.subTest(weights=invalid), self.assertRaises(ValueError):
                self.summarize(weights=invalid)
        for invalid in (np.nan, np.inf, -np.inf, True, 0.5j, [0.5]):
            with self.subTest(cut=invalid), self.assertRaisesRegex(ValueError, "cut"):
                self.summarize(cut=invalid)
        for invalid in ((0.4, np.nan, 0.9, 0.3, 0.99), (0.4, np.inf, 0.9, 0.3, 0.99)):
            with self.subTest(scores=invalid), self.assertRaisesRegex(ValueError, "scores"):
                self.summarize(scores=invalid)
        with self.assertRaisesRegex(ValueError, "pair_labels"):
            self.summarize(labels=(1, 2, 0, 1, -1))

    def test_bad_shapes_types_and_length_mismatches_rejected(self):
        arguments = [ids((1, 2)), np.array([1.0, 1.0]), ids((1,)),
                     np.array([1], dtype=np.int8), np.array([0.9]), 0.5]
        corruptions = ((0, np.array([[1, 2, 3]], dtype=float)),
                       (0, np.array([1, 2, 3], dtype=np.uint64)),
                       (1, np.array([1.0])),
                       (1, np.array([1 + 2j, 1])),
                       (2, np.zeros((1, 2), dtype=np.uint64)),
                       (3, np.array([1.0])),
                       (3, np.array([1, 1])),
                       (4, np.array([[0.9]])))
        for index, corrupted in corruptions:
            changed = list(arguments)
            changed[index] = corrupted
            with self.subTest(index=index), self.assertRaises(ValueError):
                module.summarize_vertex_efficiency(*changed)


if __name__ == "__main__":
    unittest.main()
