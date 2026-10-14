# Code style: AI-code. AI-generated fixed-family signal transfer tests.
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "utils"))
spec = importlib.util.spec_from_file_location("complete_transfer",
    Path(__file__).parents[1] / "utils/shift_vertex_complete_signal_transfer.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def ids(events):
    return np.asarray([[1, 1, event] for event in events], dtype=np.uint64).reshape(-1, 3)


class CompleteSignalTransferTests(unittest.TestCase):
    def test_only_full_size_locked_winner_seeds_are_selected(self):
        recipes = [{"name": f"{family}_f{int(fraction*100)}_s{seed}", "family": family,
                    "algorithm": algorithm, "seed": seed, "fraction": fraction}
                   for algorithm, family in (("bdt", "trees"), ("dnn", "network"), ("dnn", "unselected"))
                   for fraction in (.25, .5, 1.) for seed in module.SEEDS]
        protocol = {"domain": "current_CMSIR5", "seeds": list(module.SEEDS), "recipes": recipes}
        lock = {"domain": "current_CMSIR5", "selection_uses_test": False, "all81fits_before_development": True,
                "winners": {"bdt": {"rejection": "trees", "acceptance": None},
                            "dnn": {"rejection": "network", "acceptance": "network"}},
                "families": {"trees": {"algorithm": "bdt"}, "network": {"algorithm": "dnn"}}}
        chosen, roles = module.selected_models(protocol, lock)
        self.assertEqual(len(chosen), 6)
        self.assertEqual({spec["fraction"] for spec in chosen}, {1.})
        self.assertEqual({spec["seed"] for spec in chosen}, {71, 72, 73})
        self.assertEqual(roles["network"], ["dnn_rejection", "dnn_acceptance"])
        self.assertNotIn("unselected", {spec["family"] for spec in chosen})

    def test_unknown_bounds_count_unique_parents_and_exclude_known_non_signal(self):
        parent_ids = ids([1, 2, 3, 4, 5])
        pair_ids = ids([1, 1, 2, 2, 3, 4])
        labels = np.array([1, -1, -1, -1, 0, 1])
        qualified = np.array([True, False, False, False, False, False])
        bounds = module.unknown_response_bounds(parent_ids, np.array([2., 3., 4., 5., 6.]),
            pair_ids, labels, np.array([.9, .9, .9, .8, .9, .9]), .5, qualified)
        self.assertEqual(bounds["known_selected_parents"], 1)
        self.assertEqual(bounds["passing_unknown_additional_parents"], 1)
        self.assertEqual(bounds["upper_selected_parents"], 2)
        self.assertEqual(bounds["lower_empirical_efficiency"], .2)
        self.assertEqual(bounds["upper_empirical_efficiency"], .4)
        self.assertEqual(bounds["lower_weighted_efficiency"], 2/20)
        self.assertEqual(bounds["upper_weighted_efficiency"], 5/20)

    def test_qualified_retention_and_full_GEN_are_distinct_and_unknown_preserved(self):
        reco = {"ids": ids([1, 2, 3]), "point": np.array(["A", "A", "A"])}
        events = {"ids": ids([1, 2, 3, 4]), "point": np.array(["A"]*4), "native_weight": np.ones(4)}
        labels = {"y": np.array([1, 1, -1]), "genuine_signal_associated": np.array([True, True, False])}
        result = module.summarize_predictions(reco, labels, events, {"fixed": np.array([.8, .2, .9])}, {"fixed": .5}, {"A": {}})
        row = result["fixed"]["points"]["A"]
        self.assertEqual(row["genuine_pair_retention"]["empirical"]["efficiency"], .5)
        self.assertEqual(row["full_parent_selected"]["empirical"]["efficiency"], .25)
        self.assertEqual(row["classes"]["unknown"]["selected_pairs"], 1)
        self.assertEqual(row["unknown_truth_bounds"]["upper_empirical_efficiency"], .5)

    def test_absent_selection_lock_prevents_all_cache_and_model_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(module, "load") as loader, patch.object(module.pickle, "load") as unpickle:
                with self.assertRaisesRegex(ValueError, "Selection lock required"):
                    module.run(temporary, "missing_cache", "missing_response", "missing_output")
                loader.assert_not_called(); unpickle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
