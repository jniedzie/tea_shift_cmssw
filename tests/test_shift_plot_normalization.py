# Code style: AI-code. AI-generated regression test.
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "configs"))
from shift_sample_paths import full_gen_weight_sums


class FullGenPlotNormalizationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = "shift_detector_representative_20261007_v10"
        ordinary = self.write("ordinary.json", {"schema": "frozen-ordinary-source"})
        weighted = self.write("weighted.json", {"schema": "frozen-weighted-source"})
        self.accounting = {
            "schema": "shift-weighted-high-bin-final-accounting-v1", "complete": True,
            "manifest_sha256": self.digest(weighted),
            "strata": {"qcd_20toinf": {"accepted": 1000, "sumw": 0.5}},
        }
        accounting = self.write("accounting.json", self.accounting)
        parent = self.write("parent.json", {
            "schema": "shift-gen-to-nano-plan-v1",
            "strata": {"qcd_0to1": 100, "qcd_20toinf": 1000},
            "ordinary_manifest": str(ordinary), "ordinary_manifest_sha256": self.digest(ordinary),
            "weighted_manifest": str(weighted), "weighted_manifest_sha256": self.digest(weighted),
        })
        self.plan = {
            "schema": "shift-detector-pps-sample-v1", "parent_manifest": str(parent),
            "parent_manifest_sha256": self.digest(parent),
            "normalization_authorities": {
                "ordinary": {"manifest": str(ordinary), "manifest_sha256": self.digest(ordinary)},
                "weighted": {"manifest": str(weighted), "manifest_sha256": self.digest(weighted),
                             "accounting": str(accounting), "accounting_sha256": self.digest(accounting),
                             "strata": self.accounting["strata"]},
            },
            "strata": {
                "qcd_0to1": {"original_events": 100, "original_sumw": 100,
                             "realized_events": 9, "corrected_sumw": 90,
                             "original_normalization_records": [
                                 {"stratum": "qcd_0to1", "events": 100, "normalization_record": None}]},
                "qcd_20toinf": {"original_events": 1000, "original_sumw": 0.5,
                                "realized_events": 10, "corrected_sumw": 0.4,
                                "original_normalization_records": [
                                    {"stratum": "qcd_20toinf", "events": 1000,
                                     "normalization_record": "weighted-ledger.json"}]},
            },
        }
        self.manifest = {
            "schema": "shift-gen-to-nano-plan-v1", "eos_output": "/eos/" + self.campaign,
            "strata": {"qcd_0to1": 9, "qcd_20toinf": 10},
            "detector_sampling": {"plan": "sampling_plan.json",
                                  "parent_manifest_sha256": self.digest(parent)},
        }
        self.refresh()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh(self):
        path = self.write("sampling_plan.json", self.plan)
        self.manifest["detector_sampling"]["plan_sha256"] = self.digest(path)
        self.write("manifest.json", self.manifest)

    def sums(self):
        return full_gen_weight_sums(self.campaign, self.root)

    def test_full_parent_exposure_ignores_selected_counts_and_sums(self):
        self.assertEqual(self.sums(), {"qcd_0to1": 100., "qcd_20toinf": 0.5})

    def test_legacy_campaign_needs_no_representative_provenance(self):
        self.assertIsNone(full_gen_weight_sums("legacy_10k", self.root / "absent"))

    def test_missing_representative_authority_fails_closed(self):
        (self.root / "accounting.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            self.sums()

    def test_changed_sampling_payload_fails_closed(self):
        (self.root / "sampling_plan.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "hash differs"):
            self.sums()

    def test_wrong_campaign_fails_closed(self):
        self.manifest["eos_output"] = "/eos/another_campaign"
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "requested campaign"):
            self.sums()

    def test_weighted_selected_sum_cannot_replace_parent_sum(self):
        self.plan["strata"]["qcd_20toinf"]["original_sumw"] = 0.4
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "Weighted full GEN sum differs"):
            self.sums()

    def test_unit_weight_selected_sum_cannot_replace_parent_count(self):
        self.plan["strata"]["qcd_0to1"]["original_sumw"] = 90
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "Unit-weight full GEN exposure differs"):
            self.sums()

    def test_missing_stratum_cannot_fall_back_to_selected_cutflow(self):
        del self.plan["strata"]["qcd_20toinf"]
        self.refresh()
        with self.assertRaisesRegex(RuntimeError, "inventories differ"):
            self.sums()


if __name__ == "__main__":
    unittest.main()
