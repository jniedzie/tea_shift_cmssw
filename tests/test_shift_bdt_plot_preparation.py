import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import ROOT
import uproot

REPOSITORY = Path(__file__).resolve().parents[1]
for directory in ("configs", "tea/pylibs/plotting", "tea/pylibs/logger"):
    sys.path.insert(0, str(REPOSITORY / directory))
# Study API import uses only a pinned fixture; canonical campaign discovery
# must remain unnecessary for this optional view.
with tempfile.TemporaryDirectory() as import_directory:
    fixture = Path(import_directory) / "histograms.root"
    fixture.touch()
    inventory = Path(import_directory) / "inputs.json"
    inventory.write_text(json.dumps({
        "schema": "shift-dimuon-bdt-plot-inputs-v1", "complete": True,
        "samples": [{"name": "qcd", "process": "qcd", "histogram_path": str(fixture)}],
    }))
    with patch.multiple("shift_paths", bdt_mixing_histogram_inventory=str(inventory),
                        bdt_mixing_plot_output=None, bdt_mixing_preparation_directory=None,
                        bdt_mixing_plot_scope="all"), \
            patch("shift_sample_paths.full_gen_weight_sums", side_effect=AssertionError("Canonical discovery")), \
            patch("shift_sample_paths.single_root_file", side_effect=AssertionError("Canonical discovery")):
        import shift_bdt_mixing_plotter_config as plot_config

from utils.shift_bdt_mixing_histogram_inputs import (
    _plot_digest, bin_scale, prepare_process_histograms as prepare, readable_histogram_path,
)


def histograms(path, weighted, raw):
    output = ROOT.TFile(str(path), "RECREATE")
    output.mkdir("dimuon_bdt").cd()
    for name, value in (("bdtScore", weighted), ("bdtScoreUnweighted", raw)):
        histogram = ROOT.TH1D("ShiftDimuonVertex_" + name, "", 100, 0, 1)
        histogram.Sumw2()
        histogram.Fill(0.5, value)
        histogram.Write()
    output.Close()


class BdtPlotPreparationTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.first = self.directory / "first.root"
        self.second = self.directory / "second.root"
        histograms(self.first, 10, 2)
        histograms(self.second, 20, 3)
        self.manifest_path = self.directory / "inputs.json"
        self.output = self.directory / "combined"
        self.manifest = {
            "schema": "shift-dimuon-bdt-bin-inputs-v1", "complete": True,
            "samples": [
                {"name": "qcd_low", "process": "qcd", "label": "QCD", "histogram_path": str(self.first),
                 "cross_section_pb": 100, "full_gen_sumw": 1000},
                {"name": "qcd_high", "process": "qcd", "label": "QCD", "histogram_path": str(self.second),
                 "cross_section_pb": 2, "full_gen_sumw": 100},
            ],
        }

    def run_preparation(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return prepare(self.manifest_path, self.output)

    def test_exposure_scale_before_process_sum_and_raw_counts_preserved(self):
        receipt = self.run_preparation()
        self.assertTrue(receipt["complete"])
        self.assertEqual(len(receipt["samples"]), 1)
        self.assertEqual([row["applied_shape_scale"] for row in receipt["inputs"]], [0.1, 0.02])
        with uproot.open(self.output / "qcd.root") as source:
            weighted = source["dimuon_bdt/ShiftDimuonVertex_bdtScore"]
            raw = source["dimuon_bdt/ShiftDimuonVertex_bdtScoreUnweighted"]
            self.assertAlmostEqual(np.sum(weighted.values(flow=True)), 1.4)
            self.assertAlmostEqual(np.sum(weighted.variances(flow=True)), 1.16)
            self.assertEqual(np.sum(raw.values(flow=True)), 5)

    def test_production_merger_rows_preserve_bin_mixture(self):
        self.manifest["rows"] = self.manifest.pop("samples")
        for row in self.manifest["rows"]:
            row["label"] = row["name"]
        receipt = self.run_preparation()
        self.assertTrue(receipt["complete"])
        self.assertEqual(receipt["samples"][0]["label"], "QCD")
        self.assertEqual([row["applied_shape_scale"] for row in receipt["inputs"]], [0.1, 0.02])
        with uproot.open(self.output / "qcd.root") as source:
            self.assertAlmostEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertex_bdtScore"].values(flow=True)), 1.4)

    def test_conflicting_input_inventories_rejected_before_output(self):
        self.manifest["rows"] = self.manifest["samples"][:1]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.run_preparation()
        self.assertFalse(self.output.exists())

    def test_eos_mount_alias_preserves_provenance_and_checksum(self):
        logical = "/eos/user/a/alice/production/histograms.root"
        mounted = "/eos/home-a/alice/production/histograms.root"
        with patch.object(Path, "is_file", autospec=True, side_effect=lambda path: str(path) == mounted):
            self.assertEqual(str(readable_histogram_path(logical)), mounted)
        with patch.object(Path, "is_file", autospec=True, side_effect=lambda path: str(path) == logical):
            self.assertEqual(str(readable_histogram_path(mounted)), logical)
        self.manifest["samples"][0]["histogram_path"] = logical
        with patch("utils.shift_bdt_mixing_histogram_inputs.readable_histogram_path", side_effect=[self.first, self.second]):
            receipt = self.run_preparation()
        self.assertEqual(receipt["inputs"][0]["histogram_path"], logical)
        self.assertEqual(receipt["inputs"][0]["histogram_read_path"], str(self.first))

    def test_missing_histogram_mount_fails(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Cannot read"):
                readable_histogram_path("/eos/user/a/alice/missing.root")

    def test_selected_sum_is_not_an_implicit_denominator(self):
        self.manifest["samples"][0].pop("full_gen_sumw")
        self.manifest["samples"][0]["selected_sumw"] = 10
        with self.assertRaises(KeyError):
            self.run_preparation()
        self.assertFalse(json.loads((self.output / "plot_inputs.json").read_text())["complete"])

    def test_signal_mass_lifetime_samples_stay_separate(self):
        for index, row in enumerate(self.manifest["samples"]):
            row["process"] = "signal"
            row["name"] = f"signal_m{index}_tau"
            row["label"] = f"A' mass {index}"
        receipt = self.run_preparation()
        self.assertEqual(len(receipt["samples"]), 2)
        self.assertEqual([row["applied_shape_scale"] for row in receipt["inputs"]], [1, 1])
        self.assertEqual([row["label"] for row in receipt["samples"]], ["A' mass 0", "A' mass 1"])

    def test_duplicate_input_rejected(self):
        self.manifest["samples"][1]["name"] = self.manifest["samples"][0]["name"]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.run_preparation()

    def test_incomplete_input_manifest_rejected(self):
        self.manifest["complete"] = False
        with self.assertRaisesRegex(ValueError, "complete BDT"):
            self.run_preparation()
        self.assertFalse(self.output.exists())

    def test_checksum_mismatch_rejected(self):
        self.manifest["samples"][0]["histogram_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_preparation()

    def test_nonpositive_full_gen_exposure_rejected(self):
        self.manifest["samples"][0]["full_gen_sumw"] = 0
        with self.assertRaisesRegex(ValueError, "full GEN exposure"):
            self.run_preparation()

    def test_existing_output_directory_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_preparation()

    def test_exact_completed_preparation_can_be_reused(self):
        receipt = self.run_preparation()
        reused = prepare(self.manifest_path, self.output, reuse=True)
        self.assertEqual(reused, receipt)
        histograms(self.output / "qcd.root", 500, 50)
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare(self.manifest_path, self.output, reuse=True)

    def test_changed_inventory_cannot_reuse_an_existing_preparation(self):
        self.run_preparation()
        self.manifest["samples"][0]["full_gen_sumw"] = 2000
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "another or incomplete"):
            prepare(self.manifest_path, self.output, reuse=True)

    def test_changed_source_histogram_cannot_reuse_completed_preparation(self):
        self.run_preparation()
        histograms(self.first, 500, 50)
        with self.assertRaisesRegex(ValueError, "Source histogram checksum"):
            prepare(self.manifest_path, self.output, reuse=True)

    def test_different_physical_signal_points_cannot_share_a_plot_group(self):
        for index, row in enumerate(self.manifest["samples"]):
            row.update(name=f"signal_{index}", process="signal", plot_group="pooled_signal")
        with self.assertRaisesRegex(ValueError, "signal points"):
            self.run_preparation()

    def test_sm_scope_preserves_process_labels_and_signal_points_remain_distinct(self):
        receipt = self.run_preparation()
        receipt["samples"].append({
            "name": "m15_prompt", "process": "signal", "label": "A' 15 GeV prompt",
            "histogram_path": str(self.first),
        })
        self.assertEqual([s.name for s in plot_config.inventory_plot_samples(receipt, "sm")], ["qcd"])
        all_samples = plot_config.inventory_plot_samples(receipt, "all")
        self.assertEqual([s.legend_description for s in all_samples], ["QCD", "A' 15 GeV prompt"])
        self.assertNotEqual(all_samples[0].line_color, all_samples[1].line_color)
        self.assertEqual(all_samples[1].cross_section, -1)

    def test_canonical_score_plots_keep_score_range_and_reconstructed_axes(self):
        scores, maps = plot_config.saved_score_plots()
        self.assertEqual(len(scores), 8)
        self.assertEqual(len(maps), 6)
        self.assertTrue(all(h.x_min == 0 and h.x_max == 1 for h in scores))
        self.assertTrue(all(h.norm_type == plot_config.NormalizationType.to_one for h in scores))
        self.assertEqual({h.name.rsplit("Vs", 1)[1] for h in maps}, {"Mass", "Vz", "VertexR"})

    def test_prepared_inventory_checksums_are_verified_before_plotting(self):
        receipt = self.run_preparation()
        path = self.output / "plot_inputs.json"
        self.assertEqual(plot_config.configured_inventory(path, self.output), receipt)
        histograms(self.output / "qcd.root", 500, 50)
        with self.assertRaisesRegex(ValueError, "checksum"):
            plot_config.configured_inventory(path, self.output)

    def test_mixing_plots_do_not_apply_luminosity_or_shape_normalization(self):
        plots = plot_config.event_mixing_plots()
        self.assertEqual(len(plots), 8)
        self.assertTrue(all(h.norm_type == plot_config.NormalizationType.none for h in plots))
        self.assertTrue(all(h.scale_by_bin_width for h in plots))
        self.assertEqual({h.x_max for h in plots}, {12, 120})

    def load_config(self, settings):
        spec = importlib.util.spec_from_file_location("_shift_plot_config_test", REPOSITORY / "configs/shift_bdt_mixing_plotter_config.py")
        module = importlib.util.module_from_spec(spec)
        options = dict(bdt_mixing_histogram_inventory=None, bdt_mixing_plot_output=None,
                       bdt_mixing_preparation_directory=None, bdt_mixing_plot_scope="all")
        options.update(settings)
        with patch.multiple("shift_paths", **options), \
                patch("shift_sample_paths.full_gen_weight_sums", side_effect=AssertionError("Canonical discovery")), \
                patch("shift_sample_paths.single_root_file", side_effect=AssertionError("Canonical discovery")):
            spec.loader.exec_module(module)
        return module

    def test_study_config_prepares_raw_bin_inventory_without_a_separate_preparer(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        module = self.load_config({
            "bdt_mixing_histogram_inventory": str(self.manifest_path), "bdt_mixing_plot_output": str(self.directory / "plots"),
            "bdt_mixing_preparation_directory": str(self.output), "bdt_mixing_plot_scope": "sm",
        })
        self.assertEqual(module.plot_mode, "bdt")
        self.assertEqual([sample.name for sample in module.samples], ["qcd"])
        self.assertEqual(len(module.histograms), 8)
        self.assertTrue((self.output / "plot_inputs.json").is_file())
        self.assertFalse(module.signal_normalization_validated)
        self.assertFalse(hasattr(module, "lumi_label_value"))
        with uproot.open(module.samples[0].file_path) as source:
            self.assertAlmostEqual(np.sum(source["dimuon_bdt/ShiftDimuonVertex_bdtScore"].values(flow=True)), 1.4)

    def test_frozen_prepared_inventory_uses_optional_study_config(self):
        self.run_preparation()
        module = self.load_config({
            "bdt_mixing_histogram_inventory": str(self.output / "plot_inputs.json"),
            "bdt_mixing_plot_scope": "sm", "bdt_mixing_plot_output": str(self.directory / "legacy_plots"),
        })
        self.assertEqual(module.output_path, str(self.directory / "legacy_plots"))
        self.assertEqual([sample.legend_description for sample in module.samples], ["QCD"])

    def test_mixing_manifest_checks_model_report_and_output_hashes(self):
        report = self.directory / "mixing.json"
        report.write_text("{}")
        manifest = {
            "schema": "shift-event-mixing-plot-inputs-v1", "complete": True,
            "physics_valid": False, "rate_validated": False, "fit_reproduction_validated": True,
            "report": str(report), "report_sha256": _plot_digest(report),
            "samples": [{"name": "same_event", "kind": "model", "label": "same event",
                         "histogram_path": str(self.first), "sha256": _plot_digest(self.first)}],
        }
        self.manifest_path.write_text(json.dumps(manifest))
        module = self.load_config({"bdt_mixing_histogram_inventory": str(self.manifest_path)})
        self.assertEqual(module.plot_mode, "mixing")
        self.assertEqual(len(module.histograms), 8)
        self.assertFalse(module.physics_valid)
        self.assertIn("unvalidated", module.beam_label)
        report.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "report checksum"):
            plot_config.configured_inventory(self.manifest_path, self.output)


if __name__ == "__main__":
    unittest.main()
