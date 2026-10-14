# Code style: AI-code. AI-generated regression test.
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


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


class AvailableHistogramPlotConfigTest(unittest.TestCase):
    def setUp(self):
        import ROOT
        self.ROOT = ROOT
        self.repository = Path(__file__).resolve().parents[1]
        for name in ('tea/pylibs/plotting', 'tea/pylibs/logger'):
            directory = str(self.repository / name)
            sys.path.insert(0, directory)
            self.addCleanup(sys.path.remove, directory)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign = 'shift_detector_representative_20261007_v10'
        self.bsm_campaign = '20261013-unfinished-production'
        self.sm = self.write_histograms(
            self.base / 'qcd_pThat0to1GeV' / self.campaign / 'histograms_merged', 20.)
        self.signal = self.write_histograms(
            self.base / 'darkphoton_m15_lab_100m' / self.bsm_campaign / 'histograms_merged', 5.)

    def write_histograms(self, directory, initial_sum):
        directory.mkdir(parents=True)
        path = directory / 'partial.root'
        output = self.ROOT.TFile(str(path), 'RECREATE')
        cutflow = self.ROOT.TH1D('cutFlow', '', 2, 0, 2)
        cutflow.SetBinContent(1, initial_sum)
        cutflow.SetBinContent(2, 4.)
        cutflow.Write()
        histogram = self.ROOT.TH1D('available_observable', '', 2, 0, 2)
        histogram.Fill(.5, 3.)
        histogram.Fill(1.5, 1.)
        histogram.Write()
        output.Close()
        return path

    def config(self, *, bsm_campaign=None, flights=(100,), epsilons=None):
        with patch.multiple('shift_paths', base_path=str(self.base), campaign=self.campaign,
                            cross_sections={'qcd': {'0to1': 2., '1to2': 3.}},
                            bsm_campaign=bsm_campaign, bsm_masses_to_plot=[15],
                            bsm_mean_lab_flights_to_plot=list(flights), bsm_epsilons_to_plot=epsilons,
                            histogram_cache_directory=None), \
                patch('shift_sample_paths.full_gen_weight_sums', side_effect=AssertionError('Full GEN validation')), \
                patch('shift_sample_paths._read_normalization_record', side_effect=AssertionError('Production records')), \
                patch('shift_sample_paths._bsm_campaign_records', side_effect=AssertionError('BSM production validation')):
            return SimpleNamespace(**runpy.run_path(str(self.repository / 'configs/shift_plotter_config.py')))

    def test_partial_sm_plots_without_any_production_validation(self):
        with patch('Logger.warn') as warn:
            config = self.config()
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1'])
        self.assertEqual(config.samples[0].file_path, str(self.sm))
        self.assertEqual(config.samples[0].initial_weight_sum, -1)
        self.assertTrue(warn.called)  # The missing bin does not block the ready one.

    def test_partial_bsm_needs_no_plan_marker_or_complete_campaign(self):
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(None, 100))
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1', 'm15_lab_100m'])
        self.assertEqual(config.samples[1].file_path, str(self.signal))
        self.assertTrue(all(sample.initial_weight_sum == -1 for sample in config.samples))

    def test_epsilon_samples_use_epsilon_legends_without_production_records(self):
        signal = self.write_histograms(
            self.base / 'darkphoton_m15_epsilon_1e-06' / self.bsm_campaign / 'histograms_merged', 5.)
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign, flights=(), epsilons=[1.e-6])
        self.assertEqual([sample.name for sample in config.samples], ['qcd_0to1', 'm15_epsilon_1e-06'])
        self.assertEqual(config.samples[1].file_path, str(signal))
        self.assertIn('#epsilon = 1e-06', config.samples[1].legend_description)
        self.assertEqual(config.samples[1].line_style, 2)
        self.assertEqual(config.samples[1].initial_weight_sum, -1)

    def test_native_normalization_uses_available_cutflow_and_matches_signal_area(self):
        from HistogramNormalizer import HistogramNormalizer, NormalizationType
        with patch('Logger.warn'):
            config = self.config(bsm_campaign=self.bsm_campaign)
        config.luminosity = 3.
        normalizer = HistogramNormalizer(config)
        self.addCleanup(lambda: [file.Close() for file in normalizer.input_files.values()])
        background, signal = config.samples
        self.assertEqual(normalizer.background_initial_sum_weights[(background.name, -1)], 20.)
        self.assertEqual(normalizer.signal_initial_sum_weights[signal.name], 5.)
        def histogram(sample):
            native = normalizer.input_files[sample.name].Get('available_observable').Clone(sample.name)
            native.SetDirectory(0)
            return SimpleNamespace(name='available_observable', hist=native,
                                   norm_type=NormalizationType.to_background, scale_by_bin_width=False)
        background_hist, signal_hist = histogram(background), histogram(signal)
        normalizer.normalize(background_hist, background)
        self.assertAlmostEqual(background_hist.hist.Integral(), 1.2)
        normalizer.normalize(signal_hist, signal, total_backgrounds_integral=background_hist.hist.Integral())
        self.assertAlmostEqual(signal_hist.hist.Integral(), 1.2)


if __name__ == "__main__":
    unittest.main()
