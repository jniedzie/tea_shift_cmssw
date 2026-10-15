# Code style: AI-code. AI-generated counting-limit accounting and tea API tests.
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "bin"))
sys.path.insert(0, str(PROJECT / "utils"))

from shift_limits import bind_fits, reference_cross_section, scan_group, validate_yields
from shift_limits_combine_precision import BUILDER_SHA256, digest, verify_stable_nll
from limits_producer import read_limit_quantiles


class LimitAccountingTest(unittest.TestCase):
    def test_existing_fit_without_bindings_cannot_acquire_new_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "card.txt").write_text("counting card")
            executable = base / "combine"
            executable.write_text("synthetic executable identity")
            config = SimpleNamespace(combine_options=(), datacards_output_path=str(base),
                                     combine_options_by_datacard={})
            manifest = dict(yields_sha256="a" * 64)
            cards = [dict(name="card")]
            output = base / "higgsCombine.card.AsymptoticLimits.mH120.root"
            output.write_bytes(b"unbound old fit")
            with self.assertRaisesRegex(ValueError, "no saved bindings"):
                bind_fits(config, manifest, cards, base, executable)
            self.assertFalse((base / "fit_bindings.json").exists())
            output.unlink()
            bind_fits(config, manifest, cards, base, executable)
            output.write_bytes(b"fit after binding")
            bind_fits(config, manifest, cards, base, executable)
            (base / "card.txt").write_text("changed counting card")
            with self.assertRaisesRegex(ValueError, "bindings changed"):
                bind_fits(config, manifest, cards, base, executable)

    def test_saved_numerical_overlay_is_verified_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = SimpleNamespace(combine_path=str(base / "combine"),
                                     combine_preload_path=str(base / "numerics/overlay.so"))
            with self.assertRaises(FileNotFoundError):
                verify_stable_nll(config)
            self.assertFalse((base / "numerics").exists())
            source = base / "combine/src/CachingNLL.cc"
            source.parent.mkdir(parents=True)
            source.write_text("original source")
            library = Path(config.combine_preload_path)
            library.parent.mkdir()
            library.write_bytes(b"retained library")
            patched = library.parent / "CachingNLL.cc"
            patched.write_text("retained patched source")
            receipt = dict(source_sha256=digest(source), builder_sha256=BUILDER_SHA256,
                           library_sha256=digest(library), patched_source_sha256=digest(patched))
            (library.parent / "receipt.json").write_text(json.dumps(receipt))
            self.assertEqual(verify_stable_nll(config), receipt)
            library.write_bytes(b"changed library")
            with self.assertRaisesRegex(ValueError, "overlay changed"):
                verify_stable_nll(config)
            self.assertEqual(library.read_bytes(), b"changed library")

    def test_shared_background_normalization_uses_channel_shape(self):
        backgrounds = {"QCD": dict(yields={"a": 5.e11, "b": 1.e8})}
        signal = dict(reference_cross_section_pb=1.e-4, yields={"a": .001, "b": .01})
        reference = reference_cross_section(signal, backgrounds, ("a", "b"))
        # The relative category populations constrain a common normalization.
        self.assertLess(reference, 1.e3)
        ratio = .001 / 5.e11 + .01 / 1.e8
        self.assertTrue(math.isfinite(reference))
        # An overall signal rescaling must leave the physical reference unchanged.
        signal2 = dict(reference_cross_section_pb=2.e-4, yields={"a": .002, "b": .02})
        self.assertAlmostEqual(reference, reference_cross_section(signal2, backgrounds, ("a", "b")))

    def test_zero_background_and_zero_acceptance_are_explicit(self):
        backgrounds = {"QCD": dict(yields={"a": 0.})}
        signal = dict(reference_cross_section_pb=.1, yields={"a": 2.})
        self.assertAlmostEqual(reference_cross_section(signal, backgrounds, ("a",)), .15)
        signal["yields"]["a"] = 0.
        self.assertEqual(reference_cross_section(signal, backgrounds, ("a",)), .1)

    def test_prompt_and_epsilon_scans_remain_separate(self):
        self.assertEqual(scan_group(dict(mean_lab_flight_m=None, source=dict(name="m10_prompt"))), "prompt")
        self.assertEqual(scan_group(dict(epsilon=1.e-5, mean_lab_flight_m=None,
                                        source=dict(name="m10_epsilon_1em5"))), "epsilon_1e-05")

    def test_missing_category_or_luminosity_mismatch_rejected(self):
        config = SimpleNamespace(luminosity=3000., categories=("a", "b"))
        sample = dict(reference_cross_section_pb=1., yields={"a": 1., "b": 0.})
        report = dict(schema="shift-simplified-limit-yields-v1", luminosity_pb=3000., categories=["a", "b"],
                      backgrounds={"QCD": sample}, signals={"m10": sample})
        validate_yields(report, config)
        report["luminosity_pb"] = 30.
        with self.assertRaisesRegex(ValueError, "luminosity"):
            validate_yields(report, config)

    def test_quantiles_are_read_from_tree_and_expected_has_no_observed(self):
        import ROOT
        from array import array
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "limits.root"
            output = ROOT.TFile(str(path), "RECREATE")
            tree = ROOT.TTree("limit", "limit")
            limit = array("d", [0.])
            quantile = array("f", [0.])
            tree.Branch("limit", limit, "limit/D")
            tree.Branch("quantileExpected", quantile, "quantileExpected/F")
            for q, value in ((.84, 4.), (.025, 1.), (.975, 5.), (.5, 3.), (.16, 2.)):
                quantile[0], limit[0] = q, value
                tree.Fill()
            tree.Write()
            output.Close()
            self.assertEqual(read_limit_quantiles(path),
                             dict(observed=None, minus2=1., minus1=2., median=3., plus1=4., plus2=5.))

    def test_cards_preserve_zero_processes_and_only_the_requested_nuisances(self):
        import ROOT
        from Histogram import Histogram
        from HistogramNormalizer import NormalizationType
        from HistogramsManager import HistogramsManager
        from Sample import Sample, SampleType
        from shift_limits import write_count_histograms
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            input_files = {}
            samples = [Sample(name="signal_point", type=SampleType.signal),
                       Sample(name="QCD", type=SampleType.background),
                       Sample(name="DY", type=SampleType.background)]
            for sample, rate in zip(samples, (0., 5.e11, 0.)):
                sample.file_path = str(path / (sample.name + ".root"))
                write_count_histograms(sample.file_path, {"a": rate})
                input_files[sample.name] = ROOT.TFile.Open(sample.file_path)
            config = SimpleNamespace(histogram=Histogram(name="count_a", norm_type=NormalizationType.none),
                samples=samples, datacards_output_path=str(path) + "/", plots_output_path=str(path) + "/",
                nuisances={"signal_norm": {"signal": 1.30}, "background_norm": {"QCD": 1.30, "DY": 1.30}},
                exclude_backgrounds_for_years={-1: 0}, do_abcd=False, retain_zero_backgrounds=True,
                asimov_observation=True, include_shapes=False, include_mc_stat_uncertainties=False,
                symmetric_variations=(), add_uncertainties_on_zero=False)
            manager = HistogramsManager(config, input_files, "card")
            for sample in samples:
                manager.addHistosample(config.histogram, sample)
            manager.normalizeHistograms()
            manager.buildStacks()
            manager.saveDatacards()
            card = (path / "card.txt").read_text()
            self.assertIn("process signal QCD DY", card)
            self.assertIn("rate 0.0 500000000000.0 0.0", card)
            self.assertIn("signal_norm lnN 1.3 - -", card)
            self.assertIn("background_norm lnN - 1.3 1.3", card)
            self.assertNotIn("autoMCStats", card)
            self.assertNotIn("stat_err", card)
            self.assertNotIn("background_dummy", card)
            for file in input_files.values():
                file.Close()


if __name__ == "__main__":
    unittest.main()
