# Code style: AI-code. AI-generated regression tests for safe data control plots.
"""Synthetic aggregate inputs only, including the ordinary plotter CLI."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils import shift_blinded_data_summary as blind
from utils import shift_blinded_data_plot_inputs as inputs

try:
    import ROOT
except ImportError:
    ROOT = None


def synthetic_summary():
    policy = dict(schema="shift-data-blinding-policy-v1", version="synthetic-control-v1",
                  blinded=True, unit="candidate", mass_branch=blind.MASS_BRANCH,
                  mass_min_gev=2.6, mass_max_gev=3.6, histogram_bins=20, endpoints="inclusive")
    summary = blind.BlindedSummary(policy, "mc-as-data")
    summary.add_event((1, 1, 1), 3, [3.025, 3.025, 666.123456])
    summary.add_event((1, 1, 2), 1, [3.025])
    summary.add_event((1, 1, 3), 1, [3.025])
    summary.add_event((1, 1, 4), 1, [777.234567])
    return summary.result([])


class AggregateInputTest(unittest.TestCase):
    def test_source_context_rejects_unbound_summary_and_forbidden_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "blinded_summary.json"
            source.write_text(json.dumps(synthetic_summary()))
            context = dict(schema="shift-data-control-plot-context-v1", era="Run2023D",
                           datasets=["/Muon0/Run2023D-v1/RAW"], runs=[369943],
                           summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            path = directory / inputs.CONTEXT_NAME
            path.write_text(json.dumps(context))
            labels = inputs.data_plot_labels(source, {}, directory / "legacy/summary.json")
            self.assertEqual(labels, dict(era="Run2023D", stream="Muon0", run_label="run 369943"))
            path.write_text(json.dumps(dict(context, summary_sha256="0" * 64)))
            with self.assertRaises(blind.BlindingError):
                inputs.read_plot_context(source)
            path.write_text(json.dumps(dict(context, candidate_rows=[999.123456])))
            with self.assertRaises(blind.BlindingError):
                inputs.read_plot_context(source)

    def test_nonlegacy_plot_cannot_fall_back_to_wrong_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "summary.json"
            source.write_text(json.dumps(synthetic_summary()))
            with self.assertRaises(blind.BlindingError):
                inputs.data_plot_labels(source, {}, directory / "legacy/summary.json")
            legacy = dict(era="Run2023D", stream="ZeroBias", run="369943")
            self.assertEqual(inputs.data_plot_labels(source, legacy, source)["stream"], "ZeroBias")
            context = dict(schema="shift-data-control-plot-context-v1", era="Run2023D",
                           datasets=["/Muon0/Run2023D-v1/RAW"], runs=[369943],
                           summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            (directory / inputs.CONTEXT_NAME).write_text(json.dumps(context))
            with self.assertRaises(blind.BlindingError):
                inputs.data_plot_labels(source, legacy, source)

    def test_nested_forbidden_payload_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            summary = synthetic_summary()
            summary["candidate_rows"] = [[999.123456]]
            source = directory / "summary.json"
            source.write_text(json.dumps(summary))
            with self.assertRaises(blind.BlindingError):
                inputs.prepare_histogram(source, directory / "new/output.root", (2.6, 3.6))
            self.assertFalse((directory / "new").exists())

    def test_wider_aggregate_cannot_silently_widen_the_configured_plot_band(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "summary.json"
            source.write_text(json.dumps(synthetic_summary()))
            with self.assertRaises(blind.BlindingError):
                inputs.prepare_histogram(source, directory / "new/output.root", (2.9, 3.3))
            self.assertFalse((directory / "new").exists())


@unittest.skipIf(ROOT is None, "ROOT runtime is unavailable")
class AggregateRootPlotTest(unittest.TestCase):
    def test_current_histogram_reuse_preserves_bytes_and_rejects_changed_bin_content(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, target = directory / "summary.json", directory / "histograms.root"
            source.write_text(json.dumps(synthetic_summary()))
            inputs.prepare_histogram(source, target, (2.6, 3.6))
            before = hashlib.sha256(target.read_bytes()).hexdigest()
            inputs.prepare_histogram(source, target, (2.6, 3.6))
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), before)
            root = ROOT.TFile.Open(str(target), "UPDATE")
            histogram = root.Get(inputs.HISTOGRAM_NAME)
            histogram.SetBinContent(1, 1.)
            histogram.Write("", ROOT.TObject.kOverwrite)
            root.Close()
            with self.assertRaises(blind.BlindingError):
                inputs.prepare_histogram(source, target, (2.6, 3.6))

    def test_only_mass_histogram_zero_flows_and_unscaled_poisson_counts_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, target = directory / "summary.json", directory / "histograms.root"
            summary = synthetic_summary()
            source.write_text(json.dumps(summary))
            inputs.prepare_histogram(source, target, (2.6, 3.6))
            root = ROOT.TFile.Open(str(target))
            self.addCleanup(root.Close)
            self.assertEqual({key.GetName() for key in root.GetListOfKeys()},
                             {inputs.HISTOGRAM_NAME, inputs.METADATA_NAME})
            histogram = root.Get(inputs.HISTOGRAM_NAME)
            self.assertEqual(histogram.GetXaxis().GetXmin(), 2.6)
            self.assertEqual(histogram.GetXaxis().GetXmax(), 3.6)
            self.assertEqual(histogram.GetBinContent(0), 0.)
            self.assertEqual(histogram.GetBinContent(histogram.GetNbinsX() + 1), 0.)
            self.assertEqual(histogram.Integral(), 4.)
            self.assertEqual(histogram.GetEntries(), 4.)
            self.assertEqual(histogram.GetEffectiveEntries(), 4.)
            self.assertEqual(histogram.GetBinErrorOption(), ROOT.TH1.kPoisson)
            empty = 1
            populated = histogram.FindBin(3.025)
            self.assertEqual(histogram.GetBinContent(populated), 4.)
            self.assertEqual(histogram.GetBinErrorLow(empty), 0.)
            self.assertGreater(histogram.GetBinErrorUp(empty), 1.8)
            self.assertGreater(histogram.GetBinErrorUp(populated), 2.)
            metadata = json.loads(root.Get(inputs.METADATA_NAME).GetString().Data())
            self.assertEqual(metadata["normalization"], "none")
            self.assertEqual(metadata["unique_processed_events"], 4)
            self.assertEqual(metadata["events_with_allowed_pair"], 3)
            self.assertEqual(metadata["allowed_pairs"], 4)

    def test_actual_preparer_and_ordinary_plotter_cli_preserve_blinding_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "summary.json"
            source.write_text(json.dumps(synthetic_summary()))
            target = directory / "safe.root"
            prepared = subprocess.run([
                sys.executable, str(PROJECT / "utils/shift_blinded_data_plot_inputs.py"),
                "--summary", str(source), "--output-root", str(target),
                "--mass-min-gev", "2.6", "--mass-max-gev", "3.6",
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
            config_path = directory / "synthetic_plot_config.py"
            config_path.write_text(
                "import runpy\n"
                f"globals().update(runpy.run_path({str(PROJECT / 'configs/shift_data_plotter_config.py')!r}, "
                f"init_globals={{'summary_path': {str(source)!r}, 'output_path': {str(directory / 'plots')!r}, "
                "'era': 'Synthetic MC-as-data', 'stream': 'Synthetic', 'run': '0'}))\n")
            paths = [directory, PROJECT, PROJECT / "configs", PROJECT / "tea/pylibs/plotting",
                     PROJECT / "tea/pylibs/logger"]
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + environment.get("PYTHONPATH", "")
            result = subprocess.run([
                sys.executable, str(PROJECT / "tea/apps/examples/plotter.py"),
                "--config", "synthetic_plot_config",
            ], capture_output=True, text=True, cwd=directory, env=environment, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            figures = list((directory / "plots").glob("*.pdf"))
            self.assertEqual(len(figures), 1)
            self.assertGreater(figures[0].stat().st_size, 1000)
            if shutil.which("pdftotext"):
                text = subprocess.run(["pdftotext", str(figures[0]), "-"],
                                      capture_output=True, text=True, check=True).stdout
                for label in ("MC-as-data", "Synthetic", "4 unique processed events", "Raw dimuon pairs"):
                    self.assertIn(label, text)
                for canary in ("666.123456", "777.234567", "999.123456"):
                    self.assertNotIn(canary, text)
            for canary in ("666.123456", "777.234567", "999.123456"):
                self.assertNotIn(canary, prepared.stdout + prepared.stderr + result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
