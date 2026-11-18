#!/usr/bin/env python3
# Code style: AI-code. AI-generated preparation of blinded aggregate plot inputs.
"""Prepare one unscaled control histogram from allowlisted aggregate JSON only.

This module never opens collision Nano files, recomputes a dimuon, accesses a
classifier, or invents a SHIFT exposure.
"""

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

try:
    from .shift_blinded_data_summary import BlindingError, validate_summary
except ImportError:
    from shift_blinded_data_summary import BlindingError, validate_summary


HISTOGRAM_NAME = "ShiftDimuonVertex_mass"
METADATA_NAME = "ShiftDataControlSummary"
CONTEXT_NAME = "plot_context.json"


def validate_plot_context(context):
    keys = {"schema", "era", "datasets", "runs", "summary_sha256"}
    if (not isinstance(context, dict) or set(context) != keys
            or context["schema"] != "shift-data-control-plot-context-v1"
            or context["era"] != "Run2023D"):
        raise BlindingError("Unsupported data plot context")
    datasets, runs = context["datasets"], context["runs"]
    if (not isinstance(datasets, list) or not datasets
            or any(not isinstance(value, str) or not re.fullmatch(
                r"/[A-Za-z0-9_-]+/Run2023D-v[0-9]+/RAW", value) for value in datasets)
            or datasets != sorted(set(datasets))
            or not isinstance(runs, list) or not runs
            or any(type(value) is not int or not 0 < value < 2**32 for value in runs)
            or runs != sorted(set(runs))
            or not isinstance(context["summary_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", context["summary_sha256"])):
        raise BlindingError("Invalid recorded-data source context")
    return context


def read_plot_context(summary_path):
    summary_path = Path(summary_path)
    context_path = summary_path.parent / CONTEXT_NAME
    if not context_path.exists():
        return None
    try:
        context = validate_plot_context(json.loads(context_path.read_text()))
        digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        raise BlindingError("Data plot context is unavailable or invalid") from None
    if context["summary_sha256"] != digest:
        raise BlindingError("Data plot context belongs to a different aggregate summary")
    return context


def plot_context_labels(context):
    validate_plot_context(context)
    primaries = [value.split("/")[1] for value in context["datasets"]]
    stream = " + ".join(primaries) if len(primaries) <= 2 else f"{len(primaries)} recorded RAW streams"
    runs = context["runs"]
    run_label = (f"run {runs[0]}" if len(runs) == 1
                 else f"runs {runs[0]}-{runs[-1]} ({len(runs)} runs)")
    return dict(era=context["era"], stream=stream, run_label=run_label)


def data_plot_labels(summary_path, overrides, legacy_summary_path):
    context = read_plot_context(summary_path)
    if context is not None:
        labels = plot_context_labels(context)
        expected = dict(era=labels["era"], stream=labels["stream"], run=labels["run_label"])
        for key, value in overrides.items():
            if (f"run {value}" if key == "run" else value) != expected[key]:
                raise BlindingError("Explicit plot labels conflict with the recorded-data context")
        return labels
    if (Path(summary_path).resolve() != Path(legacy_summary_path).resolve()
            and set(overrides) != {"era", "stream", "run"}):
        raise BlindingError("A nonlegacy data plot requires its source context")
    return dict(era=overrides.get("era", "Run2023D"), stream=overrides.get("stream", "ZeroBias"),
                run_label="run " + str(overrides.get("run", "369943")))


def read_summary(path):
    try:
        payload = Path(path).read_bytes()
        summary = json.loads(payload)
    except (OSError, ValueError):
        raise BlindingError("The aggregate data summary is unavailable or invalid") from None
    validate_summary(summary)
    return summary, hashlib.sha256(payload).hexdigest()


def check_band(summary, expected_band):
    if (not isinstance(expected_band, (tuple, list)) or len(expected_band) != 2
            or list(expected_band) != [summary["policy"]["mass_min_gev"], summary["policy"]["mass_max_gev"]]):
        raise BlindingError("Aggregate input differs from the explicitly configured plot band")


def histogram_metadata(summary, source_sha256):
    return dict(
        schema="shift-data-control-histogram-v1", summary_sha256=source_sha256,
        policy_sha256=summary["policy_sha256"], qualification_sha256=summary["qualification_sha256"],
        input_provenance_sha256=summary["input_provenance_sha256"], input_kind=summary["input_kind"],
        unique_processed_events=summary["unique_processed_events"], allowed_pairs=summary["allowed_pairs"],
        events_with_allowed_pair=summary["events_with_allowed_pair"], normalization="none",
        counting_errors="Poisson", purpose="jpsi-control-commissioning",
    )


def verify_histogram(summary_path, histogram_path, expected_band):
    """Read back the exact safe histogram semantics before reusing current bytes."""
    summary, source_sha256 = read_summary(summary_path)
    check_band(summary, expected_band)
    import ROOT
    source = ROOT.TFile.Open(str(histogram_path), "READ")
    try:
        if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
            raise BlindingError("Aggregate histogram input is unreadable")
        keys = [key.GetName() for key in source.GetListOfKeys()]
        if len(keys) != 2 or set(keys) != {HISTOGRAM_NAME, METADATA_NAME}:
            raise BlindingError("Aggregate histogram input has an unsupported object schema")
        histogram = source.Get(HISTOGRAM_NAME)
        metadata = source.Get(METADATA_NAME)
        counts, edges = summary["mass_bin_counts"], summary["mass_bin_edges_gev"]
        if (not histogram.InheritsFrom("TH1") or histogram.GetDimension() != 1
                or histogram.GetNbinsX() != len(counts)
                or [histogram.GetBinLowEdge(i) for i in range(1, len(counts) + 2)] != edges
                or [histogram.GetBinContent(i) for i in range(1, len(counts) + 1)] != counts
                or histogram.GetBinContent(0) != 0 or histogram.GetBinContent(len(counts) + 1) != 0
                or histogram.GetEntries() != summary["allowed_pairs"]
                or histogram.GetBinErrorOption() != ROOT.TH1.kPoisson
                or not metadata.InheritsFrom("TObjString")
                or json.loads(metadata.GetString().Data()) != histogram_metadata(summary, source_sha256)):
            raise BlindingError("Aggregate histogram input differs from the safe source summary")
    finally:
        if source:
            source.Close()
    return summary


def prepare_histogram(summary_path, histogram_path, expected_band):
    """Publish exactly one count histogram, zero flows and compact provenance."""
    summary, source_sha256 = read_summary(summary_path)
    check_band(summary, expected_band)
    # Ordinary replots must not mutate an already qualified current bundle.
    if Path(histogram_path).exists():
        return verify_histogram(summary_path, histogram_path, expected_band)
    import ROOT
    ROOT.gROOT.SetBatch(True)
    histogram_path = Path(histogram_path)
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    output = None
    try:
        with tempfile.NamedTemporaryFile(dir=histogram_path.parent, suffix=".root", delete=False) as stream:
            temporary = Path(stream.name)
        output = ROOT.TFile(str(temporary), "RECREATE")
        if output.IsZombie():
            raise BlindingError("Cannot create the aggregate histogram input")
        edges, counts = summary["mass_bin_edges_gev"], summary["mass_bin_counts"]
        histogram = ROOT.TH1D(HISTOGRAM_NAME, "", len(counts), array("d", edges))
        for index, count in enumerate(counts, 1):
            histogram.SetBinContent(index, count)
        histogram.SetBinContent(0, 0.)
        histogram.SetBinContent(len(counts) + 1, 0.)
        histogram.SetEntries(summary["allowed_pairs"])
        histogram.SetBinErrorOption(ROOT.TH1.kPoisson)
        histogram.Write()
        metadata = histogram_metadata(summary, source_sha256)
        ROOT.TObjString(json.dumps(metadata, sort_keys=True, allow_nan=False)).Write(METADATA_NAME)
        output.Close()
        output = None
        # This is a reproducible current plot input; original summaries and
        # qualification records are never overwritten here.
        os.replace(temporary, histogram_path)
        temporary = None
    finally:
        if output is not None:
            output.Close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mass-min-gev", type=float, required=True)
    parser.add_argument("--mass-max-gev", type=float, required=True)
    args = parser.parse_args()
    try:
        prepare_histogram(args.summary, args.output_root, (args.mass_min_gev, args.mass_max_gev))
    except Exception:
        raise SystemExit("Blinded aggregate histogram preparation failed") from None
    print("Blinded aggregate histogram prepared")


if __name__ == "__main__":
    main()
