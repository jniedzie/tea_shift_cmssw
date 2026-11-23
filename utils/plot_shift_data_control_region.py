#!/usr/bin/env python3
# Code style: AI-code. AI-generated blinded control-region plot rendering.
"""Render the All and Both-Both data plots from approved aggregate counts."""

import argparse
import json
import math
from pathlib import Path
import tempfile


MASS_EDGES = tuple(round(2.6 + index * 0.05, 2) for index in range(21))
CATEGORIES = (
    ("All dimuons", "all_bins", "ShiftDimuonVertex_mass.pdf"),
    ("Both-Both dimuons", "both_both_bins", "ShiftDimuonVertexBoth-Both_mass.pdf"),
)


def luminosity_label(luminosity):
    if luminosity is None:
        return "Processed luminosity unknown"
    if (type(luminosity) not in (int, float) or not math.isfinite(luminosity)
            or luminosity <= 0):
        raise ValueError("Recorded luminosity must be positive and finite, or null if unknown")
    precision = f"{luminosity:.4f}" if luminosity >= 0.0001 else f"{luminosity:.4g}"
    return f"{precision} pb⁻¹ recorded CMS luminosity"


def validate_aggregate(aggregate):
    if not isinstance(aggregate, dict):
        raise ValueError("Expected a control-region aggregate")
    edges = aggregate.get("mass_bin_edges_gev")
    if (not isinstance(edges, list) or len(edges) != len(MASS_EDGES)
            or any(type(value) not in (int, float) or not math.isclose(
                value, expected, rel_tol=0, abs_tol=1e-12
            ) for value, expected in zip(edges, MASS_EDGES))):
        raise ValueError("Expected the approved 2.6–3.6 GeV mass bins")
    samples = aggregate.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("At least one sample is required")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Invalid control-region sample")
        selection = sample.get("selection")
        streams = sample.get("streams")
        sections = sample.get("closed_lumisections")
        if (not isinstance(selection, str) or not selection.strip()
                or not isinstance(streams, list) or not streams
                or any(not isinstance(stream, str) or not stream.strip() for stream in streams)
                or type(sections) is not int or sections < 0):
            raise ValueError("A sample requires its selection, streams, and complete lumisection count")
        if "recorded_luminosity_pb_inverse" not in sample:
            raise ValueError("A sample must declare its recorded luminosity or mark it unknown")
        luminosity = sample["recorded_luminosity_pb_inverse"]
        luminosity_label(luminosity)
        if luminosity is not None and sections == 0:
            raise ValueError("Recorded luminosity requires at least one complete lumisection")
        for _, key, _ in CATEGORIES:
            counts = sample.get(key)
            if (not isinstance(counts, list) or len(counts) != len(edges) - 1
                    or any(type(count) is not int or count < 0 for count in counts)):
                raise ValueError("Expected one nonnegative integer count per mass bin")
        if any(both > inclusive for both, inclusive in zip(sample["both_both_bins"], sample["all_bins"])):
            raise ValueError("Both-Both counts must be a subset of All in every bin")
    return edges, samples


def render_category(edges, samples, category, key, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    centers = [(left + right) / 2 for left, right in zip(edges, edges[1:])]
    upper = max(count + math.sqrt(count) for sample in samples for count in sample[key])
    with plt.rc_context({"pdf.fonttype": 42}):
        figure, axes = plt.subplots(1, len(samples), figsize=(6.2 * len(samples), 5.3),
                                    squeeze=False, constrained_layout=True)
        try:
            for axis, sample in zip(axes[0], samples):
                counts = sample[key]
                sections = sample["closed_lumisections"]
                pairs = sum(counts)
                axis.stairs(counts, edges, color="#254f77", linewidth=1.5)
                axis.errorbar(centers, counts, yerr=[math.sqrt(count) for count in counts],
                              fmt="o", color="#102738", markersize=3, linewidth=1, clip_on=True)
                axis.set(xlim=(edges[0], edges[-1]), ylim=(0, max(1.25, upper * 1.18)),
                         xlabel="Reconstructed dimuon mass [GeV]", ylabel="Raw dimuon pairs / 50 MeV")
                axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                axis.grid(axis="y", color="#e0e6eb", linewidth=0.7)
                axis.set_axisbelow(True)
                axis.set_title(sample["selection"], loc="left", fontsize=12, fontweight="bold", pad=51)
                label = luminosity_label(sample["recorded_luminosity_pb_inverse"])
                axis.text(0, 1.143, label, transform=axis.transAxes, fontsize=10, va="bottom")
                axis.text(0, 1.077, " + ".join(sample["streams"]),
                          transform=axis.transAxes, fontsize=9, va="bottom")
                axis.text(0, 1.022, (
                    f"{sections:,} complete lumisection{'s' if sections != 1 else ''}  ·  "
                    f"{pairs:,} pair{'s' if pairs != 1 else ''} in the window"
                ), transform=axis.transAxes, fontsize=9, va="bottom")
            title = f"CMS Run 2023D data control region · {category}"
            figure.suptitle(title, fontsize=15, fontweight="bold")
            figure.savefig(path, format="pdf", metadata={"Title": title})
        finally:
            plt.close(figure)


def render_control_region(aggregate, output_directory):
    """Write both figures with a luminosity label on every sample panel."""
    edges, samples = validate_aggregate(aggregate)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output / filename for _, _, filename in CATEGORIES)
    with tempfile.TemporaryDirectory(prefix=".data-control-plot-", dir=output) as scratch:
        for category, key, filename in CATEGORIES:
            path = Path(scratch) / filename
            render_category(edges, samples, category, key, path)
            if not path.read_bytes().startswith(b"%PDF"):
                raise ValueError("The control-region plot did not produce a PDF")
        for destination in outputs:
            (Path(scratch) / destination.name).replace(destination)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_control_region(json.loads(args.summary.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
