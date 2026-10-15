# Code style: AI-code. AI-generated plotter for frozen source-rate comparisons.
"""Plot audited source-rate estimates and their Monte Carlo sampling errors."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def plot_rates(comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = comparison["samples"]
    components = comparison["components"]
    figure, axes = plt.subplots(1, len(components), figsize=(10.2, 4.3), squeeze=False)
    colors = ["#777777", "#2271b2", "#e69f00"]
    for column, component in enumerate(components):
        axis = axes[0, column]
        for row, sample in enumerate(samples):
            point = sample["rates"][component["key"]]
            value, error = point["rate_pb"], point["sampling_error_pb"]
            if not math.isfinite(value) or not math.isfinite(error) or error < 0:
                raise ValueError("Invalid rate or sampling error")
            axis.errorbar(value, row, xerr=error, fmt="o", markersize=7,
                          color=colors[row % len(colors)], capsize=4, linewidth=1.8)
            axis.text(0.985, row + 0.20,
                      f"{value:.2f} ± {error:.2f} pb; {point['raw_sources']} sources, "
                      f"{point['effective_sources']:.1f} effective",
                      transform=axis.get_yaxis_transform(), ha="right", va="center", fontsize=8.5)
        axis.set_yticks(list(range(len(samples))))
        axis.set_yticklabels([sample["label"] for sample in samples], fontsize=10)
        axis.set_ylim(len(samples) - 0.45, -0.45)
        maximum = max(sample["rates"][component["key"]]["rate_pb"] +
                      sample["rates"][component["key"]]["sampling_error_pb"] for sample in samples)
        axis.set_xlim(0, 1.18 * maximum)
        axis.set_xlabel("First hard-process source rate [pb]")
        axis.set_title(component["label"], fontsize=12)
        axis.grid(axis="x", alpha=0.18)
    figure.suptitle(comparison["title"], fontsize=14)
    figure.text(0.5, 0.035, "\n".join(comparison["notes"]), ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.20, 1, 0.94), w_pad=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.input.read_text())
    plot_rates(comparison, args.output)
    receipt = {
        "schema": "shift-source-rate-comparison-figure-v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "plotter": str(Path(__file__).resolve()),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "numbers": comparison,
        "source_density_closure_established": False,
        "analysis_release": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
