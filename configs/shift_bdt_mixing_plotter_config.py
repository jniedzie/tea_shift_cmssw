"""Optional BDT and event-mixing overlays from immutable study inputs.

The standard reconstruction plotter remains shift_plotter_config.py.
This config never discovers canonical samples or their normalization records.
Histogram input validation and process-bin preparation live in shift_bdt_mixing_histogram_inputs.
"""

import json
import sys
from pathlib import Path

import ROOT
from Sample import Sample, SampleType
from Legend import Legend
from Histogram import Histogram, Histogram2D
from HistogramNormalizer import NormalizationType
from CmsLabelsManager import CmsLabel
from shift_paths import (
    bdt_mixing_histogram_inventory, bdt_mixing_plot_output,
    bdt_mixing_preparation_directory, bdt_mixing_plot_scope,
)

project_dir = str(Path(__file__).resolve().parents[1])
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from utils.shift_bdt_mixing_histogram_inputs import (
    configured_inventory, plot_input_rows, readable_histogram_path,
)

plot_inputs_path = bdt_mixing_histogram_inventory
if not plot_inputs_path:
    raise ValueError("Set bdt_mixing_histogram_inventory in configs/shift_paths.py")
plot_input_manifest = json.loads(Path(plot_inputs_path).read_text())
if plot_input_manifest.get("schema") == "shift-event-mixing-plot-inputs-v1":
    plot_mode = "mixing"
elif plot_input_manifest.get("schema") in (
    "shift-dimuon-bdt-bin-inputs-v1", "shift-dimuon-bdt-plot-inputs-v1",
):
    plot_mode = "bdt"
else:
    raise ValueError("Unsupported BDT/event-mixing histogram inventory")

show_ratio_plots = False
show_cms_labels = True
cms_label = CmsLabel.pas_sim
label_outside_axes = True
auto_adjust_axes_for_legend = True
plotting_options = {
    SampleType.signal: "nostack hist", SampleType.background: "hist", SampleType.data: "nostack e",
}


def saved_score_plots():
    """Use the same score domain and reconstructed map axes as histogramming."""
    from shift_histogrammer_config import BDT_MAP_AXES, bdt_histogram_path
    scores = [Histogram(
        name=bdt_histogram_path(category, label), log_y=False,
        norm_type=NormalizationType.to_one, x_label="dimuon vertex BDT score",
        y_label="fraction of retained dimuon pairs / bin", show_full_x_range=True,
        suffix="_shape", x_min=0.0, x_max=1.0,
    ) for category in ("", "Both-Both") for label in ("", "Genuine", "Accidental", "Unknown")]
    maps = [Histogram2D(
        name=bdt_histogram_path(category) + "Vs" + axis, log_z=True,
        norm_type=NormalizationType.none, x_label=title,
        y_label="dimuon vertex BDT score", z_label="exposure-correct weighted pairs",
    ) for category in ("", "Both-Both") for axis, (_, _, _, title) in BDT_MAP_AXES.items()]
    return scores, maps

def inventory_plot_samples(manifest, scope="all", mixing=False):
    """Process shapes and independent signal points are line overlays."""
    if scope not in ("all", "sm"):
        raise ValueError("bdt_mixing_plot_scope must be all or sm")
    rows = plot_input_rows(manifest)
    process_colors = {"jpsi": ROOT.kRed + 1, "qcd": ROOT.kBlue + 1, "dy": ROOT.kGreen + 2}
    signal_names = sorted(row["name"] for row in rows if row.get("process") == "signal")
    signal_colors = (ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kCyan + 2,
                     ROOT.kViolet + 1, ROOT.kAzure + 7, ROOT.kGray + 2,
                     ROOT.kPink + 7, ROOT.kSpring + 5, ROOT.kTeal + 2)
    result = []
    for index, row in enumerate(rows):
        if mixing:
            color = (ROOT.kBlack, ROOT.kBlue + 1, ROOT.kRed + 1,
                     ROOT.kGreen + 2, ROOT.kMagenta + 1)[index % 5]
            style = 1 + index
        else:
            process = row["process"]
            if process not in ("jpsi", "qcd", "dy", "signal"):
                raise ValueError("Unknown plot process: " + str(process))
            if scope == "sm" and process == "signal":
                continue
            signal_index = signal_names.index(row["name"]) if process == "signal" else None
            color = (process_colors[process] if signal_index is None
                     else signal_colors[signal_index % len(signal_colors)])
            style = 1 if signal_index is None else 2 + signal_index // 3
        result.append(Sample(
            name=row["name"], file_path=str(readable_histogram_path(row["histogram_path"])),
            type=SampleType.signal, legend_description=row.get("label", row["name"]),
            line_color=color, line_style=style, line_width=3, line_alpha=1.0,
            fill_alpha=0.0, marker_size=0.0,
        ))
    if not result:
        raise ValueError("Plot scope has no completed samples")
    return result

def event_mixing_plots():
    """Keep unvalidated rate models separate from raw mixing trials."""
    result = []
    for category in ("", "Both-Both"):
        for variable, label in (
            ("mass_model", "model cross section [pb/GeV]"),
            ("mass_trials", "trial pairs / GeV (shape study)"),
        ):
            for suffix, upper in (("", 120.0), ("_low_mass", 12.0)):
                result.append(Histogram(
                    name=f"mixing/ShiftDimuonVertex{category}_{variable}",
                    title="Event mixing: rate and shape transfer unvalidated", log_y=True,
                    norm_type=NormalizationType.none, x_min=0.0, x_max=upper,
                    x_label="reconstructed dimuon mass [GeV]", y_label=label,
                    scale_by_bin_width=True, suffix=suffix,
                ))
    return result


scope = bdt_mixing_plot_scope
plot_output = (bdt_mixing_plot_output or
               str(Path(plot_inputs_path).parent / ("plots_sm" if scope == "sm" else "plots")))
is_mixing = plot_mode == "mixing"
plot_inventory = configured_inventory(
    plot_inputs_path, plot_output, bdt_mixing_preparation_directory)
expected_schema = ("shift-event-mixing-plot-inputs-v1" if is_mixing else "shift-dimuon-bdt-plot-inputs-v1")
if plot_inventory["schema"] != expected_schema:
    raise ValueError("Plot mode does not match immutable inventory schema")
samples = inventory_plot_samples(plot_inventory, scope, mixing=is_mixing)
if is_mixing:
    histograms, histograms2D = event_mixing_plots(), []
else:
    histograms, histograms2D = saved_score_plots()
output_path = plot_output
output_formats = ["pdf"]
custom_stacks_order = [sample.name for sample in samples]
many_samples = len(samples) > 5
legends = {SampleType.signal: (
    Legend(0.68, max(0.18, 0.88 - 0.048 * len(samples)), 0.99, 0.88, "l", text_size=19)
    if many_samples else Legend(0.14, 0.65, 0.60, 0.88, "l", text_size=18)
)}
canvas_size = (1300, 750) if many_samples else (900, 700)
if many_samples:
    plot_margins = {"left": 0.10, "right": 0.35, "top": 0.09, "bottom": 0.15}
canvas_size_2Dhists = (900, 700)
luminosity = 1.0
beam_label = "SHIFT simulation; mixing unvalidated" if is_mixing else "SHIFT simulation"
physics_valid = normalization_ready = False
signal_normalization_validated = False
