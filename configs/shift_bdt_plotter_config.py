"""Shape overlays from explicitly prepared, exposure-correct process histograms.

Set SHIFT_BDT_PLOT_INPUTS to the completed plot_inputs.json produced by
prepare_shift_bdt_plots.py. This configuration never discovers a live campaign.
"""

import json
import os
from pathlib import Path

import ROOT
from Sample import Sample, SampleType
from Legend import Legend
from Histogram import Histogram, Histogram2D
from HistogramNormalizer import NormalizationType
from CmsLabelsManager import CmsLabel
from shift_bdt_histograms import CATEGORIES, MAP_AXES, histogram_path


manifest_path = Path(os.environ["SHIFT_BDT_PLOT_INPUTS"])
manifest = json.loads(manifest_path.read_text())
if manifest.get("schema") != "shift-dimuon-bdt-plot-inputs-v1" or manifest.get("complete") is not True:
    raise RuntimeError("BDT plots require a completed process-shape preparation manifest")
if not manifest.get("samples"):
    raise RuntimeError("BDT plot input manifest contains no samples")

process_colors = {"jpsi": ROOT.kRed + 1, "qcd": ROOT.kBlue + 1, "dy": ROOT.kGreen + 2}
samples = []
for index, row in enumerate(manifest["samples"]):
    if row["process"] not in ("jpsi", "qcd", "dy", "signal"):
        raise RuntimeError(f"Unknown BDT plot process: {row['process']}")
    samples.append(Sample(
        name=row["name"], file_path=row["histogram_path"], type=SampleType.signal,
        legend_description=row.get("label", row["name"]),
        line_color=process_colors.get(row["process"], ROOT.kMagenta + index % 5),
        line_width=3, line_alpha=1.0, fill_alpha=0.0, marker_size=0.0,
    ))

histograms = []
for category in ("", "Both-Both"):
    for label in ("", "Genuine", "Accidental", "Unknown"):
        histograms.append(Histogram(
            name=histogram_path(category, label), log_y=False,
            norm_type=NormalizationType.to_one, x_label="dimuon vertex BDT score",
            y_label="fraction of retained dimuon pairs / bin", show_full_x_range=True,
            suffix="_shape", x_min=0.0, x_max=1.0,
        ))

histograms2D = [Histogram2D(
    name=histogram_path(category) + "Vs" + axis, log_z=True,
    norm_type=NormalizationType.none, x_label=title,
    y_label="dimuon vertex BDT score", z_label="exposure-correct weighted pairs",
) for category in ("", "Both-Both") for axis, (_, _, _, title) in MAP_AXES.items()]

output_path = os.environ.get("SHIFT_BDT_PLOT_OUTPUT", str(manifest_path.parent / "plots"))
output_formats = ["pdf", "png"]
custom_stacks_order = [sample.name for sample in samples]
plotting_options = {SampleType.signal: "nostack hist", SampleType.background: "hist", SampleType.data: "nostack e"}
legends = {SampleType.signal: Legend(0.14, 0.65, 0.60, 0.88, "l", text_size=18)}
canvas_size = (900, 700)
canvas_size_2Dhists = (900, 700)
show_ratio_plots = False
show_cms_labels = True
cms_label = CmsLabel.pas_sim
label_outside_axes = True
luminosity = 1.0
beam_label = "SHIFT simulation"
