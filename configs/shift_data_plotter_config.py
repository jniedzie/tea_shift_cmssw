# Code style: AI-code. AI-generated data commissioning plot configuration.
"""Data-only J/psi control view through the ordinary TEA plotter.

The input is a previously gated aggregate JSON, never a collision Nano file.
These raw data counts have no luminosity or SHIFT-exposure normalization.
"""

from pathlib import Path
import sys

import ROOT
from CmsLabelsManager import CmsLabel
from Histogram import Histogram
from HistogramNormalizer import NormalizationType
from Legend import Legend
from Sample import Sample, SampleType

project_dir = Path(__file__).resolve().parents[1]
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
from utils.shift_blinded_data_plot_inputs import data_plot_labels, prepare_histogram, HISTOGRAM_NAME

# Keep this unscaled data view separate from luminosity-normalized MC. Tests
# may provide these globals explicitly when exercising the ordinary plotter.
output_path = globals().get("output_path", str(project_dir / "plots/data/Run2023D/jpsi_control"))
summary_path = globals().get("summary_path", str(Path(output_path) / "blinded_summary.json"))
histogram_path = str(Path(output_path) / "blinded_histograms.root")
source_labels = data_plot_labels(
    summary_path, {key: globals()[key] for key in ("era", "stream", "run") if key in globals()},
    project_dir / "plots/data/Run2023D/jpsi_control/blinded_summary.json")
era = source_labels["era"]
stream = source_labels["stream"]
control_mass_min_gev, control_mass_max_gev = 2.6, 3.6
summary = prepare_histogram(summary_path, histogram_path, (control_mass_min_gev, control_mass_max_gev))

is_data = summary["input_kind"] == "collision-data"
source_label = "Data" if is_data else "MC-as-data validation"
samples = [Sample(
    name="data_control", file_path=histogram_path, type=SampleType.data,
    legend_description=f"{source_label}: {stream}, {source_labels['run_label']}",
    line_color=ROOT.kBlack, marker_color=ROOT.kBlack, marker_style=20,
    marker_size=0.9, fill_alpha=0.0, line_width=2,
)]
histograms = [Histogram(
    name=HISTOGRAM_NAME, title="", log_y=False, norm_type=NormalizationType.none,
    x_label="Canonical reconstructed dimuon mass [GeV]", y_label="Raw dimuon pairs / bin",
    rebin=1, show_full_x_range=True, allow_empty=True, y_min=0.,
    x_min=control_mass_min_gev, x_max=control_mass_max_gev,
)]
histograms2D = []
show_ratio_plots = False
show_cms_labels = True
cms_label = CmsLabel.private_data if is_data else CmsLabel.private_sim
label_outside_axes = True
beam_label = f"{era}; 13.6 TeV"
luminosity = 1.0  # Unused: every histogram has NormalizationType.none.
plotting_options = {
    SampleType.data: "nostack e0 p", SampleType.background: "hist", SampleType.signal: "nostack hist",
}
legends = {SampleType.data: Legend(
    0.19, 0.76, 0.94, 0.89, "lep", text_size=18,
    title=(f"Commissioning; {summary['unique_processed_events']:,} unique processed events; "
           f"{summary['allowed_pairs']:,} pairs in window"),
)}
custom_stacks_order = ["data_control"]
canvas_size = (1050, 750)
plot_margins = {"left": 0.12, "right": 0.04, "top": 0.11, "bottom": 0.15}
output_formats = ["pdf"]
