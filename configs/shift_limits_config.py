# Code style: human-code.
"""First expected counting limits; run from bin with shift_limits.py."""
from Histogram import Histogram
from HistogramNormalizer import NormalizationType
from shift_limits import load_limit_samples
from shift_paths import limits_output_path, limits_yields_path, limits_combine_path
from shift_paths import limits_lifetime_output_path

output_path = limits_output_path
yields_path = limits_yields_path
combine_path = limits_combine_path
# TEA loads this only in Combine, without passing LD_PRELOAD to helper programs.
combine_preload_path = output_path + "numerics/libShiftStableNLL.so"
prepared_input_manifest = output_path + "inputs/manifest.json"
datacards_output_path = output_path + "datacards/"
results_output_path = output_path + "results/"
plots_output_path = output_path + "brazil/"

luminosity = 3000.0  # pb^-1; the assumed 1% beam-halo exposure is already included.
trigger_efficiency = 1.0
scale_factors = 1.0
confidence_levels = (0.90, 0.95)
# First scan; the remaining audited signal grid is left untouched.
signal_points = ("m2_epsilon_0p01", "m15_prompt", "m30_prompt", "m50_prompt")
categories = ("Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other")
category = "Near-Both"  # Used by the ordinary tea limits_producer entry point.
histogram = Histogram(name="count_" + category, norm_type=NormalizationType.none)

signal_samples, background_samples, limit_metadata = load_limit_samples(prepared_input_manifest)
data_samples = []
samples = signal_samples + background_samples
nuisances = {
    "signal_norm": {"signal": 1.30},
    "background_norm": {sample.name: 1.30 for sample in background_samples},
}
# Each nuisance is shared across channels; signal and background are independent.
# A one-bin shape has the counting likelihood.
include_shapes = True
asimov_observation = True
include_mc_stat_uncertainties = False
retain_zero_backgrounds = True
add_uncertainties_on_zero = False
exclude_backgrounds_for_years = {-1: 0}
symmetric_variations = ()
do_abcd = False
use_combined_limits = False
skip_combine = False
combine_max_workers = 1
combine_options = ()

# One-mass diagnostic scan; leave the original four-point result in place.
lifetime_output_path = limits_lifetime_output_path
lifetime_mass_gev = 15.0
lifetime_points = ("m15_prompt", "m15_lab_5m", "m15_lab_30m",
                   "m15_lab_100m", "m15_lab_1000m")

# The runner makes one ordinary tea Brazil plot for each mass scan and channel.
reference_cross_section = 1.0
x_title = "m_{A'} [GeV]"
y_title = "Expected upper limit on #sigma #times B(A' #rightarrow #mu#mu) [pb]"
x_min = 1.5
x_max = 100.0
y_min = 1.e-5
y_max = 1.e9
log_x = True
log_y = True
experiment_label = "#bf{SHIFT} #it{Simulation}"
luminosity_label = "3 fb^{-1}, assumed trigger efficiency 100%"
