# Code style: human-code.
"""Fine mass bins, with explicit interpolation of sparse background templates."""
from shift_limits_shapes_config import *
from shift_paths import limits_resolution_output_path

output_path = limits_resolution_output_path
source_mass_study_path = limits_mass_shape_output_path
yields_path = output_path + "p0/yields.json"
prepared_input_manifest = output_path + "p0/inputs/manifest.json"
datacards_output_path = output_path + "p0/datacards/"
results_output_path = output_path + "p0/results/"
plots_output_path = output_path + "p0/brazil/"
split_mass_bins = False

# This is a provisional resolution scale, not a measured SHIFT core resolution.
# The saved 0.1 GeV histogram grid sets the smallest available bin width.
mass_resolution_fraction = 0.02
mass_minimum_bin_gev = 0.1
background_interpolation_powers = (0.0, 2.0)
confidence_levels = (0.95, 0.90)
combine_max_workers = 1
combine_options = ()

signal_samples, background_samples, limit_metadata = load_limit_samples(prepared_input_manifest)
samples = signal_samples + background_samples
nuisances = {
    "signal_norm": {"signal": 1.30},
    "background_norm": {sample.name: 1.30 for sample in background_samples},
}
