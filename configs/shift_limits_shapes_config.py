# Code style: human-code.
"""Mass-shape expected limits, with the same exposure and nuisances as before."""
from shift_limits_config import *
from shift_paths import limits_mass_shape_output_path

audited_yields_path = limits_yields_path
output_path = limits_mass_shape_output_path
yields_path = output_path + "shape_yields.json"
prepared_input_manifest = output_path + "poisson_bins/inputs/manifest.json"
datacards_output_path = output_path + "poisson_bins/datacards/"
results_output_path = output_path + "poisson_bins/results/"
plots_output_path = output_path + "brazil/"
histogram_prefix = "mass_"
split_mass_bins = True
# The lifetime bound makes the boundary prompt at every mass.
signal_points = ("m2_prompt", "m5_prompt", "m10_prompt", "m12_prompt",
                 "m15_prompt", "m20_prompt", "m30_prompt", "m40_prompt",
                 "m50_prompt", "m70_prompt", "m90_prompt")
background_minimum_bin_ess = 10.0
confidence_levels = (0.95, 0.90)
combine_max_workers = 1
combine_options = ()

# Physical interpolation stays inside the generated mass range. Prompt
# production is extrapolated in epsilon only with the stated epsilon^2 model.
epsilon_min = 1.e-10
epsilon_max = 0.1  # Upper edge of the small-mixing coupling check.
# Plot the complete formal benchmark, including explicitly labelled extrapolation.
epsilon_plot_max = 1.0
epsilon_grid_points = 101
prompt_maximum_mean_flight_mm = 0.01
lifetime_minimum_selected_ess = 10.0

# Native TEA defaults also resolve to the prepared mass inputs.
histogram = Histogram(name=histogram_prefix + category, norm_type=NormalizationType.none)
signal_samples, background_samples, limit_metadata = load_limit_samples(prepared_input_manifest)
samples = signal_samples + background_samples
nuisances = {
    "signal_norm": {"signal": 1.30},
    "background_norm": {sample.name: 1.30 for sample in background_samples},
}
