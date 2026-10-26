# Code style: human-code.
"""Current published DP lifetimes and raw QCD+DY+J/psi mass templates."""
from shift_limits_config import *
from shift_paths import limits_current_output_path, base_path, campaign, cross_sections

output_path = limits_current_output_path
luminosity = 3000.0  # pb^-1; includes the assumed 1% halo exposure.
confidence_levels = (0.95, 0.90)
background_modes = ("background", "zero_background")
categories = ("Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other")
channel_names = ("combined",) + categories
primary_channel = "Both-Both"
plot_omitted_masses_gev = (1.5,)  # Temporary plot omission pending review of the visible spike.
native_confidence_levels = (0.95,)
signal_campaign = "latest"
signal_masses = None
signal_flights = None
signal_epsilons = None
response_anchors = ("prompt", "lab_5m", "lab_30m", "lab_100m", "lab_1000m")
mass_resolution_fraction = 0.02  # Provisional scale; no measured core resolution claimed.
mass_minimum_bin_gev = 0.1
background_interpolation = False
include_mc_stat_uncertainties = False
combine_max_workers = 1
combine_options = ()
epsilon_min = 1.e-10
epsilon_max = 1.0
small_mixing_max = 0.1
scan_grid_points = 65
# Lifetimes are interpolated using the existing reconstructed templates.
# Extend the prompt template to shorter lifetimes; do not extend displaced ones.
projection_experiments = ("SHiP", "BelleII", "LHCb300", "HL-LHC", "FCC-hh", "FCC-ee", "MuC10")
