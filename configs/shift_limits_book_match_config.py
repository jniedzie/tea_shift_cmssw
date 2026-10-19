# Code style: human-code.
"""Diagnostic comparison with the briefing-book exposure and raw MC background."""
from shift_limits_config import *
from shift_paths import limits_book_match_output_path

output_path = limits_book_match_output_path
audited_yields_path = limits_yields_path
luminosity = 30000.0  # pb^-1; 1% of 3000 fb^-1, already including halo fraction.
categories = ("Both-Both",)
category = "Both-Both"
signal_points = ("m2_prompt", "m5_prompt", "m10_prompt")
confidence_levels = (0.90, 0.95)
histogram_prefix = "mass_"
split_mass_bins = False
raw_background_zeros = True  # User-authorized optimistic diagnostic, not a tail prediction.
mass_bin_width_gev = 0.1
background_systematic = 0.30
signal_systematic = 0.30
luminosity_label = "30 fb^{-1}; raw MC background; diagnostic"
