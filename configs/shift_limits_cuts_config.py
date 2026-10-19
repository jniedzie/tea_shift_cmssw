# Code style: human-code.
"""Small Both-Both cut comparison, preserving the original exposure."""
from shift_limits_config import *
from shift_paths import limits_cut_output_path

audited_yields_path = limits_yields_path
output_path = limits_cut_output_path
categories = ("Both-Both",)
category = "Both-Both"
# Inspect the overlap with the paper and the previous lifetime benchmark.
study_masses = (2.0, 15.0)
# The source has a 0.5 m longitudinal spread. These are reconstructed distances
# from its nominal centre, not generated flight lengths or significances.
nominal_target_cm = (0.0, 0.0, 14800.0)
cut_scenarios = (
    ("baseline", 0.0, 0.0),
    ("p20", 20.0, 0.0),
    ("L1", 0.0, 1.0),
    ("p20_L1", 20.0, 1.0),
    ("p20_L10", 20.0, 10.0),
)
yields_path = output_path + "p20_L10/yields.json"
prepared_input_manifest = output_path + "p20_L10/inputs/manifest.json"
datacards_output_path = output_path + "p20_L10/datacards/"
results_output_path = output_path + "p20_L10/results/"
plots_output_path = output_path + "p20_L10/brazil/"
signal_points = ("m2_prompt", "m15_prompt")
confidence_levels = (0.95,)
histogram = Histogram(name="count_" + category, norm_type=NormalizationType.none)
signal_samples, background_samples, limit_metadata = load_limit_samples(prepared_input_manifest)
samples = signal_samples + background_samples
nuisances = {
    "signal_norm": {"signal": 1.30},
    "background_norm": {sample.name: 1.30 for sample in background_samples},
}
