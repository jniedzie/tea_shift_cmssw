from shift_paths import base_path, sample, pt_bin, campaign
from shift_storage_paths import tier_directory

sample_path = ""

input_directory = str(tier_directory(base_path, sample, pt_bin, campaign, "nanoAOD"))
output_hists_dir = input_directory
input_file_pattern = "*.root"
