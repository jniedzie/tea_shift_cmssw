from shift_paths import base_path, sample, pt_bin, campaign, nano_merge_inventory
from shift_sample_paths import load_nano_merge_groups
from shift_storage_paths import tier_directory

sample_path = ""

if nano_merge_inventory:
    merge_input_groups = load_nano_merge_groups(nano_merge_inventory, base_path)
else:
    input_directory = str(tier_directory(base_path, sample, pt_bin, campaign, "nanoAOD"))
    output_hists_dir = input_directory
    input_file_pattern = "*.root"
