from shift_paths import base_path, sample, campaign
from shift_sample_paths import latest_merged_sample

sample_path = ""
input_directory = f"{base_path}/{sample}/{campaign}/samples/step4_merged"
# Do not process both a stale legacy merge and its complete replacement.
input_file_list = [latest_merged_sample(input_directory)[0]]
output_hists_dir = f"{base_path}/{sample}/{campaign}/histograms"
