import os

from shift_paths import base_path, sample, campaign
from shift_sample_paths import latest_merged_sample, validated_nano_files

sample_path = ""
# input_directory = f"{base_path}/{sample}/{campaign}/samples/step4_merged"
# Do not process both a stale legacy merge and its complete replacement.
# input_file_list = [latest_merged_sample(input_directory)[0]]
# output_hists_dir = f"{base_path}/{sample}/{campaign}/histograms"
# Exact complete inventory: one unique histogram output per Nano file across
# all six QCD bins, six J/psi bins and seven DY mass bins, in one job factory.
input_output_file_list = validated_nano_files(
    base_path, campaign,
    allow_incomplete=os.environ.get("SHIFT_ALLOW_INCOMPLETE_NANO") == "1",
)
