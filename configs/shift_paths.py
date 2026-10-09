# base_path = "/pnfs/iihe/cms/store/user/jniedzie/shift_cmssw"  # t2b
base_path = "/eos/home-j/jniedzie/shift_cmssw/"  # lxplus

sample = "jpsi"
# sample = "qcd"
# sample = "dy"

# pt_bin = "0to1"
# pt_bin = "1to2"
# pt_bin = "2to5"
pt_bin = "5to10"
# pt_bin = "10to20"
# pt_bin = "20to-1"

# ==========================
# J/Psi samples
# ==========================


# campaign = "larger_test_condor_run"
# campaign = "small_test"
# campaign = "Charmonium_pThat_50to100GeV_13p6TeV_smallSample"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_10k_beamB"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_100k_beamB"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_timing_2023"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_occupancy_2023"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_triggerProxy_2023"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_piggybackCentral_2023_v1"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_piggybackCentral_bx0_phase0_2023_v1"
# campaign = "Charmonium_FixedTarget_pThat_1to5GeV_13p6TeV_piggybackCentral_bx4_phase0_2023_v1"
# campaign = "lssPaired_control_10k_2023_v1"
# campaign = "lssPaired_materialField_10k_2023_v1"
# campaign = "lssPaired_control_10k_2023_v2"
# campaign = "lssPaired_materialField_10k_2023_v2"

# campaign = "lssPaired_control_10k_2023_v3"
# campaign = "lssPaired_material_10k_2023_v3"
# campaign = "lssPaired_field_10k_2023_v3"
# campaign = "lssPaired_materialField_10k_2023_v3"

# campaign = "lssPaired_field_10k_2023_v4"

# campaign = "lssPaired_materialField_10k_2023_cms_v1"

campaign = "shift_detector_representative_20261007_v10"

# jpsi_campaign_base = "SamplingScan_jpsi_pThat_{}_analysis1k_chunk50_20260921_v1"
jpsi_campaign_base = "Jpsi_Unfiltered_pThat_{}_ATLASproxy_10k_20260921_v1"

# campaign = jpsi_campaign_base.format("1to2")
# campaign = jpsi_campaign_base.format("2to5")
# campaign = jpsi_campaign_base.format("5to10")
# campaign = jpsi_campaign_base.format("10to20")

# ==========================
# QCD samples
# ==========================

# qcd_campaign_base = "WeightedReplay_qcdmu_pThat_{}_p10floor0p1_5k_chunk50_20260921_v1"
qcd_campaign_base = "QCD_UnfilteredDecays_pThat_{}_ATLASproxy_100k_20260921_v1"

# campaign = "QCD_MuEnriched_FixedTarget_pThat_1to5GeV_ATLASproxy_10k_2023_v1"

# ==========================
# Cross sections
# ==========================

cross_sections = {  # pb; combined estimates from all production chunks
    "jpsi": {
        "20toinf": 0.044064467556391526,
        "10to20": 23.906403461436227,
        "5to10": 1881.374468818274,
        "2to5": 93225.77307339365,
        "1to2": 342299.260533131,
        "0to1": 1503369.0870485106,
    },
    "qcd": {
        "20toinf": 1851.4190933761722,
        "10to20": 573197.6708161484,
        "5to10": 37788127.561581455,
        "2to5": 2324492992.016241,
        "1to2": 13003378648.466198,
        "0to1": 50156300701.79254,
    },
    "dy": {
        "20to-1": 1.2761671115668445,
        "10to20": 38.46548547834682,
        "5to10": 458.0153183831834,
        "2to5": 5912.643177243027,
        "1to2": 25020.91681087829,
        "0.5to1": 132153.1881897009,
        "0.211317to0.5": 326341.85535591666,
    },
}


# ==========================
# Automatically set
# ==========================

campaign_base = jpsi_campaign_base if sample == "jpsi" else qcd_campaign_base
# campaign = campaign_base.format(pt_bin)
