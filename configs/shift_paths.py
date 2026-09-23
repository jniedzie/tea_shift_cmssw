# base_path = "/pnfs/iihe/cms/store/user/jniedzie/shift_cmssw"  # t2b
base_path = "/eos/home-j/jniedzie/shift_cmssw/"  # lxplus

# sample = "jpsi"
sample = "qcd"

pt_bin = "1to2"
# pt_bin = "2to5"
# pt_bin = "5to10"
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

cross_sections = {  # pb; combined estimates from all chunks
    "jpsi": {
        "1to2": 621357.8611509507,
        "2to5": 116404.48317891415,
        "5to10": 2030.7448141130817,
    },
    "qcd": {
        "1to2": 36563971991.23754,
        "2to5": 3655743759.3925757,
        "5to10": 40792126.38657331,
    },
}


# ==========================
# Automatically set
# ==========================

campaign_base = jpsi_campaign_base if sample == "jpsi" else qcd_campaign_base
campaign = campaign_base.format(pt_bin)
