# Code style: human-code. Keep this file short and explicit.

# base_path = "/pnfs/iihe/cms/store/user/jniedzie/shift_cmssw"  # t2b
base_path = "/eos/home-j/jniedzie/shift_cmssw/"  # lxplus

# ==========================
# BSM samples
# ==========================

# None gives SM-only plots.
# Each point's latest link keeps the newest ready sample with matching settings.
bsm_campaign = "latest"
# bsm_campaign = None


# None selects all masses, [] selects none.
# bsm_masses_to_plot = [40]  # GeV
bsm_masses_to_plot = [2, 5, 10, 15, 40, 70]  # GeV
# bsm_masses_to_plot = [2, 5, 10, 12, 15, 20, 30, 40, 50, 70, 90]  # GeV

# None for all lifetimes, or [] for none.  None in the list means prompt.
# bsm_mean_lab_flights_to_plot = [None, 5, 30, 100, 1000]  # metres
bsm_mean_lab_flights_to_plot = [None]  # metres

# None selects all epsilon samples, [] selects none. Use [] above for epsilon-only plots.
bsm_epsilons_to_plot = []

histogram_cache_directory = None


# ==========================
# SM samples
# ==========================

campaign = "shift_detector_representative_20261007_v10"
jpsi_campaign_base = "Jpsi_Unfiltered_pThat_{}_ATLASproxy_10k_20260921_v1"
qcd_campaign_base = "QCD_UnfilteredDecays_pThat_{}_ATLASproxy_100k_20260921_v1"

sample = "jpsi"
# sample = "qcd"
# sample = "dy"

# pt_bin = "0to1"
# pt_bin = "1to2"
# pt_bin = "2to5"
pt_bin = "5to10"
# pt_bin = "10to20"
# pt_bin = "20to-1"


campaign_base = jpsi_campaign_base if sample == "jpsi" else qcd_campaign_base
# campaign = campaign_base.format(pt_bin)

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
# BDT settings
# ==========================

# Optional exact Nano inputs and BDT/event-mixing histogram views.
# Leave these unset for the standard campaign workflow.
nano_merge_inventory = None
bdt_mixing_histogram_inventory = None
bdt_mixing_plot_output = None
bdt_mixing_preparation_directory = None
bdt_mixing_plot_scope = "all"

# ==========================
# Simplified expected limits
# ==========================

limits_study = "first_20261014"
limits_output_path = f"../plots/limits/{limits_study}/"
limits_yields_path = limits_output_path + "yields.json"
limits_lifetime_output_path = "../plots/limits/lifetime_m15_20261015/"
limits_combine_path = (
    "/eos/home-j/jniedzie/combine/CMSSW_16_0_0/src/HiggsAnalysis/CombinedLimit"
)
