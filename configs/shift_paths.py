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


# None selects all, [] selects none. None in the list of lifetimes means prompt.

# Mass scan:
# bsm_masses_to_plot = [0.22, 1.8, 5, 10, 15, 40, 70]  # GeV
# bsm_mean_lab_flights_to_plot = [None]  # metres

# Lifetime scan:
bsm_masses_to_plot = [15]  # GeV
bsm_mean_lab_flights_to_plot = [None, 5, 30, 100, 1000]  # metres

# All signals:
# bsm_masses_to_plot = [0.22, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9, 1.5, 1.8, 2, 5, 10, 12, 15, 20, 30, 40, 50, 70, 90]  # GeV
# bsm_mean_lab_flights_to_plot = [None, 5, 30, 100, 1000]  # metres


# None selects all epsilon samples, [] selects none. Use [] above for epsilon-only plots.
bsm_epsilons_to_plot = []

histogram_cache_directory = None

# ==========================
# SM samples
# ==========================

# Mu-enriched QCD with the existing DY and J/psi references.
# campaign = "20261023_sm_material_csc_v1"
campaign = "20261016_mu_enriched_weekend_v1"
# campaign = "shift_detector_representative_20261007_v10"
# Optional single-sample GEN campaigns; the combined plotter uses campaign above.
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
limits_mass_shape_output_path = "../plots/limits/mass_shapes_20261015/"
limits_cut_output_path = "../plots/limits/both_both_cuts_20261015/"
limits_pheno_comparison_output_path = "../plots/limits/pheno_comparison_20261016/"
limits_resolution_output_path = "../plots/limits/resolution_20261016/"
limits_book_match_output_path = "../plots/limits/book_match_20261016/"
limits_current_output_path = "../plots/limits/current_20261021/"
limits_combine_path = (
    "/eos/home-j/jniedzie/combine/CMSSW_16_0_0/src/HiggsAnalysis/CombinedLimit"
)
