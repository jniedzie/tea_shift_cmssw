"""BDT diagnostics only: every existing dimuon is retained, with no score cut."""

SCORE_BRANCH = "ShiftDimuonVertex_bdtScore"
DIRECTORY = "dimuon_bdt"
CATEGORIES = ("", "Good", "Near-Both", "Near-Far", "Both-Both", "Both-Far", "Other")
LABEL_SUFFIXES = ("", "Genuine", "Accidental", "Unknown")
SCORE_BINS = (100, 0.0, 1.0)
MAP_AXES = {
    "Mass": (100, 0.0, 100.0, "m_{#mu#mu}^{reco} [GeV]"),
    "Vz": (100, -20000.0, 20000.0, "v_{z}^{reco} [cm]"),
    "VertexR": (100, 0.0, 300.0, "r_{vertex}^{reco} [cm]"),
}


def histogram_name(category="", label="", unweighted=False):
    return f"ShiftDimuonVertex{category}_bdtScore{label}" + ("Unweighted" if unweighted else "")


def histogram_path(category="", label="", unweighted=False):
    return f"{DIRECTORY}/{histogram_name(category, label, unweighted)}"


# These default bookings can also be appended to a frozen ordinary TEA config.
# MC-labelled and two-dimensional diagnostics are filled by the dedicated tool.
defaultHistParams = tuple(
    (f"ShiftDimuonVertex{category}", "bdtScore", *SCORE_BINS, DIRECTORY)
    for category in CATEGORIES
)
