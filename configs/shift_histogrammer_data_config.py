"""Reco-only SHIFT histograms: supply --input_path and --output_hists_path."""
import runpy
from pathlib import Path
# Load shared histogram definitions without locating any simulation sample.
globals().update(runpy.run_path(str(Path(__file__).resolve().with_name("shift_histogrammer_config.py")),
                               init_globals={"recoDataMode": True}))
enableTruthDiagnostics = False
weightsBranchName = ""
extraEventCollections = {name: spec for name, spec in extraEventCollections.items() if not name.startswith("Gen")}
defaultHistParams = tuple(h for h in defaultHistParams if not h[0].startswith("Gen")
    and not h[1].startswith(("nGen", "sim", "gen", "sameGen", "chargeMatchesGen", "nAddedDTTruth"))
    and "sim" not in h[1].lower() and "signalMuon" not in h[1])
# Book only quantities observable in data; generator-dependent plots are absent.
histParams = tuple(h for h in histParams if h[0] in ("ShiftDimuonVertex", "TargetDiagnostics", "DetectorDiagnostics")
    and "Truth" not in h[1])
histParams2D = ()
irregularHistParams = ()
