# Code style: AI-code. AI-generated configuration for the blinded SHIFT skim.
"""Use TEA's event writer to save the accepted J/ψ candidates."""

from shift_jpsi_dimuon import jpsi_dimuon

nEvents = -1
eventsTreeNames = ("Events",)
inputFilePath = ""
treeOutputFilePath = ""

skimCollection = jpsi_dimuon["collection"]
skimMassWindow = tuple(str(limit) for limit in jpsi_dimuon["mass"])

floatFields = ("mass", "pt", "pz", "eta", "phi", "vx", "vy", "vz", "chi2", "normalizedChi2", "dca")
intFields = ("topologyMin", "topologyMax", "isOS", "dcaValid")
mcWeightBranches = (
    "shiftSamplingGenWeight", "shiftResponseGenWeight", "shiftRecoveryGenWeight",
    "shiftProposalGenWeight", "genWeight", "shiftSamplingProbability",
    "shiftSamplingWeight", "shiftOriginalGenWeight",
)
# Keep the existing candidate branches and compact their rows before filling.
dimuonBranches = (
    "run", "luminosityBlock", "event", f"n{skimCollection}",
    *(f"{skimCollection}_{field}" for field in floatFields + intFields),
)
branchesToKeep = dimuonBranches + mcWeightBranches
branchesToRemove = ()
branchesToAdd = ()
