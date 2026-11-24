# Code style: AI-code. AI-generated J/ψ control-region configuration.
"""Keep only dimuon candidates inside the J/ψ control region."""

# The shared 2.6–3.6 GeV mass window has inclusive endpoints.
from shift_skimmer_config import (
    branchesToAdd,
    branchesToKeep,
    branchesToRemove,
    eventsTreeNames,
    floatFields,
    intFields,
    mcWeightBranches,
    skimCollection,
    skimMassWindow,
)

nEvents = -1
inputFilePath = ""
treeOutputFilePath = ""

skimCandidateCollection = "JpsiControlDimuons"
extraEventCollections = {
    "JpsiControlDimuons": {
        "inputCollections": (skimCollection,),
        "isOS": 1,
        "topologyMin": 2,  # Both muons cross both endcaps.
        "topologyMax": 2,
        "vz": (10_000.0, float("inf")),  # Reconstructed vertex z ≥ 100 m, stored in cm.
        # Trial only. Enable these together on inputs with stored PV DCA.
        # "pvDcaValid": 1,
        # "pvDca": (0.0, 50.0),  # cm
        # Trial only, with pT in GeV.
        # "pt": (0.0, 2.8),
    },
}
