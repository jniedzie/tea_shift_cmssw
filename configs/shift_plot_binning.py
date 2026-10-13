# Code style: human-code.
"""Common density plots: retain fine populated regions and merge sparse tails.

Display edges always come from the input histogram. This permits older files
to be plotted without splitting a recorded bin or guessing where its entries
were. The same rules apply to every sample in a stack and every topology.
"""

from bisect import bisect_left
from dataclasses import dataclass


def linear_bin_edges(*segments):
    """Join (low, high, width) segments without repeated boundary edges."""
    edges = []
    for low, high, width in segments:
        bins = round((high - low) / width)
        if bins < 1 or abs(low + bins * width - high) > 1.0e-8:
            raise ValueError("A linear binning segment must contain whole bins")
        part = [round(low + index * width, 10) for index in range(bins + 1)]
        if edges and abs(edges[-1] - part[0]) > 1.0e-8:
            raise ValueError("Binning segments must meet at one common edge")
        edges.extend(part[1:] if edges else part)
    return tuple(edges)


@dataclass(frozen=True)
class AlignedBinWidths:
    """Requested (upper edge, width) regions; width zero keeps every source bin."""

    regions: tuple

    def __call__(self, histogram):
        source = tuple(
            histogram.GetBinLowEdge(index)
            for index in range(1, histogram.GetNbinsX() + 2)
        )

        def nearest_index(value, start=0):
            position = bisect_left(source, value, lo=start)
            if position >= len(source):
                return len(source) - 1
            if position > start and abs(source[position - 1] - value) <= abs(source[position] - value):
                return position - 1
            return position

        result = [source[0]]
        start = 0
        for upper, width in self.regions:
            end = nearest_index(upper, start)
            if end <= start:
                continue
            while start < end:
                following = start + 1 if width == 0 else max(
                    start + 1, nearest_index(source[start] + width, start + 1)
                )
                start = min(following, end)
                result.append(source[start])
        # Preserve the complete recorded axis, including any future extension.
        result.extend(source[start + 1:])
        return tuple(result)


muon_variable_binning = {
    "eta": AlignedBinWidths(((-8.0, 0.1), (-2.0, 0), (0.0, 0.1), (10.0, 0.5))),
    "constrainedPt": AlignedBinWidths(((6.0, 0), (10.0, 0.5), (20.0, 1.0), (50.0, 2.0))),
    "constrainedPz": AlignedBinWidths(((-800.0, 100.0), (-400.0, 40.0), (-200.0, 20.0),
                                      (-100.0, 8.4), (25.0, 0), (100.0, 21.0))),
}

dimuon_variable_binning = {
    "openingAngle": AlignedBinWidths(((0.2, 0), (0.5, 0.002), (1.0, 0.005), (3.15, 0.025))),
    "chi2": AlignedBinWidths(((4.0, 0), (10.0, 0.5), (20.0, 1.0), (100.0, 5.0))),
    "normalizedChi2": AlignedBinWidths(((1.5, 0), (4.0, 0.05), (10.0, 0.2), (50.0, 1.0))),
    "dca": AlignedBinWidths(((150.0, 0), (300.0, 6.0), (600.0, 15.0), (1500.0, 60.0))),
    "deltaEta": AlignedBinWidths(((2.0, 0), (4.0, 0.1), (8.0, 0.5), (20.0, 1.0))),
    # J/psi accumulates near zero; high-mass DY accumulates near pi. Retain
    # both endpoints and merge only the less populated middle of this axis.
    "deltaPhi": AlignedBinWidths(((0.5, 0), (1.0, 0.02), (2.0, 0.04),
                                 (2.5, 0.02), (3.15, 0))),
    "deltaR": AlignedBinWidths(((3.5, 0), (5.0, 0.1), (10.0, 0.5), (20.0, 1.0))),
    "eta": muon_variable_binning["eta"],
    "mass": AlignedBinWidths(((10.0, 0), (30.0, 0.2), (50.0, 1.0), (100.0, 5.0))),
    "massOS": AlignedBinWidths(((5.0, 0), (10.0, 0.1), (20.0, 0.2),
                               (30.0, 0.5), (50.0, 1.0), (100.0, 5.0))),
    "maxMuonPt": AlignedBinWidths(((6.0, 0), (10.0, 0.2), (20.0, 0.5), (50.0, 2.0))),
    "openingAngleFine": AlignedBinWidths(((0.1, 0), (0.2, 0.001), (0.5, 0.0025),
                                         (1.0, 0.005), (3.15, 0.025))),
    "vertexR": AlignedBinWidths(((100.0, 0), (150.0, 5.0), (250.0, 10.0), (450.0, 25.0))),
    "constrainedMass": AlignedBinWidths(((5.0, 0), (30.0, 0.2), (50.0, 1.0), (100.0, 5.0))),
    "probability": AlignedBinWidths(((0.3, 0.05), (0.7, 0.02), (1.01, 0))),
    "originCompatibilityNormalizedChi2": AlignedBinWidths(((10.0, 0), (25.0, 0.2),
                                                          (50.0, 0.5), (100.0, 2.0), (500.0, 10.0))),
}
dimuon_variable_binning["massSS"] = dimuon_variable_binning["massOS"]
dimuon_variable_binning["refittedMass"] = dimuon_variable_binning["mass"]
dimuon_variable_binning["constrainedOpeningAngle"] = dimuon_variable_binning["openingAngle"]

# Source histograms must be regenerated for extra angular/momentum resolution
# and extended axes. These source bins remain finer than the display bins.
dimuon_opening_angle_edges = linear_bin_edges(
    (0.0, 0.2, 0.0005), (0.2, 1.0, 0.002), (1.0, 3.15, 0.005)
)
dimuon_fine_opening_angle_edges = linear_bin_edges(
    (0.0, 0.2, 0.00025), (0.2, 1.0, 0.001), (1.0, 3.15, 0.005)
)
dimuon_origin_chi2_edges = linear_bin_edges(
    (0.0, 10.0, 0.05), (10.0, 50.0, 0.1), (50.0, 100.0, 0.5), (100.0, 500.0, 2.0)
)
