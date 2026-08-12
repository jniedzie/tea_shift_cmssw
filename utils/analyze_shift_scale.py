#!/usr/bin/env python3
"""Print robust SHIFT momentum-scale summaries by quality and fit strategy."""

import argparse
import math
from collections import defaultdict

import ROOT


QUALITY_NAMES = {
    0: "cosmic",
    1: "DSA",
    2: "traversing",
    3: "double_traversing",
}


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summary(values):
    q16 = quantile(values, 0.16)
    median = quantile(values, 0.50)
    q84 = quantile(values, 0.84)
    outliers = sum(abs(value) > 0.5 for value in values)
    return len(values), median, 0.5 * (q84 - q16), outliers / len(values) if values else float("nan")


def relative(reco, gen):
    return (reco - gen) / gen if abs(gen) > 1.0e-12 else None


def four_vector(event, prefix, index, constrained=False):
    qualifier = "constrained" if constrained else ""
    field = lambda name: getattr(event, f"{prefix}_{qualifier}{name}" if qualifier else f"{prefix}_{name}")[index]
    vector = ROOT.Math.PtEtaPhiMVector(field("Pt" if constrained else "pt"),
                                      field("Eta" if constrained else "eta"),
                                      field("Phi" if constrained else "phi"),
                                      field("Mass" if constrained else "mass"))
    return vector


def gen_vector(event, index):
    return ROOT.Math.PtEtaPhiMVector(event.GenPart_pt[index], event.GenPart_eta[index],
                                    event.GenPart_phi[index], event.GenPart_mass[index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NanoAOD ROOT file")
    args = parser.parse_args()

    root_file = ROOT.TFile.Open(args.input)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot open {args.input}")
    events = root_file.Get("Events")
    if not events:
        raise RuntimeError("missing Events tree")

    muon_residuals = defaultdict(list)
    dimuon_residuals = defaultdict(list)
    muon_counts = defaultdict(int)
    dimuon_counts = defaultdict(int)
    diagnostic_residuals = defaultdict(list)

    for event in events:
        for index in range(int(event.nShiftMuon)):
            quality = int(event.ShiftMuon_quality[index])
            gen_index = int(event.ShiftMuon_genPartIdx[index])
            if gen_index < 0 or gen_index >= int(event.nGenPart):
                continue
            muon_counts[(quality, "unconstrained")] += 1
            unconstrained_values = {}
            gen_p = math.hypot(event.GenPart_pt[gen_index], event.GenPart_pz[gen_index])
            for quantity, reco, gen in (
                ("pt", event.ShiftMuon_pt[index], event.GenPart_pt[gen_index]),
                ("pz", event.ShiftMuon_pz[index], event.GenPart_pz[gen_index]),
                ("p", event.ShiftMuon_p[index], gen_p),
                ("eta", event.ShiftMuon_eta[index], event.GenPart_eta[gen_index]),
            ):
                value = relative(reco, gen)
                if value is not None and math.isfinite(value):
                    muon_residuals[(quality, "unconstrained", quantity)].append(value)
                    unconstrained_values[quantity] = value
            p_residual = unconstrained_values.get("p")
            if p_residual is not None:
                momentum_bin = "p<30" if gen_p < 30.0 else ("30<=p<100" if gen_p < 100.0 else "p>=100")
                used_precision = "precision" if bool(event.ShiftMuon_directionalRefitUsedPrecisionHits[index]) else "all_hits"
                stations = int(event.ShiftMuon_nPrecisionRefitStations[index])
                diagnostic_residuals[(quality, "momentum", momentum_bin)].append(p_residual)
                diagnostic_residuals[(quality, "hit_fit", used_precision)].append(p_residual)
                diagnostic_residuals[(quality, "stations", str(stations))].append(p_residual)

            if not bool(event.ShiftMuon_constrainedValid[index]):
                continue
            muon_counts[(quality, "unconstrained_paired")] += 1
            muon_counts[(quality, "constrained")] += 1
            for quantity, value in unconstrained_values.items():
                muon_residuals[(quality, "unconstrained_paired", quantity)].append(value)
            for quantity, reco, gen in (
                ("pt", event.ShiftMuon_constrainedPt[index], event.GenPart_pt[gen_index]),
                ("pz", event.ShiftMuon_constrainedPz[index], event.GenPart_pz[gen_index]),
                ("p", event.ShiftMuon_constrainedP[index],
                 math.hypot(event.GenPart_pt[gen_index], event.GenPart_pz[gen_index])),
                ("eta", event.ShiftMuon_constrainedEta[index], event.GenPart_eta[gen_index]),
            ):
                value = relative(reco, gen)
                if value is not None and math.isfinite(value):
                    muon_residuals[(quality, "constrained", quantity)].append(value)

        for index in range(int(event.nShiftDimuonVertex)):
            first = int(event.ShiftDimuonVertex_muonIdx1[index])
            second = int(event.ShiftDimuonVertex_muonIdx2[index])
            first_gen = int(event.ShiftMuon_genPartIdx[first])
            second_gen = int(event.ShiftMuon_genPartIdx[second])
            if first_gen < 0 or second_gen < 0 or first_gen == second_gen:
                continue
            qualities = tuple(sorted((int(event.ShiftMuon_quality[first]), int(event.ShiftMuon_quality[second]))))
            gen_pair = gen_vector(event, first_gen) + gen_vector(event, second_gen)
            unconstrained_pair = four_vector(event, "ShiftMuon", first) + four_vector(event, "ShiftMuon", second)
            dimuon_counts[(qualities, "unconstrained")] += 1
            unconstrained_values = {}
            for quantity, reco, gen in (
                ("pt", unconstrained_pair.Pt(), gen_pair.Pt()),
                ("pz", unconstrained_pair.Pz(), gen_pair.Pz()),
                ("mass", unconstrained_pair.M(), gen_pair.M()),
            ):
                value = relative(reco, gen)
                if value is not None and math.isfinite(value):
                    dimuon_residuals[(qualities, "unconstrained", quantity)].append(value)
                    unconstrained_values[quantity] = value

            if not (bool(event.ShiftMuon_constrainedValid[first]) and
                    bool(event.ShiftMuon_constrainedValid[second])):
                continue
            dimuon_counts[(qualities, "unconstrained_paired")] += 1
            for quantity, value in unconstrained_values.items():
                dimuon_residuals[(qualities, "unconstrained_paired", quantity)].append(value)
            constrained_pair = (four_vector(event, "ShiftMuon", first, True) +
                                four_vector(event, "ShiftMuon", second, True))
            dimuon_counts[(qualities, "constrained")] += 1
            for quantity, reco, gen in (
                ("pt", constrained_pair.Pt(), gen_pair.Pt()),
                ("pz", constrained_pair.Pz(), gen_pair.Pz()),
                ("mass", constrained_pair.M(), gen_pair.M()),
            ):
                value = relative(reco, gen)
                if value is not None and math.isfinite(value):
                    dimuon_residuals[(qualities, "constrained", quantity)].append(value)

    print(f"events={events.GetEntries()}")
    print("\nMuon residuals: n median half68 outlier(|r|>0.5)")
    for quality in QUALITY_NAMES:
        for strategy in ("unconstrained", "unconstrained_paired", "constrained"):
            count = muon_counts[(quality, strategy)]
            print(f"{QUALITY_NAMES[quality]:19s} {strategy:13s} matched={count}")
            for quantity in ("pt", "pz", "p", "eta"):
                n, median, half68, outlier = summary(muon_residuals[(quality, strategy, quantity)])
                print(f"  {quantity:4s} {n:5d} {median:+.5f} {half68:.5f} {outlier:.3f}")

    print("\nUnconstrained total-p diagnostics (groups with at least 20 matched muons)")
    for (quality, diagnostic, category), values in sorted(diagnostic_residuals.items()):
        if len(values) < 20:
            continue
        n, median, half68, outlier = summary(values)
        print(f"{QUALITY_NAMES[quality]:19s} {diagnostic:9s} {category:10s} "
              f"{n:5d} {median:+.5f} {half68:.5f} {outlier:.3f}")

    print("\nDimuon residuals from the same two muons: n median half68 outlier(|r|>0.5)")
    for qualities in sorted({key[0] for key in dimuon_counts}):
        label = f"{QUALITY_NAMES[qualities[0]]}+{QUALITY_NAMES[qualities[1]]}"
        for strategy in ("unconstrained", "unconstrained_paired", "constrained"):
            count = dimuon_counts[(qualities, strategy)]
            if count == 0:
                continue
            print(f"{label:39s} {strategy:13s} matched={count}")
            for quantity in ("pt", "pz", "mass"):
                n, median, half68, outlier = summary(dimuon_residuals[(qualities, strategy, quantity)])
                print(f"  {quantity:4s} {n:5d} {median:+.5f} {half68:.5f} {outlier:.3f}")


if __name__ == "__main__":
    main()
