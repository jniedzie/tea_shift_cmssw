#!/usr/bin/env python3
"""Plot inclusive and topology-resolved SHIFT efficiency versus physical delay."""

import argparse
from array import array
import json
import math
from pathlib import Path
import re

import ROOT


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

MUON_CATEGORIES = [
    ("inclusive", "Inclusive", ROOT.kBlack),
    ("both_endcaps", "Both Endcaps", ROOT.kGreen + 2),
    ("near_endcap_only", "Near Endcap Only", ROOT.kViolet + 1),
    ("far_endcap_only", "Far Endcap Only", ROOT.kCyan + 2),
    ("near_endcap_barrel", "Near Endcap + Barrel", ROOT.kBlue + 1),
    ("unclassified", "Unclassified", ROOT.kGray + 2),
]
MUON_TOPOLOGY = {
    0: "near_endcap_only",
    1: "near_endcap_barrel",
    2: "both_endcaps",
    3: "far_endcap_only",
    4: "unclassified",
}
DIMUON_CATEGORIES = [
    ("inclusive", "Inclusive", ROOT.kBlack),
    ("good", "Good", ROOT.kRed + 1),
    ("both_both", "Both Endcaps + Both Endcaps", ROOT.kGreen + 2),
    ("near_both", "Both Endcaps + Near Endcap Only", ROOT.kViolet + 1),
    ("both_far", "Both Endcaps + Far Endcap Only", ROOT.kCyan + 2),
    ("near_far", "Near Endcap Only + Far Endcap Only", ROOT.kBlue + 1),
    ("other", "Other topologies", ROOT.kGray + 2),
]
DIMUON_TOPOLOGY = {
    (0, 2): "near_both",
    (0, 3): "near_far",
    (2, 2): "both_both",
    (2, 3): "both_far",
}
DELAY_PATTERN = re.compile(r"^delay_([mp])(\d+(?:p\d+)?)ns$")


def delay_from_name(name):
    match = DELAY_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"invalid delay directory name: {name}")
    value = float(match.group(2).replace("p", "."))
    return -value if match.group(1) == "m" else value


def find_delay_files(scan_dir):
    points = []
    for directory in scan_dir.glob("delay_*ns"):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("events_shiftDelayScan_part*.root"))
        if files:
            points.append((delay_from_name(directory.name), files))
    if not points:
        raise ValueError(f"no compact delay-scan ROOT files found under {scan_dir}")
    return sorted(points)


def as_list(tree, branch):
    return list(getattr(tree, branch))


def read_provenance(input_file, path, expected_delay):
    metadata = input_file.Get("tag")
    if not metadata or not metadata.InheritsFrom("TObjString"):
        raise RuntimeError(f"delay provenance is missing in {path}")
    try:
        provenance = json.loads(str(metadata.GetString().Data()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid delay provenance in {path}") from error
    if provenance.get("format") != "shift-reco-delay-scan-v1":
        raise RuntimeError(f"unsupported delay-scan format in {path}")
    if provenance.get("pileup_mode") != "none":
        raise RuntimeError(f"{path} is not a no-pileup same-SimHit control")
    try:
        stored_delay = float(provenance["delay_ns"])
        sources = provenance["source_step1"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"incomplete delay provenance in {path}") from error
    if not math.isclose(stored_delay, expected_delay, abs_tol=1.0e-9):
        raise RuntimeError(
            f"directory delay {expected_delay} ns disagrees with {stored_delay} ns in {path}"
        )
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources or not all(
        isinstance(source, str) for source in sources
    ):
        raise RuntimeError(f"invalid Step-1 source list in {path}")
    return sources


def count_file(path, counts, expected_delay):
    input_file = ROOT.TFile.Open(str(path))
    if not input_file or input_file.IsZombie():
        raise RuntimeError(f"cannot open {path}")
    sources = read_provenance(input_file, path, expected_delay)
    tree = input_file.Get("Events")
    if not tree:
        raise RuntimeError(f"Events tree is missing in {path}")

    for event in tree:
        pdg_ids = as_list(event, "GenPart_pdgId")
        statuses = as_list(event, "GenPart_status")
        mothers = as_list(event, "GenPart_genPartIdxMother")
        daughters = {}
        for index, (pdg_id, status, mother) in enumerate(zip(pdg_ids, statuses, mothers)):
            if abs(pdg_id) != 13 or status != 1 or mother < 0 or mother >= len(pdg_ids):
                continue
            if abs(pdg_ids[mother]) != 443:
                continue
            charge_key = "minus" if pdg_id == 13 else "plus"
            daughters.setdefault(mother, {"minus": [], "plus": []})[charge_key].append(index)

        reco_gen = as_list(event, "ShiftMuon_genPartIdx")
        reco_topology = as_list(event, "ShiftMuon_topology")
        reco_by_gen = {}
        for reco_index, gen_index in enumerate(reco_gen):
            if gen_index >= 0:
                reco_by_gen.setdefault(gen_index, []).append(reco_index)

        first_indices = as_list(event, "ShiftDimuonVertex_muonIdx1")
        second_indices = as_list(event, "ShiftDimuonVertex_muonIdx2")
        topology_min = as_list(event, "ShiftDimuonVertex_topologyMin")
        topology_max = as_list(event, "ShiftDimuonVertex_topologyMax")
        is_os = as_list(event, "ShiftDimuonVertex_isOS")
        dca_valid = as_list(event, "ShiftDimuonVertex_dcaValid")
        dca = as_list(event, "ShiftDimuonVertex_dca")

        for candidate in daughters.values():
            if len(candidate["minus"]) != 1 or len(candidate["plus"]) != 1:
                continue
            truth_indices = {candidate["minus"][0], candidate["plus"][0]}
            counts["muon"]["denominator"] += 2
            for gen_index in truth_indices:
                matched_reco = reco_by_gen.get(gen_index, [])
                if matched_reco:
                    counts["muon"]["inclusive"] += 1
                matched_categories = {
                    MUON_TOPOLOGY.get(int(reco_topology[index]), "unclassified")
                    for index in matched_reco
                }
                for category in matched_categories:
                    counts["muon"][category] += 1

            counts["dimuon"]["denominator"] += 1
            matched_categories = set()
            matched = False
            for index, (first, second) in enumerate(zip(first_indices, second_indices)):
                if first < 0 or second < 0 or first >= len(reco_gen) or second >= len(reco_gen):
                    continue
                if {int(reco_gen[first]), int(reco_gen[second])} != truth_indices:
                    continue
                matched = True
                matched_categories.add(
                    DIMUON_TOPOLOGY.get(
                        (int(topology_min[index]), int(topology_max[index])), "other"
                    )
                )
                if is_os[index] == 1 and dca_valid[index] == 1 and dca[index] >= 50.0:
                    matched_categories.add("good")
            if matched:
                counts["dimuon"]["inclusive"] += 1
            for category in matched_categories:
                counts["dimuon"][category] += 1
    input_file.Close()
    return sources


def empty_counts():
    return {
        "muon": {"denominator": 0, **{name: 0 for name, _, _ in MUON_CATEGORIES}},
        "dimuon": {"denominator": 0, **{name: 0 for name, _, _ in DIMUON_CATEGORIES}},
    }


def graph_for(points, object_name, category):
    graph = ROOT.TGraphAsymmErrors(len(points))
    for index, point in enumerate(points):
        denominator = point[object_name]["denominator"]
        numerator = point[object_name][category]
        efficiency = numerator / denominator if denominator else 0.0
        lower = ROOT.TEfficiency.ClopperPearson(denominator, numerator, 0.682689, False) if denominator else 0.0
        upper = ROOT.TEfficiency.ClopperPearson(denominator, numerator, 0.682689, True) if denominator else 0.0
        graph.SetPoint(index, point["delay_ns"], efficiency)
        graph.SetPointError(index, 0.0, 0.0, efficiency - lower, upper - efficiency)
    return graph


def make_plot(points, object_name, categories, output_path):
    canvas = ROOT.TCanvas(f"canvas_{object_name}", "", 900, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetBottomMargin(0.13)
    delays = [point["delay_ns"] for point in points]
    span = max(delays) - min(delays)
    margin = 0.06 * span if span else 1.0
    frame = canvas.DrawFrame(min(delays) - margin, 0.0, max(delays) + margin, 1.05)
    frame.SetTitle(f";SHIFT delay relative to central collision (ns);{object_name.capitalize()} efficiency")
    frame.GetXaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetLabelSize(0.042)
    legend = ROOT.TLegend(0.48, 0.16, 0.88, 0.48 if object_name == "muon" else 0.53)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    graphs = []
    for category, label, color in categories:
        graph = graph_for(points, object_name, category)
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetLineWidth(2)
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.9)
        graph.Draw("LP SAME")
        legend.AddEntry(graph, label, "lp")
        graphs.append(graph)
    legend.Draw()
    canvas.SaveAs(str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    scan_dir = args.scan_dir.resolve()
    output_dir = (args.output_dir or scan_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = []
    for delay, files in find_delay_files(scan_dir):
        counts = empty_counts()
        sources = set()
        for path in files:
            file_sources = count_file(path, counts, delay)
            duplicated = sources.intersection(file_sources)
            if duplicated:
                raise RuntimeError(
                    f"duplicate source Step-1 file at delay {delay} ns: {sorted(duplicated)[0]}"
                )
            sources.update(file_sources)
        points.append({"delay_ns": delay, "files": len(files), "sources": sources, **counts})
    reference_sources = points[0]["sources"]
    for point in points[1:]:
        if point["sources"] != reference_sources:
            raise RuntimeError(
                f"delay {point['delay_ns']} ns does not contain the same Step-1 inputs as "
                f"delay {points[0]['delay_ns']} ns"
            )
    for point in points:
        point["sources"] = sorted(point["sources"])
    with (output_dir / "shift_delay_efficiencies.json").open("w", encoding="utf-8") as output_file:
        json.dump({"format": "shift-delay-efficiencies-v1", "points": points}, output_file,
                  indent=2, sort_keys=True)
        output_file.write("\n")
    make_plot(points, "muon", MUON_CATEGORIES, output_dir / "muon_efficiency_vs_delay.pdf")
    make_plot(points, "dimuon", DIMUON_CATEGORIES, output_dir / "dimuon_efficiency_vs_delay.pdf")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
