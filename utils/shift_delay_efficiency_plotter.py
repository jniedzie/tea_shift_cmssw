#!/usr/bin/env python3
"""Plot inclusive and topology-resolved SHIFT efficiency versus physical delay."""

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
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
DIMUON_PLOT_CATEGORIES = [
    category for category in DIMUON_CATEGORIES if category[0] != "good"
]
DIMUON_TOPOLOGY = {
    (0, 2): "near_both",
    (0, 3): "near_far",
    (2, 2): "both_both",
    (2, 3): "both_far",
}
DELAY_PATTERN = re.compile(r"^delay_([mp])(\d+(?:p\d+)?)ns$")
COUNT_BRANCHES = (
    "GenPart_pdgId",
    "GenPart_status",
    "GenPart_genPartIdxMother",
    "ShiftMuon_genPartIdx",
    "ShiftMuon_topology",
    "ShiftDimuonVertex_muonIdx1",
    "ShiftDimuonVertex_muonIdx2",
    "ShiftDimuonVertex_topologyMin",
    "ShiftDimuonVertex_topologyMax",
    "ShiftDimuonVertex_isOS",
    "ShiftDimuonVertex_dcaValid",
    "ShiftDimuonVertex_dca",
)


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
        merged = directory / "events_shiftDelayScan_merged.root"
        if merged.is_file():
            if files:
                raise ValueError(
                    f"both per-file and merged delay outputs found in {directory}"
                )
            files = [merged]
        if files:
            points.append((delay_from_name(directory.name), files))
    if not points:
        raise ValueError(f"no compact delay-scan ROOT files found under {scan_dir}")
    return sorted(points)


def select_common_file_groups(delay_files):
    files_by_delay = [
        {path.name: path for path in files}
        for _, files in delay_files
    ]
    common_names = set.intersection(*(set(files) for files in files_by_delay))
    if not common_names:
        raise RuntimeError("no ROOT file group is present at every delay")

    all_names = set.union(*(set(files) for files in files_by_delay))
    excluded_names = sorted(all_names - common_names)
    if excluded_names:
        print(
            f"using {len(common_names)} file groups present at every delay; "
            f"excluding {len(excluded_names)} incomplete groups: "
            + ", ".join(excluded_names),
            flush=True,
        )
    return [
        (delay, [files[name] for name in sorted(common_names)])
        for (delay, _), files in zip(delay_files, files_by_delay)
    ]


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
    missing_branches = [branch for branch in COUNT_BRANCHES if not tree.GetBranch(branch)]
    if missing_branches:
        raise RuntimeError(f"required branch {missing_branches[0]} is missing in {path}")
    tree.SetBranchStatus("*", False)
    for branch in COUNT_BRANCHES:
        tree.SetBranchStatus(branch, True)

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


def graph_for(points, object_name, category, show_errors=True):
    graph = ROOT.TGraphAsymmErrors(len(points))
    for index, point in enumerate(points):
        denominator = point[object_name]["denominator"]
        numerator = point[object_name][category]
        efficiency = numerator / denominator if denominator else 0.0
        lower = (
            ROOT.TEfficiency.ClopperPearson(denominator, numerator, 0.682689, False)
            if denominator and show_errors else efficiency
        )
        upper = (
            ROOT.TEfficiency.ClopperPearson(denominator, numerator, 0.682689, True)
            if denominator and show_errors else efficiency
        )
        graph.SetPoint(index, point["delay_ns"], efficiency)
        graph.SetPointError(index, 0.0, 0.0, efficiency - lower, upper - efficiency)
    return graph


def rebin_points(points, object_name, bin_width, minimum, maximum):
    number_of_bins = math.ceil((maximum - minimum) / bin_width)
    bins = [[] for _ in range(number_of_bins)]
    for point in points:
        delay = point["delay_ns"]
        if delay < minimum or delay > maximum:
            continue
        index = min(int((delay - minimum) // bin_width), number_of_bins - 1)
        bins[index].append(point)

    result = []
    for group in bins:
        if not group:
            continue
        combined = empty_counts()[object_name]
        for point in group:
            for name, value in point[object_name].items():
                combined[name] += value
        result.append(
            {
                "delay_ns": sum(point["delay_ns"] for point in group) / len(group),
                object_name: combined,
            }
        )
    return result


def make_plot(
    points, object_name, categories, output_path, *, x_minimum, x_maximum,
    show_errors=True, average_width=None,
):
    canvas = ROOT.TCanvas(f"canvas_{object_name}", "", 1400, 700)
    canvas.SetLeftMargin(0.13)
    canvas.SetRightMargin(0.44)
    canvas.SetBottomMargin(0.13)
    maximum = max(
        point[object_name][category] / point[object_name]["denominator"]
        if point[object_name]["denominator"] else 0.0
        for point in points
        for category, _, _ in categories
    )
    y_max = min(1.05, max(0.05, 1.25 * maximum))
    frame = canvas.DrawFrame(x_minimum, 0.0, x_maximum, y_max)
    y_title = f"{object_name.capitalize()} efficiency"
    if average_width is not None:
        y_title += f" ({average_width:g} ns average)"
    frame.SetTitle(f";SHIFT delay relative to central collision (ns);{y_title}")
    frame.GetXaxis().SetTitleSize(0.050)
    frame.GetYaxis().SetTitleSize(0.050)
    frame.GetXaxis().SetLabelSize(0.042)
    frame.GetYaxis().SetLabelSize(0.042)
    legend = ROOT.TLegend(0.58, 0.50, 0.99, 0.88)
    legend.SetTextSize(0.028)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    graphs = []
    for category, label, color in categories:
        graph = graph_for(points, object_name, category, show_errors=show_errors)
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


def validate_cached_counts(data, path):
    if not isinstance(data, dict) or data.get("format") not in {
        "shift-delay-efficiencies-v1",
        "shift-delay-efficiencies-compact-v1",
    }:
        raise RuntimeError(f"unsupported efficiency-count cache in {path}")
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"efficiency-count cache has no points: {path}")

    delays = []
    required = {
        "muon": [name for name, _, _ in MUON_CATEGORIES],
        "dimuon": [name for name, _, _ in DIMUON_CATEGORIES],
    }
    for point in points:
        try:
            delay = float(point["delay_ns"])
            files = point["files"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid point in efficiency-count cache: {path}") from error
        if not math.isfinite(delay) or not isinstance(files, int) or files < 1:
            raise RuntimeError(f"invalid delay or file count in efficiency-count cache: {path}")
        delays.append(delay)
        for object_name, categories in required.items():
            counts = point.get(object_name)
            if not isinstance(counts, dict):
                raise RuntimeError(f"missing {object_name} counts in {path}")
            denominator = counts.get("denominator")
            if not isinstance(denominator, int) or denominator < 0:
                raise RuntimeError(f"invalid {object_name} denominator in {path}")
            for category in categories:
                numerator = counts.get(category)
                if (
                    not isinstance(numerator, int)
                    or numerator < 0
                    or numerator > denominator
                ):
                    raise RuntimeError(
                        f"invalid {object_name}/{category} numerator in {path}"
                    )
    if delays != sorted(set(delays)):
        raise RuntimeError(f"cached delays are duplicated or out of order: {path}")

    if data["format"] == "shift-delay-efficiencies-v1":
        reference_sources = points[0].get("sources")
        if not isinstance(reference_sources, list) or not reference_sources:
            raise RuntimeError(f"cached Step-1 sources are missing in {path}")
        if any(point.get("sources") != reference_sources for point in points[1:]):
            raise RuntimeError(f"cached delay points do not use identical inputs: {path}")
    else:
        sources = data.get("source_step1")
        if not isinstance(sources, list) or not sources:
            raise RuntimeError(f"cached Step-1 sources are missing in {path}")
    return points


def load_cached_counts(path):
    try:
        with path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read efficiency-count cache {path}") from error
    return validate_cached_counts(data, path)


def count_delay(delay, files):
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
    return {
        "delay_ns": delay,
        "files": len(files),
        "sources": sorted(sources),
        **counts,
    }


def count_scan(scan_dir, workers):
    delay_files = select_common_file_groups(find_delay_files(scan_dir))
    if workers == 1:
        points = []
        for delay_index, (delay, files) in enumerate(delay_files, start=1):
            print(
                f"counting delay {delay:g} ns "
                f"({delay_index}/{len(delay_files)}; {len(files)} files)",
                flush=True,
            )
            points.append(count_delay(delay, files))
    else:
        points = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(count_delay, delay, files): delay
                for delay, files in delay_files
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                delay = futures[future]
                try:
                    point = future.result()
                except Exception as error:
                    raise RuntimeError(f"failed counting delay {delay:g} ns") from error
                points.append(point)
                print(
                    f"counted delay {delay:g} ns ({completed}/{len(delay_files)})",
                    flush=True,
                )
        points.sort(key=lambda point: point["delay_ns"])

    reference_sources = points[0]["sources"]
    for point in points[1:]:
        if point["sources"] != reference_sources:
            raise RuntimeError(
                f"delay {point['delay_ns']} ns does not contain the same Step-1 inputs as "
                f"delay {points[0]['delay_ns']} ns"
            )
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--counts-json", type=Path,
        help="redraw directly from a previously written counts JSON without reading ROOT files",
    )
    parser.add_argument(
        "--recount", action="store_true",
        help="ignore an existing counts JSON and reread all ROOT files",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="parallel counting processes used when reading ROOT files (default: 4)",
    )
    parser.add_argument("--delay-min", type=float, default=-200.0)
    parser.add_argument("--delay-max", type=float, default=200.0)
    parser.add_argument(
        "--dimuon-bin-width", type=float, default=10.0,
        help="delay-bin width used to average the dimuon curves (default: 10 ns)",
    )
    args = parser.parse_args()
    if args.counts_json and args.recount:
        parser.error("--counts-json and --recount cannot be used together")
    if not args.counts_json and not args.scan_dir:
        parser.error("scan_dir is required unless --counts-json is used")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.delay_min >= args.delay_max:
        parser.error("--delay-min must be smaller than --delay-max")
    if args.dimuon_bin_width <= 0:
        parser.error("--dimuon-bin-width must be positive")

    if args.counts_json:
        cache_path = args.counts_json.resolve()
        output_dir = (args.output_dir or cache_path.parent).resolve()
    else:
        scan_dir = args.scan_dir.resolve()
        output_dir = (args.output_dir or scan_dir / "plots").resolve()
        cache_path = output_dir / "shift_delay_efficiencies.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.counts_json and not cache_path.is_file():
        parser.error(f"counts JSON does not exist: {cache_path}")

    if args.counts_json or (cache_path.is_file() and not args.recount):
        points = load_cached_counts(cache_path)
        message = f"using cached counts from {cache_path}"
        if not args.counts_json:
            message += "; pass --recount to reread ROOT files"
        print(message, flush=True)
    else:
        points = count_scan(scan_dir, args.workers)
        with cache_path.open("w", encoding="utf-8") as output_file:
            json.dump({"format": "shift-delay-efficiencies-v1", "points": points}, output_file,
                      indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"cached counts in {cache_path}", flush=True)
    visible_points = [
        point for point in points
        if args.delay_min <= point["delay_ns"] <= args.delay_max
    ]
    if not visible_points:
        parser.error("no cached or counted delay point is inside the requested plot range")
    dimuon_points = rebin_points(
        visible_points, "dimuon", args.dimuon_bin_width,
        args.delay_min, args.delay_max,
    )
    make_plot(
        visible_points, "muon", MUON_CATEGORIES,
        output_dir / "muon_efficiency_vs_delay.pdf",
        x_minimum=args.delay_min, x_maximum=args.delay_max,
    )
    make_plot(
        dimuon_points, "dimuon", DIMUON_PLOT_CATEGORIES,
        output_dir / "dimuon_efficiency_vs_delay.pdf",
        x_minimum=args.delay_min, x_maximum=args.delay_max,
        show_errors=True, average_width=args.dimuon_bin_width,
    )
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
