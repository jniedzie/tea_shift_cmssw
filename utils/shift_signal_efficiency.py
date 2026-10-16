#!/usr/bin/env python3
# Code style: AI-code. AI-generated signal-efficiency and MC-statistics diagnostic.
"""Measure ready signal efficiencies and plan statistics for ordinary z plots.

This diagnostic does not launch production or change plotting selections. Run
from any directory with the analysis Python dependencies available. Inputs use
the ordinary campaign resolver and do not require production validation files.
An optional compact production-credit inventory can account for independently
generated events that have not yet reached the published histograms.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta


ANALYSIS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ANALYSIS / "configs"), str(ANALYSIS / "tea/pylibs/logger")]
MASSES = [2, 5, 10, 15, 40, 70]
FLIGHTS = [None, 5, 30, 100, 1000]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(selected, total, confidence=0.682689492):
    """Exact central binomial interval; zero observations are not zero acceptance."""
    tail = (1.0 - confidence) / 2.0
    low = float(beta.ppf(tail, selected, total - selected + 1)) if selected else 0.0
    high = float(beta.ppf(1.0 - tail, selected + 1, total - selected)) if selected < total else 1.0
    return [low, high]


def plot_settings():
    """Read the existing ordinary z plot without importing every SM plot input."""
    source = ANALYSIS / "configs/shift_plotter_config.py"
    module = ast.parse(source.read_text())
    matches = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "Histogram" or len(node.args) < 9:
            continue
        name = ast.unparse(node.args[0])
        if "dimuon/ShiftDimuonVertex" in name and "_vz" in name:
            matches.append(dict(rebin=ast.literal_eval(node.args[5]),
                                minimum_cm=ast.literal_eval(node.args[6]),
                                maximum_cm=ast.literal_eval(node.args[7])))
    if len(matches) != 1:
        raise ValueError("Expected one ordinary dimuon vertex-z plot definition")
    return dict(matches[0], config_path=str(source), config_sha256=digest(source))


def open_histograms(path, native):
    import ROOT
    filename = str(path)
    if native:
        filename = filename.replace("/eos/home-j/jniedzie/", "/eos/user/j/jniedzie/")
        filename = "root://eosuser.cern.ch/" + filename
    root_file = ROOT.TFile.Open(filename)
    if not root_file or root_file.IsZombie() or root_file.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError("Unreadable or recovered histogram input: " + filename)
    return root_file


def unit_count_histogram(root_file, name):
    hist = root_file.Get(name)
    if not hist:
        raise ValueError("Missing histogram: " + name)
    for index in range(hist.GetNbinsX() + 2):
        count = hist.GetBinContent(index)
        variance = hist.GetBinError(index) ** 2
        if count < 0 or not math.isclose(count, round(count), abs_tol=1.e-7):
            raise ValueError("This diagnostic requires unit positive GEN weights: " + name)
        if not math.isclose(count, variance, rel_tol=1.e-6, abs_tol=1.e-6):
            raise ValueError("Stored errors are inconsistent with unit weights: " + name)
    return hist


def count_events(root_file, category):
    name = "event/Event_nShiftDimuonVertex" + category
    hist = unit_count_histogram(root_file, name)
    if hist.GetBinContent(0) != 0 or hist.GetXaxis().GetBinLowEdge(1) != 0:
        raise ValueError("Unexpected dimuon multiplicity histogram axis")
    total = int(round(hist.Integral(0, hist.GetNbinsX() + 1)))
    selected = total - int(round(hist.GetBinContent(1)))
    if total <= 0:
        raise ValueError("Empty generated-event denominator")
    return dict(generated_events=total, selected_events=selected,
                efficiency=selected / total,
                efficiency_interval_68=interval(selected, total),
                efficiency_interval_95=interval(selected, total, 0.95),
                multiple_dimuon_events=int(round(hist.Integral(3, hist.GetNbinsX() + 1))))


def rounded_events(value):
    return int(math.ceil(value / 20.0) * 20)


def occupancy_plan(root_file, counts, settings, relative_error, core_fraction):
    """Plan independent raw-MC entries, before shape normalization to the SM.

    The most populated bins carrying 90% of the observed entries define the
    planning core. This does not change the actual plots or discard their tails.
    The nominal estimate uses the smallest core bin. An 80% one-sided
    binomial lower bound supplies a conservative, per-bin planning estimate.
    This is not a simultaneous confidence guarantee or a fitted shape model.
    """
    hist = unit_count_histogram(root_file, "dimuon/ShiftDimuonVertexBoth-Both_vz")
    # A histogram does not retain correlations between several pairs in one
    # event. Refuse to pretend they are independent if that situation occurs.
    if counts["multiple_dimuon_events"]:
        raise ValueError("Multiple Both-Both pairs require event-level covariance before planning")
    rebinned = hist.Clone("efficiency_vertex_z")
    rebinned.SetDirectory(0)
    rebinned.Rebin(settings["rebin"])
    edges, values = [], []
    for index in range(1, rebinned.GetNbinsX() + 1):
        low = rebinned.GetXaxis().GetBinLowEdge(index)
        high = rebinned.GetXaxis().GetBinUpEdge(index)
        if high <= settings["minimum_cm"] or low >= settings["maximum_cm"]:
            continue
        if not edges:
            edges.append(low)
        edges.append(high)
        values.append(int(round(rebinned.GetBinContent(index))))
    return plan_bin_counts(values, edges, counts, relative_error, core_fraction)


def plan_bin_counts(values, edges, counts, relative_error, core_fraction):
    total = counts["generated_events"]
    visible = sum(values)
    result = dict(histogram="dimuon/ShiftDimuonVertexBoth-Both_vz",
                  edges_cm=edges, bin_events=values, visible_selected_events=visible,
                  target_effective_entries_per_bin=math.ceil(relative_error ** -2),
                  minimum_visible_selected_events=1000,
                  core_population_fraction=core_fraction, core_bin_indices=[],
                  core_definition="Highest-population bins covering the requested fraction; all ties included",
                  unobserved_visible_bin_indices=[index for index, value in enumerate(values) if not value],
                  empty_core_bin_indices=[], nominal_target_total_events=None,
                  conservative_target_total_events=None,
                  pilot_total_events=None, status="no_visible_acceptance_measurement")
    if visible == 0:
        result["pilot_total_events"] = max(10000, rounded_events(total * 10))
        result["visible_efficiency_upper_95"] = interval(0, total, 0.95)[1]
        return result
    cumulative = 0
    for value in sorted(values, reverse=True):
        cumulative += value
        if cumulative >= core_fraction * visible:
            minimum = value
            break
    core = [index for index, value in enumerate(values) if value >= minimum]
    target = result["target_effective_entries_per_bin"]
    nominal = rounded_events(max(target * total / minimum, 1000 * total / visible))
    probability_low = float(beta.ppf(0.20, minimum, total - minimum + 1))
    visible_probability_low = float(beta.ppf(0.20, visible, total - visible + 1))
    conservative = rounded_events(max(target / probability_low, 1000 / visible_probability_low))
    # Sparse shapes need a fresh pilot before committing the full extrapolation.
    sparse = visible < 30
    pilot = min(conservative, max(10000, rounded_events(100 * total / visible)))
    result.update(core_bin_indices=core,
                  empty_core_bin_indices=[index for index in core if values[index] == 0],
                  measured_core_fraction=sum(values[index] for index in core) / visible,
                  minimum_nonzero_core_bin_events=minimum,
                  nominal_target_total_events=max(total, nominal),
                  conservative_target_total_events=max(total, conservative),
                  pilot_total_events=max(total, pilot) if sparse else None,
                  status="pilot_first_sparse_shape" if sparse else "provisional_shape_target",
                  minimum_core_bin_probability_lower_80=probability_low)
    return result


def measure(point, settings, args):
    path = Path(point["histogram_path"]).resolve()
    before = path.stat()
    checksum = digest(path)
    root_file = open_histograms(path, args.native_xrootd)
    try:
        inclusive = count_events(root_file, "")
        both = count_events(root_file, "Both-Both")
        if inclusive["generated_events"] != both["generated_events"]:
            raise ValueError("Category denominators differ")
        flow = root_file.Get("cutFlow")
        if not flow or not math.isclose(flow.GetBinContent(1), both["generated_events"]):
            raise ValueError("Full unselected histogram denominator differs from cutFlow")
        shape = occupancy_plan(root_file, both, settings, args.relative_error, args.core_fraction)
    finally:
        root_file.Close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during efficiency measurement: " + str(path))
    mass, flight = point["mass_gev"], point["mean_lab_flight_m"]
    name = f"m{mass:g}_prompt" if flight is None else f"m{mass:g}_lab_{flight:g}m"
    return dict(point=name, mass_gev=mass, mean_lab_flight_m=flight,
                histogram_path=str(path), histogram_sha256=checksum,
                histogram_bytes=after.st_size, campaign_directory=str(path.parent.parent),
                inclusive=inclusive, both_both=both, vertex_z_plan=shape)


def apply_credit(rows, filename):
    """Never count unpublished or in-flight events without an explicit inventory."""
    credit = json.loads(filename.read_text()) if filename else {"points": {}}
    for row in rows:
        published = row["both_both"]["generated_events"]
        item = credit["points"].get(row["point"], {})
        completed = int(item.get("completed_total_events", published))
        pending = int(item.get("independent_inflight_events", 0))
        if completed < published or pending < 0:
            raise ValueError("Production credit must include published events and nonnegative pending counts")
        row["production_credit"] = dict(published_events=published, completed_total_events=completed,
                                         independent_inflight_events=pending,
                                         inventory_record=item)
        plan = row["vertex_z_plan"]
        target = plan["conservative_target_total_events"]
        conservative_next_stage = plan["pilot_total_events"] or target
        next_stage = plan["pilot_total_events"] or plan["nominal_target_total_events"]
        plan["planned_next_stage_total_events"] = next_stage
        plan["conservative_next_stage_total_events"] = conservative_next_stage
        plan["additional_events_after_completed"] = max(0, target - completed) if target else None
        plan["additional_events_if_inflight_succeeds"] = max(0, target - completed - pending) if target else None
        plan["planned_next_stage_additional_events"] = max(0, next_stage - completed)
        plan["planned_next_stage_additional_if_inflight_succeeds"] = max(0, next_stage - completed - pending)
        plan["conservative_next_stage_additional_events"] = max(0, conservative_next_stage - completed)
    return dict(path=str(filename.resolve()), sha256=digest(filename)) if filename else None


def maps(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    indexed = {(row["mass_gev"], row["mean_lab_flight_m"]): row for row in rows}
    flights = ["prompt", "5 m", "30 m", "100 m", "1 km"]
    figure, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)
    for axis, category, title in zip(axes, ("inclusive", "both_both"),
                                     ("At least one reconstructed dimuon", "At least one Both-Both dimuon")):
        values = np.array([[indexed[(mass, flight)][category]["efficiency"] for mass in MASSES] for flight in FLIGHTS])
        mesh = axis.imshow(np.ma.masked_less_equal(values, 0), norm=LogNorm(vmin=0.001, vmax=1), cmap="viridis", aspect="auto")
        for iy, flight in enumerate(FLIGHTS):
            for ix, mass in enumerate(MASSES):
                row = indexed[(mass, flight)][category]
                label = f"{100 * row['efficiency']:.1f}%\n{row['selected_events']}/{row['generated_events']}"
                axis.text(ix, iy, label, ha="center", va="center", fontsize=9,
                          color="black" if values[iy, ix] > 0.25 else "white")
        axis.set(xticks=range(len(MASSES)), xticklabels=MASSES, yticks=range(len(FLIGHTS)), yticklabels=flights,
                 xlabel="Dark-photon mass [GeV]", ylabel="Estimated mean lab flight", title=title)
    figure.colorbar(mesh, ax=axes, label="Reconstructed-event efficiency")
    figure.suptitle("Ready signal MC: efficiency before additional analysis cuts\nCounts are unique events; binomial intervals are in the CSV/JSON", fontsize=13)
    figure.savefig(output / "signal_efficiency_map.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14.6, 6.5), constrained_layout=True)
    largest = max(row["vertex_z_plan"]["conservative_target_total_events"] or row["vertex_z_plan"]["pilot_total_events"] for row in rows)
    for axis, field, title in zip(axes, ("planned_next_stage_total_events", "conservative_target_total_events"),
                                  ("Planned next stage: nominal targets + sparse pilots", "Conservative final projection: remeasure first")):
        values = np.array([[indexed[(mass, flight)]["vertex_z_plan"][field] or np.nan
                            for mass in MASSES] for flight in FLIGHTS], dtype=float)
        mesh = axis.imshow(values, norm=LogNorm(vmin=10000, vmax=largest), cmap="magma", aspect="auto")
        for iy, flight in enumerate(FLIGHTS):
            for ix, mass in enumerate(MASSES):
                plan = indexed[(mass, flight)]["vertex_z_plan"]
                target = plan[field]
                label = f"{target / 1000:.1f}k" if target else "pilot"
                if plan["pilot_total_events"]:
                    label += "*"
                axis.text(ix, iy, label, ha="center", va="center",
                          color="black" if target and target > 180000 else "white", fontsize=9)
        axis.set(xticks=range(len(MASSES)), xticklabels=MASSES, yticks=range(len(FLIGHTS)), yticklabels=flights,
                 xlabel="Dark-photon mass [GeV]", ylabel="Estimated mean lab flight", title=title,)
    figure.colorbar(mesh, ax=axes, label="Total generated events")
    figure.suptitle("Signal MC plan for Both-Both dimuon vertex-z plots", fontsize=13)
    figure.supxlabel("100 entries in the most populated bins covering 90% of entries; at least 1,000 selected events.\n80% lower acceptance estimates. * Fewer than 30 selected events: pilot first, then remeasure. Unobserved bins/tails remain uncertain.", fontsize=10)
    figure.savefig(output / "signal_mc_statistics_targets.pdf")
    plt.close(figure)


def write_tables(report, output):
    columns = ["point", "mass_gev", "mean_lab_flight_m", "generated_events", "inclusive_events",
               "both_both_events", "both_both_efficiency", "efficiency_low_68", "efficiency_high_68",
               "efficiency_low_95", "efficiency_high_95", "visible_z_events", "core_bins", "empty_core_bins",
               "nominal_final_target_total", "conservative_final_projection_total", "pilot_target_total", "planned_next_stage_total",
               "completed_total", "inflight_events", "additional_after_completed", "additional_after_inflight",
               "planned_next_stage_additional", "conservative_next_stage_additional", "status"]
    with (output / "signal_mc_statistics_plan.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in report["points"]:
            both, plan, credit = row["both_both"], row["vertex_z_plan"], row["production_credit"]
            writer.writerow([row["point"], row["mass_gev"], "prompt" if row["mean_lab_flight_m"] is None else row["mean_lab_flight_m"],
                             both["generated_events"], row["inclusive"]["selected_events"], both["selected_events"], both["efficiency"],
                             *both["efficiency_interval_68"], *both["efficiency_interval_95"], plan["visible_selected_events"],
                             len(plan["core_bin_indices"]), len(plan["empty_core_bin_indices"]),
                             plan["nominal_target_total_events"], plan["conservative_target_total_events"], plan["pilot_total_events"], plan["planned_next_stage_total_events"],
                             credit["completed_total_events"], credit["independent_inflight_events"],
                             plan["additional_events_after_completed"], plan["additional_events_if_inflight_succeeds"],
                             plan["planned_next_stage_additional_events"], plan["conservative_next_stage_additional_events"], plan["status"]])
    (output / "signal_mc_statistics_plan.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=None, help="Defaults to shift_paths.bsm_campaign")
    parser.add_argument("--output", type=Path, default=ANALYSIS / "plots/bsm_efficiency")
    parser.add_argument("--relative-error", type=float, default=0.10)
    parser.add_argument("--core-fraction", type=float, default=0.90)
    parser.add_argument("--production-credit", type=Path)
    parser.add_argument("--native-xrootd", action="store_true", help="Read ROOT objects through native EOS XRootD")
    args = parser.parse_args()
    if not 0 < args.relative_error < 1 or not 0 < args.core_fraction < 1:
        parser.error("Relative error and core fraction must lie between zero and one")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    from shift_paths import base_path, bsm_campaign
    from shift_sample_paths import available_bsm_campaign_histograms
    campaign = args.campaign or bsm_campaign
    points = available_bsm_campaign_histograms(base_path, campaign, MASSES, FLIGHTS, epsilons=[])
    if len(points) != len(MASSES) * len(FLIGHTS):
        raise ValueError("Efficiency grid requires all thirty ready points")
    settings = plot_settings()
    rows = []
    for point in points:
        row = measure(point, settings, args)
        rows.append(row)
        both = row["both_both"]
        print(row["point"], f"{both['selected_events']}/{both['generated_events']}",
              row["vertex_z_plan"]["nominal_target_total_events"], flush=True)
    credit = apply_credit(rows, args.production_credit)
    report = dict(schema="shift-signal-efficiency-and-mc-plan-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  campaign=campaign, source_code_sha256=digest(__file__), plot_settings=settings,
                  statistical_goal=dict(relative_raw_mc_bin_error=args.relative_error, core_fraction=args.core_fraction,
                                        core_definition="Most-populated bins containing at least 90% of observed entries, including ties",
                                        confidence_lower_bound="80% one-sided per-bin Clopper-Pearson; not simultaneous",
                                        zero_bins="Unmeasured acceptance; excluded from finite extrapolation and listed explicitly",
                                        sparse_shapes="Fewer than 30 visible events: pilot first, remeasure before full target",
                                        planned_allocation="Nominal targets for measured shapes; pilot first below 30 selected events. Conservative projections are uncertainty comparisons, not submitted allocations.",
                                        limitations="Frozen current detector/reconstruction; not a recording/trigger efficiency or sensitivity claim"),
                  production_credit=credit, points=rows)
    report["totals"] = dict(published_events=sum(row["both_both"]["generated_events"] for row in rows),
                            completed_events=sum(row["production_credit"]["completed_total_events"] for row in rows),
                            nominal_target_events=sum(row["vertex_z_plan"]["nominal_target_total_events"] or 0 for row in rows),
                            conservative_target_events=sum(row["vertex_z_plan"]["conservative_target_total_events"] or 0 for row in rows),
                            planned_next_stage_total_events=sum(row["vertex_z_plan"]["planned_next_stage_total_events"] for row in rows),
                            planned_next_stage_additional_events=sum(row["vertex_z_plan"]["planned_next_stage_additional_events"] for row in rows),
                            conservative_next_stage_additional_events=sum(row["vertex_z_plan"]["conservative_next_stage_additional_events"] for row in rows))
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(report, args.output)
    maps(rows, args.output)
    files = sorted(args.output.glob("signal_*.pdf")) + sorted(args.output.glob("signal_mc_statistics_plan.*"))
    (args.output / "artifact_checksums.json").write_text(json.dumps({path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in files}, indent=2) + "\n")
    print("Published diagnostic:", args.output, flush=True)


if __name__ == "__main__":
    main()
