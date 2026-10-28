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
        if node.func.id != "Histogram" or not node.args:
            continue
        name = ast.unparse(node.args[0])
        if "dimuon/ShiftDimuonVertex" in name and "_vz" in name:
            options = {option.arg: ast.literal_eval(option.value) for option in node.keywords
                       if option.arg in ("rebin", "x_min", "x_max")}
            for position, key in ((5, "rebin"), (6, "x_min"), (7, "x_max")):
                if len(node.args) > position:
                    options[key] = ast.literal_eval(node.args[position])
            matches.append(dict(rebin=options.get("rebin", 1),
                                minimum_cm=options.get("x_min"),
                                maximum_cm=options.get("x_max")))
    if len(matches) != 1:
        raise ValueError("Expected one ordinary dimuon vertex-z plot definition")
    return dict(matches[0], config_path=str(source), config_sha256=digest(source))


def open_histograms(path, native):
    import ROOT
    filename = str(path)
    if native:
        filename = filename.replace("/eos/home-j/jniedzie/", "/eos/user/j/jniedzie/")
        filename = "root://eosuser.cern.ch/" + filename
    root_file = ROOT.TFile.Open(filename) if native else ROOT.TFile(filename, "READ")
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
                processed_events=total, effective_processed_events=float(total),
                effective_selected_events=float(selected),
                efficiency=selected / total,
                efficiency_interval_68=interval(selected, total),
                efficiency_interval_95=interval(selected, total, 0.95),
                multiple_dimuon_events=int(round(hist.Integral(3, hist.GetNbinsX() + 1))))


def rounded_events(value):
    return int(math.ceil(value / 20.0) * 20)


def effective_statistics_plan(original, processed, effective, target, first_stage):
    """Credit the completed cohort and forecast independent, identical repeats.

    This is a nominal cost estimate, including for sparse positive samples;
    it is not an efficiency confidence bound or authorization to submit.
    """
    if (not 0 < processed <= original or not math.isfinite(effective)
            or not 0 <= effective <= processed + 1.e-6 or not 0 < first_stage <= target):
        raise ValueError('Invalid completed cohort or effective-statistics target')
    result = dict(target_effective_events=target, first_stage_effective_events=first_stage,
                  status='sparse_nominal_forecast' if effective < 30 else 'nominal_forecast')
    for name, goal in [('first_stage', first_stage), ('final', target)]:
        scale = max(0., goal / effective - 1.) if effective else None
        result[name + '_additional_gen_events'] = math.ceil(original * scale) if scale is not None else None
        result[name + '_additional_sim_events'] = math.ceil(processed * scale) if scale is not None else None
    if not effective:
        result['status'] = 'zero_pass_followup_required'
    return result


def published_weighted_counts(hist, original, processed):
    """The event multiplicity histogram retains one W/p entry per event.

    Its nonzero bins give event-level sums and variances, even if several
    dimuons were reconstructed. Raw passing counts cannot be recovered here.
    """
    if (not hist or hist.GetXaxis().GetBinLowEdge(1) != 0 or hist.GetXaxis().GetBinUpEdge(1) != 1
            or hist.GetBinContent(0) != 0
            or not math.isclose(hist.GetEntries(), processed, abs_tol=1.e-6)
            or hist.GetSumw2N() == 0):
        raise ValueError('Multiplicity histogram does not cover the complete weighted cohort')
    bins = range(2, hist.GetNbinsX() + 2)
    values = [hist.GetBinContent(i) for i in bins]
    variances = [hist.GetBinError(i)**2 for i in bins]
    if any(not math.isfinite(v) or v < 0 for v in values + variances):
        raise ValueError('Require finite positive-weight event statistics')
    sumw, sumw2 = sum(values), sum(variances)
    if (sumw > 0) != (sumw2 > 0):
        raise ValueError('Passing weights and variances disagree')
    return dict(passing_sumw=sumw, passing_sumw2=sumw2, raw_passing_events=None,
                effective_passing_events=sumw**2 / sumw2 if sumw2 else 0.,
                original_population_efficiency=sumw / original,
                poisson_mc_standard_error=math.sqrt(sumw2) / original if sumw2 else None,
                relative_mc_error=math.sqrt(sumw2) / sumw if sumw else None)


def measure_published_cohort(point, native, target, first_stage):
    """Audit complete published weighted cohorts, including histogram-only unions.

    These completion checks belong to this production diagnostic. Ordinary
    plotting continues to accept partial inputs without any such dependency.
    """
    path = Path(point['histogram_path']).resolve()
    directory = path.parent.parent
    rate_path = directory / 'cross_sections.json'
    receipt_path = directory / 'metadata/point_receipt.json'
    if not receipt_path.exists():
        receipt_path = directory / 'metadata/grouping/point_complete.json'
    rate = json.loads(rate_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    original, processed = rate['full_gen']['accepted_events'], receipt['events']
    if (receipt.get('complete') is not True or type(original) is not int
            or type(processed) is not int or not 0 < processed <= original
            or original != receipt['full_gen_events']
            or not math.isclose(rate['full_gen']['sum_weights'], original)
            or not math.isclose(rate['full_gen']['sum_weights_squared'], original)):
        raise ValueError('Require a complete response and unit native GEN exposure')
    union = receipt.get('union', receipt)
    identities = [tuple(identity) for identity in union.get('event_ids', [])]
    if identities and (len(identities) != processed or len(set(identities)) != processed):
        raise ValueError('Repeated or missing original event in completion receipt')
    root = open_histograms(path, native)
    try:
        raw = root.Get('rawEventsCutFlow')
        if not raw or not math.isclose(raw.GetBinContent(1), processed, abs_tol=1.e-6):
            raise ValueError('Published response count differs from completed cohort')
        counts = published_weighted_counts(root.Get('event/Event_nShiftDimuonVertexBoth-Both'), original, processed)
    finally:
        root.Close()
    groups = union.get('groups', receipt.get('groups', []))
    nano_objects = {group['grouped']['path']: group['grouped'] for group in groups if 'grouped' in group}
    nano_bytes = sum(item['bytes'] for item in nano_objects.values()) if nano_objects else None
    plan = effective_statistics_plan(original, processed, counts['effective_passing_events'], target, first_stage)
    if nano_bytes is not None:
        # One retained Nano copy only: GEN, scoring, merges and safety margin are extra.
        plan['final_additional_nano_bytes_lower_bound'] = math.ceil(nano_bytes / processed * plan['final_additional_sim_events']) if counts['effective_passing_events'] else None
    return dict(point=point['name'], mass_gev=point['mass_gev'],
                mean_lab_flight_m=point['mean_lab_flight_m'], histogram_path=str(path),
                original_gen_events=original, processed_events=processed, category='Both-Both',
                mechanism=rate.get('mechanism'), rate_normalization_ready=rate.get('normalization_ready'),
                native_rate_record=str(rate_path), native_rate_sha256=digest(rate_path),
                completion_receipt=str(receipt_path), completion_receipt_sha256=digest(receipt_path),
                grouped_nano_bytes=nano_bytes, allocation=plan, **counts)


def write_published_map(rows, output):
    """One compact delivery: weighted efficiencies, precision and existing credit."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    output.mkdir(parents=True, exist_ok=True)
    columns = ['point', 'original_gen_events', 'processed_events', 'original_population_efficiency',
               'poisson_mc_standard_error', 'effective_passing_events', 'relative_mc_error',
               'first_stage_additional_gen_events', 'first_stage_additional_sim_events',
               'final_additional_gen_events', 'final_additional_sim_events',
               'final_additional_nano_bytes_lower_bound', 'status']
    with (output / 'published_efficiency_allocation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dict(row, **row['allocation']) for row in rows)
    masses = sorted({row['mass_gev'] for row in rows})
    indexed = {(row['mass_gev'], row['mean_lab_flight_m']): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, field, title, bounds in [
            (axes[0], 'original_population_efficiency', 'Both-Both event efficiency', (.001, .3)),
            (axes[1], 'effective_passing_events', 'Existing effective Both-Both events', (1, 1000))]:
        values = np.full((len(FLIGHTS), len(masses)), np.nan)
        for iy, flight in enumerate(FLIGHTS):
            for ix, mass in enumerate(masses):
                row = indexed.get((mass, flight))
                if row is None:
                    continue
                value = row[field]
                values[iy, ix] = value if value else np.nan
                text = f'{100*value:.2f}%' if field == 'original_population_efficiency' else f'{value:.1f}'
                color = 'black' if value > math.sqrt(bounds[0] * bounds[1]) else 'white'
                axis.text(ix, iy, text, ha='center', va='center', color=color, fontsize=9)
        mesh = axis.imshow(np.ma.masked_invalid(values), norm=LogNorm(*bounds), cmap='viridis', aspect='auto')
        axis.set(xticks=range(len(masses)), xticklabels=[f'{m:g}' for m in masses],
                 yticks=range(len(FLIGHTS)), yticklabels=['prompt', '5 m', '30 m', '100 m', '1 km'],
                 xlabel='Dark-photon mass [GeV]', ylabel='Reference mean LAB flight', title=title)
        figure.colorbar(mesh, ax=axis)
    figure.suptitle('Completed published samples; full original GEN denominator and frozen sampling weights')
    figure.supxlabel('At least one Both-Both candidate per event; no BDT or truth matching. Low-mass source tests are separate.\nNominal allocation credits existing events; errors omit source-collision correlations and rate uncertainty.', fontsize=9)
    figure.savefig(output / 'published_efficiency_map.pdf')
    plt.close(figure)


def weighted_response_statistics(probabilities, passed, full_gen_events, target=3000):
    """Survey weights estimate original response; raw counts describe admission.

    One boolean per simulated GEN event prevents multiple pairs from inflating
    the passing statistics. The Poisson MC error is a sparse-count diagnostic,
    not a confidence interval or a source-collision clustering calculation.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    passed = np.asarray(passed)
    if (probabilities.ndim != 1 or passed.shape != probabilities.shape or passed.dtype.kind != 'b'
            or not np.isfinite(probabilities).all() or np.any(probabilities <= 0)
            or np.any(probabilities > 1) or type(full_gen_events) is not int
            or not len(passed) <= full_gen_events or full_gen_events <= 0
            or type(target) is not int or target <= 0):
        raise ValueError('Require complete admitted events, valid probabilities and original GEN count')
    with np.errstate(over='ignore', divide='ignore'):
        weights = 1. / probabilities
        selected = weights[passed]
        sumw = float(selected.sum())
        sumw2 = float(np.square(selected).sum())
    if not np.isfinite(weights).all() or not math.isfinite(sumw + sumw2):
        raise ValueError('Sampling weights overflow the response diagnostic')
    effective = sumw * sumw / sumw2 if sumw2 else 0.
    return dict(processed_events=len(passed), raw_passing_events=int(passed.sum()),
        original_gen_events=full_gen_events, passing_sumw=sumw, passing_sumw2=sumw2,
        effective_passing_events=effective, original_population_efficiency=sumw / full_gen_events,
        poisson_mc_standard_error=math.sqrt(sumw2) / full_gen_events if sumw2 else None,
        relative_mc_error=math.sqrt(sumw2) / sumw if sumw else None,
        target_effective_passing_events=target,
        projected_total_gen_events=rounded_events(full_gen_events * target / effective) if effective >= 30 else None,
        status='provisional_projection' if effective >= 30 else 'bounded_followup_required',
        uncertainty_scope='Poisson diagnostic only; source-collision correlations and physical rate uncertainty are not included',
        zero_pass_interpretation='No finite efficiency forecast or zero-error claim follows from no observed passes')


def measure_weighted_histogram(histogram_path, native=False, target=3000):
    """Read one complete main-histogram cohort and its grouped, weighted Nano.

    Cumulative histogram-only merges retain Nanos at their source locations;
    evaluate those source cohorts individually with this diagnostic.
    """
    histogram_path = Path(histogram_path)
    histograms = open_histograms(histogram_path, native)
    nano = None
    try:
        marker = histograms.Get('ShiftFullGenNormalization')
        if not marker:
            raise ValueError('Weighted response needs embedded original GEN normalization')
        normalization = json.loads(marker.GetString().Data())
        if (normalization.get('denominator_kind') != 'conditional_signal_events'
                or normalization.get('histogram_weight') != 'native_gen_weight/reco_inclusion_probability'):
            raise ValueError('Weighted response currently requires unit-weight conditional signal GEN')
        original = normalization['full_source_sumw']
        if not math.isfinite(original) or int(original) != original or original <= 0:
            raise ValueError('Invalid original GEN count')
        original = int(original)
        nano_path = histogram_path.parent.parent / 'nanoAOD_merged' / histogram_path.name
        nano = open_histograms(nano_path, native)
        tree = nano.Get('Events')
        required = ['run', 'luminosityBlock', 'event', 'nShiftDimuonVertex',
                    'ShiftDimuonVertex_topologyMin', 'ShiftDimuonVertex_topologyMax',
                    'shiftOriginalGenWeight', 'shiftSamplingProbability', 'shiftSamplingGenWeight']
        if not tree or any(not tree.GetBranch(name) for name in required):
            raise ValueError('Grouped Nano lacks response or frozen sampling branches')
        tree.SetBranchStatus('*', 0)
        for name in required:
            tree.SetBranchStatus(name, 1)
        ids = set()
        probabilities, inclusive, both = [], [], []
        for event in tree:
            identity = (int(event.run), int(event.luminosityBlock), int(event.event))
            if identity in ids:
                raise ValueError('Repeated original event in grouped Nano')
            ids.add(identity)
            if (len(event.ShiftDimuonVertex_topologyMin) != int(event.nShiftDimuonVertex)
                    or len(event.ShiftDimuonVertex_topologyMax) != int(event.nShiftDimuonVertex)):
                raise ValueError('Dimuon topology vectors differ from their event multiplicity')
            probability = float(event.shiftSamplingProbability)
            if (float(event.shiftOriginalGenWeight) != 1. or not 0 < probability <= 1
                    or not math.isclose(float(event.shiftSamplingGenWeight), 1. / probability, rel_tol=1.e-10)):
                raise ValueError('Grouped Nano changed the unit native weight or W/p')
            probabilities.append(probability)
            inclusive.append(int(event.nShiftDimuonVertex) > 0)
            both.append(any(int(low) == 2 and int(high) == 2 for low, high in
                            zip(event.ShiftDimuonVertex_topologyMin, event.ShiftDimuonVertex_topologyMax)))
        result = dict(histogram_path=str(histogram_path), nano_path=str(nano_path),
                      normalization=normalization, no_analysis_cuts_or_truth_matching=True)
        for name, category, passed in [('inclusive', '', inclusive), ('both_both', 'Both-Both', both)]:
            counts = weighted_response_statistics(probabilities, np.asarray(passed, dtype=bool), original, target)
            hist = histograms.Get('event/Event_nShiftDimuonVertex' + category)
            if (not hist or hist.GetBinContent(0) != 0
                    or not math.isclose(hist.GetEntries(), len(probabilities))
                    or not math.isclose(hist.Integral(2, hist.GetNbinsX()+1), counts['passing_sumw'], rel_tol=1.e-10, abs_tol=1.e-10)
                    or not math.isclose(sum(hist.GetBinError(i)**2 for i in range(2, hist.GetNbinsX()+2)),
                                        counts['passing_sumw2'], rel_tol=1.e-10, abs_tol=1.e-10)):
                raise ValueError('Grouped event response differs from published multiplicity histogram')
            result[name] = counts
        result['readiness'] = 'Response diagnostic; inspect tails, source correlations and costs before bulk allocation'
        return result
    finally:
        if nano:
            nano.Close()
        histograms.Close()


def event_statistics_plan(counts, target=3000, first_stage_target=500):
    """Size independent GEN events for a common number of passing events.

    This uses the processed-event acceptance, not the rate denominator or the
    number of pairs. Sparse measurements get a bounded pilot before a large
    extrapolation; zero passes never imply a finite measured acceptance.
    """
    total, selected = counts["generated_events"], counts["selected_events"]
    upper = counts["efficiency_interval_95"][1]
    nominal = rounded_events(target * total / selected) if selected else None
    lower = float(beta.ppf(.20, selected, total - selected + 1)) if selected else 0.
    conservative = rounded_events(target / lower) if lower else None
    if selected == 0:
        pilot = total + max(10000, min(100000, 9 * total))
    elif selected < 30:
        pilot = max(total, min(nominal, max(10000, rounded_events(100 * total / selected))))
    elif selected < first_stage_target:
        pilot = max(total, rounded_events(first_stage_target * total / selected))
    else:
        pilot = None
    return dict(target_selected_events=target,
                first_stage_selected_events=first_stage_target,
                nominal_target_total_events=max(total, nominal) if nominal else None,
                conservative_target_total_events=max(total, conservative) if conservative else None,
                planning_acceptance_lower_80=lower,
                optimistic_target_total_from_upper_95=rounded_events(target / upper),
                pilot_total_events=pilot,
                planned_next_stage_total_events=pilot or max(total, nominal),
                status="zero_pass_pilot" if not selected else "sparse_pilot" if selected < 30 else
                       "first_measured_stage" if selected < first_stage_target else "measured_acceptance_target")


def normalization_context(histogram_path, processed, selected):
    """Expose full GEN accounting without treating unfinished processing as loss."""
    path = Path(histogram_path).parent.parent / "cross_sections.json"
    if not path.is_file():
        return dict(status="full_gen_metadata_unavailable", processed_events=processed)
    data = json.loads(path.read_text())
    full = data.get("full_gen", {})
    events = full.get("accepted_events")
    if type(events) is not int or events < processed:
        raise ValueError("Full GEN metadata cannot cover the processed event denominator: " + str(path))
    sumw, sumw2 = full.get("sum_weights"), full.get("sum_weights_squared")
    return dict(status="full_gen_metadata_read", metadata_path=str(path), metadata_sha256=digest(path),
                full_gen_events=events, full_gen_sumw=sumw, full_gen_sumw2=sumw2,
                full_gen_effective_events=sumw * sumw / sumw2 if sumw2 and sumw is not None else None,
                processed_events=processed, processing_fraction=processed / events,
                selected_per_full_gen_available=selected / events,
                source_collisions=full.get("source_collisions"),
                mechanism=data.get("mechanism"), model=data.get("model"),
                interpretation="Efficiency uses processed events; source collisions normalize production rates. Missing processing is not detector inefficiency.")


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
        if settings["minimum_cm"] is not None and high <= settings["minimum_cm"]:
            continue
        if settings["maximum_cm"] is not None and low >= settings["maximum_cm"]:
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
        raw_flow = root_file.Get("rawEventsCutFlow")
        if not raw_flow or not math.isclose(raw_flow.GetBinContent(1), both["generated_events"]):
            raise ValueError("Raw processed event count differs from unit-weight multiplicity denominator")
        shape = occupancy_plan(root_file, both, settings, args.relative_error, args.core_fraction)
    finally:
        root_file.Close()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during efficiency measurement: " + str(path))
    mass, flight = point["mass_gev"], point["mean_lab_flight_m"]
    name = f"m{mass:g}_prompt" if flight is None else f"m{mass:g}_lab_{flight:g}m"
    name = point.get("name", name.replace(".", "p"))
    return dict(point=name, mass_gev=mass, mean_lab_flight_m=flight,
                histogram_path=str(path), histogram_sha256=checksum,
                histogram_bytes=after.st_size, campaign_directory=str(path.parent.parent),
                inclusive=inclusive, both_both=both, vertex_z_plan=shape,
                normalization=normalization_context(path, both["generated_events"], both["selected_events"]),
                event_statistics_plan=event_statistics_plan(both, args.target_selected_events,
                                                           args.first_stage_selected_events))


def apply_credit(rows, filename):
    """Never count unpublished or in-flight events without an explicit inventory."""
    credit = json.loads(filename.read_text()) if filename else {"points": {}}
    for row in rows:
        published = row["both_both"]["generated_events"]
        item = credit["points"].get(row["point"], {})
        completed = int(item.get("completed_total_events", published))
        pending = int(item.get("independent_inflight_events", 0))
        same_population = item.get("reference_histogram_is_allocation_population", True)
        if completed < (published if same_population else 0) or pending < 0:
            raise ValueError("Production credit must include published events and nonnegative pending counts")
        row["production_credit"] = dict(published_events=published, completed_total_events=completed,
                                         independent_inflight_events=pending,
                                         reference_histogram_is_allocation_population=same_population,
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
        event_plan = row.get("event_statistics_plan")
        if event_plan:
            next_total = event_plan["planned_next_stage_total_events"]
            event_plan["additional_events_after_completed"] = max(0, next_total - completed)
            event_plan["additional_events_if_inflight_succeeds"] = max(0, next_total - completed - pending)
            event_plan["credit_inventory_supplied"] = bool(filename and row["point"] in credit["points"])
    return dict(path=str(filename.resolve()), sha256=digest(filename)) if filename else None


def storage_capped_round(rows, density_inventory, budget_bytes, base_passes=30, priority_passes=100):
    """A bounded first round; final targets remain separate from storage admission.

    Densities must already include the reviewed margin and source-ledger model.
    Existing GEN credit is never counted again. This is a planning forecast:
    native queue, peer reservations and actual new-model output sizes still
    need the production controller's fresh checks before submission.
    """
    if budget_bytes <= 0 or not 0 < base_passes <= priority_passes:
        raise ValueError("Round budget and staged passing-event goals must be positive")
    points = []
    for row in rows:
        density = density_inventory["points"].get(row["point"])
        if density is None:
            continue
        if density["reserved_bytes_per_gen"] <= 0 or density["fixed_bytes_per_point"] < 0:
            raise ValueError("Storage density must be positive with a nonnegative fixed allowance")
        credit = row["production_credit"]
        points.append(dict(point=row["point"], mass_gev=row["mass_gev"],
            mean_lab_flight_m=row["mean_lab_flight_m"], reference_efficiency=row["both_both"]["efficiency"],
            reference_efficiency_interval_95=row["both_both"].get("efficiency_interval_95"),
            committed_gen_events=credit["completed_total_events"] + credit["independent_inflight_events"],
            additional_gen_events=0, reserved_bytes=0, density=density,
            allocation_status="zero_efficiency_requires_separate_exploration" if not row["both_both"]["selected_events"] else "planned"))
    points.sort(key=lambda p: (p["mean_lab_flight_m"] != 1000, p["mass_gev"],
                               -1 if p["mean_lab_flight_m"] is None else p["mean_lab_flight_m"]))
    used = 0

    def increase(point, selected_target):
        nonlocal used
        efficiency = point["reference_efficiency"]
        if not efficiency:
            return
        required = rounded_events(max(0., selected_target / efficiency - point["committed_gen_events"]))
        if required <= point["additional_gen_events"]:
            return
        density = point["density"]
        reserve = math.ceil(required * density["reserved_bytes_per_gen"] + density["fixed_bytes_per_point"])
        increment = reserve - point["reserved_bytes"]
        if used + increment > budget_bytes:
            return
        used += increment
        point.update(additional_gen_events=required, reserved_bytes=reserve)

    for target in range(1, base_passes + 1):
        for point in points:
            increase(point, target)
    for target in range(base_passes + 1, priority_passes + 1):
        for point in points:
            if point["mean_lab_flight_m"] == 1000:
                increase(point, target)
    for target in range(base_passes + 1, priority_passes + 1):
        for point in points:
            if point["mean_lab_flight_m"] != 1000:
                increase(point, target)
    for point in points:
        total = point["committed_gen_events"] + point["additional_gen_events"]
        point["expected_passing_events_after_round"] = total * point["reference_efficiency"]
        interval = point["reference_efficiency_interval_95"]
        point["mean_passing_interval_95_from_reference_efficiency"] = [total * value for value in interval] if interval else None
        ratio = point["density"].get("source_collisions_per_gen")
        point["additional_source_collisions_forecast"] = math.ceil(point["additional_gen_events"] * ratio) if ratio else None
    return dict(schema="shift-signal-storage-capped-round-v1", budget_bytes=budget_bytes,
                reserved_bytes=used, unallocated_bytes=budget_bytes - used,
                base_selected_event_goal=base_passes, priority_selected_event_goal=priority_passes,
                priority="Low-mass 1km points after broad first coverage; smaller bounded exploration before the 500/3000 pass stages",
                additional_gen_events=sum(p["additional_gen_events"] for p in points),
                production_admitted=False,
                limitations="Efficiency proxy and source/GEN ratios belong to the ready/current19 models. Recalibrate the extended coherent model, actual storage density, native queue and shared peer claims before admission; expected passes are not guaranteed.",
                points=points)


def maps(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    indexed = {(row["mass_gev"], row["mean_lab_flight_m"]): row for row in rows}
    masses = sorted({row["mass_gev"] for row in rows})
    flights = ["prompt", "5 m", "30 m", "100 m", "1 km"]
    width = max(12., .95 * len(masses))
    figure, axes = plt.subplots(2, 1, figsize=(width, 9.), constrained_layout=True)
    positive = [row[category]["efficiency"] for row in rows for category in ("inclusive", "both_both")
                if row[category]["efficiency"] > 0]
    minimum = min(positive) if positive else .001
    for axis, category, title in zip(axes, ("inclusive", "both_both"),
                                     ("At least one reconstructed dimuon", "At least one Both-Both dimuon")):
        values = np.full((len(FLIGHTS), len(masses)), np.nan)
        for iy, flight in enumerate(FLIGHTS):
            for ix, mass in enumerate(masses):
                point = indexed.get((mass, flight))
                if point is None:
                    axis.text(ix, iy, "no sample", ha="center", va="center", fontsize=7)
                    continue
                row = point[category]
                values[iy, ix] = max(minimum / 2., row["efficiency"])
                label = f"{100 * row['efficiency']:.2g}%\n{row['selected_events']}/{row['generated_events']}"
                if not row["selected_events"]:
                    label = f"<{100 * row['efficiency_interval_95'][1]:.2g}%\n0/{row['generated_events']}"
                axis.text(ix, iy, label, ha="center", va="center", fontsize=7,
                          color="black" if values[iy, ix] > .25 else "white")
        mesh = axis.imshow(np.ma.masked_invalid(values), norm=LogNorm(vmin=minimum / 2., vmax=1),
                           cmap="viridis", aspect="auto")
        axis.set(xticks=range(len(masses)), xticklabels=[f"{mass:g}" for mass in masses],
                 yticks=range(len(FLIGHTS)), yticklabels=flights, xlabel="Dark-photon mass [GeV]",
                 ylabel="Reference mean LAB flight", title=title)
    figure.colorbar(mesh, ax=axes, label="Passing-event / processed-event efficiency")
    figure.suptitle("Ready signal MC: current plot categories, before additional cuts\nUnique unit-weight events; zero-pass labels are 95% upper limits, not measured efficiencies", fontsize=12)
    figure.savefig(output / "signal_efficiency_map.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(width, 9.), constrained_layout=True)
    largest = max(row["event_statistics_plan"]["conservative_target_total_events"] or
                  row["event_statistics_plan"]["pilot_total_events"] for row in rows)
    for axis, field, title in zip(axes, ("planned_next_stage_total_events", "conservative_target_total_events"),
                                  ("Next stage: measured targets and sparse pilots", "Final projection using 80% lower acceptance; remeasure pilots")):
        values = np.full((len(FLIGHTS), len(masses)), np.nan)
        for iy, flight in enumerate(FLIGHTS):
            for ix, mass in enumerate(masses):
                point = indexed.get((mass, flight))
                if point is None:
                    continue
                plan = point["event_statistics_plan"]
                target = plan[field]
                if target:
                    values[iy, ix] = target
                label = f"{target / 1000:.1f}k" if target else "unknown"
                if plan["pilot_total_events"]:
                    label += "*"
                axis.text(ix, iy, label, ha="center", va="center", fontsize=7,
                          color="black" if target and target > largest / 2. else "white")
        mesh = axis.imshow(np.ma.masked_invalid(values), norm=LogNorm(vmin=1000, vmax=max(1001, largest)), cmap="magma", aspect="auto")
        axis.set(xticks=range(len(masses)), xticklabels=[f"{mass:g}" for mass in masses],
                 yticks=range(len(FLIGHTS)), yticklabels=flights, xlabel="Dark-photon mass [GeV]",
                 ylabel="Reference mean LAB flight", title=title)
    figure.colorbar(mesh, ax=axes, label="Total independent GEN events")
    target = rows[0]["event_statistics_plan"]["target_selected_events"]
    figure.suptitle(f"Statistics plan: {target:,} passing Both-Both events per point", fontsize=13)
    first_stage = rows[0]["event_statistics_plan"]["first_stage_selected_events"]
    figure.supxlabel(f"* Intermediate stage: {first_stage:,} passes; below 30 observations, a smaller exploration precedes remeasurement. Production credit is listed separately.\nThe low-mass source model is still being extended; these efficiencies are planning estimates for the current ready model.", fontsize=10)
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
    with (output / "signal_event_allocation.csv").open("w", newline="") as handle:
        fields = ["point", "mass_gev", "mean_lab_flight_m", "processed_events", "full_gen_events",
                  "passing_events", "efficiency", "target_passing_events", "nominal_total_gen",
                  "conservative_total_gen", "next_stage_total_gen", "additional_after_completed",
                  "additional_after_inflight", "credit_inventory_supplied", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["points"]:
            counts, plan = row["both_both"], row["event_statistics_plan"]
            writer.writerow(dict(point=row["point"], mass_gev=row["mass_gev"],
                mean_lab_flight_m="prompt" if row["mean_lab_flight_m"] is None else row["mean_lab_flight_m"],
                processed_events=counts["processed_events"], full_gen_events=row["normalization"].get("full_gen_events"),
                passing_events=counts["selected_events"], efficiency=counts["efficiency"],
                target_passing_events=plan["target_selected_events"], nominal_total_gen=plan["nominal_target_total_events"],
                conservative_total_gen=plan["conservative_target_total_events"],
                next_stage_total_gen=plan["planned_next_stage_total_events"],
                additional_after_completed=plan["additional_events_after_completed"],
                additional_after_inflight=plan["additional_events_if_inflight_succeeds"],
                credit_inventory_supplied=plan["credit_inventory_supplied"], status=plan["status"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=None, help="Defaults to shift_paths.bsm_campaign")
    parser.add_argument("--output", type=Path, default=ANALYSIS / "plots/physics/bsm_efficiency")
    parser.add_argument("--relative-error", type=float, default=0.10)
    parser.add_argument("--core-fraction", type=float, default=0.90)
    parser.add_argument("--production-credit", type=Path)
    parser.add_argument("--masses", nargs="+", type=float, help="Default: all available masses")
    parser.add_argument("--target-selected-events", type=int, default=3000)
    parser.add_argument("--first-stage-selected-events", type=int, default=500)
    parser.add_argument("--storage-density-inventory", type=Path)
    parser.add_argument("--round-budget-gb", type=float, default=100., help="Decimal GB; requires a density inventory")
    parser.add_argument("--native-xrootd", action="store_true", help="Read ROOT objects through native EOS XRootD")
    parser.add_argument('--weighted-histograms', nargs='+', type=Path,
                        help='Measure complete W/p-weighted main histograms using their grouped Nano; write a response JSON')
    parser.add_argument('--published-weighted', action='store_true',
                        help='Audit complete published W/p cohorts; map efficiencies and credit existing effective statistics')
    args = parser.parse_args()
    if not 0 < args.relative_error < 1 or not 0 < args.core_fraction < 1:
        parser.error("Relative error and core fraction must lie between zero and one")
    if not 0 < args.first_stage_selected_events <= args.target_selected_events:
        parser.error("Selected-event targets require 0 < first-stage <= final target")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if args.weighted_histograms:
        rows = [measure_weighted_histogram(path, args.native_xrootd, args.target_selected_events)
                for path in args.weighted_histograms]
        args.output.mkdir(parents=True, exist_ok=True)
        report = dict(schema='shift-weighted-signal-response-v1', points=rows,
                      source_code_sha256=digest(__file__), created_utc=datetime.now(timezone.utc).isoformat())
        (args.output / 'weighted_efficiency.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return
    from shift_paths import base_path, bsm_campaign
    from shift_sample_paths import available_bsm_campaign_histograms
    campaign = args.campaign or bsm_campaign
    points = available_bsm_campaign_histograms(base_path, campaign, args.masses, FLIGHTS, epsilons=[])
    if not points:
        raise ValueError("No ready points are available for efficiency measurement")
    rows = []
    if args.published_weighted:
        rows = [measure_published_cohort(point, args.native_xrootd, args.target_selected_events,
                                        args.first_stage_selected_events) for point in points]
        write_published_map(rows, args.output)
        report = dict(schema='shift-published-weighted-response-plan-v1', campaign=campaign, points=rows,
                      created_utc=datetime.now(timezone.utc).isoformat(), source_code_sha256=digest(__file__),
                      assumptions='Nominal linear repeats of each measured prescription; no rate qualification, source-collision covariance, guaranteed bin precision or bulk submission implied.')
        report['totals'] = {key: sum(row['allocation'].get(key, 0) or 0 for row in rows) for key in
                            ['first_stage_additional_gen_events', 'first_stage_additional_sim_events',
                             'final_additional_gen_events', 'final_additional_sim_events',
                             'final_additional_nano_bytes_lower_bound']}
        report['totals'].update(original_gen_events=sum(row['original_gen_events'] for row in rows),
                                processed_events=sum(row['processed_events'] for row in rows),
                                effective_passing_events=sum(row['effective_passing_events'] for row in rows))
        (args.output / 'published_efficiency_allocation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return
    settings = plot_settings()
    for point in points:
        row = measure(point, settings, args)
        rows.append(row)
        both = row["both_both"]
        print(row["point"], f"{both['selected_events']}/{both['generated_events']}",
              row["vertex_z_plan"]["nominal_target_total_events"], flush=True)
    credit = apply_credit(rows, args.production_credit)
    report = dict(schema="shift-signal-efficiency-and-mc-plan-v2", created_utc=datetime.now(timezone.utc).isoformat(),
                  campaign=campaign, source_code_sha256=digest(__file__), plot_settings=settings,
                  statistical_goal=dict(relative_raw_mc_bin_error=args.relative_error, core_fraction=args.core_fraction,
                                        passing_events_per_point=args.target_selected_events,
                                        allocation_basis="Both-Both events divided by processed unit-weight events; no truth, BDT or extra analysis cuts",
                                        passing_event_definition="At least one pair in the plotted topology category, including any reconstructed candidate; no A-prime truth-match requirement",
                                        submission_gate="Reconcile independent completed/in-flight events before submission; missing credit is not permission to duplicate work",
                                        core_definition="Most-populated bins containing at least 90% of observed entries, including ties",
                                        confidence_lower_bound="80% one-sided per-bin Clopper-Pearson; not simultaneous",
                                        zero_bins="Unmeasured acceptance; excluded from finite extrapolation and listed explicitly",
                                        sparse_shapes="Fewer than 30 visible events: pilot first, remeasure before full target",
                                        planned_allocation="Final target of 3000 passing events; first 500 passes and smaller sparse explorations. A supplied storage density can cap the initial round. Shape-bin projections are separate diagnostics, not submitted allocations.",
                                        limitations="Frozen current detector/reconstruction; not a recording/trigger efficiency or sensitivity claim"),
                  production_credit=credit, points=rows)
    report["totals"] = dict(published_events=sum(row["both_both"]["generated_events"] for row in rows),
                            completed_events=sum(row["production_credit"]["completed_total_events"] for row in rows),
                            nominal_target_events=sum(row["vertex_z_plan"]["nominal_target_total_events"] or 0 for row in rows),
                            conservative_target_events=sum(row["vertex_z_plan"]["conservative_target_total_events"] or 0 for row in rows),
                            planned_next_stage_total_events=sum(row["vertex_z_plan"]["planned_next_stage_total_events"] for row in rows),
                            planned_next_stage_additional_events=sum(row["vertex_z_plan"]["planned_next_stage_additional_events"] for row in rows),
                            conservative_next_stage_additional_events=sum(row["vertex_z_plan"]["conservative_next_stage_additional_events"] for row in rows))
    report["event_allocation_totals"] = dict(
        next_stage_total_gen=sum(row["event_statistics_plan"]["planned_next_stage_total_events"] for row in rows),
        additional_gen_after_completed=sum(row["event_statistics_plan"]["additional_events_after_completed"] for row in rows),
        additional_gen_after_inflight=sum(row["event_statistics_plan"]["additional_events_if_inflight_succeeds"] for row in rows),
        nominal_eventual_gen=sum(row["event_statistics_plan"]["nominal_target_total_events"] or 0 for row in rows),
        unknown_zero_efficiency_points=[row["point"] for row in rows if not row["both_both"]["selected_events"]])
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(report, args.output)
    if args.storage_density_inventory:
        density = json.loads(args.storage_density_inventory.read_text())
        allocation = storage_capped_round(rows, density, int(args.round_budget_gb * 1.e9))
        allocation["density_inventory"] = dict(path=str(args.storage_density_inventory.resolve()),
                                               sha256=digest(args.storage_density_inventory))
        (args.output / "signal_lowmass_storage_capped_round.json").write_text(json.dumps(allocation, indent=2) + "\n")
    maps(rows, args.output)
    files = sorted(args.output.glob("signal_*.pdf")) + sorted(args.output.glob("signal_mc_statistics_plan.*")) + [args.output / "signal_event_allocation.csv"]
    (args.output / "artifact_checksums.json").write_text(json.dumps({path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in files}, indent=2) + "\n")
    print("Published diagnostic:", args.output, flush=True)


if __name__ == "__main__":
    main()
