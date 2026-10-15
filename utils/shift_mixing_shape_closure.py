# Code style: AI-code. AI-generated event-mixing shape-closure diagnostic.
"""Compare real and mixed shapes without borrowing a same-sign rate estimate.

The event-replacement model in shift_mixing supplies the physical relative
weights across generator and occupancy pools. Each curve is then normalized
to unit area. This tests the shape of that independence model, not its rate.
All occurrences of an original collision share a bootstrap multiplier, and
the weighted donor normalization is resampled with the original events.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

from shift_mixing import (
    DEFAULT_CONFIG, activity_bin, digest, event_replacement, hit_truth, iter_pool,
    pools_for, same_event_closure, same_event_rows, validate_inventory, write_json,
)


OBSERVABLES = {
    "mass": ([0., 1., 2., 4., 6., 10., 20., 40., 80., 120., 200.],
             "Dimuon mass [GeV]"),
    "pt": ([0., .5, 1., 2., 4., 8., 20., 40., 80., 200., 1000.],
           "Dimuon transverse momentum [GeV]"),
    "deltaPhi": ([0., .25, .5, 1., 1.5, 2., 2.5, np.pi],
                 "Absolute azimuthal separation [rad]"),
    "openingAngle": ([0., .001, .003, .01, .03, .1, .3, np.pi],
                     "Opening angle [rad]"),
    "deltaEta": ([0., .1, .25, .5, 1., 2., 3., 5., 20.],
                 "Absolute pseudorapidity separation"),
    "ptAsymmetry": ([0., .1, .25, .5, .75, .9, 1.],
                    "Muon transverse-momentum asymmetry"),
}


def compact_row(row):
    result = {name: row[name] for name in (
        "anchor_event", "donor_event", "source_events", "stratum", "partition",
        "activity_bin", "weight", "kind", "both_endcaps", "truth_reason",
        "common_parent_pdg",
    )}
    result["pair"] = {name: row["pair"][name] for name in OBSERVABLES}
    if "muons" in row and row["anchor_event"] == row["donor_event"]:
        first, second = row["muons"]
        a, b = hit_truth(first), hit_truth(second)
        component = "unknown_hit_match"
        if a is not None and b is not None:
            duplicate = a["index"] == b["index"] or (
                first.get("hitSimTrackId", -1) >= 0
                and first["hitSimTrackId"] == second.get("hitSimTrackId", -2))
            if duplicate:
                component = "same_true_muon_twice"
            else:
                parents = [next((item for item in muon["truth"]["hit_match"]
                                 if abs(item["pdgId"]) != 13), None)
                           for muon in (first, second)]
                if any(parent is None for parent in parents):
                    component = "distinct_true_muons_unknown_parent"
                elif parents[0]["index"] == parents[1]["index"]:
                    component = "distinct_true_muons_same_parent"
                else:
                    component = "distinct_true_muons_different_parents"
        result["truth_component"] = component
    return result


def read_rows(path, events):
    rows = []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = compact_row(json.loads(line))
            indices = row["anchor_event"], row["donor_event"]
            if row["source_events"] != [events[index]["event_key"] for index in indices]:
                raise ValueError("Row identity does not match its original event")
            for index in indices:
                event = events[index]
                if (row["partition"] != event["partition"]
                        or row["stratum"] != event["stratum"]
                        or row["activity_bin"] != activity_bin(event)):
                    raise ValueError("Mixed row crossed its event partition or pool")
            rows.append(row)
    return rows


def prepare_rows(rows, event_count):
    first = np.asarray([row["anchor_event"] for row in rows], dtype=int)
    second = np.asarray([row["donor_event"] for row in rows], dtype=int)
    weights = np.asarray([row["weight"] for row in rows], dtype=float)
    if (np.any(first < 0) or np.any(second < 0) or np.any(first >= event_count)
            or np.any(second >= event_count) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("Invalid original-event index or input weight")
    if any((a == b) != row["kind"].startswith("same_")
           for a, b, row in zip(first, second, rows)):
        raise ValueError("Mixed pairs must use two distinct original events")
    indices = {}
    for variable, (edges, _) in OBSERVABLES.items():
        edges = np.asarray(edges)
        values = np.asarray([row["pair"][variable] for row in rows])
        if not np.isfinite(values).all() or np.any(values < edges[0]) or np.any(values > edges[-1]):
            raise ValueError("Shape domain does not cover every value: " + variable)
        index = np.searchsorted(edges, values, side="right") - 1
        index[values == edges[-1]] = len(edges) - 2
        indices[variable] = index
    return first, second, weights, indices


def normalized_histograms(data, weights):
    total = float(weights.sum())
    if total <= 0:
        return None
    return {name: np.bincount(data[3][name], weights=weights,
                             minlength=len(settings[0]) - 1) / total
            for name, settings in OBSERVABLES.items()}


def shape_closure(same, mixed, events, repetitions, seed, donor_pools=None):
    """Joint Poisson bootstrap of unit-area shapes on a fixed partner graph.

    Row count is never used as the independent sample size. The quoted tail
    fraction is an approximate centered-bootstrap shape diagnostic; sparse
    empirical tails cannot be validated by an absent-bin uncertainty of zero.
    """
    if repetitions < 2 or not events:
        raise ValueError("Shape closure needs original events and two bootstrap repetitions")
    real = prepare_rows(same, len(events))
    mock = prepare_rows(mixed, len(events))
    nominal_real = normalized_histograms(real, real[2])
    nominal_mock = normalized_histograms(mock, mock[2])
    if nominal_real is None or nominal_mock is None:
        raise ValueError("Both real and mixed samples need positive weight")
    groups = np.zeros(len(events), dtype=int)
    pools = pools_for(events) if donor_pools is None else donor_pools
    membership = [index for indices in pools.values() for index in indices]
    if sorted(membership) != list(range(len(events))):
        raise ValueError("Donor pools must partition all original events exactly once")
    for group, indices in enumerate(pools.values()):
        groups[indices] = group
    detector_weights = np.asarray([event["detector_weight"] for event in events])
    total_weight = np.bincount(groups, weights=detector_weights, minlength=len(pools))
    original_donor_sum = total_weight[groups[mock[0]]] - detector_weights[mock[0]]
    real_samples = {variable: [] for variable in OBSERVABLES}
    mixed_samples = {variable: [] for variable in OBSERVABLES}
    rng = np.random.default_rng(seed)
    invalid = 0
    for _ in range(repetitions):
        count = rng.poisson(1., len(events))
        donor_total = np.bincount(groups, weights=count * detector_weights,
                                  minlength=len(pools))
        donor_sum = donor_total[groups[mock[0]]] - count[mock[0]] * detector_weights[mock[0]]
        correction = np.divide(original_donor_sum, donor_sum,
                               out=np.zeros(len(mixed)), where=donor_sum > 0)
        real_hist = normalized_histograms(real, real[2] * count[real[0]])
        mixed_weight = mock[2] * count[mock[0]] * count[mock[1]] * correction
        mixed_hist = normalized_histograms(mock, mixed_weight)
        if real_hist is None or mixed_hist is None:
            invalid += 1
            continue
        for variable in OBSERVABLES:
            real_samples[variable].append(real_hist[variable])
            mixed_samples[variable].append(mixed_hist[variable])
    if repetitions - invalid < 2:
        raise ValueError("Insufficient nonempty source-event bootstrap replicas")
    source_weights = np.bincount(real[0], weights=real[2], minlength=len(events))
    summary = dict(
        real_rows=len(same), mixed_rows=len(mixed),
        real_distinct_originals=int(np.count_nonzero(source_weights)),
        mixed_distinct_originals=len(set(mock[0]) | set(mock[1])),
        real_weight_pb=float(real[2].sum()), mixed_model_weight_pb=float(mock[2].sum()),
        real_effective_originals=float(source_weights.sum() ** 2 / (source_weights @ source_weights)),
        mixed_model_over_real_rate=float(mock[2].sum() / real[2].sum()),
        rate_uses_os_for_normalization=False, shapes_normalized_to_unit_area=True,
        repetitions=repetitions, valid_repetitions=repetitions - invalid,
        empty_repetitions=invalid, observables={},
        uncertainty="Shared original-event Poisson bootstrap, including donor normalization, on a fixed partner graph",
        p_value_scope="Approximate centered-bootstrap shape diagnostic, not proof of independent muons or tail validity",
        empirical_empty_bins_do_not_establish_a_physical_zero=True,
    )
    for variable, (edges, _) in OBSERVABLES.items():
        a, b = np.asarray(real_samples[variable]), np.asarray(mixed_samples[variable])
        difference = nominal_mock[variable] - nominal_real[variable]
        bootstrap_difference = b - a
        centered = bootstrap_difference - difference
        cdf_distance = float(np.max(np.abs(np.cumsum(difference))))
        cdf_trials = np.max(np.abs(np.cumsum(centered, axis=1)), axis=1)
        std = np.std(bootstrap_difference, axis=0, ddof=1)
        pulls = np.divide(difference, std, out=np.zeros_like(difference), where=std > 0)
        result = dict(
            edges=edges, real_shape=nominal_real[variable].tolist(),
            mixed_shape=nominal_mock[variable].tolist(), difference=difference.tolist(),
            real_shape_quantiles=np.quantile(a, [.025, .16, .5, .84, .975], axis=0).tolist(),
            mixed_shape_quantiles=np.quantile(b, [.025, .16, .5, .84, .975], axis=0).tolist(),
            difference_quantiles=np.quantile(bootstrap_difference, [.025, .16, .5, .84, .975], axis=0).tolist(),
            difference_std=std.tolist(), difference_covariance=np.cov(bootstrap_difference, rowvar=False).tolist(),
            standardized_bin_differences=pulls.tolist(),
            max_absolute_standardized_bin_difference=float(np.max(np.abs(pulls))),
            binned_cdf_distance=cdf_distance,
            centered_bootstrap_cdf_tail_fraction=float((1 + np.count_nonzero(cdf_trials >= cdf_distance)) / (len(cdf_trials) + 1)),
            raw_real_bin_rows=np.bincount(real[3][variable], minlength=len(edges) - 1).tolist(),
            raw_mixed_bin_rows=np.bincount(mock[3][variable], minlength=len(edges) - 1).tolist(),
        )
        summary["observables"][variable] = result
    return summary


def draw_shapes(summary, output, category, partition, graph):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output) as pdf:
        for variable, (_, label) in OBSERVABLES.items():
            record = summary["observables"][variable]
            edges = np.asarray(record["edges"])
            figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
            for name, color, description in (
                    ("real", "black", "Real same-event pairs"),
                    ("mixed", "tab:blue", "Mixed pairs (different original events)")):
                value = np.asarray(record[name + "_shape"])
                quantiles = np.asarray(record[name + "_shape_quantiles"])
                axes[0].stairs(value, edges, color=color, label=description, linewidth=1.8)
                axes[0].fill_between(edges, np.r_[quantiles[1], quantiles[1, -1]],
                                     np.r_[quantiles[3], quantiles[3, -1]],
                                     step="post", color=color, alpha=.18)
            difference = np.asarray(record["difference"])
            quantiles = np.asarray(record["difference_quantiles"])
            axes[1].stairs(difference, edges, color="tab:blue")
            axes[1].fill_between(edges, np.r_[quantiles[1], quantiles[1, -1]],
                                 np.r_[quantiles[3], quantiles[3, -1]],
                                 step="post", color="tab:blue", alpha=.22)
            axes[1].axhline(0., color="black", linewidth=.8)
            axes[0].set(ylabel="Fraction of pairs / bin", yscale="log", ylim=(1.e-5, 2.))
            axes[1].set(xlabel=label, ylabel="Mixed - real")
            if variable in ("mass", "pt"):
                axes[1].set_xscale("symlog", linthresh=2.)
            elif variable == "openingAngle":
                axes[1].set_xscale("symlog", linthresh=.001)
            axes[1].set_xlim(edges[0], edges[-1])
            axes[0].legend(loc="upper right", fontsize=9)
            axes[0].set_title(f"QCD {category}, {partition}; {graph}\n"
                              f"{summary['real_rows']} real pairs, {summary['real_effective_originals']:.1f} effective originals; "
                              f"{summary['mixed_rows']} mixed combinations")
            axes[0].text(.02, .035, "Unit-area shape test; shared original-event 68% intervals\n"
                         f"CDF difference = {record['binned_cdf_distance']:.3f}, "
                         f"bootstrap tail fraction = {record['centered_bootstrap_cdf_tail_fraction']:.3f}\n"
                         "Sparse or empty tails remain untested; this plot does not set a background rate.",
                         transform=axes[0].transAxes, fontsize=8)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def run_hard_scale_closure(events, hard_scales, output, plots, partners=100,
                           repetitions=1000, seed=2026101502):
    """Test whether a common collision scale explains the hard-bin pT tension.

    The three predeclared generator-pThat intervals are 20--30, 30--50 and
    50+ GeV. Existing campaign, source-bin, fold and occupancy restrictions
    remain. The ordinary and scale-matched models use exactly the same
    eligible originals. Hard-scale information is an MC diagnostic, not a
    classifier input or a selection on the observed pair being tested.
    """
    output, plots = Path(output), Path(plots)
    output.mkdir(parents=True, exist_ok=False)
    plots.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    high = [event for event in events if event["stratum"] == "qcd_20toinf"]

    def scale_bin(event):
        value = float(hard_scales[event["event_key"]])
        if not np.isfinite(value) or value < 19.9:
            raise ValueError("Missing or invalid generator pThat in QCD20+")
        return "20to30" if value < 30. else "30to50" if value < 50. else "50toinf"

    def split_pools(source):
        result = {}
        for key, indices in pools_for(source).items():
            for index in indices:
                result.setdefault(key + (scale_bin(source[index]),), []).append(index)
        return result

    original_pools = split_pools(high)
    allowed = {index for indices in original_pools.values() if len(indices) >= 2 for index in indices}
    excluded = [event for index, event in enumerate(high) if index not in allowed]
    eligible = [event for index, event in enumerate(high) if index in allowed]
    if not eligible:
        raise ValueError("No generator-scale donor pools have two original events")
    matched_pools = split_pools(eligible)
    same_all = [compact_row(row) for row in same_event_rows(high, DEFAULT_CONFIG)]
    same = [compact_row(row) for row in same_event_rows(eligible, DEFAULT_CONFIG)]
    weights = lambda source: float(sum(event["physical_weight_pb"] for event in source))
    eligible_keys = {event["event_key"] for event in eligible}
    excluded_os = [row for row in same_all if row["kind"] == "same_os"
                   and row["source_events"][0] not in eligible_keys]
    report = dict(schema="shift-mixing-hard-scale-conditional-closure-v1",
                  scale_intervals_GeV=[20., 30., 50., None], source_events=len(high),
                  eligible_events=len(eligible), excluded_events=len(excluded),
                  excluded_physical_event_weight_pb=weights(excluded),
                  total_physical_event_weight_pb=weights(high),
                  excluded_real_os_rows=len(excluded_os),
                  excluded_real_both_both_os_rows=sum(row["both_endcaps"] for row in excluded_os),
                  missing_donor_pools={"/".join(key): len(indices) for key, indices in original_pools.items() if len(indices) < 2},
                  source_by_scale=dict(Counter(scale_bin(event) for event in eligible)),
                  real_both_by_scale=dict(Counter(scale_bin(eligible[row["anchor_event"]])
                      for row in same if row["kind"] == "same_os" and row["both_endcaps"])),
                  source_sha256=digest(Path(__file__)), seed=seed, partners=partners,
                  repetitions=repetitions, graphs={}, publication=[],
                  rate_validated=False,
                  limitations=["Unit-area comparison does not independently normalize the model.",
                               "Hard-scale matching is a generator-level diagnostic, not proof of independence.",
                               "Sparse or empty real tails remain untested."])
    ordinary, ordinary_accounting = event_replacement(
        eligible, DEFAULT_CONFIG, partners, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    matched, matched_accounting = [], Counter()
    for scale in ("20to30", "30to50", "50toinf"):
        indices = [index for index, event in enumerate(eligible) if scale_bin(event) == scale]
        if not indices:
            continue
        rows, accounting = event_replacement([eligible[index] for index in indices],
                                             DEFAULT_CONFIG, partners, rng)
        matched_accounting.update(accounting)
        for row in rows:
            row["anchor_event"] = indices[row["anchor_event"]]
            row["donor_event"] = indices[row["donor_event"]]
            matched.append(row)
    for name, rows, donor_pools, accounting in (
            ("ordinary", ordinary, None, ordinary_accounting),
            ("hard_scale_matched", matched, matched_pools, dict(matched_accounting))):
        rows = [compact_row(row) for row in rows]
        path = output / (name + "_mixed.jsonl.gz")
        with gzip.open(path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        results = {}
        for both in (False, True):
            for partition in (None, "test"):
                category = "Both-Both" if both else "inclusive"
                keep = lambda row: (not both or row["both_endcaps"]) and (partition is None or row["partition"] == partition)
                real = [row for row in same if row["kind"] == "same_os" and keep(row)]
                mock = [row for row in rows if keep(row)]
                label = category + "/" + (partition or "all")
                result = shape_closure(real, mock, eligible, repetitions, seed + 1, donor_pools)
                results[label] = result
                pdf = plots / (name + "_" + category.replace("-", "_") + "_" + (partition or "all") + ".pdf")
                draw_shapes(result, pdf, category + " QCD20+", partition or "all folds", name)
                report["publication"].append(dict(path=str(pdf.resolve()), sha256=digest(pdf)))
        report["graphs"][name] = dict(results=results, mixed_file=str(path.resolve()),
                                     mixed_sha256=digest(path), accounting=accounting)
        print("Hard-scale comparison", name, "mixed", len(rows), flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "hard_scale_closure.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--graph", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path, required=True)
    parser.add_argument("--partners", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--replay-mixed", type=Path,
                        help="Reuse audited compact rows from an interrupted diagnostic; seed/partners remain explicit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.plots.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    events = list(iter_pool(args.pool))
    provenance = validate_inventory(args.pool, args.inventory, events)
    pools_for(events)
    closure = same_event_closure(events, DEFAULT_CONFIG)
    if not closure["passed"]:
        raise ValueError("Same-event fit and membership closure failed")
    same = [compact_row(row) for row in same_event_rows(events, DEFAULT_CONFIG)]
    report = dict(schema="shift-mixed-real-unit-area-shape-closure-v1",
                  provenance=provenance, same_event_closure=closure,
                  source_sha256=digest(Path(__file__)), mixing_source_sha256=digest(Path(__file__).with_name("shift_mixing.py")),
                  source_events=len(events), source_muons=sum(len(event["muons"]) for event in events),
                  seed=args.seed, bootstrap=args.bootstrap, graphs={}, publication=[],
                  no_new_detector_events_or_analysis_release=True,
                  same_event_components=dict(Counter(row["truth_reason"] for row in same if row["kind"] == "same_os")),
                  limitations=["Unit-area normalization tests shape only; it does not validate rate.",
                               "Original admitted input subset retains the extraction inventory's missing jobs.",
                               "Finite original-event precision limits both real and mixed shapes.",
                               "Different production, generator, occupancy and event-partition pools never mix.",
                               "Neither same-event nor different-production-vertex muons are proven uncorrelated.",
                               "Late mixing cannot reproduce shared-hit failures or same-event track competition.",
                               "Empirical empty tails have no measured physical uncertainty or validated extrapolation."])
    graphs = []
    for directory in args.graph:
        original = json.loads((directory / "report.json").read_text())
        if original["pool_sha256"] != provenance["pool_sha256"] or not original["same_event_closure"]["passed"]:
            raise ValueError("Existing graph input or fit closure differs")
        rows = read_rows(directory / "event_replacement.jsonl.gz", events)
        graphs.append((directory.name, rows, dict(seed=original["seed"], partners=original["partners"],
                      file=str(directory / "event_replacement.jsonl.gz"),
                      sha256=digest(directory / "event_replacement.jsonl.gz"))))
    print("Input identities and same-event closure checked", flush=True)
    if args.replay_mixed:
        fresh = read_rows(args.replay_mixed, events)
        accounting = dict(replayed_audited_graph=str(args.replay_mixed.resolve()))
    else:
        fresh, accounting = event_replacement(events, DEFAULT_CONFIG, args.partners, np.random.default_rng(args.seed))
        fresh = [compact_row(row) for row in fresh]
    fresh_path = args.output / "fresh_event_replacement.jsonl.gz"
    with gzip.open(fresh_path, "wt") as stream:
        for row in fresh:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    graphs.append(("fresh_" + str(args.partners), fresh, dict(seed=args.seed, partners=args.partners,
                   file=str(fresh_path.resolve()), sha256=digest(fresh_path), accounting=accounting)))
    write_json(args.output / "fresh_graph_receipt.json", dict(
        pool_sha256=provenance["pool_sha256"], pairs=len(fresh),
        source_sha256=report["source_sha256"], mixing_source_sha256=report["mixing_source_sha256"],
        **graphs[-1][2]))
    print("Fresh mixed graph generated:", len(fresh), "pairs", flush=True)
    for name, rows, graph_provenance in graphs:
        results = {}
        for both in (False, True):
            category = "Both-Both" if both else "inclusive"
            for partition in (None, "test"):
                selected = lambda row: (not both or row["both_endcaps"]) and (partition is None or row["partition"] == partition)
                real = [row for row in same if row["kind"] == "same_os" and selected(row)]
                mixed = [row for row in rows if selected(row)]
                label = category + "/" + (partition or "all")
                result = shape_closure(real, mixed, events, args.bootstrap, args.seed + 1)
                result["real_truth_components"] = dict(Counter(row["truth_reason"] for row in real))
                results[label] = result
                pdf = args.plots / (name + "_" + category.replace("-", "_") + "_" + (partition or "all") + ".pdf")
                draw_shapes(result, pdf, category, partition or "all folds", name)
                report["publication"].append(dict(path=str(pdf.resolve()), sha256=digest(pdf)))
                print(name, label, "real", len(real), "mixed", len(mixed), "elapsed", round(time.monotonic() - started), flush=True)
        graph_record = dict(provenance=graph_provenance, results=results)
        if name.startswith("fresh_"):
            diagnostics = {}
            real_both = [row for row in same if row["kind"] == "same_os" and row["both_endcaps"]]
            mixed_both = [row for row in rows if row["both_endcaps"]]
            for component in sorted({row["truth_component"] for row in real_both}):
                selected = [row for row in real_both if row["truth_component"] == component]
                diagnostics[component] = dict(rows=len(selected), weight_pb=sum(row["weight"] for row in selected))
            graph_record["both_both_truth_components"] = diagnostics
            graph_record["both_both_generator_bin_shapes"] = {}
            for stratum in sorted({row["stratum"] for row in real_both}):
                selected_real = [row for row in real_both if row["stratum"] == stratum]
                selected_mixed = [row for row in mixed_both if row["stratum"] == stratum]
                if len(selected_real) < 2 or not selected_mixed:
                    graph_record["both_both_generator_bin_shapes"][stratum] = dict(
                        real_rows=len(selected_real), mixed_rows=len(selected_mixed),
                        available=False, reason="Insufficient same-event or mixed support")
                    continue
                diagnostic = shape_closure(selected_real, selected_mixed, events, args.bootstrap, args.seed + 1)
                diagnostic["available"] = True
                graph_record["both_both_generator_bin_shapes"][stratum] = diagnostic
                pdf = args.plots / ("fresh_Both_Both_" + stratum + ".pdf")
                draw_shapes(diagnostic, pdf, "Both-Both " + stratum, "all folds", name)
                report["publication"].append(dict(path=str(pdf.resolve()), sha256=digest(pdf)))
            selected = [row for row in real_both
                        if row["truth_component"] == "distinct_true_muons_different_parents"]
            if len(selected) >= 2:
                diagnostic = shape_closure(selected, mixed_both, events, args.bootstrap, args.seed + 1)
                diagnostic["scope"] = ("Truth-only diagnostic: different parents do not prove independence; "
                                       "mixed single-muon pool remains unchanged, so this is not a conditional closure test.")
                graph_record["different_parent_real_diagnostic"] = diagnostic
                pdf = args.plots / "fresh_Both_Both_different_parent_diagnostic.pdf"
                draw_shapes(diagnostic, pdf, "Both-Both different-parent diagnostic", "all folds", name)
                report["publication"].append(dict(path=str(pdf.resolve()), sha256=digest(pdf)))
        graph_path = args.output / (name + "_shape_closure.json")
        write_json(graph_path, graph_record)
        report["graphs"][name] = dict(result_path=str(graph_path.resolve()),
                                    result_sha256=digest(graph_path), **graph_provenance)
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / "study_receipt.json", report)
    print("Finished", report["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
