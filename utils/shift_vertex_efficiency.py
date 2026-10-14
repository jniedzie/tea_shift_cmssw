# Code style: AI-code. AI-generated parent-level vertex efficiency accounting.
"""Count fixed-cut efficiencies without dropping zero-candidate GEN parents.

The caller must establish complete reconstruction/classification coverage for
every supplied parent. Missing output files are unknown coverage, not events
with zero candidates. This helper does not establish trigger/readout coverage or
turn a simulated efficiency into an effective luminosity.
"""
import numbers

import numpy as np


def _identities(values, name):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(name + " must have shape (N, 3)")
    if values.dtype != np.dtype("uint64"):
        raise ValueError(name + " must have uint64 dtype; do not pass rounded IDs")
    return values


def _real_vector(values, length, name):
    values = np.asarray(values)
    if values.shape != (length,) or values.dtype.kind not in "iuf":
        raise ValueError(name + " must be a real numeric vector of matching length")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(name + " must contain only finite values")
    return values


def _wilson95(passed, total):
    if total == 0:
        return None
    z = 1.959963984540054
    fraction = passed / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    half_width = z * np.sqrt(
        fraction * (1 - fraction) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _ratio_report(numerator, denominator, weights, independent_trials=True):
    """Ratios of parent sums; counts may be binary or candidate multiplicities."""
    passed = int(numerator.sum())
    total = int(denominator.sum())
    empirical = {
        "passed": passed,
        "total": total,
        "efficiency": passed / total if total else None,
        "wilson95": _wilson95(passed, total) if independent_trials else None,
        "uncertainty_reason": (
            "empty_population" if total == 0 else
            None if independent_trials else
            "multiple_genuine_candidates_share_GEN_parent"
        ),
    }
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_numerator = weights * numerator
        weighted_denominator = weights * denominator
        numerator_sumw = float(weighted_numerator.sum())
        denominator_sumw = float(weighted_denominator.sum())
        denominator_sumw2 = float(np.square(weighted_denominator).sum())
    if not np.isfinite([numerator_sumw, denominator_sumw, denominator_sumw2]).all():
        raise ValueError("Weight totals overflow; rescale all parent weights together")
    weighted = {
        "numerator_sumw": numerator_sumw,
        "denominator_sumw": denominator_sumw,
        "denominator_sumw2": denominator_sumw2,
        "n_eff": 0.0,
        "efficiency": None,
        "standard_error": None,
        "uncertainty_method": "ratio_influence_clustered_by_GEN_parent",
        "undefined_reason": "empty_population" if total == 0 else "zero_total_weight",
    }
    if denominator_sumw > 0:
        # Normalize first to keep the calculation invariant under a common
        # weight scale. One influence per parent preserves candidate correlation.
        scale = float(weighted_denominator.max())
        scaled_denominator = weighted_denominator / scale
        scaled_numerator = weighted_numerator / scale
        scaled_sum = float(scaled_denominator.sum())
        efficiency = float(scaled_numerator.sum() / scaled_sum)
        influence = scaled_numerator - efficiency * scaled_denominator
        standard_error = float(np.sqrt(np.square(influence).sum()) / scaled_sum)
        weighted.update({
            "n_eff": scaled_sum ** 2 / float(np.square(scaled_denominator).sum()),
            "efficiency": efficiency,
            "standard_error": standard_error,
            "undefined_reason": None,
        })
    return {"empirical": empirical, "weighted": weighted}


def summarize_vertex_efficiency(
    event_ids, event_weights, pair_event_ids, pair_labels, scores, cut,
    *, qualified_genuine_mask=None,
):
    """Return JSON-ready efficiencies at the fixed inclusive ``score >= cut``.

    IDs are exact uint64 ``(run, lumi, event)`` rows. The event list contains
    every covered GEN parent, including parents with no reconstructed pairs.
    A parent passes the baseline if it has at least one qualified genuine pair;
    it passes the cut if at least one qualified pair has ``score >= cut``. By
    default all label-1 pairs qualify. An optional boolean mask may restrict
    them, for example to the intended signal decay, without changing raw labels.
    Truth is used only to assess efficiency, never as a classifier input.

    Pair labels are 1 (genuine), 0 (fake), or -1 (unknown). Duplicate candidate
    rows are legitimate alternatives and are retained. Parent IDs must be unique
    and every candidate must match a supplied parent. Parent weights must be
    finite and nonnegative; each pair inherits its parent's weight.

    Weighted standard errors use independent-parent ratio influences, including
    for the pair ratio. They are asymptotic plug-in errors, and vanish when all
    observed weighted trials pass or fail. They do not provide finite-sample
    coverage at those boundaries. Wilson intervals use integer counts only.
    The pair Wilson interval is omitted when any parent has multiple genuine
    candidates: those candidate trials share one parent and are not independent.
    No Wilson interval is applied to sum weights or an effective event count.
    """
    event_ids = _identities(event_ids, "event_ids")
    pair_event_ids = _identities(pair_event_ids, "pair_event_ids")
    n_events = len(event_ids)
    n_pairs = len(pair_event_ids)
    weights = _real_vector(event_weights, n_events, "event_weights")
    if np.any(weights < 0):
        raise ValueError("event_weights must be nonnegative")
    scores = _real_vector(scores, n_pairs, "scores")
    labels = np.asarray(pair_labels)
    if labels.shape != (n_pairs,) or labels.dtype.kind not in "iu":
        raise ValueError("pair_labels must be an integer vector of matching length")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("pair_labels must be -1 (unknown), 0 (fake), or 1 (genuine)")
    qualified = labels == 1
    if qualified_genuine_mask is not None:
        qualified = np.asarray(qualified_genuine_mask)
        if qualified.shape != (n_pairs,) or qualified.dtype.kind != "b":
            raise ValueError("qualified_genuine_mask must be an aligned boolean vector")
        if np.any(qualified & (labels != 1)):
            raise ValueError("qualified_genuine_mask must be a subset of label-1 pairs")
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, numbers.Real):
        raise ValueError("cut must be a finite real scalar")
    cut = float(cut)
    if not np.isfinite(cut):
        raise ValueError("cut must be a finite real scalar")

    parent_index = {}
    for index, identity in enumerate(event_ids):
        key = tuple(identity)
        if key in parent_index:
            raise ValueError("Duplicate GEN parent identity: " + str(key))
        parent_index[key] = index
    pair_parents = np.empty(n_pairs, dtype=np.int64)
    for index, identity in enumerate(pair_event_ids):
        key = tuple(identity)
        if key not in parent_index:
            raise ValueError("Orphan candidate parent identity: " + str(key))
        pair_parents[index] = parent_index[key]

    passed = scores >= cut
    parent_pair_counts = np.zeros(n_events, dtype=np.int64)
    np.add.at(parent_pair_counts, pair_parents, 1)
    classes = {}
    for name, label in (("genuine", 1), ("fake", 0), ("unknown", -1)):
        mask = labels == label
        selected_mask = mask & passed
        counts = np.zeros(n_events, dtype=np.int64)
        selected_counts = np.zeros(n_events, dtype=np.int64)
        np.add.at(counts, pair_parents[mask], 1)
        np.add.at(selected_counts, pair_parents[selected_mask], 1)
        classes[name] = {
            "pairs": int(mask.sum()),
            "selected_pairs": int(selected_mask.sum()),
            "parents": int(np.count_nonzero(counts)),
            "selected_parents": int(np.count_nonzero(selected_counts)),
        }
    genuine_counts = np.zeros(n_events, dtype=np.int64)
    genuine_passed_counts = np.zeros(n_events, dtype=np.int64)
    np.add.at(genuine_counts, pair_parents[qualified], 1)
    np.add.at(genuine_passed_counts, pair_parents[qualified & passed], 1)
    classes["qualified_genuine"] = {
        "pairs": int(qualified.sum()),
        "selected_pairs": int(np.sum(qualified & passed)),
        "parents": int(np.count_nonzero(genuine_counts)),
        "selected_parents": int(np.count_nonzero(genuine_passed_counts)),
    }

    baseline = (genuine_counts > 0).astype(np.int64)
    selected = (genuine_passed_counts > 0).astype(np.int64)
    all_parents = np.ones(n_events, dtype=np.int64)
    return {
        "schema_version": 1,
        "cut": cut,
        "cut_comparison": ">=",
        "genuine_denominator": "qualified label-1 pairs; all label-1 pairs if no qualification mask was supplied",
        "counts": {
            "GEN_parents": n_events,
            "pairs": n_pairs,
            "parents_without_candidates": int(np.sum(parent_pair_counts == 0)),
            "parents_with_genuine_pairs": int(baseline.sum()),
            "parents_selected_by_genuine_pair": int(selected.sum()),
            "raw_genuine_pairs": int(np.sum(labels == 1)),
            "qualified_genuine_pairs": int(qualified.sum()),
            "parents_with_qualified_genuine_pairs": int(baseline.sum()),
        },
        "classes": classes,
        "full_parent_baseline": _ratio_report(baseline, all_parents, weights),
        "full_parent_selected": _ratio_report(selected, all_parents, weights),
        "conditional_event_retention": _ratio_report(selected, baseline, weights),
        "genuine_pair_retention": _ratio_report(
            genuine_passed_counts, genuine_counts, weights,
            independent_trials=bool(np.all(genuine_counts <= 1)),
        ),
        "limitations": [
            "Supplied GEN parents require complete reconstruction/classification coverage; missing outputs are not zero-candidate events.",
            "GEN-parent efficiencies are simulation accounting, not validated trigger/readout efficiency or effective luminosity.",
            "Only qualified genuine candidates select a parent; raw genuine, fake and unknown counts remain separate.",
            "Pair Wilson intervals are omitted for multiple genuine candidates per parent; weighted standard errors cluster by parent.",
            "Weighted standard errors are asymptotic and omit finite-sample uncertainty at all-pass/all-fail boundaries.",
        ],
    }
