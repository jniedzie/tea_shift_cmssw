#!/usr/bin/env python3
# Code style: AI-code. AI-generated provisional event-counting sensitivity diagnostic.
"""Screen physical signal points with an explicit hypothetical luminosity.

This is an MC-only, known-background, no-nuisance expected 90% CLs check.
It does not infer exposure, use collision data, optimize a selection, or publish
a physical exclusion. Input files and full GEN weight denominators are explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


LUMINOSITY_ROLES = {.03: "Pessimistic", .3: "Baseline", 3.: "Optimistic ceiling"}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def choose_candidate(masses, opposite_sign, valid, hypothesis, fraction):
    """Pick one data-available OS candidate; ties use the stored row order."""
    if not len(masses) == len(opposite_sign) == len(valid):
        raise ValueError("Candidate columns have different lengths")
    candidates = [(abs(float(mass) - hypothesis), index)
                  for index, mass in enumerate(masses)
                  if opposite_sign[index] == 1 and valid[index] == 1
                  and math.isfinite(float(mass)) and mass > 0]
    if not candidates:
        return None
    distance, index = min(candidates)
    return index if distance < fraction * hypothesis else None


def log_cls(signal, background, observed):
    """Exact integer-count CLs for an exactly known nonnegative background."""
    from scipy.stats import poisson
    if signal < 0 or background < 0 or not math.isfinite(signal + background):
        raise ValueError("Poisson means must be finite and nonnegative")
    if isinstance(observed, bool) or int(observed) != observed or observed < 0:
        raise ValueError("An exact Poisson test needs an integer observation")
    return float(poisson.logcdf(int(observed), signal + background)
                 - poisson.logcdf(int(observed), background))


def median_expected_limit(background, alpha=0.1):
    """Median background-only exact one-bin upper limit, without nuisances."""
    from scipy.optimize import brentq
    from scipy.stats import poisson
    if background < 0 or not math.isfinite(background) or not 0 < alpha < 1:
        raise ValueError("Invalid background or confidence level")
    count = int(poisson.ppf(0.5, background))
    target = math.log(alpha)
    upper = max(10., 10. * math.sqrt(background + 1.))
    while log_cls(upper, background, count) > target:
        upper *= 2.
    limit = brentq(lambda signal: log_cls(signal, background, count) - target,
                   0., upper)
    return dict(observed_median=count, signal_events_90=limit,
                construction="exact integer-Poisson known-background CLs")


def count_sample(sample, masses, fraction):
    """Read only reconstructed selection, identity and weight branches."""
    import ROOT
    branches = ["run", "luminosityBlock", "event", "nShiftDimuonVertex", "ShiftDimuonVertex_mass",
                "ShiftDimuonVertex_isOS", "ShiftDimuonVertex_dcaValid", sample["weight_branch"]]
    result = dict(events=0, sumw=0., sumw2=0., maximum_weight=0., sources=[],
                  windows={str(mass): dict(events=0, sumw=0., sumw2=0.) for mass in masses})
    identities = set()
    identity_hash = hashlib.sha256()
    for path in sample["files"]:
        path = Path(path)
        before = path.stat()
        # Inputs are mounted local files. The TFile constructor avoids an
        # automatic XRootD redirect for EOS paths in disconnected sessions.
        root_file = ROOT.TFile(str(path), "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError("Unreadable Nano file: " + str(path))
        try:
            tree = root_file.Get("Events")
            if not tree:
                raise RuntimeError("Missing Events tree: " + str(path))
            for branch in branches:
                if not tree.GetBranch(branch):
                    raise RuntimeError("Missing selection/weight branch: " + branch)
            tree.SetBranchStatus("*", 0)
            for branch in branches:
                tree.SetBranchStatus(branch, 1)
            for event in tree:
                identity = (int(event.run), int(event.luminosityBlock), int(event.event))
                if identity in identities:
                    raise ValueError("Duplicate event identity within sample: " + str(identity))
                identities.add(identity)
                identity_hash.update(("%d:%d:%d\n" % identity).encode())
                weight = float(getattr(event, sample["weight_branch"]))
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("This preliminary screen requires finite positive event weights")
                result["events"] += 1
                result["sumw"] += weight
                result["sumw2"] += weight * weight
                result["maximum_weight"] = max(result["maximum_weight"], weight)
                for mass in masses:
                    candidate = choose_candidate(event.ShiftDimuonVertex_mass,
                        event.ShiftDimuonVertex_isOS, event.ShiftDimuonVertex_dcaValid,
                        mass, fraction)
                    if candidate is not None:
                        row = result["windows"][str(mass)]
                        row["events"] += 1
                        row["sumw"] += weight
                        row["sumw2"] += weight * weight
        finally:
            root_file.Close()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Nano input changed while counting: " + str(path))
        result["sources"].append(dict(path=str(path), bytes=after.st_size,
                                      mtime_ns=after.st_mtime_ns))
    result["event_ids_sha256"] = identity_hash.hexdigest()
    for row in result["windows"].values():
        row["ess"] = row["sumw"] ** 2 / row["sumw2"] if row["sumw2"] else 0.
    return result


def sample_prediction(sample, counts, mass, luminosity_fb):
    denominator = float(sample["full_gen_sumw"])
    cross_section = float(sample["selected_cross_section_pb"])
    if denominator <= 0 or cross_section <= 0:
        raise ValueError("Require positive full GEN denominator and physical cross section")
    window = counts["windows"][str(mass)]
    scale = luminosity_fb * 1000. * cross_section / denominator
    return dict(events=scale * window["sumw"],
                generated_dimuon_events=luminosity_fb * 1000. * cross_section,
                selection_acceptance=window["sumw"] / denominator,
                mc_error_events=scale * math.sqrt(window["sumw2"]),
                selected_mc_events=window["events"], selected_mc_ess=window["ess"],
                processed_mc_events=counts["events"], full_gen_sumw=denominator)


def empty_mc_planning_floor(sample, counts, luminosity_fb, alpha=.1):
    """Conditional single-component binomial upper rate for constant weights.

    This assumes the processed MC events are independent draws of the sampled
    population. It is a planning floor, not a joint confidence interval or a
    replacement for finite-template nuisance treatment. Variable-weight empty
    cells remain unresolved.
    """
    events, sumw, sumw2 = counts["events"], counts.get("sumw", 0.), counts.get("sumw2", 0.)
    if events <= 0 or sumw2 <= 0 or not math.isclose(sumw * sumw / sumw2, events, rel_tol=1.e-8):
        return None
    probability_upper = -math.expm1(math.log(alpha) / events)
    return luminosity_fb * 1000. * sample["selected_cross_section_pb"] * sumw * probability_upper / sample["full_gen_sumw"]


def screen(inputs, counts, luminosities, signal_only=False):
    """Keep zero-stat signal/background cells unknown, with no invented b=0."""
    masses = inputs["masses_gev"]
    scenarios = []
    for luminosity in luminosities:
        backgrounds = {}
        for mass in ([] if signal_only else masses):
            components = {sample["name"]: sample_prediction(sample, counts[sample["name"]], mass, luminosity)
                          for sample in inputs["backgrounds"]}
            total = sum(row["events"] for row in components.values())
            error = math.sqrt(sum(row["mc_error_events"] ** 2 for row in components.values()))
            zero_names = [name for name, row in components.items() if row["selected_mc_events"] == 0]
            floors = {sample["name"]: empty_mc_planning_floor(sample, counts[sample["name"]], luminosity)
                      for sample in inputs["backgrounds"] if sample["name"] in zero_names}
            unresolved = [name for name, floor in floors.items() if floor is None]
            screening_background = total + sum(floor for floor in floors.values() if floor is not None)
            backgrounds[str(mass)] = dict(events=total, mc_error_events=error,
                selected_mc_ess=total * total / (error * error) if error else 0.,
                zero_mc_components=zero_names, empty_mc_upper_planning_floors=floors,
                unresolved_variable_weight_components=unresolved,
                screen_background_events=screening_background,
                supported=total > 0 and bool(components) and not unresolved, components=components)
        points = []
        for sample in inputs["signals"]:
            mass = sample["mass_gev"]
            signal = sample_prediction(sample, counts[sample["name"]], mass, luminosity)
            background = backgrounds.get(str(mass),
                dict(screen_background_events=None, events=None, supported=False))
            supported = signal["selected_mc_events"] > 0 and background["supported"]
            point = dict(name=sample["name"], mass_gev=mass, epsilon=sample["epsilon"],
                         signal=signal, background=background["screen_background_events"],
                         background_nominal=background["events"], supported=supported,
                         partial=sample.get("partial", False),
                         status="nominal screening estimate" if supported else "unknown finite-MC acceptance/background")
            if signal_only:
                point["status"] = ("partial signal contribution" if point["partial"] else "signal yield estimate")
                if signal["selected_mc_events"] == 0:
                    point["status"] = "unknown acceptance: no selected MC events"
            if supported:
                limit = median_expected_limit(background["screen_background_events"])
                point.update(limit)
                point["log_cls"] = log_cls(signal["events"], background["screen_background_events"], limit["observed_median"])
                point["signal_over_s90"] = signal["events"] / limit["signal_events_90"]
            if sample.get("unit_gen_weights"):
                missing = max(0., sample["full_gen_sumw"] - counts[sample["name"]]["events"])
                point["unprocessed_signal_yield_maximum"] = (luminosity * 1000. *
                    sample["selected_cross_section_pb"] * missing / sample["full_gen_sumw"])
            points.append(point)
        scenarios.append(dict(luminosity_fb=luminosity, luminosity_role=LUMINOSITY_ROLES.get(luminosity, "Hypothetical"),
                              backgrounds=backgrounds, points=points))
    return dict(schema="shift-baseline-sensitivity-screen-v1", scenarios=scenarios,
        selection=dict(opposite_sign=True, vertex_validity="dcaValid == 1", mass_definition="ShiftDimuonVertex_mass",
                       candidate_arbitration="closest mass to hypothesis, then stored candidate index",
                       window_half_width_fraction=inputs["window_half_width_fraction"], bdt_selection=False),
        assumptions=(["signal yields only; no background estimate or exclusion evaluated"] if signal_only else []) +
                    ["hypothetical effective target luminosity, no empirical exposure calibration",
                     "L_eff covers unmodeled halo/exposure/recording/data-MC losses; simulated reconstructed mass acceptance is applied once",
                     "fixed no-pileup/Fake2-HLT replay is a detector scenario, not final recording acceptance",
                     "pure-DY native-Pythia rate pilot; physical rate closure remains open",
                     "MC yield errors reported separately" if signal_only else "known-background, no-nuisance exact median 90% CLs screening; MC errors reported separately",
                     "zero selected signal MC events leave acceptance unresolved" if signal_only else "empty constant-weight MC components use conditional single-component90% binomial upper planning floors; not a joint confidence bound",
                     "zero finite-MC cells are unknown; zero components do not establish zero physical rates",
                     "all selected sums retain full GEN denominators; missing detector outputs are not renormalized",
                     "mass hypotheses share events; values are pointwise, with no joint-likelihood claim"],
        counts=counts)


def draw(report, output, signal_yields=False, generated_yields=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    scenarios = report["scenarios"]
    presentation = report.get('presentation', {})
    width = 12. if presentation else 5.0 * len(scenarios)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(width, 7.5 if presentation else 6.3), squeeze=False)
    def supported(point):
        if generated_yields:
            return True
        return point["signal"]["selected_mc_events"] > 0 if signal_yields else point["supported"]

    def value(point):
        if generated_yields:
            return math.log10(point["signal"]["generated_dimuon_events"])
        return math.log10(point["signal"]["events"] if signal_yields else point["signal_over_s90"])

    values = [value(p) for s in scenarios for p in s["points"] if supported(p)]
    norm = Normalize(vmin=min(-1., min(values, default=-1.)), vmax=max(0., max(values, default=0.)))
    artist = None
    for axis, scenario in zip(axes[0], scenarios):
        points = scenario["points"]
        ready = [p for p in points if supported(p)]
        unknown = [p for p in points if not supported(p)]
        if ready:
            artist = axis.scatter([p["mass_gev"] for p in ready], [p["epsilon"] for p in ready],
                c=[value(p) for p in ready], cmap="viridis", norm=norm,
                s=100, edgecolors="black", linewidths=.5)
            partial = [p for p in points if p["partial"] and not generated_yields]
            if partial:
                axis.scatter([p["mass_gev"] for p in partial], [p["epsilon"] for p in partial],
                    facecolors="none", edgecolors="red", s=145, linewidths=1., label="Partial detector coverage")
        if unknown:
            axis.scatter([p["mass_gev"] for p in unknown], [p["epsilon"] for p in unknown],
                         marker="x", c="gray", s=65, label="Unknown: no selected MC")
        if presentation.get('show_mc_support'):
            sparse = [p for p in ready if p['signal']['selected_mc_events'] < 10]
            axis.scatter([p['mass_gev'] for p in sparse], [p['epsilon'] for p in sparse],
                facecolors='none', edgecolors='gray', s=195,
                label='Outer ring: fewer than 10 selected MC events')
        if presentation.get('annotate_yields'):
            for point in points:
                upper = point.get('zero_selected_one_sided95_upper_events')
                text = f'<{upper:.2g}*' if upper is not None else f'{point["signal"]["events"]:.2g}'
                axis.annotate(text, (point['mass_gev'], point['epsilon']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
            axis.set_xlim(min(p['mass_gev'] for p in points)*.87, max(p['mass_gev'] for p in points)*1.14)
            axis.set_ylim(min(p['epsilon'] for p in points)*.55, max(p['epsilon'] for p in points)*3.)
        luminosity_symbol = presentation.get('luminosity_symbol', 'L_{eff}')
        axis.set(xscale="log", yscale="log", xlabel=r"$m_{A'}$ [GeV]",
                 title=f"{scenario['luminosity_role']}\n${luminosity_symbol}$ = {scenario['luminosity_fb']:g} fb$^{{-1}}$")
        axis.grid(alpha=.2)
        axis.spines[["right", "top"]].set_visible(False)
        if unknown or presentation.get('show_mc_support') or any(p["partial"] for p in ready) and not generated_yields:
            axis.legend(fontsize=8, loc="lower left")
    axes[0, 0].set_ylabel(r"$\epsilon$")
    title = "SHIFT baseline signal yield — MC scenario, no BDT" if signal_yields else "SHIFT baseline sensitivity screen — MC scenario, no BDT"
    if generated_yields:
        title = "SHIFT dark-photon dimuon yield before detector selection — DY model"
    title = presentation.get('title', title)
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(left=.07, right=.88, bottom=.25, top=.86, wspace=.25)
    if artist is not None:
        colorbar = fig.colorbar(artist, cax=fig.add_axes([.90, .29, .016, .54]))
        colorbar.set_label(r"$\log_{10}(N)$, generated dimuon events" if generated_yields else
                          r"$\log_{10}(S)$, selected signal events" if signal_yields else
                          r"$\log_{10}(S/S_{90})$; 0 is the nominal screening boundary")
    width = 100. * report.get("selection", {}).get("window_half_width_fraction", 0.)
    selection_text = ("Production cross section times effective luminosity; dimuon branching fraction included once."
                      if generated_yields else f"One OS valid candidate per event; reconstructed mass within ±{width:g}% of the hypothesis.")
    selection_text = presentation.get('selection_text', selection_text)
    fig.text(.5, .13, selection_text, ha="center", fontsize=9)
    description = ("No detector, reconstruction, mass-window or BDT selection applied." if generated_yields else
                   "DY selected cross sections already include the dimuon branching fraction; simulated selection acceptance is applied once."
                   if signal_yields else "Known-background screen; empty constant-weight MC components use 90% upper planning floors. No fitted nuisances.")
    description = presentation.get('description', description)
    fig.text(.5, .09, description, ha="center", fontsize=9)
    fig.text(.5, .05, presentation.get('footer', "Native-Pythia pure-DY rate and no-pileup/Fake2-HLT detector pilots. Points only; no physical exclusion contour."), ha="center", fontsize=9)
    if presentation.get('annotate_yields'):
        fig.text(.5, .02, 'Labels: expected events. * Exact one-sided 95% MC upper bound for zero selected events; finite-MC intervals in numerical receipt.', ha='center', fontsize=8)
    fig.savefig(output, metadata={"Title": presentation.get('metadata_title', "SHIFT provisional baseline sensitivity screen")})
    plt.close(fig)


def _selected_signal_yield_rows(response_report, theory_manifest, lumi_fb):
    """Combine full-parent event-OR response with already physical dimuon rates.

    This pure calculation is separate from the older mass-window screen.
    Constant-weight parent intervals use integer trials, never effective counts.
    Variable-weight nominal responses remain usable, but finite-sample interval
    coverage is not inferred from their asymptotic standard error.
    """
    if not math.isfinite(lumi_fb) or lumi_fb <= 0:
        raise ValueError("Hypothetical luminosity must be finite and positive")
    if (theory_manifest.get("schema") != "shift-vertex-theory-yield-inputs-v1"
            or theory_manifest.get("complete") is not True):
        raise ValueError("Require a complete explicit theory yield manifest")
    for key in ("branching_included_in_sigma", "epsilon_correction_already_in_sigma"):
        if theory_manifest.get(key) is not True:
            raise ValueError("Physical selected rate must already include " + key)
    if response_report.get("selection_uses_signal") is not False:
        raise ValueError("Signal response cannot select or tune the classifier")
    threshold = float(response_report["threshold"])
    if not math.isfinite(threshold):
        raise ValueError("Nonfinite fixed classifier cut")
    if "fixed_classifier_cut" in theory_manifest and theory_manifest["fixed_classifier_cut"] != threshold:
        raise ValueError("Theory and response classifier cuts differ")
    response_points, theory_points = response_report["signal_points"], theory_manifest["points"]
    if not response_points or set(response_points) != set(theory_points):
        raise ValueError("Theory and complete response point populations differ")
    rows = {}
    for point, item in response_points.items():
        rate = theory_points[point]; response = item["response"]
        if response["cut"] != threshold or response["cut_comparison"] != ">=":
            raise ValueError("Response changed the frozen inclusive cut: " + point)
        for key in ("mass_gev", "proper_ctau_mm", "native_sumw", "native_sumw2"):
            actual, expected = float(item[key]), float(rate[key])
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.):
                raise ValueError("Response/theory mismatch: " + point + " " + key)
        if item["full_GEN_events"] != rate["full_GEN_events"]:
            raise ValueError("Full GEN population count differs: " + point)
        identity_hash = item["full_GEN_event_ids_sha256"]
        if len(identity_hash) != 64 or identity_hash != rate["full_GEN_event_ids_sha256"]:
            raise ValueError("Full GEN identities differ: " + point)
        if item["gen_audit"] != rate["gen_audit"]:
            raise ValueError("GEN audit authority differs: " + point)
        empirical = response["full_parent_selected"]["empirical"]
        weighted = response["full_parent_selected"]["weighted"]
        total, selected = empirical["total"], empirical["passed"]
        if (isinstance(total, bool) or isinstance(selected, bool)
                or int(total) != total or int(selected) != selected
                or total <= 0 or not 0 <= selected <= total):
            raise ValueError("Invalid full-parent event-OR counts: " + point)
        if total != item["full_GEN_events"] or total != response["counts"]["GEN_parents"]:
            raise ValueError("Incomplete full GEN response denominator: " + point)
        if selected != response["counts"]["parents_selected_by_genuine_pair"]:
            raise ValueError("Selected parent accounting differs: " + point)
        sumw, sumw2 = float(item["native_sumw"]), float(item["native_sumw2"])
        numerator = float(weighted["numerator_sumw"])
        if (not all(math.isfinite(v) for v in (sumw, sumw2, numerator))
                or sumw <= 0 or sumw2 <= 0 or not 0 <= numerator <= sumw):
            raise ValueError("Invalid nonnegative native GEN weight sums: " + point)
        if (not math.isclose(sumw, weighted["denominator_sumw"], rel_tol=1e-12)
                or not math.isclose(sumw2, weighted["denominator_sumw2"], rel_tol=1e-12)):
            raise ValueError("Weighted response lost the full GEN denominator: " + point)
        efficiency = numerator / sumw
        if not math.isclose(efficiency, weighted["efficiency"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Stored response ratio is inconsistent: " + point)
        sigma = float(rate["physical_selected_sigma_pb"])
        sigma_error = float(rate["physical_selected_sigma_stat_error_pb"])
        epsilon = float(rate["epsilon"])
        if (not all(math.isfinite(v) for v in (sigma, sigma_error, epsilon))
                or sigma <= 0 or sigma_error < 0 or epsilon <= 0
                or float(item["mass_gev"]) <= 0 or float(item["proper_ctau_mm"]) < 0):
            raise ValueError("Invalid physical point or selected cross section: " + point)
        normalization = lumi_fb * 1000. * sigma
        constant_weights = math.isclose(sumw * sumw / sumw2, total, rel_tol=1e-10)
        interval = None; zero_upper = None
        if constant_weights:
            if not math.isclose(efficiency, selected/total, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError("Constant-weight response differs from event counts: " + point)
            if selected == 0:
                zero_upper = -math.expm1(math.log(.05) / total)
                interval = [0., normalization * zero_upper]
                method = "exact one-sided95 binomial upper bound for zero selected parents"
            else:
                from shift_vertex_efficiency import _wilson95
                interval = [normalization * value for value in _wilson95(selected, total)]
                method = "Wilson95 independent full-GEN parent trials; constant native weights"
        else:
            method = "finite-sample weighted interval unresolved; parent-cluster asymptotic SE separate"
        response_error = weighted.get("standard_error")
        if response_error is not None and (not math.isfinite(response_error) or response_error < 0):
            raise ValueError("Invalid weighted response standard error: " + point)
        rows[point] = dict(point=point, mass_gev=float(item["mass_gev"]),
            proper_ctau_mm=float(item["proper_ctau_mm"]), epsilon=epsilon,
            full_GEN_events=total, selected_qualified_signal_parents=selected,
            full_GEN_event_ids_sha256=identity_hash, native_sumw=sumw, native_sumw2=sumw2,
            selected_native_sumw=numerator, full_parent_response=efficiency,
            physical_selected_sigma_pb=sigma, physical_selected_sigma_stat_error_pb=sigma_error,
            produced_dimuons=normalization, expected_selected_events=normalization*efficiency,
            selected_event_mc_interval95=interval, mc_interval_method=method,
            response_asymptotic_standard_error_events=None if response_error is None else normalization*response_error,
            rate_mc_standard_error_events=lumi_fb*1000.*sigma_error*efficiency,
            zero_count_one_sided95_efficiency_upper=zero_upper,
            zero_observed_selected=selected == 0, sparse_selected_mc=selected < 5,
            constant_native_weights=constant_weights, gen_audit=rate["gen_audit"])
    if "full_GEN_events" in theory_manifest and sum(row["full_GEN_events"] for row in rows.values()) != theory_manifest["full_GEN_events"]:
        raise ValueError("Theory total full GEN population differs")
    return rows


def _verify_selected_yield_references(value, checked):
    """Verify explicit metadata/cache authorities without discovering inputs."""
    from shift_vertex_signal_scores import verify
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            key = (value["path"], value["sha256"])
            if key not in checked:
                verify(value); checked.add(key)
        else:
            for child in value.values():
                _verify_selected_yield_references(child, checked)
    elif isinstance(value, list):
        for child in value:
            _verify_selected_yield_references(child, checked)


def render_selected_signal_yields(response_report, theory_manifest, outputdir, lumi_fb=3):
    """Write fixed-cut full-GEN selected yields from two sealed JSON paths.

    ``signal_points`` and theory ``points`` are dictionaries with identical
    keys. Both carry mass, proper ctau, full GEN count, native sumw/sumw2,
    GEN audit reference and SHA256 of ordered little-endian uint64 parent IDs.
    The physical selected cross section already includes epsilon and BR_mumu.
    This adds no mass window and makes no recording-acceptance or limit claim.
    """
    from shift_vertex_signal_scores import reference, verify
    response_path, theory_path, output = Path(response_report), Path(theory_manifest), Path(outputdir)
    if output.exists():
        raise ValueError("Refusing existing selected-yield output: " + str(output))
    response_ref, theory_ref = reference(response_path), reference(theory_path)
    response, theory = json.loads(response_path.read_text()), json.loads(theory_path.read_text())
    checked = set()
    for value in (response, theory):
        _verify_selected_yield_references(value, checked)
    for key in ("selection", "signal_cache"):
        if key not in response or not {"path", "sha256"}.issubset(response[key]):
            raise ValueError("Response must pin its " + key)
        theory_references = theory.get("references", {})
        # Theory-only manifests may use references for a bibliography rather
        # than cache bindings; their exact GEN authorities remain mandatory.
        theory_ref_for_key = theory_references.get(key) if isinstance(theory_references, dict) else None
        if theory_ref_for_key is not None and theory_ref_for_key != response[key]:
            raise ValueError("Theory and response have different " + key)
    points = _selected_signal_yield_rows(response, theory, float(lumi_fb))
    verify(response_ref); verify(theory_ref)
    report = dict(schema="shift-selected-signal-yields-v1", complete=True,
        luminosity_fb_inverse=float(lumi_fb), luminosity_pb_inverse=float(lumi_fb)*1000.,
        formula="L_pb_inverse * physical_selected_sigma_pb * selected_native_parent_sumw/full_GEN_sumw",
        response=response_ref, theory=theory_ref, selection=response["selection"],
        signal_cache=response["signal_cache"], threshold=response["threshold"], points=points,
        full_GEN_events=sum(row["full_GEN_events"] for row in points.values()),
        selected_qualified_signal_parents=sum(row["selected_qualified_signal_parents"] for row in points.values()),
        point_count=len(points), zero_selected_mc_points=sum(row["zero_observed_selected"] for row in points.values()),
        sparse_selected_mc_points=sum(row["sparse_selected_mc"] for row in points.values()),
        positive_selected_mc_points=sum(not row["zero_observed_selected"] for row in points.values()),
        verified_reference_count=len(checked), extra_branching_or_epsilon_factor_applied=False,
        added_mass_window=False, event_selection="OR of qualified signal pairs above the frozen inclusive score cut",
        recording_acceptance_established=False, production_rate_validated=False,
        background_neglected=True, exclusion_or_sensitivity_inferred=False,
        assumptions=["Hypothetical target luminosity; pure on-shell DY LO production model remains provisional",
            "Physical dimuon cross section includes branching and coupling corrections once",
            "Truth-matched known-signal nominal yield; unknown and known-fake passing candidates are excluded",
            "Full GEN parents retained, including parents with no reconstructed candidate",
            "No added mass window; actual recording/readout and data/MC corrections unresolved",
            "Intervals show finite response MC only; rate MC error separate, no production-model/PDF/scale uncertainty supplied",
            "Three events is only a yield reference; no contour, interpolation or background calculation"],
        source=reference(__file__))
    output.mkdir(parents=True, exist_ok=False)
    _draw_selected_signal_yields(report, output/"signal_mass_lifetime.pdf")
    report["figure"] = reference(output/"signal_mass_lifetime.pdf")
    with (output/"expected_signal_yields.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    return report


def _draw_selected_signal_yields(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    rows = list(report["points"].values())
    nonzero = [row for row in rows if row["expected_selected_events"] > 0]
    zeros = [row for row in rows if row["zero_observed_selected"]]
    values = [row["expected_selected_events"] for row in nonzero]
    lower, upper = min(values, default=1e-3), max(values, default=1.)
    if lower == upper:
        lower /= 2.; upper *= 2.
    norm = LogNorm(vmin=lower, vmax=upper)
    with PdfPages(output) as pdf:
        fig, axis = plt.subplots(figsize=(10.5, 7.5))
        if nonzero:
            dots = axis.scatter([r["mass_gev"] for r in nonzero], [r["epsilon"] for r in nonzero],
                c=values, norm=norm, cmap="viridis", s=85, edgecolors="black", linewidths=.5)
            fig.colorbar(dots, ax=axis, label="Expected qualified signal events (log color)")
        sparse = [r for r in nonzero if r["sparse_selected_mc"]]
        if sparse:
            axis.scatter([r["mass_gev"] for r in sparse], [r["epsilon"] for r in sparse],
                         s=155, facecolors="none", edgecolors="darkorange", linewidths=1.5)
        if zeros:
            axis.scatter([r["mass_gev"] for r in zeros], [r["epsilon"] for r in zeros],
                         marker="x", c="gray", s=70)
        handles = [Line2D([],[],marker="o",color="black",linestyle="none",label="Positive nominal yield"),
                   Line2D([],[],marker="o",markerfacecolor="none",color="darkorange",linestyle="none",label="1–4 selected MC parents: sparse"),
                   Line2D([],[],marker="x",color="gray",linestyle="none",label="Zero selected MC: finite upper bound, not zero sensitivity"),
                   Line2D([],[],color="none",label="3 expected events: reference only; no contour inferred")]
        axis.legend(handles=handles, fontsize=8, loc="best")
        axis.set(xscale="log",yscale="log",xlabel=r"$m_{A'}$ [GeV]",ylabel=r"$\epsilon$",
                 title=f"Frozen-classifier signal yield at hypothetical {report['luminosity_fb_inverse']:g} fb$^{{-1}}$")
        axis.grid(alpha=.2)
        fig.text(.5,.055,"Pure-DY LO rate model: provisional. Truth-matched event-OR response; full GEN denominator.\n"
                 "No added mass window; recording unresolved; backgrounds neglected. Points only, no exclusion claim.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.1,1,1));pdf.savefig(fig);plt.close(fig)
        ordered = sorted(rows, key=lambda r:(r["mass_gev"],r["epsilon"],r["point"]))
        for offset in range(0,len(ordered),24):
            chunk=ordered[offset:offset+24]
            fig, axis=plt.subplots(figsize=(11,max(5,len(chunk)*.29+2)))
            for i,row in enumerate(chunk):
                value=row["expected_selected_events"]; interval=row["selected_event_mc_interval95"]
                if row["zero_observed_selected"]:
                    if interval is not None:
                        axis.scatter(interval[1],i,marker="<",c="gray",s=45)
                    else:
                        axis.text(.01,i,"zero; weighted upper bound unresolved",transform=axis.get_yaxis_transform(),fontsize=7)
                elif interval is not None:
                    axis.errorbar(value,i,xerr=[[max(0.,value-interval[0])],[max(0.,interval[1]-value)]],
                                  fmt="o",color="darkorange" if row["sparse_selected_mc"] else "#2468a0",capsize=2)
                else:
                    axis.scatter(value,i,marker="o",c="darkorange")
            axis.set_yticks(range(len(chunk)),[f"{r['point']}  ({r['selected_qualified_signal_parents']}/{r['full_GEN_events']} MC)" for r in chunk],fontsize=7)
            axis.invert_yaxis();axis.set_xscale("log");axis.set_xlabel("Expected qualified signal events; finite-response MC95% intervals")
            axis.set_title("Fixed physical points and frozen classifier cut; no point interpolation")
            axis.grid(axis="x",alpha=.25)
            fig.text(.5,.025,"Zero-MC triangles mark one-sided95% upper yields. Rate MC error is separate; model uncertainties not included.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.06,1,1));pdf.savefig(fig);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--luminosity-fb", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=Path, help="Reuse a previously extracted, input-hash-matched count receipt")
    parser.add_argument("--signal-yields-only", action="store_true", help="Skip backgrounds and exclusion screening")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in args.luminosity_fb):
        parser.error("Luminosities must be positive and explicit")
    inputs = json.loads(args.inputs.read_text())
    if inputs.get("schema") != "shift-baseline-sensitivity-inputs-v1":
        parser.error("Unsupported explicit input schema")
    input_hash = digest(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.counts:
        receipt = json.loads(args.counts.read_text())
        if receipt["inputs_sha256"] != input_hash:
            raise ValueError("Reused count receipt belongs to different inputs")
        counts = receipt["counts"]
        for row in counts.values():
            for source in row["sources"]:
                current = Path(source["path"]).stat()
                if (current.st_size, current.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
                    raise ValueError("Reused count input changed: " + source["path"])
    else:
        counts = {}
    samples = inputs["signals"] if args.signal_yields_only else inputs["backgrounds"] + inputs["signals"]
    for sample in samples:
        if sample["name"] in counts:
            continue
        masses = inputs["masses_gev"] if sample in inputs["backgrounds"] else [sample["mass_gev"]]
        counts[sample["name"]] = count_sample(sample, masses, inputs["window_half_width_fraction"])
        counted = counts[sample["name"]]
        if "expected_processed_events" in sample and counted["events"] != sample["expected_processed_events"]:
            raise ValueError("Processed event coverage differs from its inventory: " + sample["name"])
        if "expected_event_ids_sha256" in sample and counted["event_ids_sha256"] != sample["expected_event_ids_sha256"]:
            raise ValueError("Processed event identities differ from their GEN slices: " + sample["name"])
        if sample.get("unit_gen_weights") and not math.isclose(counted["sumw"], counted["events"]):
            raise ValueError("Expected unit GEN weights: " + sample["name"])
        print(sample["name"], counts[sample["name"]]["events"], "events counted", flush=True)
        receipt = dict(inputs_sha256=input_hash, counts=counts)
        (args.output / "event_counts.json").write_text(json.dumps(receipt, indent=2) + "\n")
    report = screen(inputs, counts, args.luminosity_fb, signal_only=args.signal_yields_only)
    report["inputs"] = dict(path=str(args.inputs.resolve()), sha256=input_hash)
    report["script_sha256"] = digest(__file__)
    pdf = args.output / "baseline_mass_epsilon_screen.pdf"
    if not args.signal_yields_only:
        draw(report, pdf)
        report["figure"] = dict(path=str(pdf.resolve()), sha256=digest(pdf))
    yield_pdf = args.output / "baseline_mass_epsilon_signal_yields.pdf"
    draw(report, yield_pdf, signal_yields=True)
    report["signal_yield_figure"] = dict(path=str(yield_pdf.resolve()), sha256=digest(yield_pdf))
    generated_pdf = args.output / "dark_photon_generated_dimuon_yields.pdf"
    draw(report, generated_pdf, signal_yields=True, generated_yields=True)
    report["generated_yield_figure"] = dict(path=str(generated_pdf.resolve()), sha256=digest(generated_pdf))
    (args.output / "baseline_mass_epsilon_screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
