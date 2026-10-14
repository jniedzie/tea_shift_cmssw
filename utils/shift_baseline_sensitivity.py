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
                     "known-background, no-nuisance exact median 90% CLs screening; MC errors reported separately",
                     "empty constant-weight MC components use conditional single-component90% binomial upper planning floors; not a joint confidence bound",
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
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 6.3), squeeze=False)
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
            partial = [p for p in ready if p["partial"] and not generated_yields]
            if partial:
                axis.scatter([p["mass_gev"] for p in partial], [p["epsilon"] for p in partial],
                    facecolors="none", edgecolors="red", s=145, linewidths=1., label="Partial detector coverage")
        if unknown:
            axis.scatter([p["mass_gev"] for p in unknown], [p["epsilon"] for p in unknown],
                         marker="x", c="gray", s=65, label="Unknown: no selected MC")
        axis.set(xscale="log", yscale="log", xlabel=r"$m_{A'}$ [GeV]",
                 title=f"{scenario['luminosity_role']}\n$L_{{eff}}$ = {scenario['luminosity_fb']:g} fb$^{{-1}}$")
        axis.grid(alpha=.2)
        axis.spines[["right", "top"]].set_visible(False)
        if unknown or any(p["partial"] for p in ready) and not generated_yields:
            axis.legend(fontsize=8, loc="lower left")
    axes[0, 0].set_ylabel(r"$\epsilon$")
    title = "SHIFT baseline signal yield — MC scenario, no BDT" if signal_yields else "SHIFT baseline sensitivity screen — MC scenario, no BDT"
    if generated_yields:
        title = "SHIFT dark-photon dimuon yield before detector selection — DY model"
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(left=.07, right=.88, bottom=.25, top=.86, wspace=.25)
    if artist is not None:
        colorbar = fig.colorbar(artist, cax=fig.add_axes([.90, .29, .016, .54]))
        colorbar.set_label(r"$\log_{10}(N)$, generated dimuon events" if generated_yields else
                          r"$\log_{10}(S)$, selected signal events" if signal_yields else
                          r"$\log_{10}(S/S_{90})$; 0 is the nominal screening boundary")
    width = 100. * report["selection"]["window_half_width_fraction"]
    selection_text = ("Production cross section times effective luminosity; dimuon branching fraction included once."
                      if generated_yields else f"One OS valid candidate per event; reconstructed mass within ±{width:g}% of the hypothesis.")
    fig.text(.5, .13, selection_text, ha="center", fontsize=9)
    description = ("DY selected cross sections already include the dimuon branching fraction; simulated selection acceptance is applied once."
                   if signal_yields else "Known-background screen; empty constant-weight MC components use 90% upper planning floors. No fitted nuisances.")
    fig.text(.5, .09, description, ha="center", fontsize=9)
    fig.text(.5, .05, "Native-Pythia pure-DY rate and no-pileup/Fake2-HLT detector pilots. Points only; no physical exclusion contour.", ha="center", fontsize=9)
    fig.savefig(output, metadata={"Title": "SHIFT provisional baseline sensitivity screen"})
    plt.close(fig)


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
