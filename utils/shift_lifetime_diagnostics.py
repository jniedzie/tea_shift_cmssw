#!/usr/bin/env python3
# Code style: AI-code. AI-generated GEN boost and lifetime diagnostic utility.
"""Illustrate proper lengths and event-dependent exponential mean lab flights.

Consume a completed, unfiltered GEN boost audit and its reference-derived
lifetime mapping. Verify the original frozen reference and GEN hashes. This
diagnostic neither changes lifetimes nor evaluates detector acceptance/rates.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_inputs(audit_path, mapping_path, flights):
    audit = json.loads(audit_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    require(audit.get("schema") == "shift-bsm-unfiltered-GEN-boost-mixture-audit-v1"
            and audit.get("complete") is True, "Require a complete unfiltered GEN boost audit")
    require(mapping.get("schema") == "shift-bsm-extended-lab-flight-mapping-v1",
            "Unsupported lifetime mapping")
    reference_path = Path(mapping["reference_path"])
    require(digest(reference_path) == mapping["reference_sha256"], "Frozen reference checksum differs")
    reference = json.loads(reference_path.read_text())
    require(reference.get("schema") == "shift-dark-photon-grid-reference-v1"
            and reference.get("sample_kind") == "simulation_only", "Require the frozen simulation reference")
    samples = {}
    for row in audit["samples"]:
        if row["kind"] != "frozen_reference":
            continue
        mass = float(row["mass_gev"])
        require(mass not in samples, "Duplicate reference mass")
        authority = [r for r in reference["references"] if r["mass_gev"] == mass]
        require(len(authority) == 1, "Reference mass is absent or ambiguous")
        authority = authority[0]
        source = authority["boost_source"]
        require(row["source"] == source["gen_path"] and row["sha256"] == source["gen_sha256"]
                and digest(row["source"]) == row["sha256"], "GEN source checksum or authority differs")
        for name, key in (("contract.json", "contract_sha256"), ("manifest.json", "manifest_sha256")):
            require(digest(Path(row["source"]).parent / name) == source[key],
                    "GEN contract or manifest checksum differs")
        events = row["event_rows"]
        identities = {tuple(event["event_id"]) for event in events}
        require(len(events) == len(identities) == source["events"] == row["expected_events"],
                "Reference must retain every unique GEN event")
        boosts = np.asarray([event["beta_gamma"] for event in events], dtype=float)
        require(np.all(np.isfinite(boosts)) and np.all(boosts > 0), "Invalid GEN boost")
        for event in events:
            require(math.isclose(event["beta_gamma"], event["p_gev"] / event["mass_gev"], rel_tol=1.e-12),
                    "Boost differs from the generated parent momentum")
        mean = math.fsum(boosts) / len(boosts)
        require(math.isclose(mean, authority["mean_beta_gamma"], rel_tol=1.e-12),
                "Unfiltered reference mean boost differs")
        lengths = {}
        for flight in flights:
            points = [p for p in mapping["rows"] if p["mass_gev"] == mass
                      and p["requested_estimated_mean_lab_flight_m"] == flight]
            require(len(points) == 1, "Requested lifetime is absent or ambiguous")
            point = points[0]
            require(math.isclose(point["proper_ctau_m"] * mean, flight, rel_tol=1.e-10),
                    "Proper length does not close the reference mean lab flight")
            estimate = authority["width_authority"]["ctau_mm"] * (authority["epsilon"] / point["epsilon"]) ** 2
            require(math.isclose(point["proper_ctau_mm"], estimate, rel_tol=1.e-10),
                    "Coupling and proper length differ from the reference width model")
            lengths[flight] = point
        samples[mass] = dict(boosts=boosts, mean=mean, points=lengths, source=source)
    require(set(samples) == {float(r["mass_gev"]) for r in reference["references"]},
            "A reference mass is missing from the audit")
    return samples, reference_path


def draw(samples, flights, output):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "pdf.fonttype": 42})
    masses = np.asarray(sorted(samples))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.84, wspace=.28)
    boosts = [samples[mass]["boosts"] for mass in masses]
    quantiles = np.asarray([np.quantile(values, [.1, .5, .9]) for values in boosts])
    for mass, values in zip(masses, boosts):
        offsets = .8 * np.sin(np.arange(len(values)) * 2.399963229728653)
        axes[0].scatter(mass + offsets, values, s=7, color="#777777", alpha=.16, linewidths=0)
    axes[0].errorbar(masses, quantiles[:, 1],
                     yerr=np.asarray([quantiles[:, 1] - quantiles[:, 0], quantiles[:, 2] - quantiles[:, 1]]),
                     fmt="o", color="#222222", capsize=5, linewidth=2, label="Median and 10–90% range")
    axes[0].plot(masses, [samples[mass]["mean"] for mass in masses], "D", color="#0072B2",
                 markersize=6, label="Reference mean")
    axes[0].set(title="Generated parent boosts", ylabel=r"$\beta\gamma = |\vec{p}|/m$")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)

    handles, labels, plotted = [], [], {}
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for index, flight in enumerate(flights):
        color = colors[index % len(colors)]
        lengths = np.asarray([samples[mass]["points"][flight]["proper_ctau_m"] for mass in masses])
        laboratory = quantiles * lengths[:, None]
        position = masses + (index - (len(flights) - 1) / 2) * .8
        bars = axes[1].errorbar(position, laboratory[:, 1],
                               yerr=np.asarray([laboratory[:, 1] - laboratory[:, 0],
                                                laboratory[:, 2] - laboratory[:, 1]]),
                               fmt="o", color=color, capsize=5, linewidth=2)
        proper, = axes[1].plot(masses, lengths, "s--", color=color, markersize=5, linewidth=1.2)
        axes[1].axhline(flight, color=color, linestyle=":", alpha=.45, linewidth=1)
        title = "1 km" if flight == 1000 else f"{flight:g} m"
        handles.extend((bars, proper))
        labels.extend((title + ": lab mean (10–90%)", title + r": proper $c\tau$"))
        plotted[str(flight)] = {str(mass): dict(proper_ctau_m=float(length),
            event_lab_mean_quantiles_m=values.tolist(), reference_mean_lab_flight_m=float(flight))
            for mass, length, values in zip(masses, lengths, laboratory)}
    axes[1].set(title=r"Mean lab flight per event: $\beta\gamma\,c\tau$", ylabel="Length [m]", yscale="log")
    axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.17),
                   fontsize=9, frameon=False, ncol=2, columnspacing=1.2)
    for axis in axes:
        axis.set(xlabel=r"$m_{A'}$ [GeV]", xticks=masses, xlim=(min(masses) - 4, max(masses) + 4))
        axis.grid(axis="y", alpha=.18)
        axis.spines[["right", "top"]].set_visible(False)
    fig.suptitle("SHIFT lifetimes: the boost separates proper length from lab flight", fontsize=15, y=.96)
    count = sum(len(values) for values in boosts)
    fig.text(.5, .060, f"{count} unfiltered GEN reference events; bars show empirical 10–90% ranges, not confidence intervals.",
             ha="center", fontsize=9)
    fig.text(.5, .038, "Right: exponential means, not sampled flights; dotted lines are the requested reference mean lab flights.",
             ha="center", fontsize=9)
    fig.text(.5, .017, "Width model is a provisional native-Pythia pilot; signal rates unvalidated.",
             ha="center", fontsize=9)
    fig.savefig(output, metadata={"Title": "SHIFT reference boosts and physical lifetime mapping",
                                 "Subject": "Unfiltered GEN reference; no detector acceptance or rate normalization"})
    plt.close(fig)
    return dict(boost_quantiles={str(mass): values.tolist() for mass, values in zip(masses, quantiles)},
                lifetime_quantiles=plotted)


def read_production_flights(production, gen_cache):
    """Read timed GEN parents directly; retain signed beam projections."""
    from DataFormats.FWLite import Events
    sys.path.insert(0, str(production / "frozen/sources"))
    from audit_dark_photon_gen import objects, read_product
    plan = json.loads((production / "plan.json").read_text())
    inventory = json.loads((production / "gen_inventory.json").read_text())
    seal = json.loads((production / "seal.json").read_text())
    samples = []
    identities = set()
    for point in plan["points"]:
        name = point["point"]
        parent = inventory["points"][name]
        directory = Path(point["gen_directory"])
        evidence = {}
        for filename in ("contract.json", "manifest.json", "validation.json"):
            path = directory / filename
            checksum = digest(path)
            require(checksum == seal["files"][str(path)], "Sealed GEN metadata differs")
            evidence[str(path)] = checksum
        contract = json.loads((directory / "contract.json").read_text())
        audit = json.loads((directory / "validation.json").read_text())
        require(audit["runtime_validated"] and audit["events"] == parent["events"] == point["events"],
                "Incomplete full GEN population")
        source = gen_cache / name / "gen.root"
        require(digest(source) == parent["gen_sha256"], "GEN cache differs from the full parent")
        beam_pz = float(audit["beams"]["beam_pz_gev"])
        require(math.isfinite(beam_pz) and beam_pz != 0 and contract["source_z_mm"] * beam_pz < 0,
                "Audited beam direction does not point from the source toward CMS")
        beam_sign = math.copysign(1., beam_pz)
        rows = []
        for index, event in enumerate(Events(str(source))):
            auxiliary = event.eventAuxiliary()
            identity = [int(auxiliary.run()), int(auxiliary.luminosityBlock()), int(auxiliary.event())]
            require(identity == parent["event_ids"][index] and tuple(identity) not in identities,
                    "GEN event identity is changed or duplicated")
            identities.add(tuple(identity))
            weight = float(read_product(event, "generator", "GenEventInfoProduct").weight())
            require(weight == audit["event_rows"][index]["weight"] == 1., "Native GEN weight differs")
            hep = read_product(event, "shiftEventTime", "edm::HepMCProduct").GetEvent()
            terminal = []
            for particle in objects(hep.particles_begin(), hep.particles_end()):
                if abs(particle.pdg_id()) != contract["pdg_id"]:
                    continue
                start, end = particle.production_vertex(), particle.end_vertex()
                require(start and end, "Incomplete generated signal decay")
                children = list(objects(end.particles_out_const_begin(), end.particles_out_const_end()))
                if any(abs(child.pdg_id()) == contract["pdg_id"] for child in children):
                    continue
                prod, decay, momentum = start.position(), end.position(), particle.momentum()
                delta = [float(getattr(decay, axis)() - getattr(prod, axis)()) for axis in ("x", "y", "z", "t")]
                four_momentum = [float(getattr(momentum, axis)()) for axis in ("px", "py", "pz", "e")]
                proper = delta[3] * float(particle.generated_mass()) / four_momentum[3]
                require(proper > 0 and all(math.isfinite(value) for value in delta), "Invalid decay displacement")
                for displacement, component in zip(delta, four_momentum):
                    require(abs(displacement - component / particle.generated_mass() * proper)
                            <= 1.e-6 * max(1., abs(displacement)), "Decay spacetime differs from parent momentum")
                terminal.append(dict(event_id=identity, native_weight=weight,
                    proper_length_mm=proper, lab_path_m=math.sqrt(sum(value * value for value in delta[:3])) / 1000,
                    signed_beam_projection_m=beam_sign * delta[2] / 1000,
                    production_z_m=float(prod.z()) / 1000, decay_z_m=float(decay.z()) / 1000))
            require(len(terminal) == 1, "Require exactly one terminal physical parent per GEN event")
            rows.extend(terminal)
        require(len(rows) == point["events"], "GEN event coverage differs")
        samples.append(dict(point=name, mass_gev=point["mass_gev"],
            requested_mean_lab_flight_m=point["target_mean_lab_flight_m"], events=len(rows),
            source=str(source), source_sha256=parent["gen_sha256"], gen_remote_path=parent["gen_path"],
            metadata_sha256=evidence, beam_pz_gev=beam_pz, source_z_mm=contract["source_z_mm"], rows=rows))
    require(len(identities) == plan["events"], "Full campaign GEN denominator differs")
    return samples


def draw_decay_cdfs(samples, flights, output):
    masses = sorted({sample["mass_gev"] for sample in samples})
    maximum = max(10000., 1.05 * max(row["lab_path_m"] for sample in samples
                  if sample["requested_mean_lab_flight_m"] in flights for row in sample["rows"]))
    fig, axes = plt.subplots(1, len(masses), figsize=(15, 5.3), sharey=True, squeeze=False)
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.80, wspace=.12)
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7")
    checks = {}
    for axis, mass in zip(axes[0], masses):
        minimum = 0.
        for flight, color in zip(flights, colors):
            rows = [sample for sample in samples if sample["mass_gev"] == mass
                    and sample["requested_mean_lab_flight_m"] == flight]
            require(len(rows) == 1, "Requested CDF point is missing or ambiguous")
            sample = rows[0]
            count = sample["events"]
            values = {}
            for variable, style in (("lab_path_m", "-"), ("signed_beam_projection_m", "--")):
                data = np.sort([row[variable] for row in sample["rows"]])
                cumulative = np.arange(1, count + 1, dtype=float) / count
                require(len(data) == count and np.all(np.isfinite(data)) and cumulative[-1] == 1.
                        and np.all(np.diff(cumulative) >= 0), "CDF coverage or normalization differs")
                minimum = min(minimum, float(data[0]))
                axis.step(np.r_[min(0., data[0]), data], np.r_[0., cumulative], where="post",
                          color=color, linestyle=style, linewidth=1.6,
                          label=f"{flight:g} m target" if variable == "lab_path_m" else None)
                values[variable] = dict(mean_m=float(np.mean(data)), quantiles_m=np.quantile(data, [.1, .5, .9]).tolist(),
                    below_zero_events=int(np.sum(data < 0)), below148m_events=int(np.sum(data <= 148.)),
                    CDF_at148m=float(np.sum(data <= 148.) / count), CDF_endpoint=1.)
            checks[sample["point"]] = dict(full_GEN_events=count, distributions=values)
        axis.axvline(148., color="#333333", linestyle=":", linewidth=1.4)
        axis.set(xscale="symlog", xlim=(minimum * 1.1, maximum), ylim=(0, 1.02), title=f"$m_{{A'}}={mass:g}$ GeV",
                 xlabel="Decay travel [m]")
        axis.set_xscale("symlog", linthresh=.01)
        axis.set_xticks([0, .1, 1, 10, 100, 1000, 10000], ["0", "0.1", "1", "10", "100", "1000", "10000"])
        axis.grid(alpha=.15)
        axis.spines[["right", "top"]].set_visible(False)
        axis.legend(loc="upper left", fontsize=9, frameon=False)
    axes[0, 0].set_ylabel("Fraction of all generated parents")
    fig.suptitle("New SHIFT campaign: actual generated decay-flight distributions", y=.96, fontsize=15)
    fig.text(.5, .86, "Solid: 3D lab path; dashed: signed beam projection toward CMS; vertical dotted line: nominal source–CMS distance (148 m)",
             ha="center", fontsize=10)
    fig.text(.5, .095, "1,000 full GEN events per curve, including decays outside the detector region; requested mean labels are not retuned.", ha="center", fontsize=10)
    fig.text(.5, .050, "Forward path and beam-projection curves nearly coincide. Parent-flight probabilities are not detector acceptance or signal rates.", ha="center", fontsize=10)
    fig.savefig(output, metadata={"Title": "Actual timed-GEN decay flights in the new SHIFT campaign"})
    plt.close(fig)
    return checks


def load_coupling_widths(reference_path, darkcast_path, model_path, additional_references=()):
    """Keep spectral low-mass widths separate from the production pilot model."""
    reference = json.loads(reference_path.read_text())
    require(reference.get("schema") == "shift-dark-photon-grid-reference-v1",
            "Require a frozen production width reference")
    reference["references"] = list(reference["references"])
    for path in additional_references:
        additional = json.loads(path.read_text())
        require(additional.get("schema") == reference["schema"], "Unsupported additional reference schema")
        require(additional["common"] == reference["common"] and additional["cp5"] == reference["cp5"],
                "Additional references use different generator settings")
        reference["references"].extend(additional["references"])
    masses = []
    for point in reference["references"]:
        width = point["width_authority"]
        mass, epsilon = float(point["mass_gev"]), float(point["epsilon"])
        require(mass not in masses, "Duplicate production reference mass")
        require(math.isclose(width["mass_gev"], mass, rel_tol=1.e-12)
                and math.isclose(width["epsilon"], epsilon, rel_tol=1.e-12),
                "Production width reference has another mass or epsilon")
        require(width["ctau_mm"] > 0 and width["total_width_gev"] > 0
                and math.isclose(width["ctau_mm"] * width["total_width_gev"], 1.973269804e-13, rel_tol=1.e-10),
                "Production proper length differs from its complete physical width")
        require(math.isfinite(point["mean_beta_gamma"]) and point["mean_beta_gamma"] > 0,
                "Require a positive unfiltered production boost")
        masses.append(mass)
    reference["references"].sort(key=lambda point: point["mass_gev"])
    sys.path.insert(0, str(darkcast_path.resolve()))
    import darkcast
    spectral = darkcast.Model("dark_photon")
    spec = importlib.util.spec_from_file_location("shift_diagnostic_dark_photon_model", model_path)
    electroweak = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = electroweak
    spec.loader.exec_module(electroweak)
    return reference, spectral, electroweak


def coupling_proper_length_m(mass, epsilon, spectral, electroweak):
    """Illustrative visible-only total width; no dark decay channels assumed."""
    require(math.isfinite(mass) and mass > 0 and math.isfinite(epsilon) and epsilon > 0,
            "Mass and epsilon must be finite and positive")
    if mass < electroweak.PERTURBATIVE_MIN_MASS_GEV:
        width = spectral.width("total", mass, epsilon)
    else:
        # This curve is explicitly a tree-level diagnostic, not a native
        # Pythia width table or a validated precision result near the Z pole.
        width = math.fsum(electroweak.fermion_widths_lo(mass, epsilon).values())
    return electroweak.proper_decay_length_mm(width) / 1000.


def coupling_for_proper_length(mass, length, spectral, electroweak):
    """Invert the width instead of assuming epsilon scaling near the Z pole."""
    require(math.isfinite(length) and length > 0, "Proper length must be finite and positive")
    lower, upper = 1.e-13, 1.e-3
    require(coupling_proper_length_m(mass, lower, spectral, electroweak) > length
            > coupling_proper_length_m(mass, upper, spectral, electroweak),
            "Requested proper length lies outside the diagnostic coupling range")
    for _ in range(70):
        middle = math.sqrt(lower * upper)
        if coupling_proper_length_m(mass, middle, spectral, electroweak) > length:
            lower = middle
        else:
            upper = middle
    return math.sqrt(lower * upper)


def reference_coupling_rows(reference, flights):
    rows = []
    for point in reference["references"]:
        authority = point["width_authority"]
        native_proper = float(authority["ctau_mm"]) / 1000.
        mean_boost = float(point["mean_beta_gamma"])
        for flight in flights:
            proper = flight / mean_boost
            epsilon = float(point["epsilon"]) * math.sqrt(native_proper / proper)
            rows.append(dict(mass_gev=point["mass_gev"], requested_mean_lab_flight_m=flight,
                             mean_beta_gamma=mean_boost, proper_ctau_m=proper, epsilon=epsilon,
                             proper_lifetime_s=proper / 299792458.,
                             width_scope=authority["width_scope"]))
    return rows


def draw_coupling_diagnostics(reference, spectral, electroweak, flights, output):
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    minimum = 2 * next(f.mass_gev for f in electroweak.FERMIONS if f.name == "mu")
    masses = np.geomspace(minimum * 1.0005, 90., 800)
    # Resolve narrow spectral structure instead of hiding it with a coarse
    # logarithmic grid. These are width features, not excluded regions.
    masses = np.unique(np.r_[masses, np.linspace(.65, .90, 160), np.linspace(.99, 1.045, 110)])
    proper_lengths = [0.001, 1., 100., 1000.]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    fig, axis = plt.subplots(figsize=(10.5, 6.8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.90)
    contours = {}
    for length, color in zip(proper_lengths, colors):
        epsilons = [coupling_for_proper_length(float(mass), length, spectral, electroweak) for mass in masses]
        title = "1 mm" if length == .001 else "1 km" if length == 1000 else f"{length:g} m"
        axis.plot(masses, epsilons, color=color, linewidth=1.6, label=r"Proper $c\tau = $" + title)
        contours[str(length)] = epsilons
    reference_rows = reference_coupling_rows(reference, flights)
    flight_colors = ["#222222", "#7A5195", "#EF5675", "#FFA600"]
    for flight, color in zip(flights, flight_colors):
        selected = [row for row in reference_rows if row["requested_mean_lab_flight_m"] == flight]
        axis.scatter([row["mass_gev"] for row in selected], [row["epsilon"] for row in selected],
                     marker="o", s=40, color=color, edgecolors="white", linewidths=.5, zorder=5,
                     label="Production mean lab flight: " + ("1 km" if flight == 1000 else f"{flight:g} m"))
    axis.axvline(minimum, color="#444444", linestyle=":", linewidth=1)
    axis.axvline(12., color="#777777", linestyle=":", linewidth=1)
    axis.axvspan(85., 95., color="#AAAAAA", alpha=.18)
    axis.axvline(electroweak.DEFAULT_EW_INPUTS.m_z_gev, color="#777777", linestyle="--", linewidth=1)
    axis.text(12.8, 4.e-6, "12 GeV", fontsize=9, color="#555555")
    axis.text(89., 1.e-10, "$Z$ region", rotation=90, fontsize=9, ha="center")
    axis.set(xscale="log", yscale="log", xlim=(minimum * .95, 95.), ylim=(5.e-12, 1.e-5),
             xlabel=r"Dark-photon mass $m_{A'}$ [GeV]", ylabel=r"Kinetic mixing $\epsilon$",
             title="Dark-photon coupling and lifetime")
    axis.set_xticks([minimum, .3, 1, 3, 10, 30, 90])
    axis.set_xticklabels([r"$2m_\mu$", "0.3", "1", "3", "10", "30", "90"])
    axis.grid(which="major", alpha=.15)
    axis.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    for spine in ("right", "top"):
        axis.spines[spine].set_visible(False)
    fig.text(.5, .125, r"$c\tau = \hbar c/\Gamma_{\rm total}$;  $\langle L_{\rm lab}\rangle=\langle\beta\gamma\rangle c\tau$.  Points use the frozen SHIFT GEN mean boosts.",
             ha="center", fontsize=10)
    fig.text(.5, .081, "Curves: DarkCast spectral hadronic widths below 12 GeV; hypercharge-mixing fermion tree widths above 12 GeV.",
             ha="center", fontsize=9)
    fig.text(.5, .046, "Visible-only model. Dots: frozen production widths (spectral below 12 GeV; native Pythia above). Near-Z validation remains open.",
             ha="center", fontsize=9)
    fig.text(.5, .013, "Lifetime mapping only: these curves are not exclusion limits or detector acceptance.", ha="center", fontsize=9)
    fig.savefig(output, metadata={"Title": "Dark photon epsilon versus mass: proper lifetime contours"})
    plt.close(fig)

    lifetime_output = output.with_name(output.stem + "_proper_lengths.pdf")
    fig, axis = plt.subplots(figsize=(9.7, 6.5))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.90)
    epsilons = np.geomspace(1.e-9, 1.e-3, 220)
    lifetime_curves = {}
    references = {point["mass_gev"]: point for point in reference["references"]}
    for index, mass in enumerate([.22, 1., 15., 30., 50., 90.]):
        if mass in references:
            point = references[mass]
            lengths = point["width_authority"]["ctau_mm"] / 1000. * (point["epsilon"] / epsilons)**2
        else:
            lengths = np.asarray([coupling_proper_length_m(mass, float(epsilon), spectral, electroweak)
                                  for epsilon in epsilons])
        axis.plot(epsilons, lengths, linewidth=1.7, label=f"$m_{{A'}} = {mass:g}$ GeV")
        lifetime_curves[str(mass)] = lengths.tolist()
    for length in [.001, 1., 1000.]:
        axis.axhline(length, color="#777777", alpha=.4, linewidth=1, linestyle=":")
    axis.set(xscale="log", yscale="log", xlabel=r"Kinetic mixing $\epsilon$", ylabel=r"Proper mean length $c\tau$ [m]",
             xlim=(1.e-9, 1.e-3), ylim=(1.e-12, 1.e6), title="Smaller mixing gives longer dark-photon lifetimes")
    axis.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")
    axis.grid(which="major", alpha=.15)
    for spine in ("right", "top"):
        axis.spines[spine].set_visible(False)
    fig.text(.5, .132, r"At fixed mass away from resonant mixing: $\Gamma\propto\epsilon^2$, so $c\tau\propto\epsilon^{-2}$.", ha="center", fontsize=11)
    fig.text(.5, .085, "These are proper lengths; multiply by the generated parent boost to obtain an event's mean lab flight.",
             ha="center", fontsize=10)
    fig.text(.5, .043, "Generated reference masses: frozen production width prescription. Other masses: spectral or tree-level width diagnostic.",
             ha="center", fontsize=9)
    fig.text(.5, .011, "Visible-only lifetime mapping; no production rate or exclusion-limit claim.", ha="center", fontsize=9)
    fig.savefig(lifetime_output, metadata={"Title": "Dark photon proper mean length versus epsilon"})
    plt.close(fig)
    return dict(mass_grid_gev=masses.tolist(), proper_length_contours_m=contours,
                epsilon_grid=epsilons.tolist(), lifetime_curves_m=lifetime_curves,
                production_mapping=reference_rows,
                pdfs=[dict(path=str(path.resolve()), sha256=digest(path)) for path in (output, lifetime_output)],
                assumptions=dict(low_mass="DarkCast 2.1 photon-like spectral widths below 12 GeV",
                                 high_mass="full hypercharge-mixing fermion tree widths; HAHM defaults",
                                 production_points="frozen native-Pythia widths and unfiltered reference mean boosts",
                                 hidden_decay_channels=False, detector_acceptance_included=False,
                                 physical_sensitivity_validated=False, near_z_finite_width_validated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boost-audit", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--production", type=Path, help="Sealed campaign directory for actual timed-GEN decay CDFs")
    parser.add_argument("--gen-cache", type=Path, help="Content-verified local GEN copies, one point directory each")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--flights", type=float, nargs="+")
    parser.add_argument("--coupling-reference", type=Path, help="Production width reference for epsilon/lifetime plots")
    parser.add_argument("--additional-reference", type=Path, action="append", default=[],
                        help="Additional production width/boost reference; repeat to combine mass ranges")
    parser.add_argument("--darkcast-path", type=Path, help="Directory containing the pinned external DarkCast package")
    parser.add_argument("--model-source", type=Path, help="Workflow dark_photon_model.py for high-mass mixing widths")
    args = parser.parse_args()
    if args.coupling_reference is not None:
        require(args.darkcast_path is not None and args.model_source is not None,
                "Coupling plots require --darkcast-path and --model-source")
        flights = args.flights or [5., 30., 100., 1000.]
        require(len(flights) == len(set(flights)) and all(math.isfinite(x) and x > 0 for x in flights),
                "Require distinct positive mean lab flights")
        reference, spectral, electroweak = load_coupling_widths(
            args.coupling_reference, args.darkcast_path, args.model_source, args.additional_reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = draw_coupling_diagnostics(reference, spectral, electroweak, flights, args.output)
        result.update(schema="shift-dark-photon-coupling-lifetime-diagnostic-v1",
                      reference_path=str(args.coupling_reference.resolve()),
                      reference_sha256=digest(args.coupling_reference),
                      additional_references=[dict(path=str(path.resolve()),sha256=digest(path))
                                             for path in args.additional_reference],
                      model_source_sha256=digest(args.model_source), script_sha256=digest(__file__),
                      sources=["https://arxiv.org/abs/1412.0018", "https://arxiv.org/abs/1801.04847",
                               "https://gitlab.com/darkcast/releases"])
        package = args.darkcast_path / "darkcast"
        result["darkcast_source_files_sha256"] = {
            str(path.relative_to(package)): digest(path) for path in package.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result["pdfs"]))
        return
    if args.production is not None:
        require(args.gen_cache is not None, "Actual GEN CDFs require --gen-cache")
        flights = args.flights or [5., 30., 100., 1000.]
        samples = read_production_flights(args.production.resolve(), args.gen_cache.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checks = draw_decay_cdfs(samples, flights, args.output)
        receipt = dict(schema="shift-actual-GEN-decay-flight-CDF-v1", complete=True,
            pdf=dict(path=str(args.output.resolve()), sha256=digest(args.output)),
            production=str(args.production.resolve()), seal_sha256=digest(args.production / "seal.json"),
            plan_sha256=digest(args.production / "plan.json"), full_GEN_events=sum(sample["events"] for sample in samples),
            plotted_GEN_events=sum(row["full_GEN_events"] for row in checks.values()), samples=samples, CDF_checks=checks,
            script_sha256=digest(__file__), projected_distance="signed displacement dotted with audited beam direction; no absolute value",
            interpretation="Actual relative production-to-decay vertices; no detector selection, rate normalization or lifetime relabelling")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps(receipt["pdf"]))
        return
    require(args.boost_audit is not None and args.mapping is not None, "Reference diagnostic requires --boost-audit and --mapping")
    args.flights = args.flights or [100., 1000.]
    require(len(args.flights) == len(set(args.flights)) and all(math.isfinite(x) and x > 0 for x in args.flights),
            "Require distinct positive mean lab flights")
    samples, reference = read_inputs(args.boost_audit, args.mapping, args.flights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotted = draw(samples, args.flights, args.output)
    receipt = dict(schema="shift-lifetime-diagnostic-pdf-v1", complete=True,
        pdf=dict(path=str(args.output.resolve()), sha256=digest(args.output)),
        inputs={str(path.resolve()): digest(path) for path in (args.boost_audit, args.mapping, reference)},
        script=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
        reference_GEN={str(mass): sample["source"] for mass, sample in samples.items()},
        matplotlib=matplotlib.__version__, numpy=np.__version__, plotted=plotted,
        interpretation="Unfiltered generated boosts and event-dependent exponential mean lab paths; no acceptance/rate claim")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt["pdf"]))


if __name__ == "__main__":
    main()
