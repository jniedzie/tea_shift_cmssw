#!/usr/bin/env python3
# Code style: AI-code. AI-generated adapter for tea counting cards and Combine.
"""Prepare audited event yields for tea and combine exclusive counting channels.

Run from bin: python shift_limits.py --config shift_limits_config.py.
The input JSON owns luminosity, full GEN normalization and event assignment.
This adapter never changes selections or calculates acceptance from candidates.
"""
import argparse
import hashlib
import importlib
import json
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_limit_samples(manifest_path):
    """Keep sample inventory parsing out of the ordinary limits config."""
    from Sample import Sample, SampleType
    path = Path(manifest_path)
    if not path.exists():
        return [], [], {}
    manifest = json.loads(path.read_text())
    signals = [Sample(name=name, file_path=record["file_path"], type=SampleType.signal,
                      cross_section=record["combine_reference_cross_section_pb"])
               for name, record in manifest["signals"].items()]
    backgrounds = [Sample(name=name, file_path=record["file_path"], type=SampleType.background)
                   for name, record in manifest["backgrounds"].items()]
    return signals, backgrounds, manifest


def validate_yields(report, config):
    if report.get("schema") != "shift-simplified-limit-yields-v1":
        raise ValueError("Unsupported simplified-limit yields schema")
    if not math.isclose(float(report["luminosity_pb"]), config.luminosity):
        raise ValueError("Yield luminosity does not match the limits config")
    if list(report["categories"]) != list(config.categories):
        raise ValueError("Yield categories do not match the limits config")
    if not report["backgrounds"] or not report["signals"]:
        raise ValueError("Limits need background and signal inventories")
    for sample_type in ("signals", "backgrounds"):
        for name, record in report[sample_type].items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError("Unsafe sample name: " + name)
            if set(record["yields"]) != set(config.categories):
                raise ValueError("Incomplete category yields for " + name)
            for category, value in record["yields"].items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid expected yield for {name}/{category}: {value}")
                if "binned_yields" in record:
                    bins = record["binned_yields"][category]
                    if (not bins or any(not math.isfinite(entry) or entry < 0 for entry in bins)
                            or not math.isclose(math.fsum(bins), value, rel_tol=1.e-9, abs_tol=1.e-12)):
                        raise ValueError("Mass bins do not preserve the expected yield: " + name + "/" + category)
            if sample_type == "signals":
                sigma = float(record["reference_cross_section_pb"])
                if not math.isfinite(sigma) or sigma <= 0:
                    raise ValueError("Signal physical cross section must be positive: " + name)


def reference_cross_section(signal, backgrounds, categories):
    """Condition the fit while preserving every physical signal rate ratio."""
    physical_sigma = float(signal["reference_cross_section_pb"])
    if "binned_yields" in signal:
        cells = [(category, index) for category in categories
                 for index in range(len(signal["binned_yields"][category]))]
        rates = {cell: signal["binned_yields"][cell[0]][cell[1]] / physical_sigma for cell in cells}
        background = {cell: math.fsum(sample["binned_yields"][cell[0]][cell[1]]
                                     for sample in backgrounds.values()) for cell in cells}
        categories = cells
    else:
        rates = {category: float(signal["yields"][category]) / physical_sigma for category in categories}
        background = {category: math.fsum(float(sample["yields"][category]) for sample in backgrounds.values())
                      for category in categories}
    positive_background = [category for category in categories if background[category] > 0]
    background_total = math.fsum(background[category] for category in positive_background)
    signal_total = math.fsum(rates[category] for category in positive_background)
    choices = [3. / rates[category] for category in categories
               if background[category] == 0 and rates[category] > 0]
    if background_total > 0:
        ratio = signal_total / background_total
        # Stable profiled Fisher information: channel shape plus total rate.
        shape = math.fsum(background[category] * (rates[category] / background[category] - ratio) ** 2
                          for category in positive_background)
        total_rate = ratio * ratio * background_total / (1. + math.log(1.30) ** 2 * background_total)
        information = shape + total_rate
        if information > 0:
            # Use an approximate expected upper limit, rather than a one-sigma
            # sensitivity, so the fitted strength is close to one.
            choices.append(2. / math.sqrt(information))
    if not choices:
        return physical_sigma
    return min(choices)


def write_count_histograms(path, rates):
    import ROOT
    root_file = ROOT.TFile(str(path), "RECREATE")
    if root_file.IsZombie():
        raise RuntimeError("Cannot write prepared input " + str(path))
    for category, rate in rates.items():
        hist = ROOT.TH1D("count_" + category, "Expected event count", 1, 0., 1.)
        hist.SetBinContent(1, rate)
        hist.SetBinError(1, 0.)
        hist.SetEntries(1.)  # The native loader accepts finite zero-count channels.
        hist.Write()
    root_file.Close()


def write_binned_histograms(path, rates):
    """Write the already normalized, audited mass bins for native TEA cards."""
    import ROOT
    root_file = ROOT.TFile(str(path), "RECREATE")
    if root_file.IsZombie():
        raise RuntimeError("Cannot write prepared input " + str(path))
    for category, values in rates.items():
        hist = ROOT.TH1D("mass_" + category, "Reconstructed mass bins", len(values), 0., len(values))
        for index, value in enumerate(values, 1):
            hist.SetBinContent(index, value)
            hist.SetBinError(index, 0.)
        hist.SetEntries(1.)
        hist.Write()
        for index, value in enumerate(values):
            single = ROOT.TH1D("mass_" + category + "_bin" + str(index), "One reconstructed mass bin", 1, 0., 1.)
            single.SetBinContent(1, value)
            single.SetBinError(1, 0.)
            single.SetEntries(1.)
            single.Write()
    root_file.Close()


def prepare_inputs(config):
    report = json.loads(Path(config.yields_path).read_text())
    validate_yields(report, config)
    points = tuple(getattr(config, "signal_points", ())) or tuple(report["signals"])
    missing = set(points) - set(report["signals"])
    if missing:
        raise ValueError("Missing audited signal points: " + ", ".join(sorted(missing)))
    report["signals"] = {name: report["signals"][name] for name in points}
    output = Path(config.prepared_input_manifest).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    existing = Path(config.prepared_input_manifest)
    input_hash = digest(config.yields_path)
    if existing.exists():
        previous = json.loads(existing.read_text())
        same_inputs = previous["yields_sha256"] == input_hash and set(previous["signals"]) == set(points)
        if not same_inputs and any(Path(config.datacards_output_path).rglob("higgsCombine*.root")):
            raise ValueError("Existing fits use different yields or signal points; choose a new limits_study")
        if same_inputs:
            for kind in ("signals", "backgrounds"):
                for record in previous[kind].values():
                    if digest(record["file_path"]) != record["file_sha256"]:
                        raise ValueError("Prepared input changed: " + record["file_path"])
            return previous
    manifest = dict(schema="shift-simplified-limit-inputs-v1", categories=list(config.categories),
                    luminosity_pb=config.luminosity, yields_path=str(Path(config.yields_path).resolve()),
                    yields_sha256=input_hash, signals={}, backgrounds={})
    for name, sample in report["backgrounds"].items():
        path = output / ("background_" + name + ".root")
        if "binned_yields" in sample:
            write_binned_histograms(path, sample["binned_yields"])
        else:
            write_count_histograms(path, sample["yields"])
        manifest["backgrounds"][name] = dict(sample, file_path=str(path), file_sha256=digest(path))
    for name, sample in report["signals"].items():
        sigma = reference_cross_section(sample, report["backgrounds"], config.categories)
        physical_sigma = float(sample["reference_cross_section_pb"])
        rates = {category: value * sigma / physical_sigma for category, value in sample["yields"].items()}
        path = output / ("signal_" + name + ".root")
        if "binned_yields" in sample:
            bins = {category: [value * sigma / physical_sigma for value in values]
                    for category, values in sample["binned_yields"].items()}
            write_binned_histograms(path, bins)
        else:
            write_count_histograms(path, rates)
        manifest["signals"][name] = dict(sample, file_path=str(path), file_sha256=digest(path),
                                         combine_reference_cross_section_pb=sigma,
                                         combine_reference_yields=rates)
    Path(config.prepared_input_manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def category_config(config, manifest, category):
    from Histogram import Histogram
    from HistogramNormalizer import NormalizationType
    current = SimpleNamespace(**{key: value for key, value in vars(config).items() if not key.startswith("__")})
    current.config_source_path = getattr(config, "__file__", getattr(config, "config_source_path", None))
    current.category = category
    current.histogram = Histogram(name=getattr(config, "histogram_prefix", "count_") + category,
                                  norm_type=NormalizationType.none)
    current.signal_samples, current.background_samples, _ = load_limit_samples(config.prepared_input_manifest)
    current.samples = current.signal_samples + current.background_samples
    current.nuisances = dict(signal_norm={"signal": 1.30},
                            background_norm={sample.name: 1.30 for sample in current.background_samples})
    return current


def create_cards(config, manifest, combine_setup):
    if getattr(config, "split_mass_bins", False):
        return create_mass_bin_cards(config, manifest, combine_setup)
    import ROOT
    from HistogramsManager import HistogramsManager
    from limits_producer import get_datacard_file_name
    import DatacardsProcessor
    import HistogramsManager as histogram_module
    directory = Path(config.datacards_output_path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    signature = dict(yields_sha256=manifest["yields_sha256"],
                     sources={str(Path(module.__file__).resolve()): digest(module.__file__)
                              for module in (DatacardsProcessor, histogram_module)},
                     adapter_sha256=digest(__file__), include_shapes=config.include_shapes,
                     include_mc_stat_uncertainties=config.include_mc_stat_uncertainties,
                     retain_zero_backgrounds=config.retain_zero_backgrounds,
                     nuisances={"signal_norm": 1.30, "background_norm": 1.30})
    card_manifest = directory / "manifest.json"
    if card_manifest.exists():
        previous = json.loads(card_manifest.read_text())
        if previous["signature"] == signature:
            for path, checksum in previous["artifacts_sha256"].items():
                if digest(path) != checksum:
                    raise ValueError("Prepared card or shape changed: " + path)
            return previous["cards"]
        if any(directory.rglob("higgsCombine*.root")):
            raise ValueError("Existing card model changed; choose a new limits_study")
    cards = []
    components = []
    root_files = {name: ROOT.TFile.Open(record["file_path"], "READ")
                  for kind in ("signals", "backgrounds") for name, record in manifest[kind].items()}
    try:
        for category in config.categories:
            current = category_config(config, manifest, category)
            for signal in current.signal_samples:
                name = get_datacard_file_name(current, signal)
                manager = HistogramsManager(current, root_files, name + "_component")
                manager.addHistosample(current.histogram, signal)
                for background in current.background_samples:
                    manager.addHistosample(current.histogram, background)
                manager.normalizeHistograms()
                manager.buildStacks()
                manager.saveDatacards()
                components.append(dict(point=signal.name, channel=category, name=name + "_component"))
                reference = reference_cross_section(manifest["signals"][signal.name],
                                                    manifest["backgrounds"], (category,))
                combined_reference = manifest["signals"][signal.name]["combine_reference_cross_section_pb"]
                manager = HistogramsManager(current, root_files, name)
                manager.addHistosample(current.histogram, signal)
                manager.histosamples[-1][0].hist.Scale(reference / combined_reference)
                for background in current.background_samples:
                    manager.addHistosample(current.histogram, background)
                manager.normalizeHistograms()
                manager.buildStacks()
                manager.saveDatacards()
                cards.append(dict(point=signal.name, channel=category, name=name,
                                  combine_reference_cross_section_pb=reference))
        commands = []
        for point in manifest["signals"]:
            name = "datacard_combined_" + point
            point_cards = [record for record in components if record["point"] == point]
            arguments = [record["channel"].replace("-", "_") + "=" +
                         str(directory / (record["name"] + ".txt")) for record in point_cards]
            commands.append("combineCards.py " + shlex.join(arguments) + " > " +
                            shlex.quote(str(directory / (name + ".txt"))))
            cards.append(dict(point=point, channel="combined", name=name,
                              combine_reference_cross_section_pb=manifest["signals"][point]["combine_reference_cross_section_pb"]))
        command = "set -e -o pipefail; " + combine_setup + "; " + "; ".join(commands)
        subprocess.run(["bash", "-lc", command], text=True, capture_output=True, check=True)
        artifacts = list(directory.glob("*.txt")) + list(directory.glob("*.root"))
        card_manifest.write_text(json.dumps(dict(signature=signature, cards=cards,
            artifacts_sha256={str(path): digest(path) for path in artifacts}), indent=2) + "\n")
    finally:
        for file in root_files.values():
            file.Close()
    return cards


def create_mass_bin_cards(config, manifest, combine_setup):
    """An equivalent binned likelihood using stable one-bin Poisson channels.

    TEA writes every component; combineCards.py groups bins into categories and
    their combination. No change to rates, normalization nuisances or mass bins.
    """
    import ROOT
    from Histogram import Histogram
    from HistogramNormalizer import NormalizationType
    from HistogramsManager import HistogramsManager
    from limits_producer import get_datacard_file_name
    directory = Path(config.datacards_output_path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    signature = dict(yields_sha256=manifest["yields_sha256"], adapter_sha256=digest(__file__),
                     representation="One Poisson channel per mass bin",
                     nuisance_model={"signal_norm": 1.30, "background_norm": 1.30})
    receipt = directory / "manifest.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["signature"] != signature:
            raise ValueError("Saved Poisson-bin model changed; choose a new output directory")
        for path, checksum in old["artifacts_sha256"].items():
            if digest(path) != checksum:
                raise ValueError("Saved Poisson-bin card changed: " + path)
        return old["cards"]
    files = {name: ROOT.TFile.Open(row["file_path"], "READ")
             for kind in ("signals", "backgrounds") for name, row in manifest[kind].items()}
    cards, commands = [], []

    def combine(name, components):
        arguments = [label.replace("-", "_") + "=" + str(directory / (card_name + ".txt"))
                     for label, card_name in components]
        commands.append("combineCards.py " + shlex.join(arguments) + " > " +
                        shlex.quote(str(directory / (name + ".txt"))))

    try:
        for point, row in manifest["signals"].items():
            combined_components = []
            for category in config.categories:
                current = category_config(config, manifest, category)
                signal = next(sample for sample in current.signal_samples if sample.name == point)
                category_reference = reference_cross_section(row, manifest["backgrounds"], (category,))
                combined_reference = row["combine_reference_cross_section_pb"]
                category_components = []
                for index in range(len(row["binned_yields"][category])):
                    current.histogram = Histogram(name="mass_" + category + "_bin" + str(index),
                                                  norm_type=NormalizationType.none)
                    base_name = get_datacard_file_name(current, signal)
                    label = category + "_bin" + str(index)
                    for suffix, factor, destination in (
                            ("_component", 1., combined_components),
                            ("_category_component", category_reference/combined_reference, category_components)):
                        manager = HistogramsManager(current, files, base_name + suffix)
                        manager.addHistosample(current.histogram, signal)
                        manager.histosamples[-1][0].hist.Scale(factor)
                        for background in current.background_samples:
                            manager.addHistosample(current.histogram, background)
                        manager.normalizeHistograms()
                        manager.buildStacks()
                        manager.saveDatacards()
                        destination.append((label, base_name + suffix))
                name = "datacard_mass_" + category + "_" + point
                combine(name, category_components)
                cards.append(dict(point=point, channel=category, name=name,
                                  combine_reference_cross_section_pb=category_reference))
            name = "datacard_combined_" + point
            combine(name, combined_components)
            cards.append(dict(point=point, channel="combined", name=name,
                              combine_reference_cross_section_pb=row["combine_reference_cross_section_pb"]))
        script = "set -e -o pipefail\n" + "\n".join(commands) + "\n"
        (directory / "combine_cards.sh").write_text(script)
        subprocess.run(["bash", "-lc", "set -e -o pipefail; " + combine_setup + "; bash -s"],
                       input=script, text=True, capture_output=True, check=True)
        artifacts = list(directory.glob("*.txt")) + list(directory.glob("*.root"))
        receipt.write_text(json.dumps(dict(signature=signature, cards=cards,
            artifacts_sha256={str(path): digest(path) for path in artifacts}), indent=2) + "\n")
    finally:
        for file in files.values():
            file.Close()
    return cards


def card_has_signal(card, manifest, categories):
    signal = manifest["signals"][card["point"]]
    selected = categories if card["channel"] == "combined" else (card["channel"],)
    return any(signal["yields"][category] > 0 for category in selected)


def results_from_outputs(config, manifest, cards, confidence_level, directory):
    from limits_producer import read_limit_quantiles
    records = []
    for card in cards:
        signal = manifest["signals"][card["point"]]
        record = dict(point=card["point"], channel=card["channel"], confidence_level=confidence_level,
                      mass_gev=signal["mass_gev"], epsilon=signal.get("epsilon"),
                      mean_lab_flight_m=signal.get("mean_lab_flight_m"),
                      proper_ctau_mm=signal.get("proper_ctau_mm"), partial=signal.get("partial", False),
                      reference_cross_section_pb=signal["reference_cross_section_pb"],
                      combine_reference_cross_section_pb=card["combine_reference_cross_section_pb"])
        if not card_has_signal(card, manifest, config.categories):
            record.update(status="no_signal_acceptance", expected_quantiles=None,
                          expected_limit=None, expected_limit_cross_section_pb=None)
        else:
            path = directory / ("higgsCombine." + card["name"] + ".AsymptoticLimits.mH120.root")
            quantiles = read_limit_quantiles(path)
            scale = card["combine_reference_cross_section_pb"]
            physical = signal["reference_cross_section_pb"]
            xsecs = {key: value * scale for key, value in quantiles.items() if key != "observed"}
            strengths = {key: value / physical for key, value in xsecs.items()}
            record.update(status="ok", expected_quantiles=strengths,
                          expected_cross_section_quantiles_pb=xsecs, raw_combine_quantiles=quantiles,
                          expected_limit=strengths["median"], expected_limit_cross_section_pb=xsecs["median"],
                          combine_output=str(path), combine_output_sha256=digest(path))
        records.append(record)
    return records


def scan_group(signal):
    if "scan_group" in signal:
        return str(signal["scan_group"])
    if signal.get("sample_kind") == "epsilon" or "_epsilon_" in signal.get("source", {}).get("name", ""):
        return "epsilon_" + format(float(signal["epsilon"]), ".6g")
    flight = signal.get("mean_lab_flight_m")
    return "prompt" if flight is None else "flight_" + format(float(flight), ".6g") + "m"


def draw_brazil_plots(config, manifest, records):
    import limits_plotter
    groups = {}
    for record in records:
        if record["status"] != "ok":
            continue
        key = (record["confidence_level"], record["channel"], scan_group(manifest["signals"][record["point"]]))
        groups.setdefault(key, []).append(record)
    for (level, channel, group), selected in groups.items():
        if len({record["mass_gev"] for record in selected}) != len(selected):
            raise ValueError("Mass scan has duplicate masses; set explicit scan_group metadata")
        directory = Path(config.plots_output_path) / ("cl" + str(round(100 * level))) / channel / group
        directory.mkdir(parents=True, exist_ok=True)
        current = category_config(config, manifest, channel)
        current.results_output_path = str(directory)
        current.plots_output_path = str(directory)
        current.limit_points = {record["point"]: record["mass_gev"] for record in selected}
        current.reference_cross_section = 1.0
        current.x_min = min(record["mass_gev"] for record in selected) * .8
        current.x_max = max(record["mass_gev"] for record in selected) * 1.2
        current.experiment_label = "#bf{SHIFT} #it{Simulation}"
        current.luminosity_label = f"3 fb^{{-1}}, {round(level * 100)}% CL, {channel}"
        values = [value for record in selected for value in record["expected_cross_section_quantiles_pb"].values()]
        current.y_min = min(values) * .5
        current.y_max = max(values) * 3.
        current.histogram.name = "mass_scan"
        path = directory / "limits_mass_scan.txt"
        with path.open("w") as stream:
            for record in sorted(selected, key=lambda row: row["mass_gev"]):
                quantiles = record["expected_cross_section_quantiles_pb"]
                row = [None] + [quantiles[key] for key in ("minus2", "minus1", "median", "plus1", "plus2")]
                stream.write(record["point"] + ": " + repr(row) + "\n")
        limits_plotter.config = current
        limits_plotter.input_path = str(path)
        limits_plotter.draw_brazil_plots()


def bind_fits(config, manifest, cards, directory, combine_executable):
    """A ROOT filename alone never establishes that an old fit is reusable."""
    import limits_producer
    import DatacardsProcessor
    import HistogramsManager
    sources = [Path(__file__), Path(config.config_source_path) if getattr(config, "config_source_path", None) else None,
               Path(limits_producer.__file__), Path(DatacardsProcessor.__file__), Path(HistogramsManager.__file__)]
    source_hashes = {str(path.resolve()): digest(path) for path in sources if path is not None}
    bindings = dict(schema="shift-limit-fit-bindings-v1", yields_sha256=manifest["yields_sha256"],
                    combine_executable=str(combine_executable), combine_sha256=digest(combine_executable),
                    source_sha256=source_hashes, options=list(config.combine_options),
                    shapes_sha256={str(path.resolve()): digest(path) for path in Path(config.datacards_output_path).parent.glob("*.root")},
                    cards={card["name"]: dict(sha256=digest(directory / (card["name"] + ".txt")),
                          options=list(config.combine_options_by_datacard.get(card["name"], ()))) for card in cards})
    preload = getattr(config, "combine_preload_path", None)
    if preload:
        bindings["combine_preload"] = dict(path=str(Path(preload).resolve()), sha256=digest(preload))
    path = directory / "fit_bindings.json"
    if not path.exists() and any(directory.glob("higgsCombine*.root")):
        raise ValueError("Existing fits have no saved bindings; choose a new limits_study: " + str(path))
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != bindings and any(directory.glob("higgsCombine*.root")):
            raise ValueError("Existing fit bindings changed; choose a new limits_study: " + str(path))
    path.write_text(json.dumps(bindings, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="shift_limits_config.py")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cards-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    import ROOT
    from limits_producer import find_combine, run_combine
    ROOT.gROOT.SetBatch(True)
    config = importlib.import_module(args.config.removesuffix(".py").replace("/", "."))
    if args.plots_only:
        manifest = json.loads(Path(config.prepared_input_manifest).read_text())
        results = json.loads((Path(config.results_output_path) / "limits.json").read_text())
        draw_brazil_plots(config, manifest, results["records"])
        return
    manifest = prepare_inputs(config)
    if args.prepare_only:
        return
    setup = find_combine(config)
    if getattr(config, "combine_preload_path", None):
        library = Path(config.combine_preload_path)
        if not library.is_file() or not (library.parent / "receipt.json").is_file():
            raise FileNotFoundError("Saved numerical library and receipt are required; no automatic rebuild")
        from shift_limits_combine_precision import verify_stable_nll
        verify_stable_nll(config)
    binary = subprocess.run(["bash", "-lc", "set -e -o pipefail; " + setup + "; command -v combine"],
                            text=True, capture_output=True, check=True).stdout.strip().splitlines()[-1]
    cards = create_cards(config, manifest, setup)
    if args.cards_only:
        return
    records = []
    for level in config.confidence_levels:
        current = category_config(config, manifest, config.category)
        directory = Path(config.datacards_output_path) / ("cl" + str(round(100 * level)))
        directory.mkdir(parents=True, exist_ok=True)
        current.datacards_output_path = str(directory.resolve()) + "/"
        # Standard AsymptoticLimits defaults to 95%; only the paper comparison
        # needs a different confidence level. No fit-range or minimizer tuning.
        current.combine_options = tuple(config.combine_options)
        if level != 0.95:
            current.combine_options += ("--cl", str(level))
        current.combine_options_by_datacard = {}
        runnable = []
        for card in sorted(cards, key=lambda row: row["channel"] != "combined"):
            shutil.copy2(Path(config.datacards_output_path) / (card["name"] + ".txt"), directory)
            shape = Path(config.datacards_output_path) / (card["name"] + ".root")
            if config.include_shapes and shape.exists():
                shutil.copy2(shape, directory)
            if not card_has_signal(card, manifest, config.categories):
                continue
            output = directory / ("higgsCombine." + card["name"] + ".AsymptoticLimits.mH120.root")
            reusable = False
            if output.exists():
                from limits_producer import read_limit_quantiles
                try:
                    read_limit_quantiles(output)
                    reusable = True
                except RuntimeError:
                    pass
            if not reusable:
                runnable.append(card["name"])
        bind_fits(current, manifest, cards, directory, binary)
        if runnable:
            run_combine(current, runnable)
        records.extend(results_from_outputs(config, manifest, cards, level, directory))
        result_path = Path(config.results_output_path) / "limits.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(dict(schema="shift-simplified-limits-v1", method="AsymptoticLimits",
            expected_only=True, luminosity_pb=config.luminosity, yields_sha256=manifest["yields_sha256"],
            nuisance_model={"signal_norm": 1.30, "background_norm": 1.30},
            nuisance_correlations="each nuisance fully correlated across channels; independent signal/background",
            mc_stat_nuisances=False, records=records), indent=2) + "\n")
    draw_brazil_plots(config, manifest, records)
    receipt = dict(yields_sha256=manifest["yields_sha256"], results_sha256=digest(result_path),
                   figures={str(path.resolve()): digest(path) for path in Path(config.plots_output_path).rglob("*.pdf")})
    (Path(config.results_output_path) / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
