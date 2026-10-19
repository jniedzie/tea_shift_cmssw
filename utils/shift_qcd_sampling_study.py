# Code style: AI-code. AI-generated GEN-to-reconstruction sampling diagnostics.
"""Audit original QCD events and compare GEN-only reconstruction proposals.

This reads completed source cohorts. Its legacy detector samples do not measure
hadron survival in material. No production selections or event weights change.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_kind(pdg_id):
    value = abs(int(pdg_id))
    if value == 211:
        return "pion"
    if value in (130, 310, 311, 321):
        return "kaon"
    if value in (443, 100443, 553, 100553, 200553, 333, 113, 223):
        return "resonance"
    digits = (value // 100 % 10, value // 1000 % 10)
    if 5 in digits:
        return "bottom"
    if 4 in digits:
        return "charm"
    return "other" if value else "unknown"


def line_radius(muon, z_cm):
    """A straight-line diagnostic, with no claim to be detector acceptance."""
    if muon["pz"] == 0:
        return None
    fraction = (z_cm - muon["z_cm"]) / muon["pz"]
    if fraction <= 0:
        return None
    return math.hypot(muon["x_cm"] + fraction * muon["px"],
                      muon["y_cm"] + fraction * muon["py"])


def pair_mass(first, second):
    energy = first["energy"] + second["energy"]
    momentum2 = sum((first[key] + second[key]) ** 2 for key in ("px", "py", "pz"))
    return math.sqrt(max(0., energy * energy - momentum2))


def extract_muons(tree):
    """Read muons and nearest saved non-muon ancestors from the pruned Nano graph.

    A saved mother may skip physical intermediates. In particular a quark or
    beam proton here does not establish that the muon is prompt.
    """
    muons = []
    for index in range(int(tree.nGenPart)):
        pdg_id = int(tree.GenPart_pdgId[index])
        if abs(pdg_id) != 13 or int(tree.GenPart_status[index]) != 1:
            continue
        pt = float(tree.GenPart_pt[index])
        phi = float(tree.GenPart_phi[index])
        pz = float(tree.GenPart_pz[index])
        mass = float(tree.GenPart_mass[index])
        momentum = math.hypot(pt, pz)
        row = dict(index=index, pdg_id=pdg_id, pt=pt, p=momentum, pz=pz,
                   px=pt * math.cos(phi), py=pt * math.sin(phi),
                   energy=math.hypot(momentum, mass),
                   x_cm=float(tree.GenPart_vx[index]),
                   y_cm=float(tree.GenPart_vy[index]),
                   z_cm=float(tree.GenPart_vz[index]))
        mother = int(tree.GenPart_genPartIdxMother[index])
        seen = {index}
        while mother >= 0 and mother < int(tree.nGenPart):
            if mother in seen:
                raise ValueError("Cycle in saved GenPart mothers")
            seen.add(mother)
            if abs(int(tree.GenPart_pdgId[mother])) != 13:
                break
            mother = int(tree.GenPart_genPartIdxMother[mother])
        if mother >= int(tree.nGenPart):
            raise ValueError("Invalid GenPart mother index")
        row["parent_index"] = mother
        row["parent_pdg_id"] = int(tree.GenPart_pdgId[mother]) if mother >= 0 else 0
        row["parent_kind"] = parent_kind(row["parent_pdg_id"])
        row["parent_start_cm"] = [float(getattr(tree, "GenPart_v" + name)[mother])
                                  for name in ("x", "y", "z")] if mother >= 0 else None
        row["parent_p_GeV"] = math.hypot(float(tree.GenPart_pt[mother]),
                                         float(tree.GenPart_pz[mother])) if mother >= 0 else None
        row["parent_flight_cm"] = (math.sqrt(sum((row[name + "_cm"] -
            float(getattr(tree, "GenPart_v" + name)[mother])) ** 2 for name in ("x", "y", "z")))
            if mother >= 0 else None)
        row["radius_at_plus700_cm"] = line_radius(row, 700.)
        row["radius_at_minus700_cm"] = line_radius(row, -700.)
        row["loose_two_plane"] = all(value is not None and 30. <= value <= 800.
            for value in (row["radius_at_plus700_cm"], row["radius_at_minus700_cm"]))
        muons.append(row)
    return muons


def summarize_event(muons, pairs):
    forward = [muon for muon in muons if muon["pz"] < 0 and
               0. <= muon["z_cm"] <= 15100. and math.hypot(muon["x_cm"], muon["y_cm"]) <= 800.]
    opposite = [(first, second) for i, first in enumerate(forward)
                for second in forward[i + 1:] if first["pdg_id"] * second["pdg_id"] < 0]
    summary = dict(n_gen_muons=len(muons), n_forward_muons=len(forward),
                   max_gen_muon_pt=max([muon["pt"] for muon in muons] or [0.]),
                   max_gen_OS_mass=max([pair_mass(*pair) for pair in opposite] or [0.]),
                   has_gen_OS=bool(opposite), n_reco_BB_pairs=len(pairs))
    for threshold in (10, 20, 30, 50):
        useful = [pair for pair in opposite if all(muon["p"] >= threshold for muon in pair)]
        summary[f"gen_OS_p{threshold}"] = bool(useful)
        summary[f"gen_OS_p{threshold}_line"] = any(all(muon["loose_two_plane"] for muon in pair)
                                                   for pair in useful)
        if threshold in (10, 20, 50):
            for variable in ("mass", "pt"):
                summary[f"reco_BB_{variable}_ge{threshold}"] = sum(pair[variable] >= threshold for pair in pairs)
    for flight_cm in (1, 100, 1000):
        summary[f"gen_OS_p10_flight{flight_cm}"] = any(all(muon["p"] >= 10. and
            muon["parent_flight_cm"] is not None and muon["parent_flight_cm"] <= flight_cm
            for muon in pair) for pair in opposite)
    return summary


def extract_inventory(inventory_path, output, receipt_path, maximum_sources=None):
    import ROOT
    inventory = json.loads(Path(inventory_path).read_text())
    inputs = []
    seen = set()
    event_count = 0
    source_rows = inventory["cohort_receipts"][:maximum_sources]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as handle:
        for source_row in source_rows:
            result_path = Path(source_row["path"])
            if sha256(result_path) != source_row["sha256"]:
                raise ValueError("Completed source result changed")
            result = json.loads(result_path.read_text())
            if not result["complete"]:
                raise ValueError("Source has not completed")
            selected_path = result_path.parent.parent / "selected.json"
            descriptor = json.loads(selected_path.read_text())
            cohort_path = result_path.parent.parent / "nano_cohort.json"
            if sha256(cohort_path) != result["descriptor_sha256"]:
                raise ValueError("Completed Nano cohort descriptor changed")
            cohort = json.loads(cohort_path.read_text())
            if sha256(selected_path) != cohort["source_descriptor_sha256"]:
                raise ValueError("Original selection descriptor changed")
            selected = {tuple(row["event_id"]): row for row in descriptor["selected_events"]}
            nano = result["merged_nano"]
            path = nano["path"]
            root_file = ROOT.TFile.Open(path, "READ")
            if not root_file or root_file.IsZombie():
                raise OSError(path)
            tree = root_file.Get("Events")
            if int(tree.GetEntries()) != len(selected):
                raise ValueError("Merged Nano and source selections differ")
            branches = ["run", "luminosityBlock", "event", "genWeight", "shiftSamplingProbability",
                        "shiftSamplingGenWeight", "nGenPart", "GenPart_*", "nShiftMuon",
                        "ShiftMuon_genPartIdx", "ShiftMuon_hitGenPartIdx", "ShiftMuon_topology",
                        "ShiftMuon_simTruthMatched", "nShiftDimuonVertex", "ShiftDimuonVertex_topologyMin",
                        "ShiftDimuonVertex_topologyMax", "ShiftDimuonVertex_mass", "ShiftDimuonVertex_pt",
                        "ShiftDimuonVertex_isOS", "ShiftDimuonVertex_muonIdx1", "ShiftDimuonVertex_muonIdx2"]
            tree.SetBranchStatus("*", 0)
            for name in branches:
                tree.SetBranchStatus(name, 1)
            source_seen = set()
            for tree_row in tree:
                identity = tuple(int(getattr(tree_row, name)) for name in ("run", "luminosityBlock", "event"))
                if identity in seen or identity not in selected:
                    raise ValueError("Duplicated or unexpected original event")
                seen.add(identity)
                source_seen.add(identity)
                original = selected[identity]
                for branch, key in (("genWeight", "native_weight"),
                                    ("shiftSamplingProbability", "sampling_probability"),
                                    ("shiftSamplingGenWeight", "sampling_gen_weight")):
                    expected = original[key]
                    if branch == "genWeight" and tree.GetLeaf(branch).GetTypeName() == "Float_t":
                        expected = struct.unpack("f", struct.pack("f", expected))[0]
                    if not math.isclose(float(getattr(tree_row, branch)), expected, rel_tol=1e-11, abs_tol=1e-100):
                        raise ValueError("Native or sampling weight changed: " + branch)
                muons = extract_muons(tree_row)
                pairs = []
                for index in range(int(tree_row.nShiftDimuonVertex)):
                    if int(tree_row.ShiftDimuonVertex_topologyMin[index]) != 2 or int(tree_row.ShiftDimuonVertex_topologyMax[index]) != 2:
                        continue
                    first = int(tree_row.ShiftDimuonVertex_muonIdx1[index])
                    second = int(tree_row.ShiftDimuonVertex_muonIdx2[index])
                    pairs.append(dict(mass=float(tree_row.ShiftDimuonVertex_mass[index]),
                        pt=float(tree_row.ShiftDimuonVertex_pt[index]), isOS=int(tree_row.ShiftDimuonVertex_isOS[index]),
                        gen_indices=[int(tree_row.ShiftMuon_genPartIdx[i]) for i in (first, second)],
                        hit_gen_indices=[int(tree_row.ShiftMuon_hitGenPartIdx[i]) for i in (first, second)]))
                row = dict(event_id=identity, source_id=descriptor["source_id"], stratum=result["stratum"],
                    native_weight=original["native_weight"], sampling_probability=original["sampling_probability"],
                    corrected_weight=original["sampling_gen_weight"], training_fold=original["training_fold"],
                    feature_stratum=original["feature_stratum"], normalization=result["normalization"],
                    muons=muons, reco_BB_pairs=pairs, features=summarize_event(muons, pairs),
                    physics_scope="Legacy predecayed muons: detector response conditional on supplied GEN graph; parent survival unvalidated")
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                event_count += 1
            if source_seen != set(selected):
                raise ValueError("Missing original events")
            root_file.Close()
            inputs.append(dict(source_id=descriptor["source_id"], stratum=result["stratum"], events=len(source_seen),
                nano=nano, selection_path=str(selected_path), selection_sha256=sha256(selected_path),
                result_path=str(result_path), result_sha256=source_row["sha256"],
                normalization=result["normalization"]))
            print(json.dumps(dict(source=descriptor["source_id"], events=len(source_seen), cumulative_events=event_count)), flush=True)
    receipt = dict(schema="shift-QCD-GEN-reco-predictor-inputs-v1", complete=True,
        inventory_path=str(Path(inventory_path).resolve()), inventory_sha256=sha256(inventory_path),
        events=event_count, sources=inputs, output=str(output.resolve()), output_sha256=sha256(output),
        original_event_ids_unique=True, native_and_sampling_weights_unchanged=True,
        material_survival_calibration=False, producer_sha256=sha256(__file__),
        ancestry_scope="Nearest saved non-muon ancestor in pruned GenPart; intermediates can be missing, so parent fields are not guaranteed immediate physical mothers")
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")


def read_events(path):
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    identities = [tuple(row["event_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Original events must be unique, including across training partitions")
    for row in rows:
        probability = row["sampling_probability"]
        native = row["native_weight"]
        if not math.isfinite(probability) or not 0. < probability <= 1. or not math.isfinite(native):
            raise ValueError("Invalid original sampling probability or native weight")
        if not math.isclose(row["corrected_weight"], native / probability, rel_tol=1e-11, abs_tol=1e-100):
            raise ValueError("The original inclusion correction must appear exactly once")
    return rows


def physics_weights(rows):
    trials = {}
    sources = set()
    sigma = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id in sources:
            continue
        sources.add(source_id)
        stratum = row["stratum"]
        trials[stratum] = trials.get(stratum, 0) + row["normalization"]["full_source_trials"]
        sigma.setdefault(stratum, []).append(row["normalization"]["sigma_nd_pb"])
    means = {key: math.fsum(values) / len(values) for key, values in sigma.items()}
    return [row["corrected_weight"] * means[row["stratum"]] / trials[row["stratum"]] for row in rows]


def keep_rule(row, rule):
    features = row["features"]
    if rule == "all":
        return True
    if rule == "two_gen_muons":
        return features["n_gen_muons"] >= 2
    if rule == "gen_OS_mass_gt10":
        return features["max_gen_OS_mass"] > 10.
    if rule.startswith("gen_two_p"):
        threshold = int(rule.split("p", 1)[1].split("_", 1)[0])
        muons = [muon for muon in row["muons"] if muon["pz"] < 0. and
                 0. <= muon["z_cm"] <= 15100. and math.hypot(muon["x_cm"], muon["y_cm"]) <= 800.
                 and muon["p"] >= threshold]
        if rule.endswith("_line"):
            muons = [muon for muon in muons if muon["loose_two_plane"]]
        return len(muons) >= 2
    return bool(features[rule])


def effective_count(weights):
    denominator = math.fsum(weight * weight for weight in weights)
    return math.fsum(weights) ** 2 / denominator if denominator else 0.


def summarize_rule(rows, weights, rule):
    kept = [keep_rule(row, rule) for row in rows]
    counts = [row["features"]["n_reco_BB_pairs"] for row in rows]
    denominator = math.fsum(weight * count for weight, count in zip(weights, counts))
    raw_pairs = sum(counts)
    raw_retention = sum(count for count, keep in zip(counts, kept) if keep) / raw_pairs if raw_pairs else None
    weighted_retention = math.fsum(weight * count for weight, count, keep in zip(weights, counts, kept) if keep) / denominator if denominator else None
    return dict(rule=rule, events=len(rows), raw_kept_events=sum(kept),
        raw_event_fraction=sum(kept) / len(rows) if rows else None,
        GEN_population_fraction=(math.fsum(1. / row["sampling_probability"] for row, keep in zip(rows, kept) if keep) /
                                 math.fsum(1. / row["sampling_probability"] for row in rows)) if rows else None,
        BB_pairs=raw_pairs, BB_pairs_kept=sum(count for count, keep in zip(counts, kept) if keep),
        BB_raw_retention=raw_retention,
        BB_raw_retention_event_cluster_error=math.sqrt(math.fsum((count * (keep - raw_retention)) ** 2
            for count, keep in zip(counts, kept))) / raw_pairs if raw_pairs else None,
        BB_weighted_retention=weighted_retention,
        BB_weighted_retention_event_cluster_error=math.sqrt(math.fsum((weight * count * (keep - weighted_retention)) ** 2
            for weight, count, keep in zip(weights, counts, kept))) / denominator if denominator else None,
        BB_event_effective_count=effective_count([weight * count for weight, count in zip(weights, counts)]),
        BB_mass_ge20_kept=sum(row["features"]["reco_BB_mass_ge20"] for row, keep in zip(rows, kept) if keep),
        BB_pt_ge20_kept=sum(row["features"]["reco_BB_pt_ge20"] for row, keep in zip(rows, kept) if keep))


def muon_table(rows, weights, key, edges=None, categories=None):
    table = []
    groups = list(zip(edges[:-1], edges[1:])) if edges else [(category, category) for category in categories]
    for low, high in groups:
        total = matched = weighted_total = weighted_matched = 0.
        squared_denominator = 0.
        event_counts = []
        for row, weight in zip(rows, weights):
            matched_indices = {index for pair in row["reco_BB_pairs"] for index in pair["hit_gen_indices"] if index >= 0}
            event_total = event_matched = 0
            for muon in row["muons"]:
                value = muon.get(key)
                if value is None or not ((low <= value < high) if edges else value == low):
                    continue
                match = muon["index"] in matched_indices
                total += 1
                matched += match
                event_total += 1
                event_matched += match
                weighted_total += weight
                weighted_matched += weight * match
            event_counts.append((event_total, event_matched, weight))
            squared_denominator += (weight * event_total) ** 2
        raw_ratio = matched / total if total else None
        weighted_ratio = weighted_matched / weighted_total if weighted_total else None
        raw_error = math.sqrt(math.fsum((nmatch - raw_ratio * ntotal) ** 2
            for ntotal, nmatch, _ in event_counts)) / total if total else None
        weighted_error = math.sqrt(math.fsum((weight * (nmatch - weighted_ratio * ntotal)) ** 2
            for ntotal, nmatch, weight in event_counts)) / weighted_total if weighted_total else None
        table.append(dict(low=low, high=high, raw_muons=int(total), raw_BB_hit_matched_muons=int(matched),
            raw_fraction=raw_ratio, raw_fraction_event_cluster_error=raw_error,
            weighted_fraction=weighted_ratio, weighted_fraction_event_cluster_error=weighted_error,
            denominator_effective_original_event_count=weighted_total ** 2 / squared_denominator if squared_denominator else 0.))
    return table


def origin_summary(rows, weights):
    counts = Counter()
    parents = Counter()
    groups = defaultdict(list)
    pairs = []
    masses = []
    for row, weight in zip(rows, weights):
        lookup = {muon["index"]: muon for muon in row["muons"]}
        parents.update(muon["parent_pdg_id"] for muon in row["muons"])
        for pair in row["reco_BB_pairs"]:
            matched = [lookup[index] for index in pair["hit_gen_indices"] if index in lookup]
            if len(matched) != 2:
                category = "unmatched_leg"
            elif all(muon["parent_kind"] in ("pion", "kaon") for muon in matched):
                category = "two_pion_or_kaon"
            elif any(muon["parent_kind"] in ("charm", "bottom") for muon in matched):
                category = "heavy_flavour_leg"
            elif matched[0]["parent_index"] == matched[1]["parent_index"]:
                category = "same_saved_parent"
            else:
                category = "other_two_matched"
            truth_mass = pair_mass(*matched) if len(matched) == 2 else None
            if truth_mass is not None:
                masses.append(truth_mass)
            counts[category] += 1
            groups[category].append(weight)
            pairs.append(dict(event_id=row["event_id"], source_id=row["source_id"], stratum=row["stratum"],
                native_weight=row["native_weight"], sampling_probability=row["sampling_probability"],
                corrected_weight=row["corrected_weight"], training_fold=row["training_fold"], category=category,
                matched_parent_pdgs=[muon["parent_pdg_id"] for muon in matched],
                matched_parent_flights_cm=[muon["parent_flight_cm"] for muon in matched],
                matched_truth_muon_pdgs=[muon["pdg_id"] for muon in matched], truth_pair_mass=truth_mass,
                reco_mass=pair["mass"], reco_pt=pair["pt"], reco_isOS=pair["isOS"]))
    return dict(schema="shift-QCD-BothBoth-truth-origin-diagnostic-v1",
        scope="Legacy vacuum-predecay detector response; nearest saved ancestors in pruned Nano, not guaranteed immediate physical mothers",
        counts=dict(counts), sumweights_by_category={key: math.fsum(values) for key, values in groups.items()},
        effective_originals_by_category={key: effective_count(values) for key, values in groups.items()},
        max_hit_associated_true_pair_mass=max(masses or [0.]),
        hit_associated_true_mass_ge10=sum(value >= 10. for value in masses),
        hit_associated_true_mass_ge20=sum(value >= 20. for value in masses),
        parent_pdg_counts=parents.most_common(30), BB_events=pairs)


def analyze_events(events_path, report_path, plot_directory=None):
    rows = read_events(events_path)
    weights = physics_weights(rows)
    origins = origin_summary(rows, weights)
    origin_path = Path(report_path).with_name("BB_origin_and_mass_response.json")
    origin_path.write_text(json.dumps(origins, indent=2) + "\n")
    rules = ["all", "two_gen_muons", "gen_two_p10", "gen_two_p20", "gen_two_p30", "gen_two_p50", "gen_two_p20_line",
             "gen_OS_p10", "gen_OS_p20", "gen_OS_p30", "gen_OS_p50",
             "gen_OS_p10_line", "gen_OS_p20_line", "gen_OS_p10_flight1", "gen_OS_p10_flight100",
             "gen_OS_p10_flight1000", "gen_OS_mass_gt10"]
    report = dict(schema="shift-QCD-GEN-reco-predictor-study-v1", complete=True,
        events=len(rows), events_sha256=sha256(events_path),
        independent_originals=True, material_survival_calibration=False,
        scope="Legacy predecayed muon detector response only; no parent-in-material acceptance or rate validation",
        ancestry_scope="Parent fields mean nearest saved non-muon ancestor in pruned GenPart; quark/proton ancestors do not establish prompt muons",
        partition_definition="Existing original-event training_fold: 0-5 training, 6-7 development, 8-9 development-test; partitions were not an untouched physics closure sample",
        cost_definition="Expected number of full detector events; no measured per-event CPU model",
        uncertainty_definition="Empirical standard errors group all muon/pair contributions from one original event before squaring; zero observed losses do not establish a zero-loss population",
        hit_association_definition="Saved ShiftMuon_hitGenPartIdx; no additional hit-purity requirement",
        normalization="Native W divided by original inclusion p once; independent full-source trial denominators combined within each bin",
        raw_BB_pairs=sum(row["features"]["n_reco_BB_pairs"] for row in rows),
        raw_BB_events=sum(row["features"]["n_reco_BB_pairs"] > 0 for row in rows),
        raw_BB_mass_ge20=sum(row["features"]["reco_BB_mass_ge20"] for row in rows),
        raw_BB_pt_ge20=sum(row["features"]["reco_BB_pt_ge20"] for row in rows),
        origin_summary=dict(path=str(origin_path.resolve()), sha256=sha256(origin_path), counts=origins["counts"]),
        rules=[summarize_rule(rows, weights, rule) for rule in rules], partitions={}, by_stratum={},
        tail_events=[dict(event_id=row["event_id"], source_id=row["source_id"], stratum=row["stratum"],
            native_weight=row["native_weight"], sampling_probability=row["sampling_probability"],
            corrected_weight=row["corrected_weight"], training_fold=row["training_fold"],
            features=row["features"], reco_BB_pairs=row["reco_BB_pairs"],
            matched_muons=[muon for muon in row["muons"] if muon["index"] in {
                index for pair in row["reco_BB_pairs"] for index in pair["hit_gen_indices"]}])
            for row in rows if row["features"]["reco_BB_mass_ge20"] or row["features"]["reco_BB_pt_ge20"]])
    for name, folds in (("training", set(range(6))), ("development", {6, 7}), ("development_test", {8, 9})):
        selected = [(row, weight) for row, weight in zip(rows, weights) if row["training_fold"] in folds]
        report["partitions"][name] = [summarize_rule([row for row, _ in selected], [weight for _, weight in selected], rule) for rule in rules]
    for stratum in sorted({row["stratum"] for row in rows}):
        selected = [(row, weight) for row, weight in zip(rows, weights) if row["stratum"] == stratum]
        report["by_stratum"][stratum] = [summarize_rule([row for row, _ in selected], [weight for _, weight in selected], rule) for rule in rules]
    report["muon_momentum_GeV"] = muon_table(rows, weights, "p", edges=[0., 5., 10., 20., 30., 50., 100., 200., 500., 1000., 10000.])
    report["parent_flight_cm"] = muon_table(rows, weights, "parent_flight_cm", edges=[0., .01, 1., 10., 100., 1000., 3000., 10000., 100000.])
    report["muon_birth_z_cm"] = muon_table(rows, weights, "z_cm", edges=[-100000., -1100., 0., 1100., 5000., 10000., 13000., 14000., 14700., 14900., 100000.])
    report["parent_kind"] = muon_table(rows, weights, "parent_kind", categories=["pion", "kaon", "charm", "bottom", "resonance", "other", "unknown"])
    if plot_directory:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(13., 4.))
        for axis, (key, xlabel, factor) in zip(axes[:2], (("muon_momentum_GeV", "GEN muon momentum [GeV]", 1.),
                                                       ("parent_flight_cm", "Saved ancestor-to-muon distance [m]", .01))):
            groups = report[key]
            centers = [(max(row["low"], .001) * row["high"]) ** .5 * factor for row in groups]
            values = [100. * row["raw_fraction"] if row["raw_fraction"] is not None else float("nan") for row in groups]
            errors = [100. * row["raw_fraction_event_cluster_error"] if row["raw_fraction_event_cluster_error"] is not None else 0. for row in groups]
            axis.errorbar(centers, values, yerr=errors, fmt="o-")
            axis.set_xscale("log")
            axis.set_xlabel(xlabel)
            axis.grid(alpha=.25)
        groups = report["parent_kind"]
        axes[2].bar([row["low"] for row in groups], [100. * (row["raw_fraction"] or 0.) for row in groups],
                    yerr=[100. * (row["raw_fraction_event_cluster_error"] or 0.) for row in groups])
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(axis="y", alpha=.25)
        axes[0].set_ylabel("GEN muons hit-matched to a Both-Both pair [%]")
        figure.suptitle(f"QCD detector-response diagnostic: {len(rows):,} original events")
        figure.text(.5, .01, "Raw selected sample. Supplied vacuum-decay muons: parent survival in material is unvalidated.\n"
                    "Pruned GEN ancestry can omit physical intermediates. No hard cut or physical acceptance is established.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0., .10, 1., .93))
        directory = Path(plot_directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "legacy_gen_to_BothBoth.pdf"
        figure.savefig(path)
        plt.close(figure)
        report["plots"] = [dict(path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size)]
    Path(report_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("events", "raw_BB_pairs", "raw_BB_events", "raw_BB_mass_ge20", "raw_BB_pt_ge20")}))


def plot_material_paths(paths_path, scores_path, plot_directory, receipt_path):
    """Show a deliberately stratified geometry check without inventing a weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = json.loads(Path(paths_path).read_text())
    scores = json.loads(Path(scores_path).read_text())
    if scores.get("usable_as_physics_weight") is not False or scores.get("diagnostic_only") is not True:
        raise ValueError("Material path scores must remain diagnostic, never physics weights")
    lookup = {row["id"]: row for row in paths["paths"]}
    if len(lookup) != len(paths["paths"]):
        raise ValueError("Material paths need unique original-parent identities")
    if {row["id"] for row in scores["paths"]} != set(lookup):
        raise ValueError("Material scores do not match the exact supplied parent paths")
    categories = sorted({row["category"] for row in lookup.values()})
    figure, axes = plt.subplots(1, 2, figsize=(11., 4.4))
    summary = {}
    for category in categories:
        chosen = [(lookup[row["id"]], row) for row in scores["paths"] if lookup[row["id"]]["category"] == category]
        depths = [float(row["interaction_depth_proxy"]) for _, row in chosen]
        if any(not math.isfinite(depth) or depth < 0. for depth in depths):
            raise ValueError("Invalid integrated interaction-depth diagnostic")
        label = category.replace("_", " ") + f" ({len(chosen)})"
        flights = [max(path["parent_flight_cm"] * .01, 1.e-5) for path, _ in chosen]
        displayed = [max(depth, 1.e-5) for depth in depths]
        axes[0].scatter(flights, displayed, label=label, alpha=.7, s=20)
        ordered = sorted(depths)
        median = (ordered[len(ordered) // 2] + ordered[(len(ordered) - 1) // 2]) / 2 if ordered else None
        summary[category] = dict(parents=len(chosen), median_interaction_depth_proxy=median,
            depth_proxy_gt10=sum(depth > 10. for depth in depths),
            full_saved_chord_coverage=sum(bool(row.get("full_saved_chord_coverage")) for _, row in chosen))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Saved ancestor-to-muon distance [m]")
    axes[0].set_ylabel("Straight-chord material depth proxy [interaction lengths]")
    axes[0].axhline(1., color="0.5", linestyle=":")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)
    text = ["Deliberately chosen parent paths:", ""]
    for category, result in summary.items():
        text += [category.replace("_", " ") + f": {result['parents']} parents",
                 f"  Median depth: {result['median_interaction_depth_proxy']:.3g}",
                 f"  Above 10 lengths: {result['depth_proxy_gt10']}/{result['parents']}", ""]
    axes[1].axis("off")
    axes[1].text(.02, .98, "\n".join(text), va="top", fontsize=10)
    figure.suptitle("Do saved QCD ancestry paths cross substantial material?")
    figure.text(.5, .015, "Geometry diagnostic only. This is neither a survival probability nor an event weight.\n"
                "Pruned ancestry and straight chords omit intermediates, field bending and showers. Depths below 1e-5 are plotted at 1e-5.\n"
                f"The {len(lookup)} intentionally selected parents are not a representative rate sample.", ha="center", fontsize=8)
    figure.tight_layout(rect=(0., .14, 1., .94))
    output = Path(plot_directory) / "planned_parent_material.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    receipt = dict(schema="shift-QCD-planned-parent-material-plot-v1", complete=True, summary=summary,
        paths=dict(path=str(Path(paths_path).resolve()), sha256=sha256(paths_path)),
        scores=dict(path=str(Path(scores_path).resolve()), sha256=sha256(scores_path)),
        PDF=dict(path=str(output.resolve()), bytes=output.stat().st_size, sha256=sha256(output)),
        usable_as_physics_weight=False, representative_survival_measurement=False)
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")


def material_control_rows(inventory):
    """Keep missing stages unknown and compare only clean completed audits."""
    if inventory.get("schema") != "shift-material-aware-control-plot-inputs-v1":
        raise ValueError("Unknown material-control plot inventory")
    rows, inputs, pending, seen = [], {}, [], set()
    for case in inventory["cases"]:
        identity = tuple(case["event_id"])
        key = (identity, case["variant"])
        if key in seen:
            raise ValueError("Repeated original event and control variant")
        seen.add(key)
        sim_path = Path(case["sim_audit"])
        if not sim_path.is_file():
            pending.append(dict(case=case, reason="SIM audit not ready"))
            continue
        sim = json.loads(sim_path.read_text())
        if sim.get("schema") != "shift-material-scout-SIM-audit-v1" or not sim.get("passed"):
            pending.append(dict(case=case, reason="SIM audit did not pass"))
            continue
        matches = [row for row in sim["events"] if tuple(row["event_id"]) == identity]
        if len(matches) != 1:
            raise ValueError("Control event does not appear exactly once in its SIM audit")
        event = matches[0]
        inputs[str(sim_path)] = sha256(sim_path)
        row = dict(event_id=list(identity), label=case["label"], variant=case["variant"],
            GEN_muons=int(event["GEN_muons"]),
            true_CSC_muons=int(event["muons_with_required_CSC_stations_each_endcap"]),
            required_CSC_stations_each_endcap=int(sim["minimum_CSC_stations_each_endcap"]),
            reco_Both_muons=None, reco_BothBoth_pairs=None, reco_BothBoth_pair_isOS=None,
            reco_BothBoth_mass_GeV=None,
            reconstruction_status="not ready")
        response_path = Path(case["response_audit"]) if case.get("response_audit") else None
        if response_path and response_path.is_file():
            response = json.loads(response_path.read_text())
            if response.get("schema") != "shift-material-scout-reconstruction-response-v1":
                raise ValueError("Unknown reconstruction-response audit schema")
            if response.get("passed"):
                matches = [entry for entry in response["events"] if tuple(entry["event_id"]) == identity]
                if len(matches) != 1:
                    raise ValueError("Control event does not appear exactly once in its reconstruction audit")
                reconstructed = matches[0]
                row["reco_Both_muons"] = sum(int(muon["topology"]) == 2 for muon in reconstructed["muons"])
                pairs = [pair for pair in reconstructed["pairs"] if int(pair["topologyMin"]) == 2 and int(pair["topologyMax"]) == 2]
                row["reco_BothBoth_pairs"] = len(pairs)
                row["reco_BothBoth_pair_isOS"] = [int(pair["isOS"]) for pair in pairs]
                row["reco_BothBoth_mass_GeV"] = [float(pair["mass"]) for pair in pairs]
                row["reconstruction_status"] = "complete"
                inputs[str(response_path)] = sha256(response_path)
            else:
                row["reconstruction_status"] = "audit did not pass"
        rows.append(row)
    return rows, inputs, pending


def plot_material_controls(inventory_path, output_directory, receipt_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inventory = json.loads(Path(inventory_path).read_text())
    rows, inputs, pending = material_control_rows(inventory)
    if not rows:
        raise ValueError("No clean completed SIM controls are ready to plot")
    requirements = {row["required_CSC_stations_each_endcap"] for row in rows}
    if len(requirements) != 1:
        raise ValueError("SIM controls use different CSC station definitions")
    required = next(iter(requirements))
    figure, axis = plt.subplots(figsize=(11., 2.5 + .55 * len(rows)))
    axis.axis("off")
    labels = [row["label"] + "\n" + str(row["event_id"][0]) + ":" + str(row["event_id"][2]) for row in rows]
    columns = ["GEN muons\n(pre-material)", f"Simulated muons\n>={required} CSC stations/endcap",
               "Reconstructed\nBoth muons", "Both-Both pairs\n(total; OS / SS)"]
    values = [[str(row["GEN_muons"]), str(row["true_CSC_muons"]),
               str(row["reco_Both_muons"]) if row["reco_Both_muons"] is not None else "not ready",
               (str(row["reco_BothBoth_pairs"]) + "; " + str(row["reco_BothBoth_pair_isOS"].count(1)) + " / " + str(row["reco_BothBoth_pair_isOS"].count(0))
                + ("\nm = " + ", ".join(f"{mass:.2f}" for mass in row["reco_BothBoth_mass_GeV"]) + " GeV" if row["reco_BothBoth_pairs"] else ""))
               if row["reco_BothBoth_pairs"] is not None else "not ready"] for row in rows]
    table = axis.table(cellText=values, rowLabels=labels, colLabels=columns,
                       cellLoc="center", rowLoc="right", loc="center", colWidths=[.18, .27, .23, .23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_height(.9 / (len(rows) + 1))
        if row_index == 0:
            cell.set_facecolor("#dbe5ef")
        elif column_index >= 0:
            cell.set_facecolor("#f7f7f7" if row_index % 2 else "#edf3f8")
            if rows[row_index - 1]["reco_BothBoth_pairs"]:
                cell.set_facecolor("#d9ebda")
            if cell.get_text().get_text() == "not ready":
                cell.get_text().set_color("#777777")
    originals = len({tuple(row["event_id"]) for row in rows})
    figure.suptitle("QCD material-aware controls: measured event counts", fontsize=14)
    cap_note = ("The cap0 row changes only the offline vertex limit on the same saved reconstruction.\n"
                if inventory.get("cap_comparison_audit") else "")
    figure.text(.5, .025, f"{len(rows)} control variants from {originals} original hard events. Repeated variants are not independent events.\n"
                "Intentionally selected, unweighted diagnostics; no physical rates or population efficiency. Missing reconstruction is unknown, not zero.\n"
                "The CSC station threshold is a proxy: reconstructed Both tracks can occur below it. It is not a hard event veto.\n"
                + cap_note +
                "CSC hits precede detector readout and reconstruction. Trigger acceptance has not been established.", ha="center", fontsize=8)
    figure.subplots_adjust(left=.29, right=.98, bottom=.19, top=.87)
    output = Path(output_directory) / "material_aware_controls.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    receipt = dict(schema="shift-material-aware-control-plot-v1", complete=True,
        inventory=dict(path=str(Path(inventory_path).resolve()), sha256=sha256(inventory_path)),
        inputs=[dict(path=path, sha256=digest) for path, digest in sorted(inputs.items())],
        rows=rows, pending=pending, original_hard_events=originals, control_variants=len(rows),
        rates_or_efficiency_claim=False, missing_reconstruction_is_unknown=True,
        PDF=dict(path=str(output.resolve()), sha256=sha256(output), bytes=output.stat().st_size))
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--maximum-sources", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plot-directory", type=Path)
    parser.add_argument("--material-paths", type=Path)
    parser.add_argument("--material-scores", type=Path)
    parser.add_argument("--material-receipt", type=Path)
    parser.add_argument("--material-controls", type=Path)
    parser.add_argument("--material-controls-receipt", type=Path)
    args = parser.parse_args()
    if args.inventory:
        if not args.receipt or not args.events:
            parser.error("Extraction requires --receipt and --events")
        extract_inventory(args.inventory, args.events, args.receipt, args.maximum_sources)
    if args.report:
        if not args.events:
            parser.error("Analysis requires --events")
        analyze_events(args.events, args.report, args.plot_directory)
    if args.material_paths:
        if not args.material_scores or not args.plot_directory or not args.material_receipt:
            parser.error("Material plotting needs --material-scores, --plot-directory and --material-receipt")
        plot_material_paths(args.material_paths, args.material_scores, args.plot_directory, args.material_receipt)
    if args.material_controls:
        if not args.plot_directory or not args.material_controls_receipt:
            parser.error("Material controls need --plot-directory and --material-controls-receipt")
        plot_material_controls(args.material_controls, args.plot_directory, args.material_controls_receipt)
    if not args.inventory and not args.report and not args.material_paths and not args.material_controls:
        parser.error("Choose --inventory extraction or --report analysis")


if __name__ == "__main__":
    main()
