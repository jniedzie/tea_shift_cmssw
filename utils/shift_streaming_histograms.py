#!/usr/bin/env python3
# Code style: AI-code. AI-generated streaming production validation and preparation.
"""Histogram closed Nano cohorts and merge immutable, exposure-aware snapshots.

Run this inside the frozen TEA runtime. It calls the ordinary SHIFT histogrammer
and TEA merger; it does not implement another histogram or ROOT merge engine.
Only complete predetermined GEN-source cohorts may be used for rate previews.
The controller owns submission and cleanup of the original published Nano files.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


METADATA_KEY = "ShiftStreamingNormalization"
WEIGHT_POLICY = "native_gen_weight/reco_inclusion_probability"
WEIGHT_BRANCHES = ("shiftOriginalGenWeight", "shiftSamplingProbability",
                   "shiftSamplingWeight", "shiftSamplingGenWeight")
DEFAULT_PUBLICATION_BYTES = 32 * 1024 * 1024
MAXIMUM_PUBLICATION_BYTES = 512 * 1024 * 1024
REQUIRED_HISTOGRAMS = tuple(
    "dimuon/ShiftDimuonVertex" + category + "_" + variable
    for category in ("", "Both-Both") for variable in ("mass", "pt")
)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, record):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_close(value, expected, label):
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-300):
        raise ValueError("Value differs from frozen cohort: " + label)


def validate_cohort(record):
    """Validate closure before opening ROOT files or preparing any outputs."""
    if record.get("schema") != "shift-streaming-nano-cohort-v1" or record.get("complete") is not True:
        raise ValueError("A complete immutable Nano cohort is required")
    norm = record["normalization"]
    if (norm.get("schema") != "shift-stream-source-exposure-v1" or
            norm.get("stratum") != record["stratum"] or
            norm.get("no_second_filter_efficiency") is not True or
            norm.get("histogram_weight") != WEIGHT_POLICY or
            not isinstance(norm.get("full_source_trials"), int) or
            norm["full_source_trials"] <= 0 or
            not math.isfinite(norm.get("sigma_nd_pb", float("nan"))) or
            norm["sigma_nd_pb"] <= 0):
        raise ValueError("Invalid full GEN exposure authority")
    if not norm.get("source_id") or len(norm.get("gen_sha256", "")) != 64:
        raise ValueError("GEN source identity and checksum are required")
    jobs = record["expected_reco_job_ids"]
    ready = [row["job_id"] for row in record["verified_nano"]]
    if len(jobs) != len(set(jobs)) or len(ready) != len(set(ready)) or set(jobs) != set(ready):
        raise ValueError("Nano cohort is missing predetermined reconstruction jobs")
    selected = {}
    for row in record["selected_events"]:
        identity = tuple(row["event_id"])
        if len(identity) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in identity):
            raise ValueError("Invalid selected event identity")
        if identity in selected:
            raise ValueError("Duplicate selected original event")
        weight, probability = row["native_weight"], row["sampling_probability"]
        if not math.isfinite(weight) or not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Invalid selected event weight or inclusion probability")
        selected[identity] = row
    if sum(row["events"] for row in record["verified_nano"]) != len(selected):
        raise ValueError("Nano counts do not close the selected event cohort")
    if not selected:
        raise ValueError("An empty cohort needs an explicit empty-Nano reference")
    return selected


def stage(path, destination, expected_sha256):
    """Read EOS through native XRootD; every staged payload is SHA-256 checked."""
    import teaHelpers
    path = str(path)
    url = teaHelpers.eos_xrootd_url(path) or path
    if url.startswith("root://"):
        subprocess.run(["xrdcp", "--silent", "--cksum", "adler32", url, str(destination)],
                       check=True, timeout=600)
    else:
        shutil.copyfile(path, destination)
    if expected_sha256 is not None and sha256(destination) != expected_sha256:
        raise ValueError("Input checksum differs: " + path)


def root_file(path, mode="READ"):
    import ROOT
    source = ROOT.TFile.Open(str(path), mode)
    if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
        raise ValueError("Unreadable or recovered ROOT file: " + str(path))
    return source


def audit_nano(path, selected):
    source = root_file(path)
    identities, sumw, sumw2 = [], 0.0, 0.0
    try:
        tree = source.Get("Events")
        if not tree:
            raise ValueError("Missing Nano Events tree")
        for name in WEIGHT_BRANCHES:
            leaf = tree.GetLeaf(name)
            if not leaf or leaf.GetTypeName() != "Double_t":
                raise ValueError("Sampling weights must be Double: " + name)
        for event in tree:
            identity = (int(event.run), int(event.luminosityBlock), int(event.event))
            if identity not in selected:
                raise ValueError("Nano contains an event outside the frozen cohort")
            row = selected[identity]
            native, probability = row["native_weight"], row["sampling_probability"]
            references = (native, probability, 1.0 / probability, native / probability)
            for name, expected in zip(WEIGHT_BRANCHES, references):
                check_close(float(getattr(event, name)), expected, name)
            identities.append(identity)
            weight = native / probability
            sumw += weight
            sumw2 += weight * weight
    finally:
        source.Close()
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate Nano original event")
    return dict(event_ids=identities, events=len(identities), sumw=sumw, sumw2=sumw2)


def merge(inputs, destination, merge_script, scratch):
    job = scratch / (destination.stem + "_merge.json")
    write_json(job, dict(input_files=list(map(str, inputs)), output_file=str(destination),
                         preserve_input_compression=True, hadd_workers=1, hadd_files_per_pass=100))
    subprocess.run([sys.executable, str(merge_script), "--job-file", str(job)],
                   check=True, timeout=1800)


def histogram_summary(path):
    import ROOT
    source = root_file(path, "UPDATE")
    result = {}
    try:
        for name in REQUIRED_HISTOGRAMS:
            histogram = source.Get(name)
            if not histogram:
                raise ValueError("Missing required histogram: " + name)
            # ROOT creates Sumw2 on the first nonunit weight. A completely
            # empty or unit-weight histogram still needs its Poisson errors
            # materialized before it is merged with weighted chunks.
            if histogram.GetSumw2N() == 0:
                histogram.Sumw2()
                source.cd(str(Path(name).parent))
                histogram.Write(Path(name).name, ROOT.TObject.kOverwrite)
            values = [float(histogram.GetBinContent(i)) for i in range(histogram.GetNcells())]
            variances = [float(histogram.GetBinError(i)) ** 2 for i in range(histogram.GetNcells())]
            if not all(math.isfinite(v) for v in values + variances):
                raise ValueError("Nonfinite histogram content: " + name)
            total, variance = sum(values), sum(variances)
            tails = {}
            for threshold in (10., 20., 30., 50., 100.):
                first = histogram.GetXaxis().FindFixBin(threshold)
                tail_sum = sum(values[first:])
                tail_variance = sum(variances[first:])
                tails[str(int(threshold))] = dict(sumw=tail_sum, sumw2=tail_variance,
                    effective_count=tail_sum * tail_sum / tail_variance if tail_variance else 0.0)
            result[name] = dict(entries=float(histogram.GetEntries()), sumw=total, sumw2=variance,
                                effective_count=total * total / variance if variance else 0.0,
                                underflow=values[0], overflow=values[-1], tails_ge_gev=tails)
    finally:
        source.Close()
    return result


def embed_metadata(path, metadata):
    import ROOT
    source = root_file(path, "UPDATE")
    try:
        ROOT.TObjString(json.dumps(metadata, sort_keys=True, allow_nan=False)).Write(
            METADATA_KEY, ROOT.TObject.kOverwrite)
    finally:
        source.Close()


def audit_histogram_exposure(path, expected_events, expected_sumw, expected_sumw2):
    source = root_file(path)
    try:
        raw = source.Get("rawEventsCutFlow")
        event = source.Get("event/Event_nShiftMuon")
        if not raw or not event:
            raise ValueError("Missing histogram event-exposure checks")
        check_close(float(raw.GetBinContent(1)), expected_events, "raw original-event count")
        total = sum(float(event.GetBinContent(i)) for i in range(event.GetNcells()))
        variance = sum(float(event.GetBinError(i)) ** 2 for i in range(event.GetNcells()))
        check_close(total, expected_sumw, "histogram event weight sum")
        check_close(variance, expected_sumw2, "histogram event squared-weight sum")
    finally:
        source.Close()


def file_record(path, published_path=None):
    return dict(path=published_path or str(path), sha256=sha256(path), bytes=path.stat().st_size)


def publication_budget(files, plots, requested_limit=DEFAULT_PUBLICATION_BYTES):
    # Reserve compact JSON receipts before the first upload as well. Workers
    # A larger cohort needs an explicit matching controller reservation. Old
    # frozen descriptors retain their 32 MiB limit.
    if type(requested_limit) is not int or not 0 < requested_limit <= MAXIMUM_PUBLICATION_BYTES:
        raise ValueError("Invalid streaming publication reservation")
    limit = requested_limit
    total = sum(Path(path).stat().st_size for path in files) + sum(row["bytes"] for row in plots) + 65536
    if total > limit:
        raise ValueError(f"Streaming publication exceeds its reservation: {total} > {limit} bytes")
    return dict(maximum_bytes=limit, planned_bytes=total, receipt_reserve_bytes=65536)


def publish(path, directory):
    """Use successful-close EOS publication and verify a SHA-256 readback."""
    import teaHelpers
    destination = str(directory).rstrip("/") + "/" + path.name
    url = teaHelpers.eos_xrootd_url(destination) or destination
    if url.startswith("root://"):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        subprocess.run(["xrdfs", "root://" + parsed.netloc, "mkdir", "-p",
                        str(Path(parsed.path).parent)], check=True, timeout=120)
        subprocess.run(["xrdcp", "--silent", "--posc", "--cksum", "adler32", str(path), url],
                       check=True, timeout=600)
        with tempfile.TemporaryDirectory(prefix="stream_publish_", dir=os.environ.get("TMPDIR")) as temporary:
            stage(url, Path(temporary) / path.name, sha256(path))
    else:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        if Path(destination).exists():
            raise FileExistsError("Refusing to replace immutable published output: " + destination)
        shutil.copyfile(path, destination)
        if sha256(destination) != sha256(path):
            raise ValueError("Published readback checksum differs")
    return destination


def render_plots(inventory, plotter, plot_output, scratch, kinds, caption=None):
    plots = []
    has_references = bool(json.loads(Path(inventory).read_text()).get("reference_samples"))
    for kind in kinds:
        config = scratch / (kind + "_plot_config.py")
        # An exact snapshot is an internal production input. Ordinary users
        # select its published campaign name in shift_paths instead.
        selected_campaign = str(inventory) if kind == "rate" else None
        reference_settings = "" if has_references and kind == "rate" else "shift_paths.cross_sections = {}\n"
        beam_label = "partial SM; mu-enhanced QCD" if has_references and kind == "rate" else "partial mu-enhanced QCD; diagnostic"
        legend_bottom = 0.2 if has_references and kind == "rate" else 0.56
        raw_settings = ""
        if kind == "raw":
            raw_settings = f"""
samples = []
for row in streaming_histogram_inputs({str(inventory)!r}, raw=True):
    bin_name = row["stratum"].removeprefix("qcd_")
    samples.append(Sample(name=row["stratum"], file_path=row["histogram_path"],
        type=SampleType.background, cross_section=1, initial_weight_sum=1,
        fill_color=qcd_colors[bin_name], fill_alpha=0.7, line_alpha=0,
        marker_size=0, legend_description=f"QCD {{get_bin_label(bin_name, 'qcd')}}"))
custom_stacks_order = [sample.name for sample in samples]
for h in histograms:
    h.norm_type = NormalizationType.none
    h.y_label = "processed events" if h.name.startswith("event/") else "reconstructed pairs / GeV"
del lumi_label_value
"""
        config.write_text("import shift_paths\n"
                          "shift_paths.bsm_campaign = None\n"
                          + reference_settings +
                          f"shift_paths.campaign = {selected_campaign!r}\n"
                          "from shift_plotter_config import *\n"
                          "preview_names = ['event/Event_nShiftMuonBothEndcaps', "
                          "'event/Event_nShiftDimuonVertexBoth-Both'] + "
                          "['dimuon/ShiftDimuonVertex' + c + '_' + v "
                          "for c in ('', 'Both-Both') for v in ('mass', 'pt')]\n"
                          "histograms = [h for h in histograms if h.name in preview_names]\n"
                          + raw_settings +
                          f"output_path = {str(plot_output / kind)!r}\n"
                          "for h in histograms:\n"
                          "    h.allow_empty = True\n"
                          "    h.log_y = True\n"
                          "    h.rebin = 1 if h.name.startswith('event/') else 5\n"
                          f"    if {caption!r}:\n"
                          f"        h.legend_label = {caption!r}\n"
                          "cms_label = CmsLabel.private_sim\n"
                          f"beam_label = {beam_label!r}\n"
                          "canvas_size = (1100, 750)\n"
                          f"legends = {{SampleType.background: Legend(0.59, {legend_bottom}, 0.93, 0.90, 'f', text_size=15)}}\n")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(scratch) + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run([sys.executable, str(plotter), "--config", config.name],
                       cwd=scratch, env=environment, check=True, timeout=1800)
        plots.extend(file_record(path) for path in sorted((plot_output / kind).rglob("*.pdf")))
    return plots


def diagnostic(args):
    """Make raw plots of explicitly chosen test events, with no rate estimator."""
    descriptor = json.loads(args.descriptor.read_text())
    if (descriptor.get("schema") != "shift-streaming-nano-diagnostic-v1" or
            descriptor.get("diagnostic_only") is not True or
            descriptor.get("normalization_ready") is not False):
        raise ValueError("An explicitly unnormalized diagnostic descriptor is required")
    if args.plotter and not args.plot_output:
        raise ValueError("--plot-output is required with --plotter")
    rows = descriptor["nano_inputs"]
    if not rows or not 1 <= sum(len(row["event_ids"]) for row in rows) <= 20:
        raise ValueError("Raw diagnostic view requires 1..20 explicit test originals")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples, seen, audits = [], set(), []
    with tempfile.TemporaryDirectory(prefix="stream_diagnostic_", dir=os.environ.get("TMPDIR")) as temporary:
        scratch = Path(temporary)
        groups = defaultdict(list)
        for index, original in enumerate(rows):
            row = dict(original)
            if len(row["event_ids"]) != len(row["native_weights"]):
                raise ValueError("Diagnostic native weights do not cover exact event IDs")
            if row.get("reco_receipt"):
                receipt_path = scratch / f"reco{index}.json"
                stage(row["reco_receipt"], receipt_path, row.get("reco_receipt_sha256"))
                receipt = json.loads(receipt_path.read_text())
                identities_sha = hashlib.sha256(json.dumps(row["event_ids"], sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()
                if (receipt.get("complete") is not True or
                        receipt.get("source_descriptor_sha256") != row["source_descriptor_sha256"] or
                        receipt.get("source_gen_sha256") != row["gen_sha256"] or
                        receipt.get("source_stratum") != row["stratum"] or
                        receipt.get("selected_event_ids_sha256") != identities_sha or
                        receipt.get("events") != len(row["event_ids"]) or
                        not receipt.get("nano_sha256") or
                        receipt["nano_sha256"] != receipt.get("nano_readback_sha256")):
                    raise ValueError("Targeted RECO receipt does not match the pinned original GEN event")
                row.update(path=receipt["nano_path"], sha256=receipt["nano_sha256"],
                           reco_receipt_sha256=sha256(receipt_path))
            local = scratch / f"nano{index}.root"
            stage(row["path"], local, row["sha256"])
            source = root_file(local)
            found, muons, pairs, both_pairs = [], 0, 0, 0
            try:
                tree = source.Get("Events")
                if not tree or tree.GetEntries() != len(row["event_ids"]):
                    raise ValueError("Diagnostic Nano does not contain the declared originals")
                for event, identity, native in zip(tree, row["event_ids"], row["native_weights"]):
                    actual = [int(event.run), int(event.luminosityBlock), int(event.event)]
                    if actual != identity or tuple(actual) in seen or not math.isfinite(native):
                        raise ValueError("Diagnostic Nano original-event identity differs or repeats")
                    # Nano's standard genWeight is Float; the source native
                    # Double stays in this receipt. Sampling Double fields are
                    # audited when present, never invented for targeted tests.
                    expected_float = struct.unpack("f", struct.pack("f", native))[0]
                    if not math.isclose(float(event.genWeight), expected_float, rel_tol=2e-6, abs_tol=0):
                        raise ValueError("Diagnostic native GEN weight changed in Nano")
                    if tree.GetLeaf("shiftOriginalGenWeight"):
                        check_close(float(event.shiftOriginalGenWeight), native, "diagnostic native weight")
                    seen.add(tuple(actual)); found.append(actual)
                    muons += int(event.nShiftMuon)
                    pairs += int(event.nShiftDimuonVertex)
                    both_pairs += sum(int(event.ShiftDimuonVertex_topologyMin[i]) == 2 and
                                      int(event.ShiftDimuonVertex_topologyMax[i]) == 2
                                      for i in range(int(event.nShiftDimuonVertex)))
            finally:
                source.Close()
            audits.append(dict(**row, measured_event_ids=found, shift_muons=muons,
                               shift_dimuons=pairs, both_both_dimuons=both_pairs))
            groups[row["stratum"]].append(local)
        for stratum, inputs in sorted(groups.items()):
            nano = scratch / (stratum + "_nano.root")
            merge(inputs, nano, args.merge_script, scratch)
            config = scratch / (stratum + "_raw_config.py")
            config.write_text("from shift_histogrammer_config import *\n"
                              "enableBdtDiagnostics = False\nweightsBranchName = ''\n")
            path = output / (stratum + "_raw.root")
            subprocess.run([str(args.histogrammer), "--config", str(config), "--input_path", str(nano),
                            "--output_hists_path", str(path)], check=True, timeout=1800)
            count = sum(len(row["event_ids"]) for row in rows if row["stratum"] == stratum)
            audit_histogram_exposure(path, count, count, count)
            summary = histogram_summary(path)
            embed_metadata(path, dict(schema="shift-streaming-normalization-v1", stratum=stratum,
                diagnostic_only=True, closed_cohorts=False, normalization_ready=False,
                histogram_weight="one_per_reconstructed_entry", processed_original_events=count,
                physical_rate_claim=False, production_sample=False))
            samples.append(dict(stratum=stratum, path=str(path), raw_path=str(path),
                                histogram=file_record(path), histograms=summary))
        plot_inputs = output / "plot_inputs.json"
        write_json(plot_inputs, dict(schema="shift-streaming-plot-inputs-v1", diagnostic_only=True,
                                    normalization_ready=False, samples=samples))
        plots = []
        if args.plotter:
            plots = render_plots(plot_inputs, args.plotter, args.plot_output, scratch, ("raw",),
                                 descriptor.get("caption", "Targeted test events; no rate claim"))
        budget = publication_budget([row["path"] for row in samples], plots)
        if args.publish_directory:
            for row in samples:
                published = publish(Path(row["path"]), args.publish_directory)
                row.update(path=published, raw_path=published)
                row["histogram"]["path"] = published
            plot_inputs.write_text(json.dumps(dict(schema="shift-streaming-plot-inputs-v1",
                diagnostic_only=True, normalization_ready=False, samples=samples), indent=2) + "\n")
        receipt = dict(schema="shift-streaming-histogram-diagnostic-v1", complete=True,
            diagnostic_only=True, normalization_ready=False, production_sample=False,
            physical_rate_claim=False, descriptor_sha256=sha256(args.descriptor),
            originals=len(seen), nano_audits=audits, samples=samples, plots=plots,
            publication_budget=budget, plot_inputs=file_record(plot_inputs))
        write_json(output / "diagnostic_complete.json", receipt)
        if args.publish_directory:
            publish(output / "diagnostic_complete.json", args.publish_directory)
    print(json.dumps(dict(complete=True, completion=str(output / "diagnostic_complete.json"))))


def retire_consumed_nano(receipt, receipt_path):
    """Delete exact per-event Nano inputs only after the merged replacements exist.

    The factory assigns each original Nano to one cohort and has switched all
    future consumers to merged_nano. GEN, other tiers, and receipts are retained.
    Cleanup failure is recorded separately from completed histogram processing.
    """
    from urllib.parse import urlparse
    import teaHelpers
    from shift_nano_integrity import validate_retirement_proof

    cleanup = dict(schema="shift-streaming-consumed-nano-cleanup-v1", complete=False,
                   cohort_id=receipt["cohort_id"], retained_merged_nano=receipt["merged_nano"],
                   deleted=[], removed_bytes=0)

    def save():
        temporary = receipt_path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(cleanup, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, receipt_path)

    save()
    try:
        validate_retirement_proof(receipt)
        originals, identities = [], set()
        retained = {row["path"] for row in (receipt["merged_nano"], receipt["histogram"], receipt["raw_histogram"])}
        with tempfile.TemporaryDirectory(prefix="stream_cleanup_", dir=os.environ.get("TMPDIR")) as temporary:
            # Recheck every original before the first deletion. No directory globs.
            for index, row in enumerate(receipt["nano_inputs"]):
                url = teaHelpers.eos_xrootd_url(row["path"]) or row["path"]
                parsed = urlparse(url)
                path = "/" + parsed.path.lstrip("/")
                identity = (parsed.netloc, path)
                if (parsed.scheme != "root" or not parsed.netloc.endswith(".cern.ch") or
                        "/shift_cmssw/qcd_pThat" not in path or "/nanoAOD/" not in path or
                        not path.endswith(".root") or row["path"] in retained or identity in identities):
                    raise ValueError("Unsafe or duplicate consumed Nano path: " + row["path"])
                identities.add(identity)
                local = Path(temporary) / f"input{index}.root"
                stage(url, local, row["sha256"])
                originals.append(dict(path=row["path"], url=url, host=parsed.netloc,
                                      server_path=path, sha256=row["sha256"], bytes=local.stat().st_size))
                local.unlink()
            # Published outputs have already passed SHA-256 readback. Their
            # exact retained identities must not alias any consumed input URL.
            for row in (receipt["merged_nano"], receipt["histogram"], receipt["raw_histogram"]):
                parsed = urlparse(teaHelpers.eos_xrootd_url(row["path"]) or row["path"])
                if (parsed.netloc, "/" + parsed.path.lstrip("/")) in identities:
                    raise ValueError("Consumed Nano aliases a retained replacement")
            for row in originals:
                host = "root://" + row["host"]
                subprocess.run(["xrdfs", host, "rm", row["server_path"]], check=True, timeout=120)
                absent = subprocess.run(["xrdfs", host, "stat", row["server_path"]],
                                        capture_output=True, text=True, timeout=120)
                # XRootD's explicit not-found status proves absence. A network
                # or credential error is unknown state and stops cleanup.
                if absent.returncode == 0 or "[3011]" not in absent.stderr + absent.stdout:
                    raise RuntimeError("Cannot verify deletion of " + row["path"])
                cleanup["deleted"].append(dict(path=row["path"], sha256=row["sha256"],
                                               bytes=row["bytes"], verified_absent=True))
                cleanup["removed_bytes"] += row["bytes"]
                save()
        cleanup["complete"] = True
    except Exception as error:
        cleanup["error"] = str(error)
    save()
    return cleanup


def cohort(args):
    descriptor = json.loads(args.descriptor.read_text())
    selected = validate_cohort(descriptor)
    if args.retire_consumed_nano and not args.publish_directory:
        raise ValueError("Consumed Nano cleanup requires checksummed published replacements")
    if args.plotter and not args.plot_output:
        raise ValueError("--plot-output is required with --plotter")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="stream_cohort_", dir=os.environ.get("TMPDIR")) as temporary:
        scratch = Path(temporary)
        inputs, seen, input_records = [], set(), []
        for index, row in enumerate(descriptor["verified_nano"]):
            path = scratch / f"nano{index}.root"
            stage(row["path"], path, row["sha256"])
            audit = audit_nano(path, selected)
            identities = set(map(tuple, audit["event_ids"]))
            if audit["events"] != row["events"] or seen.intersection(identities):
                raise ValueError("Nano count mismatch or overlapping original events")
            seen.update(identities)
            inputs.append(path)
            input_records.append(dict(**row, audit=audit))
        if seen != set(selected):
            raise ValueError("Nano event identities do not close the selected cohort")
        nano = output / "nano_merged.root"
        merge(inputs, nano, args.merge_script, scratch)
        merged_audit = audit_nano(nano, selected)
        if set(map(tuple, merged_audit["event_ids"])) != seen:
            raise ValueError("Nano merge changed original event identities")
        from shift_nano_integrity import compare_in_runtime
        integrity = compare_in_runtime(inputs, nano, getattr(args, "nano_integrity_runtime", None))
        results = {}
        for kind, branch in (("histogram", "shiftSamplingGenWeight"), ("raw_histogram", "")):
            config = scratch / (kind + "_config.py")
            config.write_text("# Generated exact-input overlay for the standard SHIFT configuration.\n"
                              "from shift_histogrammer_config import *\n"
                              "enableBdtDiagnostics = False\n"
                              f"weightsBranchName = {branch!r}\n")
            path = output / ("histograms.root" if branch else "histograms_raw.root")
            with (scratch / (kind + ".log")).open("w") as log:
                command = [str(args.histogrammer), "--config", str(config),
                           "--input_path", str(nano), "--output_hists_path", str(path)]
                try:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    log.flush()
                    tail = (scratch / (kind + ".log")).read_text(errors="replace")[-8192:]
                    raise RuntimeError("Standard SHIFT histogrammer failed:\n" + tail) from error
            audit_histogram_exposure(path, len(selected),
                                     merged_audit["sumw"] if branch else len(selected),
                                     merged_audit["sumw2"] if branch else len(selected))
            summary = histogram_summary(path)
            metadata = dict(schema="shift-streaming-normalization-v1", closed_cohorts=True,
                            stratum=descriptor["stratum"], cohort_ids=[descriptor["cohort_id"]],
                            source_ids=[descriptor["normalization"]["source_id"]],
                            full_source_trials=descriptor["normalization"]["full_source_trials"],
                            sigma_nd_pb=descriptor["normalization"]["sigma_nd_pb"],
                            histogram_weight=WEIGHT_POLICY if branch else "one_per_reconstructed_entry",
                            no_second_filter_efficiency=True, partial_production=True)
            embed_metadata(path, metadata)
            results[kind] = dict(**file_record(path), histograms=summary)
        results["merged_nano"] = file_record(nano)
        plots = []
        if args.plotter:
            plot_inputs = output / "plot_inputs.json"
            write_json(plot_inputs, dict(schema="shift-streaming-plot-inputs-v1", partial_production=True,
                samples=[dict(stratum=descriptor["stratum"], path=results["histogram"]["path"],
                              raw_path=results["raw_histogram"]["path"])],
                cohort_id=descriptor["cohort_id"], view="raw diagnostic only"))
            plots = render_plots(plot_inputs, args.plotter, args.plot_output, scratch, ("raw",))
        budget = publication_budget([row["path"] for row in results.values()], plots,
                                    descriptor.get("maximum_publication_bytes", DEFAULT_PUBLICATION_BYTES))
        if args.publish_directory:
            for row in results.values():
                row["path"] = publish(Path(row["path"]), args.publish_directory)
        receipt = dict(schema="shift-streaming-histogram-cohort-v1", complete=True,
                       cohort_id=descriptor["cohort_id"], stratum=descriptor["stratum"],
                       descriptor_sha256=sha256(args.descriptor), normalization=descriptor["normalization"],
                       expected_reco_job_ids=descriptor["expected_reco_job_ids"],
                       nano_inputs=input_records, nano_audit=merged_audit, plots=plots,
                       merged_nano_integrity=integrity,
                       publication_budget=budget, **results)
        receipt_path = output / "cohort_complete.json"
        write_json(receipt_path, receipt)
        if args.publish_directory:
            publish(receipt_path, args.publish_directory)
            if args.retire_consumed_nano:
                cleanup_path = output / "nano_cleanup.json"
                retire_consumed_nano(receipt, cleanup_path)
                publish(cleanup_path, args.publish_directory)
            # All three ROOT files have complete checksummed EOS readbacks.
            for name in ("nano_merged.root", "histograms.root", "histograms_raw.root"):
                (output / name).unlink()
    print(json.dumps(dict(complete=True, cohort_id=descriptor["cohort_id"],
                          completion=str(receipt_path))))


def snapshot_authorities(receipts, expected_cohort_ids):
    cohorts, sources, gen_hashes, groups = set(), set(), set(), defaultdict(list)
    for row in receipts:
        if row.get("schema") != "shift-streaming-histogram-cohort-v1" or row.get("complete") is not True:
            raise ValueError("Snapshot requires complete histogram cohorts")
        cohort_id, norm = row["cohort_id"], row["normalization"]
        if cohort_id in cohorts or norm["source_id"] in sources or norm["gen_sha256"] in gen_hashes:
            raise ValueError("Snapshot repeats a cohort or full GEN source exposure")
        cohorts.add(cohort_id); sources.add(norm["source_id"]); gen_hashes.add(norm["gen_sha256"])
        groups[row["stratum"]].append(row)
    if cohorts != set(expected_cohort_ids) or len(expected_cohort_ids) != len(cohorts):
        raise ValueError("Snapshot is missing its predetermined source cohorts")
    for stratum, rows in groups.items():
        sigma = rows[0]["normalization"]["sigma_nd_pb"]
        for row in rows:
            check_close(row["normalization"]["sigma_nd_pb"], sigma, "common ND cross section")
            if row["normalization"]["stratum"] != stratum:
                raise ValueError("Snapshot mixes different pThat strata")
    return groups


def snapshot(args):
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("schema") != "shift-streaming-histogram-snapshot-inputs-v1":
        raise ValueError("Unsupported streaming snapshot inventory")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="stream_snapshot_", dir=os.environ.get("TMPDIR")) as temporary:
        scratch = Path(temporary)
        receipts = []
        for index, row in enumerate(inventory["cohort_receipts"]):
            path = scratch / f"receipt{index}.json"
            stage(row["path"], path, row["sha256"])
            receipts.append(json.loads(path.read_text()))
        groups = snapshot_authorities(receipts, inventory["expected_cohort_ids"])
        samples = []
        for stratum, rows in sorted(groups.items()):
            metadata = dict(schema="shift-streaming-normalization-v1", closed_cohorts=True,
                            stratum=stratum, cohort_ids=[row["cohort_id"] for row in rows],
                            source_ids=[row["normalization"]["source_id"] for row in rows],
                            full_source_trials=sum(row["normalization"]["full_source_trials"] for row in rows),
                            sigma_nd_pb=rows[0]["normalization"]["sigma_nd_pb"],
                            no_second_filter_efficiency=True, partial_production=True)
            sample = dict(stratum=stratum)
            for kind, key in (("histogram", "path"), ("raw_histogram", "raw_path")):
                inputs = []
                for index, row in enumerate(rows):
                    path = scratch / f"{stratum}_{kind}_{index}.root"
                    stage(row[kind]["path"], path, row[kind]["sha256"])
                    inputs.append(path)
                path = output / f"{stratum}_{kind}.root"
                merge(inputs, path, args.merge_script, scratch)
                summary = histogram_summary(path)
                for name in REQUIRED_HISTOGRAMS:
                    for field in ("sumw", "sumw2"):
                        check_close(summary[name][field],
                                    sum(row[kind]["histograms"][name][field] for row in rows),
                                    name + " merged " + field)
                embed_metadata(path, dict(metadata, histogram_weight=(WEIGHT_POLICY if key == "path"
                                                                      else "one_per_reconstructed_entry")))
                sample[key] = str(path)
                sample[key + "_sha256"] = sha256(path)
            samples.append(sample)
        plot_inputs = output / "plot_inputs.json"
        plot_record = dict(schema="shift-streaming-plot-inputs-v1", samples=samples,
                           snapshot_id=inventory["snapshot_id"], partial_production=True)
        reference_inputs = getattr(args, "reference_inputs", None)
        if reference_inputs:
            references = json.loads(reference_inputs.read_text())
            if references.get("schema") != "shift-sm-reference-histograms-v1":
                raise ValueError("Unsupported SM reference histogram inventory")
            plot_record["reference_samples"] = references["samples"]
        write_json(plot_inputs, plot_record)
        plots = []
        if args.plotter:
            if not args.plot_output:
                raise ValueError("--plot-output is required with --plotter")
            plots = render_plots(plot_inputs, args.plotter, args.plot_output, scratch, ("rate", "raw"))
        budget = publication_budget([row[key] for row in samples for key in ("path", "raw_path")], plots,
                                    inventory.get("maximum_publication_bytes", DEFAULT_PUBLICATION_BYTES))
        if args.publish_directory:
            for row in samples:
                for key in ("path", "raw_path"):
                    row[key] = publish(Path(row[key]), args.publish_directory)
            plot_inputs.write_text(json.dumps(plot_record, indent=2, allow_nan=False) + "\n")
            published_plot_inputs = publish(plot_inputs, args.publish_directory)
        else:
            published_plot_inputs = str(plot_inputs)
        receipt = dict(schema="shift-streaming-histogram-snapshot-v1", complete=True,
                       inventory_sha256=sha256(args.inventory), snapshot_id=inventory["snapshot_id"],
                       closed_cohort_ids=inventory["expected_cohort_ids"], samples=samples,
                       plot_inputs_record=plot_record,
                       plot_inputs=file_record(plot_inputs, published_plot_inputs), plots=plots, publication_budget=budget,
                       partial_production=True)
        write_json(output / "snapshot_complete.json", receipt)
        if args.publish_directory:
            publish(output / "snapshot_complete.json", args.publish_directory)
    print(json.dumps(dict(complete=True, completion=str(output / "snapshot_complete.json"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    first = stages.add_parser("cohort")
    first.add_argument("--descriptor", type=Path, required=True)
    first.add_argument("--histogrammer", type=Path, required=True)
    first.add_argument("--retire-consumed-nano", action="store_true")
    first.add_argument("--nano-integrity-runtime", type=Path,
                       help="Stock CMSSW CVMFS release providing complete Nano validation dependencies")
    last = stages.add_parser("snapshot")
    last.add_argument("--inventory", type=Path, required=True)
    last.add_argument("--reference-inputs", type=Path)
    check = stages.add_parser("diagnostic")
    check.add_argument("--descriptor", type=Path, required=True)
    check.add_argument("--histogrammer", type=Path, required=True)
    for stage_parser in (first, last, check):
        stage_parser.add_argument("--output-directory", type=Path, required=True)
        stage_parser.add_argument("--merge-script", type=Path, required=True)
        stage_parser.add_argument("--publish-directory")
        stage_parser.add_argument("--plotter", type=Path)
        stage_parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()
    {"cohort": cohort, "snapshot": snapshot, "diagnostic": diagnostic}[args.stage](args)


if __name__ == "__main__":
    main()
