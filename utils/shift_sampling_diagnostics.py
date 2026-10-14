# Code style: AI-code. AI-generated diagnostics for independent weighted source events.
"""Compare raw and effective GEN tail counts without treating decay copies as events."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def effective_count(weights):
    """Scale first so very small likelihood weights do not underflow when squared."""
    if not weights:
        return 0.0
    largest = max(weights)
    if largest == 0:
        return 0.0
    scaled = [weight / largest for weight in weights]
    return math.fsum(scaled) ** 2 / math.fsum(weight * weight for weight in scaled)


def source_events(rows):
    events = []
    identities = set()
    for row in rows:
        identity = tuple(row["event_id"])
        if identity in identities:
            raise ValueError(f"Repeated source event {identity}; combine its replicas first")
        identities.add(identity)
        weight = float(row["final_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid source weight for {identity}")
        pairs = []
        for pair in row["actual_OS_muon_pairs"]:
            if pair["strict_two_station_each_endcap"]:
                pairs.append(pair)
        if not pairs:
            continue
        events.append({
            "identity": identity,
            "weight": weight,
            "mass": max(pair["mass_GeV"] for pair in pairs),
            "pt": max(pair["maximum_muon_pt_GeV"] for pair in pairs),
        })
    return events


def tail_counts(events, variable, threshold):
    selected = [event for event in events if event[variable] > threshold]
    weights = [event["weight"] for event in selected]
    return len(selected), effective_count(weights)


def validate_summary(rows, events, audit):
    if not audit.get("passed"):
        raise ValueError("The source audit has not passed")
    if audit["full_trials"] <= 0:
        raise ValueError("The full attempted denominator must be positive")
    if len(rows) != audit["accepted"]:
        raise ValueError("Support and audit have different accepted counts")
    if len(rows) > audit["full_trials"]:
        raise ValueError("Accepted count exceeds the full attempted denominator")
    for variable, threshold, key in (
        ("mass", 20, "strict_line_OS_mass_above20"),
        ("mass", 40, "strict_line_OS_mass_above40"),
        ("pt", 20, "strict_line_OS_muon_pt_above20"),
    ):
        count, effective = tail_counts(events, variable, threshold)
        reference = audit["summary"][key]
        if count != reference["independent_source_events"]:
            raise ValueError(f"Independent event count does not reproduce {key}")
        if not math.isclose(effective, reference["effective_source_events"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Effective event count does not reproduce {key}")


def read_sample(label, support_path, audit_path):
    support = Path(support_path)
    audit_file = Path(audit_path)
    opener = gzip.open if support.suffix == ".gz" else open
    with opener(support, "rt") as handle:
        rows = json.load(handle)
    audit = json.loads(audit_file.read_text())
    events = source_events(rows)
    validate_summary(rows, events, audit)
    return {
        "label": label,
        "events": events,
        "full_trials": audit["full_trials"],
        "inputs": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (support, audit_file)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append", nargs=3, required=True,
                        metavar=("LABEL", "SUPPORT_JSON", "AUDIT_JSON"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-trials", type=int,
                        help="Compare observed statistical yield per this many attempted trials")
    args = parser.parse_args()
    if args.per_trials is not None and args.per_trials <= 0:
        parser.error("--per-trials must be positive")
    samples = [read_sample(*specification) for specification in args.sample]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7.3), sharex="col")
    for column, (variable, limit, xlabel) in enumerate((
        ("mass", 60, "GEN opposite-sign pair mass threshold [GeV]"),
        ("pt", 40, "GEN largest muon pT in the pair: threshold [GeV]"),
    )):
        thresholds = list(range(limit + 1))
        for sample in samples:
            counts = [tail_counts(sample["events"], variable, threshold) for threshold in thresholds]
            label = f"{sample['label']} ({sample['full_trials']:,} attempts)"
            scale = args.per_trials / sample["full_trials"] if args.per_trials else 1.0
            axes[0, column].step(thresholds, [count[0] * scale for count in counts], where="post", label=label)
            axes[1, column].step(thresholds, [count[1] * scale for count in counts], where="post", label=label)
        for row in range(2):
            axes[row, column].set_xlim(0, limit)
            axes[row, column].set_ylim(bottom=0)
            axes[row, column].axvline(20, color="0.5", linestyle=":", linewidth=1)
            axes[row, column].grid(alpha=0.2)
        axes[1, column].set_xlabel(xlabel)
    suffix = f"\nper {args.per_trials:,} attempts" if args.per_trials else ""
    axes[0, 0].set_ylabel("Raw independent event count" + suffix)
    axes[1, 0].set_ylabel("Effective count after weights" + suffix)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("QCD sampling: raw event counts and usable statistics", fontsize=14)
    figure.text(0.5, 0.018,
                "GEN only. Each muon has straight lines through at least two CSC stations in each endcap.\n"
                "This is an input diagnostic; reconstruction, physical rates and BDT readiness are unvalidated.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.065, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)

    receipt = {
        "schema": "shift-independent-source-tail-statistics-plot-v1",
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "effective_count_definition": "(sum source weights)^2 / sum squared source weights",
        "counts_displayed_per_attempted_trials": args.per_trials,
        "counts_are_independent_original_source_events": True,
        "reconstructed_acceptance_or_physical_rate_claim": False,
        "samples": [],
    }
    for sample in samples:
        receipt["samples"].append({
            "label": sample["label"], "full_trials": sample["full_trials"], "inputs": sample["inputs"],
            "mass_above20": tail_counts(sample["events"], "mass", 20),
            "mass_above40": tail_counts(sample["events"], "mass", 40),
            "muon_pt_above20": tail_counts(sample["events"], "pt", 20),
        })
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
