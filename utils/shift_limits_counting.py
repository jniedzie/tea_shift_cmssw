# Code style: AI-code. AI-generated equivalent coordinates for large counts.
"""Represent a single counting channel using its total yield as a fit variable.

The Poisson count and the two independent 30% normalization measurements are
unchanged. With z=(expected-B)/sqrt(B), the background multiplier becomes
1+z/sqrt(B)-r*(S/B)*exp(log(1.3)*signal_norm). Its log-normal constraint is
represented by an auxiliary Gaussian measurement, avoiding cancellation along
the nearly flat signal/background direction in the ordinary coordinates.
"""
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_auxiliary_card(source, destination):
    import ROOT
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    workspace_path = destination.with_name(destination.stem + "_auxiliary.root")
    receipt_path = destination.with_name(destination.stem + "_auxiliary.json")
    signature = dict(source_sha256=digest(source), helper_sha256=digest(__file__))
    if receipt_path.exists() and workspace_path.exists() and destination.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt["signature"] == signature and receipt["card_sha256"] == digest(destination)
                and receipt["workspace_sha256"] == digest(workspace_path)):
            return receipt
    lines = source.read_text().splitlines()
    rates = next(line.split()[1:] for line in lines if line.startswith("rate "))
    names = next(line.split()[1:] for line in lines if line.startswith("process "))
    signal = float(rates[0])
    background = math.fsum(map(float, rates[1:]))
    if signal <= 0 or background <= 0:
        raise ValueError("Auxiliary coordinates require positive signal and background")
    kappa = math.log(1.3)
    formula = (f"TMath::Max(1e-30,1+@0/{math.sqrt(background):.17g}"
               f"-@1*{signal/background:.17g}*exp({kappa:.17g}*@2))")
    workspace = ROOT.RooWorkspace("w")
    workspace.factory(f"expr::theta_background('log({formula})/{kappa:.17g}',"
                      "total_shift[0,-20,20],r[1,0,20],signal_norm[0,-7,7])")
    workspace.factory("Gaussian::background_aux(theta_b_aux[-12,12],theta_background,1)")
    workspace.var("theta_b_aux").setBins(240)
    workspace.var("theta_b_aux").setVal(0)
    observable = ROOT.RooArgSet(workspace.var("theta_b_aux"))
    data = ROOT.RooDataSet("data_obs", "data_obs", observable)
    data.add(observable)
    getattr(workspace, "import")(data)
    workspace.writeToFile(str(workspace_path))
    result = []
    for line in lines:
        if line.startswith("imax "):
            line = "imax 2"
        elif line.startswith("jmax "):
            line = "jmax " + str(len(rates))
        elif line.startswith("background_norm "):
            continue
        elif line.startswith("shapes "):
            parts = line.split()
            shape = Path(parts[3])
            parts[3] = str(shape if shape.is_absolute() else source.parent / shape)
            result.append(" ".join(parts))
            result.append(f"shapes background_aux aux {workspace_path} w:background_aux")
            result.append(f"shapes data_obs aux {workspace_path} w:data_obs")
            continue
        elif line.startswith(("bin ", "observation ", "rate ")):
            line += " aux" if line.startswith("bin ") else " 1"
        elif line.startswith("process "):
            line += " " + (str(len(rates)) if line.split()[1] == "0" else "background_aux")
        elif line.startswith("signal_norm "):
            line += " -"
        result.append(line)
    result.append("total_shift flatParam")
    for name in names[1:]:
        result.append(f"background_scale rateParam bin1 {name} ({formula}) total_shift,r,signal_norm")
    destination.write_text("\n".join(result) + "\n")
    receipt = dict(signature=signature, card_sha256=digest(destination),
                   workspace_sha256=digest(workspace_path), source=str(source),
                   destination=str(destination), signal_yield=signal, background_yield=background,
                   normalization_factor=1.30, total_shift_range_standard_deviations=[-20, 20],
                   auxiliary_observable_range=[-12, 12], auxiliary_expected_events=1)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
