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


def prepare_mass_auxiliary_card(source, destination):
    """Use the existing total-yield coordinate for any number of mass bins.

    The original Poisson bins and the two normalization constraints are
    unchanged. Only the shared background normalization is expressed through
    total_shift, r and signal_norm. This avoids nearly flat numerical steps
    in a normalization variable when the expected counts are very large.
    """
    import ROOT
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = destination.with_suffix('.root')
    receipt_path = destination.with_suffix('.json')
    signature = dict(source_sha256=digest(source), helper_sha256=digest(__file__))
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if (old['signature']!=signature or old['card_sha256']!=digest(destination)
                or old['workspace_sha256']!=digest(workspace_path)):
            raise ValueError('Saved auxiliary mass card changed')
        return old
    lines = source.read_text().splitlines()
    rates = list(map(float,next(line.split()[1:] for line in lines if line.startswith('rate '))))
    processes = [line.split()[1:] for line in lines if line.startswith('process ')]
    names, numbers = processes[0], list(map(int,processes[1]))
    channel_rows = [line.split()[1:] for line in lines if line.startswith('bin ')]
    observations, channels = channel_rows[0], channel_rows[1]
    if not (len(rates)==len(names)==len(numbers)==len(channels)):
        raise ValueError('Auxiliary card columns differ')
    signal = math.fsum(rate for rate,number in zip(rates,numbers) if number<=0)
    background = math.fsum(rate for rate,number in zip(rates,numbers) if number>0)
    if signal<=0 or background<=0:
        raise ValueError('Auxiliary coordinates need positive total signal and background')
    formula = (f'TMath::Max(1e-30,1+@0/{math.sqrt(background):.17g}'
               f'-@1*{signal/background:.17g}*exp({math.log(1.3):.17g}*@2))')
    workspace = ROOT.RooWorkspace('w')
    workspace.factory(f"expr::theta_background('log({formula})/{math.log(1.3):.17g}',"
                      'total_shift[0,-20,20],r[1,0,20],signal_norm[0,-7,7])')
    workspace.factory('Gaussian::background_aux(theta_b_aux[-12,12],theta_background,1)')
    workspace.var('theta_b_aux').setBins(240)
    workspace.var('theta_b_aux').setVal(0)
    data = ROOT.RooDataSet('data_obs','data_obs',ROOT.RooArgSet(workspace.var('theta_b_aux')))
    data.add(ROOT.RooArgSet(workspace.var('theta_b_aux')))
    getattr(workspace,'import')(data)
    workspace.writeToFile(str(workspace_path))
    result, shapes_added = [], False
    for line in lines:
        if line.startswith('imax '):
            line = 'imax '+str(len(observations)+1)
        elif line.startswith('jmax '):
            line = 'jmax *'
        elif line.startswith('kmax '):
            line = 'kmax *'
        elif line.startswith('background_norm '):
            continue
        elif line.startswith('shapes '):
            parts = line.split()
            path = Path(parts[3])
            parts[3] = str(path if path.is_absolute() else source.parent/path)
            result.append(' '.join(parts))
            if not shapes_added:
                result.append(f'shapes background_aux aux {workspace_path} w:background_aux')
                result.append(f'shapes data_obs aux {workspace_path} w:data_obs')
                shapes_added = True
            continue
        elif line.startswith('bin '):
            line += ' aux'
        elif line.startswith(('observation ','rate ')):
            line += ' 1'
        elif line.startswith('process '):
            parts = line.split()
            line += ' '+(str(max(numbers)+1) if parts[1].lstrip('-').isdigit() else 'background_aux')
        elif line.startswith('signal_norm '):
            line += ' -'
        result.append(line)
    if not shapes_added:
        raise ValueError('Expected native shape declarations for Poisson bins')
    result.append('total_shift extArg 0 [-20,20]')
    result.append('total_shift flatParam')
    for channel,name in sorted(set((channel,name) for channel,name,number in zip(channels,names,numbers) if number>0)):
        result.append(f'background_scale rateParam {channel} {name} ({formula}) total_shift,r,signal_norm')
    destination.write_text('\n'.join(result)+'\n')
    receipt = dict(signature=signature,source=str(source),destination=str(destination),
        card_sha256=digest(destination),workspace_sha256=digest(workspace_path),
        signal_yield=signal,background_yield=background,original_poisson_bins=len(observations),
        normalization_factor=1.30,total_shift_range_standard_deviations=[-20,20],
        auxiliary_observable_range=[-12,12],auxiliary_expected_events=1,
        likelihood='Same Poisson bins, signal lognormal and background lognormal; background nuisance uses total-yield coordinates.')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
