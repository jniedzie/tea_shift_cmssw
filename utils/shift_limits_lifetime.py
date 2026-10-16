#!/usr/bin/env python3
# Code style: AI-code. AI-generated lifetime-template and closure diagnostics.
"""Explain a fixed-mass lifetime scan using audited full-parent normalizations.

Run from bin with --stage overview, cache, diagnostics, or limits. The cache
stage reads only first-candidate RECO branches, one file at a time. GEN proper
lengths and positions come from the existing audited JSON, joined by integer
event identity. No classifier, truth matching or new selection is applied.
"""
import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def settings(config):
    report = json.loads(Path(config.yields_path).read_text())
    points = {name: report['signals'][name] for name in config.lifetime_points}
    if any(row['mass_gev'] != config.lifetime_mass_gev for row in points.values()):
        raise ValueError('Lifetime scan must have exactly one mass')
    if any(not row['source'].get('unit_gen_weights') or
           row['full_gen_sumw'] != row['counts']['events'] for row in points.values()):
        raise ValueError('This diagnostic requires the audited unit-weight signal anchors')
    output = Path(config.lifetime_output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return report, points, output


def truth_rows(row):
    reference = row['source']['gen_audit']
    if digest(reference['path']) != reference['sha256']:
        raise ValueError('GEN audit changed')
    audit = json.loads(Path(reference['path']).read_text())
    if not audit['runtime_validated'] or audit['events'] != row['counts']['events']:
        raise ValueError('Incomplete GEN audit')
    result = {}
    for event in audit['event_rows']:
        if len(event['signal_decays']) != 1:
            raise ValueError('Expected one generated dark photon per parent')
        identity = tuple(event['id'])
        if identity in result:
            raise ValueError('Repeated GEN identity')
        result[identity] = dict(event['signal_decays'][0], weight=event['weight'])
    return result


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42,
                         'axes.grid': True, 'grid.alpha': .18})
    return plt


def label(row):
    flight = row.get('mean_lab_flight_m')
    return 'prompt reference' if flight is None else f'mean lab flight {flight:g} m'


def finish(figure, output, name, caption):
    plt = plotting()
    figure.text(.02, .015, caption, fontsize=8, va='bottom')
    figure.tight_layout(rect=(0, .075, 1, .95))
    path = output / (name + '.pdf')
    figure.savefig(path)
    plt.close(figure)
    print('Saved', path, flush=True)


def overview(config):
    report, points, output = settings(config)
    plt = plotting()
    displaced = [(name, row) for name, row in points.items() if row['mean_lab_flight_m'] is not None]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    figure.suptitle(f"SHIFT simulation: {config.lifetime_mass_gev:g} GeV — lifetime changes the physical hypothesis")
    reference = displaced[0][1]
    lengths = np.geomspace(displaced[0][1]['proper_ctau_mm'] * .5,
                           displaced[-1][1]['proper_ctau_mm'] * 2, 200)
    epsilons = reference['epsilon'] * np.sqrt(reference['proper_ctau_mm'] / lengths)
    axes[0].loglog(lengths, epsilons, color='0.4', label=r'fixed mass: $\epsilon\propto(c\tau)^{-1/2}$')
    for name, row in displaced:
        axes[0].plot(row['proper_ctau_mm'], row['epsilon'], 'o', label=label(row))
        values = np.array([event['proper_length_mm'] for event in truth_rows(row).values()])
        axes[1].hist(values / row['proper_ctau_mm'], bins=np.linspace(0, 5, 26),
                     density=False, weights=np.full(len(values), 1 / (len(values) * .2)),
                     histtype='step', label=label(row))
    x = np.linspace(0, 5, 200)
    axes[1].plot(x, np.exp(-x), 'k--', label='unit exponential')
    axes[0].set(xlabel=r'Mean proper decay length $c\tau$ [mm]', ylabel=r'Physical coupling $\epsilon$')
    axes[1].set(xlabel=r'Generated proper decay length / $c\tau$', ylabel='Probability density')
    for axis in axes: axis.legend(fontsize=8)
    finish(figure, output, '01_lifetime_hypotheses',
           'Metre labels denote estimated mean lab flights, not a fixed decay position. Each curve uses every generated event.\n'
           'Coupling, total width and lifetime use the saved production model; no detector response is changed.')

    categories = report['categories']
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    figure.suptitle(f"SHIFT simulation: {config.lifetime_mass_gev:g} GeV — from 0_initial to exclusive categories")
    positions = np.arange(len(points))
    bottom = np.zeros(len(points))
    totals = np.array([row['counts']['events'] for row in points.values()])
    for category in categories:
        values = np.array([row['counts']['categories'][category]['events'] for row in points.values()])
        axes[0].bar(positions, values, bottom=bottom, label=category)
        axes[1].bar(positions, values / totals, bottom=bottom / totals, label=category)
        bottom += values
    missing = np.array([row['counts']['no_dimuon_events'] for row in points.values()])
    if not np.array_equal(bottom + missing, totals):
        raise ValueError('Selected and zero-dimuon counts do not close the full denominator')
    axes[0].bar(positions, missing, bottom=bottom, color='.85', label='zero dimuons')
    for index, selected in enumerate(bottom):
        axes[1].text(index, selected / totals[index] + .008, f'{int(selected)} / {totals[index]}', ha='center', fontsize=8)
    names = ['prompt' if row['mean_lab_flight_m'] is None else f"{row['mean_lab_flight_m']:g} m" for row in points.values()]
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(names)
        axis.set_xlabel('Generated lifetime anchor (mean lab flight label)')
    axes[0].set_ylabel('Events in full generated population')
    axes[1].set_ylabel('Selected category / full generated population')
    axes[1].set_ylim(0, max(bottom / totals) * 1.3)
    axes[0].legend(fontsize=8, ncol=2)
    finish(figure, output, '02_full_denominator_and_categories',
           '0_initial includes zero-dimuon events. The first stored dimuon assigns each selected event to exactly one category.\n'
           'Stored reconstruction includes its OS requirement; no classifier, mass, vertex-quality or additional kinematic selection.')

    figure, axis = plt.subplots(figsize=(8.6, 5.3))
    figure.suptitle(f"SHIFT simulation: {config.lifetime_mass_gev:g} GeV — reconstruction efficiency")
    x = np.array([row['proper_ctau_mm'] for _, row in displaced])
    n = np.array([row['counts']['events'] for _, row in displaced])
    k = np.array([sum(v['events'] for v in row['counts']['categories'].values()) for _, row in displaced])
    p = k / n
    center = (p + .5 / n) / (1 + 1 / n)
    half = np.sqrt(p * (1-p) / n + .25 / n**2) / (1 + 1 / n)
    axis.errorbar(x, p, yerr=[p-(center-half), center+half-p], fmt='o', capsize=3, label='Displaced anchors, 68% Wilson intervals')
    prompt = next(row for row in points.values() if row['mean_lab_flight_m'] is None)
    prompt_p = 1 - prompt['counts']['no_dimuon_events'] / prompt['counts']['events']
    prompt_error = math.sqrt(prompt_p * (1-prompt_p) / prompt['counts']['events'])
    axis.axhline(prompt_p, color='k', linestyle='--', label=f'Prompt reference: {prompt_p:.3f}')
    axis.axhspan(prompt_p-prompt_error, prompt_p+prompt_error, color='k', alpha=.08)
    axis.set(xscale='log', xlabel=r'Mean proper decay length $c\tau$ [mm]', ylabel='Events with a reconstructed dimuon / 0_initial', ylim=(0, .27))
    axis.legend(fontsize=9)
    finish(figure, output, '03_efficiency_vs_lifetime',
           'Points are independent generated samples. Error bars show finite signal-MC statistics, not the assumed 30% experimental uncertainty.\n'
           'Prompt is a separate reference, not a zero-lifetime point on the logarithmic axis.')
    save_json(output / 'overview.json', dict(mass_gev=config.lifetime_mass_gev,
        yields_path=str(Path(config.yields_path).resolve()), yields_sha256=digest(config.yields_path),
        points={name:dict(epsilon=row['epsilon'], proper_ctau_mm=row['proper_ctau_mm'],
            mean_lab_flight_m=row['mean_lab_flight_m'], generated=row['counts']['events'],
            selected=sum(v['events'] for v in row['counts']['categories'].values()),
            zero_dimuons=row['counts']['no_dimuon_events']) for name,row in points.items()}))


def build_cache(config):
    import ROOT
    from shift_limits_yields import event_category
    ROOT.gROOT.SetBatch(True)
    report, points, output = settings(config)
    records = {}
    seen = set()
    for name, row in points.items():
        destination = output / 'data' / (name + '.npz')
        receipt = destination.with_suffix('.json')
        algorithm = inspect.getsource(build_cache) + inspect.getsource(truth_rows) + inspect.getsource(event_category)
        binding = dict(yields_sha256=digest(config.yields_path), cache_algorithm_sha256=hashlib.sha256(algorithm.encode()).hexdigest(),
                       gen_audit=row['source']['gen_audit'], source_file_stats=row['source']['source_file_stats'])
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['binding'] != binding or old['sha256'] != digest(destination):
                raise ValueError('Existing lifetime cache changed; do not overwrite it')
            print('Reused cache', name, flush=True)
            records[name] = old
            continue
        truth = truth_rows(row)
        data = dict(ids=[], category=[], gen_weight=[], proper_length_mm=[], truth_position_mm=[],
                    mass_gev=[], reco_position_cm=[])
        for metadata in row['source']['source_file_stats']:
            path = Path(metadata['path'])
            before = path.stat()
            if (before.st_size, before.st_mtime_ns) != (metadata['bytes'], metadata['mtime_ns']):
                raise ValueError('Nano source differs from audited count receipt')
            file = ROOT.TFile.Open(str(path), 'READ')
            if not file or file.IsZombie(): raise RuntimeError('Cannot open Nano')
            try:
                tree = file.Get('Events')
                branches = ('run','luminosityBlock','event','genWeight','nShiftDimuonVertex',
                            'ShiftDimuonVertex_topologyMin','ShiftDimuonVertex_topologyMax',
                            'ShiftDimuonVertex_mass','ShiftDimuonVertex_vx','ShiftDimuonVertex_vy','ShiftDimuonVertex_vz')
                tree.SetBranchStatus('*', 0)
                for branch in branches:
                    if not tree.GetBranch(branch): raise ValueError('Missing branch ' + branch)
                    tree.SetBranchStatus(branch, 1)
                for event in tree:
                    identity = (int(event.run), int(event.luminosityBlock), int(event.event))
                    if identity not in truth or identity in seen: raise ValueError('Missing or repeated physical parent')
                    seen.add(identity)
                    generated = truth[identity]
                    if float(event.genWeight) != generated['weight']: raise ValueError('GEN weight changed')
                    count = int(event.nShiftDimuonVertex)
                    if count > 1: raise ValueError('This scan requires the audited one-candidate signal populations')
                    category, mass, position = -1, math.nan, [math.nan] * 3
                    if count:
                        category = report['categories'].index(event_category(int(event.ShiftDimuonVertex_topologyMin[0]), int(event.ShiftDimuonVertex_topologyMax[0])))
                        mass = float(event.ShiftDimuonVertex_mass[0])
                        position = [float(getattr(event, 'ShiftDimuonVertex_v'+axis)[0]) for axis in 'xyz']
                    data['ids'].append(identity)
                    data['category'].append(category)
                    data['gen_weight'].append(generated['weight'])
                    data['proper_length_mm'].append(generated['proper_length_mm'])
                    data['truth_position_mm'].append(generated['position_mm'][:3])
                    data['mass_gev'].append(mass)
                    data['reco_position_cm'].append(position)
            finally: file.Close()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Nano changed during read')
        arrays = {key:np.asarray(value, dtype=np.uint64 if key=='ids' else np.int16 if key=='category' else float) for key,value in data.items()}
        if len(arrays['ids']) != row['counts']['events'] or set(map(tuple, arrays['ids'])) != set(truth):
            raise ValueError('Cache lost full GEN denominator')
        for index, category in enumerate(report['categories']):
            if int(np.sum(arrays['category']==index)) != row['counts']['categories'][category]['events']:
                raise ValueError('Cache differs from audited category counts')
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        records[name] = dict(binding=binding, path=str(destination), sha256=digest(destination),
                             events=len(arrays['ids']), selected=int(np.sum(arrays['category']>=0)))
        save_json(receipt, records[name])
        print('Cached', name, 'events', len(arrays['ids']), flush=True)
    save_json(output / 'data' / 'manifest.json', dict(complete=True, records=records))


def load_cache(output, name):
    path = output / 'data' / (name+'.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    if digest(path) != receipt['sha256']: raise ValueError('Cache hash changed')
    with np.load(path, allow_pickle=False) as source:
        return {key:source[key] for key in source.files}


def reweighting_support(config):
    _,points,output=settings(config)
    target=next(row for row in points.values() if row['mean_lab_flight_m']==100.)
    plt=plotting()
    figure,axes=plt.subplots(1,2,figsize=(12,5.2))
    figure.suptitle('Lifetime reweighting: detector response is retained, statistical information changes')
    records={}
    x=np.linspace(0,5,200)
    axes[0].plot(x,np.exp(-x),'k--',label='Target 100 m exponential')
    for name,row in points.items():
        if row['mean_lab_flight_m'] not in (30.,1000.):continue
        lengths=np.array([event['proper_length_mm'] for event in truth_rows(row).values()])
        weights=lifetime_weights({'proper_length_mm':lengths},row['proper_ctau_mm'],target['proper_ctau_mm'])
        axes[0].hist(lengths/target['proper_ctau_mm'],bins=np.linspace(0,5,26),
            weights=weights/len(weights)/.2,histtype='step',
            label=f"{row['mean_lab_flight_m']:g} m source, estimated effective N = {weights.sum()**2/np.square(weights).sum():.0f}")
        ratio=target['proper_ctau_mm']/row['proper_ctau_mm']
        records[name]=dict(generated=len(weights),estimated_effective_events=float(weights.sum()**2/np.square(weights).sum()),
            target_over_source_ctau=ratio,finite_theoretical_weight_variance=ratio<2,
            weighted_total_over_original_gen=float(weights.mean()))
    ratio=np.linspace(.01,1.999,400)
    axes[1].plot(ratio,ratio*(2-ratio),label='Ideal effective N / generated N')
    axes[1].axvspan(2,4,color='tab:red',alpha=.12,label='Infinite second moment of weights')
    axes[1].axvline(1,color='0.4',ls='--')
    axes[1].axvline(.1,color='tab:green',ls=':',label='1,000 m to 100 m: ratio 0.1')
    axes[1].axvline(100/30,color='tab:orange',ls=':',label='30 m to 100 m: ratio 3.33')
    axes[0].set(xlabel=r'Generated proper decay length / target $c\tau$',ylabel='Weighted probability density')
    axes[1].set(xlabel=r'Target $c\tau$ / source $c\tau$',ylabel='Asymptotic effective fraction (unfiltered decays)',xlim=(0,4),ylim=(0,1.08))
    for axis in axes:axis.legend(fontsize=8)
    finish(figure,output,'03_reweighting_support',
        'For exponential decays, E[w^2] = 1/[r(2-r)] only for r < 2. At r >= 2 the weight variance diverges.\n'
        'A finite sample can still report an apparently reasonable effective count while missing rare long-flight events. No reco events are read here.')
    save_json(output/'reweighting_support.json',dict(target_ctau_mm=target['proper_ctau_mm'],records=records,
        derivation='Integral f_target(t)^2/f_source(t) dt; finite only when 2/lambda_target - 1/lambda_source > 0.'))


def lifetime_weights(data, source_ctau, target_ctau):
    # Keep the original full-parent denominator; never renormalize to selected events.
    return (source_ctau / target_ctau) * np.exp(data['proper_length_mm'] * (1/source_ctau-1/target_ctau))


def acceptance(data, weights, category=None):
    selected = data['category'] >= 0 if category is None else data['category'] == category
    values = weights * selected
    n = len(values)
    mean = float(values.mean())
    error = float(np.std(values, ddof=1) / np.sqrt(n))
    ess = float(values.sum()**2 / np.square(values).sum()) if values.sum() else 0.
    return dict(value=mean, error=error, selected_ess=ess)


def diagnostics(config):
    report, points, output = settings(config)
    plt = plotting()
    caches = {name:load_cache(output,name) for name in points}
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    figure.suptitle(f'SHIFT simulation: {config.lifetime_mass_gev:g} GeV — reconstructed signal shapes change with lifetime')
    summaries = {}
    for name,row in points.items():
        data = caches[name]
        selected = data['category'] >= 0
        mass = data['mass_gev'][selected]
        residual = (data['reco_position_cm'][selected,2]*10-data['truth_position_mm'][selected,2])/1000
        axes[0].hist(mass, bins=np.linspace(0,60,31), weights=np.full(len(mass),1/len(mass)), histtype='step',label=label(row))
        axes[1].hist(data['truth_position_mm'][:,2]/1000, bins=np.linspace(-200,180,39),
                     weights=np.full(len(data['category']),1/len(data['category'])),histtype='step', label=label(row))
        axes[2].hist(residual,bins=np.linspace(-100,100,41),weights=np.full(len(mass),1/len(mass)),histtype='step',label=label(row))
        summaries[name] = dict(selected=len(mass),mass_median_gev=float(np.median(mass)),
            mass_q16_q84_gev=np.quantile(mass,[.16,.84]).tolist(),
            delta_z_q16_q50_q84_m=np.quantile(residual,[.16,.5,.84]).tolist(),
            generated_z_outside_plot=int(np.sum((data['truth_position_mm'][:,2]/1000 < -200) | (data['truth_position_mm'][:,2]/1000 > 180))),
            mass_outside_plot=int(np.sum((mass<0)|(mass>60))),
            delta_z_outside_plot=int(np.sum(np.abs(residual)>100)))
    axes[0].axvline(config.lifetime_mass_gev,color='k',ls='--',lw=1)
    axes[0].set(xlabel=r'Reconstructed dimuon mass [GeV]',ylabel='Fraction of selected events / bin')
    axes[1].set(xlabel='Generated decay z [m]',ylabel='Fraction of all generated events / bin')
    axes[2].set(xlabel=r'$z_{reco}-z_{generated}$ [m]',ylabel='Fraction of selected events / bin')
    axes[0].legend(fontsize=7)
    finish(figure,output,'04_reconstructed_shapes',
           'Inclusive first stored candidate, including wrong pairs: residuals are response diagnostics, not a truth-matched resolution measurement.\n'
           'Visible ranges can omit tails; overflow counts and central quantiles are retained in shapes.json. No histogram is used to select events.')
    save_json(output/'shapes.json',summaries)

    target_name = next(name for name,row in points.items() if row['mean_lab_flight_m']==100.)
    target, target_data = points[target_name], caches[target_name]
    sources = [(name,row) for name,row in points.items() if row['mean_lab_flight_m'] in (30.,1000.)]
    figure,axes = plt.subplots(1,3,figsize=(15.5,5.1))
    figure.suptitle('Lifetime reweighting closure: predict the independent 100 m anchor')
    rows = {}
    comparisons=[(target_name,target,target_data,np.ones(len(target_data['category'])))]
    for name,row in sources:
        data=caches[name]
        weights=lifetime_weights(data,row['proper_ctau_mm'],target['proper_ctau_mm'])
        comparisons.append((name,row,data,weights))
    target_accept=[acceptance(target_data,np.ones(len(target_data['category'])),i) for i in range(len(report['categories']))]
    for offset,(name,row,data,weights) in enumerate(comparisons):
        title='direct 100 m simulation' if name==target_name else label(row)+' reweighted'
        axes[0].hist(data['proper_length_mm']/target['proper_ctau_mm'],bins=np.linspace(0,5,26),
                     weights=weights/len(weights)/.2,histtype='step',label=title)
        estimates=[acceptance(data,weights,i) for i in range(len(report['categories']))]
        axes[1].errorbar(np.arange(len(estimates))+(offset-1)*.18,[v['value'] for v in estimates],
                         yerr=[v['error'] for v in estimates],fmt='o',capsize=3,label=title)
        selected=data['category']>=0
        axes[2].hist(data['mass_gev'][selected],bins=np.linspace(0,60,21),
                     weights=weights[selected]/len(weights),histtype='step',label=title)
        pulls=[(a['value']-b['value'])/math.hypot(a['error'],b['error']) if math.hypot(a['error'],b['error']) else None
               for a,b in zip(estimates,target_accept)]
        rows[name]=dict(target=target_name,all_parent_weight_mean=float(weights.mean()),
            all_parent_weight_mean_error=float(weights.std(ddof=1)/np.sqrt(len(weights))),
            total_acceptance=acceptance(data,weights),category_acceptances=estimates,
            category_pulls=pulls,independent=name!=target_name)
    axes[0].set(xlabel=r'Generated proper decay length / target $c\tau$',ylabel='Probability density')
    axes[1].set_xticks(np.arange(len(report['categories'])))
    axes[1].set_xticklabels(report['categories'],rotation=25,ha='right',fontsize=8)
    axes[1].set_ylabel('Accepted category / original full GEN denominator')
    axes[2].set(xlabel='Reconstructed mass [GeV]',ylabel='Selected probability / original GEN event / bin')
    axes[0].legend(fontsize=7)
    finish(figure,output,'05_lifetime_reweighting_closure',
           'An event keeps its simulated detector response; only its decay-probability weight changes. The original 0_initial denominator stays fixed.\n'
           '30 m to 100 m illustrates the risky long-tail extrapolation; 1,000 m to 100 m has better tail coverage. Error bars include weight fluctuations.')
    longer = next(record for name,record in rows.items() if points[name]['mean_lab_flight_m']==1000.)
    weight_pull = (longer['all_parent_weight_mean']-1)/longer['all_parent_weight_mean_error']
    closure_passed = abs(weight_pull)<3 and all(abs(value)<3 for value in longer['category_pulls'] if value is not None)
    save_json(output/'closure.json',dict(target=target_name,records=rows,
        shrinking_lifetime_closure_passed=closure_passed,weight_normalization_pull=weight_pull,
        statement='Finite-MC closure diagnostic, not a detector-data calibration or proof of validity outside sampled support.'))


def prepare_scan_yields(config):
    report, points, output = settings(config)
    closure = json.loads((output/'closure.json').read_text())
    if not closure['shrinking_lifetime_closure_passed']:
        raise ValueError('Independent lifetime closure failed; do not fit interpolated templates')
    selected = copy.deepcopy(report)
    selected['signals'] = copy.deepcopy(points)
    anchors = [(name,row) for name,row in points.items() if row['mean_lab_flight_m'] is not None]
    interpolation = []
    for (lower_name,lower),(upper_name,upper) in zip(anchors[:-1],anchors[1:]):
        target_ctau = math.sqrt(lower['proper_ctau_mm'] * upper['proper_ctau_mm'])
        data = load_cache(output, upper_name)
        weights = lifetime_weights(data,upper['proper_ctau_mm'],target_ctau)
        epsilon = upper['epsilon'] * math.sqrt(upper['proper_ctau_mm']/target_ctau)
        sigma = upper['reference_cross_section_pb'] * (epsilon/upper['epsilon'])**2
        name = f'm{config.lifetime_mass_gev:g}_reweighted_ctau_' + format(target_ctau,'.8g').replace('.','p')
        row = copy.deepcopy(upper)
        row.update(epsilon=epsilon,proper_ctau_mm=target_ctau,mean_lab_flight_m=math.sqrt(lower['mean_lab_flight_m']*upper['mean_lab_flight_m']),
                   reference_cross_section_pb=sigma,cross_section_pb=sigma,sample_kind='lifetime_reweighted')
        row['yields'] = {}
        row['mc_errors'] = {}
        estimates = {}
        for index,category in enumerate(report['categories']):
            estimate = acceptance(data,weights,index)
            row['yields'][category] = report['luminosity_pb']*sigma*estimate['value']
            row['mc_errors'][category] = report['luminosity_pb']*sigma*estimate['error']
            mask = data['category']==index
            row['counts']['categories'][category].update(sumw=float(weights[mask].sum()),
                sumw2=float(np.square(weights[mask]).sum()),ess=estimate['selected_ess'])
            estimates[category]=estimate
        total = acceptance(data,weights)
        if total['selected_ess'] < 10:
            raise ValueError('Insufficient selected effective statistics for interpolation')
        row['lifetime_reweighting'] = dict(source_point=upper_name,source_ctau_mm=upper['proper_ctau_mm'],
            target_ctau_mm=target_ctau,full_gen_denominator=upper['full_gen_sumw'],
            target_weight_mean=float(weights.mean()),target_weight_mean_error=float(weights.std(ddof=1)/np.sqrt(len(weights))),
            category_estimates=estimates,total_estimate=total,
            convention='Exponential density ratio; original full GEN denominator fixed; existing reco response retained.')
        selected['signals'][name]=row
        interpolation.append(dict(point=name,source=upper_name,proper_ctau_mm=target_ctau,
                                  epsilon=epsilon,selected_ess=total['selected_ess']))
    selected['lifetime_scan'] = dict(source_yields=str(Path(config.yields_path).resolve()),
        source_yields_sha256=digest(config.yields_path),mass_gev=config.lifetime_mass_gev,
        direct_points=list(points),interpolated_points=interpolation,
        mc_statistics='Shown in efficiency and closure diagnostics; not included in this first counting likelihood.',
        interpretation='Conditional sensitivity under the original simplified background and systematic assumptions.')
    save_json(output/'scan_yields.json',selected)
    plt=plotting()
    figure,axes=plt.subplots(1,2,figsize=(11.8,5.2))
    figure.suptitle(f'SHIFT simulation: {config.lifetime_mass_gev:g} GeV — separate detector acceptance from production rate')
    for kind,marker in [('direct','o'),('reweighted','D')]:
        rows=[row for row in selected['signals'].values() if row['mean_lab_flight_m'] is not None
              and ('lifetime_reweighting' in row)==(kind=='reweighted')]
        x=[row['proper_ctau_mm'] for row in rows]
        axes[0].plot(x,[sum(row['yields'].values())/row['reference_cross_section_pb'] for row in rows],marker, label=kind)
        axes[1].plot(x,[sum(row['yields'].values()) for row in rows],marker,label=kind)
    axes[0].set(xscale='log',xlabel=r'$c\tau$ [mm]',ylabel=r'Selected events / ($\sigma\times\mathcal{B}=1$ pb)')
    axes[1].set(xscale='log',yscale='log',xlabel=r'$c\tau$ [mm]',ylabel=r'Selected events at the physical $\epsilon(c\tau)$')
    for axis in axes:axis.legend()
    finish(figure,output,'06_acceptance_and_physical_rate',
           'Effective luminosity 3 fb^-1. Trigger efficiency and scale factors equal one. Metre labels are converted using the saved total-width model.\n'
           'Diamonds are intermediate lifetimes reweighted from the next longer-lived generated anchor; they share its MC events.')
    print('Prepared lifetime points:',len(selected['signals']),flush=True)
    return selected


def draw_limits(config):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import limits_plotter
    _,points,output=settings(config)
    data=json.loads((output/'results'/'limits.json').read_text())
    records=data['records']
    levels=sorted({row['confidence_level'] for row in records})
    for level in levels:
        selected=sorted([row for row in records if row['confidence_level']==level and row['mean_lab_flight_m'] is not None
                         and row.get('numerically_validated',False)],key=lambda row:row['proper_ctau_mm'])
        if len(selected)<2:continue
        directory=output/'brazil'/('cl'+str(round(level*100)))
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'limits_lifetime.txt'
        with path.open('w') as stream:
            for row in selected:
                quantiles=row['expected_cross_section_quantiles_pb']
                values=[None]+[quantiles[k] for k in ('minus2','minus1','median','plus1','plus2')]
                stream.write(row['point']+': '+repr(values)+'\n')
        current=SimpleNamespace(**{key:value for key,value in vars(config).items() if not key.startswith('__')})
        current.limit_points={row['point']:row['proper_ctau_mm'] for row in selected}
        current.reference_cross_section=1.
        current.x_title='c#tau [mm]'
        current.y_title='Expected limit on #sigma #times B_{#mu#mu} [pb]'
        current.x_min=selected[0]['proper_ctau_mm']*.8
        current.x_max=selected[-1]['proper_ctau_mm']*1.2
        values=[value for row in selected for value in row['expected_cross_section_quantiles_pb'].values()]
        current.y_min=min(values)*.5
        current.y_max=max(values)*3.
        current.plots_output_path=str(directory)
        current.experiment_label=f'#bf{{SHIFT}} #it{{Simulation}}, m={config.lifetime_mass_gev:g} GeV'
        current.luminosity_label=f'3 fb^{{-1}}, {level*100:g}% CL'
        if not all(row.get('expected_bands_validated',False) for row in selected):
            current.luminosity_label='#splitline{'+current.luminosity_label+'}{#scale[0.65]{Band edges provisional}}'
        limits_plotter.config=current
        limits_plotter.input_path=str(path)
        limits_plotter.draw_brazil_plots()
    plt=plotting()
    figure,axes=plt.subplots(1,2,figsize=(12,5.5))
    figure.suptitle(f'SHIFT preliminary expectation: {config.lifetime_mass_gev:g} GeV — test the physical coupling')
    for level in levels:
        selected=sorted([row for row in records if row['confidence_level']==level and row['mean_lab_flight_m'] is not None
                         and row.get('numerically_validated',False)],key=lambda row:row['epsilon'])
        if not selected:continue
        epsilon=[row['epsilon'] for row in selected]
        axes[0].loglog(epsilon,[row['expected_limit_cross_section_pb'] for row in selected],'o--',label=f'{level*100:g}% CL cross-section limit')
        axes[1].loglog(epsilon,[row['expected_limit'] for row in selected],'o--',label=f'{level*100:g}% CL limit / model prediction')
    selected=sorted([row for row in records if row['confidence_level']==max(levels) and row['mean_lab_flight_m'] is not None],key=lambda row:row['epsilon'])
    axes[0].loglog([row['epsilon'] for row in selected],[row['reference_cross_section_pb'] for row in selected],'k-',label=r'Physical $\sigma\times\mathcal{B}_{\mu\mu}$')
    axes[1].axhline(1,color='k',ls=':',label='Exclusion boundary = 1')
    axes[0].set(xlabel=r'Physical coupling $\epsilon$',ylabel=r'$\sigma\times\mathcal{B}_{\mu\mu}$ [pb]')
    axes[1].set(xlabel=r'Physical coupling $\epsilon$',ylabel='Expected cross-section limit / physical prediction')
    for axis in axes:axis.legend(fontsize=8)
    finish(figure,output,'07_physical_coupling_test',
           'Every point uses its own lifetime-dependent category template. The plotted ratio rescales the conditioned Combine strength back to the physical model.\n'
           'Only a crossing of one establishes an exclusion boundary; lines guide the eye. MC statistics and missing background tails remain limitations.')


def write_receipt(config):
    _,_,output=settings(config)
    files=[path for path in output.rglob('*') if path.is_file() and path.suffix in ('.pdf','.json','.npz','.txt')
           and path.name!='receipt.json']
    save_json(output/'receipt.json',dict(source_sha256=digest(__file__),
        files_sha256={str(path.relative_to(output)):digest(path) for path in files},
        first_counting_scan=True,full_denominator_preserved=True))


def category_relation(config):
    report,points,output=settings(config)
    plt=plotting()
    categories=report['categories']
    background=np.array([report['background_totals'][name] for name in categories])
    background_fraction=background/background.sum()
    figure,axes=plt.subplots(1,2,figsize=(12.3,5.4))
    figure.suptitle(f'SHIFT simulation: {config.lifetime_mass_gev:g} GeV — category fractions change the signal/background relation')
    x=np.arange(len(categories))
    axes[0].bar(x,background_fraction,color='.85',label='Background: fixed category fractions')
    axes[1].axhline(1,color='k',ls='--',label='Inclusive signal/background ratio')
    for index,(name,row) in enumerate(points.items()):
        counts=np.array([row['counts']['categories'][category]['events'] for category in categories])
        n=counts.sum()
        fraction=counts/n
        center=(fraction+.5/n)/(1+1/n)
        half=np.sqrt(fraction*(1-fraction)/n+.25/n**2)/(1+1/n)
        error=np.array([fraction-(center-half),center+half-fraction])
        location=x+(index-2)*.12
        axes[0].errorbar(location,fraction,yerr=error,fmt='o',capsize=2,label=label(row))
        axes[1].errorbar(location,fraction/background_fraction,yerr=error/background_fraction,fmt='o',capsize=2,label=label(row))
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(categories,rotation=20,ha='right')
    axes[0].set_ylabel('Category fraction within the selected sample')
    axes[1].set_ylabel('(Signal/background in category) / inclusive ratio')
    axes[0].legend(fontsize=7,loc='upper left')
    axes[1].legend(fontsize=8,loc='upper right')
    finish(figure,output,'02_signal_background_relation',
        'Error bars show finite signal-MC statistics, including upper intervals for empty MC categories. Production rate cancels in the right panel.\n'
        'The counting likelihood fixes background category fractions: a shared 30% normalization uncertainty does not let those fractions vary.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='shift_limits_config')
    parser.add_argument('--stage',choices=('overview','support','categories','cache','diagnostics','prepare','limits'),required=True)
    parser.add_argument('--point',help='One cache canary, before the complete one-mass cache')
    args=parser.parse_args()
    config=importlib.import_module(args.config.removesuffix('.py'))
    if args.point:
        if args.stage!='cache' or args.point not in config.lifetime_points:raise ValueError('Unsupported cache canary')
        config.lifetime_points=(args.point,)
    {'overview':overview,'support':reweighting_support,'categories':category_relation,'cache':build_cache,'diagnostics':diagnostics,
     'prepare':prepare_scan_yields,'limits':draw_limits}[args.stage](config)
    write_receipt(config)


if __name__=='__main__': main()
