#!/usr/bin/env python3
# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Compare frozen BDT cuts and audit conditional acceptance without fitting.

The extraction is aligned to an existing, immutable performance population.
Truth and weights are stored separately from reconstructed model predictors.
Existing test populations are diagnostic evidence, never fresh tuning data.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np


AXES = ('mass', 'z', 'radius')
AXIS_KEYS = ('reconstructed_mass_gev', 'reconstructed_vertex_z_cm', 'reconstructed_vertex_radius_cm')
AXIS_LABELS = ('Reconstructed mass [GeV]', 'Reconstructed vertex z [cm]', 'Reconstructed vertex radius [cm]')
LABELS = {'genuine': 1, 'accidental': 0, 'unknown': -1}
TOPOLOGIES = ('Near-Both', 'Near-Far', 'Both-Both', 'Both-Far', 'Other')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def topology_names(low, high):
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != high.shape:
        raise ValueError('Topology arrays differ')
    result = np.full(low.shape, 'Other', dtype='U10')
    for pair, name in (((0, 2), 'Near-Both'), ((0, 3), 'Near-Far'),
                       ((2, 2), 'Both-Both'), ((2, 3), 'Both-Far')):
        result[(low == pair[0]) & (high == pair[1])] = name
    return result


def assert_alignment(actual, expected, keys):
    for key in keys:
        if actual[key].shape != expected[key].shape or not np.array_equal(actual[key], expected[key]):
            raise ValueError('Frozen candidate population differs: ' + key)


def extract(args):
    import awkward as ak
    import uproot
    sys.path.insert(0, str(args.classifier_directory.resolve()))
    from features import PREDICTOR_BRANCHES, FEATURE_NAMES, extract_reco
    from labels import TRUTH_BRANCHES, extract_labels, label_diagnostics
    from portable_inference import load_model, predict
    from bdt_performance import validate_manifest

    manifest = read(args.manifest)
    validate_manifest(manifest)
    if manifest.get('inputs_complete') is not True:
        raise ValueError('Extraction requires completed source inventory')
    expected = load_npz(args.evaluation_rows)
    model = load_model(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_ref, rows_ref, model_ref = map(reference, (args.manifest, args.evaluation_rows, args.model))
    reco_pieces, label_pieces, event_pieces = defaultdict(list), defaultdict(list), defaultdict(list)
    source_records, offset, started = [], 0, time.monotonic()
    selected = manifest['inputs'] if args.max_files is None else manifest['inputs'][:args.max_files]
    for source_index, source in enumerate(selected):
        path = Path(source['path'])
        before = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        receipt = read(source['score_receipt'])
        if (receipt.get('complete') is not True or receipt.get('model_sha256') != model_ref['sha256'] or
                receipt.get('output_sha256') != source['nano_sha256'] or Path(receipt['output']).resolve() != path.resolve()):
            raise ValueError('Source scoring receipt mismatch')
        metadata, n_events, n_pairs = None, 0, 0
        weight_branch = source.get('weight_branch', 'genWeight' if source['process'] == 'signal' else 'shiftSamplingGenWeight')
        extras = ('ShiftDimuonVertex_bdtScore', 'ShiftDimuonVertex_mass',
                  'ShiftDimuonVertex_topologyMin', 'ShiftDimuonVertex_topologyMax',
                  'ShiftDimuonVertex_isOS', 'ShiftDimuonVertex_dcaValid', weight_branch)
        wanted = list(dict.fromkeys(PREDICTOR_BRANCHES + TRUTH_BRANCHES + extras))
        with uproot.open(path, handler=uproot.source.file.MultithreadedFileSource, num_workers=1) as root:
            tree = root['Events']
            optional = [k for k in ('ShiftMuon_hitTruthLayers','ShiftMuon_hitTruthAmbiguousLayers',
                'ShiftMuon_genPartIdx','ShiftMuon_genPartDeltaR') if k in tree]
            metadata = json.loads(str(root['ShiftDimuonVertex_bdtScoreMetadata']))
            if metadata['model_sha256'] != model_ref['sha256'] or metadata['selection_applied'] is not False:
                raise ValueError('Source score metadata mismatch')
            for arrays in tree.iterate(wanted + optional, step_size=1000, library='ak', how=dict):
                reco = extract_reco({key: arrays[key] for key in PREDICTOR_BRANCHES}, include_nuisance=False)
                score = ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_bdtScore']))
                if not np.array_equal(predict(model, reco['X']), score):
                    raise ValueError('Stored score differs from frozen reco-only predictor')
                y, reason = extract_labels(arrays, reco)
                detail = label_diagnostics(arrays, reco)
                for branch in ('ShiftMuon_hitTruthLayers','ShiftMuon_hitTruthAmbiguousLayers','ShiftMuon_genPartIdx','ShiftMuon_genPartDeltaR'):
                    detail[branch.removeprefix('ShiftMuon_')] = np.asarray([
                        [float(arrays[branch][e][i]) for i in muons] for e,muons in zip(reco['event_index'],reco['muon_indices'])
                    ], dtype=float).reshape(-1,2) if branch in arrays else np.full((len(y),2),np.nan)
                truth_positions = np.full((len(y),2,3),np.nan)
                for row,(event,indices) in enumerate(zip(reco['event_index'],detail['gen_index'])):
                    for leg,g in enumerate(indices):
                        if (0 <= g < len(arrays['GenPart_pdgId'][event]) and
                                abs(int(arrays['GenPart_pdgId'][event][g])) == 13 and
                                int(arrays['GenPart_status'][event][g]) == 1):
                            truth_positions[row,leg] = [float(arrays['GenPart_'+axis][event][g]) for axis in ('vx','vy','vz')]
                detail['truth_position_cm'] = truth_positions
                event_ids = np.column_stack([ak.to_numpy(arrays[key]).astype(np.uint64) for key in ('run', 'luminosityBlock', 'event')])
                event_pieces['ids'].append(event_ids)
                event_pieces['source_index'].append(np.full(len(event_ids), source_index, dtype=np.int32))
                event_pieces['process'].append(np.full(len(event_ids), source['process'], dtype='U6'))
                event_pieces['point'].append(np.full(len(event_ids), source.get('point', ''), dtype='U20'))
                event_pieces['native_weight'].append(ak.to_numpy(arrays[weight_branch]))
                nuisance = np.column_stack([ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_mass'])),
                    ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_vz'])),
                    np.hypot(ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_vx'])), ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_vy'])))])
                count = len(y)
                result = dict(X=reco['X'], ids=reco['ids'], pair_index=reco['pair_index'], muon_indices=reco['muon_indices'],
                    vertex_position_cm=np.column_stack([ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_'+axis])) for axis in ('vx','vy','vz')]),
                    event_index=reco['event_index'] + n_events, source_index=np.full(count, source_index, dtype=np.int32),
                    nuisance=nuisance, process=np.full(count, source['process'], dtype='U6'),
                    point=np.full(count, source.get('point', ''), dtype='U20'),
                    stratum=np.full(count, source.get('stratum', source.get('point', '')), dtype='U100'), score=score,
                    evaluation_weight=ak.to_numpy(arrays[weight_branch])[reco['event_index']] * float(source['mixture_scale']),
                    native_weight=ak.to_numpy(arrays[weight_branch])[reco['event_index']],
                    topology=topology_names(ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_topologyMin'])),
                                            ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_topologyMax']))))
                result['good'] = ((ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_isOS'])) == 1) &
                    (ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_dcaValid'])) == 1) &
                    (ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_dca'])) >= 0) &
                    (ak.to_numpy(ak.flatten(arrays['ShiftDimuonVertex_dca'])) <= 50))
                frozen = {k: v[offset:offset+count] for k, v in expected.items()}
                assert_alignment({**result, 'y': y, 'reason': reason}, frozen,
                    ('ids', 'score', 'y', 'reason', 'nuisance', 'process', 'point', 'evaluation_weight'))
                result['heldout'] = frozen['heldout']
                for key, value in result.items(): reco_pieces[key].append(value)
                for key, value in dict(y=y, reason=reason, **detail).items(): label_pieces[key].append(value)
                offset += count
                n_events += len(event_ids)
                n_pairs += count
        after = path.stat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during read')
        if n_events != receipt['events'] or n_pairs != receipt['retained_pairs']:
            raise ValueError('Source pair/event receipt mismatch')
        source_records.append(dict(index=source_index, path=str(path), process=source['process'],
            point=source.get('point'), stratum=source.get('stratum'), nano_sha256=source['nano_sha256'],
            score_receipt=reference(source['score_receipt']), events=n_events, pairs=n_pairs,
            first_candidate=offset-n_pairs, end_candidate=offset, stat_signature=signature))
        print(json.dumps(dict(source_complete=source_index, events=n_events, pairs=n_pairs, candidates=offset)), flush=True)
    if args.max_files is None and offset != len(expected['y']):
        raise ValueError('Full extraction does not cover frozen evaluation rows')
    outputs = {}
    for filename, pieces in (('reco.npz', reco_pieces), ('labels.npz', label_pieces), ('events.npz', event_pieces)):
        values = {key: np.concatenate(parts) for key, parts in pieces.items()}
        if filename == 'reco.npz': values['feature_names'] = np.asarray(FEATURE_NAMES)
        np.savez_compressed(args.output / filename, **values)
        outputs[filename.replace('.npz', '_sha256')] = digest(args.output / filename)
    write(args.output/'inputs.json', dict(schema='shift-bdt-acceptance-extraction-v1', complete=True,
        sample_kind='simulation_only', frozen_population_parity=True, full_population=args.max_files is None,
        source_manifest=manifest_ref, evaluation_rows=rows_ref, model=model_ref, inputs=source_records,
        candidates=offset, events=sum(s['events'] for s in source_records), elapsed_seconds=time.monotonic()-started,
        source_code={name:reference(args.classifier_directory/name) for name in ('features.py','labels.py','portable_inference.py')},
        extraction_code=reference(__file__), **outputs))


def group_keys(data, mask):
    return [(str(p), *(int(x) for x in event)) for p, event in zip(data['process'][mask], data['ids'][mask])]


def cut_summary(data, score, threshold, mask, repeats, minimum=20):
    from bdt_performance import efficiency_summary, weighted_efficiency
    passed, groups = score[mask] >= threshold, group_keys(data, mask)
    empirical = efficiency_summary(passed, groups, repeats)
    weighted = weighted_efficiency(passed, groups, data['evaluation_weight'][mask], repeats)
    empirical['supported'] = empirical['pairs'] >= minimum and empirical['events'] >= minimum
    weighted['supported'] = bool(weighted['available'] and empirical['supported'] and weighted.get('event_weight_effective_n', 0) >= minimum)
    return dict(empirical=empirical, weighted=weighted)


def complement(summary):
    result = json.loads(json.dumps(summary))
    for item in result.values():
        if item.get('efficiency') is not None: item['efficiency'] = 1-item['efficiency']
        if 'passed_pairs' in item:item['passed_pairs']=item['pairs']-item['passed_pairs']
        if item.get('weight_sum') is not None and item.get('selected_weight_sum') is not None:
            item['selected_weight_sum']=item['weight_sum']-item['selected_weight_sum']
        if item.get('signed_mc_acceptance_ratio') is not None:
            item['signed_mc_acceptance_ratio']=1-item['signed_mc_acceptance_ratio']
        for key in ('ci95_pair_wilson', 'ci95_event_bootstrap'):
            if key in item and item[key][0] is not None: item[key] = [1-item[key][1], 1-item[key][0]]
    return result


def shape_histogram(values, passed, weights, edges):
    """Finite explicit flow bins preserve every row and signed native weights."""
    values, passed, weights, edges = map(np.asarray, (values, passed, weights, edges))
    if values.shape != passed.shape or values.shape != weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('Shape arrays must be aligned and finite')
    if len(edges) < 2 or not np.all(np.diff(edges) > 0): raise ValueError('Invalid shape edges')
    bins = np.searchsorted(edges, values, side='right')
    return dict(edges=edges.tolist(), before=np.bincount(bins, weights=weights, minlength=len(edges)+1).tolist(),
        after=np.bincount(bins[passed], weights=weights[passed], minlength=len(edges)+1).tolist(),
        before_sumw2=np.bincount(bins, weights=weights**2, minlength=len(edges)+1).tolist(),
        after_sumw2=np.bincount(bins[passed], weights=weights[passed]**2, minlength=len(edges)+1).tolist(),
        flow_bins='first=below first edge; last=at or above final edge')


def compare(args):
    sys.path.insert(0, str(args.classifier_directory.resolve()))
    from features import FEATURE_NAMES, QUALITY_NAMES
    from inference import predict
    from bdt_performance import training_reference, heldout_mask
    from shift_bdt_acceptance_statistics import paired_acceptance_difference, event_histogram_variance
    info = read(args.extraction/'inputs.json')
    if info.get('complete') is not True or info.get('full_population') is not True:
        raise ValueError('Full completed aligned extraction required')
    for name in ('reco', 'labels'):
        if digest(args.extraction/(name+'.npz')) != info[name+'_sha256']:
            raise ValueError('Extraction digest mismatch')
    data, labels = load_npz(args.extraction/'reco.npz'), load_npz(args.extraction/'labels.npz')
    data.update(y=labels['y'])
    if tuple(data['feature_names']) != FEATURE_NAMES: raise ValueError('Feature contract mismatch')
    registry, edges, training_sha = training_reference(args.training_results, args.training_dataset)
    if any(len(edges[key]) != 2 for key in AXIS_KEYS):
        raise ValueError('Comparison requires the original three-bin training contract')
    if not np.array_equal(heldout_mask(data['process'], data['ids'], registry), data['heldout']):
        raise ValueError('Original held-out selection mismatch')
    model_path = args.training_results/'models/bdt_quality_s71.pkl'
    lock = read(args.training_results/'selection_locked.json')
    if lock['selection_uses_test'] is not False: raise ValueError('Threshold lock not validation-only')
    portable=read(info['model']['path'])
    if (digest(info['model']['path'])!=info['model']['sha256'] or
            portable['provenance']['model_name']!='uniform_bdt_3p0_s71' or
            portable['threshold_metadata']!=lock['thresholds']['uniform_bdt_3p0_s71']):
        raise ValueError('Extracted uniform model and validation threshold differ')
    # Explicit CLI training-results path points to trusted local saved study models.
    with model_path.open('rb') as stream: baseline = pickle.load(stream)
    if tuple(np.asarray(FEATURE_NAMES)[baseline['columns']]) != QUALITY_NAMES:
        raise ValueError('Baseline is not the frozen quality-only feature contract')
    if baseline['threshold'] != lock['thresholds']['bdt_quality_s71']:
        raise ValueError('Saved baseline threshold differs from lock')
    quality = predict(baseline, data['X'])
    training = load_npz(args.training_dataset/'reco.npz')
    saved = load_npz(args.training_results/'bdt_quality_s71_scores.npz')
    score_key = 'score' if 'score' in saved else 'scores'
    if not np.array_equal(predict(baseline, training['X']), saved[score_key]):
        raise ValueError('Frozen baseline no longer reproduces original saved predictions')
    models = {'uniform_bdt_3p0_s71': data['score'], 'bdt_quality_s71': quality}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'comparison_scores.npz', ids=data['ids'], process=data['process'],
        pair_index=data['pair_index'], source_index=data['source_index'], heldout=data['heldout'], **models)
    sm = np.isin(data['process'], ('jpsi','dy','qcd')) & data['heldout']
    bin_arrays = [np.searchsorted(edges[key], data['nuisance'][:,i], side='right') for i,key in enumerate(AXIS_KEYS)]
    report = dict(schema='shift-bdt-acceptance-comparison-v1', complete=False, extraction=reference(args.extraction/'inputs.json'),
        training_results=str(args.training_results), training_dataset_sha256=training_sha,
        baseline=reference(model_path), selection_lock=reference(args.training_results/'selection_locked.json'),
        comparison_code=reference(__file__), threshold_selection='Existing validation lock, unchanged; no retuning.',
        test_status='Previously inspected diagnostic population; not a fresh final test.',
        bootstrap_repeats=args.bootstrap_repeats, heldout_sm_pairs=int(sm.sum()),
        label_counts={name:int((sm & (data['y']==value)).sum()) for name,value in LABELS.items()},
        nuisance_edges=edges, models={}, paired_differences=[], physics_selection_ready=False,
        runtime=dict(python=sys.version,numpy=np.__version__),
        statistics_code=reference(Path(__file__).with_name('shift_bdt_acceptance_statistics.py')),
        weighting='SM sampling weight times cross section/full-parent GEN sumw exactly once; no N_eff fit.',
        support='At least20 pairs and independent events; weighted also at least20 effective independent events. This is a support floor, not a precision guarantee.')
    for name, scores in models.items():
        threshold = lock['thresholds'][name]
        entry = dict(threshold=threshold, groups=[], marginal=[], joint=[], shapes=[])
        for process in ('sm','jpsi','dy','qcd'):
            base = sm & (True if process=='sm' else data['process']==process)
            for label, value in LABELS.items():
                target = base & (data['y']==value)
                for topology in ('inclusive', 'Good') + TOPOLOGIES:
                    mask = target & (True if topology=='inclusive' else data['good'] if topology=='Good' else data['topology']==topology)
                    result = cut_summary(data,scores,threshold,mask,args.bootstrap_repeats)
                    entry['groups'].append(dict(process=process,label=label,topology=topology,retention=result,
                        rejection=complement(result) if label=='accidental' else None))
                for i,axis in enumerate(AXES):
                    for b in range(3):
                        mask = target & (bin_arrays[i]==b)
                        entry['marginal'].append(dict(process=process,label=label,axis=axis,bin=b,
                            retention=cut_summary(data,scores,threshold,mask,args.bootstrap_repeats)))
                    plotting_edges = (np.r_[np.linspace(0,20,41),30.,50.,100.] if axis=='mass' else
                        np.linspace(*({'z':(-20000,25000),'radius':(0,200)}[axis]),41))
                    for weight_kind, weights in (('empirical',np.ones(int(target.sum()))),('weighted',data['evaluation_weight'][target])):
                        shape=shape_histogram(data['nuisance'][target,i],scores[target]>=threshold,weights,plotting_edges)
                        groups=group_keys(data,target)
                        passed=scores[target]>=threshold
                        positions=np.searchsorted(plotting_edges,data['nuisance'][target,i],side='right')
                        shape['before_event_sumw2']=event_histogram_variance(positions,groups,weights,len(plotting_edges)+1).tolist()
                        shape['after_event_sumw2']=event_histogram_variance(positions[passed],[g for g,p in zip(groups,passed) if p],weights[passed],len(plotting_edges)+1).tolist()
                        entry['shapes'].append(dict(process=process,label=label,axis=axis,weighting=weight_kind,**shape))
                for bins in itertools.product(range(3), repeat=3):
                    mask = target.copy()
                    for i,b in enumerate(bins):mask &= bin_arrays[i]==b
                    entry['joint'].append(dict(process=process,label=label,bins=list(bins),
                        retention=cut_summary(data,scores,threshold,mask,args.bootstrap_repeats)))
        report['models'][name]=entry
        print(json.dumps(dict(model_complete=name)),flush=True)
    first=models['uniform_bdt_3p0_s71']>=lock['thresholds']['uniform_bdt_3p0_s71']
    second=models['bdt_quality_s71']>=lock['thresholds']['bdt_quality_s71']
    for process in ('sm','jpsi','dy','qcd'):
        base=sm & (True if process=='sm' else data['process']==process)
        for label,value in LABELS.items():
            target=base & (data['y']==value)
            selections=[('inclusive',target)]
            selections += [(axis+str(b),target & (bin_arrays[i]==b)) for i,axis in enumerate(AXES) for b in range(3)]
            selections += [(topology,target & (data['topology']==topology)) for topology in TOPOLOGIES]
            for selection,mask in selections:
                groups=group_keys(data,mask)
                values={kind:paired_acceptance_difference(first[mask],second[mask],groups,weights=weights,repeats=args.bootstrap_repeats)
                    for kind,weights in (('empirical',None),('weighted',data['evaluation_weight'][mask]))}
                report['paired_differences'].append(dict(process=process,label=label,selection=selection,
                    interpretation='Uniform-minus-quality retention; accidental rejection improvement is its negative.',**values))
    make_plots(report,args.output)
    with (args.output/'acceptance_summary.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('model','process','label','topology','weighting','pairs','events','retention','ci95_low','ci95_high','supported'))
        for name,model in report['models'].items():
            for group in model['groups']:
                for weighting,result in group['retention'].items():
                    writer.writerow((name,group['process'],group['label'],group['topology'],weighting,
                        result['pairs'],result['events'],result['efficiency'],*result['ci95_event_bootstrap'],result['supported']))
    with (args.output/'marginal_acceptance.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('model','process','label','axis','bin','weighting','pairs','events','retention','ci95_low','ci95_high','supported'))
        for name,model in report['models'].items():
            for group in model['marginal']:
                for weighting,result in group['retention'].items():
                    writer.writerow((name,group['process'],group['label'],group['axis'],group['bin'],weighting,
                        result['pairs'],result['events'],result['efficiency'],*result['ci95_event_bootstrap'],result['supported']))
    report['complete']=True
    report['artifacts']={p.name:digest(p) for p in args.output.iterdir() if p.is_file()}
    write(args.output/'comparison.json',report)
    print(json.dumps(dict(complete=True,output=str(args.output))),flush=True)


def make_plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short = {'uniform_bdt_3p0_s71':'Uniformity BDT','bdt_quality_s71':'Quality-only BDT'}
    for weighting in ('empirical','weighted'):
        fig, axes=plt.subplots(3,3,figsize=(13,10),sharey=True)
        for row,process in enumerate(('sm','jpsi','dy')):
            for col,axis in enumerate(AXES):
                ax=axes[row,col]
                for index,(name,model) in enumerate(report['models'].items()):
                    vals=[r for r in model['marginal'] if r['process']==process and r['label']=='genuine' and r['axis']==axis]
                    y=[r['retention'][weighting]['efficiency'] for r in vals]
                    ci=[r['retention'][weighting]['ci95_event_bootstrap'] for r in vals]
                    x=np.arange(3)+(index-.5)*.12
                    line,=ax.plot(x,y,'o-',label=short[name])
                    for xi,interval in zip(x,ci):
                        if interval[0] is not None:ax.vlines(xi,*interval,color=line.get_color())
                    unsupported=[j for j,r in enumerate(vals) if not r['retention'][weighting]['supported']]
                    if unsupported:ax.scatter(x[unsupported],np.asarray(y)[unsupported],marker='x',color='black',s=70,zorder=5)
                ax.set_xticks(range(3),('low','middle','high'));ax.set_ylim(0,1.06)
                ax.set_title(process.upper()+' — '+axis);ax.grid(alpha=.2)
                limits=report['nuisance_edges'][AXIS_KEYS[col]]
                ax.set_xlabel(f'Boundaries: {limits[0]:.4g}, {limits[1]:.4g} '+('GeV' if axis=='mass' else 'cm'))
                if col==0:ax.set_ylabel('Genuine retention')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(weighting.capitalize()+' fixed-cut acceptance; bins frozen from training\n95% event bootstrap; same held-out events, existing validation thresholds; x = insufficient support')
        fig.tight_layout()
        fig.savefig(output/f'acceptance_profiles_{weighting}.pdf',dpi=140)
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
        for ax,process in zip(axes,('sm','jpsi','dy')):
            for index,(name,model) in enumerate(report['models'].items()):
                values=[next(r['retention'][weighting] for r in model['groups']
                    if r['process']==process and r['label']=='genuine' and r['topology']==topology) for topology in TOPOLOGIES]
                valid=[j for j,r in enumerate(values) if r['efficiency'] is not None]
                x=np.asarray(valid)+(index-.5)*.12
                y=np.asarray([values[j]['efficiency'] for j in valid])
                line,=ax.plot(x,y,'o',label=short[name])
                for xi,j in zip(x,valid):
                    interval=values[j]['ci95_event_bootstrap']
                    if interval[0] is not None:ax.vlines(xi,*interval,color=line.get_color())
                bad=[k for k,j in enumerate(valid) if not values[j]['supported']]
                if bad:ax.scatter(x[bad],y[bad],marker='x',color='black',s=70,zorder=5)
            ax.set_xticks(range(len(TOPOLOGIES)),TOPOLOGIES,rotation=35,ha='right');ax.set_ylim(0,1.06)
            ax.set_title(process.upper());ax.grid(alpha=.2)
        axes[0].set_ylabel('Genuine retention');axes[0].legend(fontsize=8)
        fig.suptitle(weighting.capitalize()+' topology acceptance; 95% event bootstrap; x = insufficient support')
        fig.tight_layout()
        fig.savefig(output/f'topology_acceptance_{weighting}.pdf',dpi=140)
        plt.close(fig)
        for process in ('sm','jpsi','dy'):
            fig,axes=plt.subplots(3,2,figsize=(10,12),sharex=True,sharey=True)
            for col,(name,model) in enumerate(report['models'].items()):
                for z in range(3):
                    ax=axes[z,col];matrix=np.full((3,3),np.nan)
                    cells={}
                    for entry in model['joint']:
                        if entry['process']!=process or entry['label']!='genuine' or entry['bins'][1]!=z:continue
                        m,_,r=entry['bins'];cell=entry['retention'][weighting];cells[m,r]=cell
                        if cell['supported']:matrix[r,m]=cell['efficiency']
                    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
                    im=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,origin='lower')
                    for (m,r),cell in cells.items():
                        value=cell['efficiency'];text='unmeasured' if value is None else f'{value:.1%}'
                        if not cell['supported']:text+='*'
                        ax.text(m,r,text+'\nn='+str(cell['pairs']),ha='center',va='center',fontsize=8,color='black' if not cell['supported'] or (value or 0)>.65 else 'white')
                    z_edges=report['nuisance_edges'][AXIS_KEYS[1]]
                    z_text=(f'z < {z_edges[0]:.5g}' if z==0 else f'z >= {z_edges[1]:.5g}' if z==2 else f'{z_edges[0]:.5g} <= z < {z_edges[1]:.5g}')
                    ax.set_title(short[name]+'; '+z_text+' cm',fontsize=10)
                    ax.set_xticks(range(3),('low','middle','high'));ax.set_yticks(range(3),('low','middle','high'))
                    ax.set_xlabel('Mass third');ax.set_ylabel('Radius third')
            fig.suptitle(process.upper()+' '+weighting+' genuine acceptance in fixed training bins\nGrey/* = insufficient support; n = raw pairs; full event intervals in JSON')
            fig.tight_layout(rect=(0,0,.92,.95))
            fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.02,label='Retention')
            fig.savefig(output/f'joint_acceptance_{process}_{weighting}.pdf',dpi=140)
            plt.close(fig)
        for process in ('sm','jpsi','dy','qcd'):
            fig,axes=plt.subplots(3,3,figsize=(14,11))
            for row,label in enumerate(LABELS):
                for col,axis in enumerate(AXES):
                    ax=axes[row,col]
                    plotted=False
                    for index,(name,model) in enumerate(report['models'].items()):
                        h=next(r for r in model['shapes'] if (r['process'],r['label'],r['axis'],r['weighting'])==(process,label,axis,weighting))
                        before=np.asarray(h['before']);after=np.asarray(h['after']);scale=before.sum()
                        if scale<=0:continue
                        plotted=True
                        centers=(np.asarray(h['edges'][:-1])+np.asarray(h['edges'][1:]))/2
                        if index==0:ax.step(centers,before[1:-1]/scale,where='mid',color='black',label='Before cut')
                        ax.step(centers,after[1:-1]/scale,where='mid',label=short[name])
                    if not plotted:ax.text(.5,.5,'No held-out pairs with positive total weight',ha='center',va='center',transform=ax.transAxes,fontsize=8,wrap=True)
                    info=next(r['retention'][weighting] for r in next(iter(report['models'].values()))['groups']
                        if r['process']==process and r['label']==label and r['topology']=='inclusive')
                    if not info['supported']:ax.text(.98,.96,'Insufficient support',ha='right',va='top',transform=ax.transAxes,fontsize=8,color='darkred')
                    ax.set_xlabel(AXIS_LABELS[col]);ax.set_ylabel(label+' / before total');ax.grid(alpha=.2)
            for legend_axis in axes.ravel():
                if legend_axis.get_legend_handles_labels()[0]:
                    legend_axis.legend(fontsize=8)
                    break
            fig.suptitle(process.upper()+' '+weighting+' before/after shapes; common before-cut normalization\nFlows retained in JSON; unknown labels remain separate')
            fig.tight_layout()
            fig.savefig(output/f'shapes_{process}_{weighting}.pdf',dpi=140)
            plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('extract')
    for name in ('manifest','evaluation-rows','model','classifier-directory','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-files',type=int,help='Bounded prefix pilot only; not accepted by comparison')
    p=commands.add_parser('compare')
    for name in ('extraction','classifier-directory','training-results','training-dataset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap-repeats',type=int,default=200)
    args=parser.parse_args()
    if args.command=='extract':extract(args)
    else:compare(args)


if __name__=='__main__':main()
