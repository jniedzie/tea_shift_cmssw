#!/usr/bin/env python3
# Code style: AI-code. AI-generated validation-only overall model selection.
"""Select one complete-information ensemble, then evaluate that choice only.

An immutable manifest fixes candidates and criteria before reading their
validation predictions. Development and new signal rows are opened only
after the model and global validation cut have been locked. No signal,
unknown truth label or development outcome changes the choice.
"""
import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np

from shift_vertex_signal_scores import load, read, reference, require, verify, validate_loaded_sources


def reconstructed_proper_length_mm(*args, **kwargs):
    # Import after initialize() has installed the declared frozen source paths.
    from shift_vertex_reco_lifetime import reconstructed_proper_length_mm as calculate
    return calculate(*args, **kwargs)


def reconstructed_vertex_distance_m(*args, **kwargs):
    from shift_vertex_reco_lifetime import reconstructed_vertex_distance_m as calculate
    return calculate(*args, **kwargs)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def initialize(manifest):
    for refs in manifest['sources'].values():
        for ref in refs: verify(ref)
    sys.path[:0] = manifest['import_directories']
    for module in ('vertex_complete_bdt', 'neural_vertex', 'neural_vertex_categories', 'vertex_complete_ensemble'):
        importlib.import_module(module)
    validate_loaded_sources(manifest['sources'])


def lifetime_summary(scores, y, process, length, train_length, train_y, threshold):
    from evaluation import distance_correlation, fraction
    genuine_train = (train_y == 1) & np.isfinite(train_length)
    edges = np.quantile(train_length[genuine_train], [1/3, 2/3])
    valid = np.isfinite(length)
    bins = np.searchsorted(edges, length, side='right')
    passed = scores >= threshold
    profiles, spans, efficiencies = {}, [], []
    scopes = {'pooled': np.ones(len(y), dtype=bool)}
    scopes.update({str(p): process == p for p in np.unique(process)
                   if np.count_nonzero((process == p) & (y == 1)) >= 50})
    for name, mask in scopes.items():
        rows = []
        for index in range(3):
            row = fraction(mask & valid & (y == 1) & (bins == index), passed)
            row.update(bin=index, supported=row['n'] >= 30); rows.append(row)
        values = [row['efficiency'] for row in rows if row['supported']]
        efficiencies.extend(values)
        if len(values) >= 2: spans.append(max(values)-min(values))
        profiles[name] = rows
    correlations = {str(label): distance_correlation(scores[valid & (y == label)],
        np.sign(length[valid & (y == label)])*np.log1p(np.abs(length[valid & (y == label)]))) for label in (0,1)}
    return dict(interior_edges_mm=edges.tolist(), profiles=profiles,
        minimum_supported_retention=min(efficiencies) if efficiencies else None,
        maximum_supported_span=max(spans) if spans else None,
        score_length_dcor=correlations, invalid_rows=int((~valid).sum()),
        definition='Signed m*(vertex-nominal target) dot p / |p|^2, cm converted to mm; RECO only, not a calibrated proper-time measurement')


def gate_failures(metrics, lifetime, bounds):
    from shift_vertex_complete_diagnostics import gates
    _, failures = gates(metrics)
    lower, span = lifetime['minimum_supported_retention'], lifetime['maximum_supported_span']
    if lower is None or lower < bounds['minimum_supported_genuine_retention']: failures.append('reco_length_minimum')
    if span is None or span > bounds['maximum_supported_marginal_span']: failures.append('reco_length_span')
    if any(value is None or value > bounds['maximum_class_joint_dcor'] for value in lifetime['score_length_dcor'].values()):
        failures.append('reco_length_dcor')
    return failures


def get_data(manifest):
    protocol = read(manifest['base_protocol']['path']); verify(manifest['base_protocol'])
    for key in ('reco','labels','splits','cache_manifest'): verify(protocol[key])
    reco, labels, splits = [load(protocol[key]['path']) for key in ('reco','labels','splits')]
    for key in ('ids','process','pair_index','source_index'):
        require(np.array_equal(reco[key], splits[key]), 'Candidate/split identity changed')
    require(reco['ids'].dtype == np.uint64 and reco['X'].shape[1] == 119, 'Incomplete/rounded RECO cache')
    return protocol, reco, labels, splits


def select(manifest_path, output):
    manifest = read(manifest_path)
    require(not output.exists(), 'Refusing selection evidence overwrite')
    require(manifest['selection_uses_signal'] is False, 'Signal selection is prohibited')
    for key in ('nominal_target_authority','ensemble_source','lifetime_source','selection_source'): verify(manifest[key])
    require(reference(Path(__file__)) == manifest['selection_source'], 'Execute the frozen selection utility')
    initialize(manifest)
    from neural_vertex_study import summary
    from vertex_complete_ensemble import CompleteVertexEnsemble
    protocol, reco, labels, splits = get_data(manifest)
    sm = np.isin(reco['process'], ['dy','jpsi','qcd']) & (labels['y'] >= 0)
    train, val = sm & (splits['split'] == 0), sm & (splits['split'] == 1)
    length = reconstructed_proper_length_mm(reco['X'], reco['feature_names'], reco['nuisance'][:,0], manifest['nominal_target_cm'])
    results, models = {}, {}
    for candidate in manifest['candidates']:
        name = candidate['name']; members, explicit = [], []
        for member in candidate['members']:
            for key in ('fit','model','history','protocol'): verify(member[key])
            fit = read(member['fit']['path'])
            require(fit['model'] == member['model'] and fit['protocol'] == member['protocol'], 'Changed model/fit provenance')
            with Path(member['model']['path']).open('rb') as stream: bundle = pickle.load(stream)
            require(bundle['requires_gen'] is False and bundle['model'].n_features_in_ == 119, 'Incomplete or GEN-dependent member')
            if 'columns' in bundle: require(np.array_equal(bundle['columns'],np.arange(119)), 'Feature ablation in member')
            members.append(bundle['model']); explicit.append(bundle['category_required'])
        validate_loaded_sources(manifest['sources'])
        model = CompleteVertexEnsemble(members, explicit)
        scores = np.asarray([m.predict_proba(reco['X'][val], reco['topology'][val])[:,1] if flag else
            m.predict_proba(reco['X'][val])[:,1] for m,flag in zip(members,explicit)], dtype=float)
        variants = {'all': scores.mean(axis=0)}
        for index in range(len(members)):
            variants['omit_'+str(index)] = np.delete(scores,index,axis=0).mean(axis=0) if len(members)>1 else scores[0]
        checks = {}
        for label, score in variants.items():
            metrics = summary(score, labels['y'][val], reco['process'][val], reco['nuisance'][val],
                reco['nuisance'][train], labels['y'][train], category=reco['topology'][val])
            lifetime = lifetime_summary(score, labels['y'][val], reco['process'][val], length[val], length[train], labels['y'][train], metrics['threshold'])
            checks[label] = dict(metrics=metrics, lifetime=lifetime,
                failed_gates=gate_failures(metrics,lifetime,manifest['acceptance_envelope']))
        results[name] = dict(candidate=candidate, checks=checks,
            eligible=all(not check['failed_gates'] for check in checks.values()),
            worst_variant_rejection=min(check['metrics']['rejection'] for check in checks.values()))
        models[name] = model
    eligible = [name for name,row in results.items() if row['eligible']]
    if not eligible:
        output.mkdir(parents=True)
        write(output/'validation_ranking.json',dict(candidates=results,choice=None,
            selection_uses_test=False,selection_uses_signal=False))
    require(bool(eligible), 'No candidate meets the declared performance/dependence/stability criteria')
    winner = max(eligible,key=lambda name:(results[name]['worst_variant_rejection'],results[name]['checks']['all']['metrics']['rejection'],name))
    chosen = results[winner]; threshold = chosen['checks']['all']['metrics']['threshold']
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    frozen = {}
    for name,refs in manifest['sources'].items():
        ref = refs[0]; target = snapshot/name; shutil.copyfile(ref['path'],target); frozen[name] = reference(target)
    bundle = dict(model=models[winner], threshold=threshold, requires_gen=False, category_required=True,
        feature_names=list(protocol['input_feature_names']), choice=winner, deployment_ready=False)
    with (output/'model.pkl').open('xb') as stream: pickle.dump(bundle,stream)
    with (output/'model.pkl').open('rb') as stream: restored=pickle.load(stream)
    expected=models[winner].predict_proba(reco['X'][val],reco['topology'][val])
    require(np.array_equal(expected,restored['model'].predict_proba(reco['X'][val],reco['topology'][val])),'Ensemble reload parity failed')
    write(output/'validation_ranking.json',dict(candidates=results,choice=winner,selection_uses_test=False,selection_uses_signal=False))
    lock = dict(complete=True,choice=winner,threshold=threshold,model=reference(output/'model.pkl'),
        manifest=reference(manifest_path),ranking=reference(output/'validation_ranking.json'),sources=frozen,
        validation=chosen['checks']['all'],stability=chosen['checks'],selection_uses_test=False,
        selection_uses_signal=False,requires_gen=False,physics_ready=False,reload_parity_exact=True,
        interpretation='One equal-weight complete-input ensemble; omission variants are a stability diagnostic, not independent uncertainty')
    write(output/'selection_locked.json',lock)
    print(json.dumps(dict(choice=winner,rejection=chosen['checks']['all']['metrics']['rejection'],threshold=threshold)),flush=True)


def evaluate(output, signal_cache):
    lock = read(output/'selection_locked.json')
    for key in ('model','manifest','ranking'): verify(lock[key])
    for ref in lock['sources'].values(): verify(ref)
    manifest=read(lock['manifest']['path']); initialize(manifest)
    from neural_vertex_study import summary
    from shift_vertex_efficiency import summarize_vertex_efficiency
    from shift_vertex_signal_scores import validate_parent_alignment, prediction_packet, reject_parent_overlap
    protocol,reco,labels,splits=get_data(manifest)
    with Path(lock['model']['path']).open('rb') as stream: bundle=pickle.load(stream)
    require(bundle['threshold']==lock['threshold'] and bundle['choice']==lock['choice'],'Locked choice/cut changed')
    sm=np.isin(reco['process'],['dy','jpsi','qcd']); train=sm&(splits['split']==0)&(labels['y']>=0)
    test=sm&(splits['split']==2);known=test&(labels['y']>=0)
    score=bundle['model'].predict_proba(reco['X'][test],reco['topology'][test])[:,1]
    length=reconstructed_proper_length_mm(reco['X'],reco['feature_names'],reco['nuisance'][:,0],manifest['nominal_target_cm'])
    metrics=summary(score[labels['y'][test]>=0],labels['y'][known],reco['process'][known],reco['nuisance'][known],
        reco['nuisance'][train],labels['y'][train],threshold=lock['threshold'],category=reco['topology'][known])
    lifetime=lifetime_summary(score[labels['y'][test]>=0],labels['y'][known],reco['process'][known],length[known],length[train],labels['y'][train],lock['threshold'])
    np.savez_compressed(output/'development_scores.npz',score=score,y=labels['y'][test],process=reco['process'][test],
        category=reco['topology'][test],nuisance=reco['nuisance'][test],reco_length_mm=length[test],
        ids=reco['ids'][test],evaluation_weight=reco['evaluation_weight'][test])
    metadata=read(signal_cache/'inputs_resolved.json')
    verify(metadata['protocol'])
    for ref in metadata['cache'].values(): verify(ref)
    sr,sl,se=[load(metadata['cache'][key]['path']) for key in ('reco','labels','events')]
    validate_parent_alignment(sr,sl,se,metadata);verify(protocol['events'])
    reject_parent_overlap(se['ids'],load(protocol['events']['path'])['ids'])
    packet=prediction_packet(sr,protocol['input_feature_names'])
    ss=bundle['model'].predict_proba(packet['X'],packet['topology'])[:,1]
    signal_manifest=read((signal_cache/'extraction_protocol.json'))
    info={}
    for point in signal_manifest['points']:
        audit=read(point['gen_audit']['path']);verify(point['gen_audit'])
        verify(point['gen_inventory']);verify(point['receipt'])
        parent=read(point['gen_inventory']['path'])['points'][point['point']]
        point_ids=se['ids'][se['point']==point['point']]
        require(np.array_equal(point_ids,np.asarray([row['id']for row in audit['event_rows']],dtype=np.uint64)),
            'GEN proper-lifetime parent alignment changed')
        require(np.isclose(parent['proper_ctau_mm'],audit['expected_proper_length_mm'],rtol=1e-10),
            'Proper lifetime differs from GEN authority')
        info[point['point']]=dict(mass_gev=parent['mass_gev'],proper_ctau_mm=parent['proper_ctau_mm'],
            response=summarize_vertex_efficiency(se['ids'][se['point']==point['point']],se['native_weight'][se['point']==point['point']],
                sr['ids'][sr['point']==point['point']],sl['y'][sr['point']==point['point']],ss[sr['point']==point['point']],lock['threshold'],
                qualified_genuine_mask=sl['genuine_signal_associated'][sr['point']==point['point']]))
    np.savez_compressed(output/'signal_scores.npz',score=ss,y=sl['y'],qualified=sl['genuine_signal_associated'],
        point=sr['point'],ids=sr['ids'],category=sr['topology'],nuisance=sr['nuisance'],
        reco_length_mm=reconstructed_proper_length_mm(sr['X'],sr['feature_names'],sr['nuisance'][:,0],manifest['nominal_target_cm']))
    report=dict(choice=lock['choice'],threshold=lock['threshold'],development=metrics,reconstructed_lifetime=lifetime,
        signal_points=info,selection=reference(output/'selection_locked.json'),signal_cache=reference(signal_cache/'inputs_resolved.json'),
        signal_generation_not_in_inference=True,selection_uses_signal=False,physics_ready=False,
        development_previously_inspected=True,source=reference(Path(__file__)))
    write(output/'evaluation.json',report)
    write(output/'completion.json',dict(complete=True,selection=report['selection'],evaluation=reference(output/'evaluation.json'),
        development_scores=reference(output/'development_scores.npz'),signal_scores=reference(output/'signal_scores.npz')))
    print(json.dumps(dict(choice=lock['choice'],rejection=metrics['rejection'],genuine=metrics['common_vertex']['efficiency'],signal_points=len(info))),flush=True)


def evaluate_signal(choice_output, signal_cache, output):
    """Score a complete signal grid with the already locked model and cut.

    This action never opens SM development predictions or compares models.
    Its fresh output keeps the original selection and 30-point validation
    evidence immutable while completed signal production can be extended.
    """
    require(not output.exists(), 'Refusing signal response evidence overwrite')
    lock_path = choice_output / 'selection_locked.json'
    lock = read(lock_path)
    for key in ('model', 'manifest', 'ranking'):
        verify(lock[key])
    for ref in lock['sources'].values():
        verify(ref)
    require(lock['selection_uses_signal'] is False and lock['requires_gen'] is False,
            'Signal-dependent choice or inference is prohibited')
    manifest = read(lock['manifest']['path'])
    initialize(manifest)
    from shift_vertex_efficiency import summarize_vertex_efficiency
    from shift_vertex_signal_scores import validate_parent_alignment, prediction_packet, reject_parent_overlap
    protocol = read(manifest['base_protocol']['path'])
    verify(manifest['base_protocol'])
    with Path(lock['model']['path']).open('rb') as stream:
        bundle = pickle.load(stream)
    require(bundle['threshold'] == lock['threshold'] and bundle['choice'] == lock['choice']
            and bundle['requires_gen'] is False, 'Locked choice/cut changed')
    metadata_path = signal_cache / 'inputs_resolved.json'
    metadata = read(metadata_path)
    verify(metadata['protocol'])
    for ref in metadata['cache'].values():
        verify(ref)
    sr, sl, se = [load(metadata['cache'][key]['path']) for key in ('reco', 'labels', 'events')]
    validate_parent_alignment(sr, sl, se, metadata)
    verify(protocol['events'])
    reject_parent_overlap(se['ids'], load(protocol['events']['path'])['ids'])
    packet = prediction_packet(sr, protocol['input_feature_names'])
    require(list(bundle['feature_names']) == list(protocol['input_feature_names']),
            'Chosen model feature order differs from signal cache')
    scores = bundle['model'].predict_proba(packet['X'], packet['topology'])[:, 1]
    require(scores.shape == (len(sr['ids']),) and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), 'Invalid signal scores')
    signal_manifest = read(metadata['protocol']['path'])
    info = {}
    for point in signal_manifest['points']:
        for key in ('gen_audit', 'gen_inventory', 'receipt'):
            verify(point[key])
        audit = read(point['gen_audit']['path'])
        parent = read(point['gen_inventory']['path'])['points'][point['point']]
        parent_mask = se['point'] == point['point']
        pair_mask = sr['point'] == point['point']
        require(np.array_equal(se['ids'][parent_mask], np.asarray(
            [row['id'] for row in audit['event_rows']], dtype=np.uint64)),
            'GEN response parent alignment changed')
        require(np.isclose(parent['proper_ctau_mm'], audit['expected_proper_length_mm'], rtol=1e-10),
                'Proper lifetime differs from GEN authority')
        info[point['point']] = dict(mass_gev=parent['mass_gev'], proper_ctau_mm=parent['proper_ctau_mm'],
            full_GEN_events=parent['events'], native_sumw=parent['native_sumw'], native_sumw2=parent['native_sumw2'],
            full_GEN_event_ids_sha256=hashlib.sha256(np.asarray(audit['event_ids'], dtype='<u8').tobytes(order='C')).hexdigest(),
            gen_audit=point['gen_audit'],
            response=summarize_vertex_efficiency(se['ids'][parent_mask], se['native_weight'][parent_mask],
                sr['ids'][pair_mask], sl['y'][pair_mask], scores[pair_mask], lock['threshold'],
                qualified_genuine_mask=sl['genuine_signal_associated'][pair_mask]))
    original_path = choice_output / 'evaluation.json'
    original = read(original_path)
    for point, entry in original['signal_points'].items():
        if point in info:
            require({key: info[point][key] for key in entry} == entry,
                    'Previously frozen signal response changed: ' + point)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'signal_scores.npz', score=scores, y=sl['y'],
        qualified=sl['genuine_signal_associated'], point=sr['point'], ids=sr['ids'],
        category=sr['topology'], nuisance=sr['nuisance'], reco_length_mm=reconstructed_proper_length_mm(
            sr['X'], sr['feature_names'], sr['nuisance'][:, 0], manifest['nominal_target_cm']))
    report = dict(choice=lock['choice'], threshold=lock['threshold'], signal_points=info,
        selection=reference(lock_path), signal_cache=reference(metadata_path),
        signal_generation_not_in_inference=True, selection_uses_signal=False,
        previously_inspected_signal_points=sorted(set(info).intersection(original['signal_points'])),
        original_response=reference(original_path), original_signal_response_parity_exact=True,
        source=reference(Path(__file__)), physics_ready=False)
    write(output / 'evaluation.json', report)
    write(output / 'completion.json', dict(complete=True, selection=report['selection'],
        evaluation=reference(output / 'evaluation.json'), signal_scores=reference(output / 'signal_scores.npz')))
    print(json.dumps(dict(choice=lock['choice'], signal_points=len(info),
        generated_parents=len(se['ids']), retained_pairs=len(scores))), flush=True)


def render(output, plot_output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import roc_curve,roc_auc_score
    completed=read(output/'completion.json')
    for key in ('selection','evaluation','development_scores','signal_scores'):verify(completed[key])
    lock=read(completed['selection']['path']);report=read(completed['evaluation']['path'])
    data=load(completed['development_scores']['path'])
    manifest=read(lock['manifest']['path']);protocol,reco,labels,splits=get_data(manifest)
    require(not plot_output.exists(),'Refusing final plot overwrite')
    plot_output.mkdir(parents=True)
    score,y,process=data['score'],data['y'],data['process'];passed=score>=lock['threshold']
    title='Full-information DNN ensemble'
    footer='Fixed validation cut; previously inspected SM development. Truth labels are evaluation labels, never inference inputs.'
    with PdfPages(plot_output/'roc_and_scores.pdf')as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        for p,color in [('pooled','#252525'),('dy','#278336'),('jpsi','#c94333')]:
            mask=(y>=0)&(np.ones(len(y),bool)if p=='pooled'else process==p)
            fpr,tpr,_=roc_curve(y[mask],score[mask]);auc=roc_auc_score(y[mask],score[mask])
            axes[0].plot(fpr,tpr,color=color,label=f'{p.upper() if p!="pooled" else "Pooled SM"}: AUC={auc:.3f}')
            genuine=(y==1)&mask;fake=(y==0)&mask
            axes[0].scatter(passed[fake].mean(),passed[genuine].mean(),color=color,s=28)
        axes[0].plot([0,1],[0,1],color='gray',linestyle=':',linewidth=1)
        axes[0].set(xlabel='Fake vertices retained',ylabel='Genuine vertices retained',xlim=(0,1),ylim=(0,1))
        axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
        for label,word,color in [(1,'Genuine','#c94333'),(0,'Fake','#235789'),(-1,'Unknown truth','#777777')]:
            values=score[y==label]
            if len(values):axes[1].hist(values,bins=np.linspace(0,1,41),weights=np.ones(len(values))/len(values),histtype='step',color=color,label=f'{word}: {len(values)}',linewidth=1.6)
        axes[1].axvline(lock['threshold'],color='black',linestyle='--',label=f'Cut={lock["threshold"]:.4f}')
        axes[1].set(xlabel='Mean DNN score',ylabel='Fraction per bin within each truth class',xlim=(0,1));axes[1].legend(fontsize=9)
        fig.suptitle(title+' — one selected model',fontsize=14)
        fig.text(.03,.015,footer+'\nPair-count shapes, not a cross-section-weighted SM mixture. DNN score is not a calibrated truth probability.',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,.94));pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharex=True)
        for ax,p in zip(axes,['dy','jpsi','qcd']):
            for label,word,color in [(1,'Genuine','#c94333'),(0,'Fake','#235789'),(-1,'Unknown','#777777')]:
                values=score[(y==label)&(process==p)]
                if len(values):ax.hist(values,bins=np.linspace(0,1,41),weights=np.ones(len(values))/len(values),histtype='step',color=color,label=f'{word}: {len(values)}',linewidth=1.5)
            ax.axvline(lock['threshold'],color='black',linestyle='--',linewidth=1)
            ax.set(title=p.upper(),xlabel='Mean DNN score',ylabel='Fraction per bin');ax.legend(fontsize=8)
            if p=='qcd':ax.text(.03,.85,'No genuine QCD examples\nin this development partition',transform=ax.transAxes,fontsize=8)
        fig.suptitle(title+' — separate process score distributions',fontsize=13)
        fig.text(.03,.015,footer,fontsize=8);fig.tight_layout(rect=(0,.06,1,.92));pdf.savefig(fig);plt.close(fig)

    # Resample complete physical events, retaining correlations among their pairs.
    keys=[str(p)+':'+':'.join(map(str,identity))for p,identity in zip(process,data['ids'])]
    _,inverse=np.unique(keys,return_inverse=True);parents=int(inverse.max())+1
    rng=np.random.default_rng(141071)
    draws=np.asarray([np.bincount(rng.integers(0,parents,parents),minlength=parents)for _ in range(300)])
    def stats(mask):
        n=int(mask.sum());k=int((mask&passed).sum())
        if not n:return dict(n=0,passed=0,efficiency=None,ci95=None,parents=0)
        total=np.bincount(inverse[mask],minlength=parents)
        selected=np.bincount(inverse[mask&passed],minlength=parents)
        den=draws@total;num=draws@selected;values=num[den>0]/den[den>0]
        return dict(n=n,passed=k,efficiency=k/n,ci95=np.quantile(values,[.025,.975]).tolist(),parents=int(np.count_nonzero(total)))
    train=(splits['split']==0)&(labels['y']==1)&np.isin(reco['process'],['dy','jpsi','qcd'])
    development=(splits['split']==2)&np.isin(reco['process'],['dy','jpsi','qcd'])
    require(np.array_equal(reco['ids'][development],data['ids']) and
            np.array_equal(reco['process'][development],process) and
            np.array_equal(labels['y'][development],y), 'Development coordinates/predictions do not align')
    all_distance=reconstructed_vertex_distance_m(reco['X'],reco['feature_names'],manifest['nominal_target_cm'])
    distance=all_distance[development]
    distance_edges=np.arange(0.,151.,10.)
    distance_centres=(distance_edges[:-1]+distance_edges[1:])/2
    in_range=np.isfinite(distance)&(distance>=0)&(distance<=150)
    distance_bins=np.minimum(np.searchsorted(distance_edges,distance,side='right')-1,14)
    profiles={}
    with PdfPages(plot_output/'reconstructed_mass_lifetime.pdf')as pdf:
        fig,axes=plt.subplots(2,3,figsize=(14,9),sharey=True)
        for row,(name,t,v)in enumerate([('Reconstructed mass (GeV)',reco['nuisance'][:,0],data['nuisance'][:,0]),
                                        ('Distance from nominal production point\nto fitted vertex (m)',all_distance,distance)]):
            if row==0:
                edges=np.quantile(t[train&np.isfinite(t)],np.arange(1,6)/6)
                bins=np.searchsorted(edges,v,side='right');positions=np.arange(6);count=6
                ticks=[f'< {edges[0]:.3g}']+[f'{a:.3g}\nto {b:.3g}'for a,b in zip(edges[:-1],edges[1:])]+[f'≥ {edges[-1]:.3g}']
            else:
                edges=distance_edges;bins=distance_bins;positions=distance_centres;count=15
            for column,p in enumerate(['pooled','dy','jpsi']):
                ax=axes[row,column];mask=(y==1)&np.isfinite(v)&(np.ones(len(y),bool)if p=='pooled'else process==p)
                if row==1:mask=mask&in_range
                values=[stats(mask&(bins==i))for i in range(count)]
                profiles[name+'_'+p]=dict(edges=edges.tolist(),bins=values)
                for i,s in enumerate(values):
                    position=positions[i]
                    if row==1 and s['n']<30:
                        ax.axvspan(edges[i],edges[i+1],color='#eeeeee',zorder=0)
                    ax.text(position,.51,f'n={s["n"]}',ha='center',fontsize=6 if row==1 else 7,
                            color='black'if s['n']>=30 else '#999999',rotation=90 if row==1 else 0)
                    if not s['n']:continue
                    eff=s['efficiency'];lo,hi=s['ci95'];supported=s['n']>=30
                    ax.errorbar(position,eff,yerr=[[max(0,eff-lo)],[max(0,hi-eff)]],
                                xerr=5 if row==1 else None,fmt='o',
                                color='#235789'if supported else '#aaaaaa',capsize=3)
                ax.axhline(.9,color='gray',linestyle='--',linewidth=1);ax.grid(axis='y',alpha=.2)
                ax.set(ylim=(.5,1.025),xlabel=name,title='Pooled SM'if p=='pooled'else p.upper())
                if row==0:ax.set(xticks=range(6),xticklabels=ticks)
                else:ax.set(xlim=(0,150),xticks=np.arange(0,151,25))
                if column==0:ax.set_ylabel('Known genuine vertices retained')
                ax.tick_params(axis='x',labelsize=7)
        fig.suptitle(title+' — reconstructed mass and laboratory distance',fontsize=14)
        fig.text(.03,.012,'Mass: six TRAIN-quantile bins. Distance: fixed10m bins over0–150m. Bars:95% event bootstrap (300 replicas); gray: fewer than30 genuine pairs.\nDistance is the3D separation from nominal target (0,0,148m), without a boost correction. Fitted positions can differ from true decay positions.',fontsize=8)
        fig.tight_layout(rect=(0,.075,1,.95));pdf.savefig(fig);plt.close(fig)
        mass_edges=np.quantile(reco['nuisance'][train,0],[.25,.5,.75])
        b1=np.searchsorted(mass_edges,data['nuisance'][:,0],side='right')
        fig,axes=plt.subplots(1,3,figsize=(13.5,8));last=None
        for ax,p in zip(axes,['pooled','dy','jpsi']):
            matrix=np.full((15,4),np.nan);values=[]
            for i in range(4):
                for j in range(15):
                    mask=(y==1)&(b1==i)&(distance_bins==j)&in_range&(np.ones(len(y),bool)if p=='pooled'else process==p)
                    s=stats(mask);s.update(mass_bin=i,distance_bin=j);values.append(s)
                    if s['n']>=30:matrix[j,i]=s['efficiency']
                    text=f'{100*s["efficiency"]:.0f}% {s["passed"]}/{s["n"]}'if s['n']>=30 else f'n={s["n"]}'
                    ink='white' if s['n']>=30 and s['efficiency']>=.86 else 'black'
                    ax.text(i+.5,distance_centres[j],text,ha='center',va='center',fontsize=6.5,color=ink)
            profiles['joint_'+p]=dict(mass_edges=mass_edges.tolist(),distance_edges_m=distance_edges.tolist(),bins=values)
            cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
            last=ax.pcolormesh(np.arange(5),distance_edges,matrix,vmin=.65,vmax=1,cmap=cmap)
            ax.set(title='Pooled SM'if p=='pooled'else p.upper(),xlabel='Reconstructed mass quartile',
                   ylabel='Distance from nominal production point (m)',xticks=np.arange(4)+.5,
                   xticklabels=range(1,5),ylim=(0,150),yticks=np.arange(0,151,25))
        fig.suptitle(title+' — joint reconstructed mass/distance check',fontsize=14)
        fig.text(.03,.025,'Mass quartiles fixed from TRAIN; distance bins fixed at10m. Color shown only with ≥30 known genuine pairs; gray cells remain unvalidated.\nCounts are passing/total. Distance is measured from nominal target at z=148m. These maps describe classifier retention, not full reconstruction or recording acceptance.',fontsize=8)
        fig.subplots_adjust(left=.06,right=.88,bottom=.18,top=.85,wspace=.35)
        fig.colorbar(last,cax=fig.add_axes([.92,.18,.018,.67]),label='Genuine retention')
        pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
        for ax,p in zip(axes.ravel(),['pooled','dy','jpsi','qcd']):
            scope=np.ones(len(y),bool)if p=='pooled'else process==p
            profiles['distance_coverage_'+p]={}
            for label,word,color in [(1,'Genuine','#c94333'),(0,'Fake','#235789'),(-1,'Unclassified','#777777')]:
                mask=scope&(y==label);values=distance[mask&in_range]
                coverage=dict(total=int(mask.sum()),in_range=int((mask&in_range).sum()),
                              above150=int((mask&np.isfinite(distance)&(distance>150)).sum()),
                              invalid=int((mask&~np.isfinite(distance)).sum()))
                profiles['distance_coverage_'+p][word]=coverage
                if len(values):ax.hist(values,bins=distance_edges,weights=np.ones(len(values))/mask.sum(),
                    histtype='step',color=color,linewidth=1.5,label=f'{word}: {mask.sum()}')
            ax.set(xlim=(0,150),xticks=np.arange(0,151,25),title='Pooled SM'if p=='pooled'else p.upper(),
                   xlabel='Distance from nominal production point to fitted vertex (m)',ylabel='Fraction per10m bin within truth class')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
            if p=='qcd':ax.text(.04,.78,'No genuine QCD examples in this partition',transform=ax.transAxes,fontsize=8)
        fig.suptitle(title+' — fitted-distance distributions before the DNN cut',fontsize=14)
        outside=int((np.isfinite(distance)&(distance>150)).sum());invalid=int((~np.isfinite(distance)).sum())
        fig.text(.03,.012,f'Distance = |fitted vertex - nominal production point|, in metres; all{len(score)} development candidates retained in bookkeeping.\nAbove150m:{outside}; invalid coordinates:{invalid}. Shapes use candidate counts, not cross-section weights. Genuine means common truth origin, not an accurate fitted position.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.95));pdf.savefig(fig);plt.close(fig)

    with PdfPages(plot_output/'signal_mass_lifetime.pdf')as pdf:
        fig,ax=plt.subplots(figsize=(13,8))
        supported=[];sparse=[]
        for point,entry in sorted(report['signal_points'].items()):
            s=entry['response']['genuine_pair_retention']['empirical'];n=s['total'];k=s['passed'];x=entry['mass_gev'];ct=entry['proper_ctau_mm']
            if n>=30:supported.append((x,ct,s['efficiency']))
            else:sparse.append((x,ct))
            ax.annotate(f'{k}/{n}',(x,ct),xytext=(4,4),textcoords='offset points',fontsize=8,color='black'if n>=30 else '#666666')
        if supported:
            x,ct,eff=map(np.asarray,zip(*supported));im=ax.scatter(x,ct,c=eff,cmap='viridis',vmin=.8,vmax=1,s=115,edgecolors='black',zorder=3);fig.colorbar(im,ax=ax,label='Known genuine signal retention')
        if sparse:
            x,ct=map(np.asarray,zip(*sparse));ax.scatter(x,ct,s=55,facecolors='none',edgecolors='#999999',zorder=2,label='Fewer than30 genuine pairs: unresolved')
        ax.set(xlabel='Generated mass (GeV)',ylabel='Audited mean proper cτ (mm)',yscale='log',xlim=(8,96))
        ax.set_title(title+' — 30 newly completed signal points\nOne unchanged validation cut; counts are passing/total known genuine pairs',fontsize=13)
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
        pair_stats=[entry['response']['genuine_pair_retention']['empirical']for entry in report['signal_points'].values()]
        signal_k=sum(s['passed']for s in pair_stats);signal_n=sum(s['total']for s in pair_stats)
        gen_n=sum(entry['response']['full_parent_selected']['empirical']['total']for entry in report['signal_points'].values())
        fig.text(.025,.015,'GEN mass/lifetime are plot labels only. They never enter inference, training or selection. Empty and sparse points are not evidence of uniformity.\n'
            f'Retained{signal_k}/{signal_n} known genuine pairs overall; all{gen_n} generated parents preserved separately. Real recording acceptance and data/MC corrections remain unresolved.',fontsize=8)
        fig.tight_layout(rect=(0,.075,1,1));pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,5.5))
        for i,(point,entry)in enumerate([(p,e)for p,e in sorted(report['signal_points'].items())if e['response']['genuine_pair_retention']['empirical']['total']>=30]):
            s=entry['response']['genuine_pair_retention']['empirical'];eff=s['efficiency'];ci=s['wilson95']
            if ci is not None:ax.errorbar(i,eff,yerr=[[max(0,eff-ci[0])],[max(0,ci[1]-eff)]],fmt='o',capsize=4,color='#235789')
            else:ax.scatter(i,eff,color='#235789')
            ax.text(i,.705,f'{s["passed"]}/{s["total"]}',ha='center',fontsize=9)
        rows=[(p,e)for p,e in sorted(report['signal_points'].items())if e['response']['genuine_pair_retention']['empirical']['total']>=30]
        ax.set_xticks(range(len(rows)),[f'{e["mass_gev"]:g}GeV\ncτ={e["proper_ctau_mm"]:.3g}mm'for _,e in rows])
        ax.set(ylim=(.7,1.02),ylabel='Known genuine signal pairs retained',title='Signal points with at least30 known genuine pairs')
        ax.axhline(.9,color='gray',linestyle='--');ax.grid(axis='y',alpha=.2)
        fig.text(.03,.015,'95% Wilson intervals are shown only when qualified pairs have distinct physical parents; otherwise omitted.\nThese points were not used to choose a model or cut. Higher masses and long lifetimes remain statistically sparse.',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1));pdf.savefig(fig);plt.close(fig)
    write(plot_output/'plot_numerics.json',dict(profiles=profiles,overall={name:stats(y==label)for label,name in [(1,'genuine'),(0,'fake')]},event_bootstrap_replicas=300,seed=141071,
        distance_definition='Euclidean3D fitted-vertex separation from nominal target, cm converted to m; no momentum/boost factor',
        nominal_target_cm=manifest['nominal_target_cm'],distance_edges_m=distance_edges.tolist(),
        selection=reference(output/'selection_locked.json'),evaluation=reference(output/'evaluation.json')))
    write(plot_output/'completion.json',dict(complete=True,source=reference(Path(__file__)),
        coordinate_source=reference(Path(sys.modules['shift_vertex_reco_lifetime'].__file__)),
        artifacts={p.name:reference(p)for p in sorted(plot_output.iterdir())},only_selected_model=True))
    print(json.dumps(dict(final_pdfs=3,directory=str(plot_output))),flush=True)


def selected_signal_yield(response, cross_section_pb, luminosity_fb):
    """Count selected physical parents against the full native GEN weight sum."""
    weighted = response['weighted']
    numerator, denominator = weighted['numerator_sumw'], weighted['denominator_sumw']
    require(all(math.isfinite(v) for v in (numerator, denominator, cross_section_pb, luminosity_fb)),
            'Nonfinite signal normalization')
    require(denominator > 0 and 0 <= numerator <= denominator and
            cross_section_pb > 0 and luminosity_fb > 0, 'Invalid signal normalization')
    efficiency = numerator / denominator
    require(math.isclose(efficiency, weighted['efficiency'], rel_tol=1e-12, abs_tol=0),
            'Response efficiency differs from full GEN weight ratio')
    produced = luminosity_fb * 1000 * cross_section_pb
    error = weighted['standard_error']
    return dict(expected_events=produced * efficiency, produced_dimuons=produced,
                full_response_efficiency=efficiency,
                mc_standard_error_events=None if error is None else produced * error)


def render_yields(output, yield_inputs, plot_output):
    """Reuse the signal-yield plotter with the locked DNN's event selection."""
    from shift_baseline_sensitivity import draw
    inputs = read(yield_inputs)
    require(inputs['schema'] == 'shift-vertex-theory-yield-inputs-v1' and inputs['complete'],
            'Incomplete or unsupported theory-yield inputs')
    for ref in inputs['references'].values(): verify(ref)
    verify(inputs['independent_normalization_audit'])
    require(inputs['references']['evaluation'] == reference(output/'evaluation.json') and
            inputs['references']['selection'] == reference(output/'selection_locked.json'),
            'Yield inputs do not belong to this locked classifier')
    evaluation = read(output/'evaluation.json')
    require(inputs['fixed_classifier_cut'] == evaluation['threshold'], 'Changed classifier cut')
    require(inputs['branching_included_in_sigma'] and inputs['epsilon_correction_already_in_sigma'] and
            not inputs['extra_retention_factor_applied'] and not inputs['extra_filter_factor_applied'],
            'Signal corrections must be applied exactly once')
    require(set(inputs['points']) == set(evaluation['signal_points']), 'Signal point coverage changed')
    luminosity = inputs['luminosity_fb_inverse']
    require(math.isclose(1000*luminosity, inputs['luminosity_pb_inverse']), 'Luminosity unit mismatch')
    points = []
    for name, point in sorted(inputs['points'].items(), key=lambda item:(item[1]['mass_gev'],item[1]['epsilon'])):
        for key in ('rate_reference', 'contract_reference'): verify(point[key])
        rate = read(point['rate_reference']['path'])
        corrections = rate['corrections']
        require(corrections['applications'] == 1 and
                not corrections['extra_decay_branching_fraction_applied'] and
                not corrections['extra_detector_or_generator_filter_efficiency_applied'],
                'Repeated or extra signal-rate corrections')
        sigmas = rate['physical_selected_cross_sections_pb']
        require(len(sigmas) == 1, 'Ambiguous physical signal cross section')
        sigma = sigmas[0]['value_pb']
        for key in ('mass_gev', 'epsilon', 'proper_ctau_mm'):
            require(math.isclose(point[key], rate[key], rel_tol=1e-10, abs_tol=0), 'Rate/point mismatch: '+key)
        require(math.isclose(sigma, point['sigma_mumu_pb'], rel_tol=1e-12, abs_tol=0), 'Changed physical dimuon rate')
        response = evaluation['signal_points'][name]['response']['full_parent_selected']
        empirical, weighted = response['empirical'], response['weighted']
        full_gen = rate['full_gen']
        require(full_gen['accepted_events'] == empirical['total'] == point['full_GEN_events'] and
                full_gen['sum_weights'] == weighted['denominator_sumw'] and
                full_gen['sum_weights_squared'] == weighted['denominator_sumw2'], 'Incomplete GEN denominator')
        # Present samples have unit GEN weights. Only then may Wilson count
        # intervals be scaled to yields without a separate weighted interval.
        require(weighted['denominator_sumw'] == weighted['denominator_sumw2'] == empirical['total'] and
                weighted['numerator_sumw'] == empirical['passed'], 'Nonunit GEN weights need a weighted interval')
        prediction = selected_signal_yield(response, sigma, luminosity)
        require(empirical['passed'] == point['selected_qualified_signal_parents'] and
                math.isclose(prediction['expected_events'], point['expected_selected_events_at_3fb'],
                             rel_tol=1e-12, abs_tol=0), 'Yield differs from selected parent response')
        prediction.update(point=name, mass_gev=point['mass_gev'], epsilon=point['epsilon'],
            proper_ctau_mm=point['proper_ctau_mm'], sigma_mumu_pb=sigma,
            sigma_mumu_stat_error_pb=sigmas[0]['error_pb'], selected_GEN_parents=empirical['passed'],
            full_GEN_parents=empirical['total'],
            mc_interval95_events=[max(0., prediction['produced_dimuons']*bound) for bound in empirical['wilson95']],
            zero_selected_one_sided95_upper_events=None)
        if empirical['passed'] == 0:
            upper = -math.expm1(math.log(.05)/empirical['total'])
            prediction['zero_selected_one_sided95_upper_events'] = prediction['produced_dimuons']*upper
            prediction['mc_interval95_events'][0] = 0.
        prediction['mc_low95_events'], prediction['mc_high95_events'] = prediction['mc_interval95_events']
        points.append(prediction)
    require(not plot_output.exists(), 'Refusing yield evidence overwrite')
    plot_output.mkdir(parents=True)
    report = dict(schema='shift-vertex-selected-signal-yields-v1', complete=True,
        luminosity_fb_inverse=luminosity, fixed_classifier_cut=evaluation['threshold'], points=points,
        formula='L_fb * 1000 * physical_sigma_mumu_pb * selected_unique_parent_sumw / full_GEN_sumw',
        selection_definition='Truth-matched reconstructed signal event with fixed DNN cut; no reconstructed mass window',
        model=inputs['model'], theory_status=inputs['theory_status'], collision_model=inputs['collision_model'],
        normalization_scope=inputs['normalization_scope'], background_neglected=True, exclusion_evaluated=False,
        recording_acceptance_included=False, model_choice_and_cut_unchanged=True,
        uncertainty_scope=inputs['uncertainty_scope'], input=reference(yield_inputs),
        selection=reference(output/'selection_locked.json'), evaluation=reference(output/'evaluation.json'),
        independent_normalization_audit=inputs['independent_normalization_audit'])
    plot_points = [dict(name=p['point'], mass_gev=p['mass_gev'], epsilon=p['epsilon'], partial=False,
        signal=dict(events=p['expected_events'], selected_mc_events=p['selected_GEN_parents']),
        zero_selected_one_sided95_upper_events=p['zero_selected_one_sided95_upper_events']) for p in points]
    report['scenarios'] = [dict(luminosity_fb=luminosity, luminosity_role='Assumed exposure', points=plot_points)]
    report['presentation'] = dict(
        title='SHIFT signal events after reconstruction + selected DNN',
        selection_text=f'One truth-matched reconstructed signal event; fixed DNN score >= {evaluation["threshold"]:.4f}; no mass window.',
        description='3 fb^-1 times physical dimuon cross section times full reconstruction + DNN event efficiency; backgrounds neglected.',
        footer='Pure-DY/no-pileup scenario; rates provisional, real recording acceptance unresolved. Points only; no exclusion contour.',
        annotate_yields=True, show_mc_support=True, luminosity_symbol='L',
        metadata_title='SHIFT selected DNN signal yields at 3 inverse fb')
    draw(report, plot_output/'signal_mass_lifetime.pdf', signal_yields=True)
    write(plot_output/'signal_yields.json', report)
    columns = ['point', 'mass_gev', 'epsilon', 'proper_ctau_mm', 'sigma_mumu_pb', 'sigma_mumu_stat_error_pb',
               'selected_GEN_parents', 'full_GEN_parents', 'full_response_efficiency', 'expected_events',
               'mc_standard_error_events', 'mc_low95_events', 'mc_high95_events',
               'zero_selected_one_sided95_upper_events']
    with (plot_output/'signal_yields.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(points)
    write(plot_output/'completion.json', dict(complete=True, source=reference(Path(__file__)),
        plotter_source=reference(Path(sys.modules['shift_baseline_sensitivity'].__file__)),
        artifacts={p.name:reference(p) for p in sorted(plot_output.iterdir())}, only_selected_model=True,
        input=reference(yield_inputs), PDF_only=True))
    print(json.dumps(dict(final_pdfs=1, directory=str(plot_output), points=len(points))), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('select','evaluate','signal','plots','yields'),required=True)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--signal-cache',type=Path)
    parser.add_argument('--choice-output',type=Path)
    parser.add_argument('--plot-output',type=Path)
    parser.add_argument('--yield-inputs',type=Path)
    args=parser.parse_args()
    if args.action=='select':select(args.manifest.resolve(),args.output.resolve())
    elif args.action=='evaluate':evaluate(args.output.resolve(),args.signal_cache.resolve())
    elif args.action=='signal':evaluate_signal(args.choice_output.resolve(),args.signal_cache.resolve(),args.output.resolve())
    elif args.action=='plots':render(args.output.resolve(),args.plot_output.resolve())
    else:render_yields(args.output.resolve(),args.yield_inputs.resolve(),args.plot_output.resolve())


if __name__=='__main__':main()
