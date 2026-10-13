#!/usr/bin/env python3
# Code style: AI-code. AI-generated frozen-score DNN comparison diagnostics.
"""PDF-only development diagnostics; models and cuts come from validation locks."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shift_bdt_category_diagnostics import validate_identity
from shift_bdt_candidate_diagnostics import threshold_counts
from shift_bdt_discrimination_diagnostics import clean, digest, load_npz, read, reference, require


PROCESSES = (('sm', 'Pooled SM'), ('dy', 'DY'), ('jpsi', 'J/psi'), ('qcd', 'QCD'))
REFERENCES = ('reference_uniform_bdt_3p0_s71', 'expanded_d3_s0_n400')


def checked(ref):
    path = Path(ref['path'])
    require(digest(path) == ref['sha256'], 'Changed pinned artifact: ' + str(path))
    return path


def selected_models(lock):
    require(lock.get('selection_uses_test') is False, 'Test-based model selection is forbidden')
    chosen = [lock[key] for key in ('rejection_winner', 'flatness_winner', 'acceptance_envelope_winner')
              if lock.get(key) is not None]
    require(bool(chosen), 'No validation-selected neural models')
    return list(dict.fromkeys([*REFERENCES, *chosen]))


def frozen_statistics(path):
    """Reuse exact pure functions from the pinned source without importing ROOT/NN."""
    names = {'wilson', '_group_ids', 'efficiency_summary', 'weighted_efficiency', 'roc_summary'}
    tree = ast.parse(Path(path).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names, 'Frozen statistical helper contract changed')
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_inputs(inputs, training):
    completion = read(training / 'completion.json')
    require(completion.get('complete') is True, 'Training study is incomplete')
    require(checked(completion['diagnostic_inputs']).resolve() == inputs.resolve(), 'Wrong diagnostic input manifest')
    protocol = read(checked(completion['protocol']))
    lock = read(checked(completion['selection']))
    require(checked(lock['protocol']) == checked(completion['protocol']), 'Selection protocol differs')
    validation = read(checked(lock['validation']))
    refs = read(inputs)
    for ref in refs.values():
        checked(ref)
    for name in ('cache_manifest', 'reco', 'labels'):
        require(refs[name] == protocol[name], 'Diagnostic input differs from protocol: ' + name)
    require(refs['scores'] == completion['scores'], 'Score reference differs from completion')
    diagnostic_lock = read(checked(refs['selection']))
    for key, value in lock.items():
        if key != 'thresholds':
            require(diagnostic_lock.get(key) == value, 'Diagnostic selection changed: ' + key)
    for name, cut in lock['thresholds'].items():
        require(diagnostic_lock['thresholds'].get(name) == cut, 'Neural threshold changed')
    for name in ('splits', 'bdt_scores'):
        checked(protocol[name])
    for name, ref in protocol['sources'].items():
        require(digest(training / 'source_snapshot' / name) == ref['sha256'], 'Changed frozen source: ' + name)
    manifest = read(checked(refs['cache_manifest']))
    require(manifest.get('complete') and manifest.get('sample_kind') == 'simulation_only', 'Invalid simulation cache')
    for name in ('reco', 'labels'):
        require(refs[name]['sha256'] == manifest[name + '_sha256'], 'Cache manifest hash disagreement')
    reco, labels, scores = [load_npz(refs[name]['path']) for name in ('reco', 'labels', 'scores')]
    splits = load_npz(protocol['splits']['path'])
    require(reco['ids'].dtype == np.uint64, 'Event identities must remain exact uint64')
    for key in ('ids', 'process', 'pair_index', 'source_index', 'expanded_split', 'expanded_heldout'):
        require(np.array_equal(splits[key], scores[key]), 'Frozen split identity mismatch: ' + key)
    test = validate_identity(reco, labels, scores)
    require(np.isin(scores['expanded_split'], [0, 1, 2]).all(), 'Invalid partition')
    earlier = load_npz(protocol['bdt_scores']['path'])
    earlier_root = Path(protocol['bdt_scores']['path']).parent
    earlier_completion = read(earlier_root / 'completion.json')
    require(earlier_completion['development']['scores'] == protocol['bdt_scores'], 'Original BDT score receipt differs')
    earlier_lock = read(checked(earlier_completion['selection']))
    models, histories = selected_models(lock), {}
    for name in models:
        require(name in scores and name in diagnostic_lock['thresholds'], 'Missing locked model: ' + name)
        require(scores[name].shape == labels['y'].shape and np.isfinite(scores[name]).all(), 'Invalid model scores')
        require(((scores[name] >= 0) & (scores[name] <= 1)).all(), 'Scores outside probability range')
        if name in REFERENCES:
            require(np.array_equal(scores[name], earlier[name]), 'Frozen BDT scores changed')
            require(diagnostic_lock['thresholds'][name] == earlier_lock['thresholds'][name], 'Original BDT threshold changed')
            continue
        result = validation[name]
        require(result['validation']['threshold'] == lock['thresholds'][name], 'Validation cut differs')
        checked(result['model'])
        history = read(checked(result['history']))
        require(history['restored_prediction_exact'] and history['best_state_sha256'] == history['restored_state_sha256'],
                'Best checkpoint was not restored')
        require(history['best_epoch'] == 1 + int(np.argmin([r['val_objective'] for r in history['epochs']])),
                'Selected epoch differs from minimum validation objective')
        histories[name] = history
    return protocol, diagnostic_lock, reco, labels, scores, test, models, histories


def population(reco, test, process):
    return test & ((reco['process'] != 'signal') if process == 'sm' else (reco['process'] == process))


def groups(reco, mask):
    return [(str(p), *map(int, identity)) for p, identity in zip(reco['process'][mask], reco['ids'][mask])]


def acceptance(perf, passed, identities, weights, repeats=200):
    empirical = perf.efficiency_summary(passed, identities, repeats=repeats)
    weighted = perf.weighted_efficiency(passed, identities, weights, repeats=repeats)
    empirical['supported'] = empirical['pairs'] >= 20 and empirical['events'] >= 20
    weighted['supported'] = (weighted['available'] and weighted['pairs'] >= 20 and weighted['events'] >= 20
                             and weighted.get('event_weight_effective_n', 0) >= 20)
    return {'empirical': empirical, 'weighted': weighted}


def build(workspace, inputs, training):
    protocol, lock, reco, labels, scores, test, models, histories = load_inputs(inputs, training)
    perf = frozen_statistics(training / 'source_snapshot/bdt_performance.py')
    y = labels['y']
    split = scores['expanded_split']
    train = (split == 0) & (reco['process'] != 'signal') & (y == 1)
    dimensions = [('mass', 'Reconstructed mass [GeV]', reco['nuisance'][:, 0]),
                  ('z', 'Fitted z [m]', reco['nuisance'][:, 1] * .01),
                  ('radius', 'Fitted radius [m]', reco['nuisance'][:, 2] * .01)]
    names = list(reco['feature_names'])
    for side in ('lowPt', 'highPt'):
        dimensions.append((side + '_energy', side + ' muon energy [GeV]', reco['X'][:, names.index('mu_' + side + '_energy')]))
    edges = {name: np.unique(np.quantile(values[train & np.isfinite(values)], [1/3, 2/3]))
             for name, _, values in dimensions}
    report = {'schema': 'shift-vertex-dnn-diagnostics-v1', 'complete': True, 'models': models,
              'interpretation': 'Previously inspected expanded development test only. No test tuning. Common origin does not certify fitted-vertex accuracy.',
              'weight_definition': 'Exposure evaluation weights from frozen extraction; signed weights never converted into probabilities.',
              'uncertainty': '95% event bootstrap, 200 replicates; support requires20 pairs/events and weighted effective events. Open markers are unsupported.',
              'binning': 'Tertiles of genuine reconstructed expanded-training pairs only; inclusive overflow bins.',
              'bin_edges': edges, 'counts': {}, 'threshold_counts': {}, 'results': {},
              'unknown_policy': 'Excluded from ROC and known-class acceptance; separate counts and retained fractions only.',
              'selection': {key: lock[key] for key in ('rejection_winner', 'flatness_winner', 'acceptance_envelope_winner')},
              'history_summary': {}, 'references': [reference(inputs), reference(training / 'completion.json'),
                  reference(training / 'selection_locked.json'), reference(Path(__file__)), *read(inputs).values(),
                  protocol['splits'], reference(training / 'source_snapshot/bdt_performance.py')]}
    report['references'] += [reference(Path(__file__).with_name(name)) for name in
        ('shift_bdt_category_diagnostics.py', 'shift_bdt_candidate_diagnostics.py', 'shift_bdt_discrimination_diagnostics.py')]
    earlier_root = Path(protocol['bdt_scores']['path']).parent
    report['references'] += [reference(earlier_root / 'completion.json'), reference(earlier_root / 'selection_locked.json')]
    for process, _ in PROCESSES:
        mask = population(reco, test, process)
        report['counts'][process] = {kind: int((mask & (y == label)).sum()) for label, kind in [(1, 'genuine'), (0, 'fake'), (-1, 'unknown')]}
    for name in models:
        score, cut = scores[name], lock['thresholds'][name]
        report['threshold_counts'][name] = threshold_counts(score, np.where(reco['process'] == 'signal', -1, y), split, cut)
        report['results'][name] = {}
        for process, _ in PROCESSES:
            pop = population(reco, test, process)
            known = pop & (y >= 0)
            row = {'roc': {}, 'classes': {}, 'profiles': {}}
            for kind in ('empirical', 'weighted'):
                row['roc'][kind] = perf.roc_summary(y[known], score[known], groups(reco, known), repeats=0,
                    weights=reco['evaluation_weight'][known] if kind == 'weighted' else None)
            for label, kind in [(1, 'genuine'), (0, 'fake'), (-1, 'unknown')]:
                mask = pop & (y == label)
                passed = score[mask] < cut if label == 0 else score[mask] >= cut
                row['classes'][kind] = acceptance(perf, passed, groups(reco, mask), reco['evaluation_weight'][mask])
            for dimension, _, values in dimensions:
                bins = np.searchsorted(edges[dimension], values, side='right')
                row['profiles'][dimension] = {}
                for label, kind in [(1, 'genuine'), (0, 'fake')]:
                    entries = []
                    for b in range(len(edges[dimension]) + 1):
                        mask = pop & (y == label) & np.isfinite(values) & (bins == b)
                        passed = score[mask] < cut if label == 0 else score[mask] >= cut
                        entries.append(acceptance(perf, passed, groups(reco, mask), reco['evaluation_weight'][mask]))
                    row['profiles'][dimension][kind] = entries
                row['profiles'][dimension]['nonfinite_pairs'] = int((pop & ~np.isfinite(values)).sum())
            report['results'][name][process] = row
        if name in histories:
            history = histories[name]
            report['history_summary'][name] = {key: history[key] for key in ('best_epoch', 'epochs_completed', 'best_val_objective',
                'best_state_sha256', 'last_state_sha256', 'restored_state_sha256', 'restored_prediction_exact')}
    return report, reco, labels, scores, test, dimensions, histories


def render(report, reco, labels, scores, test, dimensions, histories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = report['models']
    palette = dict(zip(models, plt.get_cmap('tab10').colors))
    titles = {REFERENCES[0]: 'Current uniform BDT', REFERENCES[1]: 'Expanded strong BDT'}
    for name in models[2:]:
        roles = [key.replace('_winner', '').replace('_', ' ') for key, value in report['selection'].items() if value == name]
        titles[name] = 'DNN: ' + ', '.join(roles)
    report['model_titles'] = titles
    paths = []

    def save(fig, name):
        path = output / (name + '.pdf')
        require(not path.exists(), 'Refusing PDF overwrite')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        paths.append(reference(path))

    fig, axes = plt.subplots(len(models), 4, figsize=(17, 3 * len(models)), squeeze=False, layout='constrained')
    for i, name in enumerate(models):
        for ax, (process, title) in zip(axes[i], PROCESSES):
            pop = population(reco, test, process)
            for label, kind, color in [(1, 'Genuine', '#0072b2'), (0, 'Fake', '#d55e00'), (-1, 'Unknown', '#777777')]:
                values = scores[name][pop & (labels['y'] == label)]
                if len(values):
                    ax.hist(values, bins=np.linspace(0, 1, 51), weights=np.full(len(values), 1/len(values)),
                            histtype='step', color=color, label=f'{kind}: {len(values)}', linestyle=':' if label == -1 else '-')
            ax.axvline(report['threshold_counts'][name]['threshold'], color='black', linestyle=':', label='Locked cut')
            if not report['counts'][process]['genuine']:
                ax.text(.03, .65, 'No genuine pairs;\nno ROC/retention estimate', transform=ax.transAxes, fontsize=8)
            ax.set(title=titles[name] + '\n' + title, xlabel='Unchanged classifier score', ylabel='Fraction within label class', xlim=(0, 1))
            ax.legend(fontsize=7)
    fig.suptitle('Expanded development test; genuine origin, accidental origin and unknown labels\nModels and cuts selected using validation only')
    save(fig, 'truth_class_scores')

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
    for i, kind in enumerate(('empirical', 'weighted')):
        for ax, (process, title) in zip(axes[i], PROCESSES):
            for name in models:
                row = report['results'][name][process]
                roc = row['roc'][kind]
                if process == 'qcd' and not report['counts'][process]['genuine']:
                    mask = population(reco, test, process) & (labels['y'] == 0)
                    weights = np.ones(mask.sum()) if kind == 'empirical' else reco['evaluation_weight'][mask]
                    if len(weights) and (weights >= 0).all() and weights.sum() > 0:
                        cuts = np.linspace(0, 1, 201)
                        values = [float(weights[scores[name][mask] < cut].sum()/weights.sum()) for cut in cuts]
                        supported = row['classes']['fake'][kind]['supported']
                        ax.plot(cuts, values, color=palette[name], linestyle='-' if supported else ':', label=titles[name])
                        ax.scatter(report['threshold_counts'][name]['threshold'], row['classes']['fake'][kind]['efficiency'],
                                   color=palette[name], s=18)
                elif roc['supported']:
                    ax.plot(roc['fpr'], roc['tpr'], color=palette[name], label=f'{titles[name]} ({roc["auc"]:.3f})')
                    ax.scatter(1-row['classes']['fake'][kind]['efficiency'], row['classes']['genuine'][kind]['efficiency'],
                               color=palette[name], s=18)
            qcd_only = process == 'qcd' and not report['counts'][process]['genuine']
            ax.set(title=title + ' — ' + kind, xlabel='Diagnostic score threshold' if qcd_only else 'Fake pairs accepted',
                   ylabel='Fake pairs rejected' if qcd_only else 'Genuine pairs retained', xlim=(0, 1), ylim=(0, 1.02))
            if qcd_only:
                ax.text(.03, .97, 'Fake-only curve; no QCD ROC\nDotted = unsupported effective events', transform=ax.transAxes, va='top', fontsize=8)
            elif not any(report['results'][name][process]['roc'][kind]['supported'] for name in models):
                ax.text(.5, .5, 'ROC omitted:\ninsufficient effective-event support', transform=ax.transAxes, ha='center')
            if ax.lines:
                ax.legend(fontsize=6, loc='lower right')
            ax.grid(alpha=.2)
    fig.suptitle('Expanded development test; dots mark locked validation cuts\nWeighted curves use exposure weights; unsupported ROC populations omitted')
    save(fig, 'truth_rocs_and_qcd_rejection')

    for kind in ('empirical', 'weighted'):
        for label in ('genuine', 'fake'):
            fig, axes = plt.subplots(4, len(dimensions), figsize=(20, 12), layout='constrained')
            for row, (process, title) in enumerate(PROCESSES):
                for ax, (dimension, axis_title, _) in zip(axes[row], dimensions):
                    edges = np.asarray(report['bin_edges'][dimension])
                    bounds = np.r_[-np.inf, edges, np.inf]
                    ticklabels = [f'{left:.3g} to {right:.3g}' for left, right in zip(bounds[:-1], bounds[1:])]
                    for i, name in enumerate(models):
                        entries = report['results'][name][process]['profiles'][dimension][label]
                        for b, entry in enumerate(entries):
                            item = entry[kind]
                            value = item['efficiency']
                            if value is None:
                                continue
                            low, high = item['ci95_event_bootstrap']
                            error = None if low is None else [[max(0, value-low)], [max(0, high-value)]]
                            ax.errorbar(b + (i-(len(models)-1)/2)*.13, value, yerr=error, fmt='o', markersize=4,
                                        markerfacecolor=palette[name] if item['supported'] else 'none', color=palette[name], capsize=2,
                                        label=titles[name] if b == 0 else None)
                    base = report['results'][models[0]][process]['profiles'][dimension][label]
                    counts = ', '.join(str(entry['empirical']['pairs']) for entry in base)
                    support = ', '.join(f'{entry["weighted"].get("event_weight_effective_n", 0):.1f}' for entry in base)
                    ax.text(.02, .04, 'n: ' + counts + ('\nEffective events: ' + support if kind == 'weighted' else ''), transform=ax.transAxes, fontsize=7)
                    ax.set_xticks(range(len(ticklabels)), ticklabels, fontsize=7)
                    ax.set(title=title + ': ' + axis_title, ylabel='Genuine retained' if label == 'genuine' else 'Fake rejected', ylim=(0, 1.03))
                    ax.grid(axis='y', alpha=.2)
            axes[0, 0].legend(fontsize=6, loc='lower left', bbox_to_anchor=(0, .12))
            fig.suptitle(f'{kind.capitalize()} {label} acceptance at frozen validation cuts; expanded development test\nTraining-defined bins; 95% event bootstrap; open markers are unsupported; empty bins are unmeasured')
            save(fig, kind + '_' + label + '_profiles')

    fig, axes = plt.subplots(len(histories), 3, figsize=(15, 3.5 * len(histories)), squeeze=False, layout='constrained')
    for row, (name, history) in enumerate(histories.items()):
        records = history['epochs']
        epoch = [r['epoch'] for r in records]
        for ax, metric in zip(axes[row], ('bce', 'auc', 'objective')):
            for split, color in [('train', '#0072b2'), ('val', '#d55e00')]:
                key = split + '_' + metric
                if key in records[0]:
                    ax.plot(epoch, [r[key] for r in records], color=color, label=split)
            ax.axvline(history['best_epoch'], color='black', linestyle='--', label=f'Restored best: {history["best_epoch"]}')
            ax.axvline(history['epochs_completed'], color='grey', linestyle=':', label=f'Last: {history["epochs_completed"]}')
            ax.set(title=titles[name] + '\n' + metric, xlabel='Epoch', ylabel='Weighted training/validation ' + metric)
            ax.legend(fontsize=7)
    fig.suptitle('Checkpoint diagnostics use training and validation only\nObjectives differ across penalty recipes; their absolute values are not a model ranking')
    save(fig, 'best_and_last_learning_curves')
    report['plots'] = paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'inputs', 'training', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    require(output.is_relative_to(workspace / 'tea_shift_cmssw/plots'), 'Final diagnostics must be inside project plots')
    require(not output.exists(), 'Refusing to overwrite an existing diagnostic directory')
    result = build(workspace, args.inputs.resolve(), args.training.resolve())
    output.mkdir(parents=True)
    render(*result, output)
    (output / 'diagnostics.json').write_text(json.dumps(clean(result[0]), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'output': str(output), 'pdfs': len(result[0]['plots'])}))


if __name__ == '__main__':
    main()
