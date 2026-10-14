#!/usr/bin/env python3
# Code style: AI-code. AI-generated locked-development classifier diagnostics.
"""Render completed, locked full-information studies without loading models.

The completion receipt is mandatory. All score, metric, protocol, selection
and fit JSON hashes are checked before a score table is opened. No thresholds
or families are selected here; every displayed choice comes from the lock.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from shift_vertex_complete_diagnostics import reference, read_reference, write


SEEDS = (71, 72, 73)
FRACTIONS = (1., .5, .25)
PROCESSES = ('pooled', 'dy', 'jpsi', 'qcd')
PROCESS_LABELS = {'pooled': 'Pooled SM', 'dy': 'DY', 'jpsi': 'J/psi', 'qcd': 'QCD'}
AXES = ('mass', 'vertex_z', 'vertex_radius')
CATEGORIES = ('Both-Both', 'Near-Both', 'Both-Far', 'Near-Far', 'Other')
TABLE_METADATA = {'ids', 'process', 'category', 'nuisance', 'y', 'evaluation_weight'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family_label(spec):
    if spec['algorithm'].lower() == 'bdt':
        return {'shallow_d3': 'BDT depth3', 'medium_d5': 'BDT depth5',
                'regularized_d5': 'BDT regularized depth5'}.get(spec.get('config'), spec['family'])
    architecture = 'Category heads' if spec.get('architecture') == 'heads' else 'Shared DNN'
    width = '/'.join(map(str, spec.get('hidden', ())))
    loss = 'classification only' if spec.get('loss') == 'classification' else 'decorrelated'
    return f'{architecture} {width}, {loss}'


def population(table, process):
    return np.ones(len(table['y']), dtype=bool) if process == 'pooled' else table['process'] == process


def empirical(mask, passed, minimum=20):
    n = int(mask.sum())
    k = int(np.count_nonzero(mask & passed))
    return dict(n=n, passed=k, efficiency=k / n if n else None, supported=n >= minimum)


def check_fraction(saved, measured, context):
    require(saved['n'] == measured['n'] and saved['passed'] == measured['passed'],
            'Saved metric count differs: ' + context)
    if measured['n']:
        require(math.isclose(saved['efficiency'], measured['efficiency'], rel_tol=0., abs_tol=1e-12),
                'Saved metric efficiency differs: ' + context)
    else:
        require(saved['efficiency'] is None, 'Empty class must remain undefined: ' + context)


def validate_table(table, specs, metrics, cuts):
    """Reconcile every saved operating point, process/category and profile count."""
    expected = {spec['name'] for spec in specs}
    require(set(table) == TABLE_METADATA | expected, 'Score columns differ from locked selected recipes')
    require(set(metrics) == expected, 'Metric models differ from locked selected recipes')
    n = len(table['y'])
    require(table['ids'].shape == (n, 3) and table['ids'].dtype == np.uint64, 'Exact uint64 identities required')
    for key in ('process', 'category', 'y', 'evaluation_weight'):
        require(table[key].shape == (n,), 'Unaligned table metadata: ' + key)
    require(table['y'].dtype.kind in 'iu' and np.isin(table['y'], [0, 1]).all(), 'Known labels only in development table')
    require(set(table['process']).issubset(PROCESSES[1:]), 'Non-SM process in development table')
    require(set(table['category']).issubset(CATEGORIES), 'Unknown reconstructed category')
    require(table['nuisance'].shape == (n, 3) and np.isfinite(table['nuisance']).all(), 'Invalid reconstructed nuisance coordinates')
    require(np.isfinite(table['evaluation_weight']).all() and np.all(table['evaluation_weight'] >= 0), 'Invalid evaluation weights')
    require(set(table['y']) == {0, 1}, 'Pooled development needs both truth classes')
    reports, fixed_edges = {}, None
    for spec in specs:
        name = spec['name']
        score, saved = table[name], metrics[name]
        require(score.shape == (n,) and np.isfinite(score).all() and np.all((score >= 0) & (score <= 1)),
                'Invalid probability score: ' + name)
        cut = cuts[name]
        require(isinstance(cut, (int, float)) and math.isfinite(cut) and saved['threshold'] == cut,
                'Development threshold differs from validation lock: ' + name)
        passed = score >= cut
        checks = 0
        for label, key in ((1, 'common_vertex'), (0, 'accidental')):
            check_fraction(saved[key], empirical(table['y'] == label, passed), name + ':' + key)
            checks += 1
        auc = float(roc_auc_score(table['y'], score))
        require(abs(auc - saved['auc']) <= 1e-12, 'Saved AUC differs: ' + name)
        require(abs(saved['rejection'] - (1 - saved['accidental']['efficiency'])) <= 1e-12,
                'Saved rejection differs: ' + name)
        edges = {axis: saved['flatness'][axis]['interior_edges'] for axis in AXES}
        if fixed_edges is None:
            fixed_edges = edges
        require(edges == fixed_edges, 'Reporting bins changed across fits/fractions')
        for values in edges.values():
            require(np.isfinite(values).all() and np.all(np.diff(values) > 0), 'Invalid fixed reporting edges')
        groups = {}
        for process in PROCESSES:
            pmask = population(table, process)
            genuine = pmask & (table['y'] == 1)
            fake = pmask & (table['y'] == 0)
            genuine_result, fake_result = empirical(genuine, passed), empirical(fake, passed)
            if process != 'pooled' and np.any(pmask):
                for key, measured in (('common_vertex', genuine_result), ('accidental', fake_result)):
                    check_fraction(saved['per_process'][process][key], measured, name + ':' + process + ':' + key)
                    checks += 1
            profiles = {}
            for column, axis in enumerate(AXES):
                assignments = np.searchsorted(np.asarray(edges[axis]), table['nuisance'][:, column], side='right')
                bins = [empirical(genuine & (assignments == index), passed) for index in range(len(edges[axis]) + 1)]
                profiles[axis] = dict(interior_edges=edges[axis], bins=bins)
                original = saved['flatness'] if process == 'pooled' else saved['per_process_flatness'].get(process)
                if original is not None:
                    require(original[axis]['interior_edges'] == edges[axis], 'Per-process bin edges differ')
                    require(len(original[axis]['bins']) == len(bins), 'Saved profile bin count differs')
                    for before, measured in zip(original[axis]['bins'], bins):
                        check_fraction(before, measured, name + ':' + process + ':' + axis)
                        require(before['supported'] == measured['supported'], 'Saved profile support differs')
                        checks += 1
            categories = {category: empirical(genuine & (table['category'] == category), passed, minimum=50)
                          for category in CATEGORIES}
            if process == 'pooled':
                for category in set(table['category']):
                    check_fraction(saved['categories'][category]['common_vertex'], categories[category], name + ':' + category)
                    checks += 1
            groups[process] = dict(genuine=genuine_result, accidental=fake_result, profiles=profiles,
                                   categories=categories, roc_defined=bool(genuine.any() and fake.any()))
        reports[name] = dict(recipe=spec, threshold=cut, auc=auc, rejection=saved['rejection'],
                             populations=groups, reconciled_count_checks=checks)
    return reports, fixed_edges


def load_completed(training):
    """Do not open any score array until completion and all locked inputs pass."""
    completion_ref = reference(training / 'completion.json')
    completion = read_reference(completion_ref)
    require(completion.get('complete') is True and completion.get('fits') == 81, 'Evaluation is incomplete')
    expected_paths = {'protocol': 'protocol.json', 'selection': 'selection_locked.json',
                      'scores': 'development_scores.npz', 'metrics': 'development_metrics.json'}
    for key, filename in expected_paths.items():
        require(completion[key] == reference(training / filename), 'Completion input changed: ' + key)
    protocol = read_reference(completion['protocol'])
    lock = read_reference(completion['selection'])
    metric_report = read_reference(completion['metrics'])
    require(lock['protocol'] == completion['protocol'] and lock.get('selection_uses_test') is False,
            'Selection was not locked independently of development')
    require(metric_report['selection'] == completion['selection'] and metric_report.get('selection_uses_test') is False,
            'Metrics have another selection authority')
    domain = protocol['domain']
    require(domain in ('current_CMSIR5', 'historical_proxy') and
            completion['domain'] == lock['domain'] == metric_report['domain'] == domain, 'Study domains differ')
    recipes = {spec['name']: spec for spec in protocol['recipes']}
    require(len(recipes) == len(protocol['recipes']) == 81 and set(lock['validation_fits']) == set(recipes),
            'All81 fits must be locked before development')
    families = {spec['family'] for spec in recipes.values()}
    require(set(lock['winners']) == {'bdt', 'dnn'}, 'Both algorithm selections are required')
    chosen = set()
    for algorithm, winners in lock['winners'].items():
        require(set(winners) == {'rejection', 'acceptance'} and winners['rejection'] in families,
                'Incomplete locked family choices')
        for family in winners.values():
            if family is not None:
                require(family in families and all(spec['algorithm'].lower() == algorithm
                        for spec in recipes.values() if spec['family'] == family), 'Selected family belongs to another algorithm')
                chosen.add(family)
    specs = [spec for spec in recipes.values() if spec['family'] in chosen]
    for family in chosen:
        members = [spec for spec in specs if spec['family'] == family]
        require(len(members) == 9 and {(spec['seed'], spec['fraction']) for spec in members}
                == {(seed, size) for seed in SEEDS for size in FRACTIONS}, 'Selected family grid is incomplete')
    fits = {}
    for name, spec in recipes.items():
        ref = lock['validation_fits'][name]
        require(ref == reference(training / 'fits' / (name + '.json')), 'Locked fit path/hash differs')
        fit = read_reference(ref)
        require(fit['recipe'] == spec and fit['protocol'] == completion['protocol'], 'Fit belongs to another recipe/domain')
        require(fit['validation']['threshold'] == lock['thresholds'][name], 'Locked threshold differs from validation')
        if spec['family'] in chosen:
            fits[name] = fit
    with np.load(completion['scores']['path'], allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    models, edges = validate_table(table, specs, metric_report['metrics'], lock['thresholds'])
    counts = {process: {label: int(np.count_nonzero(population(table, process) & (table['y'] == value)))
                        for label, value in (('genuine', 1), ('accidental', 0))} for process in PROCESSES}
    report = dict(schema='shift-complete-locked-development-diagnostics-v1', domain=domain,
        completion=completion_ref, protocol=completion['protocol'], selection=completion['selection'],
        scores=completion['scores'], metrics=completion['metrics'], winners=lock['winners'],
        models=models, class_counts=counts, reporting_edges=edges,
        training_counts={name: fit['training_counts'] for name, fit in fits.items()},
        no_model_payload_read=True, no_new_model_selection=True, thresholds_unchanged=True,
        curve_normalization='Empirical candidate counts; each score histogram normalized separately within its process and truth class. Pooled SM uses observed counts, not equal-process or cross-section weights.',
        limitations=[
            'Current-domain development partition was previously inspected; this is not independent confirmation.',
            'Historical proxy is a separate detector/production domain with strong process/class confounding; no cross-domain pooling or CMS-acceptance claim.',
            'All81 validation fits and family choices precede evaluation; the selected100percent family remains fixed along each size curve.',
            'No development cut adjustment or score remapping. Cuts are the individual validation90percent genuine cuts.',
            'Seed variation is not a confidence interval. Full-size BDT fits can be identical across seeds; zero seed spread is not zero statistical uncertainty.',
            'Candidate trials may share physical parents; this saved table lacks historical physical-parent context. No independent-event confidence interval is claimed.',
            'Support labels use20 genuine candidates per nuisance bin and50 per category, not independent-event counts.',
            'Genuine labels assess common muon origin, not fitted vertex accuracy. Reconstructed mass/z/r are not true proper lifetime.',
            'Unknown labels and signal are excluded from these development tables; missing genuine classes have undefined ROC and retention.'])
    return report, table


def render(report, table, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    models = report['models']
    families = sorted({row['recipe']['family'] for row in models.values()})
    colors = {family: plt.get_cmap('tab10')(index) for index, family in enumerate(families)}
    styles = {71: '-', 72: '--', 73: ':'}
    labels = {family: family_label(next(row['recipe'] for row in models.values() if row['recipe']['family'] == family))
              for family in families}
    domain = 'Current CMS IR5 — reused development' if report['domain'] == 'current_CMSIR5' else 'Historical proxy — separate stress domain'
    footer = 'Empirical candidate counts; pooled SM keeps the observed process mixture. Seeds71/72/73: solid/dashed/dotted; spread is not a CI.'
    if report['domain'] == 'historical_proxy':
        footer += '\nStrong process/class confounding; this proxy domain does not establish current CMS performance.'
    else:
        footer += '\nPreviously inspected development. No new model choices or cuts; absent classes remain undefined.'
    absent = [algorithm.upper() for algorithm, choices in report['winners'].items() if choices['acceptance'] is None]
    if absent:
        footer += '\nNo acceptance-qualified family: ' + ', '.join(absent) + '. Zero full-size BDT seed spread does not imply zero statistical uncertainty.'
    else:
        footer += '\nZero full-size BDT seed spread does not imply zero statistical uncertainty.'

    def entries(size, family=None):
        return sorted(((name, row) for name, row in models.items()
                       if row['recipe']['fraction'] == size and (family is None or row['recipe']['family'] == family)),
                      key=lambda item: (item[1]['recipe']['family'], item[1]['recipe']['seed']))

    def finish(fig, pdf):
        fig.text(.025, .018, footer, fontsize=8)
        fig.tight_layout(rect=(0, .075, 1, .94))
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(output / 'development_roc.pdf') as pdf:
        for size in FRACTIONS:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for ax, process in zip(axes.flat, PROCESSES):
                mask = population(table, process)
                counts = report['class_counts'][process]
                ax.set_title(f"{PROCESS_LABELS[process]}: {counts['genuine']} genuine, {counts['accidental']} accidental", fontsize=10)
                if not all(counts.values()):
                    ax.text(.5, .5, 'ROC undefined\nBoth truth classes are required', ha='center', va='center', transform=ax.transAxes)
                else:
                    for name, row in entries(size):
                        family, seed = row['recipe']['family'], row['recipe']['seed']
                        false_positive, true_positive, _ = roc_curve(table['y'][mask], table[name][mask])
                        ax.plot(true_positive, 1-false_positive, color=colors[family], linestyle=styles[seed],
                                linewidth=1.4 if size == 1. else 1., label=labels[family] if seed == 71 else None)
                        point = row['populations'][process]
                        ax.scatter(point['genuine']['efficiency'], 1-point['accidental']['efficiency'], color=colors[family], s=14)
                    ax.legend(fontsize=7, loc='lower left')
                    if min(counts.values()) < 20:
                        ax.text(.98, .04, 'Low support: fewer than20\ncandidates in a truth class',
                                ha='right', va='bottom', transform=ax.transAxes, fontsize=8)
                ax.set(xlabel='Genuine retention', ylabel='Accidental rejection', xlim=(0, 1.01), ylim=(0, 1.01))
                ax.grid(alpha=.2)
            fig.suptitle(f'ROC at {int(size*100)}% training — {domain}', fontsize=13)
            finish(fig, pdf)

    with PdfPages(output / 'development_truth_scores.pdf') as pdf:
        for family in families:
            fig, axes = plt.subplots(3, 4, figsize=(15, 11))
            for row_index, size in enumerate(FRACTIONS):
                for column, process in enumerate(PROCESSES):
                    ax = axes[row_index, column]
                    pmask = population(table, process)
                    for name, row in entries(size, family):
                        seed = row['recipe']['seed']
                        for truth, wording, color in ((1, 'Genuine', '#267147'), (0, 'Accidental', '#b54a26')):
                            mask = pmask & (table['y'] == truth)
                            if mask.any():
                                ax.hist(table[name][mask], bins=np.linspace(0, 1, 31), density=True, histtype='step',
                                        color=color, linestyle=styles[seed], linewidth=1.4 if size == 1. else 1.,
                                        label=wording if seed == 71 else None)
                    counts = report['class_counts'][process]
                    if row_index == 0:
                        ax.set_title(f"{PROCESS_LABELS[process]}\n{counts['genuine']} genuine / {counts['accidental']} accidental", fontsize=9)
                    if column == 0: ax.set_ylabel(f'{int(size*100)}% training\nDensity within truth class')
                    if row_index == 2: ax.set_xlabel('Classifier score (unchanged)')
                    if not counts['genuine']: ax.text(.03, .9, 'No genuine candidates', transform=ax.transAxes, fontsize=7)
                    if not counts['accidental']: ax.text(.03, .8, 'No accidental candidates', transform=ax.transAxes, fontsize=7)
                    ax.set_xlim(0, 1); ax.grid(alpha=.15)
                    if row_index == 0 and column == 0: ax.legend(fontsize=8)
            fig.suptitle(labels[family] + ' — ' + domain, fontsize=13)
            finish(fig, pdf)

    with PdfPages(output / 'development_genuine_retention.pdf') as pdf:
        for size in FRACTIONS:
            fig, axes = plt.subplots(4, 4, figsize=(16, 14))
            for row_index, process in enumerate(PROCESSES):
                n = report['class_counts'][process]['genuine']
                for column, variable in enumerate((*AXES, 'category')):
                    ax = axes[row_index, column]
                    first = next(iter(models.values()))['populations'][process]
                    if variable == 'category':
                        bins = [first['categories'][category] for category in CATEGORIES]
                        tick_labels = [category + f"\nn={item['n']}" for category, item in zip(CATEGORIES, bins)]
                        minimum = 50
                    else:
                        profile = first['profiles'][variable]
                        bins = profile['bins']
                        edges = profile['interior_edges']
                        bounds = [-np.inf, *edges, np.inf]
                        tick_labels = []
                        for i, item in enumerate(bins):
                            lo, hi = bounds[i], bounds[i+1]
                            interval = f'<{hi:.3g}' if i == 0 else f'≥{lo:.3g}' if i == len(bins)-1 else f'{lo:.3g}–{hi:.3g}'
                            tick_labels.append(interval + f"\nn={item['n']}")
                        minimum = 20
                    for name, model in entries(size):
                        family, seed = model['recipe']['family'], model['recipe']['seed']
                        group = model['populations'][process]
                        cells = [group['categories'][cat] for cat in CATEGORIES] if variable == 'category' else group['profiles'][variable]['bins']
                        values = [100*cell['efficiency'] if cell['efficiency'] is not None else np.nan for cell in cells]
                        ax.plot(range(len(cells)), values, color=colors[family], linestyle=styles[seed], alpha=.85,
                                linewidth=1.3 if size == 1. else .9, label=labels[family] if seed == 71 else None)
                        for index, (cell, value) in enumerate(zip(cells, values)):
                            if cell['n']:
                                ax.scatter(index, value, s=15, edgecolors=colors[family],
                                           facecolors=colors[family] if cell['supported'] else 'none', linewidths=.8)
                    for index, cell in enumerate(bins):
                        if cell['n'] < minimum: ax.axvspan(index-.45, index+.45, color='gray', alpha=.1)
                    ax.axhline(90, color='gray', linestyle='--', linewidth=.7)
                    ax.set_ylim(0, 104); ax.set_xticks(range(len(bins)), tick_labels, fontsize=7)
                    if not n: ax.text(.5, .5, 'No genuine candidates\nRetention undefined', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    if column == 0: ax.set_ylabel(f'{PROCESS_LABELS[process]} (n={n})\nGenuine retention (%)')
                    if row_index == 0:
                        ax.set_title({'mass': 'Reconstructed mass (GeV)', 'vertex_z': 'Reconstructed z (cm)',
                                      'vertex_radius': 'Reconstructed radius (cm)', 'category': 'Reconstruction category'}[variable], fontsize=10)
                    ax.grid(axis='y', alpha=.15)
            handles, texts = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, texts, loc='upper center', bbox_to_anchor=(.5, .954), ncol=2, fontsize=8)
            fig.suptitle(f'Genuine retention at {int(size*100)}% training — {domain}', fontsize=13)
            fig.text(.025, .056, 'Filled markers: supported candidate count. Open markers/shading: fewer than20 per nuisance bin or50 per category. No independent-event CI.', fontsize=8)
            fig.text(.025, .018, footer, fontsize=8)
            fig.tight_layout(rect=(0, .08, 1, .91)); pdf.savefig(fig); plt.close(fig)

    with PdfPages(output / 'development_learning_curves.pdf') as pdf:
        for measures in (('rejection', 'auc'), ('genuine_retention', 'category_span')):
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            for ax, measure in zip(axes, measures):
                for family in families:
                    for seed in SEEDS:
                        rows = sorted(((name, row) for name, row in models.items()
                                       if row['recipe']['family'] == family and row['recipe']['seed'] == seed),
                                      key=lambda item: item[1]['recipe']['fraction'])
                        x, values = [], []
                        for name, row in rows:
                            group = row['populations']['pooled']
                            if measure in ('auc', 'rejection'): value = row[measure]
                            elif measure == 'genuine_retention': value = group['genuine']['efficiency']
                            else:
                                supported = [cell['efficiency'] for cell in group['categories'].values() if cell['supported']]
                                value = max(supported)-min(supported) if len(supported) >= 2 else np.nan
                            x.append(report['training_counts'][name]['pairs'])
                            values.append(value if measure == 'auc' else value * 100.)
                        ax.plot(x, values, color=colors[family], linestyle=styles[seed], marker='o', markersize=4,
                                label=labels[family] if seed == 71 else None)
                ax.set_xlabel('Known training pairs (actual count)')
                ax.set_ylabel({'rejection': 'Accidental rejection (%)', 'auc': 'Development AUC',
                               'genuine_retention': 'Genuine retention (%)', 'category_span': 'Supported category retention span (pp)'}[measure])
                ax.grid(alpha=.2); ax.legend(fontsize=7)
            fig.suptitle('Fixed validation-selected families — ' + domain, fontsize=13)
            finish(fig, pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report, table = load_completed(args.training.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    render(report, table, args.output)
    write(args.output / 'development_plot_summary.json', report)
    write(args.output / 'completion.json', dict(complete=True, no_model_payload_read=True,
        source=reference(__file__), helper_source=reference(Path(__file__).with_name('shift_vertex_complete_diagnostics.py')),
        input_completion=report['completion'], artifacts={path.name: reference(path) for path in sorted(args.output.iterdir())}))
    print(json.dumps(reference(args.output / 'completion.json')))


if __name__ == '__main__':
    main()
