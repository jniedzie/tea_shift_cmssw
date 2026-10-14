#!/usr/bin/env python3
# Code style: AI-code. AI-generated validation-only complete-information curves.
"""Review 81 predeclared fits using only protocol, fit and selection JSON.

No model payload, history, ROOT file, feature cache or test score is opened.
The training driver owns the authoritative selection and evaluation lock.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


SEEDS = (71, 72, 73)
FRACTIONS = (.25, .5, 1.)
BOUNDS = dict(minimum_supported_genuine_retention=.8,
              maximum_supported_marginal_span=.15,
              maximum_process_efficiency_span=.05,
              maximum_class_joint_dcor=.3,
              maximum_category_efficiency_span=.15)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_reference(ref):
    path = Path(ref['path'])
    if path.suffix != '.json':
        raise ValueError('Only JSON inputs may be opened')
    if reference(path)['sha256'] != ref['sha256']:
        raise ValueError('Changed JSON input: ' + str(path))
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gates(validation):
    profiles = [validation['flatness'], *validation['per_process_flatness'].values()]
    bins = [row['efficiency'] for profile in profiles
            for axis in ('mass', 'vertex_z', 'vertex_radius')
            for row in profile[axis]['bins'] if row['supported']]
    correlations = [validation['dependence'][label]['joint_dcor']
                    for label in ('common_vertex', 'accidental')]
    values = dict(minimum_bin=min(bins) if bins and all(map(finite, bins)) else None,
                  marginal_span=validation['maximum_marginal_efficiency_span'],
                  process_span=validation['process_efficiency_span'],
                  minimum_category=validation['minimum_supported_category_efficiency'],
                  category_span=validation['category_efficiency_span'],
                  class_joint_dcor=max(correlations) if all(map(finite, correlations)) else None)
    checks = (('minimum_bin', 'minimum_supported_genuine_retention', True),
              ('minimum_category', 'minimum_supported_genuine_retention', True),
              ('marginal_span', 'maximum_supported_marginal_span', False),
              ('category_span', 'maximum_category_efficiency_span', False),
              ('process_span', 'maximum_process_efficiency_span', False),
              ('class_joint_dcor', 'maximum_class_joint_dcor', False))
    failures = [key for key, bound, lower in checks if not finite(values[key]) or
                (values[key] < BOUNDS[bound] if lower else values[key] > BOUNDS[bound])]
    return values, failures


def aggregate(protocol, results):
    recipes = protocol['recipes']
    expected = {spec['name']: spec for spec in recipes}
    if len(recipes) != 81 or len(expected) != 81 or set(results) != set(expected):
        raise ValueError('Exactly81 unique predeclared completed fits are required')
    if 'acceptance_envelope' in protocol and protocol['acceptance_envelope'] != BOUNDS:
        raise ValueError('Acceptance envelope differs from reviewed fixed bounds')
    families, controls = {}, {}
    shared_counts = {}
    for name, spec in expected.items():
        result = results[name]
        if result['recipe'] != spec:
            raise ValueError('Fit recipe differs from protocol: ' + name)
        seed, fraction, family = spec['seed'], spec['fraction'], spec['family']
        algorithm = str(spec['algorithm']).upper()
        if seed not in SEEDS or fraction not in FRACTIONS or algorithm not in ('BDT', 'DNN'):
            raise ValueError('Undeclared seed, fraction or algorithm')
        invariant = {key: value for key, value in spec.items() if key not in ('name', 'seed', 'fraction')}
        if family in controls and controls[family] != invariant:
            raise ValueError('Family changes recipe across seed/fraction: ' + family)
        controls[family] = invariant
        row = families.setdefault(family, dict(algorithm=algorithm, recipe=invariant, fits=[]))
        if any(f['seed'] == seed and f['fraction'] == fraction for f in row['fits']):
            raise ValueError('Duplicate family seed/fraction')
        counts = result['training_counts']
        if (any(type(counts[key]) is not int or counts[key] <= 0 for key in ('pairs', 'genuine', 'fake')) or
                counts['pairs'] != counts['genuine'] + counts['fake']):
            raise ValueError('Invalid known training pair counts')
        key = seed, fraction
        if key in shared_counts and counts != shared_counts[key]:
            raise ValueError('Training subset counts differ between recipes')
        shared_counts[key] = counts
        val, train = result['validation'], result['training_at_validation_cut']
        rejection = val['rejection']
        if any(not finite(value) or not 0 <= value <= 1 for value in
               (rejection, val['auc'], val['common_vertex']['efficiency'], val['accidental']['efficiency'])):
            raise ValueError('Invalid validation performance')
        if abs(rejection - (1 - val['accidental']['efficiency'])) > 1e-12:
            raise ValueError('Rejection disagrees with accidental retention')
        if not finite(val['threshold']):
            raise ValueError('Invalid validation cut')
        values, failures = gates(val)
        for key in ('auc', 'genuine_retention', 'accidental_rejection'):
            if not finite(train[key]) or not 0 <= train[key] <= 1:
                raise ValueError('Invalid training performance')
        row['fits'].append(dict(name=name, seed=seed, fraction=fraction,
            training_counts=counts, rejection=rejection, auc=val['auc'],
            genuine_retention=val['common_vertex']['efficiency'], threshold=val['threshold'],
            validation_counts=dict(genuine=val['common_vertex'], accidental=val['accidental']),
            weighted_metrics={key: value for key, value in val.items() if 'weighted' in key or 'balanced' in key},
            gate_values=values, failed_gates=failures, acceptance_passed=not failures,
            training_minus_validation_auc=train['auc'] - val['auc'],
            training_minus_validation_rejection=train['accidental_rejection'] - rejection,
            training_minus_validation_genuine=train['genuine_retention'] - val['common_vertex']['efficiency'],
            optimizer=result['optimizer'], fit_seconds=result['fit_seconds'],
            payload_references={key: result[key] for key in ('model', 'history')}))
    if len(families) != 9:
        raise ValueError('Exactly nine fixed recipe families are required')
    expected_grid = {(seed, fraction) for seed in SEEDS for fraction in FRACTIONS}
    if any({(f['seed'], f['fraction']) for f in row['fits']} != expected_grid for row in families.values()):
        raise ValueError('Incomplete family seed/fraction grid')
    if len({json.dumps(shared_counts[(seed, 1.)], sort_keys=True) for seed in SEEDS}) != 1:
        raise ValueError('Full training population differs by seed')
    for seed in SEEDS:
        if not all(shared_counts[(seed, left)]['pairs'] < shared_counts[(seed, right)]['pairs']
                   for left, right in ((.25, .5), (.5, 1.))):
            raise ValueError('Training sizes are not increasing')
    winners, rankings = {}, {}
    for name, row in families.items():
        full = [fit for fit in row['fits'] if fit['fraction'] == 1.]
        row['full_training'] = dict(minimum_rejection=min(f['rejection'] for f in full),
                                    minimum_auc=min(f['auc'] for f in full),
                                    all_seeds_pass=all(f['acceptance_passed'] for f in full))
    for algorithm in ('BDT', 'DNN'):
        ranked = sorted((name for name, row in families.items() if row['algorithm'] == algorithm),
                        key=lambda name: (families[name]['full_training']['minimum_rejection'],
                                          families[name]['full_training']['minimum_auc'], name), reverse=True)
        if not ranked:
            raise ValueError('Both BDT and DNN families are required')
        eligible = [name for name in ranked if families[name]['full_training']['all_seeds_pass']]
        rankings[algorithm.lower()] = ranked
        winners[algorithm.lower()] = dict(rejection=ranked[0], acceptance=eligible[0] if eligible else None)
    return dict(families=families, winners=winners, full_training_rankings=rankings,
                acceptance_envelope=BOUNDS,
                ranking_rule='Within algorithm: descending minimum seed rejection, then minimum seed AUC, then descending family name. Acceptance additionally requires all three seeds to pass every supported gate.',
                interpretation=[
                    'Validation used for fitting/checkpoint/selection; current-domain validation was previously inspected. Descriptive development evidence, not independent confirmation.',
                    'All recipes receive full reconstructed information and category information; no information ablation.',
                    'Selected family is chosen only at100percent training and held fixed along each learning curve.',
                    'Three-seed spread is not a confidence interval. Subsets are nested within seed and shared across recipes.',
                    'At100percent the data are identical across seeds; below100percent seed variation mixes sampling and fitting.',
                    'Subset-fitted phase weights and unequal optimizer updates can contribute to apparent training-size dependence.',
                    'JSON counts cannot independently prove event disjointness/nesting; the sealed driver and subset artifacts own those checks.',
                    'No model payload, history, test score or feature cache was opened. Payload hashes are recorded but not reverified.',
                    'Sparse acceptance regions remain unvalidated; this is not true-lifetime or recording-efficiency validation.'])


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    families = report['families']
    labels = {'bdt_shallow_d3': 'BDT: depth3', 'bdt_medium_d5': 'BDT: depth5',
              'bdt_regularized_d5': 'BDT: regularized depth5'}
    for architecture, wording in (('shared64', 'Shared64/32'), ('shared128', 'Shared128/64'),
                                  ('heads64', 'Category heads64/32')):
        for loss, description in (('classification', 'classification only'), ('decorrelated', 'decorrelated')):
            labels['dnn_' + architecture + '_' + loss] = wording + ', ' + description
    label = lambda name: labels.get(name, name.replace('_', ' ')) if name else 'None passes all gates'
    domain = report.get('domain_metadata', {}).get('domain', '')
    domain_label = {'current_CMSIR5': 'Current CMS IR5', 'historical_proxy': 'Historical proxy stress domain'}.get(domain, domain)
    selected = {name for choices in report['winners'].values() for name in choices.values() if name}
    names = sorted(families)
    colors = dict(zip(names, plt.get_cmap('tab10').colors))
    seed_styles = {71: '-', 72: '--', 73: ':'}
    footer = ('Validation for checkpoint/selection; fixed90% genuine cuts. Three-seed spread is not a confidence interval.\n'
              'Full RECO and category information for every fit. Family selection uses100% training only; sparse regions remain unvalidated.')

    def curve(ax, family, metric, scale=1.):
        row = families[family]
        for seed in SEEDS:
            fits = sorted((fit for fit in row['fits'] if fit['seed'] == seed), key=lambda fit: fit['fraction'])
            values = [fit['gate_values'][metric] if metric in fit['gate_values'] else fit[metric] for fit in fits]
            ax.plot([fit['training_counts']['pairs'] for fit in fits],
                    [scale * value if value is not None else np.nan for value in values],
                    linestyle=seed_styles[seed], marker='o', markersize=3,
                    color=colors[family], alpha=.9 if family in selected else .55,
                    linewidth=1.7 if family in selected else 1., label=label(family) if seed == 71 else None)
        ax.set_xlabel('Known training pairs (actual count)')
        ax.grid(alpha=.2)

    pages = [
        ('Accidental rejection at the fixed validation operating point', 'rejection', 100., 'Accidental rejection (%)'),
        ('Separation with increasing training population', 'auc', 1., 'Validation AUC'),
    ]
    with PdfPages(output / 'validation_learning_curves.pdf') as pdf:
        for title, metric, scale, ylabel in pages:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 7))
            for ax, algorithm in zip(axes, ('BDT', 'DNN')):
                for name in names:
                    if families[name]['algorithm'] == algorithm:
                        curve(ax, name, metric, scale)
                choices = report['winners'][algorithm.lower()]
                ax.set_title(algorithm + '\nRejection: ' + label(choices['rejection']) + '\nAcceptance: ' + label(choices['acceptance']), fontsize=9)
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7, loc='best')
            fig.suptitle(title + '\n' + domain_label + ' — solid:seed71; dashed:seed72; dotted:seed73', fontsize=12)
            fig.text(.025, .015, footer, fontsize=8)
            fig.tight_layout(rect=(0, .075, 1, .9)); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
        for column, algorithm in enumerate(('BDT', 'DNN')):
            for row_index, metric in enumerate(('marginal_span', 'category_span')):
                ax = axes[row_index, column]
                for name in names:
                    if families[name]['algorithm'] == algorithm:
                        curve(ax, name, metric, 100.)
                ax.axhline(15., color='black', linestyle='--', linewidth=.8)
                ax.set_ylabel(('Largest supported mass/z/r span' if metric == 'marginal_span' else 'Supported category span') + ' (pp)')
                ax.set_title(algorithm)
                if row_index == 0: ax.legend(fontsize=6)
        fig.suptitle('Acceptance variation; dashed horizontal line is the15pp gate\n' + domain_label, fontsize=12)
        fig.text(.025, .015, footer + '\nPassing these two spans alone is insufficient: minimum retention, process spread and class dependence must also pass.', fontsize=8)
        fig.tight_layout(rect=(0, .085, 1, .95)); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
        for column, algorithm in enumerate(('BDT', 'DNN')):
            for row_index, metric in enumerate(('training_minus_validation_auc', 'training_minus_validation_rejection')):
                ax = axes[row_index, column]
                for name in names:
                    if families[name]['algorithm'] == algorithm:
                        curve(ax, name, metric, 1. if row_index == 0 else 100.)
                ax.axhline(0., color='black', linewidth=.8)
                ax.set_ylabel('Training minus validation ' + ('AUC' if row_index == 0 else 'rejection (pp)'))
                ax.set_title(algorithm)
                if row_index == 0: ax.legend(fontsize=6)
        capped = sum(bool(fit['optimizer'].get('cap_reached')) for family in families.values() for fit in family['fits'])
        fig.suptitle('Training–validation gaps at the same validation-derived cuts\n' + domain_label, fontsize=12)
        fig.text(.025, .015, footer + f'\nFits reaching their declared optimization cap: {capped}/81; full optimizer diagnostics are preserved in JSON.', fontsize=8)
        fig.tight_layout(rect=(0, .085, 1, .95)); pdf.savefig(fig); plt.close(fig)


def build_report(training, selection=None):
    protocol_ref = reference(training / 'protocol.json')
    protocol = read_reference(protocol_ref)
    results, refs = {}, {}
    for spec in protocol['recipes']:
        name = spec['name']
        if Path(name).name != name:
            raise ValueError('Recipe name must be a simple filename')
        refs[name] = reference(training / 'fits' / (name + '.json'))
        result = read_reference(refs[name])
        if result['protocol'] != protocol_ref:
            raise ValueError('Fit receipt references another protocol')
        results[name] = result
    report = aggregate(protocol, results)
    report.update(schema='shift-complete-information-validation-learning-curves-v1',
                  protocol=protocol_ref, fit_references=refs, source=reference(__file__),
                  domain_metadata={key: value for key, value in protocol.items()
                                   if any(term in key.lower() for term in ('domain', 'feature', 'input', 'subset', 'split'))},
                  no_model_or_test_data_read=True)
    if selection is not None:
        ref = reference(selection)
        lock = read_reference(ref)
        if lock['protocol'] != protocol_ref or lock['winners'] != report['winners']:
            raise ValueError('Independent full-fraction selection disagrees with authoritative lock')
        if lock.get('selection_uses_test') is not False or lock.get('validation_fits') != refs:
            raise ValueError('Selection does not seal these exact validation receipts')
        if lock.get('thresholds') != {name: result['validation']['threshold'] for name, result in results.items()}:
            raise ValueError('Selection thresholds differ from validation receipts')
        report['authoritative_selection_agreement'] = ref
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    report = build_report(args.training.resolve(), args.selection)
    args.output.mkdir(parents=True, exist_ok=False)
    render(report, args.output)
    write(args.output / 'validation_summary.json', report)
    write(args.output / 'completion.json', dict(complete=True, source=reference(__file__),
        protocol=report['protocol'], no_model_or_test_data_read=True,
        artifacts={path.name: reference(path) for path in sorted(args.output.iterdir())}))
    print(json.dumps(reference(args.output / 'completion.json')))


if __name__ == '__main__':
    main()
