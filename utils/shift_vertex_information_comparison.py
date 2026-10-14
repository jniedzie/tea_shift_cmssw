# Code style: AI-code. AI-generated validation-only feature and seed comparison.
"""Compare a complete, predeclared study using JSON receipts only.

No model pickle, cache, development score or ROOT file is read. The result is
an independent descriptive review; the study driver owns the selection lock.
"""
import argparse
import hashlib
import json
from pathlib import Path


def reference(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_reference(ref):
    path = Path(ref['path'])
    if reference(path)['sha256'] != ref['sha256']:
        raise ValueError('Changed JSON/source artifact: ' + str(path))
    return json.loads(path.read_text())


def gate_values(entry):
    efficiencies = [row['efficiency']
                    for profiles in [entry['flatness'], *entry['per_process_flatness'].values()]
                    for variable in ('mass', 'vertex_z', 'vertex_radius')
                    for row in profiles[variable]['bins'] if row['supported']]
    return dict(minimum_bin=min(efficiencies) if efficiencies else None,
                marginal_span=entry['maximum_marginal_efficiency_span'],
                process_span=entry['process_efficiency_span'],
                class_joint_dcor=max(entry['dependence'][label]['joint_dcor'] or 0.
                                     for label in ('common_vertex', 'accidental')),
                minimum_category=entry['minimum_supported_category_efficiency'],
                category_span=entry['category_efficiency_span'])


def failures(entry, bounds):
    values = gate_values(entry)
    checks = [('minimum_bin', 'minimum_supported_genuine_retention', True),
              ('marginal_span', 'maximum_supported_marginal_span', False),
              ('process_span', 'maximum_process_efficiency_span', False),
              ('class_joint_dcor', 'maximum_class_joint_dcor', False),
              ('minimum_category', 'minimum_supported_genuine_retention', True),
              ('category_span', 'maximum_category_efficiency_span', False)]
    return [key for key, bound, lower in checks
            if values[key] is None or (values[key] < bounds[bound] if lower else values[key] > bounds[bound])]


def aggregate(protocol, results):
    expected = {spec['name']: spec for spec in protocol['recipes']}
    if len(expected) != 24 or set(results) != set(expected):
        raise ValueError('All24 unique predeclared fit receipts are required')
    families = {}
    for name, result in results.items():
        if result['recipe'] != expected[name]:
            raise ValueError('Fit recipe differs from protocol')
        spec, val = result['recipe'], result['validation']
        failed = failures(val, protocol['acceptance_envelope'])
        family = families.setdefault(spec['family'], {})
        if spec['seed'] in family:
            raise ValueError('Duplicate family seed')
        family[spec['seed']] = dict(name=name, rejection=val['rejection'], auc=val['auc'],
                                  genuine_retention=val['common_vertex']['efficiency'],
                                  gate_values=gate_values(val), failed_gates=failed,
                                  passed=not failed, threshold=val['threshold'])
    summaries = {}
    for name, seeds in families.items():
        if set(seeds) != {71, 72}:
            raise ValueError('Each family requires both declared seeds')
        summaries[name] = dict(seeds=seeds, all_seeds_pass=all(s['passed'] for s in seeds.values()),
                               minimum_rejection=min(s['rejection'] for s in seeds.values()),
                               minimum_auc=min(s['auc'] for s in seeds.values()))
    eligible = [name for name, row in summaries.items() if row['all_seeds_pass']]
    winner = max(eligible, key=lambda name: (summaries[name]['minimum_rejection'],
                                            summaries[name]['minimum_auc'], name)) if eligible else None
    contrasts = []
    pairs = [('global_rich', 'global_raw50', 'Added information: global'),
             ('shared_rich', 'shared_raw50', 'Added information: shared category'),
             ('heads_rich', 'heads_raw50', 'Added information: category heads'),
             ('global_rich', 'global_muon', 'Added event occupancy: global'),
             ('shared_rich_accept30', 'shared_rich_accept10', 'Soft acceptance strength30 minus10')]
    for first, second, label in pairs:
        for seed in (71, 72):
            a, b = families[first][seed], families[second][seed]
            contrasts.append(dict(label=label, first=first, second=second, seed=seed,
                                  rejection_change_pp=100*(a['rejection']-b['rejection']),
                                  auc_change=a['auc']-b['auc'],
                                  category_span_change_pp=100*(a['gate_values']['category_span']-b['gate_values']['category_span']),
                                  marginal_span_change_pp=100*(a['gate_values']['marginal_span']-b['gate_values']['marginal_span'])))
    return dict(families=summaries, robust_eligible_families=eligible, robust_family=winner,
                paired_seed_contrasts=contrasts)


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    families = report['families']
    names = list(families)
    labels = {'global_raw50': 'Global: original50', 'global_rich': 'Global: rich119',
              'shared_raw50': 'Shared category: original50', 'shared_rich': 'Shared category: rich119',
              'heads_raw50': 'Category heads: original50', 'heads_rich': 'Category heads: rich119',
              'heads_rich_weak': 'Category heads: weaker loss', 'heads_rich_strong': 'Category heads: stronger loss',
              'shared_rich_strong': 'Shared category: stronger loss', 'global_muon': 'Global: rich without occupancy',
              'shared_rich_accept10': 'Shared: soft acceptance10', 'shared_rich_accept30': 'Shared: soft acceptance30'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for seed, color, offset in [(71, '#235789', -.12), (72, '#c45c21', .12)]:
        for i, name in enumerate(names):
            row = families[name]['seeds'][seed]
            for ax, value in zip(axes, [100*row['rejection'], 100*row['gate_values']['marginal_span'],
                                       100*row['gate_values']['category_span']]):
                ax.scatter(value, i+offset, color=color, marker='o' if row['passed'] else 'x', s=46,
                           label=f'Seed{seed}' if i == 0 else None)
    axes[0].set_yticks(range(len(names)), [labels[n] for n in names]); axes[0].invert_yaxis()
    axes[0].set_xlabel('Accidental rejection (%)')
    axes[1].set_xlabel('Largest mass/z/r retention span (pp)')
    axes[2].set_xlabel('Category retention span (pp)')
    for ax in axes[1:]: ax.axvline(15, color='gray', linestyle='--', linewidth=1)
    for ax in axes: ax.grid(axis='x', alpha=.25)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Validation only: paired seeds and fixed genuine90% operating points', fontsize=13)
    fig.text(.03, .015, 'Circle: all acceptance gates pass; cross: at least one fails. Dashed:15pp span limit.\n'
             'Sparse bins/categories remain unvalidated. No independent-test claim; error bars are not seed uncertainty.', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95)); fig.savefig(output/'validation_families.pdf'); plt.close(fig)
    contrasts = report['paired_seed_contrasts']; unique = list(dict.fromkeys(r['label'] for r in contrasts))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for seed, color, offset in [(71, '#235789', -.1), (72, '#c45c21', .1)]:
        rows = [r for r in contrasts if r['seed'] == seed]
        axes[0].scatter([r['rejection_change_pp'] for r in rows], np.arange(len(rows))+offset, color=color, label=f'Seed{seed}')
        axes[1].scatter([r['auc_change'] for r in rows], np.arange(len(rows))+offset, color=color)
    axes[0].set_yticks(range(len(unique)), unique); axes[0].invert_yaxis()
    axes[0].set_xlabel('Change in accidental rejection (percentage points)'); axes[1].set_xlabel('Change in AUC')
    for ax in axes: ax.axvline(0, color='gray', linewidth=1); ax.grid(axis='x', alpha=.25)
    axes[0].legend(fontsize=8); fig.suptitle('Matched validation comparisons: positive means improved rejection or AUC')
    fig.text(.025, .02, 'Raw/rich pairs share loss and optimizer; occupancy comparison retains original pair predictors.\n'
             'Soft acceptance10/30 share all settings except their acceptance coefficient. Two seeds are not a confidence interval.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95)); fig.savefig(output/'validation_information_changes.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); study = args.study.resolve(); output = args.output.resolve()
    protocol_ref = reference(study/'protocol.json'); protocol = read_reference(protocol_ref)
    results, refs = {}, {}
    for spec in protocol['recipes']:
        path = study/(spec['name']+'_validation.json'); refs[spec['name']] = reference(path)
        result = read_reference(refs[spec['name']])
        if result['protocol'] != protocol_ref: raise ValueError('Fit receipt protocol differs')
        history = read_reference(result['history'])
        if not history['restored_prediction_exact'] or history['best_state_sha256'] != history['restored_state_sha256']:
            raise ValueError('Checkpoint restoration failed')
        if history['best_epoch'] != min(history['epochs'], key=lambda row: row['val_objective'])['epoch']:
            raise ValueError('Checkpoint objective differs')
        results[spec['name']] = result
    report = aggregate(protocol, results)
    report.update(schema='shift-independent-information-validation-comparison-v1', protocol=protocol_ref,
                  receipt_references=refs, source=reference(Path(__file__)),
                  no_model_pickles_or_test_scores_read=True,
                  interpretation='Descriptive validation ablations. The driver owns the authoritative pretest lock; no test-based model choice.',
                  model_payload_hashes_rechecked=False)
    lock_path = study/'selection_locked.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if report['robust_family'] != lock['robust_family']: raise ValueError('Independent family ranking disagrees')
        report['authoritative_lock_agreement'] = reference(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    render(report, output)
    with (output/'validation_ablation.json').open('x') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    receipt = dict(complete=True, artifacts={p.name: reference(p) for p in sorted(output.iterdir())},
                   source=reference(Path(__file__)), no_model_pickles_or_test_scores_read=True)
    with (output/'completion.json').open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(reference(output/'completion.json')))


if __name__ == '__main__': main()
