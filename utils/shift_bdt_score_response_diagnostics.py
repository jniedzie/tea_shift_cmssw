#!/usr/bin/env python3
# Code style: AI-code. AI-generated diagnostic of a frozen BDT response.
"""Resolve score peaks into exact source bins, energy response and tree cuts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from shift_bdt_discrimination_diagnostics import digest, load_npz, read, reference, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    require(output.is_relative_to(workspace / 'tea_shift_cmssw/plots'), 'Final output must be in repository plots')
    require(not output.exists(), 'Refusing to overwrite earlier diagnostics')
    cache = workspace / 'validation/bdt_acceptance_study_20261012/comparison/extraction'
    training = workspace / 'validation/bdt_discrimination_improvement_20261012/expanded_training'
    manifest, completion = read(cache / 'inputs.json'), read(training / 'completion.json')
    for name in ('reco', 'labels'):
        require(digest(cache / (name + '.npz')) == manifest[name + '_sha256'], 'Changed cache: ' + name)
    require(digest(training / 'development_scores.npz') == completion['development']['scores']['sha256'], 'Changed scores')
    reco, labels = load_npz(cache / 'reco.npz'), load_npz(cache / 'labels.npz')
    scores = load_npz(training / 'development_scores.npz')
    for name in ('ids', 'process', 'pair_index', 'source_index'):
        require(np.array_equal(scores[name], reco[name]), 'Misaligned identities: ' + name)
    test = (scores['expanded_split'] == 2) & (scores['process'] != 'signal')
    require(np.array_equal(test, scores['expanded_heldout']), 'Changed test mask')
    genuine = test & (labels['y'] == 1)
    require(int(genuine.sum()) == 2642, 'Changed genuine population')
    name = 'expanded_d3_s0_n400'
    parity = read(training / 'all_cache_portable_parity.json')
    model_ref = parity['models'][name]['model']
    model_path = Path(model_ref['path'])
    require(digest(model_path) == model_ref['sha256'], 'Changed portable model')
    sys.path.insert(0, str(workspace / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier'))
    from portable_inference import load_model, predict
    model = load_model(model_path)
    score = scores[name]
    require(np.array_equal(predict(model, reco['X']), score), 'Portable score parity failed')
    names = list(reco['feature_names'])
    energy = reco['X'][:, names.index('mu_lowPt_energy')]
    lower, upper = (score >= .45) & (score < .60), (score >= .75) & (score < .88)
    strata = ['dy_10to20', 'dy_20to-1', 'jpsi_10to20', 'jpsi_20toinf']
    source_rows = {}
    for stratum in strata:
        mask = genuine & (reco['stratum'] == stratum)
        source_rows[stratum] = {'pairs': int(mask.sum()), 'middle_pairs': int((mask & lower).sum()),
                               'high_pairs': int((mask & upper).sum())}
    roots = []
    for index, tree in enumerate(model['trees']):
        feature = int(tree['feature'][0])
        if feature >= 0:
            roots.append({'tree': index + 1, 'feature': str(names[model['columns'][feature]]),
                          'threshold': float(tree['threshold'][0]), 'coefficient': float(tree['coefficient'])})
    energy_roots = [row for row in roots if row['feature'] == 'mu_lowPt_energy']
    root_groups = {}
    for row in energy_roots:
        group = root_groups.setdefault(row['threshold'], {'trees': 0, 'coefficient_sum': 0.0})
        group['trees'] += 1
        group['coefficient_sum'] += row['coefficient']
    dominant_threshold, dominant = max(root_groups.items(), key=lambda item: item[1]['coefficient_sum'])
    dominant['fraction_total_tree_weight'] = dominant['coefficient_sum'] / model['sum_abs_coefficients']
    edges = np.array([0, 50, 75, 100, 125, 150, 175, 200, 225, 250, 260, 270, 280, 290, 300, 350, 400, 500, 1000, 3000.])
    response = {}
    for process in ('dy', 'jpsi'):
        response[process] = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = genuine & (reco['process'] == process) & (energy >= low) & (energy < high)
            if mask.any():
                response[process].append({'energy_low_GeV': float(low), 'energy_high_GeV': float(high),
                    'pairs': int(mask.sum()), 'energy_median_GeV': float(np.median(energy[mask])),
                    'score_quartiles': np.quantile(score[mask], [.25, .5, .75]).tolist()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for axis, process, title in [(axes[0, 0], 'dy', 'DY generator mass bins'),
                                  (axes[0, 1], 'jpsi', 'J/psi generator hard-scattering pT bins')]:
        for stratum, label, color in [(strata[0 if process == 'dy' else 2], '10-20 GeV', '#0072b2'),
                                       (strata[1 if process == 'dy' else 3], '>20 GeV', '#d95f02')]:
            mask = genuine & (reco['stratum'] == stratum)
            values = score[mask]
            axis.hist(values, bins=np.linspace(0, 1, 61), weights=np.full(len(values), 1 / len(values)),
                      histtype='step', linewidth=1.5, color=color, label=f'{label}: n={len(values)}')
        axis.set(title=title, xlabel='BDT score', ylabel='Fraction within source bin', xlim=(0, 1))
        axis.legend()
    for process, title, color in [('dy', 'DY', '#0072b2'), ('jpsi', 'J/psi', '#d95f02')]:
        rows = [row for row in response[process] if row['pairs'] >= 10]
        x = np.array([row['energy_median_GeV'] for row in rows])
        quartiles = np.array([row['score_quartiles'] for row in rows])
        axes[1, 0].plot(x, quartiles[:, 1], 'o-', color=color, label=title)
        axes[1, 0].fill_between(x, quartiles[:, 0], quartiles[:, 2], color=color, alpha=.15)
    axes[1, 0].axvline(dominant_threshold, color='black', linestyle=':', label=f'Repeated root cut: {dominant_threshold:.1f} GeV')
    axes[1, 0].set(title='Observed score versus reconstructed muon energy', xlabel='Energy of lower-pT muon [GeV]',
                   ylabel='Median score; shading = middle 50% of pairs', xscale='log', xlim=(25, 1500), ylim=(.2, .9))
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].hist([row['threshold'] for row in energy_roots], bins=np.linspace(0, 1000, 81),
                    weights=[100 * row['coefficient'] / model['sum_abs_coefficients'] for row in energy_roots],
                    histtype='step', linewidth=1.5, color='black')
    axes[1, 1].set(title='Root cuts on lower-pT muon energy', xlabel='Tree energy threshold [GeV]',
                   ylabel='Percent of total ensemble coefficient weight', xlim=(0, 600))
    axes[1, 1].text(.97, .96, f'{dominant["trees"]}/{len(model["trees"])} trees share {dominant_threshold:.1f} GeV\n'
                    f'{dominant["fraction_total_tree_weight"]:.1%} of ensemble weight', transform=axes[1, 1].transAxes,
                    ha='right', va='top')
    fig.suptitle('Two score peaks: source populations plus a repeated learned energy boundary\n'
                 'Expanded-test genuine pairs; raw counts; observed response averages over other inputs', fontsize=12)
    output.mkdir(parents=True)
    pdf = output / 'score_response_and_exact_generator_bins.pdf'
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    report = {'model': name, 'genuine_pairs': int(genuine.sum()), 'all_cache_portable_parity': True,
        'interpretation': 'Consumed development diagnostic; no new training or model selection. Generator bins are not classifier inputs. Response is conditional observational data, not a controlled single-feature intervention.',
        'source_windows': {'middle': [.45, .60], 'high': [.75, .88]}, 'exact_source_bins': source_rows,
        'dominant_energy_root': {'threshold_GeV': dominant_threshold, **dominant}, 'energy_roots': energy_roots,
        'observed_energy_response': response, 'pdf': reference(pdf),
        'references': [reference(path) for path in [cache / 'inputs.json', cache / 'reco.npz', cache / 'labels.npz',
            training / 'completion.json', training / 'development_scores.npz', training / 'all_cache_portable_parity.json',
            model_path, Path(__file__)]]}
    with (output / 'score_response_findings.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'pdf': str(pdf), 'dominant_energy_root': report['dominant_energy_root']}))


if __name__ == '__main__':
    main()
