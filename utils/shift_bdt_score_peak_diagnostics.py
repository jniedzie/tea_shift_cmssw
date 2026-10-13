#!/usr/bin/env python3
# Code style: AI-code. AI-generated diagnostic of frozen BDT score populations.
"""Describe genuine-score peaks without fitting or changing model selections."""
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
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    plot_root = workspace / 'tea_shift_cmssw/plots'
    require(output.is_relative_to(plot_root), 'Final diagnostics must be inside the repository plots directory')
    require(not output.exists(), 'Refusing to overwrite a previous diagnostic')
    cache = workspace / 'validation/bdt_acceptance_study_20261012/comparison/extraction'
    training = workspace / 'validation/bdt_discrimination_improvement_20261012/expanded_training'
    manifest = read(cache / 'inputs.json')
    completion = read(training / 'completion.json')
    for name in ('reco', 'labels'):
        require(digest(cache / (name + '.npz')) == manifest[name + '_sha256'], 'Cache hash changed: ' + name)
    require(digest(training / 'development_scores.npz') == completion['development']['scores']['sha256'], 'Score hash changed')
    require(digest(training / 'selection_locked.json') == completion['selection']['sha256'], 'Selection lock changed')
    reco = load_npz(cache / 'reco.npz')
    labels = load_npz(cache / 'labels.npz')
    scores = load_npz(training / 'development_scores.npz')
    for name in ('ids', 'process', 'pair_index', 'source_index'):
        require(np.array_equal(scores[name], reco[name]), 'Row identity mismatch: ' + name)
    test = (scores['expanded_split'] == 2) & (scores['process'] != 'signal')
    require(np.array_equal(test, scores['expanded_heldout']), 'Expanded test mask mismatch')
    genuine = test & (labels['y'] == 1)
    require(int(genuine.sum()) == 2642, 'Expanded genuine population changed')
    model_name = 'expanded_d3_s0_n400'
    score = scores[model_name]
    feature_index = {str(name): i for i, name in enumerate(reco['feature_names'])}
    energy = reco['X'][:, feature_index['mu_lowPt_energy']]
    chi2 = reco['X'][:, feature_index['pair_normalizedChi2']]
    harder = np.isin(reco['stratum'], ['dy_10to20', 'dy_20to-1', 'jpsi_10to20', 'jpsi_20toinf'])
    report = {
        'model': model_name,
        'population': 'Expanded-test genuine pairs only; previously inspected development data',
        'interpretation': 'Descriptive associations; no causal claim, fit, new cut, or model selection',
        'source_bin_definition': 'DY generator mHat bins; J/psi generator pThat bins; lower edge at least 10 GeV',
        'score_windows': {'middle': [0.45, 0.60], 'high': [0.75, 0.88]},
        'processes': {},
    }
    for process in ('dy', 'jpsi'):
        rows = {}
        for name, lower, upper in [('middle', .45, .60), ('high', .75, .88)]:
            mask = genuine & (reco['process'] == process) & (score >= lower) & (score < upper)
            require(mask.any(), 'Empty score window')
            rows[name] = {
                'pairs': int(mask.sum()),
                'harder_source_pairs': int((mask & harder).sum()),
                'median_lower_pt_leg_energy_GeV': float(np.median(energy[mask])),
                'median_pair_chi2_ndof': float(np.median(chi2[mask])),
                'median_reconstructed_mass_GeV': float(np.median(reco['nuisance'][mask, 0])),
                'median_reconstructed_radius_cm': float(np.median(reco['nuisance'][mask, 2])),
            }
        report['processes'][process] = rows
    # Verify that the plotted score is the sealed reconstructed-only inference.
    classifier = workspace / 'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier'
    sys.path.insert(0, str(classifier))
    from portable_inference import load_model, predict
    parity = read(training / 'all_cache_portable_parity.json')
    model_reference = parity['models'][model_name]['model']
    model_path = Path(model_reference['path'])
    require(digest(model_path) == model_reference['sha256'], 'Portable model hash changed')
    require(np.array_equal(predict(load_model(model_path), reco['X']), score), 'Portable score parity failed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for row, (process, title, source_variable) in enumerate([
        ('dy', 'DY', 'generator mass'), ('jpsi', 'J/psi', 'generator hard-scattering pT')
    ]):
        mask = genuine & (reco['process'] == process)
        for group, color, label in [(~harder, '#0072b2', '< 10 GeV'), (harder, '#d95f02', '>= 10 GeV')]:
            selected = mask & group
            values = score[selected]
            axes[row, 0].hist(values, bins=np.linspace(0, 1, 61), weights=np.full(len(values), 1 / len(values)),
                              histtype='step', color=color, linewidth=1.5, label=f'{label}: n={len(values)}')
        axes[row, 0].set(title=f'{title}: genuine pairs by {source_variable} bin',
                         xlabel='BDT score', ylabel='Fraction within source group', xlim=(0, 1))
        axes[row, 0].legend(fontsize=8)
        for column, values, ylabel in [(1, energy, 'Energy of lower-pT muon [GeV]'),
                                       (2, chi2, 'Pair fit chi2 / ndof')]:
            selected = mask & np.isfinite(values) & (values > 0)
            axes[row, column].hexbin(score[selected], np.log10(values[selected]), gridsize=35, mincnt=1,
                                      bins='log', cmap='viridis')
            axes[row, column].set(title=title + ': genuine pairs', xlabel='BDT score',
                                   ylabel='log10(' + ylabel + ')', xlim=(0, 1))
    fig.suptitle('Origin of the genuine-score peaks: harder inputs receive higher scores\n'
                 'Expanded test only; raw pair counts; associations, not isolated causal effects', fontsize=12)
    output.mkdir(parents=True)
    pdf = output / 'genuine_score_peak_breakdown.pdf'
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    report['verified_all_cache_portable_parity'] = True
    report['references'] = [reference(path) for path in [cache / 'inputs.json', cache / 'reco.npz',
        cache / 'labels.npz', training / 'completion.json', training / 'development_scores.npz',
        training / 'selection_locked.json', model_path, Path(__file__)]]
    report['pdf'] = reference(pdf)
    with (output / 'score_peak_findings.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'pdf': str(pdf), 'report': str(output / 'score_peak_findings.json'),
                      'genuine_pairs': int(genuine.sum()), 'score_parity': True}))


if __name__ == '__main__':
    main()
