#!/usr/bin/env python3
# Code style: AI-code. AI-generated frozen-score category diagnostic utility.
"""Compare vertex categories on an identity-checked expanded development test.

Optional --inputs JSON contains path/sha256 references named cache_manifest,
reco, labels, scores and selection. Scores use the historical NPZ identity and
split columns plus one array per model; selection contains a thresholds mapping.
This also permits frozen DNN scores without changing plotting or selections.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from shift_bdt_discrimination_diagnostics import clean, digest, load_npz, read, reference, require

CATEGORIES = ('Inclusive', 'Both-Both', 'Near-Both', 'Both-Far', 'Near-Far', 'Other')
PROCESSES = ('pooled', 'dy', 'jpsi', 'qcd')
CLASSES = ((1, 'genuine', '#0072b2'), (0, 'fake', '#d55e00'), (-1, 'unknown', '#777777'))
MIN_SUPPORT = 20


def validate_identity(reco, labels, scores):
    n = len(reco['ids'])
    for key in ('ids', 'process', 'pair_index', 'source_index'):
        require(np.array_equal(reco[key], scores[key]), 'Score identity mismatch: ' + key)
    require(len(labels['y']) == n, 'Label length mismatch')
    require(set(np.unique(labels['y'])).issubset({-1, 0, 1}), 'Unknown truth label')
    identities = [(str(p), *map(int, ids), int(pair)) for p, ids, pair in
                  zip(reco['process'], reco['ids'], reco['pair_index'])]
    require(len(set(identities)) == n, 'Duplicate candidate identities')
    partitions = {}
    for p, ids, split in zip(reco['process'], reco['ids'], scores['expanded_split']):
        key = (str(p), *map(int, ids))
        require(key not in partitions or partitions[key] == int(split), 'Parent crosses splits')
        partitions[key] = int(split)
    test = (scores['expanded_split'] == 2) & (reco['process'] != 'signal')
    require(np.array_equal(test, scores['expanded_heldout']), 'Expanded test mask mismatch')
    require(set(np.unique(reco['topology'])).issubset(set(CATEGORIES[1:])), 'Unmapped topology')
    # Raw binomial intervals require one candidate per parent in this population.
    parents = [(str(p), *map(int, ids)) for p, ids in zip(reco['process'][test], reco['ids'][test])]
    require(len(set(parents)) == len(parents), 'Multiple test candidates per parent: clustered errors needed')
    return test


def interval(passed, total):
    if not total:
        return None
    fraction = passed / total
    z = 1.959963984540054
    denominator = 1 + z*z/total
    center = (fraction + z*z/(2*total)) / denominator
    half = z*np.sqrt(fraction*(1-fraction)/total + z*z/(4*total*total))/denominator
    return [float(center-half), float(center+half)]


def support(mask, train, y, label):
    return int(np.sum(mask & train & (y == label))) >= MIN_SUPPORT


def fractions(mask, accepted, weights):
    n, passed = int(mask.sum()), int((mask & accepted).sum())
    result = {'n': n, 'passed': passed, 'raw': passed/n if n else None,
              'raw_wilson95': interval(passed, n)}
    for name, weight in weights.items():
        values = weight[mask]
        total = float(values.sum())
        squares = float(np.sum(values*values))
        result[name] = {'sumw': total, 'sumw2': squares,
                        'passed_sumw': float(weight[mask & accepted].sum()),
                        'fraction': float(weight[mask & accepted].sum()/total) if total else None,
                        'effective_parents': total*total/squares if squares else 0,
                        'negative_parents': int(np.sum(values < 0))}
    return result


def roc(score, y):
    order = np.argsort(-score, kind='stable')
    score, y = score[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score)-1]
    genuine = np.r_[0, np.cumsum(y == 1)[ends]/np.sum(y == 1)]
    fake = np.r_[0, np.cumsum(y == 0)[ends]/np.sum(y == 0)]
    return genuine, 1-fake, float(np.trapz(genuine, fake))


def default_inputs(workspace):
    cache = workspace/'validation/bdt_acceptance_study_20261012/comparison/extraction'
    training = workspace/'validation/bdt_discrimination_improvement_20261012/expanded_training'
    completion = read(training/'completion.json')
    manifest_ref = completion['development']['cache']
    require(digest(manifest_ref['path']) == manifest_ref['sha256'], 'Changed frozen cache manifest')
    manifest = read(manifest_ref['path'])
    return {'cache_manifest': manifest_ref,
            'reco': {'path': str(cache/'reco.npz'), 'sha256': manifest['reco_sha256']},
            'labels': {'path': str(cache/'labels.npz'), 'sha256': manifest['labels_sha256']},
            'scores': completion['development']['scores'], 'selection': completion['selection']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--models', nargs='+', default=['reference_uniform_bdt_3p0_s71',
                        'reference_uniform_d3_s0_n400', 'expanded_d3_s0_n400', 'expanded_d3_s3_n400'])
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    require(output.is_relative_to(workspace/'tea_shift_cmssw/plots'), 'Output must be in project plots')
    require(not output.exists(), 'Refusing to overwrite existing diagnostics')
    refs = read(args.inputs) if args.inputs else default_inputs(workspace)
    for name in ('cache_manifest', 'reco', 'labels', 'scores', 'selection'):
        require(digest(refs[name]['path']) == refs[name]['sha256'], 'Changed input: ' + name)
    manifest = read(refs['cache_manifest']['path'])
    for name in ('reco', 'labels'):
        require(refs[name]['sha256'] == manifest[name+'_sha256'], 'Cache manifest disagreement: '+name)
    reco, labels, scores = [load_npz(refs[name]['path']) for name in ('reco', 'labels', 'scores')]
    selection = read(refs['selection']['path'])
    require(selection.get('selection_uses_test') is False, 'Selection must declare no test use')
    test = validate_identity(reco, labels, scores)
    train = (scores['expanded_split'] == 0) & (reco['process'] != 'signal')
    y = labels['y']
    weights = {'native_weight': reco['native_weight'], 'sampling_weight': reco['evaluation_weight']}
    for value in weights.values():
        require(np.isfinite(value).all(), 'Nonfinite weight')
    for name in args.models:
        require(name in selection['thresholds'] and name in scores, 'Missing locked model: '+name)
        require(scores[name].shape == y.shape and np.isfinite(scores[name]).all(), 'Invalid scores')
        require(np.all((scores[name] >= 0) & (scores[name] <= 1)), 'Score outside [0,1]')
    names = list(reco['feature_names'])
    observables = {'lower_pt_muon_energy_GeV': reco['X'][:, names.index('mu_lowPt_energy')],
                   'mass_GeV': reco['nuisance'][:, 0], 'vertex_z_cm': reco['nuisance'][:, 1],
                   'vertex_radius_cm': reco['nuisance'][:, 2]}
    edges = {}
    for name, values in observables.items():
        require(np.isfinite(values[test | train]).all(), 'Nonfinite observable: '+name)
        boundaries = np.unique(np.quantile(values[train & (y == 1)], [.25, .5, .75]))
        edges[name] = np.r_[-np.inf, boundaries, np.inf]
    report = {'schema': 'shift-frozen-category-diagnostics-v1', 'references': refs,
              'source': reference(Path(__file__)), 'population': 'Expanded event-disjoint development test; previously inspected',
              'selection_changed': False, 'test_pairs': int(test.sum()), 'minimum_support': MIN_SUPPORT,
              'support_policy': 'ROC: >=20 train and test parents in each class. Profile: >=20 in the displayed class in train and test. This is a display support gate, not a precision guarantee.',
              'binning': 'Pooled genuine training quartiles, inclusive underflow/overflow',
              'profile_edges': clean(edges), 'categories_merged': False,
              'weights_policy': 'Raw shapes primary; native and cached exposure/sampling weights retained separately without abs/clipping. No full physical SM-mixture claim.',
              'unknown_policy': 'Unknown shown separately; never treated as fake or included in ROC',
              'categories': {}, 'models': {}}
    masks = {}
    for process in PROCESSES:
        for category in CATEGORIES:
            mask = np.ones(len(y), dtype=bool)
            if process != 'pooled': mask &= reco['process'] == process
            if category != 'Inclusive': mask &= reco['topology'] == category
            key = process+'/'+category
            masks[key] = mask
            report['categories'][key] = {'test': {label: int(np.sum(mask & test & (y == value))) for value,label,_ in CLASSES},
                                         'train': {label: int(np.sum(mask & train & (y == value))) for value,label,_ in CLASSES}}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output.mkdir(parents=True)
    score_edges = np.linspace(0, 1, 51)
    for model in args.models:
        score = scores[model]
        cut = float(selection['thresholds'][model])
        accepted = score >= cut
        rows = {}
        with PdfPages(output/(model+'_categories.pdf')) as pdf:
            for process in PROCESSES:
                fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
                for axis, category in zip(axes.flat, CATEGORIES):
                    key = process+'/'+category
                    mask = masks[key]
                    row = {'cut': cut, 'classes': {}, 'profiles': {}}
                    for value, label, color in CLASSES:
                        selected = mask & test & (y == value)
                        values = score[selected]
                        record = fractions(selected, accepted, weights)
                        record['supported'] = support(mask, train, y, value) and len(values) >= MIN_SUPPORT
                        record['score_histogram'] = {'edges': score_edges.tolist(), 'raw': np.histogram(values, score_edges)[0].tolist()}
                        for weight_name, weight in weights.items():
                            record['score_histogram'][weight_name] = np.histogram(values, score_edges, weights=weight[selected])[0].tolist()
                        record['middle_score_n'] = int(np.sum((values >= .45) & (values < .60)))
                        record['upper_score_n'] = int(np.sum((values >= .75) & (values < .88)))
                        row['classes'][label] = record
                        if len(values):
                            axis.hist(values, score_edges, weights=np.ones(len(values))/len(values), histtype='step', color=color,
                                      label=f'{label}: n={len(values)}', linewidth=1.4)
                    axis.axvline(cut, color='black', linestyle=':', linewidth=1)
                    axis.set(title=category, xlabel='Score', ylabel='Fraction within truth class', xlim=(0,1))
                    if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
                    else: axis.text(.5,.5,'No test pairs',ha='center',transform=axis.transAxes)
                    labelled = mask & test & (y >= 0)
                    supported = all(row['classes'][label]['supported'] for label in ('genuine','fake'))
                    row['roc'] = {'supported': supported}
                    if supported:
                        efficiency, rejection, auc = roc(score[labelled], y[labelled])
                        row['roc'].update(genuine_efficiency=efficiency.tolist(), fake_rejection=rejection.tolist(), auc=auc)
                    for obs, values in observables.items():
                        row['profiles'][obs] = []
                        for low, high in zip(edges[obs][:-1], edges[obs][1:]):
                            in_bin = mask & (values >= low) & (values < high)
                            cell = {'low': clean(float(low)), 'high': clean(float(high)), 'classes': {}}
                            for value, label, _ in CLASSES:
                                selected = in_bin & test & (y == value)
                                entry = fractions(selected, accepted, weights)
                                entry['supported'] = support(in_bin, train, y, value) and selected.sum() >= MIN_SUPPORT
                                entry['score_median'] = float(np.median(score[selected])) if selected.any() else None
                                cell['classes'][label] = entry
                            row['profiles'][obs].append(cell)
                    rows[key] = row
                fig.suptitle(f'{model} | {process} | locked cut {cut:.4f}\nExpanded test; raw pair shapes; unknown separate; sparse categories descriptive', fontsize=11)
                pdf.savefig(fig); plt.close(fig)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
            for axis, process in zip(axes.flat, PROCESSES):
                for category in CATEGORIES:
                    row = rows[process+'/'+category]
                    if row['roc']['supported']:
                        axis.plot(row['roc']['genuine_efficiency'], row['roc']['fake_rejection'], label=f"{category} AUC={row['roc']['auc']:.3f}")
                axis.set(title=process, xlabel='Genuine retention', ylabel='Fake rejection', xlim=(0,1), ylim=(0,1))
                if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
                else: axis.text(.5,.5,'Insufficient class support for ROC',ha='center',transform=axis.transAxes)
            fig.suptitle(model+' | Truth ROC, raw parents; >=20 train/test per class required',fontsize=11)
            pdf.savefig(fig); plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
            for axis, label, title in [(axes[0],'genuine','Genuine retention'),(axes[1],'fake','Fake rejection')]:
                for index, process in enumerate(PROCESSES):
                    xx, yy, errors = [], [], []
                    for ci, category in enumerate(CATEGORIES):
                        entry = rows[process+'/'+category]['classes'][label]
                        if not entry['supported']: continue
                        value = entry['raw']; low, high = entry['raw_wilson95']
                        if label == 'fake': value, low, high = 1-value, 1-high, 1-low
                        xx.append(ci+(index-1.5)*.14); yy.append(value); errors.append([value-low,high-value])
                    if xx: axis.errorbar(xx, yy, yerr=np.array(errors).T, fmt='o', label=process, capsize=2)
                axis.set(xticks=range(6), xticklabels=CATEGORIES, ylim=(0,1.03), title=title)
                axis.tick_params(axis='x',rotation=35); axis.legend(fontsize=8)
            fig.suptitle(model+f' | Fixed validation cut {cut:.4f}; Wilson 95% intervals; unsupported cells omitted',fontsize=11)
            pdf.savefig(fig); plt.close(fig)
            for label in ('genuine','fake'):
                fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
                for axis, (obs, bins) in zip(axes.flat, edges.items()):
                    for category in CATEGORIES:
                        cells = rows['pooled/'+category]['profiles'][obs]
                        xx, yy, errors = [], [], []
                        for index, cell in enumerate(cells):
                            entry = cell['classes'][label]
                            if not entry['supported']: continue
                            value = entry['raw']; low, high = entry['raw_wilson95']
                            if label == 'fake': value, low, high = 1-value, 1-high, 1-low
                            xx.append(index); yy.append(value); errors.append([value-low, high-value])
                        if xx: axis.errorbar(xx,yy,yerr=np.array(errors).T,fmt='o-',label=category,capsize=2)
                    labels_text = [f'{low:.3g} to {high:.3g}' for low,high in zip(bins[:-1],bins[1:])]
                    axis.set(title=obs,xticks=range(len(labels_text)),xticklabels=labels_text,ylim=(0,1.03),ylabel='Retention' if label=='genuine' else 'Rejection')
                    axis.tick_params(axis='x',rotation=25)
                    if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=7)
                fig.suptitle(model+f' | Pooled {label}; fixed cut; training-defined bins, supported cells only\nRaw sampled process mixture; not a physical SM rate prediction',fontsize=11)
                pdf.savefig(fig); plt.close(fig)
        report['models'][model] = rows
    report['pdfs'] = [reference(path) for path in sorted(output.glob('*.pdf'))]
    (output/'category_report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'test_pairs':int(test.sum()),'pdfs':len(report['pdfs'])}))


if __name__ == '__main__':
    main()
