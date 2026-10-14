# Code style: AI-code. AI-generated frozen-model cross-domain stress check.
"""Apply locked source full-size families to target development, without tuning.

One direction per fresh process. Both domains must have all81 fits, selection
locks and completed native development evaluation before any NPZ is opened.
Only reconstructed119 features and reconstructed categories enter inference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def verify(reference):
    assert ref(reference['path'])['sha256'] == reference['sha256'], reference['path']


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def selected(protocol, locked):
    families = {family for roles in locked['winners'].values() for family in roles.values() if family}
    return [spec for spec in protocol['recipes'] if spec['fraction'] == 1. and spec['family'] in families]


def rates(y, passed, mask):
    result = {}
    for label, name in ((1, 'genuine_retention'), (0, 'fake_rejection')):
        use = mask & (y == label)
        n = int(use.sum()); k = int((use & (passed if label else ~passed)).sum())
        # Wilson 95% interval displays small support without hiding sparse cells.
        if n:
            probability = k/n; z = 1.959963984540054
            center = (probability + z*z/(2*n))/(1+z*z/n)
            radius = z*np.sqrt(probability*(1-probability)/n+z*z/(4*n*n))/(1+z*z/n)
            interval = [max(0., center-radius), min(1., center+radius)]
        else:
            probability = None; interval = None
        result[name] = {'n': n, 'accepted': k, 'efficiency': probability,
                        'wilson95': interval, 'supported_n50': n >= 50}
    return result


def seed_range(values):
    """Keep zero-width ranges nonnegative despite floating mean roundoff."""
    center = float(np.mean(values))
    return center, max(0., center - min(values)), max(0., max(values) - center)


def execute(source, target, output, review_helper, review_sha256):
    assert source.resolve() != target.resolve()
    assert not output.exists(), 'Refusing transfer overwrite'
    verify({'path': str(review_helper), 'sha256': review_sha256})
    helper_spec = importlib.util.spec_from_file_location('independent_transfer_review', review_helper)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    review = helper.review
    source_audit, target_audit = review(source, True), review(target, True)
    assert 'completion' in source_audit and 'completion' in target_audit
    sp, tp = read(source/'protocol.json'), read(target/'protocol.json')
    sl, tl = read(source/'selection_locked.json'), read(target/'selection_locked.json')
    assert sp['domain'] != tp['domain']
    assert sp['input_feature_names'] == tp['input_feature_names']
    assert sp['category_names'] == tp['category_names']
    # Both protocols pin the same implementation bytes; model classes resolve
    # exclusively from the SOURCE snapshot in this fresh process.
    assert set(sp['sources']) == set(tp['sources'])
    assert all(sp['sources'][name]['sha256'] == tp['sources'][name]['sha256'] for name in sp['sources'])
    sys.path.insert(0, str(source/'source_snapshot'))
    import vertex_complete_study as driver
    assert Path(driver.__file__).resolve() == (source/'source_snapshot/vertex_complete_study.py').resolve()
    reco, labels, splits = [load(tp[name]['path']) for name in ('reco', 'labels', 'splits')]
    for key in ('ids', 'process', 'pair_index', 'source_index'):
        assert np.array_equal(reco[key], splits[key])
    mask = (splits['split'] == 2) & np.isin(reco['process'], ['dy', 'jpsi', 'qcd']) & (labels['y'] >= 0)
    y, process = labels['y'][mask], reco['process'][mask]
    assert len(set(splits['parent_key'][mask])) == len(y), 'Wilson intervals require one pair per target parent'
    target_scores = load(read(target/'completion.json')['scores']['path'])
    assert np.array_equal(target_scores['ids'], reco['ids'][mask])
    assert np.array_equal(target_scores['process'], process)
    assert np.array_equal(target_scores['y'], y)
    populations = {'pooled': np.ones(len(y), dtype=bool)}
    populations.update({str(name): process == name for name in np.unique(process)})
    source_specs, target_specs = selected(sp, sl), selected(tp, tl)
    predictions, rows = {}, {}
    for origin, directory, lock, specs in [('source', source, sl, source_specs), ('target_own', target, tl, target_specs)]:
        for spec in specs:
            name = spec['name']; receipt = read(directory/'fits'/(name+'.json'))
            verify(receipt['model']); cut = lock['thresholds'][name]
            assert receipt['validation']['threshold'] == cut
            if origin == 'source':
                with Path(receipt['model']['path']).open('rb') as stream:
                    bundle = pickle.load(stream)
                assert bundle['threshold'] == cut and bundle['requires_gen'] is False
                assert bundle['feature_names'] == sp['input_feature_names']
                score = driver.model_score(bundle, reco, mask)
                repeat = driver.model_score(bundle, reco, mask)
                assert np.array_equal(score, repeat), 'Inference reproducibility failure'
            else:
                score = target_scores[name]
            assert score.shape == y.shape and np.isfinite(score).all()
            key = origin+'__'+name; predictions[key] = score
            rows[key] = {'origin': origin, 'recipe': spec, 'cut': cut, 'model': receipt['model'],
                         'cut_population': 'source validation' if origin == 'source' else 'target validation',
                         'per_process': {name: rates(y, score >= cut, population) for name, population in populations.items()}}
    for filename, source_ref in sp['sources'].items():
        module = sys.modules.get(Path(filename).stem)
        if module is not None:
            assert Path(module.__file__).resolve() == Path(source_ref['path']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'scores.npz', ids=reco['ids'][mask], source_index=reco['source_index'][mask],
                        pair_index=reco['pair_index'][mask], parent_key=splits['parent_key'][mask],
                        process=process, y=y, **predictions)
    report = {'complete': True, 'study': 'historical domain-shift stress check',
              'source_domain': sp['domain'], 'target_domain': tp['domain'],
              'source_protocol': ref(source/'protocol.json'), 'target_protocol': ref(target/'protocol.json'),
              'source_selection': ref(source/'selection_locked.json'), 'target_selection': ref(target/'selection_locked.json'),
              'source_completion': ref(source/'completion.json'), 'target_completion': ref(target/'completion.json'),
              'inputs_verified_before_target_development_read': True,
              'target_development_known_pairs': len(y), 'unique_target_parents': len(y),
              'target_development_unknown_pairs_excluded': int(((splits['split'] == 2) & (labels['y'] < 0) & np.isin(reco['process'], ['dy','jpsi','qcd'])).sum()),
              'selection_or_tuning_performed': False, 'thresholds_unchanged': True,
              'physical_mixture_normalization_claim': False, 'models': rows,
              'scores': ref(output/'scores.npz'), 'source': ref(__file__),
              'review_helper': ref(review_helper)}
    render(report, output/'transfer.pdf')
    report['pdf'] = ref(output/'transfer.pdf')
    with (output/'receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'source':sp['domain'], 'target':tp['domain'], 'pairs':len(y), 'models':len(rows)}))


def render(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    rows = report['models']
    groups = sorted({(row['origin'], row['recipe']['family']) for row in rows.values()})
    processes = list(next(iter(rows.values()))['per_process'])
    with PdfPages(path) as pdf:
        for process in processes:
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4.8, len(groups)*.48+2.5)), sharey=True)
            for i, (origin, family) in enumerate(groups):
                fitted = [row for row in rows.values() if row['origin'] == origin and row['recipe']['family'] == family]
                assert len(fitted) == 3
                for axis, metric in zip(axes, ['genuine_retention','fake_rejection']):
                    cells = [row['per_process'][process][metric] for row in fitted]
                    finite = [cell['efficiency'] for cell in cells if cell['efficiency'] is not None]
                    n = cells[0]['n']; color = '#2468a0' if origin == 'source' else '#777777'
                    if finite:
                        center, lower_error, upper_error = seed_range(finite)
                        axis.errorbar(100*center, i, xerr=[[100*lower_error],[100*upper_error]], fmt='o', color=color, capsize=4)
                    else:
                        axis.text(1, i, 'undefined: no support', va='center', fontsize=8)
                    axis.text(101, i, f'n={n}'+(' sparse' if n<50 else ''), va='center', fontsize=8)
            axes[0].set_yticks(range(len(groups)), [origin.replace('_',' ')+' | '+family for origin,family in groups], fontsize=8)
            for axis, title in zip(axes, ['Genuine retention [%]','Fake rejection [%]']):
                axis.set_xlim(-2,122);axis.set_xticks([0,25,50,75,100]);axis.set_xlabel(title);axis.grid(axis='x', alpha=.25)
            axes[0].invert_yaxis()
            fig.suptitle(f"{report['source_domain']} → {report['target_domain']} | {process}", fontsize=12)
            fig.text(.5,.025,'Frozen source validation cuts; target-own cuts from target validation. Markers: seed mean; bars: seed range, not statistical CI.\n'
                     'Known-label target development only. Sparse support shown; no physical mixture or current-geometry confirmation claim.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.085,1,.94));pdf.savefig(fig);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--review-helper',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    args=parser.parse_args();execute(args.source.resolve(),args.target.resolve(),args.output.resolve(),
                                    args.review_helper.resolve(),args.review_sha256)
