# Code style: AI-code. AI-generated auditable source-grouped classifier comparison.
"""Compare saved cross-fit predictions without fitting or changing any cut."""
import argparse
import json
from pathlib import Path

import numpy as np

from shift_classifier_crossfit import SM_PROCESSES, digest, fold_masks, validate_data, validation_cut


def wilson(fraction, n):
    if not n:
        return [None, None]
    z = 1.96; denominator = 1 + z*z/n
    center = (fraction + z*z/(2*n)) / denominator
    half = z*np.sqrt(fraction*(1-fraction)/n + z*z/(4*n*n)) / denominator
    return [max(0., float(center-half)), min(1., float(center+half))]


def verified_predictions(data, input_hash, directory):
    """Reject row substitutions, cut changes and any model/source overlap."""
    directory = Path(directory); receipt = json.loads((directory/'completion.json').read_text())
    if (not receipt.get('completed') or receipt.get('input_sha256') != input_hash
            or receipt.get('unscored_rows', 0) or len(receipt.get('models', [])) != 2):
        raise ValueError('A complete two-fold result on this exact input is required')
    path = directory/'predictions.npz'
    if digest(path) != receipt['predictions_sha256']:
        raise ValueError('Predictions changed after completion')
    prediction = dict(np.load(path, allow_pickle=False)); n = len(data['y'])
    np.testing.assert_array_equal(prediction['row_index'], np.arange(n))
    for key in ('ids', 'process', 'point', 'category', 'pair_index', 'group_id', 'event_owner_id', 'y', 'weight', 'mass'):
        if key not in data or key not in prediction:
            raise ValueError('Exact original-event ownership and row metadata are required: '+key)
        np.testing.assert_array_equal(prediction[key], data[key])
    score, cut = prediction['crossfit_score'], prediction['crossfit_cut']
    if not np.isfinite(score).all() or not np.isfinite(cut).all() or np.any((score<0)|(score>1)):
        raise ValueError('Invalid saved scores or thresholds')
    np.testing.assert_array_equal(prediction['crossfit_pass'], score>=cut)
    if {r['scored_fold'] for r in receipt['models']} != {0, 1}:
        raise ValueError('Distinct original-source folds are required')
    for model in receipt['models']:
        fold = model['scored_fold']; train, validation, scored = fold_masks(data, fold, receipt['fit_mode'])
        for name, mask in (('train', train), ('validation', validation), ('scored', scored)):
            if model['groups'][name] != sorted(set(map(str, data['group_id'][mask]))):
                raise ValueError('Model source ownership differs from the original input')
        np.testing.assert_array_equal(prediction['scored_fold'][scored], fold)
        np.testing.assert_array_equal(cut[scored], model['threshold'])
        fold_dir = directory/('fold'+str(fold)); partitions = dict(np.load(fold_dir/'partitions.npz', allow_pickle=False))
        for name, mask in (('train', train), ('validation', validation), ('scored', scored)):
            np.testing.assert_array_equal(partitions[name], np.flatnonzero(mask))
        calibration = np.full(n, np.nan); calibration[validation] = partitions['validation_scores']
        if validation_cut(calibration, data['y'], validation, data['process']) != model['threshold']:
            raise ValueError('Saved cut differs from its own inner-validation genuine SM scores')
        model_path = fold_dir/'model.pkl'
        if digest(model_path) != model['model_sha256']:
            raise ValueError('Model changed after cross-fit scoring')
    return prediction, receipt


def paired_metric(data, mask, outcomes, replicates=200, seed=711926):
    """Paired two-level Poisson bootstrap of source groups and original events.

    Raw pair means and native-weighted means are separate. Reweighted responses
    and several pairs sharing an original event receive one event multiplier.
    Wilson envelopes on original-event count prevent false boundary precision;
    one source cannot establish between-source uncertainty.
    """
    indices = np.flatnonzero(mask)
    if not len(indices):
        return dict(raw_pairs=0, independent_event_owners=0, source_groups=0,
                    baseline=None, candidate=None, paired_difference=None, unknown=True)
    owners, first, reverse = np.unique(data['event_owner_id'][indices], return_index=True, return_inverse=True)
    group_names, event_group = np.unique(data['group_id'][indices[first]], return_inverse=True)
    counts = np.bincount(reverse); native = np.asarray(data['weight'][indices], dtype=float)
    event_weight = np.bincount(reverse, weights=native)
    passed = np.vstack([np.bincount(reverse, weights=np.asarray(value)[indices]) for value in outcomes])
    weighted = np.vstack([np.bincount(reverse, weights=native*np.asarray(value)[indices]) for value in outcomes])
    raw_eff = passed.sum(axis=1)/len(indices); sumw = float(native.sum())
    weight_eff = weighted.sum(axis=1)/sumw if sumw else np.full(2, np.nan)
    rng = np.random.default_rng(seed); boot = []; weighted_boot = []
    for _ in range(replicates):
        draw = rng.poisson(1., len(group_names))[event_group]*rng.poisson(1., len(owners))
        denominator = draw@counts; weighted_denominator = draw@event_weight
        if denominator:
            boot.append(passed@draw/denominator)
        if weighted_denominator:
            weighted_boot.append(weighted@draw/weighted_denominator)
    intervals = []
    for i, efficiency in enumerate(raw_eff):
        boundary = wilson(float(efficiency), len(owners))
        sampled = np.quantile(np.asarray(boot)[:, i], [.025, .975]) if boot else boundary
        intervals.append([min(boundary[0], float(sampled[0])), max(boundary[1], float(sampled[1]))])
    delta = float(raw_eff[1]-raw_eff[0])
    delta_ci = np.quantile(np.diff(np.asarray(boot), axis=1)[:, 0], [.025, .975]).tolist() if boot else [-1., 1.]
    if delta_ci[0] == delta_ci[1]:
        # No observed disagreement does not bound unseen disagreements to zero.
        width = wilson(0., len(owners))[1]
        delta_ci = [max(-1., delta-width), min(1., delta+width)]
    rows = []
    for i in range(2):
        weighted_ci = np.quantile(np.asarray(weighted_boot)[:, i], [.025, .975]).tolist() if weighted_boot else [None, None]
        if weighted_ci[0] == weighted_ci[1] and weighted_ci[0] is not None:
            weighted_ci = [None, None]
        rows.append(dict(raw_pair_fraction=float(raw_eff[i]), ci95=intervals[i],
                         native_weighted_fraction=float(weight_eff[i]) if np.isfinite(weight_eff[i]) else None,
                         native_weighted_bootstrap_ci95=weighted_ci))
    sumw2_events = float(event_weight@event_weight)
    weighted_delta = float(weight_eff[1]-weight_eff[0]) if np.isfinite(weight_eff).all() else None
    weighted_delta_ci = np.quantile(np.diff(np.asarray(weighted_boot), axis=1)[:, 0], [.025, .975]).tolist() if weighted_boot else [None, None]
    if weighted_delta_ci[0] == weighted_delta_ci[1]:
        weighted_delta_ci = [None, None]
    return dict(raw_pairs=len(indices), independent_event_owners=len(owners), source_groups=len(group_names),
                native_sumw=sumw, native_pair_sumw2=float(native@native),
                native_event_owner_sumw2=sumw2_events,
                native_effective_event_owners=sumw*sumw/sumw2_events if sumw2_events else None,
                signed_native_weights=bool(np.any(native<0)), baseline=rows[0], candidate=rows[1],
                paired_difference=dict(candidate_minus_baseline=delta, ci95=delta_ci,
                                       native_weighted_candidate_minus_baseline=weighted_delta,
                                       native_weighted_bootstrap_ci95=weighted_delta_ci),
                supported_event_count=len(owners)>=30, source_variation_measurable=len(group_names)>=2,
                unknown=False)


def empirical_roc(y, score):
    y, score = np.asarray(y), np.asarray(score)
    genuine, fake = int(np.sum(y==1)), int(np.sum(y==0))
    if not genuine or not fake:
        return dict(genuine_pairs=genuine, fake_pairs=fake, auc=None, fpr=[], tpr=[])
    order = np.argsort(-score, kind='stable'); sorted_y = y[order]
    last = np.r_[np.flatnonzero(np.diff(score[order])), len(order)-1]
    tpr = np.r_[0., np.cumsum(sorted_y==1)[last]/genuine]
    fpr = np.r_[0., np.cumsum(sorted_y==0)[last]/fake]
    auc = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    return dict(genuine_pairs=genuine, fake_pairs=fake, auc=auc, fpr=fpr.tolist(), tpr=tpr.tolist())


def compare(data, predictions, replicates=200, inventory_physical_points=None):
    passes = [p['crossfit_pass'] for p in predictions]; y = data['y']; process = data['process']
    sm = np.isin(process, SM_PROCESSES); bsm = ~sm
    domain = data.get('domain', np.full(len(y), 'unspecified'))
    current = np.array([str(value).lower().startswith('current') or str(value)=='recent' for value in domain])
    historical = np.array([str(value).lower().startswith('historical') for value in domain])
    current_bsm = bsm&current; old_bsm = bsm&~current
    views = {'pooled_SM': sm, 'current_SM': sm&current, 'historical_SM': sm&historical,
             'current_BSM': current_bsm, 'historical_BSM_control': old_bsm, 'pooled_BSM_diagnostic': bsm}
    views.update({'process:'+str(value): process==value for value in np.unique(process)})
    views.update({'category:'+str(value): sm&(data['category']==value) for value in np.unique(data['category'])})
    views.update({'domain:'+str(value): domain==value for value in np.unique(domain)})
    def metrics(mask):
        return {name: paired_metric(data, mask&(y==label), outcomes, replicates)
                for label, name, outcomes in ((1, 'genuine_retention', passes),
                                              (0, 'fake_rejection', [~p for p in passes]),
                                              (-1, 'unknown_retention', passes))}
    report = dict(views={name: metrics(mask) for name, mask in views.items()}, roc={}, signal_points={},
                  pooled_signal_points_diagnostic={}, historical_signal_control_points={})
    for name in ('current_SM', 'pooled_SM'):
        mask = views[name]&(y>=0)
        report['roc'][name] = dict(
            composition={str(value): {str(label): int(np.sum(mask&(domain==value)&(y==label))) for label in (0, 1)}
                         for value in np.unique(domain[mask])},
            models={name: empirical_roc(y[mask], prediction['crossfit_score'][mask])
                    for name, prediction in zip(('baseline', 'candidate'), predictions)})
    inventory_points = []
    for value in inventory_physical_points or []:
        point = value if isinstance(value, str) else value.get('point', value.get('name'))
        if not isinstance(point, str) or not point:
            raise ValueError('Inventory physical points must be nonempty names')
        inventory_points.append(point)
    current_points = sorted(set(inventory_points) | set(map(str, data['point'][current_bsm])))
    for point in current_points:
        mask = current_bsm&(data['point']==point); item = metrics(mask)
        item['coverage'] = dict(input_candidate_pairs=int(mask.sum()), inventory_member=point in inventory_points,
            genuine_retention_status='measured_on_reconstructed_pairs' if np.any(mask&(y==1)) else 'unknown_no_genuine_reconstructed_support',
            absent_from_candidate_rows=not bool(mask.any()))
        report['signal_points'][point] = item
    for point in np.unique(data['point'][bsm]):
        report['pooled_signal_points_diagnostic'][str(point)] = metrics(bsm&(data['point']==point))
    for point in np.unique(data['point'][old_bsm]):
        report['historical_signal_control_points'][str(point)] = metrics(old_bsm&(data['point']==point))
    report['signal_point_census'] = dict(primary_population='Current BSM only; historical controls excluded',
        declared_inventory_points=len(set(inventory_points)), current_points_with_candidate_rows=len(np.unique(data['point'][current_bsm])),
        reported_current_points=len(current_points),
        declared_points_without_candidate_rows=sorted(set(inventory_points)-set(map(str,data['point'][current_bsm]))),
        absent_point_interpretation='Unknown conditional retention, not zero efficiency or zero physical acceptance')
    names = list(map(str, data['feature_names']))
    if all('pair_'+key in names for key in ('vx', 'vy', 'vz')):
        vertices = data['X'][:, [names.index('pair_'+key) for key in ('vx', 'vy', 'vz')]]
        flight = np.linalg.norm(vertices-np.array([0., 0., 14800.]), axis=1)/100.
        mass_edges = [0., .3, 1., 3., 10., 30., 100., np.inf]
        flight_edges = [0., .1, 1., 5., 30., 100., np.inf]
        for key, population in (('signal_mass_flight', current_bsm), ('pooled_signal_mass_flight_diagnostic', bsm)):
            cells = []
            for lo, hi in zip(mass_edges[:-1], mass_edges[1:]):
                for a, b in zip(flight_edges[:-1], flight_edges[1:]):
                    mask = population&(data['mass']>=lo)&(data['mass']<hi)&(flight>=a)&(flight<b)
                    cells.append(dict(reco_mass_gev=[lo, hi if np.isfinite(hi) else None],
                                      reco_flight_m=[a, b if np.isfinite(b) else None], metrics=metrics(mask)))
            report[key] = dict(population='current_BSM' if key=='signal_mass_flight' else 'pooled_BSM_diagnostic',
                              coordinates='Reconstructed mass and distance to nominal PV (0,0,14800)cm', cells=cells)
    direct_key = next((key for key in ('direct443', 'truth_direct_jpsi') if key in data), None)
    report['direct_Jpsi_window'] = (dict(truth_flag=direct_key, inclusive_reco_mass_gev=[2.6, 3.6],
        metrics=metrics((process=='jpsi')&(np.asarray(data[direct_key])==1)&(data['mass']>=2.6)&(data['mass']<=3.6)))
        if direct_key else dict(qualified=False, reason='No explicit direct-Jpsi truth ancestry flag supplied; common-origin labels do not establish direct Jpsi'))
    return report


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = ('#0072b2', '#d55e00'); model_names = ('SM-only', 'SM+BSM')
    with PdfPages(output) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        for ax, population in zip(axes, ('current_SM', 'pooled_SM')):
            for i, key in enumerate(('baseline', 'candidate')):
                curve = report['roc'][population]['models'][key]
                ax.plot(curve['fpr'], curve['tpr'], color=colors[i],
                        label=model_names[i]+(' AUC %.3f'%curve['auc'] if curve['auc'] is not None else ' unmeasured'))
            curve = report['roc'][population]['models']['baseline']
            ax.set(title=f"{population}: {curve['genuine_pairs']} genuine / {curve['fake_pairs']} fake raw pairs",
                   xlabel='Fake-pair acceptance', ylabel='Genuine-pair retention', xlim=(0, 1), ylim=(0, 1)); ax.grid(alpha=.2); ax.legend()
            composition = report['roc'][population]['composition']
            ax.text(.03, .03, '\n'.join(str(k)+': G='+str(v['1'])+', F='+str(v['0']) for k, v in composition.items()),
                    transform=ax.transAxes, fontsize=7, va='bottom')
        fig.suptitle('Exploratory source-grouped cross-fit; empirical ROC, no luminosity normalization')
        pdf.savefig(fig); plt.close(fig)
        names = ['current_SM', 'historical_SM', 'pooled_SM']+[name for name in report['views'] if name.startswith('category:')]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
        for ax, metric, title in zip(axes, ('genuine_retention', 'fake_rejection'), ('Genuine retention', 'Fake rejection')):
            for i, key in enumerate(('baseline', 'candidate')):
                for x, name in enumerate(names):
                    entry = report['views'][name][metric]
                    if entry[key] is None:
                        continue
                    value = entry[key]; lo, hi = value['ci95']; fraction = value['raw_pair_fraction']
                    ax.errorbar(x+(i-.5)*.16, fraction, yerr=[[fraction-lo], [hi-fraction]], fmt='o',
                                color=colors[i], label=model_names[i] if x==0 else None)
            ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
            ax.set(title=title, ylabel='Raw-pair fraction', ylim=(-.03, 1.03)); ax.grid(alpha=.2); ax.legend()
        fig.suptitle('Saved fold-specific inner-validation cuts; paired source/event bootstrap with finite-count envelopes')
        pdf.savefig(fig); plt.close(fig)
        points = list(report['signal_points'])
        for start in range(0, len(points), 30):
            subset = points[start:start+30]; fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
            for i, key in enumerate(('baseline', 'candidate')):
                first_visible = next((x for x, point in enumerate(subset)
                                      if report['signal_points'][point]['genuine_retention'][key] is not None), None)
                for x, point in enumerate(subset):
                    entry = report['signal_points'][point]['genuine_retention']; value = entry[key]
                    if value is None:
                        continue
                    lo, hi = value['ci95']; fraction = value['raw_pair_fraction']
                    ax.errorbar(x+(i-.5)*.16, fraction, yerr=[[fraction-lo], [hi-fraction]], fmt='o',
                                color=colors[i] if entry['supported_event_count'] else '#888888',
                                label=model_names[i] if x==first_visible else None)
            labels = [point+' (N='+str(report['signal_points'][point]['genuine_retention']['independent_event_owners'])+')' for point in subset]
            ax.set_xticks(range(len(subset)), labels, rotation=90, fontsize=6)
            ax.set(title='Current BSM physical points; gray: fewer than30 original genuine events',
                   xlabel='Missing markers: no genuine reconstructed support; retention unknown',
                   ylabel='Conditional genuine-pair retention', ylim=(-.03, 1.03)); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend()
            pdf.savefig(fig); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'sm-only', 'sm-bsm', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cohort-receipt', type=Path)
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args(); data = dict(np.load(args.input, allow_pickle=False)); validate_data(data)
    if args.bootstrap < 20:
        parser.error('At least20 paired bootstrap replicates are required')
    loaded = [verified_predictions(data, digest(args.input), directory) for directory in (args.sm_only, args.sm_bsm)]
    if [receipt['fit_mode'] for _, receipt in loaded] != ['sm-only', 'sm-bsm']:
        raise ValueError('Expected one SM-only baseline and one SM+BSM candidate')
    cohort = json.loads(args.cohort_receipt.read_text()) if args.cohort_receipt else {}
    report = compare(data, [prediction for prediction, _ in loaded], args.bootstrap, cohort.get('inventory_physical_points'))
    report.update(schema='shift-vertex-crossfit-comparison-v1', input_sha256=digest(args.input),
                  simulation_only=True, deployment_ready=False,
                  inference='Every row scored by the opposite original-ownership half only',
                  threshold='Each saved model uses its own inner-validation genuine-SM cut; none fitted on scored folds',
                  metric_interpretation='Raw pairs, original-event owners and native-weight diagnostics; no events normalized to lumi, physical yields or full-acceptance claim',
                  bootstrap='Paired source-group Poisson multiplier times original-event Poisson multiplier; same draws for both models; finite-event boundary envelopes',
                  limitations=['Previously inspected development populations; cross-fit does not create a new blind test.',
                               'Unknown labels remain unknown; common-origin truth does not certify fitted vertex accuracy.',
                               'Sparse BSM points or mass/flight cells do not establish zero rate or universal transfer.'])
    if args.cohort_receipt:
        compact = {key: value for key, value in cohort.items()
                   if value is None or isinstance(value, (str, int, float, bool)) or key in ('counts', 'notes', 'limitations')}
        report['cohort_receipt'] = dict(path=str(args.cohort_receipt.resolve()), sha256=digest(args.cohort_receipt),
                                        summary=compact)
    args.output.mkdir(parents=True, exist_ok=False)
    render(report, args.output/'classifier_comparison.pdf')
    report['pdf_sha256'] = digest(args.output/'classifier_comparison.pdf')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True, report=str(args.output/'report.json'), pdf=str(args.output/'classifier_comparison.pdf'))))


if __name__ == '__main__':
    main()
