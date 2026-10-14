#!/usr/bin/env python3
# Code style: AI-code. AI-generated fixed-classifier signal response diagnostic.
"""Join scored complete signal populations to authoritative timed GEN audits.

Scores and cuts are supplied by an existing locked evaluation. This utility
does not train, choose a cut, inspect collision data, or submit production.
Missing detector populations remain unknown. Proper lengths come from GEN,
never from reconstructed position or a mean laboratory flight label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from shift_vertex_efficiency import summarize_vertex_efficiency


LENGTH_EDGES_MM = (0., 25., 50., 100., 200., 400., 800., 1600., float('inf'))
LOCAL_LIFETIME_FACTORS = (.5, .75, 1., 1.25, 1.5)
MODEL_LABELS = {
    'info_global_rich_s71': 'Rich DNN, seed 71',
    'info_global_rich_s72': 'Rich DNN, seed 72',
    'info_shared_rich_accept10_s71': 'Strong DNN, seed 71',
    'info_shared_rich_accept10_s72': 'Strong DNN, seed 72',
    'dnn_r4_l3_c3_e0_h64_s71': 'Previous strict DNN',
    'reference_uniform_bdt_3p0_s71': 'Current BDT',
    'expanded_d3_s0_n400': 'Strong BDT',
    'expanded_d3_s3_n400': 'Flat BDT',
}
SHOWN_MODELS = ('info_global_rich_s71', 'info_global_rich_s72',
                'info_shared_rich_accept10_s72', 'expanded_d3_s0_n400',
                'expanded_d3_s3_n400')


def reference(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def verify(ref):
    if reference(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed input artifact: ' + ref['path'])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def align_gen(point, inventory, audit, contract, ids, weights):
    """Require full detector coverage and one timed signal decay per parent."""
    rows = audit['event_rows']
    expected = np.asarray(inventory['event_ids'], dtype=np.uint64)
    audited = np.asarray([row['id'] for row in rows], dtype=np.uint64)
    if (inventory['events'] != len(ids) or audit['events'] != len(ids) or
            audit.get('runtime_validated') is not True or
            not np.array_equal(ids, expected) or not np.array_equal(ids, audited)):
        raise ValueError('Incomplete or changed full GEN population: ' + point)
    native = np.asarray([row['weight'] for row in rows], dtype=float)
    if (not np.array_equal(native, weights) or not np.all(native == 1.) or
            audit['sum_weights'] != len(ids) or audit['sum_weights_squared'] != len(ids) or
            inventory['native_sumw'] != len(ids) or inventory['native_sumw2'] != len(ids)):
        raise ValueError('This lifetime estimator requires exact native unit weights: ' + point)
    ctau = float(inventory['proper_ctau_mm'])
    if (ctau <= 0 or not np.isfinite(ctau) or
            not np.isclose(ctau, contract['width_authority']['ctau_mm'], rtol=1e-10, atol=0) or
            not np.isclose(ctau, audit['expected_proper_length_mm'], rtol=1e-10, atol=0) or
            contract.get('filtering') != 'none'):
        raise ValueError('Proper lifetime or unfiltered generation contract differs: ' + point)
    lengths = []
    for row in rows:
        if len(row['signal_decays']) != 1:
            raise ValueError('Exactly one audited signal decay per parent required: ' + point)
        decay = row['signal_decays'][0]
        value = float(decay['proper_length_mm'])
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid GEN proper length: ' + point)
        lo, hi = contract['generated_mass_support_gev']
        if not lo <= decay['mass_gev'] <= hi:
            raise ValueError('GEN signal mass outside its declared support: ' + point)
        lengths.append(value)
    return np.asarray(lengths)


def event_masks(ids, pair_ids, genuine, values, cut):
    lookup = {tuple(identity): index for index, identity in enumerate(ids)}
    baseline, selected = np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=bool)
    for identity, valid, value in zip(pair_ids, genuine, values):
        parent = lookup[tuple(identity)]
        if valid:
            baseline[parent] = True
            if value >= cut:
                selected[parent] = True
    return baseline, selected


def precision_forecast(efficiency, retention, relative_goal=.1, absolute_goal=.03):
    """Planning approximations only; no confidence or future-count guarantee."""
    return {
        'relative_full_response_goal': relative_goal,
        'gen_for_relative_response_se':
            int(np.ceil((1 - efficiency) / (efficiency * relative_goal**2))) if efficiency else None,
        'conditional_retention_95_halfwidth_goal': absolute_goal,
        'genuine_for_retention_95_halfwidth':
            int(np.ceil(1.959963984540054**2 * retention * (1 - retention) / absolute_goal**2))
            if retention is not None and 0 < retention < 1 else None,
        'interpretation': 'Plug-in binomial planning estimates; boundary retention requires a conservative planning value and remains undefined here.',
    }


def make_report(cache, signal_report, production, selected_models):
    metadata = read(cache / 'inputs_resolved.json')
    if (metadata.get('complete') is not True or metadata.get('all_parent_events_retained') is not True or
            metadata.get('sample_role') != 'validation_only'):
        raise ValueError('A complete validation-only signal cache is required')
    for ref in metadata['cache'].values():
        verify(ref)
    reco, labels, events = [load(metadata['cache'][key]['path']) for key in ('reco', 'labels', 'events')]
    for key in ('ids', 'pair_index', 'source_index', 'point'):
        if key not in reco:
            raise ValueError('Missing candidate identity column: ' + key)
    report = read(signal_report)
    if (report.get('complete') is not True or report.get('selection_uses_signal') is not False or
            report.get('model_selection_allowed') is not False or report.get('thresholds_unchanged') is not True):
        raise ValueError('Existing complete evaluation must preserve signal-independent model choices and cuts')
    for key in ('selection', 'protocol', 'cache'):
        verify(report[key])
    verify(report['artifacts']['scores'])
    if reference(cache / 'inputs_resolved.json')['sha256'] != report['cache']['sha256']:
        raise ValueError('Scored population and cache manifest differ')
    scores = load(report['artifacts']['scores']['path'])
    for key in ('ids', 'pair_index', 'source_index', 'point'):
        if not np.array_equal(scores[key], reco[key]):
            raise ValueError('Score row identity/order differs: ' + key)
    if (len(labels['y']) != len(reco['ids']) or
            not np.array_equal(labels['genuine_signal_associated'],
                               (labels['y'] == 1) & labels['signal_associated'])):
        raise ValueError('Signal label contract differs')
    for key in ('signal_associated', 'genuine_signal_associated'):
        if labels[key].dtype != bool or labels[key].shape != (len(reco['ids']),):
            raise ValueError('Signal association flags must be aligned boolean vectors')
    inventory = read(production / 'gen_inventory.json')
    plan = read(production / 'plan.json')
    plans = {row['point']: row for row in plan['points']}
    if len(plans) != len(plan['points']) or inventory.get('complete') is not True:
        raise ValueError('Complete unique GEN plan and inventory required')
    if not set(events['point']).issubset(plans):
        raise ValueError('Cached physical point is absent from GEN plan')
    if set(selected_models) - set(report['models']):
        raise ValueError('Model has no existing frozen signal evaluation')
    cuts = {name: report['models'][name]['threshold'] for name in selected_models}
    verify(report['previous_signal_transfer'])
    previous = read(report['previous_signal_transfer']['path'])
    verify(previous['diagnostic_selection'])
    previous_lock = read(previous['diagnostic_selection']['path'])
    verify(previous_lock['previous_bdt_selection'])
    bdt_lock = read(previous_lock['previous_bdt_selection']['path'])
    current_lock = read(report['selection']['path'])
    if any(lock.get('selection_uses_test') is not False for lock in (current_lock, previous_lock, bdt_lock)):
        raise ValueError('Model selection used development test')
    for name in selected_models:
        lock = current_lock if name.startswith('info_') else bdt_lock if name in bdt_lock['thresholds'] else previous_lock
        if name not in lock['thresholds'] or cuts[name] != lock['thresholds'][name]:
            raise ValueError('Score threshold differs from its locked authority: ' + name)
        if name not in scores or scores[name].shape != (len(reco['ids']),) or not np.isfinite(scores[name]).all():
            raise ValueError('Missing or invalid score column: ' + name)
        model_ref = report['model_references'][name]
        if 'path' in model_ref:
            verify(model_ref)
        else:
            for ref in model_ref.values():
                verify(ref)
    seal, submission = read(production / 'seal.json'), read(production / 'submission_receipt.json')
    if reference(production / 'seal.json')['sha256'] != submission['seal_sha256']:
        raise ValueError('GEN production seal differs from submission authority')
    for filename in ('plan.json', 'gen_inventory.json'):
        path = (production / filename).resolve()
        if seal['files'].get(str(path)) != reference(path)['sha256']:
            raise ValueError('GEN plan/inventory differs from sealed production: ' + filename)
    if metadata['schema'] == 'shift-rich-reconstructed-information-v1':
        verify(metadata['adapter_original_manifest'])
        original = read(metadata['adapter_original_manifest']['path'])
        verify(original['protocol'])
        extraction = read(original['protocol']['path'])
    elif metadata['schema'] == 'shift-complete-signal-rich-cache-v1':
        verify(metadata['protocol'])
        extraction = read(metadata['protocol']['path'])
    else:
        raise ValueError('Unsupported complete signal cache schema')
    complete_inputs = {row['point']: row for row in extraction['complete_point_inputs']}
    if set(complete_inputs) != set(events['point']):
        raise ValueError('Full-parent processing authorities differ from cache points')
    for point, row in complete_inputs.items():
        verify(row['source_receipt'])
        receipt = read(row['source_receipt']['path'])
        if (receipt.get('complete') is not True or receipt['point'] != point or
                receipt['events'] != int(np.sum(events['point'] == point))):
            raise ValueError('Detector point processing is incomplete: ' + point)
    # Existing estimator supplies full-GEN importance sampling, moment errors,
    # normalization closure and support checks. No ratio renormalization here.
    scripts = production.parents[2] / 'shift_cmssw_workflow/scripts'
    sys.path.insert(0, str(scripts))
    from dark_photon_model import HBAR_C_GEV_MM
    from dark_photon_scan import LifetimeProposal, LifetimeEvent, estimate_lifetime_acceptance
    points, references = {}, [reference(production / name) for name in
                             ('plan.json', 'gen_inventory.json', 'seal.json', 'submission_receipt.json')]
    references.extend(row['source_receipt'] for row in complete_inputs.values())
    for point, spec in plans.items():
        emask, pmask = events['point'] == point, reco['point'] == point
        authority = inventory['points'][point]
        entry = {key: spec[key] for key in ('mass_gev', 'epsilon', 'proper_ctau_mm', 'events')}
        entry.update(point=point, target_mean_lab_flight_m=spec.get('target_mean_lab_flight_m'),
                     response_available=bool(emask.any()), models={})
        if not emask.any():
            entry['unknown_GEN_response_events'] = spec['events']
            points[point] = entry
            continue
        gen = Path(spec['gen_directory'])
        refs = {name: reference(gen / (name + '.json')) for name in ('validation', 'contract')}
        for name, ref in refs.items():
            if ref['sha256'] != authority[name + '_sha256']:
                raise ValueError('GEN authority changed: ' + point + '/' + name)
        references.extend(refs.values())
        contract, audit = read(refs['contract']['path']), read(refs['validation']['path'])
        if any(spec[key] != authority[key] for key in ('mass_gev', 'epsilon', 'proper_ctau_mm', 'events')):
            raise ValueError('Plan and inventory physical point differ: ' + point)
        lengths = align_gen(point, authority, audit, contract, events['ids'][emask], events['native_weight'][emask])
        pair_ids, y = reco['ids'][pmask], labels['y'][pmask]
        genuine = labels['genuine_signal_associated'][pmask]
        parent_ids, weights = events['ids'][emask], events['native_weight'][emask]
        observed = np.asarray([np.sum(np.all(pair_ids == identity, axis=1)) for identity in parent_ids])
        if not np.array_equal(observed, events['pair_count'][emask]):
            raise ValueError('Candidate multiplicities differ from full-parent cache: ' + point)
        lookup = {tuple(identity): index for index, identity in enumerate(parent_ids)}
        parent_index = np.asarray([lookup[tuple(identity)] for identity in pair_ids], dtype=int)
        if not np.array_equal(reco['native_weight'][pmask], weights[parent_index]):
            raise ValueError('Candidate and parent weights differ: ' + point)
        entry.update(unknown_GEN_response_events=0, proper_length_source=refs['validation'],
                     parents_without_any_candidate=int(np.sum(observed == 0)),
                     reconstructed_pairs=int(pmask.sum()),
                     measured_mean_proper_length_mm=float(np.mean(lengths)))
        for name in selected_models:
            values = scores[name][pmask]
            result = summarize_vertex_efficiency(parent_ids, weights, pair_ids, y, values, cuts[name],
                                                 qualified_genuine_mask=genuine)
            baseline, accepted = event_masks(parent_ids, pair_ids, genuine, values, cuts[name])
            _, possibly_accepted = event_masks(parent_ids, pair_ids, genuine | (y == -1), values, cuts[name])
            result['unknown_truth_bounds'] = {
                'known_signal_selected_events': int(accepted.sum()),
                'possible_signal_selected_events': int(possibly_accepted.sum()),
                'full_GEN_events': len(parent_ids),
                'response_lower': float(accepted.mean()),
                'response_upper': float(possibly_accepted.mean()),
                'interpretation': 'Deterministic label bounds, not confidence intervals: unknown candidates could all be signal or none; known fake/non-signal genuine pairs are excluded.',
            }
            bins = []
            # Realized proper decay length is distinct from the model mean c-tau.
            for lo, hi in zip(LENGTH_EDGES_MM[:-1], LENGTH_EDGES_MM[1:]):
                selected = (lengths >= lo) & (lengths < hi)
                pair_selected = selected[parent_index]
                stats = summarize_vertex_efficiency(parent_ids[selected], weights[selected],
                    pair_ids[pair_selected], y[pair_selected], values[pair_selected], cuts[name],
                    qualified_genuine_mask=genuine[pair_selected])
                bins.append({'lo_mm': lo, 'hi_mm': hi if np.isfinite(hi) else None, 'response': stats})
            result['proper_length_bins'] = bins
            result['categories'] = {}
            for category in sorted(set(reco['topology'][pmask])):
                subset = reco['topology'][pmask] == category
                result['categories'][str(category)] = summarize_vertex_efficiency(
                    parent_ids, weights, pair_ids[subset], y[subset], values[subset], cuts[name],
                    qualified_genuine_mask=genuine[subset])
            # Each original independently fixed anchor is used separately.
            # Incomplete points never become zero-count mixture strata.
            context = reference(gen / 'contract.json')['sha256']
            proposal = LifetimeProposal((float(spec['proper_ctau_mm']),), (len(parent_ids),), context)
            lifetime_events = [LifetimeEvent(':'.join(map(str, identity)), float(length), 0,
                                            bool(passed), context)
                               for identity, length, passed in zip(parent_ids, lengths, accepted)]
            result['local_lifetime_estimates'] = [estimate_lifetime_acceptance(proposal, lifetime_events,
                HBAR_C_GEV_MM / (factor * spec['proper_ctau_mm']), len(parent_ids))
                for factor in LOCAL_LIFETIME_FACTORS]
            result['reweighting_role'] = 'Development lifetime-response diagnostic; no production/BR or data correction.'
            empirical = result['full_parent_selected']['empirical']['efficiency']
            retention = result['conditional_event_retention']['empirical']['efficiency']
            result['planning'] = precision_forecast(empirical, retention)
            result['selected_signal_event_ids'] = parent_ids[accepted].tolist()
            entry['models'][name] = result
        points[point] = entry
    return {'schema': 'shift-vertex-full-parent-response-v1', 'points': points,
            'models': selected_models, 'thresholds': cuts, 'references': references,
            'complete_response_points': sum(row['response_available'] for row in points.values()),
            'planned_points': len(points), 'all_parents': len(events['ids']),
            'pairs': len(reco['ids']), 'physics_ready': False, 'selection_uses_signal': False,
            'recording_acceptance_included': False, 'data_accessed': False,
            'stage_definition': 'Unfiltered GEN to any qualified signal pair in unchanged no-pileup/Fake2-HLT replay, then fixed score cut.',
            'limitations': ['Previously inspected development signals; no independent confirmation.',
                'No measured real recording acceptance, source exposure, or physical signal rate.',
                'Unknown and genuine non-signal pairs remain explicit; only qualified signal-associated genuine pairs select a signal event.',
                'Mean proper lifetime anchors are irregular across masses; no mass interpolation.',
                'Local lifetime reweighting holds production/kinematics/decay/response fixed; physical coupling/rate transfer needs separate closure.',
                'Category events may overlap; category responses must not be added blindly.']}


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    points = sorted(report['points'], key=lambda name: (report['points'][name]['mass_gev'],
                                                      report['points'][name]['proper_ctau_mm']))
    models = [name for name in SHOWN_MODELS if name in report['models']]
    colors = ('#0072b2', '#56b4e9', '#d55e00', '#222222', '#009e73')
    labels = [f"{report['points'][name]['mass_gev']:g} GeV\n"
              f"cτ={report['points'][name]['proper_ctau_mm']:.3g} mm" for name in points]
    note = 'Complete GEN denominators only. Missing points are unknown. No-pileup/Fake2-HLT replay; real recording acceptance and data corrections unresolved.'
    with PdfPages(output / 'mass_proper_lifetime_response.pdf') as pdf:
        for field, title in [('full_parent_selected', 'Full generated-event response after the score cut'),
                             ('conditional_event_retention', 'Score-cut retention among events with a qualified signal pair')]:
            fig, ax = plt.subplots(figsize=(13.8, 6.3))
            largest_upper_percent = 0.
            for offset, (name, color) in enumerate(zip(models, colors)):
                rows = [(i, report['points'][point]['models'][name][field]['empirical'])
                        for i, point in enumerate(points) if report['points'][point]['response_available']
                        and report['points'][point]['models'][name][field]['empirical']['efficiency'] is not None]
                if not rows:
                    continue
                x = np.asarray([i for i, _ in rows]) + (offset - (len(models) - 1) / 2) * .12
                y = np.asarray([row['efficiency'] for _, row in rows])
                bounds = np.asarray([row['wilson95'] for _, row in rows])
                largest_upper_percent = max(largest_upper_percent, 100*float(bounds[:, 1].max()))
                ax.errorbar(x, 100*y, yerr=100*np.maximum(0, [y - bounds[:, 0], bounds[:, 1] - y]),
                            fmt='o', capsize=2, color=color, label=MODEL_LABELS.get(name, name))
            for i, point in enumerate(points):
                if not report['points'][point]['response_available']:
                    ax.axvspan(i-.45, i+.45, color='gray', alpha=.14)
                    ax.text(i, .5, 'unknown', ha='center', rotation=90, transform=ax.get_xaxis_transform())
                elif field == 'conditional_event_retention' and all(
                        report['points'][point]['models'][name][field]['empirical']['total'] == 0 for name in models):
                    ax.text(i, .5, 'no qualified pair', ha='center', rotation=90,
                            transform=ax.get_xaxis_transform())
            ax.set_xticks(range(len(points)), labels, fontsize=9)
            ax.set_ylabel('Selected events (%)' if field == 'full_parent_selected' else 'Retained events (%)')
            upper = 102 if field == 'conditional_event_retention' else min(
                102, max(30, 5*np.ceil(1.1*largest_upper_percent / 5)))
            ax.set_ylim(0, upper)
            ax.set_title(title + '\nFrozen validation cuts; empirical Wilson 95% intervals')
            ax.grid(axis='y', alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=9, loc='lower left' if field != 'full_parent_selected' else 'upper right')
            fig.text(.03, .015, note, fontsize=8); fig.tight_layout(rect=(0, .045, 1, 1)); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
        for ax, field, title in [(axes[0], 'full_parent_baseline', 'Before the classifier'),
                                 (axes[1], 'full_parent_selected', 'After rich DNN, seed 71')]:
            available = [report['points'][p] for p in points if report['points'][p]['response_available']]
            vals = [100*p['models'][models[0]][field]['empirical']['efficiency'] for p in available]
            scatter = ax.scatter([p['mass_gev'] for p in available], [p['proper_ctau_mm'] for p in available],
                                 c=vals, cmap='viridis', vmin=0, vmax=25, s=150, edgecolors='black')
            for p, value in zip(available, vals):
                ax.annotate(f'{value:.1f}%', (p['mass_gev'], p['proper_ctau_mm']),
                            xytext=(9, 3), textcoords='offset points', fontsize=9)
            missing = [report['points'][p] for p in points if not report['points'][p]['response_available']]
            ax.scatter([p['mass_gev'] for p in missing], [p['proper_ctau_mm'] for p in missing],
                       marker='x', c='gray', s=70, label='Response unknown')
            ax.set_yscale('log'); ax.set_xlim(10, 59); ax.set_xticks([15, 30, 50]); ax.set_xlabel('Generated signal mass (GeV)')
            ax.set_ylabel('Model mean proper decay length cτ (mm)'); ax.set_title(title); ax.legend(fontsize=8)
            fig.colorbar(scatter, ax=ax, label='Full GEN response (%)')
        fig.suptitle('Measured anchors only; no interpolation across unknown mass or lifetime regions')
        fig.text(.03, .015, note, fontsize=8); fig.tight_layout(rect=(0, .055, 1, .94)); pdf.savefig(fig); plt.close(fig)
    displaced = [p for p in points if report['points'][p]['response_available'] and report['points'][p]['proper_ctau_mm'] > 1.]
    if not displaced:
        for filename in ('realized_proper_length_profiles.pdf', 'local_lifetime_reweighting.pdf'):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.axis('off')
            ax.text(.5, .55, 'No complete displaced signal point is available.', ha='center', fontsize=14)
            ax.text(.5, .35, 'Prompt response is shown at its measured anchor; displaced lifetime response remains unknown.',
                    ha='center', fontsize=10)
            fig.savefig(output / filename); plt.close(fig)
        return
    with PdfPages(output / 'realized_proper_length_profiles.pdf') as pdf:
        for point in displaced:
            entry = report['points'][point]
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
            for ax, field, title in [(axes[0], 'full_parent_selected', 'Full GEN response in each realized-length bin'),
                                     (axes[1], 'conditional_event_retention', 'Conditional score-cut retention')]:
                for offset, (name, color) in enumerate(zip(models, colors)):
                    rows = entry['models'][name]['proper_length_bins']
                    for i, row in enumerate(rows):
                        stats = row['response'][field]['empirical']
                        if stats['efficiency'] is None:
                            continue
                        y, interval = stats['efficiency'], stats['wilson95']
                        ax.errorbar(i + (offset - (len(models)-1)/2)*.1, 100*y,
                                    yerr=100*np.maximum(0, [[y-interval[0]], [interval[1]-y]]),
                                    fmt='o' if stats['total'] >= 20 else 'x', color=color, capsize=2,
                                    label=MODEL_LABELS.get(name, name) if i == 0 else None)
                ax.set_xticks(range(len(LENGTH_EDGES_MM)-1), ['0–25', '25–50', '50–100', '100–200',
                    '200–400', '400–800', '800–1600', '≥1600'], rotation=30, ha='right')
                ax.set_xlabel('Realized GEN proper decay length (mm)'); ax.set_ylabel('Events (%)')
                ax.set_ylim(0, 102); ax.set_title(title); ax.grid(axis='y', alpha=.2)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"{entry['mass_gev']:g} GeV, model cτ={entry['proper_ctau_mm']:.2f} mm; {entry['events']} complete parents")
            fig.text(.03, .02, 'Wilson 95% intervals. Crosses: fewer than 20 denominator events; empty bins omitted. This is realized proper length, not a new mean-lifetime sample.', fontsize=8)
            fig.tight_layout(rect=(0, .065, 1, .93)); pdf.savefig(fig); plt.close(fig)
    fig, axes = plt.subplots(1, len(displaced), figsize=(max(8, 4.4*len(displaced)), 5.8), squeeze=False)
    for ax, point in zip(axes[0], displaced):
        entry = report['points'][point]
        for name, color in zip(models, colors):
            rows = entry['models'][name]['local_lifetime_estimates']
            x = [row['target_proper_mean_length_mm'] for row in rows]
            y = [100*row['acceptance']['estimate'] for row in rows]
            err = [100*row['acceptance']['standard_error'] for row in rows]
            ax.plot(x, y, linestyle=':', color=color)
            for index, row in enumerate(rows):
                ax.errorbar(x[index], y[index], yerr=err[index],
                            fmt='o' if row['numerical_support_checks_passed'] else 'x',
                            color=color, capsize=2,
                            label=MODEL_LABELS.get(name, name) if index == 0 else None)
        ax.axvline(entry['proper_ctau_mm'], color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Target model cτ (mm)'); ax.set_ylabel('Estimated full GEN response (%)')
        ax.set_title(f"{entry['mass_gev']:g} GeV; single fixed anchor"); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Local lifetime importance sampling: development estimates with MC standard errors')
    fig.text(.03, .015, 'Each anchor uses all parents. Crosses fail precision/closure goals; circles pass numerical checks. Shared-parent errors are correlated; physical transfer needs independent validation.', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .94)); fig.savefig(output / 'local_lifetime_reweighting.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--signal-report', type=Path, required=True)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parents[1] / 'plots'):
        raise ValueError('Final outputs must be in this repository plots directory')
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source_snapshot'; source.mkdir()
    files = [Path(__file__), Path(__file__).with_name('shift_vertex_efficiency.py')]
    scripts = args.production.resolve().parents[2] / 'shift_cmssw_workflow/scripts'
    files.extend(scripts / name for name in ('dark_photon_scan.py', 'dark_photon_model.py'))
    for path in files:
        shutil.copyfile(path, source / path.name)
    signal = read(args.signal_report)
    models = list(signal['models'])
    protocol = {'schema': 'shift-vertex-response-diagnostic-protocol-v1',
        'inputs': [reference(path) for path in (args.cache / 'inputs_resolved.json', args.signal_report,
                    args.production / 'plan.json', args.production / 'gen_inventory.json')],
        'sources': [{**reference(source / path.name), 'original_path': str(path.resolve())}
                    for path in files], 'model_names': models,
        'models_and_thresholds_unchanged': True, 'selection_uses_signal': False,
        'proper_length_edges_mm': [x if np.isfinite(x) else None for x in LENGTH_EDGES_MM],
        'local_lifetime_factors': LOCAL_LIFETIME_FACTORS,
        'lifetime_precision': {'min_selected_ess': 100, 'max_relative_mc_error': .10},
        'empirical_intervals': 'Wilson 95%; native independent complete parents',
        'lifetime_estimator': 'Single original fixed anchor, sum(f_target/f_anchor * accepted)/N_GEN; no density renormalization.'}
    write(output / 'protocol.json', protocol)
    report = make_report(args.cache, args.signal_report, args.production, models)
    render(report, output)
    for path in files:
        if reference(path)['sha256'] != reference(source / path.name)['sha256']:
            raise ValueError('Diagnostic source changed during evaluation: ' + str(path))
    report['artifacts'] = {path.name: reference(path) for path in output.glob('*.pdf')}
    write(output / 'response.json', report)
    write(output / 'completion.json', {'complete': True, 'physics_ready': False,
        'protocol': reference(output / 'protocol.json'), 'response': reference(output / 'response.json'),
        'plots': report['artifacts'], 'source_snapshots': [reference(path) for path in source.iterdir()]})
    print(json.dumps({'output': str(output), 'complete_points': report['complete_response_points'],
                      'planned_points': report['planned_points'], 'parents': report['all_parents']}))


if __name__ == '__main__':
    main()
