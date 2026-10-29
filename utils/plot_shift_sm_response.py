#!/usr/bin/env python3
# Code style: AI-code. AI-generated plots of measured SM reconstruction response.
"""Plot raw original-event successes, without estimating physical event yields.

Input is the measured-response summary exported by the workflow study. Counts
from generator bins and frozen source folds are added before computing a
probability. These pooled probabilities describe that observed proposal mix,
not a cross-section-weighted mixture or a qualified production selection.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path


FEATURE = 'min_parent_prefix_depth'
FEATURE_LABELS = {
    FEATURE: ('known parent material',
              'Smallest known incoming OS-pair parent depth [interaction lengths]',
              'Known prefixes only: zero does not establish complete zero material.'),
    'best_pair_miss_cm': ('pointing', 'GEN pair miss distance [cm]',
                         'Straight-line pointing is a predictor; transport can change directions.'),
    'best_pair_min_energy_GeV': ('energy', 'Lower muon energy of the best-pointing OS pair [GeV]',
                              'Total energy, with no muon-pT requirement.'),
    'best_pair_max_column_loss_over_p': ('muon material and momentum',
        'Larger estimated material energy loss / total momentum in the best-pointing pair',
        'This material predictor is not a measured survival probability.'),
    'best_pair_birth_z_min_cm': ('birth position',
        'Smaller birth z of the best-pointing OS pair [cm]',
        'Unavailable features are shown separately, never interpreted as zero material.'),
}
CATEGORIES = ('both_both', 'both_both_good')
PROCESSES = ('qcd', 'dy', 'jpsi')
PROCESS_LABELS = {'qcd': 'QCD', 'dy': 'DY', 'jpsi': r'J/$\psi$'}


def wilson_interval(passed, count):
    """Two-sided 95% interval for raw independent-original event successes."""
    if not 0 <= passed <= count:
        raise ValueError('Invalid original-event counts')
    if count == 0:
        return None
    z = 1.959963984540054
    value = passed/count
    scale = 1.+z*z/count
    center = (value+z*z/(2*count))/scale
    half = z*math.sqrt(value*(1.-value)/count+z*z/(4*count*count))/scale
    low = 0. if passed == 0 else max(0., center-half)
    high = 1. if passed == count else min(1., center+half)
    return low, high


def pool_response(summary, feature=FEATURE, split=None):
    """Conserve N and K in every group; never average bin efficiencies."""
    if summary.get('schema') != 'shift-sm-measured-response-summary-v1':
        raise ValueError('Require the measured response summary schema')
    if summary.get('original_event_is_error_unit') is not True:
        raise ValueError('Pairs must already be grouped by original event')
    pooled = {process: dict(events=0, categories={category: dict(positive=0, pairs=0)
        for category in CATEGORIES}, cells=defaultdict(lambda: {
            category: dict(events=0, positive=0, pairs=0) for category in CATEGORIES}))
        for process in PROCESSES}
    edges = None
    selected_events = 0
    for group in summary['groups']:
        if split is not None and group['study_split'] != split:
            continue
        process = group['stratum'].split('_', 1)[0]
        if process not in pooled:
            raise ValueError('Unexpected SM process: '+process)
        axis = group['efficiency_maps'][feature]
        if edges is None:
            edges = axis['edges']
        if axis['edges'] != edges:
            raise ValueError('Cannot pool different material-depth axes')
        for category in CATEGORIES:
            cells = [cell[category] for cell in axis['cells'].values()]
            totals = group['totals'][category]
            for field in ('events', 'positive_originals', 'pairs'):
                if sum(cell[field] for cell in cells) != totals[field]:
                    raise ValueError('Feature cells do not cover group totals: '+field)
            if not 0 <= totals['positive_originals'] <= totals['events']:
                raise ValueError('Positive original count exceeds exposure')
            pooled[process]['categories'][category]['positive'] += totals['positive_originals']
            pooled[process]['categories'][category]['pairs'] += totals['pairs']
        pooled[process]['events'] += group['totals']['both_both']['events']
        selected_events += group['totals']['both_both']['events']
        for key, cell in axis['cells'].items():
            for category in CATEGORIES:
                result = pooled[process]['cells'][key][category]
                result['events'] += cell[category]['events']
                result['positive'] += cell[category]['positive_originals']
                result['pairs'] += cell[category]['pairs']
    expected_events = summary['events'] if split is None else selected_events
    if sum(row['events'] for row in pooled.values()) != expected_events:
        raise ValueError('Process totals do not cover the input response population')
    if edges is None:
        raise ValueError('No response groups match the requested split')
    for process in PROCESSES:
        pooled[process]['cells'] = dict(pooled[process]['cells'])
    return edges, pooled


def bin_labels(edges, keys):
    labels = {}
    for key in keys:
        if key == 'unknown':
            labels[key] = 'Unavailable'
        elif key == 'underflow':
            labels[key] = '<'+format(edges[0], 'g')
        elif key == 'overflow':
            labels[key] = r'$\geq$'+format(edges[-1], 'g')
        else:
            index = int(key)
            labels[key] = format(edges[index], 'g')+'–'+format(edges[index+1], 'g')
    return labels


def plot(summary, output, feature=FEATURE, split=None, pdf=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    edges, pooled = pool_response(summary, feature, split)
    title, xlabel, note = FEATURE_LABELS[feature]
    keys = [str(i) for i in range(len(edges)-1)]+['overflow', 'unknown']
    if any('underflow' in row['cells'] for row in pooled.values()):
        keys.insert(0, 'underflow')
    labels = bin_labels(edges, keys)
    x = np.arange(len(keys), dtype=float)
    colors = {'both_both': '#2468a0', 'both_both_good': '#d57828'}
    offsets = {'both_both': -.10, 'both_both_good': .10}
    names = {'both_both': 'Both-Both', 'both_both_good': 'Both-Both + Good'}
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'legend.fontsize': 10, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.4), sharex='col',
        gridspec_kw={'height_ratios': [2.2, 1.]})
    fig.subplots_adjust(left=.065, right=.985, top=.805, bottom=.205,
                        hspace=.065, wspace=.19)
    fig.suptitle('Measured dimuon response versus '+title,
                 fontsize=17, y=.978)
    events = sum(row['events'] for row in pooled.values())
    scope = split+' only' if split else 'development and validation pooled for description'
    description = summary.get('sample_description', 'originals from completed source cohorts')
    fig.text(.5, .937, f'{events:,} '+description+'; '+scope,
             ha='center', fontsize=11)
    handles = [Line2D([], [], color=colors[c], marker='o', linestyle='none', label=names[c])
               for c in CATEGORIES]
    handles.append(Line2D([], [], color='#505050', marker='v', linestyle='none',
                          label='Zero successes: one-sided 95% upper bound'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .91),
               ncol=3, frameon=False, handletextpad=.5, columnspacing=2.)
    for column, process in enumerate(PROCESSES):
        row = pooled[process]
        top, bottom = axes[:, column]
        total = row['events']
        both = row['categories']['both_both']['positive']
        good = row['categories']['both_both_good']['positive']
        sampling_note = summary.get('process_sampling_notes', {}).get(process)
        title_text = PROCESS_LABELS[process]+f'\nN = {total:,}; Both-Both = {both:,}; Good = {good:,}'
        if sampling_note:
            title_text += '\n'+sampling_note
        top.set_title(title_text, pad=10)
        for category in CATEGORIES:
            for i, key in enumerate(keys):
                cell = row['cells'].get(key, {}).get(category, {})
                count, passed = cell.get('events', 0), cell.get('positive', 0)
                if count == 0:
                    continue
                position = x[i]+offsets[category]
                if passed == 0:
                    upper = -math.expm1(math.log(.05)/count)
                    top.errorbar(position, upper, yerr=upper*.30, uplims=True,
                        color=colors[category], fmt='v', markersize=4, capsize=2, linewidth=1.)
                else:
                    value = passed/count
                    low, high = wilson_interval(passed, count)
                    top.errorbar(position, value, yerr=[[value-low], [high-value]],
                        color=colors[category], fmt='o', markersize=4.8, capsize=2.5, linewidth=1.2)
        top.set_yscale('log')
        top.set_ylim(1.e-5, 1.3)
        top.set_xlim(-.6, len(keys)-.4)
        top.grid(axis='y', color='#dddddd', linewidth=.7)
        top.tick_params(labelbottom=False)
        populations = [row['cells'].get(key, {}).get('both_both', {}).get('events', 0) for key in keys]
        bottom.bar(x, populations, color='#cad3da', edgecolor='#64717c', linewidth=.6)
        bottom.set_yscale('log')
        bottom.set_ylim(.65, max(max(populations), 1)*4.)
        bottom.grid(axis='y', color='#e7e7e7', linewidth=.6)
        bottom.set_axisbelow(True)
        for i, count in enumerate(populations):
            bottom.text(x[i], max(count, 1)*1.18, f'{count:,}', ha='center', va='bottom', fontsize=8)
        bottom.set_xticks(x)
        bottom.set_xticklabels([labels[key] for key in keys], rotation=25, ha='right', fontsize=8.5)
        if column == 0:
            top.set_ylabel('Raw MC probability of ≥1 selected dimuon')
            bottom.set_ylabel('Original events')
        for ax in (top, bottom):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    fig.text(.525, .112, xlabel,
             ha='center', fontsize=10.5)
    fig.text(.065, .066, 'Intervals describe the observed GEN proposal mix in completed cohorts, not physical yields. Rate normalization and production selection remain provisional.',
             fontsize=9)
    fig.text(.065, .039, note+' Good = OS, valid DCA, 0 ≤ DCA ≤ 50 cm.',
             fontsize=9)
    fig.text(.065, .014, 'These are one-variable profiles. A joint admission decision must evaluate direction, energy and material for the same pair.',
             fontsize=8.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    if pdf is None:
        fig.savefig(output)
    else:
        pdf.savefig(fig)
    plt.close(fig)
    return edges, pooled


def plot_benchmark(summary, output):
    """Show fixed-draw counts and census admission, without claiming efficiency."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    report = summary['stratified_benchmark']
    regions = ('credible_shallow_pointing', 'border_pointing', 'deep_pointing',
               'no_both_sides_pointing_pair', 'unknown_parent')
    names = ('Pointing + shallow parent', 'Borderline parent path', 'Deep parent path',
             'No pair pointing at both sides', 'Unknown shallow parent path')
    table_rows, counts = [], {}
    for region, name in zip(regions, names):
        cells = [cell for cell in report['cells'] if cell['region'] == region]
        n = sum(cell['sampled_events'] for cell in cells)
        both = sum(cell['targets']['both_both']['positive_originals'] for cell in cells)
        good = sum(cell['targets']['both_both_good']['positive_originals'] for cell in cells)
        counts[region] = dict(tested=n, both_both=both, good=good)
        table_rows.append([name, f'{n:,}', str(both), str(good)])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), gridspec_kw={'width_ratios': [1.2, 1.]})
    fig.subplots_adjust(left=.035, right=.98, top=.82, bottom=.27, wspace=.16)
    fig.suptitle('QCD response study: what is established, and what still limits savings',
                 fontsize=16, y=.98)
    fig.text(.5, .91, f'{report["observed_events"]} reconstructed test events; '
             f'{report["population_events"]:,} events in the eligible GEN census',
             ha='center', fontsize=11)
    left, right = axes
    left.axis('off')
    table = left.table(cellText=table_rows,
        colLabels=['GEN group', 'Tested', 'Both-Both', '+ Good'],
        colWidths=[.57, .14, .16, .13], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1., 2.05)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#c5cdd3')
        if row == 0:
            cell.set_facecolor('#e3eaf0')
            cell.set_text_props(weight='bold')
        elif column == 0:
            cell.set_text_props(ha='left')
    left.set_title('Measured raw counts in the fixed validation draw', fontsize=11, pad=15)
    strata = sorted({row['stratum'] for row in report['planning_forecasts']},
                    key=lambda name: float(name.split('_')[1].split('to')[0]))
    x = np.arange(len(strata))
    largest_fraction = 0.
    for offset, floor, color in ((-.24, .1, '#256b9a'), (0., .01, '#c97f28'),
                                 (.24, .001, '#64834a')):
        fractions = [next(row['expected_admission_fraction'] for row in report['planning_forecasts']
                         if row['stratum'] == name and row['control_floor'] == floor)
                     for name in strata]
        largest_fraction = max(largest_fraction, *fractions)
        right.bar(x+offset, np.array(fractions)*100., width=.23,
                  color=color, label=f'{100*floor:g}% random controls')
    right.set_ylim(0., max(20., 120.*largest_fraction))
    right.set_ylabel('GEN events still admitted to SIM [%]', fontsize=11)
    right.set_xticks(x)
    right.set_xticklabels([name.split('_')[1].replace('to', '–').replace('inf', '∞') for name in strata])
    right.set_xlabel(r'QCD $\hat p_T$ bin [GeV]', fontsize=11)
    right.set_title('Planning: unknown paths of pointing pairs stay protected', fontsize=11, pad=15)
    right.legend(frameon=False, fontsize=9, loc='upper right')
    right.grid(axis='y', alpha=.2)
    right.set_axisbelow(True)
    pairs = sum(cell['targets']['inclusive']['pairs'] for cell in report['cells'])
    both = sum(cell['targets']['both_both']['pairs'] for cell in report['cells'])
    good = sum(cell['targets']['both_both_good']['pairs'] for cell in report['cells'])
    mass_tail = sum(group['tails']['both_both']['mass_gt_20']['positive_originals'] for group in summary['groups'])
    pt_tail = sum(group['tails']['both_both']['pt_gt_10']['positive_originals'] for group in summary['groups'])
    fig.text(.04, .17, f'Reconstructed pairs: {pairs}; Both-Both: {both}; Both-Both + Good: {good}. Good requires OS and valid DCA ≤ 50 cm.', fontsize=11)
    fig.text(.04, .12, f'Both-Both tail events: mass > 20 GeV: {mass_tail}; pair pT > 10 GeV: {pt_tail}. No qualified tail gain or zero-background claim.', fontsize=10)
    fig.text(.04, .075, 'Sampling differs between groups. Raw counts are not population efficiencies; zero controls still allow missed useful events.', fontsize=10)
    fig.text(.04, .03, 'Forecasts apply to the existing selected GEN census. Unknown pointing paths/models stay protected; the production default is unchanged.', fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    reserve = summary.get('reserve_opportunities')
    with PdfPages(output) as pdf:
        pdf.savefig(fig)
        if reserve:
            second, axes = plt.subplots(1, 2, figsize=(14, 5.7),
                gridspec_kw={'width_ratios': [1.1, 1.]})
            second.subplots_adjust(left=.04, right=.97, top=.82, bottom=.30, wspace=.20)
            second.suptitle('Unused QCD GEN: a concrete starting point for rare-decay sampling',
                            fontsize=16, y=.98)
            second.text(.5, .91, 'Original pion parents before natural decay; no detector yield or tail improvement is inferred',
                        ha='center', fontsize=11)
            axes[0].axis('off')
            table = axes[0].table(cellText=[[
                row['stratum'].split('_')[1].replace('to', '–').replace('inf', '∞'),
                str(row['original_events']), str(row['strict_parent_tail_events']),
                str(row['unclassified_original_pions'])] for row in reserve['cohorts']],
                colLabels=[r'$\hat p_T$ [GeV]', 'GEN events', 'Parent mass >20', 'Unclassified pions'],
                loc='center', cellLoc='center', colWidths=[.18, .21, .29, .32])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1., 2.2)
            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor('#c5cdd3')
                if row == 0:
                    cell.set_facecolor('#e3eaf0')
                    cell.set_text_props(weight='bold')
            axes[0].set_title('Independent original-event opportunities', fontsize=11)
            masses = reserve['largest_strict_parent_pair_mass_per_original_GeV']
            axes[1].hist(masses, bins=np.arange(20., 42., 2.), color='#256b9a', edgecolor='white')
            axes[1].set_xlabel('Largest eligible pion-parent pair mass per original [GeV]')
            axes[1].set_ylabel('Original GEN events, unweighted')
            axes[1].set_title('One entry per original; parent mass differs from dimuon mass', fontsize=10)
            axes[1].grid(axis='y', alpha=.2)
            second.text(.04, .20, 'Both pion parents have OS charge and straight rays through ≥2 station envelopes in each endcap. No muon-pT cut.', fontsize=10)
            second.text(.04, .15, 'Rare early decays could provide useful daughters. Physical parent absorption, energy loss and detector response must remain unchanged.', fontsize=10)
            second.text(.04, .10, 'A biased lifetime draw needs its exact full-mixture correction once; repeated draws of one original are not independent events.', fontsize=10)
            second.text(.04, .05, 'This is a lower-bound opportunity census. Unclassified pions are unknown, not zero; these counts are not a physical event rate.', fontsize=10)
            pdf.savefig(second)
            plt.close(second)
        decay = summary.get('decay_pilot')
        if decay:
            third, axes = plt.subplots(1, 2, figsize=(14, 5.7))
            third.subplots_adjust(left=.06, right=.97, top=.79, bottom=.30, wspace=.35)
            third.suptitle('Early-decay test: reconstructed pairs, limited weighted statistics',
                           fontsize=16, y=.98)
            description = decay.get('comparison_description',
                f"{decay['arms']['ordinary']['events']} selected original QCD events, each reconstructed with ordinary and biased lifetimes")
            third.text(.5, .91, description,
                       ha='center', fontsize=11)
            axes[0].axis('off')
            rows = []
            for arm, label in [('ordinary', 'Ordinary'), ('mixture', 'Early-decay mixture')]:
                stats = decay['arms'][arm]['categories']['both_both_good']['mass_gt20']
                rows.append([label, str(decay['muons'][arm]), str(stats['raw_pairs']),
                             f"{stats['effective_originals']:.3f}"])
            table = axes[0].table(cellText=rows,
                colLabels=['Lifetime draw', 'Reco muons', 'OS Both-Both\n+ Good, mass >20', 'Effective\ntail events'],
                colWidths=[.32, .19, .27, .22], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1., 2.5)
            for (row, column), cell in table.get_celld().items():
                cell.set_edgecolor('#c5cdd3')
                if row == 0:
                    cell.set_facecolor('#e3eaf0')
            pairs = decay['positive_pairs']
            axes[1].scatter([p['mass_GeV'] for p in pairs], [p['final_weight'] for p in pairs],
                            color='#256b9a', s=60)
            axes[1].set_yscale('log')
            axes[1].set_xlabel('Reconstructed dimuon mass [GeV]')
            axes[1].set_ylabel('Final event weight (including sampling)')
            axes[1].set_title('Each point is one original collision', fontsize=11)
            axes[1].grid(alpha=.2)
            notes = decay.get('notes', [
                'The 136 GeV entry has inflated reconstructed momentum and a weak second hit association.',
                'Neither pair has two strict hit-to-GEN matches. Reconstructed mass is not evidence of a true 136 GeV pair.',
                'The two weights differ greatly: effective tail statistics are about one event. No orders-of-magnitude gain is established.',
                'Conditional mechanism test, not a physical yield prediction. Zero ordinary pairs do not establish zero rate.'])
            for y, note in zip([.20, .15, .10, .05], notes):
                third.text(.04, y, note, fontsize=10)
            pdf.savefig(third)
            plt.close(third)
    plt.close(fig)
    return dict(raw_fixed_draw_counts=counts, forecast_scope=report['forecast_scope'],
                physical_yield_estimate=False, production_policy_qualified=False,
                reserve_opportunities_included=reserve is not None,
                decay_pilot_included=summary.get('decay_pilot') is not None)


def candidate_comparison_input(report):
    """Validate disjoint completed-source results before displaying fractions."""
    if report.get('schema') != 'shift-sm-candidate-response-analysis-v1':
        raise ValueError('Require the candidate response analysis schema')
    scope = report['scope']
    if scope.get('completion_conditioned') is not True:
        raise ValueError('Candidate comparison must declare its completed-source scope')
    groups = {row['region_group']: row for row in report['pooled_groups']}
    if set(groups) != {'candidate', 'controls'}:
        raise ValueError('Require separate candidate and random-control groups')
    for group in groups.values():
        count = group['rows']
        if type(count) is not int or not 0 <= group['muon_events'] <= count:
            raise ValueError('Muon successes exceed the completed original-event count')
        for name in ('inclusive', 'both_both', 'both_both_good'):
            stats = group['targets'][name]
            if (stats['events'] != count or not 0 <= stats['positive_originals'] <= count
                    or stats['pairs'] < stats['positive_originals']):
                raise ValueError('Response counts do not describe the completed originals')
    if sum(group['rows'] for group in groups.values()) != scope['complete_events']:
        raise ValueError('Candidate and control counts do not cover completed events')
    for name, group in groups.items():
        bins = [row for row in report['groups'] if row['region_group'] == name]
        if sum(row['rows'] for row in bins) != group['rows']:
            raise ValueError('Generator-bin counts do not cover the pooled group')
        for category in ('inclusive', *CATEGORIES):
            for field in ('positive_originals', 'pairs'):
                if sum(row['targets'][category][field] for row in bins) != group['targets'][category][field]:
                    raise ValueError('Generator-bin successes do not cover the pooled group')
    return scope, groups


def pooled_candidate_histograms(report):
    """Add disjoint original-event histogram contributions, keeping flow bins."""
    result = {}
    for row in report['histograms']:
        if row.get('stratum') == 'all_qcd':
            # Per-bin rows below are the nonoverlapping authority.
            continue
        if row['category'] not in CATEGORIES or row['field'] not in ('mass', 'pt'):
            continue
        key = row['region_group'], row['category'], row['field']
        edges = row['edges']
        if (len(edges) < 2 or any(not math.isfinite(x) for x in edges)
                or any(a >= b for a, b in zip(edges, edges[1:]))):
            raise ValueError('Require finite increasing candidate histogram edges')
        if len(row['raw_counts']) != len(edges)+1:
            raise ValueError('Candidate histograms must retain underflow and overflow')
        if any(type(value) is not int or value < 0 for value in row['raw_counts']):
            raise ValueError('Raw histogram counts must be nonnegative integers')
        if key not in result:
            result[key] = dict(edges=list(edges), raw_counts=[0]*(len(edges)+1), nonfinite=0)
        if result[key]['edges'] != edges:
            raise ValueError('Cannot pool candidate histograms with different axes')
        target = result[key]['raw_counts']
        for index, value in enumerate(row['raw_counts']):
            target[index] += value
        result[key]['nonfinite'] += row.get('nonfinite', 0)
    return result


def partial_both_both_pairs(report):
    """Keep partial observations descriptive; require their explicit raw census."""
    pairs, originals = [], set()
    for row in report.get('partial_positive_originals', []):
        key = row['gen_sha256'], tuple(row['event_id'])
        if key in originals or row.get('rates_and_intervals_withheld') is not True:
            raise ValueError('Partial pairs require unique originals and no efficiency estimate')
        originals.add(key)
        count = 0
        for pair in row['retained_pairs']:
            if pair['topologyMin'] != 2 or pair['topologyMax'] != 2:
                continue
            count += 1
            good = (pair['isOS'] == 1 and pair['dcaValid'] == 1
                    and pair['dca'] is not None and math.isfinite(pair['dca'])
                    and 0. <= pair['dca'] <= 50.)
            pairs.append(dict(source_id=row['source_id'], event_id=row['event_id'],
                region_group=row['region_group'], pair=pair, good=good))
        if count != row['pair_counts']['both_both']:
            raise ValueError('Partial Both-Both pair payload differs from its count')
    totals = [row for row in report.get('partial_groups', []) if row['stratum'] == 'all_qcd']
    if sum(row['targets']['both_both']['pairs'] for row in totals) != len(pairs):
        raise ValueError('Partial pair payload does not cover the raw observed count')
    return pairs


def plot_candidate_profiles(report, pdf):
    """Keep the same GEN pair's predictors together and exclude partial sources."""
    import matplotlib.pyplot as plt

    features = {
        'min_muon_p_GeV': 'Lower total muon momentum [GeV]',
        'max_muon_column_loss_over_p': 'Larger column-ionization / total momentum',
        'two_station_miss_cm': 'Larger two-station miss [cm]',
        'known_union_parent_depth': 'Known parent material [interaction lengths]',
    }
    profiles = {row['feature']:row for row in report.get('energy_material_profiles', [])
                if row['stratum']=='all_qcd' and row['region_group']=='candidate'}
    if not profiles or not report['scope']['complete_events']:
        return 0
    count = next(row['rows'] for row in report['pooled_groups'] if row['region_group']=='candidate')
    if not count:
        return 0
    fig, axes = plt.subplots(2,2,figsize=(14,9.2))
    fig.subplots_adjust(left=.075,right=.975,top=.80,bottom=.27,hspace=.66,wspace=.24)
    policy = report.get('selection_policy_version', 'unspecified')
    fig.suptitle(f'Energy, direction and material response (selection {policy})',fontsize=18,y=.975)
    fig.text(.5,.92,f'{count} candidates from complete source cohorts only; 95% raw-sample intervals.',
             ha='center',fontsize=11)
    fig.text(.5,.875,'Every predictor uses one frozen GEN pair. No reconstructed result chooses the pair.',
             ha='center',fontsize=11)
    for axis,(feature,label) in zip(axes.flat,features.items()):
        profile = profiles[feature]
        cells = profile['cells']
        if (sum(cell['rows'] for cell in cells)!=count
                or profile.get('same_pair_components') is not True
                or profile.get('physical_survival_model') is not False):
            raise ValueError('Same-pair profiles must cover complete candidates without a survival claim')
        edges = profile['edges']
        def order(cell):
            return {'underflow':-1,'overflow':len(edges),'unknown':len(edges)+1}.get(
                cell['bin'],int(cell['bin']) if cell['bin'].isdigit() else len(edges)+2)
        cells = sorted(cells,key=order)
        ticklabels = []
        for index,cell in enumerate(cells):
            n = cell['rows']
            for offset,passed,color,name in ((-.08,cell['muon_events'],'#2468a0','≥1 reco muon'),
                    (.08,cell['targets']['both_both_good']['positive_originals'],'#64834a','Good Both-Both')):
                low,high = wilson_interval(passed,n)
                axis.errorbar(index+offset,passed/n,yerr=[[passed/n-low],[high-passed/n]],fmt='o',
                              color=color,capsize=3,label=name if index==0 else None)
            key = cell['bin']
            interval = f'{edges[int(key)]:g}–{edges[int(key)+1]:g}' if key.isdigit() else key
            ticklabels.append(f'{interval}\nN={n}')
        axis.set_xticks(range(len(cells)))
        axis.set_xticklabels(ticklabels,rotation=25,ha='right',fontsize=9)
        axis.set_xlabel(label)
        axis.set_ylabel('Raw sample fraction')
        axis.set_ylim(-.04,1.04)
        axis.grid(axis='y',alpha=.25)
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.86),ncol=2,frameon=False,fontsize=10)
    fig.text(.04,.135,'Unknown material and censored pair lists stay unknown. These small marginal profiles are not calibrated efficiencies.',fontsize=10)
    fig.text(.04,.09,'The column predictor covers external material with minimum ionization; radiative losses and central CMS material are omitted.',fontsize=10)
    fig.text(.04,.045,'No survival probability, fitted threshold, new selection cut or full-production improvement is claimed.',fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def plot_candidate_comparison(report, output):
    """Keep different frozen selection policies in separate parts of one PDF."""
    policies = report.get('policy_groups')
    if not policies:
        return _plot_candidate_comparison(report,output)
    versions = [row['selection_policy_version'] for row in policies]
    if (len(versions)!=len(set(versions)) or not report.get('mixed_policy_efficiencies_withheld')
            or report['pooled_groups'] or report['groups'] or report['histograms']):
        raise ValueError('Mixed policies need unique versions and separately reported response statistics')
    from matplotlib.backends.backend_pdf import PdfPages
    output.parent.mkdir(parents=True,exist_ok=True)
    receipts = []
    with PdfPages(output) as pdf:
        for index,child in enumerate(policies):
            child = dict(child)
            if index==0 and 'historical_baseline' in report:
                child['historical_baseline'] = report['historical_baseline']
            receipt = _plot_candidate_comparison(child,output,_pdf=pdf)
            receipts.append(dict(receipt,selection_policy_version=child['selection_policy_version']))
    return dict(schema='shift-sm-candidate-response-plot-v1',pages=sum(r['pages'] for r in receipts),
        complete_sources=sum(r['complete_sources'] for r in receipts),
        complete_events=sum(r['complete_events'] for r in receipts),policy_groups=receipts,
        completion_conditioned=True,mixed_policy_efficiencies_withheld=True,
        physical_yield_estimate=False,production_policy_qualified=False)


def _plot_candidate_comparison(report, output, *, _pdf=None):
    """Show response and actual pair kinematics, without physical-yield claims."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    scope, groups = candidate_comparison_input(report)
    histograms = pooled_candidate_histograms(report)
    partial_pairs = partial_both_both_pairs(report)
    for name, group in groups.items():
        for category in CATEGORIES:
            for field in ('mass', 'pt'):
                histogram = histograms.get((name, category, field), dict(raw_counts=[], nonfinite=0))
                if sum(histogram['raw_counts'])+histogram['nonfinite'] != group['targets'][category]['pairs']:
                    raise ValueError('Kinematic histogram counts do not cover retained pairs')
    labels = {'candidate': 'Pointing + shallow parent', 'controls': 'Random controls'}
    colors = {'candidate': '#2468a0', 'controls': '#c47b29'}
    stages = [('muons', '≥1 reco muon'), ('inclusive', 'Any dimuon'),
              ('both_both', 'Both-Both'), ('both_both_good', 'Both-Both + Good')]
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.7), gridspec_kw={'width_ratios': [1.15, 1.]})
    fig.subplots_adjust(left=.04, right=.98, top=.77, bottom=.27, wspace=.17)
    policy = report.get('selection_policy_version')
    suffix = f' (selection {policy})' if policy else ''
    fig.suptitle('QCD candidate selection: reconstructed results'+suffix, fontsize=18, y=.975)
    fig.text(.5, .925,
        f"{scope['complete_events']:,} originals from {scope['complete_sources']:,} complete source cohorts"
        f" / {scope['total_sources']:,} planned cohorts", ha='center', fontsize=12)
    fig.text(.5, .878, 'Partial campaign: only complete source draws enter these comparisons.',
             ha='center', fontsize=11)
    if 'pending_events' in scope:
        coverage = (f"{scope['frozen_draw_sources']:,}/{scope['total_sources']:,} source draws frozen; "
                    if 'frozen_draw_sources' in scope else '')
        fig.text(.5, .835,
            coverage+f"{scope['pending_events']:,} individual responses pending. Pending outcomes are not physics zeros.",
            ha='center', fontsize=10)
    left, right = axes
    left.axis('off')
    rows = []
    for group_name in ('candidate', 'controls'):
        group = groups[group_name]
        values = [group['muon_events'], group['targets']['inclusive']['positive_originals'],
                  group['targets']['both_both']['positive_originals'],
                  group['targets']['both_both_good']['positive_originals']]
        rows.append([labels[group_name], str(group['rows']),
                     *(str(value) if group['rows'] else '—' for value in values)])
    table = left.table(cellText=rows,
        colLabels=['GEN group', 'Tested', '≥1 muon', 'Dimuon', 'Both-Both', '+ Good'],
        colWidths=[.36, .11, .13, .12, .15, .13], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1., 2.45)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#c5cdd3')
        if row == 0:
            cell.set_facecolor('#e3eaf0')
            cell.set_text_props(weight='bold')
        elif column == 0:
            cell.set_text_props(ha='left')
    left.set_title('Independent original events with at least one success', pad=12)
    partial = [row for row in report.get('partial_groups', []) if row['stratum'] == 'all_qcd']
    if partial:
        left.text(.5, .22, 'Partial-source responses: counts only', transform=left.transAxes,
                  ha='center', fontsize=10)
        table = left.table(cellText=[[labels[row['region_group']], str(row['rows']),
            str(row['muon_events']), str(row['targets']['inclusive']['positive_originals'])]
            for row in partial], colLabels=['GEN group', 'Finished', '≥1 muon', 'Dimuon'],
            colWidths=[.50, .17, .17, .16], cellLoc='center', bbox=[0., -.025, 1., .18])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#c5cdd3')
            if row == 0:
                cell.set_facecolor('#e3eaf0')
    x = np.arange(len(stages), dtype=float)
    largest_upper = 0.
    for offset, group_name in ((-.10, 'candidate'), (.10, 'controls')):
        group = groups[group_name]
        n = group['rows']
        first = True
        for index, (key, _) in enumerate(stages):
            passed = group['muon_events'] if key == 'muons' else group['targets'][key]['positive_originals']
            if not n:
                continue
            low, high = wilson_interval(passed, n)
            largest_upper = max(largest_upper, high)
            rate = passed/n
            right.errorbar(index+offset, rate,
                yerr=[[rate-low], [high-rate]], fmt='o', color=colors[group_name],
                capsize=4, markersize=6, label=labels[group_name] if first else None)
            first = False
    right.set_xticks(x)
    right.set_xticklabels([name for _, name in stages], rotation=20, ha='right')
    right.set_ylabel('Raw fraction in the reconstructed test sample')
    right.set_ylim(-.025, min(1.04, max(.20, largest_upper*1.2)))
    right.grid(axis='y', color='#dddddd')
    if scope['complete_events']:
        right.legend(frameon=False, fontsize=10)
    else:
        right.set_ylim(-.035, 1.04)
        right.text(.5, .52, 'No complete current source draws yet.\nNo reconstruction efficiency is estimated.',
                   transform=right.transAxes, ha='center', va='center', fontsize=12)
    right.set_title('Raw sample response with 95% intervals')
    fig.text(.04, .16, 'Good = opposite sign, valid closest approach, and 0 ≤ DCA ≤ 50 cm.', fontsize=11)
    fig.text(.04, .105, 'Candidates and controls have different sampling probabilities. These raw fractions are not population efficiencies.', fontsize=10)
    fig.text(.04, .055, 'Completed cohorts can differ from unfinished cohorts. No full-campaign gain or physical event yield is claimed.', fontsize=10)
    if scope.get('blocked_sources'):
        fig.text(.04,.022,f"{scope['blocked_selected_events']} pending responses belong to {scope['blocked_sources']} sources blocked by full-GEN preflight; these are missing responses, not rejections.",fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (nullcontext(_pdf) if _pdf is not None else PdfPages(output)) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        if not scope['complete_events'] and not scope.get('partial_completed_events',0):
            return dict(schema='shift-sm-candidate-response-plot-v1',pages=1,
                complete_sources=scope['complete_sources'],complete_events=0,
                completion_conditioned=True,physical_yield_estimate=False,
                production_policy_qualified=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.7))
        from matplotlib.ticker import MaxNLocator
        fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.47, wspace=.24)
        fig.suptitle('Actual Both-Both pair kinematics and tail statistics'+suffix, fontsize=18, y=.975)
        fig.text(.5, .925, 'Complete source cohorts only. Several pairs from one collision are not independent observations.',
                 ha='center', fontsize=11)
        fig.text(.5, .878, 'Solid: Both-Both + Good; dotted: all Both-Both vertices.', ha='center', fontsize=11)
        flow_notes = []
        for axis, field, xlabel in zip(axes, ('mass', 'pt'),
                ('Reconstructed dimuon mass [GeV]', 'Reconstructed dimuon pT [GeV]')):
            for group_name in ('candidate', 'controls'):
                for category, style in (('both_both', ':'), ('both_both_good', '-')):
                    histogram = histograms.get((group_name, category, field))
                    if histogram is None:
                        continue
                    edges, counts = histogram['edges'], histogram['raw_counts']
                    axis.step(edges, [*counts[1:-1], counts[-2]], where='post',
                        color=colors[group_name], linestyle=style, linewidth=2.,
                        label=labels[group_name] if category == 'both_both_good' else None)
                    if category == 'both_both_good':
                        flow_notes.append(f"{group_name} {field} {counts[0]}/{counts[-1]}/{histogram['nonfinite']}")
            for threshold in ((20., 50.) if field == 'mass' else (10., 20.)):
                axis.axvline(threshold, color='#777777', linewidth=.8, alpha=.6)
            axis.set_xlabel(xlabel)
            axis.set_ylabel('Reconstructed pairs, unweighted')
            axis.set_ylim(bottom=0)
            if axis.get_ylim()[1] < 1.:
                axis.set_ylim(top=1.)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis='y', color='#dddddd')
            if histograms:
                axis.legend(frameon=False, fontsize=9)
        tail_rows = []
        for group_name in ('candidate', 'controls'):
            for category, category_name in (('both_both', 'Both-Both'), ('both_both_good', '+ Good')):
                tail = groups[group_name]['tails'][category]
                target = groups[group_name]['targets'][category]
                available = groups[group_name]['rows'] > 0
                tail_rows.append([labels[group_name], category_name,
                    str(target['positive_originals']) if available else '—',
                    str(target['pairs']) if available else '—',
                    *(str(tail[name]['positive_originals']) if available else '—' for name in
                      ('mass_gt_20', 'mass_gt_50', 'pt_gt_10', 'pt_gt_20'))])
        table_axis = fig.add_axes([.035, .18, .93, .22])
        table_axis.axis('off')
        table = table_axis.table(cellText=tail_rows,
            colLabels=['GEN group', 'Vertex selection', 'Events', 'Pairs',
                       'm >20', 'm >50', 'pT >10', 'pT >20'],
            colWidths=[.25, .16, .08, .13, .095, .095, .095, .095],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1., 1.85)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#c5cdd3')
            if row == 0:
                cell.set_facecolor('#e3eaf0')
                cell.set_text_props(weight='bold')
        fig.text(.04, .12, 'Tail columns count independent original events. Weighted statistics are shown separately for each generator bin.', fontsize=10)
        fig.text(.04, .080, 'These raw kinematic distributions describe the sampled test mix, not the physical QCD spectrum.', fontsize=10)
        fig.text(.04, .043, 'Zero observed tails do not establish zero physical rate.', fontsize=9)
        fig.text(.04, .019, 'Underflow/overflow/nonfinite pair counts: '+ '; '.join(flow_notes), fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(14, 8.7))
        fig.subplots_adjust(left=.035, right=.97, top=.79, bottom=.34)
        fig.suptitle('Useful weighted statistics within each QCD generator bin'+suffix, fontsize=18, y=.975)
        fig.text(.5, .918, 'Both-Both + Good; completed source draws only. No weighted spectrum is pooled across bins.',
                 ha='center', fontsize=11)
        axis.axis('off')
        rows = []
        for group in sorted(report['groups'], key=lambda row: (
                float(row['stratum'].split('_')[1].split('to')[0]), row['region_group'] != 'candidate')):
            stats = group['targets']['both_both_good']
            available = group['rows'] > 0
            bin_name = group['stratum'].split('_', 1)[1].replace('to', '–').replace('inf', '∞')
            tails = group['tails']['both_both_good']
            rows.append([bin_name, labels[group['region_group']], str(group['rows']),
                str(stats['positive_originals']) if available else '—',
                f"{stats['effective_pair_contributions']:.3f}" if available else '—',
                str(tails['mass_gt_20']['positive_originals']) if available else '—',
                str(tails['pt_gt_10']['positive_originals']) if available else '—',
                f"{group['total_sim_seconds']/3600.:.2f}" if available else '—'])
        if rows:
            table = axis.table(cellText=rows,
                colLabels=['Generator bin\n[GeV]', 'GEN group', 'Tested', 'Good events',
                           'Weighted Neff', 'm >20', 'pT >10', 'SIM hours'],
                colWidths=[.13, .25, .10, .12, .13, .09, .09, .10],
                loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1., 2.1)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor('#c5cdd3')
                if row == 0:
                    cell.set_facecolor('#e3eaf0')
                    cell.set_text_props(weight='bold')
        else:
            axis.text(.5, .5, 'No completed source cohorts yet.', ha='center', va='center', fontsize=14)
        fig.text(.04, .275, 'Neff uses W / (p × q) once, with all pairs from an original grouped before its second moment.', fontsize=10)
        fig.text(.04, .230, 'Different generator bins have different source exposure. Their native weighted sums are not a physical combined yield.', fontsize=10)
        fig.text(.04, .195, 'No luminosity scaling; rate normalization remains unqualified.', fontsize=10)
        baseline = report.get('historical_baseline')
        if baseline:
            historical = {row['region_group']: row for row in baseline['pooled_groups']}
            a, b = historical['candidate'], historical['controls']
            fig.text(.04, .165, f"Earlier independent validation: {a['rows']} candidates → "
                f"{a['targets']['both_both_good']['positive_originals']} Good Both-Both events; "
                f"{b['rows']} controls → {b['targets']['both_both_good']['positive_originals']}.", fontsize=11)
            fig.text(.04, .118, 'Earlier candidates used a looser one-station pointing rule; this search requires two stations. The samples are separate.', fontsize=10)
            fig.text(.04, .073, 'Those earlier successes support testing the idea; they do not establish a gain for the current selection or its tails.', fontsize=10)
        else:
            fig.text(.04, .155, 'Completion conditioning and limited positive statistics still apply. No full-production gain is inferred.', fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)
        pages = 3
        if partial_pairs:
            fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
            fig.subplots_adjust(left=.075, right=.97, top=.79, bottom=.38, wspace=.24)
            fig.suptitle('First Both-Both pairs from partial source cohorts'+suffix, fontsize=18, y=.975)
            fig.text(.5, .922, f'{len(partial_pairs)} observed pairs; raw case diagnostics only. No efficiency or normalized rate.',
                     ha='center', fontsize=11)
            shown = [row for row in partial_pairs if row['pair']['dcaValid'] == 1
                and row['pair']['dca'] is not None and math.isfinite(row['pair']['dca'])
                and row['pair']['dca'] >= 0.]
            maximum_dca = max([row['pair']['dca'] for row in shown]+[70.])*1.22
            for axis, field, xlabel in zip(axes, ('mass', 'pt'),
                    ('Reconstructed dimuon mass [GeV]', 'Reconstructed dimuon pT [GeV]')):
                axis.axhspan(0., 50., color='#64834a', alpha=.12)
                axis.axhline(50., color='#64834a', linestyle='--', label='Good DCA limit: 50 cm')
                axis.scatter([], [], s=65, color='#2468a0', label='Observed Both-Both pair')
                if any(row['good'] for row in shown):
                    axis.scatter([],[],s=100,color='#64834a',marker='*',label='Good Both-Both pair')
                for row in shown:
                    pair = row['pair']
                    if pair[field] is None or not math.isfinite(pair[field]):
                        continue
                    axis.scatter(pair[field],pair['dca'],s=100 if row['good'] else 65,
                                 color='#64834a' if row['good'] else '#2468a0',
                                 marker='*' if row['good'] else 'o',zorder=3)
                    axis.annotate(str(row['event_id'][-1]), (pair[field], pair['dca']),
                        xytext=(7, 5), textcoords='offset points', fontsize=10)
                axis.set_ylim(0., maximum_dca)
                values = [row['pair'][field] for row in shown if row['pair'][field] is not None
                          and math.isfinite(row['pair'][field])]
                axis.set_xlim(0., max([5.]+values)*1.15)
                axis.set_xlabel(xlabel)
                axis.set_ylabel('Dimuon distance of closest approach [cm]')
                axis.grid(alpha=.2)
                axis.legend(frameon=False, fontsize=10, loc='upper right')
            table_axis = fig.add_axes([.04, .15, .92, .16])
            table_axis.axis('off')
            def number(value):
                return 'Unavailable' if value is None or not math.isfinite(value) else f'{value:.3f}'
            table = table_axis.table(cellText=[[
                row['source_id'].removeprefix('supply_qcd_'), str(row['event_id'][-1]),
                number(row['pair']['mass']), number(row['pair']['pt']), number(row['pair']['dca']),
                str(row['pair']['isOS']), str(row['pair']['dcaValid']), 'Yes' if row['good'] else 'No']
                for row in partial_pairs],
                colLabels=['Source', 'Event', 'Mass [GeV]', 'pT [GeV]', 'DCA [cm]', 'OS', 'Valid DCA', 'Good'],
                loc='center', cellLoc='center', colWidths=[.22, .08, .13, .13, .13, .07, .13, .09])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1., 1.7)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor('#c5cdd3')
                if row == 0:
                    cell.set_facecolor('#e3eaf0')
            good = sum(row['good'] for row in partial_pairs)
            if good:
                observed = '; '.join(f"event {row['event_id'][-1]}: pT {row['pair']['pt']:.2f} GeV, mass {row['pair']['mass']:.2f} GeV"
                                     for row in partial_pairs if row['good'])
                fig.text(.04,.118,'Observed Good Both-Both '+observed,fontsize=11,color='#46602f')
            fig.text(.04, .088, f'Good pairs in these observations: {good}/{len(partial_pairs)}. '
                     'Good requires OS, valid DCA and 0 ≤ DCA ≤ 50 cm.', fontsize=11)
            fig.text(.04, .045, f'{len(partial_pairs)-len(shown)} pairs have no usable DCA coordinate. '
                     'Partial-cohort completion can bias these observations; they are excluded from the efficiency plots.', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
        pages += plot_candidate_profiles(report,pdf)
    return dict(schema='shift-sm-candidate-response-plot-v1', pages=pages,
        complete_sources=scope['complete_sources'], complete_events=scope['complete_events'],
        completion_conditioned=True, physical_yield_estimate=False,
        production_policy_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--candidate-comparison', action='store_true',
                        help='Plot complete-source candidate/control response and reconstructed mass/pT')
    parser.add_argument('--all-features', action='store_true',
                        help='Write the measured direction, energy, material and birth profiles into one PDF')
    parser.add_argument('--split', choices=('development', 'validation'))
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    if args.candidate_comparison:
        if args.stratified or args.all_features or args.split:
            parser.error('--candidate-comparison is separate from marginal or old benchmark reports')
        receipt = plot_candidate_comparison(summary, args.output)
        receipt.update(input=str(args.summary), input_sha256=hashlib.sha256(raw).hexdigest(),
            output=str(args.output), output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        print('PDF SHA256', receipt['output_sha256'])
        return
    if args.all_features:
        if args.stratified:
            parser.error('--all-features and --stratified are separate report types')
        from matplotlib.backends.backend_pdf import PdfPages
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pages = {}
        with PdfPages(args.output) as pdf:
            for feature in FEATURE_LABELS:
                edges, pooled = plot(summary, args.output, feature, args.split, pdf)
                pages[feature] = dict(edges=edges, processes=pooled)
        receipt = dict(schema='shift-sm-response-profiles-v1',
            input=str(args.summary), input_sha256=hashlib.sha256(raw).hexdigest(),
            output=str(args.output), output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
            study_split=args.split, pages=pages, physical_yield_estimate=False,
            raw_proposal_mix_not_calibrated_population_efficiency=True,
            production_policy_qualified=False)
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        print('PDF SHA256', receipt['output_sha256'])
        return
    if args.stratified:
        receipt = plot_benchmark(summary, args.output)
        receipt.update(input=str(args.summary), input_sha256=hashlib.sha256(raw).hexdigest(),
            output=str(args.output), output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
        if args.receipt:
            args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
        print('PDF SHA256', receipt['output_sha256'])
        return
    edges, pooled = plot(summary, args.output, split=args.split)
    receipt = dict(schema='shift-sm-response-plot-v1', input=str(args.summary),
        input_sha256=hashlib.sha256(raw).hexdigest(), output=str(args.output),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        feature=FEATURE, edges=edges, processes=pooled,
        pooling='sum raw independent-original N and K across bins and frozen folds, then recompute intervals',
        zero_observation_interval='one-sided exact 95% raw binomial upper bound',
        positive_observation_interval='two-sided Wilson 95% raw binomial interval',
        physics_normalization_ready=summary['physics_normalization_ready'],
        production_policy_qualified=summary['production_policy_qualified'],
        complete_cohort_conditioning_may_be_informative=summary['complete_cohort_conditioning_may_be_informative'],
        physical_yield_estimate=False)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: dict(events=value['events'], categories=value['categories'])
                      for key, value in pooled.items()}, indent=2))
    print('PDF SHA256', receipt['output_sha256'])


if __name__ == '__main__':
    main()
