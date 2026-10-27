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
import hashlib
import json
import math
from pathlib import Path


FEATURE = 'min_parent_prefix_depth'
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


def pool_response(summary):
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
    for group in summary['groups']:
        process = group['stratum'].split('_', 1)[0]
        if process not in pooled:
            raise ValueError('Unexpected SM process: '+process)
        axis = group['efficiency_maps'][FEATURE]
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
        for key, cell in axis['cells'].items():
            for category in CATEGORIES:
                result = pooled[process]['cells'][key][category]
                result['events'] += cell[category]['events']
                result['positive'] += cell[category]['positive_originals']
                result['pairs'] += cell[category]['pairs']
    if sum(row['events'] for row in pooled.values()) != summary['events']:
        raise ValueError('Process totals do not cover the input response population')
    for process in PROCESSES:
        pooled[process]['cells'] = dict(pooled[process]['cells'])
    return edges, pooled


def bin_labels(edges, keys):
    labels = {}
    for key in keys:
        if key == 'unknown':
            labels[key] = 'No incoming\nOS pair'
        elif key == 'underflow':
            labels[key] = '<'+format(edges[0], 'g')
        elif key == 'overflow':
            labels[key] = r'$\geq$'+format(edges[-1], 'g')
        else:
            index = int(key)
            labels[key] = format(edges[index], 'g')+'–'+format(edges[index+1], 'g')
    return labels


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    edges, pooled = pool_response(summary)
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
    fig.suptitle('Measured dimuon response versus known parent material',
                 fontsize=17, y=.978)
    fig.text(.5, .937, f'{summary["events"]:,} originals from completed source cohorts; development and validation pooled for description',
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
        top.set_title(PROCESS_LABELS[process]+f'\nN = {total:,}; Both-Both = {both:,}; Good = {good:,}', pad=10)
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
        bottom.set_ylim(.65, max(populations)*4.)
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
    fig.text(.525, .112, 'Smallest summed known parent-path depth of an incoming OS pair [interaction-length units]',
             ha='center', fontsize=10.5)
    fig.text(.065, .066, 'Intervals describe the observed GEN proposal mix in completed cohorts, not physical yields. Rate normalization and production selection remain provisional.',
             fontsize=9)
    fig.text(.065, .039, 'Only known external parent-path prefixes are included; zero depth does not prove zero total material. Good = OS, valid DCA, 0 ≤ DCA ≤ 50 cm.',
             fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return edges, pooled


def plot_benchmark(summary, output):
    """Show fixed-draw counts and census admission, without claiming efficiency."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
    fig.savefig(output)
    plt.close(fig)
    return dict(raw_fixed_draw_counts=counts, forecast_scope=report['forecast_scope'],
                physical_yield_estimate=False, production_policy_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--stratified', action='store_true')
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    if args.stratified:
        receipt = plot_benchmark(summary, args.output)
        receipt.update(input=str(args.summary), input_sha256=hashlib.sha256(raw).hexdigest(),
            output=str(args.output), output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
        if args.receipt:
            args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
        print('PDF SHA256', receipt['output_sha256'])
        return
    edges, pooled = plot(summary, args.output)
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
