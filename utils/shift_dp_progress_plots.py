#!/usr/bin/env python3
# Code style: AI-code. AI-generated presentation plots of audited DP progress.
"""Compare frozen signal grids and limits without rerunning or changing fits.

The before/after limits must use matched exposure, mass bins and backgrounds.
Legacy DY and qualified new-model support are deliberately shown separately.
"""
import argparse
import hashlib
import json
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS / 'bin'))
from shift_limits_current_plots import draw_regions, supported_contour_rows
from shift_limits_lifetime import plotting


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_snapshot(directory):
    path = directory / 'snapshot.json'
    value = read_json(path)
    cache = Path(value['histogram_cache'])
    if digest(cache) != value['histogram_cache_sha256']:
        raise ValueError('Frozen histogram cache changed: ' + str(cache))
    scan = read_json(directory / 'scan.json')
    if scan['snapshot_sha256'] != digest(path):
        raise ValueError('Scan does not match its frozen snapshot: ' + str(path))
    if scan.get('include_mc_stat_uncertainties', False):
        raise ValueError('This matched comparison uses the nominal scans without MC statistics')
    return value, scan


def background_closure(snapshots):
    reference = snapshots[0]
    report = []
    for current in snapshots[1:]:
        for key in ('luminosity_pb', 'categories', 'mass_edges_gev'):
            if current[key] != reference[key]:
                raise ValueError('Unmatched comparison setting: ' + key)
        if len(current['backgrounds']) != len(reference['backgrounds']):
            raise ValueError('Unmatched number of background templates')
        with np.load(reference['histogram_cache'], allow_pickle=False) as old:
            with np.load(current['histogram_cache'], allow_pickle=False) as new:
                for left, right in zip(reference['backgrounds'], current['backgrounds']):
                    for key in ('name', 'cross_section_pb', 'full_gen_sumw'):
                        if left[key] != right[key]:
                            raise ValueError('Unmatched background normalization: ' + left['name'])
                    for suffix in ('', '_sumw2'):
                        if not np.array_equal(old[left['array_key'] + suffix],
                                              new[right['array_key'] + suffix]):
                            raise ValueError('Unmatched background contents/variance: ' + left['name'])
        report.append(dict(background_templates=len(current['backgrounds']),
                           arrays_and_normalization_identical=True))
    return report


def selected_rows(scan):
    rows = [row for row in scan['records']
            if row['channel'] == 'Both-Both' and row['confidence_level'] == .9]
    if not rows:
        raise ValueError('No Both-Both 90% CL scan records')
    return rows


def draw_limits(before, legacy, after, output, zoom=False):
    directories = [before, legacy, after]
    loaded = [checked_snapshot(path) for path in directories]
    closure = background_closure([item[0] for item in loaded])
    current_strength = read_json(after/'signal_strength_map.json')
    previous_path = after/'previous_signal_strength.json'
    previous_strength = (read_json(previous_path) if previous_path.exists() else
        dict(records=evaluated_strength_grid(legacy)[2], snapshot_sha256=digest(legacy/'snapshot.json')))
    if previous_strength['snapshot_sha256'] != digest(legacy/'snapshot.json'):
        raise ValueError('Previous strength grid belongs to another signal snapshot')
    plt = plotting()
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    colors = ['#b33636', '#176b9c']
    projections = read_json(before / 'projection_inventory.json')
    pheno = [curve for curve in projections['curves']
             if np.max(np.abs(np.asarray(curve['color']) - (0., .7490196, 1.))) < .002]
    if len(pheno) != 1:
        raise ValueError('Expected one pinned briefing-book SHIFT reference curve')
    for axis, mode, color in zip(axes, ['background', 'zero_background'], colors):
        points = np.asarray(pheno[0]['points'])
        axis.plot(points[:, 0], points[:, 1], color='#168c76', lw=1.6,
                  label=r'Pheno SHIFT reference: 30 fb$^{-1}$')
        draw_strength_contours(axis, previous_strength, mode, '#929292',
                               'Previous prescription: generated grid', lw=2.2, linestyle='--')
        draw_strength_contours(axis, current_strength, mode, color,
                               'Updated prescription: generated grid', lw=2.6)
        axis.axhspan(.1, 1, color='#eeeeee', zorder=-10)
        axis.set(xscale='log', yscale='log', xlim=(.2, 100), ylim=((1.e-6, .3) if zoom else (1.e-7, 1)),
                 xlabel=r'$m_{A\prime}$ [GeV]',
                 title='SM background' if mode == 'background' else 'Zero background')
        axis.legend(loc='lower right', fontsize=8.2)
    axes[0].set_ylabel(r'Kinetic mixing $\epsilon$')
    figure.suptitle(r'DP sensitivity progress: expected 90% CLs, 0.3 fb$^{-1}$', fontsize=14)
    figure.text(.035, .02,
                'Matched assumptions: Both-Both, same 19 SM templates, no BDT/additional cuts; trigger/SFs = 1.\n'
                'Provisional 2% mass resolution; independent 30% signal/background normalization uncertainties; finite-MC uncertainty omitted.\n'
                'Grey: previous prescription; coloured: updated prescription. Both use independently fitted generated points.\n'
                'All current contours: linear log(r) interpolation in log(mass), log(epsilon), inside each generated grid hull.\n'
                'No template morphing or extrapolation. Sparse-MC and wide-triangle contours are provisional; blank support is unknown.',
                fontsize=8.1, va='bottom')
    figure.tight_layout(rect=(0, .20, 1, .94))
    destination = output / ('limits_before_after_zoom.pdf' if zoom else 'limits_before_after.pdf')
    figure.savefig(destination)
    plt.close(figure)
    return dict(inputs=[dict(directory=str(path), snapshot_sha256=digest(path / 'snapshot.json'),
                             scan_sha256=digest(path / 'scan.json')) for path in directories],
                displayed_reference_includes_saved_prompt_extrapolation=False,
                omitted_historical_curve=str(before),
                matched_backgrounds=closure, confidence_level=.9, channel='Both-Both',
                luminosity_pb=loaded[0][0]['luminosity_pb'], fits_rerun=True,
                method='Per-point fitted r, shared logarithmic triangulation; no template morphing',
                include_mc_stat_uncertainties=False,
                pheno_reference=dict(path=str(before / 'projection_inventory.json'),
                                     sha256=digest(before / 'projection_inventory.json'),
                                     drawing_index=pheno[0]['drawing_index'], luminosity_fb=30),
                output=dict(path=str(destination), sha256=digest(destination)))


def grid_cells(census, epoch):
    result = {}
    for row in census['rows']:
        if row['epoch'] != epoch:
            continue
        if row['event_histogram_readback'] != 'passed':
            raise ValueError('Unverified event-multiplicity histogram: ' + row['point'])
        identity = (row['mass_gev'], row['mean_lab_flight_m'])
        if identity in result:
            raise ValueError('Duplicate physical grid point: ' + row['point'])
        result[identity] = row
    return result


def draw_grid(census_path, output):
    census = read_json(census_path)
    before = grid_cells(census, census['primary_before_epoch'])
    published = grid_cells(census, census['primary_published_after_epoch'])
    qualified = grid_cells(census, census['benchmark_qualified_epoch'])
    identities = set(before) | set(published) | set(qualified)
    masses = sorted({mass for mass, flight in identities})
    flights = [None] + sorted({flight for mass, flight in identities if flight is not None})
    plt = plotting()
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch, Rectangle
    norm = LogNorm(vmin=1, vmax=1000)
    cmap = plt.get_cmap('viridis')

    def panel(axis, cells, title, hatch_legacy):
        axis.set(xlim=(-.5, len(masses) - .5), ylim=(len(flights) - .5, -.5),
                 xticks=range(len(masses)), yticks=range(len(flights)), title=title,
                 xlabel=r'$m_{A\prime}$ [GeV]')
        axis.set_xticklabels([f'{mass:g}' for mass in masses], rotation=60, ha='right', fontsize=8)
        axis.set_yticklabels(['prompt' if flight is None else f'{flight:g} m' for flight in flights])
        axis.grid(False)
        axis.set_ylabel('Reference mean lab flight')
        for i, mass in enumerate(masses):
            for j, flight in enumerate(flights):
                row = cells.get((mass, flight))
                if row is None:
                    color = 'white'
                    label = '\u2014'
                else:
                    value = row['event_neff']
                    if value is None or value < 0 or not np.isfinite(value):
                        raise ValueError('Invalid effective statistics: ' + row['point'])
                    color = '#dddddd' if value == 0 else cmap(norm(max(value, 1)))
                    label = '0' if value == 0 else (f'{value:.1f}' if value < 10 else f'{value:.0f}')
                axis.add_patch(Rectangle((i - .5, j - .5), 1, 1,
                                         facecolor=color, edgecolor='#d4d4d4', linewidth=.6))
                if row and hatch_legacy and row['model_family'] == 'legacy_NNPDF_DY':
                    axis.add_patch(Rectangle((i - .5, j - .5), 1, 1,
                                             facecolor='none', edgecolor='#999999',
                                             linewidth=.1, hatch='///', alpha=.65))
                value = row['event_neff'] if row else 0
                textcolor = 'white' if row and 0 < value < 100 else '#111111'
                axis.text(i, j, label, ha='center', va='center', color=textcolor, fontsize=7.4)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    panel(axes[0], before, '29 October: 38 available anchors', False)
    panel(axes[1], published, 'Now: 81 available anchors', True)
    figure.suptitle('DP grid progress: effective weighted Both-Both passing events', fontsize=14)
    figure.subplots_adjust(left=.065, right=.92, bottom=.29, top=.83, wspace=.07)
    colorbar_axis = figure.add_axes([.935, .29, .012, .54])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(scalar, cax=colorbar_axis, label=r'$N_{\mathrm{eff}}=(\sum w)^2/\sum w^2$')
    figure.legend(handles=[
        Patch(facecolor='white', edgecolor='#aaaaaa', label='Dash: no input in this snapshot'),
        Patch(facecolor='#dddddd', edgecolor='#aaaaaa', label='0: no passing MC events observed'),
        Patch(facecolor='white', edgecolor='#999999', hatch='///', label='Previous DY model: retained; MSHT20 replacement pending')],
        loc='lower center', bbox_to_anchor=(.5, .13), ncol=3, frameon=False, fontsize=8.2)
    figure.text(.045, .025,
                'Statistics from actual weighted event-multiplicity histograms; raw MC event counts are not reconstructed from weighted sums.\n'
                'These are simulation statistics, not events normalized to lumi. Source-collision correlations can further reduce independent precision.\n'
                'Same axes and color scale. Changes include added points and changed models; they do not by themselves establish improved efficiency.',
                fontsize=8.2, va='bottom')
    destination = output / 'signal_grid_statistics.pdf'
    figure.savefig(destination)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    panel(axes[0], published, 'Available grid: 81 anchors', True)
    panel(axes[1], qualified, 'Qualified new-benchmark support: 43 anchors', False)
    figure.suptitle('Available samples and new-model readiness are separate', fontsize=14)
    figure.subplots_adjust(left=.065, right=.92, bottom=.28, top=.83, wspace=.07)
    colorbar_axis = figure.add_axes([.935, .28, .012, .55])
    figure.colorbar(scalar, cax=colorbar_axis, label='Effective weighted passing events')
    figure.text(.045, .025,
                'Left: compatible SensCalc plus retained previous-DY samples. Right: 41 SensCalc anchors plus the two new MSHT20 prompt tests.\n'
                'A missing replacement does not mean old events were lost. The new 1.5 GeV prompt mixture has 3 raw MC passing events, N_eff = 1.3.\n'
                'The new 70 GeV prompt control has 19 raw MC passing events, N_eff = 18.9. These controls are not larger replacements for the old grid.\n'
                'Zero observed MC response is not zero physical efficiency. Source-collision clustering is included for the 1.5 GeV mixture only.',
                fontsize=8.2, va='bottom')
    benchmark = output / 'signal_model_readiness.pdf'
    figure.savefig(benchmark)
    plt.close(figure)
    return dict(census=dict(path=str(census_path), sha256=digest(census_path)),
                epochs=[census['primary_before_epoch'], census['primary_published_after_epoch'],
                        census['benchmark_qualified_epoch']],
                input_points=[len(before), len(published), len(qualified)],
                missing_is_zero=False, counts_normalized_to_luminosity=False,
                outputs=[dict(path=str(path), sha256=digest(path))
                         for path in [destination, benchmark]])


def with_retained_statistics(current, retained):
    """Keep unchanged references; replace, never add, different model populations."""
    selected={}
    for group,reference in [(retained['rows'],True),(current['rows'],False)]:
        seen=set()
        for original in group:
            name=original.get('name',original.get('point'))
            if not name or name in seen:raise ValueError('Repeated or missing statistics point name')
            if original['event_histogram_readback']!='passed':raise ValueError('Unverified statistics input: '+name)
            seen.add(name)
            selected[name]=dict(original,retained_reference=reference)
    return dict(current,rows=list(selected.values()),title='Now: total available inputs',
                qualified_anchors=len(current['rows']),
                retained_reference_anchors=sum(r['retained_reference'] for r in selected.values()))


def draw_snapshot_statistics(before, current, output, cutoff):
    """Show stored weighted passing statistics, with model changes explicit."""
    snapshots=[read_json(before),read_json(current)]
    rows=[snapshot['rows'] for snapshot in snapshots]
    masses=sorted({row['mass_gev'] for group in rows for row in group})
    def flight_label(row):
        # Keep nominal sample labels together despite small boost differences.
        name=row.get('name',row.get('point',''))
        match=re.search(r'_lab_([0-9p]+)m$',name)
        return float(match[1].replace('p','.')) if match else row.get('mean_lab_flight_m')
    flights=[None]+sorted({flight_label(row) for group in rows for row in group
                          if flight_label(row) is not None})
    plt=plotting()
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Patch,Rectangle
    cells=[];maximum=10.
    for group in rows:
        lookup={}
        for row in group:
            key=(row['mass_gev'],flight_label(row))
            if key in lookup:raise ValueError('Repeated physical point in statistics map')
            if row['event_histogram_readback']!='passed':raise ValueError('Missing event-histogram readback')
            value=row['event_neff']
            maximum=max(maximum,value);lookup[key]=(value,row)
        cells.append(lookup)
    norm=LogNorm(1,10**math.ceil(math.log10(maximum)))
    cmap=plt.get_cmap('viridis')
    figure,axes=plt.subplots(1,2,figsize=(15,6.1),sharey=True)
    for epoch,(axis,lookup,group) in enumerate(zip(axes,cells,rows)):
        for i,mass in enumerate(masses):
            for j,flight in enumerate(flights):
                cell=lookup.get((mass,flight));value=cell[0] if cell else None
                color='white' if value is None else '#dddddd' if value==0 else cmap(norm(value))
                axis.add_patch(Rectangle((i-.5,j-.5),1,1,facecolor=color,edgecolor='#cccccc',linewidth=.5))
                if cell and (cell[1].get('retained_reference', False) or
                             (epoch==0 and 'drell-yan' in cell[1]['mechanism'])):
                    axis.add_patch(Rectangle((i-.5,j-.5),1,1,facecolor='none',edgecolor='#999999',
                                             hatch='///',linewidth=.1,alpha=.6))
                label='\u2014' if value is None else '0' if value==0 else f'{value:.1f}' if value<10 else f'{value:.0f}'
                axis.text(i,j,label,ha='center',va='center',fontsize=7.3,
                          color='white' if value is not None and 0<value<100 else '#111111')
        axis.set(xlim=(-.5,len(masses)-.5),ylim=(len(flights)-.5,-.5),
                 xticks=range(len(masses)),yticks=range(len(flights)),xlabel=r'$m_{A\prime}$ [GeV]',
                 title=('29 October: available inputs' if epoch==0 else snapshots[1].get('title', 'Now: qualified benchmark'))+f' ({len(group)} anchors)')
        axis.set_xticklabels([f'{mass:g}' for mass in masses],rotation=60,ha='right',fontsize=8)
        axis.set_yticklabels(['prompt' if flight is None else f'{flight:g} m' for flight in flights]);axis.grid(False)
    axes[0].set_ylabel('Nominal mean lab-flight label')
    figure.suptitle('DP simulation statistics: effective weighted Both-Both passing events',fontsize=14)
    figure.subplots_adjust(left=.06,right=.92,bottom=.30,top=.85,wspace=.07)
    bar=figure.add_axes([.935,.30,.012,.55]);figure.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='Effective weighted passing events')
    figure.legend(handles=[Patch(facecolor='white',edgecolor='#aaa',label='Dash: no input in this snapshot'),
        Patch(facecolor='#ddd',edgecolor='#aaa',label='0: no passing MC event observed'),
        Patch(facecolor='white',edgecolor='#999',hatch='///',label='Retained previous-model reference (not pooled with new model)')],
        loc='lower center',bbox_to_anchor=(.5,.16),ncol=3,frameon=False,fontsize=8.5)
    figure.text(.035,.025,
        'Current catalog cutoff: '+cutoff+'. Same axes and colour scale; weights and original GEN exposure are preserved.\n'
        'N_eff = (sum w)^2 / sum w^2 from positive event-multiplicity bins, including overflow; each event is counted once.\n'
        'These are MC statistics, not raw event counts or events normalized to lumi; source-collision correlations can reduce independent precision.\n'
        'Current totals retain previous-model inputs where no replacement exists; hatching identifies retained references, not new-model qualification.\n'
        'Lab-flight rows use the nominal sample labels; exact epsilon, proper lifetime and reference boost remain in the saved inputs.',fontsize=8.1,va='bottom')
    path=output/'signal_statistics_before_after.pdf';figure.savefig(path);plt.close(figure)
    return dict(path=str(path),sha256=digest(path),before_anchors=len(rows[0]),current_anchors=len(rows[1]),
                missing_is_zero=False,source_collision_correlations_generally_omitted=True)


def signal_strength_contours(snapshot, scan):
    """Use the same generated positive-response brackets as the limit curves."""
    return supported_contour_rows(selected_rows(scan),snapshot['signals'])


def evaluated_strength_grid(directory, confidence_level=.9):
    """Fit only generated hypotheses; interpolation never changes a template."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from shift_limits_current import Model
    from shift_limits_profile import expected_limit, profile_q
    snapshot, scan = checked_snapshot(directory)
    config = SimpleNamespace(output_path=str(directory), luminosity=snapshot['luminosity_pb'],
        categories=snapshot['categories'], primary_channel='Both-Both',
        response_anchors=tuple(r['name'].split('_', 1)[1] for r in snapshot['signals']),
        include_mc_stat_uncertainties=False)
    model = Model(config)
    modes = ('background', 'zero_background')
    zero_limit, threshold = expected_limit(np.ones(1), np.zeros(1), confidence_level)

    def evaluate(item):
        mass, anchor = item
        epsilon = anchor['epsilon']
        records = []
        for mode in modes:
            signal, background = model.cells(mass, epsilon, 'Both-Both', mode)
            total = float(signal.sum())
            if total <= 0:
                raise ValueError('Selected positive-response anchor became empty')
            # Solve in event-yield units to avoid absolute root tolerances on tiny r.
            limit = zero_limit if mode == 'zero_background' else expected_limit(
                signal / total, background, confidence_level)[0]
            r = limit / total
            q = model.q(mass, epsilon, 'Both-Both', mode)
            if (r < 1) != (q > threshold):
                raise ValueError('r and the saved exclusion statistic disagree')
            if not math.isclose(profile_q(limit, signal / total, background), threshold,
                                rel_tol=2.e-5, abs_tol=2.e-6):
                raise ValueError('Signal-strength root failed likelihood readback')
            records.append(dict(name=anchor['name'], mass_gev=mass, epsilon=epsilon,
                mode=mode, r=r, predicted_events_normalized_to_lumi=total,
                excluded=r < 1, q=q, histogram_sha256=anchor['sha256']))
        return records

    items = [(mass, anchor) for mass, anchors in sorted(model.signals.items()) for anchor in anchors]
    records = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for result in workers.map(evaluate, items):
            records.extend(result)
    for record in records:
        saved=next(row for row in selected_rows(scan) if row['mass_gev']==record['mass_gev']
                   and row['mode']==record['mode'])
        matches=[q for epsilon,q in zip(saved['epsilon_grid'],saved['q_grid'])
                 if math.isclose(epsilon,record['epsilon'],rel_tol=1.e-12)]
        if not matches or not all(math.isclose(q,record['q'],rel_tol=1.e-7,abs_tol=1.e-8) for q in matches):
            raise ValueError('Anchor does not reproduce saved exclusion scan: '+record['name'])
    return snapshot, scan, records, model


def draw_signal_strength_map(directory, output, confidence_level=.9, reference_directory=None):
    import csv
    snapshot, scan, records, model = evaluated_strength_grid(directory, confidence_level)
    supported_rows = []  # Contours come exclusively from the fitted-r surface.
    modes = ('background', 'zero_background')
    masses=sorted({r['mass_gev'] for r in snapshot['signals']})
    if reference_directory is not None:
        reference=read_json(reference_directory/'snapshot.json')
        masses=sorted(set(masses) | {r['mass_gev'] for r in reference['signals']})
    gaps = []
    for mass in masses:
        for mode in modes:
            anchors = sorted((r for r in records if r['mass_gev']==mass and r['mode']==mode),
                             key=lambda r:r['epsilon'])
            brackets = [[a['epsilon'], b['epsilon']] for a,b in zip(anchors, anchors[1:])
                        if (a['r']-1)*(b['r']-1)<0]
            status = ('no positive-response new-model anchor' if not anchors else
                      'bracketed' if brackets else 'need smaller epsilon' if all(r['r']<1 for r in anchors)
                      else 'no excluded anchor; extend coverage or improve sensitivity')
            saved=next((r for r in supported_rows if r['mass_gev']==mass and r['mode']==mode),None)
            gaps.append(dict(mass_gev=mass, mode=mode, anchors=len(anchors),
                             r1_brackets=brackets, interpolated_roots=saved['roots'] if saved else [],
                             bracket_epsilon_ratios=[b/a for a,b in brackets], status=status))
    excluded_masses = sorted({r['mass_gev'] for r in snapshot['signals']} - set(model.signals))
    report = dict(definition='r = expected signal-yield upper limit / predicted signal yield',
        confidence_level=confidence_level, luminosity_pb=snapshot['luminosity_pb'], channel='Both-Both',
        snapshot_sha256=digest(directory/'snapshot.json'), scan_sha256=digest(directory/'scan.json'),
        include_mc_stat_uncertainties=False, mass_interpolation=False,
        unknown_response_masses=excluded_masses, zero_observed_mc=model.unused_signals,
        records=records, coverage=gaps,
        physical_domain=dict(mass_boundary_gev=12., below_maximum_epsilon=.03, above_maximum_epsilon=.1))
    save_json(output/'signal_strength_map.json', report)
    with (output/'signal_strength_anchors.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report['figure'] = draw_root_strength_map(report, output)
    save_json(output/'signal_strength_map.json', report)
    return report


def strength_interpolator(records, mode):
    """The shared fitted-r field in logarithmic physical coordinates."""
    from scipy.interpolate import LinearNDInterpolator
    rows = [r for r in records if r['mode'] == mode]
    points = np.array([[math.log10(r['mass_gev']), math.log10(r['epsilon'])] for r in rows])
    r = np.array([row['r'] for row in rows])
    if not np.isfinite(r).all() or np.any(r<=0):
        raise ValueError('Signal-strength anchors must have finite positive r')
    values = np.log10(r)
    return LinearNDInterpolator(points, values), points, values


def strength_surface(records, mode, bins=220, physical_domain=None):
    """Sample the shared field for COLZ; never extrapolate outside its hull."""
    interpolator, _, _ = strength_interpolator(records, mode)
    all_points = np.array([[r['mass_gev'], r['epsilon']] for r in records])
    edges = [np.geomspace(all_points[:, i].min(), all_points[:, i].max(), bins+1)
             for i in range(2)]
    centres = [np.sqrt(e[:-1]*e[1:]) for e in edges]
    xx, yy = np.meshgrid(*[np.log10(c) for c in centres])
    surface = interpolator(xx, yy)
    if physical_domain:
        cap = np.where(10.**xx < physical_domain['mass_boundary_gev'],
                       physical_domain['below_maximum_epsilon'], physical_domain['above_maximum_epsilon'])
        surface[10.**yy > cap] = np.nan
    return edges, centres, surface


def clip_strength_domain(segments, domain):
    """Clip log-linear contour edges to the pinned model's epsilon domains."""
    if not domain:
        return segments
    xm = math.log10(domain['mass_boundary_gev'])
    yl = math.log10(domain['below_maximum_epsilon'])
    yh = math.log10(domain['above_maximum_epsilon'])
    output = []
    for segment in segments:
        active = []
        for left, right in zip(np.log10(segment[:-1]), np.log10(segment[1:])):
            delta = right-left
            cuts = sorted({0., 1.} | {float((edge-left[axis])/delta[axis])
                for axis,edge in [(0,xm),(1,yl),(1,yh)] if delta[axis]!=0
                and 0 < (edge-left[axis])/delta[axis] < 1})
            for a,b in zip(cuts,cuts[1:]):
                middle = left+(a+b)/2*delta
                if middle[1] <= (yl if middle[0]<xm else yh)+1.e-12:
                    start, end = left+a*delta, left+b*delta
                    if active and not np.allclose(active[-1],start,rtol=0,atol=1.e-10):
                        output.append(10.**np.asarray(active)); active=[]
                    if not active:active.append(start)
                    active.append(end)
                elif active:
                    output.append(10.**np.asarray(active)); active=[]
        if active:output.append(10.**np.asarray(active))
    return output


def strength_contour_segments(report, mode):
    """Extract r=1 on the actual triangles, avoiding raster/hull artifacts."""
    interpolator, points, values = strength_interpolator(report['records'], mode)
    if values.min()>0 or values.max()<0:
        return []
    from matplotlib.tri import Triangulation
    triangles = Triangulation(points[:, 0], points[:, 1], interpolator.tri.simplices)
    plt = plotting()
    figure, axis = plt.subplots()
    contour = axis.tricontour(triangles, values, levels=[0.])
    segments = [10.**segment.copy() for segment in contour.allsegs[0] if len(segment)>1]
    plt.close(figure)
    return clip_strength_domain(segments, report.get('physical_domain'))


def draw_strength_contours(axis, report, mode, color, label, **style):
    for i, segment in enumerate(strength_contour_segments(report, mode)):
        axis.plot(segment[:, 0], segment[:, 1], color=color,
                  label=label if i==0 else None, **style)


def draw_root_strength_map(report, output):
    """ROOT COLZ display and masked r=1 contours from evaluated anchors.

    Likelihoods are evaluated only at generated points. All updated figures
    use the same triangulated strength surface and its r=1 contour.
    """
    import ROOT
    from array import array
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kBird)
    target = output/'signal_strength_map.pdf'
    root_path = output/'signal_strength_map.root'
    root_file = ROOT.TFile(str(root_path), 'RECREATE')
    canvas = ROOT.TCanvas('signal_strength', 'Expected signal-strength limit', 1500, 750)
    canvas.SetCanvasSize(1500, 750)
    canvas.Divide(2, 1)
    keep = []
    contour_counts = {}
    for index, mode in enumerate(('background', 'zero_background'), 1):
        edges, centres, surface = strength_surface(report['records'], mode, physical_domain=report.get('physical_domain'))
        title = 'SM background' if mode == 'background' else 'Zero background'
        hist = ROOT.TH2D('r_'+mode, title+";m_{A'} [GeV];Kinetic mixing #epsilon;r",
                         len(edges[0])-1, array('d', edges[0]),
                         len(edges[1])-1, array('d', edges[1]))
        mask = hist.Clone('support_'+mode)
        mask.Reset()
        hist.SetDirectory(0)
        mask.SetDirectory(0)
        for iy, ix in np.argwhere(np.isfinite(surface)):
            hist.SetBinContent(int(ix)+1, int(iy)+1, 10.**float(surface[iy, ix]))
            mask.SetBinContent(int(ix)+1, int(iy)+1, 1.)
        hist.SetMinimum(1.e-3)
        hist.SetMaximum(1.e3)
        pad = canvas.cd(index)
        pad.SetLogx(); pad.SetLogy(); pad.SetLogz()
        pad.SetLeftMargin(.13); pad.SetRightMargin(.18); pad.SetBottomMargin(.16)
        hist.GetYaxis().SetTitleOffset(1.4)
        hist.Draw('COLZ')
        # A masked contour avoids spurious r=1 edges against empty ROOT bins.
        segments = strength_contour_segments(report, mode)
        graphs = []
        for segment in segments:
            if len(segment) < 2:
                continue
            graph = ROOT.TGraph(len(segment), array('d', segment[:, 0]),
                                array('d', segment[:, 1]))
            graph.SetName('r1_'+mode+'_'+str(len(graphs)))
            graph.SetLineWidth(3); graph.SetLineColor(ROOT.kBlack)
            graph.Draw('L SAME'); graph.Write(); graphs.append(graph)
        anchors = [r for r in report['records'] if r['mode'] == mode]
        markers = ROOT.TGraph(len(anchors), array('d', [r['mass_gev'] for r in anchors]),
                             array('d', [r['epsilon'] for r in anchors]))
        markers.SetName('anchors_'+mode); markers.SetMarkerStyle(20)
        markers.SetMarkerSize(.35); markers.Draw('P SAME'); markers.Write()
        label = ROOT.TLatex(); label.SetNDC(); label.SetTextSize(.028)
        empty = [r for r in report.get('zero_observed_mc', [])
                 if r.get('unused_reason', '').startswith('Zero observed')]
        unknown = ROOT.TGraph(len(empty), array('d', [r['mass_gev'] for r in empty]),
                              array('d', [r['epsilon'] for r in empty]))
        unknown.SetName('unknown_MC_'+mode); unknown.SetMarkerStyle(5)
        unknown.SetMarkerColor(ROOT.kGray+2); unknown.SetMarkerSize(.7)
        unknown.Draw('P SAME'); unknown.Write(); keep.append(unknown)
        label.SetTextSize(.024)
        label.DrawLatex(.13, .09, 'Black: r = 1; dots: anchors; grey x: empty MC')
        label.DrawLatex(.13, .05, 'Log(r) interpolation; blank: outside grid hull')
        hist.Write(); mask.Write()
        keep.extend([hist, mask, markers, label]+graphs)
        contour_counts[mode] = len(graphs)
    canvas.cd()
    label = ROOT.TLatex(); label.SetNDC(); label.SetTextSize(.025)
    label.DrawLatex(.12, .975, 'Both-Both, 90% CLs, '+str(report['luminosity_pb']/1000)+' fb^{-1}')
    ROOT.gStyle.SetPaperSize(26., 13.)
    canvas.SaveAs(str(target)); canvas.Write()
    metadata = dict(interpolation='Per-point likelihood fits, then linear log10(r) on the log10(mass), log10(epsilon) triangulation',
                    extrapolation=False,
                    blank_bins='Outside the grid hull or pinned physical epsilon domain',
                    physical_domain=report.get('physical_domain'),
                    contour='r=1 on the interpolated surface; no contours against blank bins',
                    contour_segments=contour_counts,
                    snapshot_sha256=report['snapshot_sha256'])
    ROOT.TNamed('interpolation_metadata', json.dumps(metadata)).Write()
    canvas.Close()
    root_file.Close()
    report['mass_interpolation'] = True
    report['root_surface'] = metadata
    return dict(path=str(target), sha256=digest(target), root_path=str(root_path),
                root_sha256=digest(root_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--census', type=Path)
    parser.add_argument('--signal-strength-map', action='store_true', help='Evaluate r at every generated anchor')
    parser.add_argument('--zoom', action='store_true', help='Focus the epsilon axis on SHIFT sensitivity')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.signal_strength_map:
        draw_signal_strength_map(arguments.after.resolve(), arguments.output.resolve(), reference_directory=arguments.legacy.resolve())
    report = draw_limits(arguments.before.resolve(), arguments.legacy.resolve(),
                         arguments.after.resolve(), arguments.output.resolve(), zoom=arguments.zoom)
    if arguments.signal_strength_map:
        report['signal_strength'] = draw_signal_strength_map(arguments.after.resolve(), arguments.output.resolve(), reference_directory=arguments.legacy.resolve())
    if arguments.census:
        report['grid'] = draw_grid(arguments.census.resolve(), arguments.output.resolve())
    report['source'] = dict(path=str(Path(__file__).resolve()), sha256=digest(__file__))
    save_json(arguments.output / 'plot_validation.json', report)


if __name__ == '__main__':
    main()
