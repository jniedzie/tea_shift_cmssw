#!/usr/bin/env python3
# Code style: AI-code. AI-generated presentation plots of audited DP progress.
"""Compare frozen signal grids and limits without rerunning or changing fits.

The before/after limits must use matched exposure, mass bins and backgrounds.
Legacy DY and qualified new-model support are deliberately shown separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS / 'bin'))
from shift_limits_current_plots import draw_regions
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


def draw_limits(before, legacy, after, output):
    directories = [before, legacy, after]
    loaded = [checked_snapshot(path) for path in directories]
    closure = background_closure([item[0] for item in loaded])
    rows = [selected_rows(item[1]) for item in loaded]
    plt = plotting()
    from matplotlib.lines import Line2D
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
        draw_regions(axis, rows[0], mode, '#202020', '29 Oct: then-available samples',
                     fill=False, linewidth=1.8, linestyle=':')
        draw_regions(axis, rows[1], mode, '#929292', 'Now: completed grid, previous DY',
                     fill=False, linewidth=2.2, linestyle='--')
        draw_regions(axis, rows[2], mode, color, 'Now: qualified benchmark support',
                     fill=False, linewidth=2.6, linestyle='-', point_marker_size=38)
        axis.axhspan(.1, 1, color='#eeeeee', zorder=-10)
        axis.set(xscale='log', yscale='log', xlim=(.2, 100), ylim=(1.e-7, 1),
                 xlabel=r'$m_{A\prime}$ [GeV]',
                 title='SM background' if mode == 'background' else 'Zero background')
        axis.legend(loc='lower right', fontsize=8.2)
    axes[0].set_ylabel(r'Kinetic mixing $\epsilon$')
    figure.suptitle(r'DP sensitivity progress: expected 90% CLs, 0.3 fb$^{-1}$', fontsize=14)
    figure.legend(handles=[
        Line2D([], [], marker='D', markerfacecolor='white', markeredgecolor='black',
               color='none', label='Diamond: tested lifetime only; open = not excluded, filled = excluded')],
        loc='lower center', bbox_to_anchor=(.5, .115), fontsize=9, frameon=False)
    figure.text(.035, .02,
                'Matched assumptions: Both-Both, same 19 SM templates, no BDT/additional cuts; trigger/SFs = 1.\n'
                'Provisional 2% mass resolution; independent 30% signal/background normalization uncertainties; finite-MC uncertainty omitted.\n'
                'New DY: prompt tests at 1.5 and 70 GeV, epsilon = 0.001 only. Missing new DY lifetimes are not supplied by the legacy curves.\n'
                'Mass connections are interpolation; support endpoints are not physical boundaries. Grey band: epsilon > 0.1 extrapolation.',
                fontsize=8.1, va='bottom')
    figure.tight_layout(rect=(0, .18, 1, .94))
    destination = output / 'limits_before_after.pdf'
    figure.savefig(destination)
    plt.close(figure)
    return dict(inputs=[dict(directory=str(path), snapshot_sha256=digest(path / 'snapshot.json'),
                             scan_sha256=digest(path / 'scan.json')) for path in directories],
                matched_backgrounds=closure, confidence_level=.9, channel='Both-Both',
                luminosity_pb=loaded[0][0]['luminosity_pb'], fits_rerun=False,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--census', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    report = draw_limits(arguments.before.resolve(), arguments.legacy.resolve(),
                         arguments.after.resolve(), arguments.output.resolve())
    if arguments.census:
        report['grid'] = draw_grid(arguments.census.resolve(), arguments.output.resolve())
    report['source'] = dict(path=str(Path(__file__).resolve()), sha256=digest(__file__))
    save_json(arguments.output / 'plot_validation.json', report)


if __name__ == '__main__':
    main()
