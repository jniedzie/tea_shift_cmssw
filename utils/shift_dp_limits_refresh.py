#!/usr/bin/env python3
# Code style: AI-code. AI-generated bounded refresh of qualified DP limits.
"""Freeze a ready-signal catalog, stage its EOS inputs and reuse the limits code.

This is a finite analysis job. It never generates events, changes aliases or
combines production populations. The qualified publisher owns those decisions.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Require an explicit SHA-256 publication pin')
    return value


def native_url(path):
    """Normalize the user mount alias before choosing its explicit EOS instance."""
    from shift_eos_storage import xrootd_url
    path = str(path)
    match = re.match(r'^/eos/home-([a-z])/([^/]+)(/.*)?$', path)
    if match:
        path = '/eos/user/' + match[1] + '/' + match[2] + (match[3] or '')
    return xrootd_url(path)


def select_signals(catalog, baseline):
    if (catalog.get('schema') != 'shift-dp-qualified-signal-catalog-v1'
            or catalog.get('complete') is not True):
        raise ValueError('Require a complete qualified-signal catalog, even during partial production')
    retained = {row['name']: row for row in baseline['signals'] if row['mass_gev'] < 1.5}
    result = []
    names = set()
    for original in catalog['ready_signals']:
        row = copy.deepcopy(original)
        name = row['name']
        if name in names:
            raise ValueError('Catalog repeats a physical point: ' + name)
        names.add(name)
        if (not 2 * .1056583755 < row['mass_gev'] < 91.1876
                or not math.isfinite(row['epsilon']) or not 0 < row['epsilon'] <= .1
                or not row.get('full_benchmark_components')):
            raise ValueError('Point is outside the qualified complete benchmark: ' + name)
        if not row.get('cumulative_compatible_inputs_included'):
            raise ValueError('A latest small increment cannot replace cumulative exposure: ' + name)
        check_digest(row['histogram_sha256'])
        check_digest(row['rate_sha256'])
        check_digest(row['qualification']['sha256'])
        if type(row['histogram_bytes']) is not int or not 0 < row['histogram_bytes'] <= 64 * 1024**2:
            raise ValueError('Invalid bounded histogram publication size: ' + name)
        if row.get('retained_reference'):
            previous = retained.get(name)
            if (previous is None or row['histogram_sha256'] != previous['sha256']
                    or row['rate_sha256'] != previous['rate_authority_sha256']):
                raise ValueError('Retained point differs from its exact reference: ' + name)
        result.append(row)
    missing = sorted(set(retained) - names)
    if missing:
        raise ValueError('Catalog omits compatible retained SensCalc exposure: ' + ', '.join(missing))
    return sorted(result, key=lambda row: (row['mass_gev'], row['epsilon'], row['name']))


class Stager:
    def __init__(self, directory, maximum_bytes):
        self.directory = directory
        self.maximum_bytes = maximum_bytes
        self.bytes = 0
        self.records = []
        self.cache = {}

    def stage(self, path, expected, *, size=None, url=None):
        check_digest(expected)
        key = (str(path), expected)
        if key in self.cache:
            return self.cache[key]
        destination = self.directory / (str(len(self.records)) + '_' + PurePosixPath(str(path)).name)
        if str(path).startswith('/eos/'):
            canonical = native_url(path)
            if url is not None:
                supplied=urlsplit(url)
                if (supplied.query or supplied.fragment
                        or supplied.netloc!=urlsplit(canonical).netloc
                        or native_url('/'+supplied.path.lstrip('/'))!=canonical):
                    raise ValueError('Publication URL differs from its explicit canonical storage path')
            subprocess.run(['/usr/bin/xrdcp', '--silent', '--cksum', 'adler32', canonical, str(destination)],
                           timeout=120, check=True)
        else:
            source = Path(path)
            if not source.is_absolute() or not str(source).startswith('/afs/cern.ch/work/j/jniedzie/private/shift_cmssw/'):
                raise ValueError('A non-EOS authority must be an explicit workspace file')
            shutil.copy2(source, destination)
        observed = destination.stat().st_size
        if (size is not None and observed != size) or digest(destination) != expected:
            raise ValueError('Publication readback differs from its pinned bytes: ' + str(path))
        self.bytes += observed
        if self.bytes > self.maximum_bytes:
            raise ValueError('Analysis input staging exceeded its bounded storage budget')
        self.records.append(dict(path=str(path), sha256=expected, bytes=observed))
        self.cache[key] = destination
        return destination


def qualify_row(row, qualification, baseline_sha):
    if qualification.get('complete') is not True:
        raise ValueError('Incomplete source qualification: ' + row['name'])
    if row.get('retained_reference'):
        if qualification.get('schema') != 'shift-dp-retained-SensCalc-qualification-v1':
            raise ValueError('Retained source lacks its frozen SensCalc qualification')
        if qualification['reference_snapshot']['sha256'] != baseline_sha:
            raise ValueError('Retained source qualification names another reference snapshot')
        point = qualification['points'][row['name']]
        if (point['histogram_sha256'] != row['histogram_sha256']
                or point['rate_sha256'] != row['rate_sha256']):
            raise ValueError('Retained publication differs from its qualified point')
    elif (qualification.get('schema') != 'shift-dp-cumulative-publication-qualification-v1'
            or qualification.get('point') != row['name']
            or qualification.get('input_signature') != row['input_signature']
            or qualification.get('all_ready_compatible_inputs_included') is not True
            or qualification.get('mechanism_denominators_retained') is not True):
        raise ValueError('New source lacks complete cumulative component qualification: ' + row['name'])


def photon_benchmark_domain(rate):
    """Require the explicit comparison approximation before widening photon rates."""
    approximation = rate.get('photon_benchmark_approximation')
    if rate.get('experimental_photon_benchmark') is not True:
        if approximation is not None:
            raise ValueError('Photon benchmark metadata requires its explicit experimental flag')
        return None
    expected = dict(schema='shift-experimental-photon-benchmark-epsilon-v1',
        maximum_epsilon=.03, source_scaling='epsilon squared',
        physical_width='Unchanged pinned DarkCast photon-like spectral widths',
        neutral_current_width_included=False)
    if (not isinstance(approximation, dict)
            or any(approximation.get(k) != v for k, v in expected.items())
            or approximation.get('neutral_current_width_included') is not False):
        raise ValueError('Unqualified experimental photon-benchmark approximation')
    return dict(maximum_epsilon=.03, prescription='experimental-photon-benchmark',
                experimental_photon_benchmark=True, approximation=copy.deepcopy(approximation))


def rate_epsilon_domain(rate):
    """Read each pinned rate's qualified domain; a v2 point never widens v1."""
    if rate.get('schema') == 'shift-dark-photon-senscalc-dy-component-rates-v1':
        domains = [rate_epsilon_domain(component['rate']) for component in rate['components']]
        return dict(maximum_epsilon=min(d['maximum_epsilon'] for d in domains),
                    prescription='qualified-component-intersection', components=domains)
    photon = photon_benchmark_domain(rate)
    if rate.get('mechanism') == 'senscalc-meson-fragmentation-bremsstrahlung':
        return photon or dict(maximum_epsilon=.01, prescription='SensCalc-photon-benchmark')
    prescription = rate.get('dy_prescription', {})
    name = prescription.get('name')
    hashes = {
        'msht20-nnlo': ('18fac0c030160c00d8fd50a3ce145a5e435546a5b3426d6caa4942af1c2384e5',),
        'msht20-nnlo-v2': ('c238aaeb64640dab129cf26b101f293ddcd34357d04d852fe50dffbd3587af5c',
                           '38a24e4a1c2533f2d306d815fa15114dac5f99f84c9bea6a98bb1228a8c0c44e')}
    if name not in hashes or prescription.get('authority', {}).get('sha256') not in hashes[name]:
        raise ValueError('Old NNPDF DY is not a qualified MSHT20 response')
    if name == 'msht20-nnlo':
        return dict(maximum_epsilon=.001, prescription=name)
    masses = (20, 30, 50) if prescription['authority']['sha256'] == hashes[name][-1] else (1.5, 1.8, 2, 5, 10, 15, 40, 70)
    if (prescription.get('epsilon_upper_bound') != .1 or rate['mass_gev'] not in masses):
        raise ValueError('Unqualified MSHT20-v2 mass or epsilon domain')
    return photon or dict(maximum_epsilon=.1 if rate['mass_gev'] >= 12 else .01,
                          prescription=name, spectral_photon_width=rate['mass_gev'] < 12)


def qualify_rate(row, rate):
    from shift_limits_current import signal_physical_point
    physical, mechanism = signal_physical_point(rate)
    for key in ('mass_gev', 'epsilon', 'proper_ctau_mm'):
        if not math.isclose(physical[key], row[key], rel_tol=1.e-10, abs_tol=0):
            raise ValueError('Catalog physical point differs from its rate authority: ' + row['name'])
    dy = rate
    if 1.5 <= row['mass_gev'] < 2:
        if rate.get('schema') != 'shift-dark-photon-senscalc-dy-component-rates-v1':
            raise ValueError('The overlap region requires the actual SensCalc plus DY component sum')
        dy = next(item['rate'] for item in rate['components'] if item['mechanism'] == 'pure-on-shell-drell-yan')
    elif row['mass_gev'] < 1.5:
        if mechanism != 'senscalc-meson-fragmentation-bremsstrahlung':
            raise ValueError('An old custom meson model cannot enter the pinned benchmark')
    # Selection admits candidates up to the largest qualified domain. Only this
    # hash-pinned authority check can grant a particular point's wider domain.
    if row['mass_gev'] >= 1.5:
        rate_epsilon_domain(dy)
    domain = rate_epsilon_domain(rate)
    if not 0 < row['epsilon'] <= domain['maximum_epsilon']:
        raise ValueError('Point exceeds its qualified epsilon domain: ' + row['name'])
    return domain


def settings(output, background, inventory, figure_directory):
    # These are the reviewed November 10 defaults consumed by snapshot/Model/scan.
    rows = read(inventory)['signals']
    return SimpleNamespace(output_path=str(output), background_snapshot=str(background),
        signal_input_inventory=str(inventory), signal_campaign='Qualified PBC/SensCalc/MSHT20 ready increments',
        campaign='20261016_mu_enriched_weekend_v1', base_path='', cross_sections={},
        categories=('Near-Both', 'Near-Far', 'Both-Both', 'Both-Far', 'Other'),
        luminosity=300., confidence_levels=(.9,), background_modes=('background', 'zero_background'),
        channel_names=('Both-Both',), primary_channel='Both-Both',
        mass_resolution_fraction=.02, mass_minimum_bin_gev=.1,
        include_mc_stat_uncertainties=False, scan_grid_points=65,
        epsilon_min=1.e-10, epsilon_max=max(r['epsilon_domain']['maximum_epsilon'] for r in rows),
        response_anchors=None, response_anchor_names=tuple(r['name'] for r in rows),
        plot_omitted_masses_gev=(), comparison_figure_directory=str(figure_directory))


def passing_event_statistics(path):
    """Count events with a positive candidate multiplicity, including overflow."""
    import ROOT
    name='event/Event_nShiftDimuonVertexBoth-Both'
    source=ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():raise ValueError('Cannot read staged event histogram')
    try:
        histogram=source.Get(name)
        if not histogram:raise ValueError('Missing event-multiplicity histogram: '+name)
        if histogram.GetXaxis().FindBin(0)!=1:
            raise ValueError('Expected zero candidates in the first multiplicity bin')
        bins=range(2,histogram.GetNbinsX()+2)
        values=[histogram.GetBinContent(index) for index in bins]
        variances=[histogram.GetBinError(index)**2 for index in bins]
        if any(not math.isfinite(value) or value<0 for value in values+variances):
            raise ValueError('Invalid weighted event-multiplicity moments')
        sumw=math.fsum(values);sumw2=math.fsum(variances)
        if sumw>0 and sumw2<=0:raise ValueError('Positive event yield lacks a variance')
        return dict(event_passing_sumw=sumw,event_passing_sumw2=sumw2,
            event_neff=sumw**2/sumw2 if sumw2 else 0.,event_histogram=name,
            event_histogram_readback='passed',raw_passing_events=None,
            event_neff_source_collision_correlations_included=False)
    finally:source.Close()


def mass_fingerprint(model, mass):
    rows = []
    for row in model.signals.get(mass, []):
        item = {key: row[key] for key in ('epsilon', 'proper_ctau_mm', 'cross_section_pb',
                                          'full_gen_sumw', 'prompt', 'mechanism')}
        for key in ('name','sha256','rate_authority_sha256'):
            item[key]=row.get(key)
        for suffix in ('', '_sumw2'):
            array = model.arrays[row['array_key'] + suffix]
            item['array' + suffix] = hashlib.sha256(array.tobytes()).hexdigest()
        rows.append(item)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def same_scan_support(row, anchors, epsilon_max):
    from shift_limits_current import epsilon_support
    low, high = epsilon_support(anchors, epsilon_max)
    return all(math.isclose(left,right,rel_tol=1.e-12,abs_tol=0.)
               for left,right in zip(row['epsilon_support'],[low,high]))


def reuse_scan(reference, model, config):
    from shift_limits_current import Model
    from shift_dp_progress_plots import background_closure
    baseline = read(reference / 'snapshot.json')
    background_closure([baseline, model.snapshot])
    reference_config = copy.copy(config)
    reference_config.output_path = str(reference)
    previous = Model(reference_config)
    scan = read(reference / 'scan.json')
    if scan['snapshot_sha256'] != digest(reference / 'snapshot.json') or scan.get('include_mc_stat_uncertainties', False):
        raise ValueError('Reference scan does not match its nominal frozen inputs')
    reused = []
    checks = []
    for mass in sorted(model.signals):
        if mass_fingerprint(previous, mass) != mass_fingerprint(model, mass):
            continue
        rows = [row for row in scan['records'] if row['mass_gev'] == mass
                and row['channel'] == config.primary_channel and row['confidence_level'] == .9]
        valid = True
        for row in rows:
            if not same_scan_support(row,model.signals[mass],config.epsilon_max):
                valid=False
                checks.append(dict(mass_gev=mass,mode=row['mode'],paired_profile_checks=False,
                                   reason='Generated lifetime or epsilon scan support changed'))
                break
            for index in sorted({0, len(row['epsilon_grid']) // 2, len(row['epsilon_grid']) - 1}):
                epsilon = row['epsilon_grid'][index]
                actual = model.q(mass, epsilon, config.primary_channel, row['mode'])
                expected = row['q_grid'][index]
                if actual is None or not math.isclose(actual, expected, rel_tol=2.e-7, abs_tol=1.e-8):
                    valid = False
                    break
            checks.append(dict(mass_gev=mass, mode=row['mode'], paired_profile_checks=valid))
        if valid and len(rows) == len(config.background_modes):
            reused.extend(copy.deepcopy(rows))
    write(model.output / 'scan.checkpoint.json', dict(records=reused,
          snapshot_sha256=digest(model.output / 'snapshot.json'), include_mc_stat_uncertainties=False))
    return dict(reference=str(reference), reference_scan_sha256=digest(reference / 'scan.json'),
                records_reused=len(reused), checks=checks,
                rule='Exact moment/normalization/support fingerprints and background arrays; paired profile checks at saved grid endpoints and midpoint')


def finite_mc_diagnostics(config, model):
    """Keep a bounded diagnostic beside the matched nominal contours."""
    from shift_limits_current import Model
    from shift_limits_profile import profile_q_mc
    current=copy.copy(config)
    current.include_mc_stat_uncertainties=True
    finite=Model(current)
    scanned=read(model.output/'scan.json')['records']
    result=[]
    for mass,anchors in sorted(model.signals.items()):
        if mass<1.5:continue
        for mode in config.background_modes:
            row=next(item for item in scanned if item['mass_gev']==mass and item['mode']==mode)
            epsilon=row['roots'][0] if row['roots'] else anchors[-1]['epsilon']
            signal,variance,backgrounds,background_variance,omissions=finite.mc_cells(
                mass,epsilon,config.primary_channel,mode)
            result.append(dict(mass_gev=mass,epsilon=epsilon,mode=mode,
                reference='first nominal supported root' if row['roots'] else 'highest generated epsilon; no supported root',
                nominal_q=model.q(mass,epsilon,config.primary_channel,mode),
                finite_MC_q=profile_q_mc(signal,variance,backgrounds,background_variance,
                                        allow_zero_background=mode=='zero_background'),
                source_MC_yield=float(signal.sum()),source_MC_variance=float(variance.sum()),
                omitted_bins=omissions))
    report=dict(records=result,
        nominal_curves_include_finite_MC=False,
        method='Existing effective-Poisson profile; source histogram variances and unchanged independent normalization nuisances',
        limitation='Histogram variances do not generally include source-collision clustering or cross-point covariance; zero SM process bins remain unbounded.')
    write(model.output/'finite_mc_diagnostics.json',report)
    return report


def render(config, model, legacy, before, output, cutoff, before_event_statistics):
    from shift_limits_current import Model
    from shift_limits_current_plots import draw_original_comparisons, draw_model_update, draw_inputs
    from shift_dp_progress_plots import (draw_limits,draw_snapshot_statistics,
                                         draw_signal_strength_map,with_retained_statistics)
    progress = dict(signal_strength=draw_signal_strength_map(output,output,reference_directory=legacy))
    from shift_dp_progress_plots import evaluated_strength_grid
    previous_strength = dict(records=evaluated_strength_grid(legacy)[2],
                             snapshot_sha256=digest(legacy/'snapshot.json'))
    write(output/'previous_signal_strength.json', previous_strength)
    reference = output / 'legacy_reference'
    reference.mkdir(exist_ok=True)
    for name in ('snapshot.json', 'scan.json'):
        shutil.copy2(legacy / name, reference / name)
    notes = [
        '0.3 fb^-1, Both-Both; no BDT/additional cuts; trigger efficiency and scale factors are 1.',
        'Grey: previous MC grid. New model: qualified SensCalc plus MSHT20 NNLO grid. Both interpolate fitted log(r).',
        'Catalog cutoff: ' + cutoff + '. Missing lifetimes and support endpoints are not physical exclusion edges.',
        'Independent 30% signal/background normalization uncertainties; finite-MC uncertainty omitted from these matched curves.',
        'Provisional 2% mass resolution. Literal SM histogram bins; no invented background tail or interpolation.',
        'Pheno SHIFT: 30 fb^-1. Most reference curves: 90% CL; FCC: 95% CL. Contours interpolate per-point log(r) in log(mass), log(epsilon); no template morphing or extrapolation.']
    for modes in [('background', 'zero_background'), ('background',), ('zero_background',)]:
        old = copy.copy(config)
        old.output_path = str(reference)
        old.background_modes = modes
        old.epsilon_max = 1.
        old.plot_omitted_masses_gev = (1.5,)
        old.signal_model_label = 'Previous DY (NNPDF)'
        old.include_prompt_reference_extrapolation = False
        old.strength_report = previous_strength
        old.comparison_legend_fontsize = 10
        old.comparison_mode_styles = [
            ('background', (.5, .5, .5), ': background', True),
            ('zero_background', (.68, .68, .68), ': zero background', True)]
        old.comparison_notes = notes
        old.comparison_notes_are_complete = True
        draw_original_comparisons(old, Model(old))
        current = copy.copy(config)
        current.background_modes = modes
        draw_model_update(current, model, reference)
    draw_inputs(config, model)
    progress['limits']=draw_limits(before, legacy, output, output)
    progress['zoomed_limits']=draw_limits(before,legacy,output,output,zoom=True)
    current_statistics=output/'event_statistics.json'
    retained_statistics=legacy/'event_statistics.json'
    if retained_statistics.exists():
        total=with_retained_statistics(read(current_statistics),read(retained_statistics))
        current_statistics=output/'current_total_event_statistics.json'
        write(current_statistics,total)
    progress['statistics_map']=draw_snapshot_statistics(before_event_statistics,
        current_statistics,output,cutoff)
    return progress


def check_pdfs(output):
    import fitz
    required={'limits_current_cl90_comparison.pdf',
        'limits_current_cl90_background_comparison.pdf',
        'limits_current_cl90_zero_background_comparison.pdf',
        'background_mass_templates.pdf','signal_coverage.pdf','limits_before_after.pdf',
        'limits_before_after_zoom.pdf','signal_strength_map.pdf','signal_statistics_before_after.pdf'}
    missing=required-{path.name for path in output.glob('*.pdf')}
    if missing:raise ValueError('Missing expected final figures: '+', '.join(sorted(missing)))
    records=[]
    for path in sorted(output.glob('*.pdf')):
        with fitz.open(path) as document:
            if len(document)!=1 or document[0].rect.width<300 or not document[0].get_drawings():
                raise ValueError('Final figure is missing its expected single plotted page: '+path.name)
            records.append(dict(path=str(path),sha256=digest(path),pages=len(document)))
    return records


def scan_generated_points(config, model):
    """Record likelihood checks only at the generated signal hypotheses."""
    records = []
    for mass, anchors in sorted(model.signals.items()):
        epsilons = [r['epsilon'] for r in anchors]
        for mode in config.background_modes:
            records.append(dict(mass_gev=mass, mode=mode, channel=config.primary_channel,
                confidence_level=.9, roots=[], excluded_intervals=[],
                epsilon_support=[min(epsilons),max(epsilons)],
                upper_prompt_continuation=False, epsilon_grid=epsilons,
                q_grid=[model.q(mass,e,config.primary_channel,mode) for e in epsilons],
                include_mc_stat_uncertainties=False, source='generated_hypotheses_only'))
    write(model.output/'scan.json', dict(records=records,
        snapshot_sha256=digest(model.output/'snapshot.json'),
        include_mc_stat_uncertainties=False, interpolation='None; generated hypotheses only'))


def refresh(args):
    from shift_limits_current import Model, snapshot
    started = time.monotonic()
    reference = args.reference.resolve()
    baseline = read(reference / 'snapshot.json')
    if digest(reference / 'snapshot.json') != args.reference_sha256:
        raise ValueError('Reference snapshot changed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    write(output / 'catalog_snapshot.json', catalog)
    for key in ('configuration', 'publisher_code'):
        pin = catalog[key]
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('Qualified catalog authority changed: ' + key)
    if catalog['configuration']['sha256'] != args.publisher_config_sha256:
        raise ValueError('Catalog belongs to another publisher configuration')
    selected = select_signals(catalog, baseline)
    report = dict(complete=False, catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                  catalog_cutoff_utc=catalog['updated_utc'], started_utc=datetime.now(timezone.utc).isoformat(),
                  signals=len(selected), pending_signals=catalog.get('pending_signals', []),
                  source=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)))
    write(output / 'refresh_status.json', report)
    with tempfile.TemporaryDirectory(prefix='shift-dp-limits-', dir=args.scratch_base) as temporary:
        scratch = Path(temporary)
        stager = Stager(scratch, args.maximum_input_bytes)
        inventory = []
        event_statistics=[]
        for row in selected:
            qualifier = row['qualification']
            proof = stager.stage(qualifier['path'], qualifier['sha256'])
            qualify_row(row, read(proof), args.reference_sha256)
            histogram = stager.stage(row['histogram_path'], row['histogram_sha256'],
                                     size=row['histogram_bytes'], url=row.get('histogram_url'))
            rate = stager.stage(row['rate_path'], row['rate_sha256'], url=row.get('rate_url'))
            domain = qualify_rate(row, read(rate))
            event_statistics.append(dict(name=row['name'],mass_gev=row['mass_gev'],
                mean_lab_flight_m=row.get('mean_lab_flight_m'),
                mechanism=read(rate).get('mechanism','senscalc-meson-fragmentation-bremsstrahlung-plus-DY'),
                histogram_path=row['histogram_path'],histogram_sha256=row['histogram_sha256'],
                **passing_event_statistics(histogram)))
            inventory.append(dict(name=row['name'], histogram_path=row['histogram_path'],
                local_histogram_path=str(histogram), sha256=row['histogram_sha256'],
                rate_path=row['rate_path'], local_rate_path=str(rate), rate_sha256=row['rate_sha256'],
                mean_lab_flight_m=row.get('mean_lab_flight_m'), epsilon_domain=domain,
                input_role='qualified ready cumulative benchmark'))
        local_inventory = scratch / 'local_inventory.json'
        write(local_inventory, dict(signals=inventory))
        write(output / 'signal_input_inventory.json', dict(signals=[
            {key: value for key, value in row.items() if not key.startswith('local_')} for row in inventory]))
        write(output/'event_statistics.json',dict(schema='shift-dp-passing-event-statistics-v1',
            catalog_cutoff_utc=catalog['updated_utc'],rows=event_statistics))
        config = settings(output, reference / 'snapshot.json', local_inventory, args.figure_directory.resolve())
        snapshot(config)
        model = Model(config)
        for source, row in zip(model.snapshot['signals'], selected):
            for key in ('mass_gev', 'epsilon', 'proper_ctau_mm'):
                if not math.isclose(source[key], row[key], rel_tol=1.e-10):
                    raise ValueError('Ordinary snapshot reader changed the catalog physical point')
        reuse = dict(method='Independent fits at generated points; no interpolated-template scans reused')
        scan_generated_points(config, model)
        diagnostics=finite_mc_diagnostics(config,model)
        plots = render(config, model, args.legacy.resolve(), args.before.resolve(), output,
            catalog['updated_utc'],args.before_event_statistics.resolve())
        pdfs=check_pdfs(output)
        report.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started, staged_bytes=stager.bytes,
                      staged_inputs=stager.records, scan_reuse=reuse, plots=plots,
                      snapshot_sha256=digest(output / 'snapshot.json'), scan_sha256=digest(output / 'scan.json'),
                      all_source_paths_preserve_canonical_provenance=True,
                      old_NNPDF_DY_response_used=False, sample_aliases_changed=False,
                      nominal_finite_MC_uncertainty_included=False,
                      finite_MC_diagnostic_points=len(diagnostics['records']),pdf_readback=pdfs)
        write(output / 'refresh_status.json', report)
        (output / 'README.md').write_text(
            '<!-- Code style: AI-code. Generated qualified signal-limit update. -->\n'
            '# DP limits from ready signal increments\n\n'
            'Catalog cutoff: ' + catalog['updated_utc'] + '. Only qualified complete component combinations enter.\n\n'
            'Open `limits_current_cl90_comparison.pdf` for both background assumptions and the other-experiment/pheno comparison,\n'
            'or the separately named background/zero-background PDFs. `limits_before_after.pdf` compares to 29 October.\n\n'
            'Same 0.3 fb^-1, Both-Both, literal 19 SM templates and provisional 2% mass resolution; independent 30% normalization nuisances.\n'
            'Nominal curves omit finite-MC uncertainty. Missing lifetime support is not a physical exclusion edge.\n'
            'No old NNPDF detector response is used for the revised benchmark. Grey curves retain the explicitly named older model.\n'
            'Each signal retains its qualified epsilon domain. Photon continuations above 0.01 require the explicit experimental approximation; v1 DY remains capped at 0.001.\n\n'
            '`catalog_snapshot.json`, `signal_input_inventory.json`, `snapshot.json`, `histograms.npz`, `scan.json` and\n'
            '`refresh_status.json` preserve inputs, full original exposure, variances and qualification. Limits interpolate fitted log(r) on the generated grid, never signal templates.\n')
    print(json.dumps(dict(complete=True, output=str(output), signals=len(selected),
                          elapsed_seconds=report['elapsed_seconds'], catalog_cutoff_utc=catalog['updated_utc'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--publisher-config-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--reuse-scan',type=Path)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--before-event-statistics',type=Path,required=True)
    parser.add_argument('--figure-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch-base', type=Path, default=Path('/tmp'))
    parser.add_argument('--maximum-input-bytes', type=int, default=500_000_000)
    args = parser.parse_args()
    check_digest(args.publisher_config_sha256)
    check_digest(args.reference_sha256)
    if not 0 < args.maximum_input_bytes <= 700_000_000:
        parser.error('Analysis staging requires a bounded input allocation below 700 MB')
    owns_output=not args.output.exists()
    try:
        refresh(args)
    except Exception as error:
        if owns_output and args.output.is_dir():
            status=args.output/'refresh_status.json'
            report=read(status) if status.exists() else {}
            report.update(complete=False,error=repr(error),failed_utc=datetime.now(timezone.utc).isoformat())
            write(status,report)
        raise


if __name__ == '__main__':
    main()
