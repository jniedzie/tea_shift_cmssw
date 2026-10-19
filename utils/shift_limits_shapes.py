#!/usr/bin/env python3
# Code style: AI-code. AI-generated audited mass-template and contour helpers.
"""Mass likelihood inputs for the native TEA limits producer.

Saved histograms are checked against the full-GEN counting receipts. All mass
flows are retained. Adaptive binning depends on background MC alone; it never
turns an empty simulated tail into a zero-background measurement.
"""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shift_limits import digest, reference_cross_section
from shift_limits_lifetime import save_json


def merge_background_bins(values, variances, minimum_ess):
    """Merge from the high-mass end, including a sparse remainder in its neighbour."""
    if len(values) != len(variances) or minimum_ess <= 0:
        raise ValueError('Invalid background binning inputs')
    ends = [len(values)]
    total, variance = 0., 0.
    for index in range(len(values)-1, -1, -1):
        total += values[index]
        variance += variances[index]
        if total > 0 and variance > 0 and total**2 / variance >= minimum_ess:
            ends.append(index)
            total, variance = 0., 0.
    ends = sorted(set(ends))
    if ends[0] != 0:
        if len(ends) > 1:
            ends[0] = 0
        else:
            ends.insert(0, 0)
    return ends


def rebin(values, boundaries):
    return [float(math.fsum(values[left:right])) for left, right in zip(boundaries, boundaries[1:])]


def read_saved_histograms(config):
    """Read each existing merged histogram once; no NanoAOD background scan."""
    import ROOT
    from shift_paths import base_path
    from shift_sample_paths import single_root_file
    from shift_storage_paths import tier_directory
    ROOT.gROOT.SetBatch(True)
    original = json.loads(Path(config.audited_yields_path).read_text())
    output = Path(config.output_path).resolve()
    cache = output / 'data'
    cache.mkdir(parents=True, exist_ok=True)
    records, arrays = {}, {}
    for kind in ('backgrounds', 'signals'):
        for name, sample in original[kind].items():
            if sample['counts']['multiple_dimuon_events']:
                raise ValueError('Inclusive histograms would count events more than once: '+name)
            if kind == 'signals':
                reference = sample['source']['scored_group_reference']
                if digest(reference['path']) != reference['sha256']:
                    raise ValueError('Signal publication changed')
                marker = json.loads(Path(reference['path']).read_text())
                published = marker['published']['histograms']
                path = Path(published['path'])
                checksum = digest(path)
                if checksum != published['sha256']:
                    raise ValueError('Published histogram changed: '+name)
            else:
                process, interval = name.split('_', 1)
                path = Path(single_root_file(tier_directory(base_path, process, interval,
                    sample['source']['campaign'], 'histograms_merged')))
                checksum = digest(path)
            binding = dict(yields_sha256=digest(config.audited_yields_path), path=str(path),
                           histogram_sha256=checksum, categories=list(config.categories),
                           flow_policy='Underflow in first bin; overflow in last bin; no mass cut.')
            target = cache / (name+'_mass.npz')
            receipt = target.with_suffix('.json')
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old['binding'] != binding or old['sha256'] != digest(target):
                    raise ValueError('Saved mass cache changed: '+name)
                with np.load(target) as data:
                    arrays[name] = {key: data[key].copy() for key in data.files}
                records[name] = old
                print('Reused mass histogram cache', name, flush=True)
                continue
            file = ROOT.TFile.Open(str(path), 'READ')
            if not file or file.IsZombie():
                raise RuntimeError('Cannot read merged histogram: '+str(path))
            data, checks = {}, {}
            try:
                for category in config.categories:
                    hist = file.Get('dimuon/ShiftDimuonVertex'+category+'_mass')
                    if not hist or hist.GetNbinsX() != 1000 or hist.GetXaxis().GetXmax() != 100.:
                        raise ValueError('Unexpected mass histogram binning')
                    values = np.array([hist.GetBinContent(i) for i in range(1002)])
                    variance = np.array([hist.GetBinError(i)**2 for i in range(1002)])
                    audited = sample['counts']['categories'][category]
                    if not math.isclose(float(np.sum(values)), audited['sumw'], rel_tol=2.e-5, abs_tol=1.e-12):
                        raise ValueError('Mass integral differs from audited event weights: '+name+'/'+category)
                    if not math.isclose(float(np.sum(variance)), audited['sumw2'], rel_tol=4.e-5, abs_tol=1.e-18):
                        raise ValueError('Mass sumw2 differs from audited event weights: '+name+'/'+category)
                    correction = audited['sumw']/float(np.sum(values)) if audited['sumw'] else 1.
                    values *= correction
                    variance *= correction**2
                    values[1] += values[0]
                    variance[1] += variance[0]
                    values[-2] += values[-1]
                    variance[-2] += variance[-1]
                    data[category] = values[1:-1]
                    data[category+'_sumw2'] = variance[1:-1]
                    checks[category] = dict(sumw=audited['sumw'], sumw2=audited['sumw2'],
                                           floating_point_integral_correction=correction)
            finally:
                file.Close()
            np.savez_compressed(target, **data)
            records[name] = dict(binding=binding, path=str(target), sha256=digest(target), checks=checks)
            save_json(receipt, records[name])
            arrays[name] = data
            print('Checked mass histogram', name, flush=True)
    save_json(cache/'mass_manifest.json', records)
    return original, arrays


def prepare_shapes(config):
    original, arrays = read_saved_histograms(config)
    report = copy.deepcopy(original)
    output = Path(config.output_path).resolve()
    bins = {}
    for category in config.categories:
        values = np.zeros(1000)
        variance = np.zeros(1000)
        for name, row in original['backgrounds'].items():
            factor = config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
            values += arrays[name][category]*factor
            variance += arrays[name][category+'_sumw2']*factor**2
        boundaries = merge_background_bins(values, variance, config.background_minimum_bin_ess)
        rebinned, rebinned_variance = rebin(values, boundaries), rebin(variance, boundaries)
        if any(value <= 0 for value in rebinned):
            raise ValueError('An adaptive mass bin still has no background')
        bins[category] = dict(fine_bin_boundaries=boundaries,
            mass_edges_gev=[value/10. for value in boundaries],
            yields=rebinned, variances=rebinned_variance,
            effective_events=[b*b/v if v else None for b,v in zip(rebinned,rebinned_variance)])
    for kind in ('backgrounds', 'signals'):
        for name, sample in report[kind].items():
            factor = config.luminosity*sample['cross_section_pb']/sample['full_gen_sumw']
            sample['binned_yields'] = {category: rebin(arrays[name][category]*factor,
                bins[category]['fine_bin_boundaries']) for category in config.categories}
            sample['binned_variances'] = {category: rebin(arrays[name][category+'_sumw2']*factor**2,
                bins[category]['fine_bin_boundaries']) for category in config.categories}
            for category in config.categories:
                if not math.isclose(math.fsum(sample['binned_yields'][category]), sample['yields'][category],
                                    rel_tol=1.e-10, abs_tol=1.e-12):
                    raise ValueError('Mass shape lost the audited normalization: '+name)
    report['mass_binning'] = bins
    report['mass_shape_provenance'] = dict(audited_yields_sha256=digest(config.audited_yields_path),
        histogram_manifest_sha256=digest(output/'data'/'mass_manifest.json'),
        minimum_background_bin_ess=config.background_minimum_bin_ess,
        mc_stat_nuisances=False, no_mass_selection=True,
        caveat='Finite MC errors are diagnosed, not profiled. Missing QCD tails remain uncorrected.')
    save_json(Path(config.yields_path), report)
    save_json(output/'binning.json', bins)
    references = {name:reference_cross_section(row, report['backgrounds'], config.categories)
                  for name,row in report['signals'].items()}
    save_json(output/'reference_estimates.json', references)
    print('Prepared mass model:', {key:len(value['yields']) for key,value in bins.items()}, flush=True)
    return report


def flatten_model(report, point, categories):
    signal = report['signals'][point]
    rates, backgrounds = [], []
    for category in categories:
        for index, value in enumerate(signal['binned_yields'][category]):
            rates.append(value/signal['reference_cross_section_pb'])
            backgrounds.append(math.fsum(row['binned_yields'][category][index]
                                         for row in report['backgrounds'].values()))
    return rates, backgrounds


def draw_mass_diagnostics(config, report):
    from shift_limits_lifetime import plotting, finish
    plt = plotting()
    output = Path(config.output_path).resolve()
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for category, axis in zip(config.categories, axes.flat):
        row = report['mass_binning'][category]
        edges = np.array(row['mass_edges_gev'])
        centers = .5*(edges[1:]+edges[:-1])
        widths = np.diff(edges)
        axis.stairs(np.array(row['yields'])/widths, edges, label='Total expected background')
        axis.errorbar(centers, np.array(row['yields'])/widths,
            yerr=np.sqrt(row['variances'])/widths, fmt='.', color='black', label='Finite MC error')
        axis.set(title=category, xlabel=r'Reconstructed $m_{\mu\mu}$ [GeV]',
                 ylabel='Expected events / GeV', yscale='log')
        axis.legend(fontsize=7)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0, .95, '3 fb$^{-1}$; five exclusive categories\n\n'
        'Binning uses background MC only.\nAll masses and flow bins are retained.\n'
        f"Target: at least {config.background_minimum_bin_ess:g} effective MC events/bin.\n\n"
        'No background-tail correction.\nMC errors are not nuisance parameters.\n'
        '30% global signal/background nuisances.', va='top')
    finish(figure, output, '01_background_mass_bins',
        'Sparse tails merge into neighbouring bins. An empty simulated tail is never assigned zero physical background.')
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    selected = ['m2_prompt','m5_prompt','m10_prompt','m15_prompt','m30_prompt','m90_prompt']
    for point, axis in zip(selected, axes.flat):
        row = report['signals'][point]
        for category in config.categories:
            edges = np.array(report['mass_binning'][category]['mass_edges_gev'])
            values = np.array(row['binned_yields'][category])/row['reference_cross_section_pb']
            axis.stairs(values/np.diff(edges), edges, label=category)
        axis.set(title=f"{row['mass_gev']:g} GeV, prompt; {row['counts']['events']} generated",
            xlabel=r'Reconstructed $m_{\mu\mu}$ [GeV]', ylabel='Events / pb / GeV', yscale='log')
        axis.legend(fontsize=7)
    finish(figure, output, '02_signal_mass_templates',
        'Signal rates use the original full GEN denominator, including events without a reconstructed dimuon. No shape smoothing.')


def prepare_lifetime_caches(config, report):
    """Reuse the five existing caches; create only missing signal response caches."""
    import shift_limits_lifetime as lifetime
    from shift_paths import limits_lifetime_output_path
    output = Path(config.output_path).resolve()
    records = {}
    for mass in sorted({row['mass_gev'] for row in report['signals'].values()}):
        selected = {name:row for name,row in report['signals'].items() if row['mass_gev']==mass}
        reused = tuple(name for name in selected if
            (Path(limits_lifetime_output_path)/'data'/(name+'.json')).exists())
        missing = tuple(name for name in selected if name not in reused)
        for points, destination in ((reused, limits_lifetime_output_path),
                                     (missing, str(output/'lifetime_cache'))):
            if not points:
                continue
            current = SimpleNamespace(**vars(config))
            current.yields_path = config.audited_yields_path
            current.lifetime_points = points
            current.lifetime_mass_gev = mass
            current.lifetime_output_path = destination
            # build_cache also verifies existing cache hashes and source receipts.
            records.update(lifetime.build_cache(current))
    save_json(output/'lifetime_cache_manifest.json', records)
    return records


def weighted_shape(data, weights, report, categories):
    """Estimate rates per pb with the original N, never the selected/reweighted sum."""
    result, ess = {}, {}
    total_count = len(data['ids'])
    for index, category in enumerate(categories):
        selected = data['category']==index
        fine_index = np.clip((data['mass_gev'][selected]*10).astype(int), 0, 999)
        values = np.bincount(fine_index, weights=weights[selected], minlength=1000)
        values2 = np.bincount(fine_index, weights=weights[selected]**2, minlength=1000)
        boundaries = report['mass_binning'][category]['fine_bin_boundaries']
        result[category] = [value*report['luminosity_pb']/total_count for value in rebin(values, boundaries)]
        ess[category] = float(np.sum(values)**2/np.sum(values2)) if np.sum(values2) else 0.
    return result, ess


def clipped_boundary_segments(points, ceiling):
    """Clip log-linear mass interpolation to the stated coupling coverage."""
    segments = []
    current = []
    for left, right in zip(points, points[1:]):
        if left is None or right is None:
            if len(current)>1: segments.append(current)
            current = []
            continue
        x1,y1 = left
        x2,y2 = right
        if y1>ceiling and y2>ceiling:
            if len(current)>1: segments.append(current)
            current = []
            continue
        pair = [list(left),list(right)]
        if (y1>ceiling)!=(y2>ceiling):
            fraction = math.log(ceiling/y1)/math.log(y2/y1)
            intersection = [x1*math.exp(fraction*math.log(x2/x1)),ceiling]
            pair[0 if y1>ceiling else 1] = intersection
        if current and all(math.isclose(a,b,rel_tol=1.e-10) for a,b in zip(current[-1],pair[0])):
            current.append(pair[1])
        else:
            if len(current)>1: segments.append(current)
            current = pair
    if len(current)>1: segments.append(current)
    return segments


def boundary_outline_segments(segments, minimum_mass, ceiling):
    """Add the upper-left display corner without inventing a fitted point."""
    outline = [[list(point) for point in segment] for segment in segments]
    if outline and math.isclose(outline[0][0][0], minimum_mass):
        if outline[0][0][1] < ceiling:
            outline[0].insert(0, [minimum_mass, ceiling])
    return outline


def physical_boundaries(config, report, results):
    """Invert the physical production model, then verify the lifetime at each crossing."""
    masses = sorted({row['mass_gev'] for row in report['signals'].values()})
    ceiling = getattr(config, 'epsilon_plot_max', config.epsilon_max)
    boundaries = {}
    for level in config.confidence_levels:
        channels = ('combined',)+tuple(config.categories) if level==.95 else ('combined',)
        for channel in channels:
            selected = {row['mass_gev']:row for row in results if row['channel']==channel and
                row['confidence_level']==level and row['point'].endswith('_prompt')}
            points = []
            for mass in masses:
                result = selected.get(mass)
                name = f'm{mass:g}_prompt'
                sample = report['signals'][name]
                if result is None or result['status']!='ok' or not result['validation']['median_valid']:
                    points.append(dict(mass_gev=mass,status='no_validated_limit'))
                    continue
                epsilons = {key:sample['epsilon']*math.sqrt(value/sample['reference_cross_section_pb'])
                    for key,value in result['expected_cross_section_quantiles_pb'].items()}
                proper = sample['proper_ctau_mm']*(sample['epsilon']/epsilons['median'])**2
                boost = 5000./report['signals'][f'm{mass:g}_lab_5m']['proper_ctau_mm']
                mean_flight = boost*proper
                if mean_flight>config.prompt_maximum_mean_flight_mm:
                    raise ValueError('A putative boundary requires a displaced template')
                points.append(dict(mass_gev=mass,status='ok',epsilon_quantiles=epsilons,
                    proper_ctau_at_boundary_mm=proper,mean_lab_flight_at_boundary_mm=mean_flight,
                    within_coupling_coverage=epsilons['median']<=ceiling,
                    small_mixing_extrapolation=epsilons['median']>config.epsilon_max,
                    bands_valid=result['validation']['bands_valid'],point=name,
                    sigma_limit_pb=result['expected_limit_cross_section_pb'],
                    sigma_model_pb=sample['reference_cross_section_pb'],
                    reference_epsilon=sample['epsilon']))
            curve = [(row['mass_gev'],row['epsilon_quantiles']['median']) if row['status']=='ok' else None for row in points]
            segments = clipped_boundary_segments(curve,ceiling)
            polygons = [segment+[[segment[-1][0],ceiling],[segment[0][0],ceiling]]
                        for segment in segments]
            key = f'cl{round(100*level)}_{channel}'
            content = dict(confidence_level=level,channel=channel,points=points,segments=segments,
                outline_segments=boundary_outline_segments(segments,min(masses),ceiling),
                outline_corner_is_display_only=True,
                excluded_polygons=polygons,mass_range_gev=[min(masses),max(masses)],
                epsilon_range=[config.epsilon_min,ceiling],
                interpretation='Formal epsilon^2 production benchmark; large-epsilon extrapolation above 0.1; log interpolation in mass.',
                small_mixing_check_maximum=config.epsilon_max,
                expected_only=True,production_normalization_provisional=True)
            save_json(Path(config.output_path)/'contours'/(key+'.json'),content)
            boundaries[key] = content
    return boundaries


def draw_contours(config, report, results, boundaries):
    from shift_limits_lifetime import plotting, finish
    plt = plotting()
    output = Path(config.output_path).resolve()
    curve = boundaries['cl95_combined']
    ceiling = curve['epsilon_range'][1]
    figure, axes = plt.subplots(1,2,figsize=(12,6))
    figure.suptitle(r'SHIFT $\epsilon^2$ benchmark — expected 95% CLs, 3 fb$^{-1}$')
    valid = [row for row in curve['points'] if row['status']=='ok']
    masses = [row['mass_gev'] for row in valid]
    colors = ['#ffdf40','#70c070']
    for axis in axes:
        for lower,upper,color,label in [('minus2','plus2',colors[0],r'Expected $\pm2\sigma$'),
                                        ('minus1','plus1',colors[1],r'Expected $\pm1\sigma$')]:
            low = [row['epsilon_quantiles'][lower] if row['bands_valid'] else np.nan for row in valid]
            high = [row['epsilon_quantiles'][upper] if row['bands_valid'] else np.nan for row in valid]
            axis.fill_between(masses,low,high,color=color,alpha=.65,label=label)
        for index,polygon in enumerate(curve['excluded_polygons']):
            axis.fill(*np.array(polygon).T,color='#dc6632',alpha=.22,
                      label='Expected excluded region' if index==0 else None)
        for index,segment in enumerate(curve['outline_segments']):
            axis.plot(*np.array(segment).T,color='#a33200',linewidth=2.2,
                      label='Median boundary' if index==0 else None)
        inside = [row for row in valid if row['within_coupling_coverage']]
        axis.scatter([row['mass_gev'] for row in inside],
                     [row['epsilon_quantiles']['median'] for row in inside],color='#a33200',s=24,zorder=6)
        axis.set(xscale='log',yscale='log',xlim=(2,90),ylim=(config.epsilon_min,ceiling),
                 xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Mixing $\epsilon$')
        coverage = json.loads((output/'model_coverage.json').read_text())['maximum_coupling_by_mass']
        domain_mass, domain_epsilon = np.array(coverage).T
        axis.fill_between(domain_mass,domain_epsilon,ceiling,where=domain_epsilon<ceiling,facecolor='none',
            edgecolor='gray',hatch='///',label='Outside full-mixing domain',zorder=8)
        axis.axhline(config.epsilon_max,color='gray',linestyle=':',linewidth=1,
                     label=r'Above $\epsilon=0.1$: benchmark extrapolation')
    # The full view retains the independently generated displaced anchors.
    for name,row in report['signals'].items():
        axes[0].scatter(row['mass_gev'],row['epsilon'],s=13,facecolors='none',edgecolors='#4575a8',linewidths=.7)
    axes[1].set_ylim(1.e-3,ceiling)
    axes[0].set_title('Full lifetime range; open circles: generated anchors',fontsize=10)
    axes[1].set_title('Boundary region — prompt lifetimes',fontsize=10)
    axes[1].legend(fontsize=8,loc='lower left')
    finish(figure,output,'mass_epsilon_limits95',
        r'Formal $\epsilon^2$ benchmark; above $\epsilon=0.1$ is an unvalidated rate/width extrapolation. Lines connect all mass anchors before clipping.'+'\n'
        '30% signal/background normalizations; finite MC shape errors and missing QCD tails are not covered. Expected sensitivity, not a data exclusion.')

    figure,axis = plt.subplots(figsize=(9,6))
    figure.suptitle('SHIFT category comparison — expected 95% CLs')
    for channel in ('combined',)+tuple(config.categories):
        current = boundaries['cl95_'+channel]
        if not current['segments']:
            axis.plot([],[],linestyle=':',label=channel+' (no boundary shown)')
        for index,segment in enumerate(current['segments']):
            axis.plot(*np.array(segment).T,label=channel if index==0 else None,
                      linewidth=2.8 if channel=='combined' else 1.4)
    axis.set(xscale='log',yscale='log',xlim=(2,90),ylim=(1.e-3,ceiling),
             xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Median boundary in $\epsilon$')
    axis.legend(fontsize=9)
    finish(figure,output,'category_epsilon_boundaries95',
        'Validated cross-section medians; formal epsilon-squared extrapolation above 0.1. Missing boundaries are not interpolated.\n'
        '30% normalization uncertainties are shared across mass bins and categories; finite MC shape errors remain unprofiled.')


def draw_mass_improvement(config, report, results, counting_profile):
    if counting_profile is None:
        raise ValueError('Mass comparison requires --counting-profile with saved counting predictions')
    saved_path = Path(counting_profile)
    if not saved_path.is_file():
        raise FileNotFoundError('Counting profile input does not exist: '+str(saved_path))
    saved = json.loads(saved_path.read_text())
    if saved['yields_sha256']!=digest(config.audited_yields_path):
        raise ValueError('Counting baseline does not use the same audited yields')
    baseline = {row['point']:row['expected95_pb'] for row in saved['records']}
    from shift_limits_lifetime import plotting, finish
    plt = plotting()
    output = Path(config.output_path).resolve()
    selected = sorted([row for row in results if row['channel']=='combined' and row['confidence_level']==.95
        and row['point'].endswith('_prompt') and row['status']=='ok'],key=lambda row:row['mass_gev'])
    figure,axes = plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    masses = [row['mass_gev'] for row in selected]
    counting = [baseline[row['point']] for row in selected]
    shaped = [row['expected_limit_cross_section_pb'] for row in selected]
    axes[0].plot(masses,counting,'o--',label='Audited counting likelihood (independent profile)')
    axes[0].plot(masses,shaped,'o-',label='Mass bins + categories (native TEA / Combine)')
    axes[0].set(yscale='log',ylabel=r'Expected 95% limit on $\sigma\times B_{\mu\mu}$ [pb]')
    axes[0].legend(fontsize=9)
    axes[1].plot(masses,np.array(counting)/shaped,'o-')
    axes[1].axhline(1,color='gray',linestyle=':')
    axes[1].set(xscale='log',yscale='log',xlabel=r'$m_{A\prime}$ [GeV]',ylabel='Improvement')
    finish(figure,output,'03_mass_information_gain',
        'Same full-GEN signal normalization, background rates, categories and two 30% uncertainties; only the use of mass changes.\n'
        'Signal and background templates have limited MC statistics. An improvement in this toy model is not a calibrated sensitivity gain.')


def draw_lifetime_mass_diagnostics(config, report):
    from shift_limits_lifetime import plotting, finish
    plt = plotting()
    output = Path(config.output_path).resolve()
    figure,axis = plt.subplots(figsize=(9,6))
    for suffix,label in [('prompt','prompt'),('lab_5m','5 m'),('lab_30m','30 m'),('lab_100m','100 m'),('lab_1000m','1000 m')]:
        rows = sorted([row for name,row in report['signals'].items() if name.endswith('_'+suffix)],key=lambda row:row['mass_gev'])
        n = np.array([row['counts']['events'] for row in rows])
        k = np.array([sum(v['events'] for v in row['counts']['categories'].values()) for row in rows])
        probability = k/n
        center = (probability+.5/n)/(1+1/n)
        half = np.sqrt(probability*(1-probability)/n+.25/n**2)/(1+1/n)
        axis.errorbar([row['mass_gev'] for row in rows],probability,
            yerr=[probability-center+half,center+half-probability],fmt='o-',capsize=2,label=label)
    axis.set(xscale='log',xlabel=r'$m_{A\prime}$ [GeV]',ylabel='Events with a dimuon / full GEN denominator',ylim=(0,.5))
    axis.legend(title='Generated mean lab-flight label',fontsize=8)
    finish(figure,output,'04_efficiency_all_masses_and_lifetimes',
        'Wilson 68% intervals include zero-reconstruction events through 0_initial. Labels are mean lab flights, not proper lifetimes.\n'
        'Prompt/displaced populations are independent MC samples; they are alternative hypotheses, not channels to combine.')
    figure,axes = plt.subplots(2,2,figsize=(11,7))
    for mass,axis in zip((2,10,15,30),axes.flat):
        for suffix,label in [('prompt','prompt'),('lab_5m','5 m'),('lab_100m','100 m'),('lab_1000m','1000 m')]:
            path = output/'data'/(f'm{mass}_{suffix}_mass.npz')
            with np.load(path) as data:
                values = sum(data[category] for category in config.categories)
            # Rebin only for this display; no smoothing or selection enters the fit.
            displayed = np.sum(values.reshape(50,20),axis=1)
            if sum(displayed)>0:
                axis.stairs(displayed/sum(displayed),np.linspace(0,100,51),label=label)
        axis.set(title=f'{mass:g} GeV',xlabel=r'Reconstructed $m_{\mu\mu}$ [GeV]',ylabel='Fraction of reconstructed dimuons',yscale='log')
        axis.legend(fontsize=8)
    finish(figure,output,'05_mass_response_changes_with_lifetime',
        'Inclusive reconstructed candidates, including wrong pairs; these are response distributions, not truth-matched resolution measurements.\n'
        'The likelihood retains each anchor’s own response. The physical exclusion boundary is prompt; displaced production is too small.')


def draw_lifetime_bound(config, report, boundaries, proof):
    from shift_limits_lifetime import plotting, finish
    plt = plotting()
    rows = sorted([row for row in proof['records'] if row['confidence_level']==.95],key=lambda row:row['mass_gev'])
    masses = [row['mass_gev'] for row in rows]
    floor = [row['epsilon_below_which_no_response_can_exclude'] for row in rows]
    prompt = []
    for mass in masses:
        source = report['signals'][f'm{mass:g}_prompt']
        boost = 5000/report['signals'][f'm{mass:g}_lab_5m']['proper_ctau_mm']
        prompt.append(source['epsilon']*math.sqrt(boost*source['proper_ctau_mm']/config.prompt_maximum_mean_flight_mm))
    if any(left<right for left,right in zip(floor,prompt)):
        raise ValueError('Lifetime bound leaves a gap requiring displaced templates')
    figure,axis = plt.subplots(figsize=(9,6))
    axis.fill_between(masses,config.epsilon_min,floor,color='#bed5e8',alpha=.6,
                      label='Cannot exclude even with 100% efficiency and ideal bin assignment')
    axis.plot(masses,floor,'o-',color='#326b96',label='Most optimistic possible boundary in this background model')
    axis.plot(masses,prompt,'s--',color='#72549c',label='Mean flight = 0.01 mm; shorter above this line')
    for index,segment in enumerate(boundaries['cl95_combined']['segments']):
        axis.plot(*np.array(segment).T,color='#a33200',linewidth=2.2,label='Actual median boundary' if index==0 else None)
    axis.set(xscale='log',yscale='log',xlim=(2,90),ylim=(1.e-6,config.epsilon_max),
             xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Mixing $\epsilon$')
    axis.legend(fontsize=8,loc='lower left')
    finish(figure,Path(config.output_path),'06_lifetime_consistency_bound',
        'Any potentially excluded coupling already has a negligible physical displacement. The actual boundary lies still farther inside the prompt regime.\n'
        'The bound uses the fixed positive background bins and the provisional production model; it does not apply to a different selection or background model.')


def draw_native_brazil(config, groups):
    """Keep raw native bands visible, explicitly labelling failed edge audits."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import limits_plotter
    from Histogram import Histogram
    omitted = []
    for (channel,level,group),rows in groups.items():
        valid = [row for row in rows if row['status']=='ok']
        if not valid:
            omitted.append((channel,level,group))
            continue
        current = SimpleNamespace(**vars(config))
        destination = Path(config.plots_output_path).resolve()/f'cl{round(100*level)}'/channel/group
        destination.mkdir(parents=True,exist_ok=True)
        current.plots_output_path = current.results_output_path = str(destination)
        current.limit_points = {row['point']:row['mass_gev'] for row in valid}
        current.reference_cross_section = 1.
        current.x_min = min(row['mass_gev'] for row in valid)*.8
        current.x_max = max(row['mass_gev'] for row in valid)*1.2
        values = [value for row in valid for value in row['expected_cross_section_quantiles_pb'].values()]
        current.y_min, current.y_max = min(values)*.5,max(values)*3.
        current.histogram = Histogram(name='mass_scan')
        current.experiment_label = '#splitline{#bf{SHIFT} #it{Simulation}}{#scale[0.65]{'+channel+'}}'
        label = f'{config.luminosity/1000:g} fb^{{-1}}, {round(100*level)}% CL'
        warning = None
        if any(not row['validation']['bands_valid'] for row in valid): warning = 'Band edges provisional'
        if len(valid)!=len(rows): warning = 'Unresolved points omitted'
        method_label = getattr(config, 'projection_method_label', None)
        if method_label: warning = method_label + ('; '+warning if warning else '')
        current.luminosity_label = '#splitline{'+label+'}{#scale[0.65]{'+warning+'}}' if warning else label
        path = destination/'limits_mass_scan.txt'
        with path.open('w') as stream:
            for row in sorted(valid,key=lambda item:item['mass_gev']):
                quantiles = row['expected_cross_section_quantiles_pb']
                stream.write(row['point']+': '+repr([None]+[quantiles[key] for key in
                    ('minus2','minus1','median','plus1','plus2')])+'\n')
        save_json(destination/'validation.json',dict(channel=channel,confidence_level=level,
            band_edges_provisional=any(not row['validation']['bands_valid'] for row in valid),
            omitted_points=[row['point'] for row in rows if row['status']!='ok'],
            band_audit_failures={row['point']:row['validation']['relative_errors'] for row in valid
                                 if not row['validation']['bands_valid']}))
        limits_plotter.config, limits_plotter.input_path = current,str(path)
        limits_plotter.draw_brazil_plots()
    return omitted


def publish(config, proof_path, counting_profile):
    """Publish verified native results and the physical interpretation separately."""
    import shutil
    import shift_limits
    import shift_limits_comparison
    from shift_paths import limits_output_path
    import fitz
    output = Path(config.output_path).resolve()
    report = json.loads(Path(config.yields_path).read_text())
    result_path = Path(config.results_output_path)/'limits.json'
    saved = json.loads(result_path.read_text())
    results = saved['records']
    if saved['yields_sha256']!=digest(config.yields_path):
        raise ValueError('Fit results and mass templates differ')
    expected = set()
    for name in report['signals']:
        if name.endswith('_prompt'):
            expected.add((name,'combined',.95))
            expected.update((name,category,.95) for category in config.categories)
            expected.add((name,'combined',.90))
    actual = {(row['point'],row['channel'],row['confidence_level']) for row in results}
    if actual!=expected:
        raise ValueError('The requested mass scan is incomplete: '+str(len(expected-actual))+' missing results')
    for row in results:
        if row.get('combine_output') and digest(row['combine_output'])!=row['combine_output_sha256']:
            raise ValueError('A saved Combine output changed')
    boundaries = physical_boundaries(config,report,results)
    draw_mass_improvement(config,report,results,counting_profile)
    draw_contours(config,report,results,boundaries)
    proof = json.loads(Path(proof_path).read_text())
    draw_lifetime_bound(config,report,boundaries,proof)
    shutil.copy2(proof_path,output/'no_exclusion_lifetime_bound.json')
    manifest = json.loads(Path(config.prepared_input_manifest).read_text())
    # Raw native bands retain explicit provisional labels if an edge fails.
    groups = {}
    for row in results:
        key = (row['channel'],row['confidence_level'],shift_limits.scan_group(report['signals'][row['point']]))
        groups.setdefault(key,[]).append(row)
    omitted = draw_native_brazil(config,groups)
    primary = [row for row in results if row['channel']=='combined' and row['confidence_level']==.95
               and row['point'].endswith('_prompt')]
    saved.update(complete=True,computation_complete=True,
        all_medians_validated=all(row['status'] in ('ok','no_signal_acceptance') for row in results),
        all_expected_bands_validated=all(row.get('validation',{}).get('bands_valid',False)
                                        for row in results if row['status']!='no_signal_acceptance'),
        primary_medians_validated=all(row['status']=='ok' for row in primary),
        primary_bands_validated=all(row.get('validation',{}).get('bands_valid',False) for row in primary),
        mc_unresolved_points=[dict(point=row['point'],channel=row['channel']) for row in results
                              if row['status']=='no_signal_acceptance'],
        no_signal_acceptance_is_not_zero_physical_efficiency=True,
        mass_interpretation='Prompt boundary verified against a response-independent no-exclusion bound at longer lifetime.',
        omitted_native_brazil_groups=omitted)
    canonical = output/'results/limits.json'
    save_json(canonical,saved)
    source = Path(limits_output_path).resolve()
    args = SimpleNamespace(input=canonical,channel='combined',contour=output/'contours/cl90_combined.json',
        source_figure=source/'arxiv_2511.03883v1_fig8_15.pdf',
        source_provenance=source/'paper_figure_provenance.json',output=output/'paper_comparison90.pdf',
        lumi_fb=3.,systematic_percent=30.)
    shift_limits_comparison.make_comparison(args)
    pages = [output/name for name in ('01_background_mass_bins.pdf','02_signal_mass_templates.pdf',
        '03_mass_information_gain.pdf','04_efficiency_all_masses_and_lifetimes.pdf',
        '05_mass_response_changes_with_lifetime.pdf','06_lifetime_consistency_bound.pdf',
        'mass_epsilon_limits95.pdf','category_epsilon_boundaries95.pdf','paper_comparison90.pdf')]
    pages += sorted((output/'brazil/cl95/combined/prompt').glob('*.pdf'))
    document = fitz.open()
    for path in pages:
        with fitz.open(path) as part:
            document.insert_pdf(part)
    document.set_metadata({'title':'SHIFT mass-shape limits and lifetime consistency',
                           'subject':'Preliminary expected 95% CLs; separate 90% paper comparison'})
    booklet = output/'mass_shape_limits_walkthrough.pdf'
    document.save(booklet,garbage=4,deflate=True)
    document.close()
    receipt = dict(yields_sha256=digest(config.yields_path),results_sha256=digest(canonical),
        plots={str(path.relative_to(output)):dict(sha256=digest(path),bytes=path.stat().st_size)
               for path in output.rglob('*.pdf')},
        contour_inputs={str(path.relative_to(output)):digest(path) for path in (output/'contours').glob('*.json')},
        complete=True,primary_medians_validated=saved['primary_medians_validated'],
        primary_bands_validated=saved['primary_bands_validated'])
    save_json(output/'receipt.json',receipt)
    print('Published',len(receipt['plots']),'PDFs;',len(results),'results;',
          sum(row['status']=='numerical_validation_failed' for row in results),'failed median audits',flush=True)


if __name__=='__main__':
    import argparse
    import importlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='shift_limits_shapes_config')
    parser.add_argument('--stage',choices=('diagnostics','preview','publish'),required=True)
    parser.add_argument('--proof',type=Path)
    parser.add_argument('--counting-profile',type=Path,
                        help='Saved counting predictions using the same audited yields; required for preview/publish')
    args = parser.parse_args()
    if args.stage in ('preview','publish'):
        if args.counting_profile is None:
            parser.error('--counting-profile is required for preview/publish')
        if not args.counting_profile.is_file():
            parser.error('Counting profile input does not exist: '+str(args.counting_profile))
    configuration = importlib.import_module(args.config.removesuffix('.py'))
    if args.stage=='diagnostics':
        draw_lifetime_mass_diagnostics(configuration,json.loads(Path(configuration.yields_path).read_text()))
    elif args.stage=='preview':
        report = json.loads(Path(configuration.yields_path).read_text())
        results = json.loads((Path(configuration.results_output_path)/'limits.json').read_text())['records']
        primary = [row for row in results if row['channel']=='combined' and row['confidence_level']==.95
                   and row['point'].endswith('_prompt') and row['status']=='ok']
        if len(primary)!=len({row['mass_gev'] for row in report['signals'].values()}):
            raise ValueError('The primary mass scan is not complete')
        boundaries = physical_boundaries(configuration,report,results)
        draw_contours(configuration,report,results,boundaries)
        draw_mass_improvement(configuration,report,results,args.counting_profile)
        if args.proof:
            draw_lifetime_bound(configuration,report,boundaries,json.loads(args.proof.read_text()))
        save_json(Path(configuration.output_path)/'preview.json',dict(primary_medians_complete=True,
            full_scan_complete=False,input_results_sha256=digest(Path(configuration.results_output_path)/'limits.json')))
    else:
        if args.proof is None:
            raise ValueError('Publication needs the saved no-exclusion bound')
        publish(configuration,args.proof,args.counting_profile)
