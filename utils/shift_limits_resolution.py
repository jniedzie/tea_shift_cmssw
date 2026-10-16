#!/usr/bin/env python3
# Code style: AI-code. AI-generated fine-binned likelihood input and response diagnostics.
"""Separate likelihood mass bins from sparse-MC background interpolation regions."""
import copy
import json
import math
from pathlib import Path

import numpy as np
from shift_limits import digest
from shift_limits_lifetime import save_json
from shift_limits_shapes import rebin


def resolution_boundaries(fraction=.02, minimum=.1):
    """Stay on the cached 0.1 GeV grid; never widen bins to collect MC events."""
    if not 0<fraction<1 or minimum<.1:raise ValueError('Invalid resolution scale')
    edges=[0]
    while edges[-1]<1000:
        mass=edges[-1]/10.
        step=max(1,int(math.floor(10*max(minimum,fraction*mass)+1.e-9)))
        edges.append(min(1000,edges[-1]+step))
    return edges


def interpolate_region(total,left,right,power):
    """Positive density with exact integral; an explicit model for missing shape."""
    edges=np.arange(left,right+1,dtype=float)/10.+.1
    if power==0:weights=np.diff(edges)
    elif power==1:weights=np.log(edges[1:]/edges[:-1])
    else:weights=(edges[1:]**(1-power)-edges[:-1]**(1-power))/(1-power)
    return total*weights/math.fsum(weights)


def prepare(config):
    source=Path(config.source_mass_study_path).resolve()
    original=json.loads(Path(config.audited_yields_path).read_text())
    previous=json.loads((source/'shape_yields.json').read_text())
    arrays={}
    points=tuple(config.signal_points)
    for name,row in [(name,row) for kind in ('backgrounds','signals')
                     for name,row in original[kind].items() if kind=='backgrounds' or name in points]:
        path=source/'data'/(name+'_mass.npz')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=receipt['sha256'] or receipt['binding']['yields_sha256']!=digest(config.audited_yields_path):
            raise ValueError('Cached mass input changed: '+name)
        with np.load(path,allow_pickle=False) as data:arrays[name]={key:data[key].copy() for key in data.files}
    common=resolution_boundaries(config.mass_resolution_fraction,config.mass_minimum_bin_gev)
    for power in config.background_interpolation_powers:
        directory=Path(config.output_path).resolve()/('p'+str(int(power)))
        directory.mkdir(parents=True,exist_ok=True)
        report=copy.deepcopy(original)
        report['signals']={name:report['signals'][name] for name in points}
        background=dict(yields={},binned_yields={},binned_variances={},
            source_components={name:dict(full_gen_sumw=row['full_gen_sumw'],cross_section_pb=row['cross_section_pb'])
                               for name,row in original['backgrounds'].items()},
            interpretation='Exact sum of components sharing the same global 30% nuisance.')
        bins={}
        for category in config.categories:
            regions=previous['mass_binning'][category]['fine_bin_boundaries']
            # Keep every interpolation join visible as a likelihood edge.
            boundaries=sorted(set(common+regions))
            density=np.zeros(1000)
            variance=np.zeros(1000)
            total_raw=np.zeros(1000)
            variance_raw=np.zeros(1000)
            for name,row in original['backgrounds'].items():
                factor=config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
                total_raw+=arrays[name][category]*factor
                variance_raw+=arrays[name][category+'_sumw2']*factor**2
            for left,right in zip(regions,regions[1:]):
                total=math.fsum(total_raw[left:right])
                fractions=interpolate_region(1.,left,right,power)
                density[left:right]=total*fractions
                variance[left:right]=math.fsum(variance_raw[left:right])*fractions**2
            values=rebin(density,boundaries);variances=rebin(variance,boundaries)
            if min(values)<=0:raise ValueError('Interpolation cannot establish support for an empty region')
            background['binned_yields'][category]=values
            background['binned_variances'][category]=variances
            background['yields'][category]=math.fsum(values)
            if not math.isclose(math.fsum(total_raw),math.fsum(values),rel_tol=1.e-12):
                raise ValueError('Background interpolation changed its audited total')
            bins[category]=dict(fine_bin_boundaries=boundaries,mass_edges_gev=[i/10 for i in boundaries],
                yields=values,variances=variances,background_interpolation_edges_gev=[i/10 for i in regions],
                raw_fine_yields=total_raw.tolist(),effective_events=[b*b/v if v else None for b,v in zip(values,variances)])
            for name,row in report['signals'].items():
                factor=config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
                row.setdefault('binned_yields',{})[category]=rebin(arrays[name][category]*factor,boundaries)
                row.setdefault('binned_variances',{})[category]=rebin(arrays[name][category+'_sumw2']*factor**2,boundaries)
                if not math.isclose(sum(row['binned_yields'][category]),row['yields'][category],rel_tol=2.e-10,abs_tol=1.e-12):
                    raise ValueError('Signal lost its full-GEN normalization')
        report['backgrounds']={'background_total':background}
        report['mass_binning']=bins
        report['mass_shape_provenance']=dict(
            audited_yields_sha256=digest(config.audited_yields_path),
            source_binning_sha256=digest(source/'shape_yields.json'),
            likelihood_bins='Independent of MC event population; resolution-scale grid plus interpolation joins',
            resolution_fraction=config.mass_resolution_fraction,minimum_bin_gev=config.mass_minimum_bin_gev,
            detector_resolution_measured=False,background_density='(mass+0.1 GeV)^(-power)',power=power,
            background_total_preserved=True,mc_stat_nuisances=False,
            caveat='Sparse-region interpolation is a model assumption, not measured tail support. Two powers do not cover all unknown shapes.')
        save_json(directory/'yields.json',report)
        print('Prepared',directory.name,'bins/category',{c:len(bins[c]['yields']) for c in bins},flush=True)


def model_config(config,power):
    from types import SimpleNamespace
    current=SimpleNamespace(**vars(config))
    directory=Path(config.output_path).resolve()/('p'+str(int(power)))
    current.output_path=str(directory)+'/'
    current.yields_path=str(directory/'yields.json')
    current.prepared_input_manifest=str(directory/'inputs/manifest.json')
    current.datacards_output_path=str(directory/'datacards')+'/'
    current.results_output_path=str(directory/'results')+'/'
    current.plots_output_path=str(directory/'brazil')+'/'
    current.config_source_path=config.__file__
    return current


def poisson_card(config,report,manifest,card):
    """Expand the native TEA model into equivalent scalar Poisson channels.

    Reuse the one-bin histograms already written in each prepared input ROOT
    file. This changes numerical representation, not rates or nuisances.
    """
    source=Path(config.datacards_output_path)/(card['name']+'.txt')
    native=source.read_text()
    if 'signal_norm' not in native or 'background_norm' not in native:
        raise ValueError('Expected the native two-normalization model')
    directory=Path(config.output_path)/'poisson_cards'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/source.name
    receipt_path=target.with_suffix('.json')
    if target.exists() and receipt_path.exists():
        cached=json.loads(receipt_path.read_text())
        template=cached.get('signal_template_path')
        if (template and cached.get('native_sha256')==digest(source)
                and cached.get('expanded_sha256')==digest(target)
                and cached.get('signal_template_sha256')==digest(template)):
            return target
    sample=report['signals'][card['point']]
    categories=config.categories if card['channel']=='combined' else (card['channel'],)
    cells=[];observations=[];columns=[];processes=[];numbers=[];rates=[];declarations=[]
    background=report['backgrounds']['background_total']
    signal_source=manifest['signals'][card['point']]['file_path']
    background_source=manifest['backgrounds']['background_total']['file_path']
    ratio=card['combine_reference_cross_section_pb']/manifest['signals'][card['point']]['combine_reference_cross_section_pb']
    if not math.isclose(ratio,1.,rel_tol=1.e-14):
        import ROOT
        scaled=directory/(card['name']+'_signal.root')
        source_file=ROOT.TFile.Open(signal_source,'READ')
        target_file=ROOT.TFile(str(scaled),'RECREATE')
        if not source_file or target_file.IsZombie():raise RuntimeError('Cannot scale category signal templates')
        try:
            for category in categories:
                for index in range(len(background['binned_yields'][category])):
                    histogram='mass_'+category+'_bin'+str(index)
                    hist=source_file.Get(histogram)
                    if not hist:raise ValueError('Missing prepared one-bin template')
                    clone=hist.Clone(histogram);clone.SetDirectory(target_file);clone.Scale(ratio);clone.Write()
        finally:
            target_file.Close();source_file.Close()
        signal_source=str(scaled)
    for category in categories:
        for index,b in enumerate(background['binned_yields'][category]):
            cell='mass_'+category.replace('-','_')+'_bin'+str(index)
            histogram='mass_'+category+'_bin'+str(index)
            cells.append(cell);observations.append(b)
            signal=sample['binned_yields'][category][index]*card['combine_reference_cross_section_pb']/sample['reference_cross_section_pb']
            if signal>0:
                columns.append(cell);processes.append('signal');numbers.append(0);rates.append(signal)
                declarations.append(f'shapes signal {cell} {signal_source} {histogram}')
            columns.append(cell);processes.append('background_total');numbers.append(1);rates.append(b)
            declarations.append(f'shapes background_total {cell} {background_source} {histogram}')
            declarations.append(f'shapes data_obs {cell} {background_source} {histogram}')
    lines=['# Code style: AI-code. Equivalent scalar expansion of native TEA card.',
        'imax '+str(len(cells)),'jmax 1','kmax *']+declarations+[
        'bin '+' '.join(cells),'observation '+' '.join(format(x,'.17g') for x in observations),
        'bin '+' '.join(columns),'process '+' '.join(processes),
        'process '+' '.join(map(str,numbers)),'rate '+' '.join(format(x,'.17g') for x in rates),
        'signal_norm lnN '+' '.join('1.30' if i==0 else '-' for i in numbers),
        'background_norm lnN '+' '.join('1.30' if i>0 else '-' for i in numbers)]
    target.write_text('\n'.join(lines)+'\n')
    save_json(target.with_suffix('.json'),dict(native_card=str(source),native_sha256=digest(source),
        expanded_sha256=digest(target),bins=len(cells),signal_rate=math.fsum(x for x,n in zip(rates,numbers) if n==0),
        background_rate=math.fsum(observations),category_signal_rescale=ratio,
        signal_template_path=signal_source,signal_template_sha256=digest(signal_source)))
    return target


def publish(config):
    from shift_limits_lifetime import plotting,finish
    from shift_limits_shapes import clipped_boundary_segments,boundary_outline_segments,draw_native_brazil
    output=Path(config.output_path).resolve()
    original=json.loads(Path(config.audited_yields_path).read_text())
    plt=plotting()
    curves={};summaries={}
    figure,axes=plt.subplots(1,2,figsize=(12,6))
    for power,color in zip(config.background_interpolation_powers,['#a33200','#3975a8']):
        current=model_config(config,power);path=Path(current.results_output_path)/'limits.json'
        if not path.exists():continue
        records=json.loads(path.read_text())['records'];groups={}
        for row in records:
            groups.setdefault((row['channel'],row['confidence_level'],'prompt'),[]).append(row)
        draw_native_brazil(current,groups)
        for level,axis in zip((.95,.90),axes):
            selected=sorted([r for r in records if r['channel']=='combined' and r['confidence_level']==level and r['status']=='ok'],key=lambda r:r['mass_gev'])
            if not selected:continue
            points=[]
            for row in selected:
                sample=original['signals'][row['point']]
                epsilon=sample['epsilon']*math.sqrt(row['expected_limit_cross_section_pb']/sample['reference_cross_section_pb'])
                boost=5000/original['signals'][f"m{row['mass_gev']:g}_lab_5m"]['proper_ctau_mm']
                flight=boost*sample['proper_ctau_mm']*(sample['epsilon']/epsilon)**2
                points.append(dict(mass_gev=row['mass_gev'],epsilon=epsilon,mean_lab_flight_mm=flight,
                    prompt_consistent=flight<config.prompt_maximum_mean_flight_mm))
            curves[(power,level)]=points
            segments=clipped_boundary_segments([(p['mass_gev'],p['epsilon']) for p in points],1.)
            outline=boundary_outline_segments(segments,2.,1.)
            for index,segment in enumerate(outline):
                axis.plot(*np.array(segment).T,color=color,label=f'Fine bins, background power {power:g}' if index==0 else None)
            if all(p['prompt_consistent'] for p in points):
                for segment in segments:
                    polygon=segment+[[segment[-1][0],1.],[segment[0][0],1.]]
                    axis.fill(*np.array(polygon).T,color=color,alpha=.08)
            summaries[f'p{int(power)}_cl{round(100*level)}']=dict(points=points,segments=segments,
                outline_segments=outline,confidence_level=level,conditional_on_background_model=True,
                all_boundaries_prompt_consistent=all(p['prompt_consistent'] for p in points),
                lower_coupling_islands_not_excluded_by_this_prompt_fit=True)
    baseline=json.loads((Path(config.source_mass_study_path).resolve()/'results/limits.json').read_text())['records']
    for level,axis in zip((.95,.90),axes):
        selected=sorted([r for r in baseline if r['channel']=='combined' and r['confidence_level']==level and r['status']=='ok'],key=lambda r:r['mass_gev'])
        axis.plot([r['mass_gev'] for r in selected],[r['epsilon']*math.sqrt(r['expected_limit_cross_section_pb']/r['reference_cross_section_pb']) for r in selected],':',color='gray',label='Previous adaptive-bin result')
        axis.axvspan(.2114,2,color='gray',alpha=.12,label='Below 2 GeV: no simulated response')
        axis.axhline(.1,color='gray',linestyle=':',linewidth=.8)
        axis.set(xscale='log',yscale='log',xlim=(.2114,90),ylim=(1.e-6,1.),xlabel='Dark-photon mass [GeV]',ylabel='Mixing epsilon',title=f'{round(level*100)}% CLs, 3 fb$^{{-1}}$')
        axis.legend(fontsize=7,loc='lower right')
    figure.suptitle('Fine mass-bin sensitivity: background-shape assumptions remain explicit')
    finish(figure,output,'mass_epsilon_comparison',
        'Bins follow a provisional 2% mass scale, limited by the saved 0.1 GeV grid; this is not a measured SHIFT core resolution.\n'
        'Actual broad signal response retained. Shading is conditional on the interpolated background and prompt consistency; lower-coupling islands remain a separate check.')
    save_json(output/'contours.json',summaries)
    # Show directly that fine likelihood bins do not inherit the broad MC regions.
    figure,axes=plt.subplots(1,2,figsize=(12,5.8))
    for power,color in zip(config.background_interpolation_powers,['#a33200','#3975a8']):
        row=json.loads((output/f'p{int(power)}'/'yields.json').read_text())['mass_binning']['Both-Both']
        edges=np.array(row['mass_edges_gev']);widths=np.diff(edges)
        axes[0].stairs(np.array(row['yields'])/widths,edges,label=f'Background power {power:g}',color=color)
        if power==0:
            axes[0].stairs(np.array(rebin(np.array(row['raw_fine_yields']),row['fine_bin_boundaries']))/widths,edges,color='gray',alpha=.6,label='Raw sparse MC')
            axes[1].stairs(widths,edges,color='black',label='Likelihood mass-bin width')
    axes[0].set(yscale='log',xlim=(0,100),xlabel='Dimuon mass [GeV]',ylabel='Expected background / GeV',title='Both-Both background interpolation')
    axes[1].set(yscale='log',xlim=(0,100),xlabel='Dimuon mass [GeV]',ylabel='Bin width [GeV]',title='Independent of MC population')
    for axis in axes:axis.legend(fontsize=8)
    finish(figure,output,'background_and_binning',
        'Each sparse-region total is preserved exactly; only its internal density is modelled. Zero raw bins are not assigned known zero background.\n'
        'The two densities are a sensitivity check, not a complete uncertainty envelope. The 30% global nuisances are retained.')
    print('Published fine-bin comparison',len(curves),'curves',flush=True)
