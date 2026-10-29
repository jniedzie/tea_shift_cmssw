#!/usr/bin/env python3
# Code style: AI-code. AI-generated comparison of saved limits inputs with phenomenology.
"""Audit saved yields and response without rerunning fits or ROOT scans."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from shift_limits import digest
from shift_limits_lifetime import save_json
from shift_paths import (limits_output_path, limits_mass_shape_output_path,
                         limits_cut_output_path, limits_pheno_comparison_output_path)


def saved_mass(name, original, directory):
    path = directory/'data'/(name+'_mass.npz')
    receipt = json.loads(path.with_suffix('.json').read_text())
    if (receipt['sha256'] != digest(path) or
            receipt['binding']['yields_sha256'] != original):
        raise ValueError('Mass cache does not match the audited yield receipt: '+name)
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


def quantile(values, probability):
    total = float(sum(values))
    if not total:return None
    cumulative = np.cumsum(values)
    index = min(int(np.searchsorted(cumulative,probability*total)),999)
    before = cumulative[index-1] if index else 0.
    return (index+(probability*total-before)/values[index])/10.


def summarize():
    source = Path(limits_output_path).resolve()/'yields.json'
    report = json.loads(source.read_text())
    directory = Path(limits_mass_shape_output_path).resolve()
    output = Path(limits_pheno_comparison_output_path).resolve()
    output.mkdir(parents=True,exist_ok=True)
    categories = report['categories']
    histograms,backgrounds,signals = {},{},{}
    for name,row in report['backgrounds'].items():
        data = saved_mass(name,digest(source),directory)
        factor = report['luminosity_pb']*row['cross_section_pb']/row['full_gen_sumw']
        histograms[name] = {category:data[category]*factor for category in categories}
        variances = {category:data[category+'_sumw2']*factor**2 for category in categories}
        populations = {}
        for category in categories+['inclusive']:
            values = np.sum([histograms[name][cat] for cat in categories],axis=0) if category=='inclusive' else histograms[name][category]
            variance = np.sum([variances[cat] for cat in categories],axis=0) if category=='inclusive' else variances[category]
            selected = float(sum(values[110:600]))
            selected_variance = float(sum(variance[110:600]))
            populations[category] = dict(all_mass=float(sum(values)),mass_11_60=selected,
                mass_11_60_variance=selected_variance,
                mass_11_60_effective_mc=selected**2/selected_variance if selected_variance else 0.)
        backgrounds[name] = dict(process=name.split('_')[0],cross_section_pb=row['cross_section_pb'],
            full_gen_sumw=row['full_gen_sumw'],processed_events=row['counts']['events'],
            yields=populations)
    process_yields = {}
    for process in ('qcd','dy','jpsi'):
        rows = [row for row in backgrounds.values() if row['process']==process]
        process_yields[process] = {category:{selection:math.fsum(row['yields'][category][selection] for row in rows)
            for selection in ('all_mass','mass_11_60')} for category in categories+['inclusive']}
    cut_directory = Path(limits_cut_output_path).resolve()
    joint_backgrounds = {}
    for name,row in report['backgrounds'].items():
        path = cut_directory/'data'/(name+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if digest(path)!=receipt['sha256']:raise ValueError('Joint cache changed')
        with np.load(path,allow_pickle=False) as archive:
            mass,weight,p = archive['mass'],archive['weight'],archive['minimum_muon_p']
            factor=report['luminosity_pb']*row['cross_section_pb']/row['full_gen_sumw']
            selected=(mass>=11)&(mass<60)&np.isfinite(p)&(p>30)
            selected_mass=(mass>=11)&(mass<60)
            reconstructed_mass_yield=factor*float(sum(weight[selected_mass]))
            if not math.isclose(reconstructed_mass_yield,backgrounds[name]['yields']['Both-Both']['mass_11_60'],rel_tol=2.e-5,abs_tol=1.e-8):
                raise ValueError('Joint and histogram mass selections disagree: '+name)
            joint_backgrounds[name]=dict(yield_events=factor*float(sum(weight[selected])),
                selected_mc=int(sum(selected)),mass_only_events=int(sum(selected_mass)),
                variance=factor**2*float(sum(weight[selected]**2)))
    for name,row in report['signals'].items():
        if row['mass_gev'] not in (2.,5.,10.,15.,30.):continue
        if not (name.endswith('_prompt') or '_lab_' in name):continue
        data=saved_mass(name,digest(source),directory)
        population={}
        for category in ('inclusive','Both-Both'):
            values=np.sum([data[cat] for cat in categories],axis=0) if category=='inclusive' else data[category]
            reconstructed=float(sum(values))
            mass=row['mass_gev']
            low=max(0,int(round(10*(mass-.5))));high=min(1000,int(round(10*(mass+.5))))
            population[category]=dict(acceptance=reconstructed/row['full_gen_sumw'],
                acceptance_mass_11_60=float(sum(values[110:600]))/row['full_gen_sumw'],
                mass_median_gev=quantile(values,.5),mass_q16_gev=quantile(values,.16),mass_q84_gev=quantile(values,.84),
                fraction_within_half_gev=float(sum(values[low:high]))/reconstructed if reconstructed else None)
        signals[name]=dict(mass_gev=row['mass_gev'],proper_ctau_m=row['proper_ctau_mm']/1000.,
            mean_lab_flight_m=row['mean_lab_flight_m'],epsilon=row['epsilon'],
            sigma_times_br_pb=row['reference_cross_section_pb'],response=population)
        if name.endswith('_prompt'):
            rate = json.loads(Path(row['source']['rate_reference']['path']).read_text())
            signals[name]['physical_br_mumu'] = rate['physical_br_mumu']
    reference=dict(source='https://inspirehep.net/files/f5de083d995be2caf06e3c1c63431949',
        version='Published JHEP10(2024)204, Tables 1-2',luminosity_pb=7150.,mass_range_gev=[11,60],
        each_muon_energy_min_gev=30.,expected_yields={'dy':7.e4,'qcd':8.e5},
        approximate_total_efficiencies={'dy':.3,'qcd':2.e-9},
        reconstructed_mass_bin_width_gev=.5,signal_and_background_model_differ=True)
    zero_background={}
    for name,row in signals.items():
        if not name.endswith('_prompt'):continue
        epsilon=row['epsilon'];sigma=row['sigma_times_br_pb']
        zero_background[name]={}
        for category in ('inclusive','Both-Both'):
            efficiency=row['response'][category]['acceptance']
            zero_background[name][category]={}
            for luminosity in (3000.,30000.):
                value=epsilon*math.sqrt(math.log(10)/(luminosity*sigma*efficiency)) if efficiency else None
                zero_background[name][category][str(luminosity)]=value
    content=dict(complete=True,luminosity_pb=report['luminosity_pb'],original_paper=reference,
        backgrounds=backgrounds,process_yields=process_yields,
        both_both_mass_11_60_and_p30=joint_backgrounds,signals=signals,
        hypothetical_zero_background_fixed_prompt_response_epsilon90=zero_background,
        zero_background_caveat='2.3026-event SensCalc counting criterion, no normalization nuisances, fixed prompt response; not a new Combine result or a lifetime-validated coupling contour.',
        source_yields=dict(path=str(source),sha256=digest(source)),
        no_root_scan_or_fit=True)
    content['generator_rate_sanity'] = dict(
        qcd_sigma_pb_current=math.fsum(row['cross_section_pb'] for row in backgrounds.values() if row['process']=='qcd'),
        qcd_sigma_pb_implied_by_rounded_paper_table=8.e5/(7150.*2.e-9),
        dy_sigma_pb_current_generated_above_10=math.fsum(row['cross_section_pb'] for name,row in backgrounds.items() if name in ('dy_10to20','dy_20to-1')),
        dy_sigma_pb_implied_by_rounded_paper_table=7.e4/(7150.*.3),
        caveat='Approximate checks only; DY supports differ at 10 versus 11 GeV and paper efficiencies are rounded.')
    native = json.loads((directory/'results/limits.json').read_text())['records']
    mass2 = next(row for row in native if row['point']=='m2_prompt' and row['channel']=='combined' and row['confidence_level']==.90)
    content['coupling_budget_2gev'] = dict(
        current_combined_epsilon90=mass2['epsilon']*math.sqrt(mass2['expected_limit_cross_section_pb']/mass2['reference_cross_section_pb']),
        current_equivalent_selected_signal_events=report['luminosity_pb']*mass2['expected_limit_cross_section_pb']*signals['m2_prompt']['response']['inclusive']['acceptance'],
        two_point_three_events_at_3fb=zero_background['m2_prompt']['inclusive']['3000.0'],
        two_point_three_events_at_30fb=zero_background['m2_prompt']['inclusive']['30000.0'])
    reference_table = output/'reference_data/senscalc_dy_cross_sections.txt'
    if reference_table.exists():
        coefficients = dict(np.loadtxt(reference_table))
        comparisons = {}
        for name,row in signals.items():
            if not name.endswith('_prompt') or row['mass_gev'] not in coefficients:continue
            coefficient = row['sigma_times_br_pb']/(row['epsilon']**2*row['physical_br_mumu'])
            comparisons[name] = dict(current_inclusive_coefficient_pb_per_epsilon2=coefficient,
                public_senscalc_coefficient_pb_per_epsilon2=coefficients[row['mass_gev']],
                senscalc_over_current=coefficients[row['mass_gev']]/coefficient)
        content['public_senscalc_production_comparison'] = dict(points=comparisons,
            table_sha256=digest(reference_table),
            caveat='Public fork cited by the report; not established as the exact input version of Fig.8.15. No cause assigned to its normalization difference.')
    curve_path = output/'reference_data/briefing_curve_readback.json'
    if curve_path.exists():content['briefing_vector_curve']=json.loads(curve_path.read_text())
    lifetime_path = output/'matched_lifetime_acceptance.json'
    if lifetime_path.exists():
        content['matched_proper_lifetime_check']=json.loads(lifetime_path.read_text())
    save_json(output/'comparison.json',content)
    archive={}
    for process in ('qcd','dy','jpsi'):
        names=[name for name in backgrounds if backgrounds[name]['process']==process]
        for category in ('inclusive','Both-Both'):
            archive[process+'_'+category]=np.sum([histograms[name][cat] for name in names for cat in (categories if category=='inclusive' else [category])],axis=0)
    np.savez_compressed(output/'comparison_mass_spectra.npz',**archive)
    for process,rows in process_yields.items():
        published=reference['expected_yields'].get(process)
        print(process,'all',rows['inclusive']['all_mass'],'11-60',rows['inclusive']['mass_11_60'],
              'paper_at_3fb',published*3000/7150 if published else None,flush=True)
    print('Both-Both + mass 11-60 + each p>30:',math.fsum(row['yield_events'] for row in joint_backgrounds.values()),flush=True)
    print('Wrote',output/'comparison.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--current-study',type=Path)
    parser.add_argument('--darkcast-path',type=Path)
    parser.add_argument('--darkcast-receipt',type=Path)
    parser.add_argument('--meson-model',type=Path)
    parser.add_argument('--source-grid',type=Path,help='Plot the bounded low-mass source-bank grid')
    parser.add_argument('--matched-benchmark',type=Path,
                        help='Compare paired original GEN and published RECO at fixed rates')
    parser.add_argument('--workflow-root',type=Path)
    parser.add_argument('--production-root',type=Path)
    parser.add_argument('--benchmark-events',type=int,default=1000)
    parser.add_argument('--benchmark-read',action='store_true',
                        help='Read the paired GEN/Nano cache in a CMSSW FWLite environment')
    args=parser.parse_args()
    if args.source_grid is not None:
        plot_source_grid(args.source_grid)
        return
    if args.matched_benchmark is not None:
        if args.benchmark_read:
            if args.workflow_root is None or args.production_root is None:
                parser.error('Reading the benchmark requires workflow and production roots')
            read_matched_benchmark(args.matched_benchmark,args.workflow_root,
                                   args.production_root,args.benchmark_events)
        else:
            matched_benchmark(args.matched_benchmark)
        return
    if args.current_study is not None:
        if any(value is None for value in (args.darkcast_path,args.darkcast_receipt,args.meson_model)):
            parser.error('Current-study diagnostics require the pinned width provider and meson model')
        current_shape_diagnostics(args.current_study,args.darkcast_path,args.darkcast_receipt,args.meson_model)
        return
    plot_saved() if args.plot else summarize()


def plot_saved():
    from shift_limits_lifetime import plotting,finish
    output=Path(limits_pheno_comparison_output_path).resolve()
    data=json.loads((output/'comparison.json').read_text())
    with np.load(output/'comparison_mass_spectra.npz',allow_pickle=False) as archive:
        spectra={key:archive[key].copy() for key in archive.files}
    plt=plotting()
    colors={'qcd':'#8b8b8b','dy':'#467aad','jpsi':'#a94f9a'}
    figure,axes=plt.subplots(1,2,figsize=(12,6))
    labels=['Published paper\nscaled to 3 fb$^{-1}$','Current inclusive\nall masses','Current inclusive\n11–60 GeV','Current Both–Both\n11–60 GeV']
    x=np.arange(4)
    for i,process in enumerate(('qcd','dy','jpsi')):
        paper=data['original_paper']['expected_yields'].get(process)
        current=data['process_yields'][process]
        values=[paper*3000/7150 if paper else np.nan,current['inclusive']['all_mass'],
                current['inclusive']['mass_11_60'],current['Both-Both']['mass_11_60']]
        axes[0].bar(x+(i-1)*.23,values,.23,label=process,color=colors[process])
    axes[0].set_xticks(x);axes[0].set_xticklabels(labels,fontsize=8)
    axes[0].set(yscale='log',ylabel='Expected events at 3 fb$^{-1}$',ylim=(1.e3,2.e12),title='Different selected populations')
    axes[0].legend(fontsize=8)
    categories=['Near-Both','Near-Far','Both-Both','Both-Far','Other']
    for i,process in enumerate(('qcd','dy','jpsi')):
        axes[1].bar(np.arange(5)+(i-1)*.23,[data['process_yields'][process][c]['mass_11_60'] for c in categories],.23,label=process,color=colors[process])
    axes[1].set_xticks(np.arange(5));axes[1].set_xticklabels(categories,fontsize=8,rotation=15)
    axes[1].set(yscale='log',ylabel='Expected events, 11–60 GeV',title='Current mass-window yield by category')
    figure.suptitle('Background yields: exposure matched, selections still different')
    finish(figure,output,'01_background_yields',
        'Original published Table 1 includes angular/material acceptance, E(mu)>30 GeV and mass 11–60 GeV.\n'
        'Current bars use saved RECO candidates. Missing bars mean empty/unprovided MC, not known zero physical background.')

    figure,axes=plt.subplots(1,2,figsize=(12,5.8))
    joint=data['both_both_mass_11_60_and_p30']
    for axis,process,title in zip(axes,('dy','jpsi'),('DY: generator mass interval [GeV]','J/psi: generator pThat interval [GeV]')):
        names=[n for n in data['backgrounds'] if n.startswith(process+'_')]
        names=sorted(names,key=lambda n:float(n.split('_',1)[1].split('to')[0]))
        values=[joint[n]['yield_events'] for n in names]
        axis.bar(np.arange(len(names)),values,color=colors[process])
        axis.set_xticks(np.arange(len(names)));axis.set_xticklabels([n.split('_',1)[1] for n in names],rotation=35,ha='right',fontsize=8)
        axis.set(yscale='log',ylabel='Expected events at 3 fb$^{-1}$',title=title,ylim=(1,max(values)*5))
        for i,name in enumerate(names):
            if values[i]>0:axis.text(i,values[i]*1.15,str(joint[name]['selected_mc'])+' MC',ha='center',fontsize=7)
    figure.suptitle('Which inputs populate reconstructed mass 11–60 GeV? Both–Both, p(mu)>30 GeV')
    finish(figure,output,'02_low_mass_feedthrough',
        'Most high-mass DY candidates arise in low-generation-mass samples; J/psi also develops a reconstructed high-mass tail.\n'
        'This is candidate-level feedthrough, not a matched-pair resolution measurement. Weighted yields can be dominated by very few MC events.')

    figure,axes=plt.subplots(1,2,figsize=(12,5.8))
    edges=np.linspace(0,100,201)
    source=Path(limits_output_path).resolve()/'yields.json'
    report=json.loads(source.read_text())
    for axis,category in zip(axes,('inclusive','Both-Both')):
        for process in ('qcd','dy','jpsi'):
            values=spectra[process+'_'+category].reshape(200,5).sum(axis=1)
            axis.stairs(np.where(values>0,values,np.nan),edges,color=colors[process],label=process,linewidth=1.3)
        signal=saved_mass('m30_prompt',digest(source),Path(limits_mass_shape_output_path).resolve())
        values=np.sum([signal[c] for c in report['categories']],axis=0) if category=='inclusive' else signal[category]
        values=values.reshape(200,5).sum(axis=1)*3000*.2/report['signals']['m30_prompt']['full_gen_sumw']
        axis.stairs(np.where(values>0,values,np.nan),edges,color='#d95319',label='30 GeV prompt, sigma*BR=0.2 pb')
        axis.set(xlim=(11,60),yscale='log',ylim=(.1,1.e10),xlabel='Reconstructed dimuon mass [GeV]',ylabel='Events / 0.5 GeV at 3 fb$^{-1}$',title=category)
        axis.legend(fontsize=7)
    figure.suptitle('Current unmerged mass spectra in the paper\'s mass range')
    finish(figure,output,'03_background_mass_spectra',
        '0.5 GeV display bins match the paper; empty MC bins are omitted. These are not the adaptive bins used for fits.\n'
        'The sparse QCD and J/psi tails are not smooth predictions. Signal is rescaled to the same 0.2 pb visual benchmark as the paper.')

    figure,axes=plt.subplots(1,2,figsize=(12,5.8))
    for axis,mass in zip(axes,(15.,30.)):
        name=f'm{mass:g}_prompt';sample=report['signals'][name]
        values=saved_mass(name,digest(source),Path(limits_mass_shape_output_path).resolve())
        for category,color in [('inclusive','#467aad'),('Both-Both','#d95319')]:
            weights=np.sum([values[c] for c in report['categories']],axis=0) if category=='inclusive' else values[category]
            weights=weights.reshape(200,5).sum(axis=1)
            axis.stairs(weights/sum(weights),edges,label=category,color=color)
        axis.axvline(mass,color='black',linestyle=':',label='Generated pole mass')
        axis.axvspan(mass-.5,mass+.5,color='gray',alpha=.18)
        axis.set(xlim=(0,60),xlabel='Reconstructed mass [GeV]',ylabel='Fraction of reconstructed candidates / 0.5 GeV',title=f'{mass:g} GeV prompt')
        axis.legend(fontsize=8)
    figure.suptitle('Signal candidates no longer form the narrow peak assumed by the pheno study')
    finish(figure,output,'04_signal_mass_response',
        'Grey band: nominal mass +/-0.5 GeV. Only 6.2% (15 GeV) and 3.6% (30 GeV) of Both–Both candidates enter it.\n'
        'Includes the actual first-candidate response used in limits, with incomplete matching; not the resolution of a truth-matched signal pair.')

    figure,axes=plt.subplots(1,2,figsize=(12,5.8))
    points=[(name,row) for name,row in data['signals'].items() if row['mass_gev']==30.]
    points=sorted(points,key=lambda item:item[1]['proper_ctau_m'])
    for category in ('inclusive','Both-Both'):
        axes[0].plot([r['proper_ctau_m'] for _,r in points],[r['response'][category]['acceptance_mass_11_60'] for _,r in points],'o-',label='Current '+category)
    axes[0].plot([1.e-3,10.,1000.],[.5,.2,.002],'s--',color='black',label='Published Table 2, 30 GeV')
    matched=data.get('matched_proper_lifetime_check')
    if matched:
        response=matched['results']['Both-Both']
        axes[0].errorbar(matched['target_ctau_m'],response['acceptance'],yerr=response['mc_error'],
            fmt='D',color='#a33200',capsize=4,label='Current, reweighted to proper 10 m')
    axes[0].set(xscale='log',yscale='log',xlabel='Proper mean decay length c*tau [m]',ylabel='Selected / full GEN denominator',title='Use the same lifetime coordinate')
    axes[0].legend(fontsize=8)
    actual=json.loads((Path(limits_mass_shape_output_path).resolve()/'shape_yields.json').read_text())['mass_binning']['Both-Both']
    widths=np.diff(actual['mass_edges_gev'])
    axes[1].bar(np.arange(len(widths)),widths,color='#467aad')
    axes[1].axhline(.5,color='black',linestyle=':',label='Paper: 0.5 GeV')
    axes[1].set(yscale='log',ylabel='Mass bin width [GeV]',xlabel='Current Both–Both bin number',title='Mass information lost to sparse background MC')
    axes[1].legend(fontsize=8)
    figure.suptitle('Acceptance and binning are different from the idealized projection')
    finish(figure,output,'05_lifetime_and_binning',
        'Our 100/1000 m labels refer to mean LAB flights; at 30 GeV these correspond to proper c*tau=1.47/14.7 m.\n'
        'At matched proper c*tau=10 m: current 2.69 +/-0.61%, paper 20%. Current production, transport and reconstruction differ.\n'
        'Paper also requires E(mu)>30 GeV; this check does not impose that extra cut. Full GEN denominator includes failed reconstruction.')

    budget=data['coupling_budget_2gev'];reference=data['briefing_vector_curve']['epsilon_crossings']['2.0'][0]
    values=[budget['current_combined_epsilon90'],budget['two_point_three_events_at_3fb'],budget['two_point_three_events_at_30fb'],reference]
    figure,axis=plt.subplots(figsize=(11,5.8))
    labels=['Current combined\n90% CLs, 3 fb$^{-1}$','2.3-event criterion\nfixed response, 3 fb$^{-1}$','2.3-event criterion\nfixed response, 30 fb$^{-1}$','Briefing-book SHIFT\nvector curve at 2 GeV']
    axis.scatter(np.arange(4),values,s=65,color=['#a33200','#467aad','#467aad','#00bfff'])
    axis.set_xticks(np.arange(4));axis.set_xticklabels(labels,fontsize=9)
    axis.set(yscale='log',ylabel='Mixing epsilon',ylim=(1.e-6,.2),title='2 GeV: the event requirement explains most of the gap')
    for i,value in enumerate(values):axis.annotate(f'{value:.3g}',(i,value),xytext=(0,10),textcoords='offset points',ha='center')
    finish(figure,output,'06_coupling_gap_budget',
        'Middle points are rate-budget counterfactuals: zero background, no nuisances and fixed measured prompt response. They are not new physical limits.\n'
        'Report uses the 2.3-event criterion; the exact background prescription/version of the briefing-book curve remains unverified. Epsilon changes the lifetime too.')


def radiative_phase_space(masses,parent_mass):
    """Pointlike P -> photon A' phase space, normalized to its massless value."""
    masses=np.asarray(masses,dtype=float)
    if parent_mass<=0 or np.any(masses<0):raise ValueError('Positive parent and nonnegative daughter masses required')
    return np.maximum(1-(masses/parent_mass)**2,0.)**3


def current_shape_diagnostics(study,darkcast_path,receipt_path,meson_model):
    """Separate sparse response, physical mass dependence and model coverage."""
    import sys
    from types import SimpleNamespace
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import norm
    from shift_limits_current import Model
    from shift_limits_profile import profile_q
    from shift_limits_lifetime import plotting

    study=study.resolve();output=study/'shape_diagnostics';output.mkdir(exist_ok=True)
    audit=json.loads((output/'numerical_audit.json').read_text())
    source=json.loads((output/'source_audit.json').read_text())
    if audit['snapshot']['sha256']!=digest(study/'snapshot.json'):
        raise ValueError('Numerical audit belongs to another input snapshot')
    receipt=json.loads(receipt_path.read_text())
    for name,expected in receipt['files'].items():
        if digest(darkcast_path/name)!=expected:raise ValueError('Pinned width provider changed: '+name)
    sys.path.insert(0,str(darkcast_path.resolve()))
    import darkcast
    photon=darkcast.Model('dark_photon');threshold=2*darkcast.pars.mfs['mu']
    parents=json.loads(meson_model.read_text())['parents']
    parent_masses={row['pdg_id']:row['mass_gev'] for row in parents}
    masses=np.unique(np.r_[np.linspace(.209,.26,160),np.geomspace(.26,5.,500),
        threshold,threshold+1.e-8,parent_masses[221],parent_masses[331],.77549,.78265,1.01946])
    total=np.array([photon.width('total',float(m),.001) for m in masses])
    muon=np.array([photon.width('mu_mu',float(m),.001) for m in masses])
    branching=np.divide(muon,total,out=np.zeros_like(muon),where=total>0)
    if np.any(muon[masses<=threshold]!=0) or not np.all(muon[masses>threshold]>0):
        raise ValueError('Dimuon threshold closure failed in the pinned provider')

    snapshot=json.loads((study/'snapshot.json').read_text())
    components={}
    for name in ['m0p25_lab_0p01m','m0p7_lab_0p01m']:
        row=next(row for row in snapshot['signals'] if row['name']==name)
        if digest(row['rate_authority'])!=row['rate_authority_sha256']:
            raise ValueError('Rate authority changed: '+name)
        rate=json.loads(Path(row['rate_authority']).read_text());channels={}
        for weight,population in zip(rate['combination']['normalized_weights'],rate['per_population']):
            for channel,value in population['record']['source_exposure']['raw_source_summary']['channels'].items():
                channels[channel]=channels.get(channel,0.)+weight*value['sigma_over_epsilon_squared_pb']
        total_rate=sum(channels.values())
        expected=sum(value['value_pb'] for value in rate['physical_inclusive_cross_sections_pb'])/rate['epsilon']**2
        if not math.isclose(total_rate,expected,rel_tol=2.e-6):raise ValueError('Production channels do not close: '+name)
        components[name]={channel:value/total_rate for channel,value in channels.items()}
    config=SimpleNamespace(output_path=str(study),luminosity=300.,categories=tuple(snapshot['categories']),
        primary_channel='Both-Both',response_anchors=('prompt','lab_0p01m','lab_5m','lab_30m','lab_100m','lab_1000m'))
    model=Model(config);category=config.categories.index('Both-Both')
    anchors={row['point']:row for row in audit['anchor_table']}
    chosen=[anchors[name] for name in ['m0p25_lab_0p01m','m0p7_lab_0p01m','m2_prompt','m5_prompt']]
    epsilon_grid=np.geomspace(1.e-5,.004,100);q_values={key:[] for key in ['actual','without_tail','fixed_cm_shape','fixed_prompt_shape']}
    cm=next(row for row in snapshot['signals'] if row['name']=='m0p7_lab_0p01m')
    prompt=next(row for row in snapshot['signals'] if row['name']=='m0p7_prompt')
    cm_shape=model.arrays[cm['array_key']][category];cm_shape=cm_shape/cm_shape.sum()
    prompt_shape=model.arrays[prompt['array_key']][category];prompt_shape=prompt_shape/prompt_shape.sum()
    driver=audit['low07_single_tail_driver'];edges=np.array(snapshot['mass_edges_gev'])
    bin_index=int(np.flatnonzero(np.isclose(edges[:-1],driver['mass_bin_gev'][0]))[0])
    for epsilon in epsilon_grid:
        signal,background=model.cells(.7,float(epsilon),'Both-Both','background')
        omitted=signal.copy();omitted[bin_index]=0.
        for key,values in [('actual',signal),('without_tail',omitted),
                           ('fixed_cm_shape',cm_shape*signal.sum()),('fixed_prompt_shape',prompt_shape*signal.sum())]:
            q_values[key].append(profile_q(1.,values,background))
    save_json(output/'mass_shape_diagnostic.json',dict(schema='shift-limit-mass-shape-diagnostic-v1',
        snapshot_sha256=digest(study/'snapshot.json'),width_provider_receipt_sha256=digest(receipt_path),
        meson_model_sha256=digest(meson_model),threshold_gev=threshold,masses_gev=masses.tolist(),
        BR_mumu=branching.tolist(),eta_phase_space=radiative_phase_space(masses,parent_masses[221]).tolist(),
        eta_prime_phase_space=radiative_phase_space(masses,parent_masses[331]).tolist(),
        source_channel_fractions=components,
        epsilon_grid=epsilon_grid.tolist(),profile_q=q_values,
        interpretation='Width/phase-space curves are physics diagnostics, not a densely simulated exclusion. Fixed shapes and tail omission are counterfactuals; published inputs and limits are unchanged.'))

    plt=plotting();path=output/'limit_shape_diagnostics.pdf'
    with PdfPages(path) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(13,10));fig.suptitle('Why the current mass contour misses expected structures')
        axis=axes[0,0];axis.plot(masses,branching,color='tab:blue',label='Pinned DarkCast BR to dimuons')
        axis.axvspan(threshold,.25,color='.85',label='No new RECO mass anchor')
        for mass,label in [(threshold,'2 mu'),(.77549,'rho / omega'),(1.01946,'phi')]:axis.axvline(mass,color='.6',ls=':',lw=1)
        axis.scatter([r['mass_gev'] for r in chosen if r['mass_gev']<=5],
                     [r['physical_BR_mumu'] for r in chosen if r['mass_gev']<=5],color='black',s=22,label='Actual limit mass anchors')
        axis.set(xscale='log',xlim=(.209,5),ylim=(0,.55),xlabel='Dark-photon mass [GeV]',ylabel='BR(A prime -> mu mu)',title='The dimuon turn-on is present in the width model');axis.legend(fontsize=8)
        axis=axes[0,1]
        for pdg,label,color in [(221,'eta -> gamma A prime','tab:orange'),(331,"eta prime -> gamma A prime",'tab:green')]:
            axis.plot(masses,radiative_phase_space(masses,parent_masses[pdg]),color=color,label=label)
            axis.axvline(parent_masses[pdg],color=color,ls=':',lw=1)
        for mass in [.25,.7]:axis.axvline(mass,color='black',ls='--',lw=.8)
        axis.set(xlim=(.21,1.05),ylim=(0,1),xlabel='Dark-photon mass [GeV]',ylabel='Normalized pointlike phase space',title='Radiative eta endpoints lie between missing mass points');axis.legend(fontsize=8)
        axis=axes[1,0]
        for group,color,label in [(chosen[:2],'tab:green','Mesons + fragmentation + FR'),(chosen[2:],'tab:blue','Drell-Yan only')]:
            axis.plot([r['mass_gev'] for r in group],[r['selected_sigma_over_epsilon_squared_pb'] for r in group],'o-',color=color,label=label)
        axis.axvspan(.7,2,color='.9',label='Unsampled mechanism transition')
        axis.set(xscale='log',yscale='log',xlabel='Mass [GeV]',ylabel='sigma times BR / epsilon squared [pb]',title='The current connected contour changes production recipe');axis.legend(fontsize=8)
        axis.text(.03,.04,f"0.25 GeV: eta = {100*components['m0p25_lab_0p01m']['eta-decay']:.1f}% of source",transform=axis.transAxes,fontsize=8)
        axis=axes[1,1]
        current=[r for r in chosen if r['mass_gev'] in [2,5,10,15]]
        reference=source['reference_table'];table=np.loadtxt(reference['path'])
        if digest(reference['path'])!=reference['sha256']:raise ValueError('Public reference DY table changed')
        public={float(m):float(value) for m,value in table}
        axis.plot([r['mass_gev'] for r in current],[r['inclusive_sigma_over_epsilon_squared_pb']/current[0]['inclusive_sigma_over_epsilon_squared_pb'] for r in current],'o-',label='Current DY, normalized at 2 GeV')
        axis.plot([r['mass_gev'] for r in current],[public[r['mass_gev']]/public[2.] for r in current],'s--',label='Public author fork DY, normalized at 2 GeV')
        axis.set(xscale='log',yscale='log',xlabel='Mass [GeV]',ylabel='Inclusive rate coefficient / value at 2 GeV',title='Neither DY prediction has a flat production rate');axis.legend(fontsize=8)
        fig.text(.025,.02,'Dense curves show decay/phase-space physics only; actual detector templates exist at sparse black markers.\nMeson endpoints suppress radiative production; vector rho/omega/phi structures are distinct. Public fork is not an authenticated record of the book run.',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.95));pdf.savefig(fig);plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(13,10));fig.suptitle('The current low-mass limit is sensitive to single reconstructed tail events')
        axis=axes[0,0];axis.stairs(cm_shape,edges,label='0.7 GeV, LAB1cm (N effective = 16.9)',color='tab:blue')
        axis.stairs(prompt_shape,edges,label='0.7 GeV, prompt (N effective = 1)',color='tab:orange')
        axis.axvspan(*driver['mass_bin_gev'],color='red',alpha=.2,label='One effective event at 14.8--15 GeV')
        axis.set(yscale='log',xlim=(0,20),ylim=(.001,2),xlabel='Reconstructed dimuon mass [GeV]',ylabel='Fraction of selected weighted signal',title='Generated mass 0.7 GeV; the driver is reconstructed near 15 GeV');axis.legend(fontsize=7.5)
        axis=axes[0,1]
        for key,label,style in [('actual','Actual interpolated templates','-'),('without_tail','Only the 15 GeV signal bin omitted','--'),('fixed_cm_shape','Fixed 1 cm shape, same total yield',':'),('fixed_prompt_shape','Fixed prompt shape, same total yield','-.')]:
            axis.plot(epsilon_grid,q_values[key],style,label=label)
        axis.axhline(norm.ppf(.975)**2,color='black',lw=1,label='95% threshold')
        axis.set(xscale='log',yscale='log',ylim=(1.e-4,1.e3),xlabel='Mixing epsilon',ylabel='Profile q',title='One reconstructed tail bin creates\nthe extra exclusion island');axis.legend(fontsize=7)
        axis=axes[1,0];names=['m0p25_lab_0p01m','m0p7_lab_0p01m','m0p7_prompt','m2_prompt','m5_prompt']
        cases=[next(r for r in audit['template_background_counterfactuals'] if r['point']==name) for name in names]
        axis.bar(np.arange(len(cases)),[r['required_events_background'] for r in cases],color='#b33636',label='Required signal with current SM/template shape')
        axis.axhline(audit['required_events_zero_background'],color='#176b9c',ls='--',label='Zero-background requirement')
        axis.set_xticks(np.arange(len(cases)),['0.25 cm','0.7 cm','0.7 prompt','2 prompt','5 prompt']);axis.set(yscale='log',ylabel='Required selected signal events',title='Background and template shape set a second mass dependence');axis.legend(fontsize=7)
        axis=axes[1,1]
        for study_name,offset,color in [('old',-.08,'gray'),('new',.08,'tab:blue')]:
            rows=sorted([r for r in audit['matched_luminosity_prompt_comparison'] if r['study']==study_name],key=lambda r:r['mass'])
            for i,row in enumerate(rows):
                axis.plot([i+offset]*2,[row['epsilon_limit_zeroB'],row['epsilon_limit_fixed_prompt']],color=color,ls=':')
            axis.scatter(np.arange(len(rows))+offset,[r['epsilon_limit_fixed_prompt'] for r in rows],color=color,label=study_name+' fixed prompt, current background')
            axis.scatter(np.arange(len(rows))+offset,[r['epsilon_limit_zeroB'] for r in rows],facecolors='white',edgecolors=color,label=study_name+' fixed prompt, zero background')
        axis.set_xticks([0,1],['2 GeV','5 GeV']);axis.set(yscale='log',ylabel='Mixing epsilon',title='Old/new at the SAME 0.3 fb^-1:\nsensitivity to reconstructed tails');axis.legend(fontsize=7)
        fig.text(.025,.02,'All counterfactuals use the unchanged 0.3 fb^-1 likelihood and 30% proxy nuisances; MC-stat uncertainty is absent.\nTail omission/fixed shapes are diagnostics, not new selections or published limits. More GEN alone cannot repair a sparse mass grid or authenticate the reference.',fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.95));pdf.savefig(fig);plt.close(fig)
    print('Saved',path,flush=True)


def ideal_pair_pointing(muons, energy_min=0., plane_m=160., inner_m=1., outer_m=7.5):
    """Straight GEN rays from a common point source; no transport or magnetic field.

    Rows contain [px,py,pz,E] in GeV, with CMS in the negative-z direction.
    This is an explicitly simplified geometry diagnostic, not a RECO selection.
    """
    values=np.asarray(muons,dtype=float)
    if (values.shape[-2:]!=(2,4) or not np.all(np.isfinite(values))
            or not 0<=inner_m<outer_m or plane_m<=0 or energy_min<0):
        raise ValueError('Two finite muon momenta and a valid forward annulus are required')
    forward=values[...,2]<0
    radii=plane_m*np.hypot(values[...,0],values[...,1])/np.where(forward,-values[...,2],1.)
    return np.all(forward & (radii>inner_m) & (radii<outer_m) & (values[...,3]>energy_min),axis=-1)


def weighted_response_summary(contributions):
    """Full-original-event mean, including zero contributions from unsampled IDs."""
    values=np.asarray(contributions,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.all(np.isfinite(values)):
        raise ValueError('A finite original-event response vector is required')
    total=float(values.sum());squared=float(values@values);count=len(values)
    return dict(original_events=count,sumw=total,sumw2=squared,efficiency=total/count,
                standard_error=math.sqrt(max(0.,squared-total**2/count)/(count*(count-1))),
                effective_events=total**2/squared if squared else None,
                zero_observed_response_unknown=not bool(np.any(values)))


def reference_counting_thresholds(mass,luminosity_pb):
    """Keep two different statistical prescriptions explicitly separate.

    The public notebook hardcodes 3000/750; extending it linearly in exposure
    is a declared diagnostic assumption, not recovered metadata of its cache.
    """
    if mass<=0 or luminosity_pb<=0:raise ValueError('Positive mass and luminosity required')
    background=luminosity_pb/750.*10**(6.1*math.exp(-.06*mass)+2.8*mass**(-.2))
    return dict(smooth_background_events=background,
        exact_zero_observation_Poisson90=math.log(10),
        SensCalc_style_one_sided_Gaussian90_proxy=1.28*math.sqrt(background))


def fixed_response_coupling(required,luminosity_pb,sigma_over_epsilon2,efficiency):
    if (not all(math.isfinite(v) for v in (required,luminosity_pb,sigma_over_epsilon2,efficiency))
            or min(required,luminosity_pb,sigma_over_epsilon2)<=0 or efficiency<0):
        raise ValueError('Positive count, exposure and physical selected rate required')
    return math.sqrt(required/(luminosity_pb*sigma_over_epsilon2*efficiency)) if efficiency>0 else None


def benchmark_admission_probabilities(selection,original_ids):
    indices=selection['selected_indices'];probabilities=selection['sampling_probabilities']
    if (len(indices)!=len(probabilities) or len(set(indices))!=len(indices)
            or any(type(i) is not int or not 0<=i<len(original_ids) for i in indices)
            or any(not math.isfinite(p) or not 0<p<=1 for p in probabilities)):
        raise ValueError('Invalid frozen full-GEN admission indices or probabilities')
    return {tuple(original_ids[i]):p for i,p in zip(indices,probabilities)}


def read_matched_benchmark(study,workflow,production,events):
    """Read one fixed original prefix and its complete existing Nano response."""
    import sys
    import gzip
    import hashlib
    import ROOT
    from DataFormats.FWLite import Events
    sys.path.insert(0,str(workflow.resolve()/'scripts'))
    from audit_dark_photon_gen import objects,read_product
    ROOT.gSystem.Load('libFWCoreFWLite');ROOT.FWLiteEnabler.enable()
    study=study.resolve();output=study/'shape_diagnostics';baseline=output/'ideal_gen_pointing.json'
    points=json.loads(baseline.read_text())['points'];results=[]
    target=output/'matched_benchmark_events.json.gz'
    if target.exists():raise FileExistsError('Reuse the frozen paired benchmark cache')
    for point in points:
        if events!=point['events']:
            raise ValueError('Benchmark prefix must match the existing frozen ideal-pointing test')
        wave=point['path'].split('/GEN/')[0].rsplit('-w',1)[1]
        deployment=production.resolve()/'direct/waves'/('w'+wave)/'production'
        plan=json.loads((deployment/'plan.json').read_text());row=plan['points'][0]
        if row['mass_gev']!=point['mass_gev']:raise ValueError('Paired physical point differs')
        source_path=Path('/eos/'+point['path'].split('/eos/',1)[1])
        source_hash=hashlib.sha256()
        with source_path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):source_hash.update(block)
        if source_hash.hexdigest()!=point['original_gen_sha256']:raise ValueError('Original GEN hash changed')
        validation=Path(point['membership_audit']['path'])
        if digest(validation)!=point['membership_audit']['sha256']:raise ValueError('GEN identity audit changed')
        original_ids=json.loads(validation.read_text())['event_ids'];ids=original_ids[:events];records=[]
        for event in Events(point['path']):
            if len(records)==events:break
            aux=event.eventAuxiliary();identity=[int(aux.run()),int(aux.luminosityBlock()),int(aux.event())]
            if identity!=ids[len(records)]:raise ValueError('Original GEN ordering changed')
            hep=read_product(event,'generatorSmeared','edm::HepMCProduct').GetEvent()
            candidates=[]
            for particle in objects(hep.particles_begin(),hep.particles_end()):
                if abs(int(particle.pdg_id()))!=32 or not particle.end_vertex():continue
                daughters=list(objects(particle.end_vertex().particles_out_const_begin(),
                                       particle.end_vertex().particles_out_const_end()))
                if len(daughters)==2 and all(abs(int(d.pdg_id()))==13 for d in daughters):
                    candidates.append((particle,daughters))
            if len(candidates)!=1:raise ValueError('GEN event needs one DP with two direct muons')
            parent,daughters=candidates[0]
            p4=lambda particle:[float(getattr(particle.momentum(),key)()) for key in ('px','py','pz','e')]
            weight=float(read_product(event,'generator','GenEventInfoProduct').weight())
            if weight!=1.:raise ValueError('This fixed-prefix binomial benchmark requires unit native GEN weights')
            records.append(dict(identity=identity,parent=p4(parent),muons=[p4(d) for d in daughters],
                                native_weight=weight,reco=None))
        if len(records)!=events:raise ValueError('Incomplete original GEN prefix')
        report_path=deployment/'completed_points'/(row['point']+'.json')
        report=json.loads(report_path.read_text())
        if report.get('complete') is not True:raise ValueError('Paired reconstruction has not completed')
        marker=report['main_histograms'];nano=marker['published']['merged_nano']
        by_id={tuple(record['identity']):record for record in records}
        selection=row['material_selection']
        probabilities=benchmark_admission_probabilities(selection,original_ids)
        selected_ids={tuple(identity) for identity in ids if tuple(identity) in probabilities}
        f=ROOT.TFile.Open('root://eosuser.cern.ch/'+nano['path']);tree=f.Get('Events')
        names=['run','luminosityBlock','event','genWeight','shiftSamplingProbability','shiftSamplingGenWeight',
               'nShiftDimuonVertex','ShiftDimuonVertex_topologyMin','ShiftDimuonVertex_topologyMax','ShiftDimuonVertex_mass']
        tree.SetBranchStatus('*',0)
        for name in names:
            if not tree.GetBranch(name):raise ValueError('Missing paired Nano branch: '+name)
            tree.SetBranchStatus(name,1)
        seen=set();full_sum=0.;full_sum2=0.
        for index in range(tree.GetEntries()):
            tree.GetEntry(index);value=lambda name,i=0:float(tree.GetLeaf(name).GetValue(i))
            identity=tuple(int(value(name)) for name in ['run','luminosityBlock','event'])
            probability=value('shiftSamplingProbability');weight=value('shiftSamplingGenWeight')
            if not math.isclose(weight,value('genWeight')/probability,rel_tol=2e-12):
                raise ValueError('Nano does not contain native W/p exactly once')
            masses=[value('ShiftDimuonVertex_mass',i) for i in range(int(value('nShiftDimuonVertex')))
                    if value('ShiftDimuonVertex_topologyMin',i)==2 and value('ShiftDimuonVertex_topologyMax',i)==2]
            if len(masses)>1:raise ValueError('Benchmark assumes at most one Both-Both dimuon per original event')
            if masses:full_sum+=weight;full_sum2+=weight**2
            if identity not in by_id:continue
            if identity in seen or identity not in selected_ids:raise ValueError('Unexpected or duplicate paired Nano ID')
            seen.add(identity)
            if not math.isclose(probability,probabilities[identity],rel_tol=2e-12):
                raise ValueError('Frozen admission probability changed')
            by_id[identity]['reco']=dict(probability=probability,weight=weight,masses=masses)
        f.Close()
        if seen!=selected_ids:raise ValueError('Missing detector responses are not physical rejection')
        histogram=marker['published']['merged_histogram'];f=ROOT.TFile.Open('root://eosuser.cern.ch/'+histogram['path'])
        h=f.Get('dimuon/ShiftDimuonVertexBoth-Both_mass')
        if not math.isclose(h.Integral(0,h.GetNbinsX()+1),full_sum,rel_tol=2e-6):
            raise ValueError('Nano Both-Both selection differs from the ordinary main histogrammer')
        f.Close()
        muons=np.asarray([record['muons'] for record in records])
        for label,cut in [('geometry',0.),('geometry_E20',20.),('geometry_E30',30.)]:
            if int(ideal_pair_pointing(muons,cut).sum())!=point['counts'][label]:
                raise ValueError('Paired GEN reader does not reproduce the frozen ideal control')
        results.append(dict(mass_gev=point['mass_gev'],source=point,
            production_plan=dict(path=str(deployment/'plan.json'),sha256=digest(deployment/'plan.json')),
            completed_report=dict(path=str(report_path),sha256=digest(report_path)),nano=nano,
            matched_original_events=events,matched_processed_events=len(seen),full_cohort_BothBoth_sumw=full_sum,
            full_cohort_BothBoth_sumw2=full_sum2,missing_selected_responses=0,records=records))
        print('Read paired',point['mass_gev'],'GeV:',len(seen),'processed responses',flush=True)
    payload=dict(schema='shift-matched-GEN-RECO-benchmark-events-v1',complete=True,
        ideal_control=dict(path=str(baseline),sha256=digest(baseline)),reader=dict(path=str(Path(__file__).resolve()),
        sha256=digest(__file__)),points=results)
    with gzip.open(target,'wt') as stream:json.dump(payload,stream,separators=(',',':'),allow_nan=False)
    save_json(output/'matched_benchmark_events_receipt.json',dict(path=str(target),sha256=digest(target)))


def matched_benchmark(study):
    """Publish the same-event acceptance/rate/background budget, not a new limit."""
    import gzip
    import csv
    from shift_limits_lifetime import plotting
    study=study.resolve();output=study/'shape_diagnostics';receipt_path=output/'matched_benchmark_events_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    if digest(receipt['path'])!=receipt['sha256']:raise ValueError('Paired event cache changed')
    with gzip.open(receipt['path'],'rt') as stream:cache=json.load(stream)
    snapshot=json.loads((study/'snapshot.json').read_text());rows=[];points=[]
    for point in cache['points']:
        mass=point['mass_gev'];source=next(r for r in snapshot['signals'] if r['mass_gev']==mass and r['prompt'])
        if digest(source['rate_authority'])!=source['rate_authority_sha256']:raise ValueError('Frozen physical rate changed')
        records=point['records'];muons=np.asarray([r['muons'] for r in records])
        actual=np.asarray([r['reco']['weight'] if r['reco'] and r['reco']['masses'] else 0. for r in records])
        variants={label:ideal_pair_pointing(muons,cut).astype(float)
                  for label,cut in [('ideal_annulus',0.),('ideal_annulus_E20',20.),('ideal_annulus_E30',30.)]}
        variants['actual_BothBoth']=actual
        coefficient=source['cross_section_pb']/source['epsilon']**2
        thresholds=reference_counting_thresholds(mass,300.);responses={}
        for label,values in variants.items():
            summary=weighted_response_summary(values);responses[label]=summary
            for method in ['exact_zero_observation_Poisson90','SensCalc_style_one_sided_Gaussian90_proxy']:
                required=thresholds[method]
                rows.append(dict(mass_gev=mass,response=label,method=method,luminosity_pb=300.,
                    sigma_over_epsilon2_pb=coefficient,efficiency=summary['efficiency'],efficiency_SE=summary['standard_error'],
                    effective_events=summary['effective_events'],signal_at_epsilon_0p001=300.*coefficient*1e-6*summary['efficiency'],
                    required_signal_events=required,epsilon_fixed_prompt_response=fixed_response_coupling(
                        required,300.,coefficient,summary['efficiency'])))
        points.append(dict(mass_gev=mass,physical_selected_rate=source,
            matched_original_events=len(records),matched_processed_events=point['matched_processed_events'],
            missing_selected_responses=point['missing_selected_responses'],responses=responses,thresholds=thresholds,
            weighted_reco_inside_ideal=float(actual[variants['ideal_annulus']>0].sum()),
            weighted_reco_outside_ideal=float(actual[variants['ideal_annulus']==0].sum())))
    table=output/'matched_acceptance_background.csv'
    with table.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(schema='shift-matched-acceptance-background-benchmark-v1',complete=True,luminosity_pb=300.,
        original_prefix_events=cache['points'][0]['matched_original_events'],event_cache=receipt,
        snapshot=dict(path=str(study/'snapshot.json'),sha256=digest(study/'snapshot.json')),points=points,table=rows,
        definitions=dict(ideal='Both direct GEN daughter muons: negative pz, straight rays from reset source to160m plane,1<r<7.5m; no field/material/transport/reconstruction.',
            actual='Ordinary unmodified Both-Both topologyMin=topologyMax=2; no added charge, vertex-quality or energy selection; native W/p once.',
            denominator='The exact same first1000 original pre-admission GEN identities; all planned selected IDs have a completed Nano response.',
            rates='Frozen snapshot physical selected sigma/epsilon², including dimuon BR once, identical for all response/background variants.',
            background='Explicit diagnostic linear exposure extension B(m,L)=(L/750pb^-1)*10^[6.1exp(-.06m)+2.8m^(-.2)]; cached source hardcodes3000/750 and does not establish run luminosity.',
            statistics='Zero observed known background: exact Poisson90% ln10. SensCalc-style proxy:1.28sqrtB (one-sided Gaussian90%), not90% CLs; not equivalent to current profiled30% nuisance limits.',
            uncertainty='Event-level MC standard errors for the mean of W/p contributions; binomial ideal controls. No claim of independent validation beyond these fixed prefixes.'),
        limitations=['This is a simplified geometric benchmark, not full SensCalc reproduction: no magnetic bending, decay-angle reweighting, nuclear material or acceptance-table integration.',
            'GEN decay positions are reset to the source only for the ideal prompt geometry diagnostic. Real reconstruction remains unchanged.',
            'A reconstructed event may lie outside the simplified annulus; the two selections are not nested detector-efficiency stages.',
            'All coupling values assume fixed prompt response as epsilon changes; no lifetime-dependent exclusion contour is inferred.',
            'Physical rates remain provisional; trigger and scale-factor assumptions are unchanged.'],
        statistical_sources=['https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf',
                             'https://arxiv.org/abs/1007.1727'],
        table_file=dict(path=str(table),sha256=digest(table)))
    save_json(output/'matched_acceptance_background.json',result)
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(12.5,5.8));labels=['ideal_annulus','ideal_annulus_E20','ideal_annulus_E30','actual_BothBoth']
    names=['Ideal annulus','Ideal + E>20GeV','Ideal + E>30GeV','Actual Both-Both']
    for index,point in enumerate(points):
        x=np.arange(4)+(index-.5)*.2;y=[point['responses'][label]['efficiency'] for label in labels]
        error=[point['responses'][label]['standard_error'] for label in labels]
        axes[0].errorbar(x,y,yerr=error,fmt='o',capsize=3,label=f"{point['mass_gev']:g}GeV")
    axes[0].set_xticks(range(4),names,rotation=15);axes[0].set(ylabel='Per original GEN event',title='Paired1000-event prefixes: response stage');axes[0].legend()
    for mass,color in [(2.,'tab:blue'),(5.,'tab:orange')]:
        for method,style in [('exact_zero_observation_Poisson90','--'),('SensCalc_style_one_sided_Gaussian90_proxy','-')]:
            data=[next(r for r in rows if r['mass_gev']==mass and r['method']==method and r['response']==label) for label in labels]
            axes[1].plot(range(4),[r['epsilon_fixed_prompt_response'] for r in data],ls=style,marker='o',color=color,
                label=f"{mass:g}GeV: "+('Poisson0B90%' if style=='--' else 'SC-style smooth-B proxy'))
    axes[1].set_xticks(range(4),names,rotation=15);axes[1].set(yscale='log',ylabel='Mixing epsilon',title='Same0.3fb^-1, rate and fixed prompt response');axes[1].legend(fontsize=7)
    fig.suptitle('Separate geometry, reconstruction and background/statistical assumptions')
    fig.text(.03,.02,'Ideal rays ignore field/material; E thresholds are explicit diagnostic variants. Actual Both-Both includes all frozen simulation/reconstruction losses.\nThese fixed-response count comparisons are not equivalent CLs prescriptions or validated SensCalc contours. MC errors are shown only in the response panel.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.94));path=output/'matched_acceptance_background.pdf';fig.savefig(path);plt.close(fig)
    print('Saved matched benchmark:',path,flush=True)


def source_grid_rows(data):
    """Require inclusive channel closure and dimuon BR exactly once."""
    rows=data['masses']
    if not rows or any(a['mass_gev']>=b['mass_gev'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Source-grid masses must be nonempty and strictly increasing')
    for row in rows:
        channels=row['meson_fragmentation']['channels']
        mesons=math.fsum(value['sigma_over_epsilon_squared_pb'] for value in channels.values())
        inclusive=row['inclusive_sigma_over_epsilon_squared_pb'];branch=row['BR_mumu']
        values=[row['mass_gev'],inclusive,row['FR_sigma_over_epsilon_squared_pb'],mesons,
                row['dimuon_sigma_over_epsilon_squared_pb'],branch]
        if not all(math.isfinite(value) and value>=0 for value in values) or not 0<branch<=1:
            raise ValueError('Nonphysical source-grid rate or branching fraction')
        if not math.isclose(mesons+row['FR_sigma_over_epsilon_squared_pb'],inclusive,rel_tol=1.e-10):
            raise ValueError('Source-grid inclusive production channels do not close')
        if not math.isclose(inclusive*branch,row['dimuon_sigma_over_epsilon_squared_pb'],rel_tol=1.e-12):
            raise ValueError('Source-grid dimuon branching must be applied exactly once')
    return rows


def plot_source_grid(source):
    """Show source physics and reference LAB anchors, without inventing RECO."""
    import csv
    from shift_limits_lifetime import plotting
    source=source.resolve();data=json.loads(source.read_text());rows=source_grid_rows(data)
    output=source.parent;mass=np.array([row['mass_gev'] for row in rows])
    plt=plotting();fig,axes=plt.subplots(2,2,figsize=(13,10))
    axis=axes[0,0]
    groups=[('eta-decay','Eta','tab:orange'),('eta-prime-decay','Eta prime','tab:green'),
            ('fragmentation','rho/omega/phi fragmentation','tab:purple'),
            ('FR','Primary-proton FR','tab:red')]
    for key,label,color in groups:
        values=[]
        for row in rows:
            channels=row['meson_fragmentation']['channels']
            if key=='FR':value=row['FR_sigma_over_epsilon_squared_pb']
            elif key=='fragmentation':value=sum(v['sigma_over_epsilon_squared_pb'] for k,v in channels.items() if k.endswith('fragmentation'))
            else:value=channels.get(key,{}).get('sigma_over_epsilon_squared_pb',0.)
            values.append(value*row['BR_mumu'])
        axis.plot(mass,np.where(np.array(values)>0,values,np.nan),'.-',label=label,color=color,lw=1)
    axis.errorbar(mass,[r['dimuon_sigma_over_epsilon_squared_pb'] for r in rows],
                  yerr=[r['dimuon_sigma_statistical_error_over_epsilon_squared_pb'] for r in rows],
                  fmt='ko',ms=3,capsize=2,label='Total; source MC error only')
    axis.set(yscale='log',xlim=(.205,2),xlabel='Dark-photon mass [GeV]',
             ylabel='sigma times BR / epsilon squared [pb]',title='Threshold, meson endpoints and vector peaks are present')
    for x in [.547862,.95778]:axis.axvline(x,color='.7',ls=':',lw=.8)
    axis.legend(fontsize=7)
    axis=axes[0,1];axis.plot(mass,[r['BR_mumu'] for r in rows],'o-',ms=3)
    axis.axvline(.21132,color='.5',ls=':',label='Pinned dimuon threshold')
    for x,label in [(.25,'Current 0.25'),(.7,'Current 0.7')]:axis.axvline(x,color='.7',ls='--',label=label)
    axis.set(xlim=(.205,2),ylim=(0,.5),xlabel='Dark-photon mass [GeV]',ylabel='BR to dimuons',
             title='A decay-model feature is not a measured limit feature');axis.legend(fontsize=7)
    axis=axes[1,0]
    for lab,label in [('0.01','Mean LAB 1 cm'),('1.0','Mean LAB 1 m'),('5.0','Mean LAB 5 m'),('1000.0','Mean LAB 1 km')]:
        values=np.array([r['epsilon_for_reference_mean_lab_m'][lab] for r in rows])
        errors=values*np.array([r['epsilon_reference_statistical_relative_error'] for r in rows])
        axis.errorbar(mass,values,yerr=errors,fmt='.-',ms=3,capsize=2,label=label)
    axis.set(yscale='log',xlim=(.205,2),xlabel='Dark-photon mass [GeV]',ylabel='Physical epsilon',
             title='Reference LAB distance uses the measured source boost');axis.legend(fontsize=7)
    axis=axes[1,1];axis.plot(mass,[r['expected_unweighted_GEN_per_50000_source_collisions'] for r in rows],'o-',ms=3)
    axis.set(yscale='log',xlim=(.205,2),xlabel='Dark-photon mass [GeV]',ylabel='Expected unweighted GEN / 50k source collisions',
             title='Source yield helps size pilots; RECO efficiency is still unknown')
    candidates=data['next_detector_mass_candidates_gev']
    axis.text(.03,.97,'Suggested new response masses [GeV]:\n'+', '.join(f'{value:g}' for value in candidates),
              transform=axis.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.suptitle('Bounded low-mass source pilot: 15 masses, 20,000 collisions each')
    fig.text(.025,.02,'Pinned SensCalc-compatible mesons/fragmentation + central FR; DY is excluded. No detector efficiency. Lines guide the eye between source points.\nErrors show source-collision statistics only; production-model, form-factor and width uncertainties are not included. Existing 0.25/0.7 GeV detector work continues.',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,.95));path=output/'low_mass_source_grid.pdf';fig.savefig(path);plt.close(fig)
    fields=['mass_gev','inclusive_sigma_over_epsilon_squared_pb','BR_mumu','dimuon_sigma_over_epsilon_squared_pb',
            'dimuon_sigma_statistical_error_over_epsilon_squared_pb','combined_mean_beta_gamma',
            'combined_mean_beta_gamma_statistical_error','expected_unweighted_GEN_per_50000_source_collisions']
    with (output/'low_mass_source_grid.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:row[key] for key in fields} for row in rows)
    save_json(output/'low_mass_source_grid_receipt.json',dict(input_path=str(source),input_sha256=digest(source),
        figure=dict(path=str(path),sha256=digest(path)),no_detector_production=True))
    print('Saved source-grid diagnostic:',path,flush=True)


if __name__=='__main__':main()
