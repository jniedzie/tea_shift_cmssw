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
    args=parser.parse_args()
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


if __name__=='__main__':main()
