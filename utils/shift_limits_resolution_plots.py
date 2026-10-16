#!/usr/bin/env python3
# Code style: AI-code. AI-generated publication of audited fine-bin sensitivity.
"""Publish native medians and explicitly marked exact-profile numerical fallbacks."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shift_paths import limits_resolution_output_path,limits_mass_shape_output_path,limits_output_path
from shift_limits import digest
from shift_limits_lifetime import plotting,finish,save_json
from shift_limits_shapes import clipped_boundary_segments,boundary_outline_segments


def load_projection(output,original):
    records=[]
    for power in [0,2]:
        path=output/f'completed/p{power}/limits.json'
        if not path.exists():path=output/f'p{power}/results/limits.json'
        if not path.exists():continue
        saved=json.loads(path.read_text())
        for native in saved['records']:
            median=None;source='unresolved'
            if native['status']=='ok':median=native['expected_limit_cross_section_pb'];source=native.get('limit_source','native_tea_combine')
            elif native['status']=='numerical_validation_failed':
                median=native.get('validation',{}).get('independent_quantiles_pb',{}).get('median')
                if median is not None:source='exact_asimov_profile_median_fallback'
            sample=original['signals'][native['point']]
            epsilon=sample['epsilon']*math.sqrt(median/sample['reference_cross_section_pb']) if median is not None else None
            boost=5000/original['signals'][f"m{sample['mass_gev']:g}_lab_5m"]['proper_ctau_mm']
            flight=boost*sample['proper_ctau_mm']*(sample['epsilon']/epsilon)**2 if epsilon else None
            records.append(dict(power=power,point=native['point'],mass_gev=native['mass_gev'],channel=native['channel'],
                confidence_level=native['confidence_level'],median_cross_section_pb=median,epsilon=epsilon,
                source=source,native_status=native.get('native_status',native['status']),native_validation=native.get('validation'),
                mean_lab_flight_at_boundary_mm=flight,prompt_consistent=flight is not None and flight<.01,
                physical_reference_epsilon=sample['epsilon'],reference_cross_section_pb=sample['reference_cross_section_pb'],
                native_output=native.get('combine_output'),native_output_sha256=native.get('combine_output_sha256')))
    return records


def draw_projection(output,original,records):
    plt=plotting();figure,axes=plt.subplots(1,2,figsize=(12,6))
    masses=sorted({s['mass_gev'] for s in original['signals'].values()})
    contours={}
    for level,axis in zip([.95,.90],axes):
        for power,color in [(0,'#a33200'),(2,'#3975a8')]:
            rows={r['mass_gev']:r for r in records if r['power']==power and r['channel']=='combined' and r['confidence_level']==level}
            curve=[(mass,rows[mass]['epsilon']) if mass in rows and rows[mass]['epsilon'] else None for mass in masses]
            segments=clipped_boundary_segments(curve,1.)
            for i,segment in enumerate(boundary_outline_segments(segments,2.,1.)):
                axis.plot(*np.array(segment).T,color=color,label=f'Background power {power}' if i==0 else None)
            for row in rows.values():
                if row['epsilon'] is None or row['epsilon']>1:continue
                fallback=row['source']!='native_tea_combine'
                axis.scatter(row['mass_gev'],row['epsilon'],marker='D' if fallback else 'o',
                    facecolors='none' if fallback else color,edgecolors=color,s=33,zorder=5)
        envelope=[]
        for mass in masses:
            values=[r for r in records if r['channel']=='combined' and r['confidence_level']==level and r['mass_gev']==mass and r['epsilon'] is not None]
            envelope.append((mass,max(r['epsilon'] for r in values)) if len({r['power'] for r in values})==2 else None)
        segments=clipped_boundary_segments(envelope,1.)
        polygons=[segment+[[segment[-1][0],1.],[segment[0][0],1.]] for segment in segments]
        for i,polygon in enumerate(polygons):axis.fill(*np.array(polygon).T,color='#8b7070',alpha=.18,label='Excluded in both tested models' if i==0 else None)
        key='cl'+str(round(100*level))
        contours[key]=dict(confidence_level=level,segments=segments,
            outline_segments=boundary_outline_segments(segments,2.,1.),excluded_polygons=polygons,
            interpretation='Worst of two background interpolation models; same exact two-nuisance Asimov likelihood. Diamonds are reference-profile median fallbacks after failed native numerical checks.',
            lower_lifetime_region_scope='See lifetime_envelope.json; m2 bound is conditional on mixtures of saved MC responses.')
        save_json(output/(key+'_envelope.json'),contours[key])
        axis.axvspan(.2114,2,color='gray',alpha=.14,label='Below 2 GeV: no detector samples')
        axis.axhline(.1,color='gray',linestyle=':',linewidth=.8)
        coverage=json.loads((Path(limits_mass_shape_output_path).resolve()/'model_coverage.json').read_text())
        domain_mass,domain_epsilon=np.array(coverage['maximum_coupling_by_mass']).T
        axis.fill_between(domain_mass,domain_epsilon,1.,where=domain_epsilon<1.,facecolor='none',
            edgecolor='gray',hatch='///',label='Outside full-mixing domain',zorder=7)
        axis.plot([],[],marker='D',markerfacecolor='none',color='black',linestyle='',label='Reference-profile fallback')
        axis.set(xscale='log',yscale='log',xlim=(.2114,90),ylim=(1.e-6,1.),xlabel='Dark-photon mass [GeV]',ylabel='Mixing epsilon',title=f'{round(100*level)}% CLs, 3 fb$^{{-1}}$')
        axis.legend(fontsize=7,loc='lower right')
    figure.suptitle('Fine-bin expected sensitivity with explicit background-model dependence')
    finish(figure,output,'complete_mass_epsilon_limits',
        'Actual reconstructed signal response and full GEN denominator retained; 30% signal/background normalization uncertainties.\n'
        'Background interpolation and finite MC shape support remain provisional. Above epsilon=0.1 is formal rate/width extrapolation, not a validated full-mixing model.')
    figure,axis=plt.subplots(figsize=(9,6))
    for channel in ['combined']+original['categories']:
        rows={r['mass_gev']:r for r in records if r['power']==0 and r['channel']==channel and r['confidence_level']==.95}
        values=[rows[m]['epsilon'] if m in rows and rows[m]['epsilon'] else np.nan for m in masses]
        axis.plot(masses,values,'o-',label=channel,linewidth=2.5 if channel=='combined' else 1.2,markersize=3)
    axis.set(xscale='log',yscale='log',xlim=(2,90),ylim=(1.e-3,1.),xlabel='Dark-photon mass [GeV]',ylabel='Mixing epsilon',title='Category comparison, 95% CLs; background power 0')
    axis.legend(fontsize=8)
    finish(figure,output,'complete_category_comparison',
        'All numerical outcomes are retained. Reference medians replace failed native medians with explicit provenance in projection.json.\n'
        'Gaps with zero reconstructed signal MC remain unresolved; no physical zero efficiency is inferred.')
    return contours


def draw_statistics(output):
    plt=plotting();figure,axes=plt.subplots(1,2,figsize=(12,5.7))
    report=json.loads((output/'p0/yields.json').read_text())
    names=sorted(report['signals'],key=lambda n:report['signals'][n]['mass_gev'])
    masses=[report['signals'][n]['mass_gev'] for n in names]
    summary={}
    for category in report['categories']:
        values=[]
        for name in names:
            row=report['signals'][name];s=sum(row['binned_yields'][category]);v=sum(row['binned_variances'][category])
            values.append(s*s/v if v else np.nan)
        axes[0].plot(masses,values,'o-',label=category,markersize=3)
        summary[category]=[None if not math.isfinite(v) else v for v in values]
    axes[0].axhline(10,color='gray',linestyle=':')
    axes[0].set(xscale='log',yscale='log',xlabel='Dark-photon mass [GeV]',ylabel='Selected signal MC effective count',title='Fine bins do not create MC information')
    axes[0].legend(fontsize=7)
    proof_path=output/'lifetime_envelope.json'
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        for power,color in [(0,'#a33200'),(2,'#3975a8')]:
            rows=sorted([r for r in proof['universal'] if r['power']==power],key=lambda r:r['mass_gev'])
            axes[1].plot([r['mass_gev'] for r in rows],[max(r['optimistic_q_max'],1.e-20) for r in rows],'o-',color=color,label=f'Acceptance-one bound, power {power}')
            extra=next(r for r in proof['m2_response_envelope'] if r['power']==power)
            axes[1].scatter(2,extra['q_optimistic_response_envelope'],marker='D',color=color,s=45)
        axes[1].axhline(proof['q90_threshold'],color='black',linestyle='--',label='90% threshold')
    axes[1].set(xscale='log',yscale='log',xlabel='Dark-photon mass [GeV]',ylabel='Optimistic q outside prompt region',title='Longer-lived exclusion check')
    axes[1].legend(fontsize=7)
    finish(figure,output,'mc_statistics_and_lifetime_bound',
        'For masses >=5 GeV, even acceptance one and the most favourable mass bin cannot exclude the nonprompt region.\n'
        'At 2 GeV diamonds bound mixtures of saved response templates; this does not prove coverage of unseen finite-MC shapes or replace low-mass simulation.')
    save_json(output/'signal_mc_statistics.json',dict(masses_gev=masses,selected_effective_mc=summary))


def paper_overlay(output,original,records):
    from shift_limits_comparison import make_comparison
    source=Path(limits_output_path).resolve()
    points=[]
    for mass in sorted({r['mass_gev'] for r in records}):
        selected=[r for r in records if r['channel']=='combined' and r['confidence_level']==.90 and r['mass_gev']==mass and r['epsilon']]
        if len(selected)!=2:continue
        row=max(selected,key=lambda r:r['epsilon']);sample=original['signals'][row['point']]
        points.append(dict(channel='combined',confidence_level=.90,mass_gev=mass,epsilon=sample['epsilon'],
            expected_limit=row['median_cross_section_pb']/sample['reference_cross_section_pb'],source=row['source']))
    if not points:return
    path=output/'paper_points90.json';save_json(path,dict(records=points))
    make_comparison(SimpleNamespace(input=path,source_figure=source/'arxiv_2511.03883v1_fig8_15.pdf',
        source_provenance=source/'paper_figure_provenance.json',output=output/'paper_comparison90.pdf',
        contour=output/'cl90_envelope.json',channel='combined',lumi_fb=3.,systematic_percent=30.))


def main():
    output=Path(limits_resolution_output_path).resolve()
    original=json.loads((Path(limits_output_path).resolve()/'yields.json').read_text())
    records=load_projection(output,original)
    save_json(output/'projection.json',dict(records=records,
        source_policy='Validated native TEA/Combine medians plus explicitly marked exact Asimov profile completion for failed or missing native outcomes. Raw results are unchanged.',
        native_fallbacks=[{k:r[k] for k in ['power','point','channel','confidence_level','native_status']} for r in records if r['source'] not in ['native_tea_combine','unresolved']]))
    draw_projection(output,original,records)
    draw_statistics(output)
    paper_overlay(output,original,records)
    print('Published',len(records),'audited outcomes, including',sum(r['source'] not in ['native_tea_combine','unresolved'] for r in records),'explicit reference medians.')


if __name__=='__main__':main()
