#!/usr/bin/env python3
# Code style: AI-code. AI-generated plots of current lifetime-dependent limits.
"""Publish the native TEA crossings and the literal histogram background model."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shift_limits import digest
from shift_limits_current import Model
from shift_limits_lifetime import plotting,save_json

MODE_COLORS={'background':'#b33636','zero_background':'#176b9c'}
MODE_LABELS={'background':'SHIFT: QCD + DY + J/psi','zero_background':'SHIFT: zero background'}


def draw_true_muonium(axis):
    """Vector-state equivalent coupling, not a dimuon signal hypothesis.

    Phys. Rev. D 100, 053003, Eqs. (2)-(3): m = 2 m_mu - BE,
    epsilon_eff = alpha**2/2. The bound state decays predominantly to ee.
    https://doi.org/10.1103/PhysRevD.100.053003
    """
    point = (0.211315, 2.66e-5)
    axis.scatter(*point, marker='*', s=95, facecolor='#efbd32', edgecolor='black',
                 linewidth=.65, zorder=30, label=r'True muonium ($e^+e^-$ reference)')
    axis.annotate('TM', point, xytext=(7, -10), textcoords='offset points', fontsize=8)
    return dict(mass_gev=point[0], equivalent_epsilon=point[1],
                source='https://doi.org/10.1103/PhysRevD.100.053003',
                role='Vector ee reference only; outside the dimuon signal grid')


def comparison_figure_directory(config):
    """Let frozen analysis workers pin the figure without importing live configs."""
    explicit=getattr(config,'comparison_figure_directory',None)
    if explicit is not None:return Path(explicit).resolve()
    from shift_paths import limits_output_path
    return Path(limits_output_path).resolve()


def finish(fig,path,caption,plt):
    fig.text(.025,.018,caption,fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.105,1,.94));fig.savefig(path);plt.close(fig)


def nominal_scan(model):
    saved=json.loads((model.output/'scan.json').read_text())
    if getattr(model,'mc_statistics',False) or saved.get('include_mc_stat_uncertainties',False):
        raise ValueError('Finite-MC scans need the dedicated mc_statistics comparison; full native plotting is not qualified')
    return saved['records']


def boundaries(model,level,channel,omitted_masses=()):
    scan=nominal_scan(model)
    native=json.loads((model.output/'native_results.json').read_text())['records'] if (model.output/'native_results.json').exists() else []
    result=[]
    for row in scan:
        if row['confidence_level']!=level or row['channel']!=channel:continue
        if row['mass_gev'] in omitted_masses:continue
        row=copy.deepcopy(row);roots=list(row['roots']);sources=[]
        for i,value in enumerate(roots):
            found=next((r for r in native if r['mass_gev']==row['mass_gev'] and r['mode']==row['mode'] and r['channel']==channel and r['confidence_level']==level and r['edge']==i),None)
            if found and found['validated']:roots[i]=found['epsilon'];sources.append('native_tea_AsymptoticLimits')
            else:sources.append('independent_profile_not_native_checked')
        mapping=dict(zip(row['roots'],roots))
        row['excluded_intervals']=[[mapping.get(a,a),mapping.get(b,b)] for a,b in row['excluded_intervals']]
        row['roots']=roots;row['boundary_sources']=sources;result.append(row)
    supported=supported_contour_rows(result,model.snapshot['signals'])
    if getattr(model.config,'include_prompt_reference_extrapolation',False):
        keys={(r['mass_gev'],r['mode']) for r in supported}
        supported.extend(row for row in result if (row['mass_gev'],row['mode']) not in keys
                         and row['upper_prompt_continuation'] and row['roots'] and row['excluded_intervals'])
    return supported


def single_lifetime(row):
    low,high=row['epsilon_support']
    return math.isclose(low,high,rel_tol=1.e-12)


def supported_contour_rows(rows, signals):
    """Keep lower boundaries bracketed by generated, positive-response anchors."""
    from scipy.stats import norm
    result=[]
    for row in rows:
        if single_lifetime(row) or not row['excluded_intervals']:continue
        usable=[]
        for signal in signals:
            if signal['mass_gev']!=row['mass_gev']:continue
            counts=signal.get('counts')
            if counts is not None:
                selected=(sum(c['sumw'] for c in counts.values()) if row['channel']=='combined'
                          else counts[row['channel']]['sumw'])
                if selected<=0:continue
            usable.append(signal['epsilon'])
        epsilons=sorted(set(usable))
        anchors=[]
        for epsilon in epsilons:
            values=[q for e,q in zip(row['epsilon_grid'],row['q_grid'])
                    if math.isclose(e,epsilon,rel_tol=1.e-12)]
            if values:anchors.append((epsilon,values[0]))
        threshold=norm.ppf((1+row['confidence_level'])/2)**2
        brackets=[(a,b) for (a,qa),(b,qb) in zip(anchors,anchors[1:])
                  if (qa-threshold)*(qb-threshold)<0]
        intervals=[(low,high) for low,high in row['excluded_intervals']
                   if any(math.isclose(low,root,rel_tol=1.e-7) for root in row['roots'])
                   and any(a<=low<=b for a,b in brackets)]
        if intervals:
            current=copy.deepcopy(row);current['excluded_intervals']=intervals;result.append(current)
    return result


def draw_regions(axis,rows,mode,color,label,fill=True,linewidth=2,linestyle=None,point_marker_size=28):
    # Presentation rule: omit unsupported masses, then connect supported lower
    # boundaries. No diamonds, support-edge closures or vertical island lines.
    nodes=sorted((r for r in rows if r['mode']==mode and not single_lifetime(r)
                  and r['excluded_intervals']),key=lambda r:r['mass_gev'])
    nodes=[r for r in nodes if any(math.isclose(r['excluded_intervals'][-1][0],root,rel_tol=1.e-7)
                                  for root in r['roots'])]
    if not nodes:return
    masses=[r['mass_gev'] for r in nodes]
    low=[r['excluded_intervals'][-1][0] for r in nodes]
    high=[r['excluded_intervals'][-1][1] for r in nodes]
    style=linestyle or ('-' if mode=='background' else '--')
    axis.plot(masses,low,color=color,ls=style,lw=linewidth,label=label)
    for left,right in zip(range(len(nodes)-1),range(1,len(nodes))):
        if all(any(math.isclose(high[i],root,rel_tol=1.e-7) for root in nodes[i]['roots'])
               for i in (left,right)):
            axis.plot(masses[left:right+1],high[left:right+1],color=color,ls=style,lw=linewidth)
    if fill and len(nodes)>1:axis.fill_between(masses,low,high,color=color,alpha=.13)


def select_projections(config,output):
    inventory=json.loads((output/'projection_inventory.json').read_text());selected=[]
    # Exact RGB values are checked against the pinned vector figure's legend.
    specs={
        'SHiP':((65/255,105/255,225/255),'SHiP, 15 years'),
        'BelleII':((0.,206/255,209/255),r'Belle II, 50 ab$^{-1}$'),
        'LHCb300':((34/255,139/255,34/255),r'LHCb, 300 fb$^{-1}$'),
        'HL-LHC':((0.,0.,0.),r'HL-LHC, 6 ab$^{-1}$'),
        'FCC-hh':((0.,0.,139/255),r'FCC-hh, 30 ab$^{-1}$'),
        'FCC-ee':((1.,20/255,147/255),'FCC-ee, combined'),
        'MuC10':((148/255,0.,211/255),r'Muon collider, 10 TeV (incl./VBF)')}
    for key in config.projection_experiments:
        color,label=specs[key]
        candidates=[c for c in inventory['curves'] if max(abs(a-b) for a,b in zip(c['color'],color))<.012]
        if key=='LHCb300':candidates=[c for c in candidates if c['dashes']=='[] 0']
        if not candidates:raise ValueError('Projection color needs verified mapping: '+key)
        for curve in candidates:selected.append(dict(curve,experiment=key,label=label))
    save_json(output/'selected_projections.json',{'source':inventory['source'],'curves':selected,'removed':['DarkQuest','FASER','NA62','NA64','FPF','old SHIFT SensCalc','LCF variants','weaker collider variants']})
    return selected


def draw_comparisons(config,model):
    plt=plotting();output=model.output;selected=select_projections(config,output)
    for level in config.confidence_levels:
        rows=boundaries(model,level,config.primary_channel,config.plot_omitted_masses_gev);save_json(output/f'contours_cl{round(100*level)}.json',{'records':rows,'channel':config.primary_channel,'omitted_masses_gev':list(config.plot_omitted_masses_gev),'interpolation':'Lifetime response in log epsilon; plot boundaries connect all available mass points across coverage gaps as requested.'})
        for external in [False,True]:
            fig,ax=plt.subplots(figsize=(11,7.3));fig.suptitle(f'SHIFT Both-Both: expected {round(100*level)}% CLs, {config.luminosity/1000:g} fb$^{{-1}}$')
            if external:
                labels=set()
                for curve in selected:
                    points=np.asarray(curve['points']);ax.plot(points[:,0],points[:,1],color=curve['color'],lw=1.35,alpha=.85,label=curve['label'] if curve['experiment'] not in labels else None)
                    labels.add(curve['experiment'])
            for mode in config.background_modes:draw_regions(ax,rows,mode,MODE_COLORS[mode],MODE_LABELS[mode])
            ax.axhspan(.1,1,facecolor='gray',alpha=.07,hatch='///',label=r'$\epsilon>0.1$: model extrapolation')
            ax.set(xscale='log',yscale='log',xlim=(.01,1000) if external else (.2,100),ylim=(1e-9,1),xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Mixing $\epsilon$')
            ax.legend(fontsize=8.2,loc='lower right',ncol=2 if external else 1)
            note='Existing mass templates; no classifier or added kinematic selection. Trigger/SFs = 1; independent 30% signal/background normalization uncertainties.\n'
            checked=sum(source=='native_tea_AsymptoticLimits' for row in rows for source in row['boundary_sources']);total=sum(len(row['roots']) for row in rows)
            note+=f'Published QCD + DY + J/psi only; empty MC bins stay empty. Asimov profile; {checked}/{total} boundaries checked with native TEA/Combine.\n'
            note+=('External curves: briefing-book Fig. 8.15; mostly 90% CL, FCC curves 95% CL. Production/response remain provisional.' if external else 'Open upper endpoints mark missing lifetime coverage, not physical exclusion edges. Sparse MC and missing background tails remain provisional.')
            note+='\nOnly boundaries bracketed by generated positive-response anchors are drawn.'
            name=f'limits_current_cl{round(100*level)}'+('_comparison' if external else '')+'.pdf';finish(fig,output/name,note,plt)


def draw_original_comparisons(config,model):
    """Keep the source figure's vector axes, existing bounds, fonts and logo."""
    import fitz
    import xml.etree.ElementTree as ET
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    source_path=comparison_figure_directory(config)/'arxiv_2511.03883v1_fig8_15.pdf'
    provenance=json.loads(source_path.with_name('paper_figure_provenance.json').read_text())
    assert digest(source_path)==provenance['figure_sha256']
    source=fitz.open(source_path);original=source[0]
    tree=ET.fromstring(original.get_svg_image(text_as_path=True))
    removed={'#808000','#dda0dd','#ff00ff','#ee82ee','#b22222','#ff0000','#2e8b57'}
    recolor={'#00ced1':'#d17a00','#228b22':'#8c564b','#00bfff':'#1b9e77'}
    removed_elements=0
    for parent in tree.iter():
        for child in list(parent):
            color=child.get('stroke') or child.get('fill')
            weaker_lhcb=child.get('stroke')=='#228b22' and child.get('stroke-dasharray')
            if color in removed or weaker_lhcb:
                parent.remove(child);removed_elements+=1;continue
            for attribute in ('stroke','fill'):
                if child.get(attribute) in recolor:child.set(attribute,recolor[child.get(attribute)])
    vector=fitz.open(stream=ET.tostring(tree),filetype='svg')
    clean=fitz.open(stream=vector.convert_to_pdf(),filetype='pdf');vector.close()
    width,height=original.rect.width,original.rect.height;offset=25
    drawings=original.get_drawings()
    omitted_masses=tuple(mass for mass in config.plot_omitted_masses_gev if mass in model.signals)
    pointwise=any(single_lifetime(row) for row in boundaries(model,config.confidence_levels[0],config.primary_channel,omitted_masses))
    footer_extra=(13 if omitted_masses else 0)+(13 if pointwise else 0)
    complete_notes=getattr(config,'comparison_notes_are_complete',False)
    if complete_notes:footer_extra=0
    # Copy retained legend rows directly from the original PDF, preserving fonts
    # and styles. Keep the original SHIFT curve as an explicitly named reference.
    legend_indices=[195,197,198,199,203,205,207,211,212]
    left,top,right,bottom=provenance['axis_rectangle_points_top_left_origin']
    def point(mass,epsilon):
        return fitz.Point(left+(math.log10(mass)+3)*(right-left)/6,
                          offset+top-math.log10(epsilon)*(bottom-top)/10)
    def luminosity_label(page,position,prefix,suffix,color,fontsize=12):
        x,y=position
        page.insert_text((x,y),prefix,fontsize=fontsize,fontname='tiro',color=color)
        x+=fitz.get_text_length(prefix,fontname='tiro',fontsize=fontsize)
        page.insert_text((x,y-.4*fontsize),'-1',fontsize=.65*fontsize,fontname='tiro',color=color)
        x+=fitz.get_text_length('-1',fontname='tiro',fontsize=.65*fontsize)+1
        page.insert_text((x,y),suffix,fontsize=fontsize,fontname='tiro',color=color)
    receipts=[]
    caption_height=33+13*len(config.comparison_notes) if complete_notes else max(235,33+13*len(getattr(config,'comparison_notes',())))
    mode_styles=getattr(config,'comparison_mode_styles',
        [('background',(0,90/255,50/255),': QCD + DY + J/psi',False),
         ('zero_background',(102/255,166/255,30/255),': zero background',True)])
    for level in config.confidence_levels:
        doc=fitz.open();page=doc.new_page(width=width,height=height+offset+caption_height+footer_extra)
        page.show_pdf_page(fitz.Rect(0,offset,width,height+offset),source,0)
        # SVG import does not preserve nested PDF clipping or masked images.
        # Restrict it to the axes, keeping original PDF text around the axes
        # and restoring the original logo with its transparency mask.
        axis_clip=fitz.Rect(left,top,right,bottom)
        axis_target=axis_clip+fitz.Rect(0,offset,0,offset)
        page.draw_rect(axis_target,color=None,fill=(1,1,1))
        page.show_pdf_page(axis_target,clean,0,clip=axis_clip)
        logo=fitz.Rect(489,305,right,bottom)
        page.show_pdf_page(logo+fitz.Rect(0,offset,0,offset),source,0,clip=logo)
        page.draw_rect(fitz.Rect(635,offset,width,bottom+offset),color=None,fill=(1,1,1))
        page.draw_rect(fitz.Rect(638.3,bottom+offset-1,width,height+offset),color=None,fill=(1,1,1))
        for index,drawing_index in enumerate(legend_indices):
            y=drawings[drawing_index]['rect'].y0
            clip=fitz.Rect(638,y-9,width,y+9)
            row_top=offset+top+index*20
            target=fitz.Rect(638,row_top,width,row_top+18)
            page.show_pdf_page(target,source,0,clip=clip)
            if drawing_index in (195,197):
                color=(209/255,122/255,0) if drawing_index==195 else (140/255,86/255,75/255)
                page.draw_rect(fitz.Rect(638.5,row_top+7,665,row_top+11),color=None,fill=(1,1,1))
                page.draw_line((640,row_top+9),(665,row_top+9),color=color,width=2)
        shift_legend_y=offset+top+len(legend_indices)*20+22
        original_color=(27/255,158/255,119/255)
        page.draw_line((640,shift_legend_y),(665,shift_legend_y),color=original_color,width=2)
        luminosity_label(page,(672,shift_legend_y+4),'SHIFT (briefing book), 30 fb','',original_color)
        rows=boundaries(model,level,config.primary_channel,omitted_masses);counts=[]
        for mode,color,label,dashed in mode_styles:
            if mode not in config.background_modes:continue
            strength = getattr(config, 'strength_report', None)
            if strength is not None:
                from shift_dp_progress_plots import strength_contour_segments
                segments = strength_contour_segments(strength, mode)
                for segment in segments:
                    shape = page.new_shape()
                    shape.draw_polyline([point(mass, epsilon) for mass, epsilon in segment])
                    shape.finish(color=color, width=2.5,
                                 dashes='[6 3] 0' if dashed else None, closePath=False)
                    shape.commit()
                y=shift_legend_y+(40 if mode=='zero_background' else 20)
                page.draw_line((640,y),(665,y),color=color,width=2.5,
                               dashes='[6 3] 0' if dashed else None)
                luminosity_label(page,(672,y+4),f"{getattr(config,'signal_model_label','SHIFT')}, {config.luminosity/1000:g} fb",label,color,fontsize=getattr(config,'comparison_legend_fontsize',12))
                counts.append(dict(mode=mode, contour_segments=len(segments),
                                   interpolation='Per-point log(r) triangulation; no template morphing'))
                continue
            tested=[]  # Unsupported single-epsilon tests are omitted from contours.
            nodes=sorted((r for r in rows if r['mode']==mode and r['excluded_intervals'] and not single_lifetime(r)),key=lambda r:r['mass_gev'])
            if not nodes:continue
            nodes=[dict(row,excluded_intervals=[row['excluded_intervals'][-1]]) for row in nodes]
            lower=[point(r['mass_gev'],min(a for a,b in r['excluded_intervals'])) for r in nodes]
            upper=[point(r['mass_gev'],max(b for a,b in r['excluded_intervals'])) for r in nodes]
            polygon=lower+upper[::-1]
            shape=page.new_shape();shape.draw_polyline(polygon);shape.finish(color=None,fill=color,fill_opacity=.10,closePath=True);shape.commit()
            # The plot-top caps are drawing closures, not measured upper limits.
            # Do not join the isolated finite upper edge to those artificial
            # caps: that produced the sharp V in the zero-background outline.
            shape=page.new_shape();shape.draw_polyline(lower)
            shape.finish(color=color,width=2.5,dashes='[6 3] 0' if dashed else None,closePath=False);shape.commit()
            finite=[]
            for row,xy in zip(nodes,upper):
                if any(math.isclose(max(b for a,b in row['excluded_intervals']),root,rel_tol=1e-7) for root in row['roots']):
                    finite.append(xy)
                elif finite:
                    if len(finite)>1:
                        shape=page.new_shape();shape.draw_polyline(finite);shape.finish(color=color,width=2.5,dashes='[6 3] 0' if dashed else None,closePath=False);shape.commit()
                    else:page.draw_circle(finite[0],2.2,color=color,fill=(1,1,1))
                    finite=[]
            if finite:
                if len(finite)>1:
                    shape=page.new_shape();shape.draw_polyline(finite);shape.finish(color=color,width=2.5,dashes='[6 3] 0' if dashed else None,closePath=False);shape.commit()
                else:page.draw_circle(finite[0],2.2,color=color,fill=(1,1,1))
            for xy in lower:page.draw_circle(xy,2.2,color=color,fill=color)
            y=shift_legend_y+(40 if mode=='zero_background' else 20)
            page.draw_line((640,y),(665,y),color=color,width=2.5,dashes='[6 3] 0' if dashed else None)
            luminosity_label(page,(672,y+4),f"{getattr(config,'signal_model_label','SHIFT')}, {config.luminosity/1000:g} fb",label,color,fontsize=getattr(config,'comparison_legend_fontsize',12))
            counts.append({'mode':mode,'connected_mass_points':[r['mass_gev'] for r in nodes],
                           'single_lifetime_tests':[dict(mass_gev=r['mass_gev'],epsilon=r['epsilon_support'][0],excluded=bool(r['excluded_intervals'])) for r in tested]})
        page.draw_rect(fitz.Rect(638.3,shift_legend_y+60,width,height+offset),color=None,fill=(1,1,1))
        luminosity_label(page,(84,17),f'SHIFT Both-Both, {config.luminosity/1000:g} fb',f': expected {round(100*level)}% CLs',(0,0,0),fontsize=13)
        notes=[
            f'Effective halo luminosity: {config.luminosity/1000:g} fb^-1 (assumed exposure; 3 fb^-1 is the optimistic ceiling).',
            'Trigger efficiency: 100%.',
            'MC is assumed to match data; all scale factors are 1.',
            'Both-Both dimuons only.',
            'No vertex classifier is applied.',
            'No additional signal-versus-background cuts are applied.',
            'Background: published QCD + DY + J/psi histograms only; empty bins stay empty and no missing tails are invented.',
            'Zero-background curve: background is set to zero as an idealized benchmark.',
            'Independent 30% signal and background normalization uncertainties.',
            'MC statistical uncertainties are omitted.',
            'Sub-GeV signals: SensCalc-compatible mesons + fragmentation + proton radiation; sparse MC and rates are provisional.',
            'Mass bins use the saved 0.1 GeV grid and a provisional 2% resolution scale.',
            'Mass boundaries are interpolated; lifetime coverage ends are not physical exclusion edges. Empty signal controls are omitted.',
            'Epsilon above 0.1 is a formal model extrapolation.',
            f'Current SHIFT: expected {round(100*level)}% CLs. Most source projections: 90% CL; FCC: 95% CL.']
        notes=list(getattr(config,'comparison_notes',notes))
        if omitted_masses and not complete_notes:notes.append(getattr(config,'comparison_omission_label','Temporarily omitted pending review')+': '+', '.join(f'{m:g} GeV' for m in omitted_masses)+'. Neighbouring points are connected.')
        if pointwise and not complete_notes:notes.append('Diamonds test one lifetime only: filled/open means excluded/not excluded. No lifetime interval is inferred.')
        for index,note in enumerate(notes):
            y=height+offset+13+index*13
            page.draw_circle((87,y-3),1.5,color=None,fill=(0,0,0))
            page.insert_text((96,y),note,fontsize=10,fontname='tiro')
        page.insert_text((84,height+offset+caption_height-24+footer_extra),'Source: arXiv:2511.03883v1, Fig. 8.15 (CC BY 4.0). Original SHIFT exposure: 30 fb^-1 (1% of 3 ab^-1).',fontsize=9.2,fontname='tiro')
        suffix='_'+config.background_modes[0] if len(config.background_modes)==1 else ''
        path=model.output/f'limits_current_cl{round(100*level)}{suffix}_comparison.pdf';temporary=path.with_suffix('.tmp.pdf')
        doc.save(temporary,garbage=4,deflate=True);doc.close();temporary.replace(path)
        receipts.append({'file':path.name,'boundaries':counts,'sha256':digest(path)})
    clean.close();source.close()
    suffix='_'+config.background_modes[0] if len(config.background_modes)==1 else ''
    save_json(model.output/f'original_overlay{suffix}_provenance.json',{'source_sha256':provenance['figure_sha256'],'removed_svg_elements':removed_elements,
        'removed_projection_colors':sorted(removed),'retained_legend_drawing_indices':legend_indices,'figures':receipts,
        'recolored_source_curves':recolor,'original_shift_luminosity_fb_inverse':30,
        'omitted_masses_gev':list(omitted_masses),'omission_reason':'Temporary user-requested presentation omission pending review; raw scan and fit results retained.',
        'upper_outline':'Finite upper edges are not joined to artificial plot-top caps; exclusion shading and numerical limits are unchanged.',
        'style':'Original vector axes, existing exclusion shading, curve styles, labels and logo retained; retained legend rows copied from source.'})



def draw_model_update(config, model, reference_directory):
    """Overlay only qualified response support on a named older-model reference.

    The reference PDF is rendered separately by draw_original_comparisons with
    muted styles. This function adds no response anchors or old-model lifetime
    templates. Mass-boundary interpolation follows the existing plotting rule.
    """
    import fitz
    reference_directory=Path(reference_directory)
    provenance=json.loads((comparison_figure_directory(config)/'paper_figure_provenance.json').read_text())
    left,top,right,bottom=provenance['axis_rectangle_points_top_left_origin']
    receipts=[]
    for level in config.confidence_levels:
        suffix='_'+config.background_modes[0] if len(config.background_modes)==1 else ''
        name=f'limits_current_cl{round(100*level)}{suffix}_comparison.pdf'
        base=reference_directory/name
        document=fitz.open(base);page=document[0]
        width,height=page.rect.width,page.rect.height
        plt=plotting();figure=plt.figure(figsize=(width/72,height/72))
        axis=figure.add_axes([left/width,1-(bottom+25)/height,(right-left)/width,(bottom-top)/height])
        axis.set(xscale='log',yscale='log',xlim=(1.e-3,1.e3),ylim=(1.e-10,1.))
        axis.set_axis_off()
        strength=json.loads((model.output/'signal_strength_map.json').read_text())
        from shift_dp_progress_plots import draw_strength_contours
        rows=[]
        background_update=getattr(config,'background_comparison_update',False)
        styles=[('background','#b43c88','Updated SM background' if background_update else 'New model: QCD + DY + J/psi'),
                ('zero_background','#166db0','Zero background (same signals)' if background_update else 'New model: zero background')]
        for mode,color,label in styles:
            if mode in config.background_modes:
                draw_strength_contours(axis,strength,mode,color,label,lw=2.1)
        draw_true_muonium(axis)
        import io
        buffer=io.BytesIO();figure.savefig(buffer,format='pdf',transparent=True);plt.close(figure)
        layer=fitz.open(stream=buffer.getvalue(),filetype='pdf')
        page.show_pdf_page(page.rect,layer,0,overlay=True);layer.close()
        # Existing reference legend ends above this deliberately reserved box.
        legend_y=337
        for mode,color,label in styles:
            if mode not in config.background_modes:continue
            rgb=tuple(int(color[i:i+2],16)/255 for i in (1,3,5))
            page.draw_line((640,legend_y),(665,legend_y),color=rgb,width=2.1,
                           dashes='[6 3] 0' if mode=='zero_background' else None)
            page.insert_text((672,legend_y+4),label,fontsize=10,fontname='tiro',color=rgb)
            legend_y+=19
        target=model.output/name;temporary=target.with_suffix('.tmp.pdf')
        document.save(temporary,garbage=4,deflate=True);document.close();temporary.replace(target)
        receipts.append(dict(file=name,sha256=digest(target),reference_pdf=str(base),
            reference_sha256=digest(base),new_model_scan_sha256=digest(model.output/'scan.json'),
            tested_support=[{k:row[k] for k in ('mass_gev','mode','epsilon_support','excluded_intervals')}for row in rows]))
    save_json(model.output/f'model_update{suffix}_provenance.json',dict(figures=receipts,
        new_model_support='Per-point log(r) triangulation inside the generated grid hull; no template morphing',
        reference='Older signal prescription, retained as an explicitly named comparison'))


def draw_inputs(config,model):
    plt=plotting();edges=model.snapshot['mass_edges_gev'];fig,ax=plt.subplots(figsize=(9,6));fig.suptitle('Background templates actually used in the limits')
    for ax,category in [(ax,config.primary_channel)]:
        for process,color in [('qcd','#666666'),('dy','#3073a5'),('jpsi','#a656a2')]:
            values=model.processes[process].sum(axis=0) if category=='combined' else model.processes[process][config.categories.index(category)]
            ax.stairs(np.where(values>0,values,np.nan),edges,label=process.upper(),color=color)
        ax.set(xscale='log',yscale='log',xlim=(.1,100),xlabel='Reconstructed dimuon mass [GeV]',ylabel='Expected events / likelihood bin',title=category);ax.legend()
    finish(fig,model.output/'background_mass_templates.pdf',f'{config.luminosity/1000:g} fb^-1. Rates use each published file\'s full source/GEN exposure and weights once; no additional filtering efficiency or scale factor.\nBin widths follow the saved 0.1 GeV grid and a provisional 2% scale. There is no MC-population-driven bin merging or background-tail interpolation.',plt)
    fig,ax=plt.subplots(figsize=(11,6.5));fig.suptitle('Published signal coverage used in this update')
    xx=[];yy=[];counts=[];mechanisms=set();unused=[]
    used={r['name'] for rows in model.signals.values() for r in rows}
    for row in model.snapshot['signals']:
        if row['name'] not in used:unused.append(row);continue
        stats=row['counts'][config.primary_channel]
        xx.append(row['mass_gev']);yy.append(row['epsilon']);counts.append(stats['sumw']**2/stats['sumw2'] if stats['sumw2'] else 0.);mechanisms.add(row['mechanism'])
    from matplotlib.colors import LogNorm
    obj=ax.scatter(xx,yy,c=np.maximum(counts,1),s=35,norm=LogNorm(),cmap='viridis',edgecolor='black',linewidth=.25)
    if unused:
        ax.scatter([r['mass_gev'] for r in unused],[r['epsilon'] for r in unused],marker='x',s=18,color='gray',label='Zero selected MC: response unknown');ax.legend(fontsize=8)
    fig.colorbar(obj,ax=ax,label='Effective Both-Both signal MC events')
    ax.set(xscale='log',yscale='log',xlim=(.2,100),ylim=(1e-9,1.5),xlabel='Mass [GeV]',ylabel='Generated physical epsilon')
    finish(fig,model.output/'signal_coverage.pdf',f'{len(xx)} response anchors from {len(model.snapshot["signals"])} frozen published files; full GEN denominators include zero-dimuon events.\nMeson and DY rates include BR(mu mu) once. Tiny legacy epsilon canaries do not override the larger prompt response. No new events were generated.',plt)


def draw_luminosity_scenarios(config,model):
    scenarios=getattr(config,'luminosity_scenarios',())
    if not scenarios:return
    plt=plotting()
    for level in config.confidence_levels:
        fig,axis=plt.subplots(figsize=(10.5,7))
        for (luminosity,directory),style in zip(scenarios,(':','-','--')):
            current=SimpleNamespace(**vars(config));current.luminosity=luminosity
            current.output_path=str(model.output/directory)
            if not (Path(current.output_path)/'scan.json').exists():continue
            rows=boundaries(Model(current),level,config.primary_channel,config.plot_omitted_masses_gev)
            for mode in config.background_modes:
                label=('QCD + DY + J/psi' if mode=='background' else 'Zero background')+f', {luminosity/1000:g} fb$^{{-1}}$'
                draw_regions(axis,rows,mode,MODE_COLORS[mode],label,fill=False,
                    linewidth=2.6 if luminosity==config.luminosity else 1.4,linestyle=style)
        axis.axhspan(.1,1,facecolor='gray',alpha=.07,hatch='///')
        axis.set(xscale='log',yscale='log',xlim=(.2,100),ylim=(1.e-7,1),
                 xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Mixing $\epsilon$',
                 title=f'SHIFT Both-Both: expected {round(100*level)}% CLs, luminosity scenarios')
        axis.legend(fontsize=8,loc='lower right',ncol=2)
        finish(fig,model.output/f'limits_luminosity_cl{round(100*level)}.pdf',
            'MC benchmark sensitivity: 0.3 fb$^{-1}$ baseline, 0.03 fb$^{-1}$ degraded, 3 fb$^{-1}$ optimistic ceiling.\n'
            'Open endpoints mark missing lifetime coverage; disconnected islands are not filled across gaps.\n'
            '30% normalization nuisances; finite-MC uncertainty omitted. Sub-GeV controls have very low effective statistics.',plt)


def brazil(config,model):
    from shift_limits_shapes import draw_native_brazil
    rows=json.loads((model.output/'native_results.json').read_text())['records'];groups={}
    for row in rows:
        if row['mass_gev'] in config.plot_omitted_masses_gev:continue
        if not row['validated'] or row.get('equivalent_counting_reuse') or row['channel']!=config.primary_channel or row['edge']!=0:continue
        if abs(row['quantiles']['median']/row['quantiles']['observed']-1)>.05:continue
        point='m'+format(row['mass_gev'],'g').replace('.','p')
        entry=dict(row,point=point,status='ok',validation={'bands_valid':False,'relative_errors':{}},
            expected_cross_section_quantiles_pb={k:v*row['combine_reference_cross_section_pb'] for k,v in row['quantiles'].items() if k!='observed'})
        groups.setdefault((config.primary_channel,row['confidence_level'],row['mode']),[]).append(entry)
    groups={key:rows for key,rows in groups.items() if len(rows)>1}
    current=SimpleNamespace(**vars(config));current.plots_output_path=str(model.output/'brazil');current.projection_method_label='Lifetime templates at boundary; bands provisional'
    draw_native_brazil(current,groups)


def qa(output):
    import fitz
    receipt={}
    for path in sorted(output.rglob('*.pdf')):
        outside=[]
        with fitz.open(path) as document:
            for number,page in enumerate(document):
                bounds=page.rect*page.derotation_matrix;bounds=fitz.Rect(0,0,bounds.width,bounds.height)+(-2,-2,2,2)
                for block in page.get_text('blocks'):
                    if block[6]==0 and not bounds.contains(fitz.Rect(block[:4])):outside.append({'page':number+1,'text':block[4][:80]})
            receipt[str(path.relative_to(output))]={'pages':len(document),'bytes':path.stat().st_size,'sha256':digest(path),'outside_page':outside}
            if path.name in ['limits_current_cl95_comparison.pdf','signal_coverage.pdf']:document[0].get_pixmap().save(output/(path.stem+'_preview.jpg'))
    save_json(output/'pdf_qa.json',receipt)
    print('PDFs checked:',len(receipt),'outside text blocks:',sum(len(r['outside_page']) for r in receipt.values()))


def main():
    import shift_limits_current_config as config
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-only',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--luminosity-pb',type=float)
    args=parser.parse_args()
    if args.output is not None:config.output_path=str(args.output.resolve())
    if args.luminosity_pb is not None:config.luminosity=args.luminosity_pb
    model=Model(config)
    nominal_scan(model)
    if not args.original_only:draw_comparisons(config,model);draw_inputs(config,model);draw_luminosity_scenarios(config,model);brazil(config,model)
    draw_original_comparisons(config,model);qa(model.output)

if __name__=='__main__':main()
