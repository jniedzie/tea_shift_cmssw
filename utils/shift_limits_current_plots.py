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


def finish(fig,path,caption,plt):
    fig.text(.025,.018,caption,fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.105,1,.94));fig.savefig(path);plt.close(fig)


def boundaries(model,level,channel,omitted_masses=()):
    scan=json.loads((model.output/'scan.json').read_text())['records']
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
    return result


def draw_regions(axis,rows,mode,color,label,fill=True,linewidth=2):
    nodes=sorted((r for r in rows if r['mode']==mode),key=lambda r:r['mass_gev']);labelled=False
    # Requested presentation: connect every available mass boundary, including
    # across gaps in generated response coverage. This is plot interpolation.
    run=[]
    for row in nodes:
        if not row['excluded_intervals']:continue
        row=copy.deepcopy(row)
        row['excluded_intervals']=[[min(a for a,b in row['excluded_intervals']),max(b for a,b in row['excluded_intervals'])]]
        run.append(row)
    runs=[run] if run else []
    for run in runs:
        masses=[r['mass_gev'] for r in run]
        for index in range(len(run[0]['excluded_intervals'])):
            low=[r['excluded_intervals'][index][0] for r in run];high=[r['excluded_intervals'][index][1] for r in run]
            style='-' if mode=='background' else '--'
            if len(run)==1:
                axis.plot([masses[0]]*2,[low[0],high[0]],color=color,ls=style,lw=linewidth)
                axis.scatter(masses,low,color=color,s=22,marker='o',label=label if not labelled else None);labelled=True
                continue
            axis.plot(masses,low,color=color,ls=style,lw=linewidth,label=label if not labelled else None);labelled=True
            axis.scatter(masses,low,color=color,s=12,zorder=6)
            if any(e<.999 for e in high):axis.plot(masses,high,color=color,ls=style,lw=linewidth)
            if fill:axis.fill_between(masses,low,high,color=color,alpha=.13)
            # Close only at the support edge. The first vertical segment starts
            # at the top rather than cutting diagonally through later masses.
            axis.plot([masses[0]]*2,[high[0],low[0]],color=color,lw=.9,ls=style,alpha=.7)
            axis.plot([masses[-1]]*2,[low[-1],high[-1]],color=color,lw=.9,ls=style,alpha=.7)


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
            note+=('External curves: briefing-book Fig. 8.15; mostly 90% CL, FCC curves 95% CL. Production/response remain provisional.' if external else 'Shading marks interpolated expected exclusion. All available mass boundaries are connected; sparse MC and missing background tails remain provisional.')
            name=f'limits_current_cl{round(100*level)}'+('_comparison' if external else '')+'.pdf';finish(fig,output/name,note,plt)


def draw_original_comparisons(config,model):
    """Keep the source figure's vector axes, existing bounds, fonts and logo."""
    import fitz
    import xml.etree.ElementTree as ET
    from shift_paths import limits_output_path
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    source_path=Path(limits_output_path).resolve()/'arxiv_2511.03883v1_fig8_15.pdf'
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
    omitted_masses=config.plot_omitted_masses_gev
    footer_extra=13 if omitted_masses else 0
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
    for level in config.confidence_levels:
        doc=fitz.open();page=doc.new_page(width=width,height=height+offset+235+footer_extra)
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
        for mode,color,label,dashed in [('background',(0,90/255,50/255),': QCD + DY + J/psi',False),
                                        ('zero_background',(102/255,166/255,30/255),': zero background',True)]:
            nodes=sorted((r for r in rows if r['mode']==mode and r['excluded_intervals']),key=lambda r:r['mass_gev'])
            if not nodes:continue
            lower=[point(r['mass_gev'],min(a for a,b in r['excluded_intervals'])) for r in nodes]
            upper=[point(r['mass_gev'],max(b for a,b in r['excluded_intervals'])) for r in nodes]
            polygon=lower+upper[::-1]
            shape=page.new_shape();shape.draw_polyline(polygon);shape.finish(color=None,fill=color,fill_opacity=.10,closePath=True);shape.commit()
            # The plot-top caps are drawing closures, not measured upper limits.
            # Do not join the isolated finite upper edge to those artificial
            # caps: that produced the sharp V in the zero-background outline.
            shape=page.new_shape();shape.draw_polyline([upper[0]]+lower+[upper[-1]])
            shape.finish(color=color,width=2.5,dashes='[6 3] 0' if dashed else None,closePath=False);shape.commit()
            finite=[]
            for row,xy in zip(nodes,upper):
                if max(b for a,b in row['excluded_intervals'])<.999:
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
            y=shift_legend_y+(40 if dashed else 20)
            page.draw_line((640,y),(665,y),color=color,width=2.5,dashes='[6 3] 0' if dashed else None)
            luminosity_label(page,(672,y+4),'SHIFT, 3 fb',label,color)
            counts.append({'mode':mode,'connected_mass_points':[r['mass_gev'] for r in nodes]})
        page.draw_rect(fitz.Rect(638.3,shift_legend_y+60,width,height+offset),color=None,fill=(1,1,1))
        luminosity_label(page,(84,17),'SHIFT Both-Both, 3 fb',f': expected {round(100*level)}% CLs',(0,0,0),fontsize=13)
        notes=[
            'Effective halo luminosity: 3 fb^-1, assuming 1% of the base collision exposure contains halo interactions.',
            'Trigger efficiency: 100%.',
            'MC is assumed to match data; all scale factors are 1.',
            'Both-Both dimuons only.',
            'No vertex classifier is applied.',
            'No additional signal-versus-background cuts are applied.',
            'Background: published QCD + DY + J/psi histograms only; empty bins stay empty and no missing tails are invented.',
            'Zero-background curve: background is set to zero as an idealized benchmark.',
            'Independent 30% signal and background normalization uncertainties.',
            'MC statistical uncertainties are omitted.',
            'Lifetime-dependent efficiencies and mass shapes use the available DP samples, including meson decays.',
            'Mass bins use the saved 0.1 GeV grid and a provisional 2% resolution scale.',
            'All available mass boundaries are connected in log mass / log epsilon; gaps and sparse MC remain provisional.',
            'Epsilon above 0.1 is a formal model extrapolation.',
            f'Current SHIFT: expected {round(100*level)}% CLs. Most source projections: 90% CL; FCC: 95% CL.']
        if omitted_masses:notes.append('Temporarily omitted pending review: '+', '.join(f'{m:g} GeV' for m in omitted_masses)+'. Neighbouring points are connected.')
        for index,note in enumerate(notes):
            y=height+offset+13+index*13
            page.draw_circle((87,y-3),1.5,color=None,fill=(0,0,0))
            page.insert_text((96,y),note,fontsize=10,fontname='tiro')
        page.insert_text((84,height+offset+211+footer_extra),'Source: arXiv:2511.03883v1, Fig. 8.15 (CC BY 4.0). Original SHIFT exposure: 30 fb^-1 (1% of 3 ab^-1).',fontsize=9.2,fontname='tiro')
        path=model.output/f'limits_current_cl{round(100*level)}_comparison.pdf';temporary=path.with_suffix('.tmp.pdf')
        doc.save(temporary,garbage=4,deflate=True);doc.close();temporary.replace(path)
        receipts.append({'file':path.name,'boundaries':counts,'sha256':digest(path)})
    clean.close();source.close()
    save_json(model.output/'original_overlay_provenance.json',{'source_sha256':provenance['figure_sha256'],'removed_svg_elements':removed_elements,
        'removed_projection_colors':sorted(removed),'retained_legend_drawing_indices':legend_indices,'figures':receipts,
        'recolored_source_curves':recolor,'original_shift_luminosity_fb_inverse':30,
        'omitted_masses_gev':list(omitted_masses),'omission_reason':'Temporary user-requested presentation omission pending review; raw scan and fit results retained.',
        'upper_outline':'Finite upper edges are not joined to artificial plot-top caps; exclusion shading and numerical limits are unchanged.',
        'style':'Original vector axes, existing exclusion shading, curve styles, labels and logo retained; retained legend rows copied from source.'})


def draw_inputs(config,model):
    plt=plotting();edges=model.snapshot['mass_edges_gev'];fig,ax=plt.subplots(figsize=(9,6));fig.suptitle('Background templates actually used in the limits')
    for ax,category in [(ax,config.primary_channel)]:
        for process,color in [('qcd','#666666'),('dy','#3073a5'),('jpsi','#a656a2')]:
            values=model.processes[process].sum(axis=0) if category=='combined' else model.processes[process][config.categories.index(category)]
            ax.stairs(np.where(values>0,values,np.nan),edges,label=process.upper(),color=color)
        ax.set(xscale='log',yscale='log',xlim=(.1,100),xlabel='Reconstructed dimuon mass [GeV]',ylabel='Expected events / likelihood bin',title=category);ax.legend()
    finish(fig,model.output/'background_mass_templates.pdf','3 fb^-1. Rates use each published file\'s full source/GEN exposure and weights once; no additional filtering efficiency or scale factor.\nBin widths follow the saved 0.1 GeV grid and a provisional 2% scale. There is no MC-population-driven bin merging or background-tail interpolation.',plt)
    fig,ax=plt.subplots(figsize=(11,6.5));fig.suptitle('Published signal coverage used in this update')
    xx=[];yy=[];counts=[];mechanisms=set();unused=[]
    used={r['name'] for rows in model.signals.values() for r in rows}
    for row in model.snapshot['signals']:
        if row['name'] not in used:unused.append(row);continue
        xx.append(row['mass_gev']);yy.append(row['epsilon']);counts.append(row['counts'][config.primary_channel]['sumw']);mechanisms.add(row['mechanism'])
    from matplotlib.colors import LogNorm
    obj=ax.scatter(xx,yy,c=np.maximum(counts,1),s=35,norm=LogNorm(),cmap='viridis',edgecolor='black',linewidth=.25)
    if unused:
        ax.scatter([r['mass_gev'] for r in unused],[r['epsilon'] for r in unused],marker='x',s=18,color='gray',label='Legacy epsilon canaries: not used');ax.legend(fontsize=8)
    fig.colorbar(obj,ax=ax,label='Selected Both-Both signal MC events (zero shown at color minimum)')
    ax.set(xscale='log',yscale='log',xlim=(.2,100),ylim=(1e-9,1.5),xlabel='Mass [GeV]',ylabel='Generated physical epsilon')
    finish(fig,model.output/'signal_coverage.pdf',f'{len(xx)} response anchors from {len(model.snapshot["signals"])} frozen published files; full GEN denominators include zero-dimuon events.\nMeson and DY rates include BR(mu mu) once. Tiny legacy epsilon canaries do not override the larger prompt response. No new events were generated.',plt)


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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-only',action='store_true');args=parser.parse_args()
    model=Model(config)
    if not args.original_only:draw_comparisons(config,model);draw_inputs(config,model);brazil(config,model)
    draw_original_comparisons(config,model);qa(model.output)

if __name__=='__main__':main()
