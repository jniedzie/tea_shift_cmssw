#!/usr/bin/env python3
# Code style: AI-code. AI-generated plots for controlled sensitivity comparisons.
import json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from shift_limits_lifetime import plotting,save_json
from shift_limits import digest
from shift_limits_gap_analysis import histogram,physical_pair_mass

LABELS={'raw3':'Original QCD, raw mass bins, 3 fb⁻¹','raw30':'Original QCD, raw mass bins, 30 fb⁻¹',
 'quality':'Plus p(mu)>30 GeV and loose fit quality','constrained':'Stored target-constrained mass',
 'constrained_quality':'Constrained mass + loose quality',
 'stream_raw':'Partial mu-enriched QCD, ordinary mass',
 'stream_constrained':'Partial mu-enriched QCD, constrained mass',
 'ideal_mass':'Truth mass of matched pairs (diagnostic)',
 'ideal_mass_no_qcd':'Truth mass, omit QCD (diagnostic)',
 'dy_only':'Keep only DY background (diagnostic)',
 'zero_background':'Zero background (diagnostic)',
 'ideal_mass_dy_only':'Truth mass, DY background only (diagnostic)',
 'no_late_reco':'Omit late pion/kaon daughters (diagnostic)',
 'no_late_ideal':'Same omission + truth mass (diagnostic)'}

def main():
 import fitz
 import shift_limits_book_match_config as config
 from shift_paths import limits_output_path,limits_pheno_comparison_output_path
 output=Path(config.output_path).resolve();plt=plotting();pages=[]
 comparison=json.loads((output/'comparison.json').read_text())['records']
 native=json.loads((output/'native_results.json').read_text())['records'] if (output/'native_results.json').exists() else []
 samples={p.stem:json.loads(p.read_text()) for p in (output/'candidates').glob('*.json')}
 prompt=samples['m2_prompt']
 prompt_count=len(prompt['candidates'])
 exposure=', '.join(f'{float(name.split("_")[0][1:]):g} GeV: full GEN sumw={samples[name]["full_gen_sumw"]:g}' for name in config.signal_points)
 book=json.loads((Path(limits_pheno_comparison_output_path)/'reference_data/briefing_curve_readback.json').read_text())['epsilon_crossings']['2.0'][0]
 def finish(fig,name,caption):
  fig.text(.025,.018,caption,fontsize=8,va='bottom');fig.tight_layout(rect=(0,.09,1,.94))
  path=output/(name+'.pdf');fig.savefig(path);plt.close(fig);pages.append(path)
 rows=[r for r in comparison if r['point']=='m2_prompt']
 fig,ax=plt.subplots(figsize=(11,8.5));fig.suptitle('What changes the 2 GeV reach?  Expected 90% CLs')
 y=np.arange(len(rows))
 for i,r in enumerate(rows):
  value=r['confidence_levels'].get('0.9',{}).get('epsilon')
  if value:ax.plot(value,i,'o',mfc='white',mec='tab:blue',ms=8)
  hit=next((n for n in native if n['mode']==r['mode'] and n['point']==r['point'] and n.get('validated')),None)
  if hit:
   e=hit['epsilon']*math.sqrt(hit['expected_limit_cross_section_pb']/hit['reference_cross_section_pb']);ax.plot(e,i,'x',color='black',ms=8)
  if value:ax.annotate(f"{value:.2g}; {r['selected_mc']:g} MC pairs",(value,i),xytext=(9,0),textcoords='offset points',fontsize=8,va='center')
 ax.axvline(book,color='#00bfff',lw=2,label=f'Book SHIFT at 2 GeV: {book:.3g}')
 ax.plot([],[],'o',mfc='white',mec='tab:blue',label='Exact Asimov profile, 30% signal/background')
 ax.plot([],[],'kx',label='Validated native TEA / Combine median')
 ax.set_yticks(y,labels=[LABELS[r['mode']] for r in rows]);ax.invert_yaxis();ax.set(xscale='log',xlabel=r'Kinetic mixing $\epsilon$',xlim=(book/3,.3));ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=(1.,1.13))
 finish(fig,'01_sensitivity_changes',f'Both-Both only. 2 GeV prompt input: full GEN sumw={prompt["full_gen_sumw"]:g}, {prompt_count} reconstructed pairs; cuts are not optimized.\nRaw empty bins are taken literally. Truth/background-removal rows are counterfactuals; zero-background asymptotics are not an exact Poisson interval.')

 fig,axes=plt.subplots(1,3,figsize=(13,4.7));fig.suptitle('Signal mass response with the latest ready prompt samples')
 for ax,name in zip(axes,config.signal_points):
  row=samples[name];m=float(name.split('_')[0][1:]);edges=np.linspace(0,2*m,41)
  for mode,label,color in [('raw30','Ordinary mass','tab:blue'),('constrained','Constrained mass','tab:orange'),('ideal_mass','Matched GEN mass','tab:green')]:
   selected=[r for r in row['candidates'] if (mode!='constrained' or r['constrainedValid']==1) and (mode!='ideal_mass' or r['both_hit_matched'])]
   vals=[physical_pair_mass(r['matched_gen_muons']) if mode=='ideal_mass' else r['constrainedMass'] if mode=='constrained' else r['mass'] for r in selected]
   ax.hist(vals,bins=edges,weights=[r['weight']/row['full_gen_sumw'] for r in selected],histtype='step',label=label,color=color)
  ax.axvline(m,color='.5',ls=':');ax.set(xlabel='Dimuon mass [GeV]',ylabel='Selected / all generated per display bin',title=f'{m:g} GeV; NGEN={row["full_gen_sumw"]:g}');ax.legend(fontsize=8)
 finish(fig,'02_signal_mass_response','Constrained means the existing stored target-constrained fit; failures are discarded, never filled at zero.\nGEN matching is a diagnostic only. The likelihood uses 0.1 GeV bins; display bins are wider for readability.')

 fig,axes=plt.subplots(1,3,figsize=(13,4.7));fig.suptitle('Low generated masses feed the reconstructed background')
 from matplotlib.colors import LogNorm
 for ax,prefix,title in zip(axes,['qcd_','jpsi_','dy_'],['Original QCD','Direct J/psi','Drell–Yan']):
  xx=[];yy=[];ww=[]
  for name,row in samples.items():
   if not name.startswith(prefix):continue
   for r in row['candidates']:
    if not r['both_hit_matched'] or r['same_truth_particle']:continue
    xx.append(physical_pair_mass(r['matched_gen_muons']));yy.append(r['mass']);ww.append(r['weight']*30000*row['sigma_pb']/row['full_gen_sumw'])
  im=ax.hist2d(xx,yy,bins=(np.linspace(0,12,49),np.linspace(0,20,81)),weights=ww,norm=LogNorm(),cmap='viridis')
  ax.plot([0,12],[0,12],'r--',lw=1);ax.set(xlabel='Mass from matched GEN muons [GeV]',ylabel='Reconstructed mass [GeV]',title=title);fig.colorbar(im[3],ax=ax,label='Expected events / cell at 30 fb⁻¹')
 finish(fig,'03_background_mass_migration','First Both-Both candidates with two distinct hit-matched muons. Truth is not used to select fitted data candidates.\nThis compares the same two muons at production and in reconstruction; energy loss and angular response both contribute.')

 fig,ax=plt.subplots(figsize=(9.5,5));fig.suptitle('Mass response and raw background assumptions control the comparison')
 colors={'raw30':'tab:red','constrained':'tab:orange','stream_raw':'tab:blue','stream_constrained':'tab:purple','zero_background':'tab:green'}
 for mode,color in colors.items():
  selected=sorted((r for r in comparison if r['mode']==mode and '0.9' in r['confidence_levels']),key=lambda r:r['mass_gev'])
  ax.plot([r['mass_gev'] for r in selected],[r['confidence_levels']['0.9']['epsilon'] for r in selected],'o--',color=color,label=LABELS[mode])
 ax.plot([2],[book],'s',color='#00bfff',ms=8,label='Book SHIFT: only common simulated mass')
 ax.set(xscale='log',yscale='log',xlabel=r'$m_{A\prime}$ [GeV]',ylabel=r'Expected 90% $\epsilon$ boundary',xlim=(1.7,12));ax.legend(fontsize=8)
 finish(fig,'04_mass_comparison','30 fb⁻¹; Both-Both; independent 30% normalization uncertainties. Dashed connectors guide the eye; no new validated contour is inferred.\nPrompt inputs: '+exposure+'.')

 baseline=next(r for r in rows if r['mode']=='raw30');rate=baseline['sigma_reference_pb']*(book/baseline['epsilon_reference'])**2
 senscalc_bkg=4*10**(6.1*math.exp(-.06*2)+2.8*2**(-.2))
 produced=30000*rate;selected=produced*baseline['efficiency'];senscalc_ratio=10.93
 budget={'book_epsilon_2gev':book,'our_dimuon_produced_30fb':produced,'our_both_both_selected_30fb':selected,
  'public_senscalc_dy_rate_ratio':senscalc_ratio,'senscalc_dy_produced_before_acceptance':produced*senscalc_ratio,
  'public_fork_background_at_2gev':senscalc_bkg,'public_fork_required_default_2p3sqrtB':2.3*math.sqrt(senscalc_bkg),
  'alternative_1p28sqrtB':1.28*math.sqrt(senscalc_bkg),
  'scope':'Public pinned source is not an authenticated record of the briefing-book run; DY-only comparison. Other production mechanisms remain separate.',
  'zero_background_exact90_epsilon':baseline['zero_background_exact90_no_systematic_epsilon']}
 save_json(output/'book_rate_budget.json',budget)
 fig,ax=plt.subplots(figsize=(11,5.5));fig.suptitle('At the published 2 GeV coupling, how many events are available?')
 labels=['Our DY dimuons, before acceptance','Our reconstructed Both-Both dimuons','Public SensCalc DY rate, before acceptance','Public fork default: 2.3 sqrt(background)']
 values=[produced,selected,produced*senscalc_ratio,2.3*math.sqrt(senscalc_bkg)]
 ax.barh(range(4),values,color=['tab:blue','tab:blue','tab:cyan','tab:red']);ax.set_yticks(range(4),labels=labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(.3,max(values)*4);ax.set_xlabel('Events at 30 fb⁻¹')
 for i,v in enumerate(values):ax.text(v*1.12,i,f'{v:.3g}',va='center')
 finish(fig,'05_book_rate_budget','The public fork background formula gives B(m)=4×10^[6.1 exp(-0.06m)+2.8 m^-0.2]. Its exact relation to the book run is unverified.\nThis DY-only event budget shows why luminosity and reconstruction efficiency alone cannot reconcile that background model with the curve.')

 # Reuse the original vector figure. Multiple diagnostic boundaries are drawn
 # explicitly as open lines; no new exclusion shading is claimed from 3 masses.
 original=Path(limits_output_path).resolve();source=fitz.open(original/'arxiv_2511.03883v1_fig8_15.pdf')
 provenance=json.loads((original/'paper_figure_provenance.json').read_text());width,height=provenance['figure_size_points']
 doc=fitz.open();page=doc.new_page(width=width+40,height=height+190)
 page.insert_text((22,25),'SHIFT sensitivity diagnosis: 30 fb^-1, Both-Both, expected 90% CLs',fontsize=14)
 top=70;page.show_pdf_page(fitz.Rect(20,top,20+width,top+height),source,0)
 from shift_limits_comparison import paper_coordinates
 rgb={'raw30':(.8,.1,.1),'stream_constrained':(.5,.15,.7),'zero_background':(.1,.55,.2)}
 for j,(mode,color) in enumerate(rgb.items()):
  data=sorted((r for r in comparison if r['mode']==mode and '0.9' in r['confidence_levels']),key=lambda r:r['mass_gev'])
  coords=[paper_coordinates(r['mass_gev'],r['confidence_levels']['0.9']['epsilon'],provenance,top) for r in data]
  shape=page.new_shape();shape.draw_polyline(coords);shape.finish(color=color,width=2,closePath=False);shape.commit()
  for point in coords:page.draw_circle(point,3,color=color,fill=(1,1,1))
  page.draw_line((22+j*245,48),(47+j*245,48),color=color,width=2)
  label={'raw30':'Original QCD, raw bins','stream_constrained':'Partial QCD, constrained mass','zero_background':'Zero background diagnostic'}[mode]
  page.insert_text((52+j*245,52),label,fontsize=9)
 footer=top+height+24
 for i,text in enumerate(['Colored points: exact Asimov-profile medians with 30% signal/background uncertainties; native cross-checks in the study.',
   f'Only 2 GeV overlaps the cyan SHIFT curve. 2 GeV has {prompt_count} prompt Both-Both candidates.',
   'Raw MC zeros are optimistic assumptions. Lines connect tested masses only; no new exclusion shading is inferred.',
   'Zero-background curve is a diagnostic. SensCalc background/production inputs for the book remain unauthenticated.']):page.insert_text((22,footer+15*i),text,fontsize=8)
 overlay=output/'paper_comparison90.pdf';doc.save(overlay,garbage=4,deflate=True);doc.close();source.close();pages.append(overlay)

 valid=[r for r in native if r.get('validated') and r['mode']=='raw30']
 if valid:
  from shift_limits_shapes import draw_native_brazil
  current=SimpleNamespace(**vars(config));current.plots_output_path=str(output/'brazil');current.projection_method_label=None
  for row in valid:row['validation']={'bands_valid':False,'relative_errors':{'median':row['relative_median_error']}}
  draw_native_brazil(current,{('Both-Both',.9,'prompt'):valid})
  pages+=sorted((output/'brazil').rglob('*.pdf'))
 booklet=fitz.open()
 for path in pages:
  with fitz.open(path) as part:booklet.insert_pdf(part)
 booklet.save(output/'sensitivity_gap_walkthrough.pdf',garbage=4,deflate=True);booklet.close()
 receipt={}
 for path in output.rglob('*.pdf'):
  with fitz.open(path) as pdf:
   assert len(pdf)>0
   assert all(page.get_text().strip() for page in pdf)
   receipt[str(path.relative_to(output))]={'sha256':digest(path),'bytes':path.stat().st_size,'pages':len(pdf)}
 save_json(output/'plot_receipt.json',receipt)
 print('Published',len(receipt),'PDFs; native validated medians',sum(r.get('validated',False) for r in native),flush=True)

if __name__=='__main__':main()
