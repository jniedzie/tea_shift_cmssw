# Code style: AI-code. AI-generated focused checks of the remaining sensitivity gap.
import json,math,sys
from pathlib import Path
import numpy as np
from shift_limits_lifetime import save_json
from shift_limits_profile import expected_limit
from shift_limits_gap_analysis import histogram,physical_pair_mass
from shift_limits import digest

def main():
 import shift_limits_book_match_config as config
 output=Path(config.output_path).resolve()
 samples={p.stem:json.loads(p.read_text()) for p in (output/'candidates').glob('*.json')}
 signal=samples['m2_prompt'];selected=histogram(signal['candidates'],'ideal_mass')
 s=selected*config.luminosity/signal['full_gen_sumw']
 background=np.zeros(1000);origins={};examples=[]
 for name,row in samples.items():
  if row['kind']!='background':continue
  if name.startswith('dy_'):background+=histogram(row['candidates'],'ideal_mass')*config.luminosity*row['sigma_pb']/row['full_gen_sumw']
  totals={'matched':0.,'common_jpsi':0.,'common_dy':0.,'other':0.,'late_pion_kaon':0.,'mc':0}
  for r in row['candidates']:
   if not r['both_hit_matched'] or r['same_truth_particle']:continue
   mass=physical_pair_mass(r['matched_gen_muons']);index=min(999,int(mass/.1))
   if not s[index]:continue
   weight=r['weight']*config.luminosity*row['sigma_pb']/row['full_gen_sumw']
   shared=set(map(tuple,r['ancestor_chains'][0]))&set(map(tuple,r['ancestor_chains'][1]))
   shared_pdg={abs(x[1]) for x in shared}
   kind='common_jpsi' if 443 in shared_pdg else 'common_dy' if shared_pdg&{22,23} else 'other'
   totals['matched']+=weight;totals[kind]+=weight;totals['mc']+=1
   if any(chain and abs(chain[0][1]) in (211,321,130,310) and abs(g['vz']-14800)>100 for chain,g in zip(r['ancestor_chains'],r['matched_gen_muons'])):totals['late_pion_kaon']+=weight
   if name.startswith('jpsi_') and len(examples)<6:examples.append({'sample':name,'id':r['id'],'gen_mass':mass,'reco_mass':r['mass'],'common_ancestors':sorted(shared),'yield_30fb':weight})
  origins[name]=totals
 sigma,_=expected_limit(s,background,.90)
 epsilon=signal['authority']['epsilon']*math.sqrt(sigma/signal['sigma_pb'])
 item={'mode':'ideal_mass_dy_only','point':'m2_prompt','mass_gev':2.,'sigma_reference_pb':signal['sigma_pb'],
  'epsilon_reference':signal['authority']['epsilon'],'full_gen':signal['full_gen_sumw'],
  'selected_mc':float(selected.sum()),'efficiency':float(selected.sum()/signal['full_gen_sumw']),
  'luminosity_pb':config.luminosity,'background_in_signal_cells':float(background[s>0].sum()),'background_total':float(background.sum()),
  'confidence_levels':{'0.9':{'cross_section_pb':sigma,'epsilon':epsilon,'source':'exact_asimov_profile','selected_events_at_limit':float(sigma*s.sum())}}}
 comparison=json.loads((output/'comparison.json').read_text())
 comparison['records']=[r for r in comparison['records'] if r['mode']!='ideal_mass_dy_only']+[item]
 save_json(output/'comparison.json',comparison)
 save_json(output/'ideal_mass_background_origins.json',{'signal_cells_gev':[[float(i/10),float((i+1)/10)] for i in np.flatnonzero(s)],
  'origins':origins,'examples':examples,'ideal_dy_limit':item,'note':'MC-only diagnostic. Shared ancestor is not a data selection.'})
 # Keep the originally observed light-mass placeholder and make the corrected
 # diagnostic field explicit without rereading any ROOT input.
 corrections=[]
 for path in sorted((output/'candidates').glob('*.json')):
  row=json.loads(path.read_text());before=digest(path);changed=False
  for r in row['candidates']:
   if not r['both_hit_matched']:continue
   corrected=physical_pair_mass(r['matched_gen_muons'])
   if 'matched_gen_mass_from_nano_mass_field' not in r:
    r['matched_gen_mass_from_nano_mass_field']=r['matched_gen_mass'];r['matched_gen_mass']=corrected;changed=True
  if changed:
   row['truth_mass_convention']='Physical muon mass restored; Nano GenPart_mass contains zero for light species. No RECO field or selection changed.'
   save_json(path,row);corrections.append({'path':str(path),'before_sha256':before,'after_sha256':digest(path)})
 save_json(output/'truth_mass_correction_receipt.json',corrections)
 save_json(output/'candidate_cache_receipt.json',{p.name:digest(p) for p in sorted((output/'candidates').glob('*.json'))})
 print('Ideal mass and DY only: epsilon90',epsilon,'B in signal cells',item['background_in_signal_cells'],flush=True)

def material_proxy():
 """Bound the effect of late hadron decays, without treating truth as a cut."""
 import shift_limits_book_match_config as config
 output=Path(config.output_path).resolve()
 samples={p.stem:json.loads(p.read_text()) for p in (output/'candidates').glob('*.json')}
 signal=samples['m2_prompt'];summary=[];origins={};non_muons={}
 def late(row):
  return row['both_hit_matched'] and any(
   chain and abs(chain[0][1]) in (211,321,130,310) and abs(g['vz']-14800)>100
   for chain,g in zip(row['ancestor_chains'],row['matched_gen_muons']))
 for name,row in samples.items():
  non_muons[name]=sum(r['both_hit_matched'] and any(abs(g['pdgId'])!=13 for g in r['matched_gen_muons']) for r in row['candidates'])
 assert not any(non_muons.values()), 'Truth-mass diagnostic requires actual generator muons; inspect cached PDG IDs'
 for mode,shape in [('no_late_reco','raw30'),('no_late_ideal','ideal_mass')]:
  selected=histogram(signal['candidates'],shape)
  s=selected*config.luminosity/signal['full_gen_sumw'];background=np.zeros(1000);components={}
  for name,row in samples.items():
   if row['kind']!='background':continue
   scale=config.luminosity*row['sigma_pb']/row['full_gen_sumw']
   all_rows=histogram(row['candidates'],shape)*scale
   kept=histogram([r for r in row['candidates'] if not late(r)],shape)*scale
   background+=kept;components[name]=float(kept[s>0].sum())
   origins.setdefault(mode,{})[name]={'before':float(all_rows[s>0].sum()),'after':components[name]}
  sigma,_=expected_limit(s,background,.90)
  epsilon=signal['authority']['epsilon']*math.sqrt(sigma/signal['sigma_pb'])
  summary.append({'mode':mode,'point':'m2_prompt','mass_gev':2.,'selected_mc':float(selected.sum()),
   'luminosity_pb':config.luminosity,'full_gen':signal['full_gen_sumw'],
   'background_in_signal_cells':float(background[s>0].sum()),'background_components_in_signal_cells':components,
   'confidence_levels':{'0.9':{'cross_section_pb':sigma,'epsilon':epsilon,'source':'exact_asimov_profile',
    'selected_events_at_limit':float(sigma*s.sum())}},
   'caveat':'Diagnostic removal of pairs with a hit-matched pion/kaon daughter born >1 m in z from nominal target. No material path is calculated; this is neither a data selection nor validated hadron survival.'})
  print(mode,'epsilon90',epsilon,'B in signal cells',float(background[s>0].sum()),flush=True)
 comparison=json.loads((output/'comparison.json').read_text())
 comparison['records']=[r for r in comparison['records'] if r['mode'] not in ('no_late_reco','no_late_ideal')]+summary
 save_json(output/'comparison.json',comparison)
 save_json(output/'material_proxy.json',{'records':summary,'background_origins':origins,'non_muon_matched_pairs':non_muons})

if __name__=='__main__':
 if sys.argv[1:]==['--material-proxy']:material_proxy()
 else:main()
