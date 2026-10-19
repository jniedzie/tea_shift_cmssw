#!/usr/bin/env python3
# Code style: AI-code. AI-generated controlled sensitivity ingredient comparisons.
import argparse, copy, json, math, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import shift_limits as limits
from shift_limits_lifetime import save_json
from shift_limits_profile import expected_limit

MODES=('raw3','raw30','quality','constrained','constrained_quality','stream_raw','stream_constrained',
       'ideal_mass','ideal_mass_no_qcd','dy_only','zero_background')

def select(row,mode):
    constrained='constrained' in mode
    if constrained and row['constrainedValid']!=1:return False
    if mode.startswith('ideal_mass') and (not row['both_hit_matched'] or row['same_truth_particle']):return False
    if 'quality' in mode:
        if row['minimum_p']<=30:return False
        if row['normalizedChi2'] is None or not 0<=row['normalizedChi2']<5:return False
        if row['originCompatibilityNormalizedChi2'] is None or not 0<=row['originCompatibilityNormalizedChi2']<5:return False
    return True

def histogram(rows,mode):
    result=np.zeros(1000)
    for row in rows:
        if not select(row,mode):continue
        mass=row['constrainedMass'] if 'constrained' in mode else row['mass']
        if mode.startswith('ideal_mass'):mass=physical_pair_mass(row['matched_gen_muons'])
        if mass is None or not math.isfinite(mass) or mass<0:raise ValueError('Invalid candidate mass')
        result[min(999,int(mass/.1))]+=row['weight']
    return result

def physical_pair_mass(muons):
    vectors=[]
    for g in muons:
        mass=.1056583755 if abs(g['pdgId'])==13 else g['mass']
        pt,pz,phi=g['pt'],g['pz'],g['phi']
        vectors.append((math.sqrt(pt*pt+pz*pz+mass*mass),pt*math.cos(phi),pt*math.sin(phi),pz))
    total=[sum(v[i] for v in vectors) for i in range(4)]
    return math.sqrt(max(0,total[0]**2-sum(x*x for x in total[1:])))

def stream_candidates(output):
    snapshot_path=output/'stream_snapshot.json'
    if not snapshot_path.is_file():
        raise FileNotFoundError('Missing streaming diagnostic input: '+str(snapshot_path)+
            '. Supply the manually prepared, audited stream snapshot; this utility does not produce it.')
    snapshot=json.loads(snapshot_path.read_text())
    import ROOT
    from shift_limits_gap import candidate
    prefix=[r for r in snapshot if int(r['task'].split('_')[-2])<4]
    strata=sorted(set(r['stratum'] for r in prefix));assert len(strata)==6
    result=[]
    for name in strata:
        sources=[r for r in prefix if r['stratum']==name]
        assert len(sources)==4
        trials=sum(r['source_trials'] for r in sources)
        sigma=sum(r['source_trials']*r['sigma_nd_pb'] for r in sources)/trials
        rows=[]
        for source in sources:
            if not source['both_both_raw']:continue
            first_source_row=len(rows)
            record=source['merged_nano']
            assert limits.digest(record['path'])==record['sha256']
            file=ROOT.TFile.Open(record['path']);tree=file.Get('Events')
            available={b.GetName() for b in tree.GetListOfBranches()}
            for event in tree:
                if not int(event.nShiftDimuonVertex):continue
                if (int(event.ShiftDimuonVertex_topologyMin[0]),int(event.ShiftDimuonVertex_topologyMax[0]))!=(2,2):continue
                row=candidate(event,available)
                # This branch already contains native GEN weight / inclusion probability.
                row['weight']=float(event.shiftSamplingGenWeight)
                row['id']=[int(event.run),int(event.luminosityBlock),int(event.event)]
                rows.append(row)
            file.Close()
            assert math.isclose(sum(r['weight'] for r in rows[first_source_row:]),source['both_both_sumw'],rel_tol=1.e-12)
        result.append({'name':name,'sigma_pb':sigma,'full_gen_sumw':trials,'candidates':rows,'sources':sources})
    save_json(output/'stream_candidates.json',{'prefix':'indices 0000--0003 in each of six strata','strata':result,
        'caveat':'Partial fixed prefix; raw zeros treated as zero only for this diagnostic. W/p once, full ND trials once.'})
    return result

def prepare(config):
    output=Path(config.output_path).resolve()
    samples={p.stem:json.loads(p.read_text()) for p in sorted((output/'candidates').glob('*.json'))}
    save_json(output/'candidate_cache_receipt.json',{p.name:limits.digest(p) for p in sorted((output/'candidates').glob('*.json'))})
    original=json.loads(Path(config.audited_yields_path).read_text())
    assert len(samples)==len(original['backgrounds'])+len(config.signal_points)
    for name,row in samples.items():
        if name in original['backgrounds']:
            old=original['backgrounds'][name]['source']['source_file_stats']
            assert row['source_file_stats']==old,'Original audited source metadata changed: '+name
    stream=stream_candidates(output)
    summary=[];truth={};models={}
    for name,row in samples.items():
        matched=[r for r in row['candidates'] if r['both_hit_matched'] and not r['same_truth_particle']]
        for r in matched:r['matched_gen_mass']=physical_pair_mass(r['matched_gen_muons'])
        total=sum(r['weight'] for r in row['candidates']);matchedw=sum(r['weight'] for r in matched)
        inflated=sum(r['weight'] for r in matched if r['mass']>2*r['matched_gen_mass'])
        late_pion=sum(r['weight'] for r in matched if any(
            chain and abs(chain[0][1]) in (211,321,130,310) and abs(g['vz']-14800)>100
            for chain,g in zip(r['ancestor_chains'],r['matched_gen_muons'])))
        truth[name]={'selected_mc':len(row['candidates']),'sumw':total,'both_hit_matched_mc':len(matched),
            'both_hit_matched_sumw':matchedw,'mass_more_than_twice_truth_sumw':inflated,
            'pion_kaon_daughter_beyond_nominal_target_1m_sumw':late_pion,
            'weighted_mass_quantiles':None,
            'source_note':'Nano GenPart_mass is zero for light particles; diagnostic uses physical muon mass.'}
    save_json(output/'truth_diagnosis.json',truth)
    for mode in MODES:
        luminosity=3000. if mode=='raw3' else config.luminosity
        backgrounds={}
        if mode!='zero_background':
            for name,row in samples.items():
                if row['kind']!='background':continue
                if mode.startswith('stream_') and name.startswith('qcd_'):continue
                if mode=='ideal_mass_no_qcd' and name.startswith('qcd_'):continue
                if mode=='dy_only' and not name.startswith('dy_'):continue
                backgrounds[name]=histogram(row['candidates'],mode)*luminosity*row['sigma_pb']/row['full_gen_sumw']
            if mode.startswith('stream_'):
                for row in stream:backgrounds['stream_'+row['name']]=histogram(row['candidates'],mode)*config.luminosity*row['sigma_pb']/row['full_gen_sumw']
        total=sum(backgrounds.values(),np.zeros(1000))
        for name in config.signal_points:
            row=samples[name];selected=histogram(row['candidates'],mode)
            signal=selected*luminosity/row['full_gen_sumw']
            item={'mode':mode,'point':name,'mass_gev':float(name.split('_')[0][1:]),'sigma_reference_pb':row['sigma_pb'],
                'epsilon_reference':row['authority']['epsilon'],'full_gen':row['full_gen_sumw'],
                'selected_mc':float(selected.sum()),'efficiency':float(selected.sum()/row['full_gen_sumw']),'luminosity_pb':luminosity,
                'background_total':float(total.sum()),'background_in_signal_cells':float(total[signal>0].sum()),
                'background_components_in_signal_cells':{n:float(b[signal>0].sum()) for n,b in backgrounds.items()},
                'zero_background_signal_fraction':float(signal[total==0].sum()/signal.sum()) if signal.sum() else None,
                'confidence_levels':{}}
            if signal.sum():
                for cl in config.confidence_levels:
                    value,target=expected_limit(signal,total,cl)
                    item['confidence_levels'][str(cl)]={'cross_section_pb':value,'epsilon':row['authority']['epsilon']*math.sqrt(value/row['sigma_pb']),
                        'q_target':target,'source':'exact_asimov_profile','selected_events_at_limit':float(value*signal.sum())}
                item['zero_background_exact90_no_systematic_epsilon']=row['authority']['epsilon']*math.sqrt(-math.log(.1)/(signal.sum()*row['sigma_pb']))
            # Combining all zero-signal cells into one control is exactly lossless
            # for the common background normalization nuisance.
            keep=signal>0;bg=np.r_[total[keep],total[~keep].sum()];sg=np.r_[signal[keep],0.]
            model={'schema':'shift-simplified-limit-yields-v1','luminosity_pb':luminosity,'categories':['Both-Both'],
                'backgrounds':{'background_total':{'yields':{'Both-Both':float(bg.sum())},'binned_yields':{'Both-Both':bg.tolist()}}},
                'signals':{name:{'mass_gev':item['mass_gev'],'epsilon':row['authority']['epsilon'],'sample_kind':'prompt',
                    'proper_ctau_mm':row['authority']['proper_ctau_mm'],'mean_lab_flight_m':None,'partial':False,
                    'reference_cross_section_pb':row['sigma_pb'],'cross_section_pb':row['sigma_pb'],
                    'full_gen_sumw':row['full_gen_sumw'],'yields':{'Both-Both':float(sg.sum()*row['sigma_pb'])},
                    'binned_yields':{'Both-Both':(sg*row['sigma_pb']).tolist()}}},
                'selection':{'mode':mode,'mass_bin_width_gev':.1,'zero_signal_bins_combined_into_control':True},
                'assumptions':['30 fb^-1; trigger/SF one; 30% signal and background lnN','Raw zero bins are optimistic assumptions; finite-MC uncertainty omitted']}
            directory=output/mode/name;save_json(directory/'yields.json',model)
            summary.append(item);save_json(output/'comparison.json',{'records':summary,'complete':False})
            print(mode,name,'selected',item['selected_mc'],'epsilon90',item['confidence_levels'].get('0.9',{}).get('epsilon'),flush=True)
    save_json(output/'comparison.json',{'records':summary,'complete':True})

def native(config,logs):
    import limits_producer
    from shift_limits_resolution import poisson_card
    from shift_limits_counting import prepare_mass_auxiliary_card
    from shift_limits_combine_precision import verify_stable_nll
    output=Path(config.output_path).resolve();results=[]
    setup=limits_producer.find_combine(config);limits_producer.find_combine=lambda c:setup
    verify_stable_nll(config)
    binary=subprocess.run(['bash','-lc',setup+'; command -v combine'],capture_output=True,text=True,check=True).stdout.strip().splitlines()[-1]
    def command(text):
        log=logs/('native_'+str(len(results))+'.log')
        with log.open('a') as f:
            f.write(text+'\n');f.flush()
            return subprocess.run(text,shell=True,executable='/bin/bash',stdout=f,stderr=subprocess.STDOUT,timeout=110).returncode
    limits_producer.run_command=command
    reference=json.loads((output/'comparison.json').read_text())['records']
    queue=[('raw30',n) for n in config.signal_points]+[('constrained','m2_prompt'),('stream_raw','m2_prompt'),('stream_constrained','m2_prompt')]
    for mode,name in queue:
        directory=output/mode/name;current=SimpleNamespace(**vars(config))
        current.output_path=str(directory)+'/'
        for attr,suffix in [('yields_path','yields.json'),('prepared_input_manifest','inputs/manifest.json'),('datacards_output_path','datacards/'),('results_output_path','results/'),('plots_output_path','brazil/')]:setattr(current,attr,str(directory/suffix)+('/' if suffix.endswith('/') else ''))
        current.signal_points=(name,);current.config_source_path=config.__file__
        report=json.loads(Path(current.yields_path).read_text());manifest=limits.prepare_inputs(current)
        cards=limits.create_cards(current,manifest,setup);card=next(c for c in cards if c['channel']=='Both-Both')
        if not limits.card_has_signal(card,manifest,config.categories):continue
        source=poisson_card(current,report,manifest,card);fitdir=directory/'native90'
        prepare_mass_auxiliary_card(source,fitdir/source.name)
        fit=limits.category_config(current,manifest,'Both-Both');fit.datacards_output_path=str(fitdir)+'/'
        fit.combine_options=('--cl','0.90');fit.combine_options_by_datacard={}
        limits.bind_fits(fit,manifest,[card],fitdir,binary)
        try:
            root=fitdir/('higgsCombine.'+card['name']+'.AsymptoticLimits.mH120.root')
            if not root.exists():limits_producer.run_combine(fit,[card['name']],method='AsymptoticLimits')
            row=limits.results_from_outputs(current,manifest,[card],.9,fitdir)[0]
            expected=next(x for x in reference if x['mode']==mode and x['point']==name)['confidence_levels']['0.9']['cross_section_pb']
            error=row['expected_limit_cross_section_pb']/expected-1
            row.update(mode=mode,relative_median_error=error,validated=abs(error)<.05)
            print('Native TEA',mode,name,'median difference',error,flush=True)
        except Exception as error:
            row={'mode':mode,'point':name,'validated':False,'error':str(error)[:400]}
            print('Native fit not validated:',row,flush=True)
        results.append(row);save_json(output/'native_results.json',{'records':results})
        if len(results)==1 and not results[0]['validated']:break

def main():
    import shift_limits_book_match_config as config
    parser=argparse.ArgumentParser();parser.add_argument('--logs',type=Path,required=True);parser.add_argument('--stage',choices=['prepare','native'],required=True);args=parser.parse_args()
    if args.stage=='prepare':prepare(config)
    else:native(config,args.logs)

if __name__=='__main__':main()
