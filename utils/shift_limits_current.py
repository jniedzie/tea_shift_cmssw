#!/usr/bin/env python3
# Code style: AI-code. AI-generated update of limits using published histograms.
"""Snapshot ready inputs, interpolate physical lifetime templates, and use TEA cards."""
import argparse
import copy
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
from scipy.stats import norm
from shift_limits import digest
from shift_limits_lifetime import save_json
from shift_limits_resolution import resolution_boundaries
from shift_limits_shapes import rebin
from shift_limits_profile import profile_q


def histogram_array(file, name, edges):
    hist = file.Get(name)
    if not hist or hist.GetXaxis().GetXmin() != 0 or not math.isclose(hist.GetBinWidth(1),.1,rel_tol=1e-8) or hist.GetXaxis().GetXmax()<100:
        raise ValueError('Expected saved 0.1 GeV mass histogram: '+name)
    size=hist.GetNbinsX()
    values = np.array([hist.GetBinContent(i) for i in range(size+2)])
    variances = np.array([hist.GetBinError(i)**2 for i in range(size+2)])
    if np.any(values < 0) or not np.all(np.isfinite(values+variances)):
        raise ValueError('Invalid histogram content: '+name)
    # Older SM files extend to 200 GeV, while some signals end at 100 GeV.
    # Fold all stored content above 100 into the last bin, without splitting
    # any source bin or filling an empty background bin.
    values[1] += values[0]; values[1000] += values[1001:].sum()
    variances[1] += variances[0]; variances[1000] += variances[1001:].sum()
    return np.asarray(rebin(values[1:1001],edges)), np.asarray(rebin(variances[1:1001],edges))


def snapshot(config):
    import ROOT
    from shift_plot_samples import prepare_plot_samples
    from shift_sample_paths import available_bsm_campaign_histograms
    ROOT.gROOT.SetBatch(True)
    output=Path(config.output_path).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'snapshot.json').exists():
        raise ValueError('A frozen snapshot already exists; reuse it instead of repeating input reads')
    backgrounds,_=prepare_plot_samples(config.base_path,config.cross_sections,config.campaign,None,None,None,None,None)
    points=available_bsm_campaign_histograms(config.base_path,config.signal_campaign,config.signal_masses,config.signal_flights,epsilons=config.signal_epsilons)
    edges=resolution_boundaries(config.mass_resolution_fraction,config.mass_minimum_bin_gev)
    arrays={};records={'backgrounds':[],'signals':[]}
    for kind,items in [('backgrounds',backgrounds),('signals',points)]:
        for index,item in enumerate(items):
            name=item.name if kind=='backgrounds' else item['name']
            path=Path(item.file_path if kind=='backgrounds' else item['histogram_path']).resolve()
            key=f'{kind}_{index}';source={'name':name,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}
            file=ROOT.TFile.Open(str(path),'READ')
            if not file or file.IsZombie():raise OSError(str(path))
            try:
                flow=file.Get('cutFlow');initial=next(flow.GetBinContent(i) for i in range(1,flow.GetNbinsX()+1) if flow.GetXaxis().GetBinLabel(i)=='0_initial')
                if kind=='signals':
                    # The histogram owns its actually published denominator. Rate
                    # metadata already includes BR(mu mu), including meson sources.
                    authority_path=path.parent.parent/'cross_sections.json'
                    authority=json.loads(authority_path.read_text())
                    sigma=sum(row['value_pb'] for row in authority['physical_selected_cross_sections_pb'])
                    denominator=authority['full_gen']['sum_weights']
                    if not math.isclose(initial,denominator,rel_tol=2e-6):raise ValueError('Published GEN denominator mismatch: '+name)
                    source.update(mass_gev=authority['mass_gev'],epsilon=authority['epsilon'],proper_ctau_mm=authority['proper_ctau_mm'],
                        mechanism=authority['mechanism'],prompt=name.endswith('_prompt'),rate_authority=str(authority_path),rate_authority_sha256=digest(authority_path),
                        physics_valid=authority.get('physics_valid'),normalization_ready=authority.get('normalization_ready'))
                else:
                    sigma=item.cross_section;denominator=item.initial_weight_sum if item.initial_weight_sum>0 else initial
                if not sigma>0 or not denominator>0:raise ValueError('Invalid normalization: '+name)
                histograms=[];variances=[];counts={}
                for category in config.categories:
                    values,variance=histogram_array(file,'dimuon/ShiftDimuonVertex'+category+'_mass',edges)
                    multiplicity=file.Get('event/Event_nShiftDimuonVertex'+category)
                    multi=sum(multiplicity.GetBinContent(i) for i in range(3,multiplicity.GetNbinsX()+2))
                    if abs(multi)>1e-10:raise ValueError('More than one dimuon/event; cannot treat categories as exclusive: '+name)
                    selected=multiplicity.GetBinContent(2)
                    if not math.isclose(float(values.sum()),selected,rel_tol=2e-4,abs_tol=1e-8):raise ValueError('Mass/category count mismatch: '+name)
                    histograms.append(values);variances.append(variance)
                    counts[category]={'sumw':float(values.sum()),'sumw2':float(variance.sum())}
                arrays[key]=np.array(histograms);arrays[key+'_sumw2']=np.array(variances)
                source.update(array_key=key,cross_section_pb=sigma,full_gen_sumw=denominator,cutflow_initial=initial,counts=counts)
                records[kind].append(source)
            finally:file.Close()
            print('Snapshotted',kind,name,flush=True)
    target=output/'histograms.npz';np.savez_compressed(target,**arrays)
    save_json(output/'snapshot.json',dict(schema='shift-current-limit-snapshot-v1',luminosity_pb=config.luminosity,categories=list(config.categories),
        background_campaign=config.campaign,signal_campaign=config.signal_campaign,mass_edges_gev=[e/10 for e in edges],
        histogram_cache=str(target),histogram_cache_sha256=digest(target),flow_policy='Underflow and overflow folded into first and last mass bin',
        background_policy='Only published QCD/DY/Jpsi; no tail interpolation, smoothing, floor or truth removal',
        lifetime_policy='Convex interpolation of rate-weighted reconstructed shapes in log epsilon between generated anchors; prompt-only continuation towards shorter lifetimes',
        **records))
    print('Frozen',len(backgrounds),'backgrounds and',len(points),'signals.',flush=True)


class Model:
    def __init__(self,config):
        self.config=config;self.output=Path(config.output_path).resolve();self.snapshot=json.loads((self.output/'snapshot.json').read_text())
        assert digest(self.snapshot['histogram_cache'])==self.snapshot['histogram_cache_sha256']
        with np.load(self.snapshot['histogram_cache'],allow_pickle=False) as saved:self.arrays={k:saved[k].copy() for k in saved.files}
        self.background=np.zeros_like(self.arrays[self.snapshot['signals'][0]['array_key']])
        self.processes={name:np.zeros_like(self.background) for name in ['qcd','dy','jpsi']}
        self.signals={}
        for row in self.snapshot['backgrounds']:
            value=self.arrays[row['array_key']]*config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
            self.background+=value;self.processes[row['name'].split('_')[0]]+=value
        self.unused_signals=[]
        for row in self.snapshot['signals']:
            if any(row['name'].endswith('_'+anchor) for anchor in config.response_anchors):self.signals.setdefault(row['mass_gev'],[]).append(row)
            else:self.unused_signals.append(row)
        for rows in self.signals.values():rows.sort(key=lambda r:r['epsilon'])

    def signal(self,mass,epsilon):
        rows=self.signals[mass]
        if epsilon<rows[0]['epsilon'] and math.isclose(epsilon,rows[0]['epsilon'],rel_tol=1e-12):epsilon=rows[0]['epsilon']
        if epsilon>rows[-1]['epsilon'] and math.isclose(epsilon,rows[-1]['epsilon'],rel_tol=1e-12):epsilon=rows[-1]['epsilon']
        x=math.log(epsilon)
        def coefficient(row):
            return self.arrays[row['array_key']]*self.config.luminosity*row['cross_section_pb']/(row['full_gen_sumw']*row['epsilon']**2)
        if epsilon<rows[0]['epsilon']:return None
        if epsilon>rows[-1]['epsilon']:
            if not rows[-1]['prompt']:return None
            return coefficient(rows[-1])*epsilon**2
        for a,b in zip(rows,rows[1:]):
            if a['epsilon']<=epsilon<=b['epsilon']:
                if a['epsilon']==b['epsilon']:raise ValueError('Same-epsilon mechanisms need explicit rate-summed populations')
                fraction=(x-math.log(a['epsilon']))/math.log(b['epsilon']/a['epsilon'])
                assert -1e-12<=fraction<=1+1e-12
                fraction=min(1.,max(0.,fraction))
                return ((1-fraction)*coefficient(a)+fraction*coefficient(b))*epsilon**2
        return coefficient(rows[0])*epsilon**2

    def cross_section(self,mass,epsilon):
        rows=self.signals[mass]
        if epsilon>=rows[-1]['epsilon']:return rows[-1]['cross_section_pb']*(epsilon/rows[-1]['epsilon'])**2
        for a,b in zip(rows,rows[1:]):
            if a['epsilon']<=epsilon<=b['epsilon']:
                fraction=math.log(epsilon/a['epsilon'])/math.log(b['epsilon']/a['epsilon'])
                fraction=min(1.,max(0.,fraction))
                return epsilon**2*((1-fraction)*a['cross_section_pb']/a['epsilon']**2+fraction*b['cross_section_pb']/b['epsilon']**2)
        return rows[0]['cross_section_pb']*(epsilon/rows[0]['epsilon'])**2

    def cells(self,mass,epsilon,channel,mode):
        signal=self.signal(mass,epsilon)
        if signal is None:return None,None
        background=self.background if mode=='background' else np.zeros_like(self.background)
        if channel!='combined':
            index=self.config.categories.index(channel);signal=signal[index:index+1];background=background[index:index+1]
        return signal.ravel(),background.ravel()

    def q(self,mass,epsilon,channel,mode):
        signal,background=self.cells(mass,epsilon,channel,mode)
        if signal is None:return None
        if not np.any(signal):return 0.
        if mode=='zero_background':
            total=float(signal.sum());k=math.log(1.3);theta=-float(lambertw(k*k*total).real)/k
            return 2*total*math.exp(k*theta)+theta*theta
        return profile_q(1.,signal,background)


def scan(config):
    model=Model(config);records=[]
    path=model.output/'scan.json'
    if path.exists():raise ValueError('A completed scan exists; reuse it')
    checkpoint=model.output/'scan.checkpoint.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        assert saved['snapshot_sha256']==digest(model.output/'snapshot.json')
        records=saved['records']
    done={(r['mass_gev'],r['mode'],r['channel'],r['confidence_level']) for r in records}
    for mass,anchors in sorted(model.signals.items()):
        low=anchors[0]['epsilon'];high=config.epsilon_max if anchors[-1]['prompt'] else anchors[-1]['epsilon']
        grid=sorted(set(np.geomspace(low,high,config.scan_grid_points).tolist()+[r['epsilon'] for r in anchors]))
        for mode in config.background_modes:
            for channel in config.channel_names:
                if all((mass,mode,channel,level) in done for level in config.confidence_levels):continue
                values=[model.q(mass,e,channel,mode) for e in grid]
                for level in config.confidence_levels:
                    threshold=norm.ppf((1+level)/2)**2;roots=[]
                    for a,b,qa,qb in zip(grid,grid[1:],values,values[1:]):
                        if (qa-threshold)*(qb-threshold)<0:
                            root=math.exp(brentq(lambda loge:model.q(mass,math.exp(loge),channel,mode)-threshold,math.log(a),math.log(b),xtol=1e-9))
                            roots.append(root)
                    bounds=[low]+roots+[high];intervals=[]
                    for a,b in zip(bounds,bounds[1:]):
                        if model.q(mass,math.sqrt(a*b),channel,mode)>threshold:intervals.append([a,b])
                    records.append(dict(mass_gev=mass,mode=mode,channel=channel,confidence_level=level,roots=roots,
                        excluded_intervals=intervals,epsilon_support=[low,high],upper_prompt_continuation=anchors[-1]['prompt'],
                        q_grid=values,epsilon_grid=grid,source='independent_exact_asimov_profile'))
                print('Scanned',mass,mode,channel,flush=True)
        save_json(checkpoint,{'records':records,'snapshot_sha256':digest(model.output/'snapshot.json')})
    save_json(path,{'records':records,'snapshot_sha256':digest(model.output/'snapshot.json')})


def native(config,canary=False):
    import subprocess,time
    import shift_limits as limits
    import limits_producer
    from shift_limits_resolution import poisson_card
    from shift_limits_counting import prepare_mass_auxiliary_card
    from shift_limits_combine_precision import verify_stable_nll
    model=Model(config);scan_rows=json.loads((model.output/'scan.json').read_text())['records']
    destination=model.output/'native_results.json'
    results=json.loads(destination.read_text())['records'] if destination.exists() else []
    setup=limits_producer.find_combine(config)
    limits_producer.find_combine=lambda c:setup
    verify_stable_nll(config)
    binary=subprocess.run(['bash','-lc',setup+'; command -v combine'],capture_output=True,text=True,check=True).stdout.strip().splitlines()[-1]
    queue=[]
    for row in scan_rows:
        if row['channel']!=config.primary_channel or row['confidence_level'] not in config.native_confidence_levels:continue
        if canary and row['mass_gev']!=2.:continue
        for edge,epsilon in enumerate(row['roots']):queue.append((row,edge,epsilon))
    queue.sort(key=lambda entry:(entry[0]['channel']!='combined',entry[0]['confidence_level']!=.95,entry[0]['mass_gev'],entry[0]['mode']))
    primary_published=False
    for row,edge,epsilon in queue:
        identity=(row['mass_gev'],row['mode'],row['channel'],row['confidence_level'],edge)
        previous=next((r for r in results if tuple(r[k] for k in ['mass_gev','mode','channel','confidence_level','edge'])==identity),None)
        if previous and previous['validated']:
            assert digest(previous['combine_output'])==previous['combine_output_sha256']
            print('Reused native result',identity,flush=True);continue
        started=time.monotonic();mass,mode,channel,level,_=identity
        zero_check=next((r for r in results if r['mode']=='zero_background' and r['channel']==channel
                         and r['confidence_level']==level and r['validated']
                         and not r.get('equivalent_counting_reuse')),None)
        if mode=='zero_background' and zero_check:
            # One category with zero observations is exactly a single Poisson
            # count, regardless of its mass shape. Reuse that identical native
            # likelihood instead of fitting it again for every mass/edge.
            original=float(model.signal(zero_check['mass_gev'],zero_check['profile_epsilon'])[config.categories.index(channel)].sum())
            local=float(model.signal(mass,epsilon)[config.categories.index(channel)].sum())
            if not math.isclose(original,local,rel_tol=1e-6):raise ValueError('Zero-background counting identity failed')
            target=zero_check['profile_q_at_native_epsilon']
            brackets=[(a,b) for a,b,qa,qb in zip(row['epsilon_grid'],row['epsilon_grid'][1:],row['q_grid'],row['q_grid'][1:]) if (qa-target)*(qb-target)<=0]
            a,b=min(brackets,key=lambda ab:abs(math.log(math.sqrt(ab[0]*ab[1])/epsilon)))
            corrected=math.exp(brentq(lambda loge:model.q(mass,math.exp(loge),channel,mode)-target,math.log(a),math.log(b),xtol=1e-9))
            result=dict(row,edge=edge,profile_epsilon=epsilon,epsilon=corrected,validated=True,
                equivalent_counting_reuse=True,expected_bands_valid=False,
                native_over_profile_rate=zero_check['native_over_profile_rate'],profile_q_at_native_epsilon=target,
                combine_output=zero_check['combine_output'],combine_output_sha256=zero_check['combine_output_sha256'],
                source='native_tea_AsymptoticLimits_exact_zero_count_reuse',
                count_identity={'source_selected_count':original,'local_selected_count':local},
                elapsed_seconds=time.monotonic()-started)
            results.append(result);save_json(destination,{'records':results,'snapshot_sha256':digest(model.output/'snapshot.json')})
            print('Reused exact native zero-background count',identity,flush=True);continue
        point='m'+format(mass,'g').replace('.','p')+'_edge'+str(edge)
        directory=model.output/'native_v2'/('cl'+str(round(level*100)))/mode/channel/point
        directory.mkdir(parents=True,exist_ok=True)
        categories=config.categories if channel=='combined' else (channel,)
        signal=model.signal(mass,epsilon);background=model.background if mode=='background' else np.zeros_like(model.background)
        sigma=model.cross_section(mass,epsilon);sg={};bg={};compression={}
        for category in categories:
            index=config.categories.index(category);s=signal[index];b=background[index]
            if mode=='zero_background':
                # With zero Asimov observations only the total selected signal
                # matters; merging mass cells is exactly likelihood-preserving.
                sg[category]=[float(s.sum())];bg[category]=[0.]
                compression[category]={'exact_zero_observation_counting_reduction':True}
            else:
                active=s>0;sg[category]=s[active].tolist();bg[category]=b[active].tolist()
                if float(b[~active].sum())>0:sg[category].append(0.);bg[category].append(float(b[~active].sum()))
                compression[category]={'signal_mass_bin_indices':np.flatnonzero(active).tolist(),'zero_signal_background_control_sum':float(b[~active].sum())}
        categories=tuple(c for c in categories if sum(sg[c])+sum(bg[c])>0)
        sg={c:sg[c] for c in categories};bg={c:bg[c] for c in categories}
        report={'schema':'shift-simplified-limit-yields-v1','luminosity_pb':config.luminosity,'categories':list(categories),
            'signals':{point:{'mass_gev':mass,'epsilon':epsilon,'sample_kind':'lifetime_interpolation','proper_ctau_mm':None,'mean_lab_flight_m':None,
                'reference_cross_section_pb':sigma,'yields':{c:sum(sg[c]) for c in categories},'binned_yields':sg}},
            'backgrounds':{'background_total':{'yields':{c:sum(bg[c]) for c in categories},'binned_yields':bg}},
            'compression':compression,'snapshot_sha256':digest(model.output/'snapshot.json')}
        save_json(directory/'yields.json',report)
        current=SimpleNamespace(**vars(config));current.categories=categories;current.category=categories[0];current.signal_points=(point,)
        current.output_path=str(directory)+'/'
        current.config_source_path=config.__file__;current.histogram_prefix='mass_';current.split_mass_bins=False
        current.yields_path=str(directory/'yields.json');current.prepared_input_manifest=str(directory/'inputs/manifest.json')
        current.datacards_output_path=str(directory/'cards')+'/'
        manifest=limits.prepare_inputs(current);cards=limits.create_cards(current,manifest,setup)
        card=next(c for c in cards if c['channel']==('combined' if channel=='combined' else categories[0]))
        scalar=poisson_card(current,report,manifest,card)
        fitdir=directory/'fit';fitdir.mkdir(exist_ok=True)
        if mode=='background':prepare_mass_auxiliary_card(scalar,fitdir/scalar.name)
        else:
            import shutil
            shutil.copy2(scalar,fitdir/scalar.name)
        fit=limits.category_config(current,manifest,current.category);fit.datacards_output_path=str(fitdir)+'/'
        # The parser needs this switch for the user's literal empty MC bins.
        # It changes input validation only; it adds no background or fit tuning.
        fit.combine_options=('--text2workspace=--X-allow-no-background',)
        if level!=.95:fit.combine_options+=('--cl',str(level))
        fit.combine_options_by_datacard={}
        limits.bind_fits(fit,manifest,[card],fitdir,binary)
        root=fitdir/('higgsCombine.'+card['name']+'.AsymptoticLimits.mH120.root')
        if not root.exists():limits_producer.run_combine(fit,[card['name']],method='AsymptoticLimits')
        quantiles=limits_producer.read_limit_quantiles(root)
        # The card's observations are the exact background-only Asimov counts,
        # including zero cells and the auxiliary measurement at zero. Its
        # observed CLs limit is therefore the expected Asimov sensitivity.
        # Keep Combine's internally generated bands for diagnosis: they are
        # not validated for this custom auxiliary representation.
        reference=card['combine_reference_cross_section_pb'];ratio=quantiles['observed']*reference/sigma
        # Convert the fitted rate back along the physical lifetime-dependent
        # model. A square-root rescaling would hold the reco response fixed.
        source_signal,source_background=model.cells(mass,epsilon,channel,mode)
        native_q=profile_q(ratio,source_signal,source_background)
        brackets=[(a,b) for a,b,qa,qb in zip(row['epsilon_grid'],row['epsilon_grid'][1:],row['q_grid'],row['q_grid'][1:]) if (qa-native_q)*(qb-native_q)<=0]
        if not brackets:raise RuntimeError('Native crossing lies outside the sampled physical response support')
        a,b=min(brackets,key=lambda ab:abs(math.log(math.sqrt(ab[0]*ab[1])/epsilon)))
        corrected_epsilon=math.exp(brentq(lambda loge:model.q(mass,math.exp(loge),channel,mode)-native_q,math.log(a),math.log(b),xtol=1e-9))
        q_after=model.q(mass,corrected_epsilon,channel,mode)
        threshold=norm.ppf((1+level)/2)**2
        # Near the checked crossing, native and exact-profile medians must agree.
        validated=bool(abs(ratio-1)<.10 and q_after is not None and abs(q_after/threshold-1)<.15)
        result=dict(row,edge=edge,profile_epsilon=epsilon,epsilon=corrected_epsilon,
            reference_cross_section_pb=sigma,combine_reference_cross_section_pb=reference,
            expected_limit_cross_section_pb=quantiles['observed']*reference,quantiles=quantiles,
            expected_median_statistic='observed_on_explicit_background_only_Asimov_card',expected_bands_valid=False,
            native_over_profile_rate=ratio,profile_q_at_native_epsilon=q_after,validated=validated,
            coupling_conversion='Map native median q onto the interpolated physical lifetime response',
            combine_output=str(root),combine_output_sha256=digest(root),elapsed_seconds=time.monotonic()-started,
            source='native_tea_AsymptoticLimits')
        if previous:results.remove(previous)
        results.append(result);save_json(destination,{'records':results,'snapshot_sha256':digest(model.output/'snapshot.json')})
        print('Native',identity,'epsilon',corrected_epsilon,'rate/profile',ratio,'validated',validated,'seconds',round(result['elapsed_seconds'],1),flush=True)
        if not validated:raise RuntimeError('Native boundary failed independent validation; inspect saved output before proceeding')
    print('Native stage complete:',len(results),'saved boundary fits.',flush=True)
    if not canary:
        subprocess.run(['python3','-u','-m','shift_limits_current_plots'],check=True)
        (model.output/'all_native.complete').write_text('Both-Both 95% Asimov CLs crossings completed using native TEA/Combine.\n')


def projection_inventory(config):
    import fitz
    from shift_paths import limits_output_path
    source=Path(limits_output_path).resolve();provenance=json.loads((source/'paper_figure_provenance.json').read_text());path=source/'arxiv_2511.03883v1_fig8_15.pdf'
    assert digest(path)==provenance['figure_sha256']
    left,top,right,bottom=provenance['axis_rectangle_points_top_left_origin'];curves=[]
    with fitz.open(path) as doc:
        for index,drawing in enumerate(doc[0].get_drawings()):
            if not drawing['color'] or drawing.get('fill') is not None:continue
            if len(drawing['items'])<5:continue
            points=[]
            for item in drawing['items']:
                if item[0]=='l':points.extend([(item[1].x,item[1].y),(item[2].x,item[2].y)])
                elif item[0]=='c':
                    p0,p1,p2,p3=item[1:]
                    for t in np.linspace(0,1,13):
                        p=(1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3;points.append((p.x,p.y))
            if not points:continue
            if not all(left-1<=x<=right+1 and top-1<=y<=bottom+1 for x,y in points):continue
            physical=[(10**(-3+6*(x-left)/(right-left)),10**(-10*(y-top)/(bottom-top))) for x,y in points]
            curves.append({'drawing_index':index,'color':drawing['color'],'dashes':drawing['dashes'],'width':drawing['width'],
                           'points':physical,'mass_range':[min(x for x,y in physical),max(x for x,y in physical)],
                           'epsilon_range':[min(y for x,y in physical),max(y for x,y in physical)]})
    save_json(Path(config.output_path).resolve()/'projection_inventory.json',{'source':provenance,'curves':curves})
    print('Projection paths decoded:',len(curves),flush=True)



def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['snapshot','scan','native_canary','native','projections'],required=True);parser.add_argument('--config',default='shift_limits_current_config')
    args=parser.parse_args();config=importlib.import_module(args.config)
    {'snapshot':snapshot,'scan':scan,'native_canary':lambda c:native(c,True),'native':native,'projections':projection_inventory}[args.stage](config)

if __name__=='__main__':main()
