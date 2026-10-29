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
from shift_limits_profile import profile_q,profile_q_mc


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


def signal_normalization(authority, preserved=None):
    """Keep original GEN exposure; sampled cutflows already contain W/p."""
    sigma=math.fsum(row['value_pb'] for row in authority['physical_selected_cross_sections_pb'])
    denominator=float(authority['full_gen']['sum_weights'])
    if preserved is not None:
        if (not math.isclose(sigma,preserved['cross_section_pb'],rel_tol=2e-6)
                or not math.isclose(denominator,preserved['initial_weight_sum'],rel_tol=2e-6)):
            raise ValueError('Histogram and rate sidecar disagree on original normalization')
        sigma=preserved['cross_section_pb'];denominator=preserved['initial_weight_sum']
    if not math.isfinite(sigma+denominator) or sigma<=0 or denominator<=0:
        raise ValueError('Invalid original signal cross section or GEN exposure')
    return sigma,denominator


def signal_inputs(config):
    from shift_sample_paths import available_bsm_campaign_histograms
    inventory=getattr(config,'signal_input_inventory',None)
    if inventory:
        rows=json.loads(Path(inventory).read_text())['signals']
        if len({row['name'] for row in rows})!=len(rows):
            raise ValueError('Signal inventory repeats a physical lifetime point')
        return rows
    return available_bsm_campaign_histograms(config.base_path,config.signal_campaign,
        config.signal_masses,config.signal_flights,epsilons=config.signal_epsilons)


def snapshot(config):
    import ROOT
    from shift_plot_samples import prepare_plot_samples
    from shift_sample_paths import histogram_gen_normalization
    ROOT.gROOT.SetBatch(True)
    output=Path(config.output_path).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'snapshot.json').exists():
        raise ValueError('A frozen snapshot already exists; reuse it instead of repeating input reads')
    backgrounds,_=prepare_plot_samples(config.base_path,config.cross_sections,config.campaign,None,None,None,None,None)
    points=signal_inputs(config)
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
                    preserved=histogram_gen_normalization(str(path),'darkphoton_'+name)
                    sigma,denominator=signal_normalization(authority,preserved)
                    source.update(mass_gev=authority['mass_gev'],epsilon=authority['epsilon'],proper_ctau_mm=authority['proper_ctau_mm'],
                        mechanism=authority['mechanism'],prompt=name.endswith('_prompt'),rate_authority=str(authority_path),rate_authority_sha256=digest(authority_path),
                        physics_valid=authority.get('physics_valid'),normalization_ready=authority.get('normalization_ready'),
                        normalization_source='histogram_full_GEN' if preserved else 'published_full_GEN_sidecar',
                        mean_lab_flight_m=item.get('mean_lab_flight_m'),input_role=item.get('input_role','current'),
                        production_rate_validated=authority.get('production_rate_validated'))
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
        self.mc_statistics=bool(getattr(config,'include_mc_stat_uncertainties',False))
        self.background_components=[];self.background_variances=[]
        self.signals={}
        for row in self.snapshot['backgrounds']:
            scale=config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
            value=self.arrays[row['array_key']]*scale
            self.background+=value;self.processes[row['name'].split('_')[0]]+=value
            variance_key=row['array_key']+'_sumw2'
            if self.mc_statistics and variance_key not in self.arrays:raise ValueError('Missing original MC variance: '+row['name'])
            self.background_components.append(value)
            self.background_variances.append(self.arrays.get(variance_key,np.zeros_like(value))*scale**2)
        self.background_components=np.asarray(self.background_components).reshape((-1,)+self.background.shape)
        self.background_variances=np.asarray(self.background_variances).reshape((-1,)+self.background.shape)
        self.unused_signals=[]
        for row in self.snapshot['signals']:
            if not any(row['name'].endswith('_'+anchor) for anchor in config.response_anchors):
                self.unused_signals.append(dict(row,unused_reason='Not a configured lifetime anchor'))
                continue
            channel=getattr(config,'primary_channel','Both-Both')
            values=self.arrays[row['array_key']]
            selected=values.sum() if channel=='combined' else values[config.categories.index(channel)].sum()
            if selected<=0:
                self.unused_signals.append(dict(row,unused_reason='Zero observed selected MC; response remains unknown'))
                continue
            self.signals.setdefault(row['mass_gev'],[]).append(row)
        for rows in self.signals.values():rows.sort(key=lambda r:r['epsilon'])
        for mass,rows in self.signals.items():
            if len({row['mechanism'] for row in rows})!=1:
                raise ValueError('Different signal production prescriptions at one mass: '+str(mass))

    def signal_moments(self,mass,epsilon):
        rows=self.signals[mass]
        if epsilon<rows[0]['epsilon'] and math.isclose(epsilon,rows[0]['epsilon'],rel_tol=1e-12):epsilon=rows[0]['epsilon']
        if epsilon>rows[-1]['epsilon'] and math.isclose(epsilon,rows[-1]['epsilon'],rel_tol=1e-12):epsilon=rows[-1]['epsilon']
        x=math.log(epsilon)
        def coefficient(row):
            scale=self.config.luminosity*row['cross_section_pb']/(row['full_gen_sumw']*row['epsilon']**2)
            key=row['array_key'];variance_key=key+'_sumw2'
            if self.mc_statistics and variance_key not in self.arrays:raise ValueError('Missing original MC variance: '+row['name'])
            return self.arrays[key]*scale,self.arrays.get(variance_key,np.zeros_like(self.arrays[key]))*scale**2
        if epsilon<rows[0]['epsilon']:return None,None
        if epsilon>rows[-1]['epsilon']:
            if not rows[-1]['prompt']:return None,None
            value,variance=coefficient(rows[-1]);return value*epsilon**2,variance*epsilon**4
        for a,b in zip(rows,rows[1:]):
            if a['epsilon']<=epsilon<=b['epsilon']:
                if a['epsilon']==b['epsilon']:raise ValueError('Same-epsilon mechanisms need explicit rate-summed populations')
                fraction=(x-math.log(a['epsilon']))/math.log(b['epsilon']/a['epsilon'])
                assert -1e-12<=fraction<=1+1e-12
                fraction=min(1.,max(0.,fraction))
                av,avar=coefficient(a);bv,bvar=coefficient(b)
                # Endpoint samples are independent populations. Hypotheses
                # along the interpolated curve remain mutually correlated.
                return ((1-fraction)*av+fraction*bv)*epsilon**2,((1-fraction)**2*avar+fraction**2*bvar)*epsilon**4
        value,variance=coefficient(rows[0]);return value*epsilon**2,variance*epsilon**4

    def signal(self,mass,epsilon):
        return self.signal_moments(mass,epsilon)[0]

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
        if self.mc_statistics:
            signal,variance,backgrounds,background_variances,_=self.mc_cells(mass,epsilon,channel,mode)
            if signal is None:return None
            if not np.any(signal):return 0.
            return profile_q_mc(signal,variance,backgrounds,background_variances,
                                allow_zero_background=mode=='zero_background')
        signal,background=self.cells(mass,epsilon,channel,mode)
        if signal is None:return None
        if not np.any(signal):return 0.
        if mode=='zero_background':
            total=float(signal.sum());k=math.log(1.3);theta=-float(lambertw(k*k*total).real)/k
            return 2*total*math.exp(k*theta)+theta*theta
        return profile_q(1.,signal,background)

    def mc_cells(self,mass,epsilon,channel,mode):
        signal,variance=self.signal_moments(mass,epsilon)
        if signal is None:return None,None,None,None,None
        backgrounds=self.background_components
        background_variances=self.background_variances
        if mode=='zero_background':
            backgrounds=np.zeros((1,)+signal.shape)
            background_variances=np.zeros_like(backgrounds)
        if channel!='combined':
            index=self.config.categories.index(channel)
            signal=signal[index:index+1];variance=variance[index:index+1]
            backgrounds=backgrounds[:,index:index+1];background_variances=background_variances[:,index:index+1]
        signal=signal.ravel();variance=variance.ravel()
        backgrounds=backgrounds.reshape(len(backgrounds),signal.size)
        background_variances=background_variances.reshape(len(backgrounds),signal.size)
        known=np.ones_like(signal,dtype=bool) if mode=='zero_background' else backgrounds.sum(axis=0)>0
        report={'omitted_total_zero_background_mc_bins':int((~known).sum()),
                'omitted_signal_yield':float(signal[~known].sum()),
                'total_signal_yield':float(signal.sum()),
                'known_zero_background_scenario':mode=='zero_background',
                'limitation':'Positive total MC does not bound contributions from individually empty process bins.'}
        return signal[known],variance[known],backgrounds[:,known],background_variances[:,known],report


def scan(config):
    model=Model(config);records=[]
    metadata={'snapshot_sha256':digest(model.output/'snapshot.json'),
              'include_mc_stat_uncertainties':model.mc_statistics}
    path=model.output/'scan.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if bool(saved.get('include_mc_stat_uncertainties',False))!=model.mc_statistics:
            raise ValueError('Completed scan uses a different finite-MC mode; use a separate output directory')
        raise ValueError('A completed scan exists; reuse it')
    checkpoint=model.output/'scan.checkpoint.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        assert saved['snapshot_sha256']==metadata['snapshot_sha256']
        if bool(saved.get('include_mc_stat_uncertainties',False))!=model.mc_statistics:
            raise ValueError('Scan checkpoint uses a different finite-MC mode; use a separate output directory')
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
                        q_grid=values,epsilon_grid=grid,include_mc_stat_uncertainties=model.mc_statistics,
                        source='independent_effective_poisson_mc_profile' if model.mc_statistics else 'independent_exact_asimov_profile'))
                print('Scanned',mass,mode,channel,flush=True)
        save_json(checkpoint,dict(records=records,**metadata))
    save_json(path,dict(records=records,**metadata))


def native(config,canary=False):
    if getattr(config,'include_mc_stat_uncertainties',False):
        raise ValueError('The legacy native cards omit finite-MC constraints; use the dedicated mc_statistics comparison and validated controls')
    if json.loads((Path(config.output_path)/'scan.json').read_text()).get('include_mc_stat_uncertainties',False):
        raise ValueError('The legacy native cards cannot consume a finite-MC scan; use the dedicated mc_statistics comparison')
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
        if (getattr(config,'native_validation_masses',None) is not None
                and row['mass_gev'] not in config.native_validation_masses):continue
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
        subprocess.run(['python3','-u','-m','shift_limits_current_plots','--output',str(model.output),
                        '--luminosity-pb',str(config.luminosity)],check=True)
        name='native_validation.complete' if getattr(config,'native_validation_masses',None) else 'all_native.complete'
        (model.output/name).write_text('Selected Both-Both Asimov CLs crossings checked using native TEA/Combine; see native_results.json.\n')


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


def mc_statistics_study(config,masses=(.7,2.,5.)):
    """Compare the frozen nominal result with a declared finite-MC variant."""
    import time
    from matplotlib.backends.backend_pdf import PdfPages
    from shift_limits_lifetime import plotting
    started=time.monotonic()
    nominal_config=SimpleNamespace(**vars(config));nominal_config.include_mc_stat_uncertainties=False
    mc_config=SimpleNamespace(**vars(config));mc_config.include_mc_stat_uncertainties=True
    nominal=Model(nominal_config);finite=Model(mc_config)
    output=nominal.output/'mc_statistics';output.mkdir(exist_ok=True)
    if (output/'comparison.json').exists():raise ValueError('Reuse the completed frozen MC-statistics comparison')
    original=json.loads((nominal.output/'scan.json').read_text())
    if original.get('include_mc_stat_uncertainties',False):raise ValueError('The MC comparison requires a nominal baseline scan')
    threshold=float(norm.ppf(.975)**2);records=[];original=original['records']
    for mass in masses:
        anchors=nominal.signals[mass]
        low=anchors[0]['epsilon'];high=min(config.small_mixing_max,config.epsilon_max) if anchors[-1]['prompt'] else anchors[-1]['epsilon']
        grid=sorted(set(np.geomspace(low,high,config.scan_grid_points).tolist()+[r['epsilon'] for r in anchors if low<=r['epsilon']<=high]))
        for mode in config.background_modes:
            nominal_values=[nominal.q(mass,e,config.primary_channel,mode) for e in grid]
            mc_values=[finite.q(mass,e,config.primary_channel,mode) for e in grid]
            roots=[]
            for a,b,qa,qb in zip(grid,grid[1:],mc_values,mc_values[1:]):
                if (qa-threshold)*(qb-threshold)<0:
                    roots.append(math.exp(brentq(lambda x:finite.q(mass,math.exp(x),config.primary_channel,mode)-threshold,math.log(a),math.log(b),xtol=1.e-9)))
            bounds=[low]+roots+[high]
            intervals=[[a,b] for a,b in zip(bounds,bounds[1:]) if finite.q(mass,math.sqrt(a*b),config.primary_channel,mode)>threshold]
            previous=next(row for row in original if row['mass_gev']==mass and row['mode']==mode and row['channel']==config.primary_channel and row['confidence_level']==.95)
            comparisons=[]
            for epsilon in previous['roots']:
                if not low<=epsilon<=high:continue
                s,v,b,bv,coverage=finite.mc_cells(mass,epsilon,config.primary_channel,mode)
                q_signal=profile_q_mc(s,v,b,bv*0,allow_zero_background=mode=='zero_background')
                q_background=profile_q_mc(s,v*0,b,bv,allow_zero_background=mode=='zero_background')
                comparisons.append(dict(epsilon=epsilon,nominal_q=nominal.q(mass,epsilon,config.primary_channel,mode),
                    nominal_q_on_positive_background_support=profile_q(1.,s,b.sum(axis=0)),
                    signal_mc_only_q=q_signal,background_mc_only_q=q_background,both_mc_q=finite.q(mass,epsilon,config.primary_channel,mode),**coverage))
            records.append(dict(mass_gev=mass,mode=mode,channel=config.primary_channel,confidence_level=.95,epsilon_grid=grid,
                nominal_q_grid=nominal_values,mc_q_grid=mc_values,nominal_roots=previous['roots'],mc_roots=roots,
                mc_excluded_intervals=intervals,epsilon_support=[low,high],at_nominal_boundaries=comparisons))
            print('MC comparison',mass,mode,'roots',roots,flush=True)
    report=dict(schema='shift-finite-mc-comparison-v1',luminosity_pb=config.luminosity,confidence_level=.95,
        channel=config.primary_channel,snapshot_path=str(nominal.output/'snapshot.json'),snapshot_sha256=digest(nominal.output/'snapshot.json'),
        histograms_sha256=nominal.snapshot['histogram_cache_sha256'],
        source_hashes={name:digest(Path(__file__).parent/name) for name in ['shift_limits_current.py','shift_limits_profile.py']},
        method='Independent per-source/per-bin effective-Poisson constraints: n_eff=sumw^2/sumw2; deviance2*n_eff*(beta-1-log(beta)); two unchanged30percent global lognormal nuisances.',
        source_urls=['https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/bin-wise-stats/'],
        limitations=['Moment-matched approximation for positive weights, not exact autoMCStats or an exact weighted-event likelihood.',
            'Independent endpoint MC populations assumed. Interpolated hypotheses along epsilon remain correlated; no simultaneous contour coverage claim.',
            'Bins with zero TOTAL background MC are omitted in the background scenario. Individually empty source bins remain unbounded, even where other sources are populated.',
            'Asymptotic CLs proxy retained; finite-MC constraints alone do not validate sparse-count coverage.',
            'No smoothing, invented background, response extrapolation or selection changes. Baseline published limits remain unchanged.'],
        records=records,elapsed_seconds=time.monotonic()-started)
    save_json(output/'comparison.json',report)
    plt=plotting()
    with PdfPages(output/'finite_mc_comparison.pdf') as pdf:
        for mode in config.background_modes:
            fig,axes=plt.subplots(1,len(masses),figsize=(12,4.2),squeeze=False)
            for axis,mass in zip(axes[0],masses):
                row=next(r for r in records if r['mass_gev']==mass and r['mode']==mode)
                axis.plot(row['epsilon_grid'],row['nominal_q_grid'],color='black',label='Nominal MC templates')
                axis.plot(row['epsilon_grid'],row['mc_q_grid'],color='tab:blue',label='Finite MC constraints')
                axis.axhline(threshold,color='tab:red',linestyle='--',label='95% Asymptotic CLs threshold')
                axis.set(xscale='log',yscale='log',ylim=(1.e-4,1.e3),xlabel='Kinetic mixing epsilon',ylabel='Profile likelihood q',title=f"{mass:g} GeV")
                axis.grid(alpha=.2)
            axes[0,0].legend(fontsize=7)
            fig.suptitle(f"Both-Both, {config.luminosity/1000:g} fb$^{{-1}}$, "+('SM background' if mode=='background' else 'ideal zero background'))
            fig.text(.5,.01,'Positive-weight effective-Poisson approximation; empty total-SM MC bins omitted only in background case.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,.93));pdf.savefig(fig);plt.close(fig)
    print('MC comparison retained in',output,flush=True)


def sm_statistics_study(config,masses=(.7,2.,5.)):
    """Rank existing SM MC limitations; forecast unchanged-mean MC top-ups."""
    import csv
    import time
    from matplotlib.backends.backend_pdf import PdfPages
    from shift_limits_lifetime import plotting
    started=time.monotonic();current=SimpleNamespace(**vars(config));current.include_mc_stat_uncertainties=True
    model=Model(current);output=model.output/'mc_statistics';output.mkdir(exist_ok=True)
    destination=output/'sm_precision.json'
    if destination.exists():raise ValueError('Reuse the completed frozen SM-statistics diagnosis')
    nominal=json.loads((model.output/'scan.json').read_text())
    if nominal.get('include_mc_stat_uncertainties',False):raise ValueError('SM-statistics diagnosis requires the nominal baseline scan')
    comparison=json.loads((output/'comparison.json').read_text())
    if comparison['snapshot_sha256']!=digest(model.output/'snapshot.json'):raise ValueError('Finite-MC comparison uses a different snapshot')
    channel=config.primary_channel;category=config.categories.index(channel);edges=model.snapshot['mass_edges_gev']
    original_bins=np.flatnonzero(model.background[category]>0)
    source_records=[];bin_records=[];source_bin_records=[];cases=[];scenarios=[];current_priorities=[]
    threshold=float(norm.ppf(.975)**2)

    def q_with_exposure(mass,epsilon,multiplier):
        s,v,b,bv,_=model.mc_cells(mass,epsilon,channel,'background')
        return profile_q_mc(s,v,b,bv/multiplier)

    for mass in masses:
        old=next(r for r in nominal['records'] if r['mass_gev']==mass and r['mode']=='background'
                 and r['channel']==channel and r['confidence_level']==.95)
        epsilon=old['roots'][0];s,v,b,bv,coverage=model.mc_cells(mass,epsilon,channel,'background')
        base=profile_q_mc(s,v,b,bv);case=dict(mass_gev=mass,reference_epsilon=epsilon,
            reference='First nominal95percent background boundary',finite_mc_q=base,**coverage)
        case['q_after_all_sm_variance_reduction']={str(factor):profile_q_mc(s,v,b,bv/factor) for factor in (10.,100.)}
        case['q_with_known_sm_and_fixed_signal_mc']=profile_q_mc(s,v,b,bv*0)
        cases.append(case)
        for k,row in enumerate(model.snapshot['backgrounds']):
            if not np.any(bv[k]):continue
            result=dict(mass_gev=mass,reference_epsilon=epsilon,source=row['name'],baseline_q=base)
            for factor in (10.,100.):
                changed=bv.copy();changed[k]/=factor;q=profile_q_mc(s,v,b,changed)
                result['q_if_source_variance_divided_by_'+str(int(factor))]=q
                result['q_gain_for_source_exposure_x'+str(int(factor))]=q-base
            source_records.append(result)
        for j in np.flatnonzero(s>0):
            ib=int(original_bins[j]);total_b=float(b[:,j].sum());total_v=float(bv[:,j].sum())
            result=dict(mass_gev=mass,reference_epsilon=epsilon,original_bin=ib,
                reco_mass_low_gev=edges[ib],reco_mass_high_gev=edges[ib+1],signal_yield=float(s[j]),
                background_yield=total_b,background_mc_variance=total_v,
                background_effective_mc_events=total_b**2/total_v if total_v else None,baseline_q=base)
            changed=bv.copy();changed[:,j]/=10.;q=profile_q_mc(s,v,b,changed)
            result.update(q_if_bin_sm_variance_divided_by_10=q,q_gain_for_bin_sm_exposure_x10=q-base)
            bin_records.append(result)
            for k,row in enumerate(model.snapshot['backgrounds']):
                if bv[k,j]<=0:continue
                raw=float(model.arrays[row['array_key']][category,ib]);raw_variance=float(model.arrays[row['array_key']+'_sumw2'][category,ib])
                scale=config.luminosity*row['cross_section_pb']/row['full_gen_sumw']
                changed=bv.copy();changed[k,j]/=10.;q=profile_q_mc(s,v,b,changed)
                source_bin_records.append(dict(mass_gev=mass,reference_epsilon=epsilon,source=row['name'],original_bin=ib,
                    reco_mass_low_gev=edges[ib],reco_mass_high_gev=edges[ib+1],signal_yield=float(s[j]),
                    source_raw_sumw=raw,source_raw_sumw2=raw_variance,source_effective_mc_events=raw**2/raw_variance,
                    source_expected_yield=float(b[k,j]),source_expected_mc_variance=float(bv[k,j]),
                    source_fraction_of_bin_mc_variance=float(bv[k,j]/total_v),
                    effective_native_weight=raw_variance/raw,expected_events_per_native_weight=scale,
                    expected_events_per_effective_mc_event=float(bv[k,j]/b[k,j]),
                    source_full_gen_sumw=row['full_gen_sumw'],source_histogram_sha256=row['sha256'],
                    q_if_source_bin_variance_divided_by_10=q,q_gain_for_source_bin_exposure_x10=q-base))
        reference=next(r for r in comparison['records'] if r['mass_gev']==mass and r['mode']=='background')
        grid=reference['epsilon_grid']
        for factor in (1.,10.,100.):
            values=[q_with_exposure(mass,e,factor) for e in grid];roots=[]
            for a,z,qa,qz in zip(grid,grid[1:],values,values[1:]):
                if (qa-threshold)*(qz-threshold)<0:
                    roots.append(math.exp(brentq(lambda x:q_with_exposure(mass,math.exp(x),factor)-threshold,math.log(a),math.log(z),xtol=1.e-9)))
            bounds=[grid[0]]+roots+[grid[-1]]
            intervals=[[a,z] for a,z in zip(bounds,bounds[1:]) if q_with_exposure(mass,math.sqrt(a*z),factor)>threshold]
            scenarios.append(dict(mass_gev=mass,sm_mc_exposure_multiplier=factor,epsilon_grid=grid,q_grid=values,
                roots=roots,excluded_intervals=intervals,epsilon_support=[grid[0],grid[-1]]))
            print('SM precision',mass,'exposure x',factor,'roots',roots,flush=True)
        present=next(r for r in scenarios if r['mass_gev']==mass and r['sm_mc_exposure_multiplier']==1.)
        present_epsilon=present['roots'][0] if present['roots'] else grid[-1]
        ps,pv,pb,pbv,_=model.mc_cells(mass,present_epsilon,channel,'background')
        present_q=profile_q_mc(ps,pv,pb,pbv);priorities=[];cells=[]
        for k,row in enumerate(model.snapshot['backgrounds']):
            changed=pbv.copy();changed[k]/=10.
            priorities.append(dict(source=row['name'],q_gain_if_source_mc_exposure_x10=profile_q_mc(ps,pv,pb,changed)-present_q))
        for j in np.flatnonzero(ps>0):
            changed=pbv.copy();changed[:,j]/=10.;ib=int(original_bins[j])
            cells.append(dict(original_bin=ib,reco_mass_gev=edges[ib:ib+2],
                q_gain_if_bin_sm_mc_exposure_x10=profile_q_mc(ps,pv,pb,changed)-present_q))
        priorities.sort(key=lambda r:r['q_gain_if_source_mc_exposure_x10'],reverse=True)
        cells.sort(key=lambda r:r['q_gain_if_bin_sm_mc_exposure_x10'],reverse=True)
        j=int(np.flatnonzero(original_bins==cells[0]['original_bin'])[0]);moments=[]
        for k,row in enumerate(model.snapshot['backgrounds']):
            if pb[k,j]<=0:continue
            moments.append(dict(source=row['name'],yield_expected=float(pb[k,j]),variance=float(pbv[k,j]),
                effective_mc_events=float(pb[k,j]**2/pbv[k,j]),expected_effective_event_weight=float(pbv[k,j]/pb[k,j])))
        moments.sort(key=lambda r:r['variance'],reverse=True)
        current_priorities.append(dict(mass_gev=mass,epsilon=present_epsilon,baseline_q=present_q,
            reference='Current finite-MC95percent boundary' if present['roots'] else 'Upper scan support; no exclusion',
            sources=priorities[:5],bins=cells[:3],dominant_bin_source_moments=moments[:4]))
    source_records.sort(key=lambda r:(r['mass_gev'],-r['q_gain_for_source_exposure_x10']))
    bin_records.sort(key=lambda r:(r['mass_gev'],-r['q_gain_for_bin_sm_exposure_x10']))
    source_bin_records.sort(key=lambda r:(r['mass_gev'],-r['q_gain_for_source_bin_exposure_x10']))
    for name,rows in [('sm_precision_sources.csv',source_records),('sm_precision_bins.csv',bin_records),('sm_precision_source_bins.csv',source_bin_records)]:
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(schema='shift-sm-mc-precision-diagnosis-v1',snapshot_sha256=digest(model.output/'snapshot.json'),
        histogram_cache_sha256=model.snapshot['histogram_cache_sha256'],luminosity_pb=config.luminosity,channel=channel,
        background_campaign=model.snapshot['background_campaign'],
        source_hashes={name:digest(Path(__file__).parent/name) for name in ['shift_limits_current.py','shift_limits_profile.py']},
        counterfactual='Central SM yields and all signal MC moments fixed; divide selected SM variances by10 or100. Equivalent to independent increased MC exposure only if its weight/selection distribution remains the same.',
        ranking='Exact profiled q improvement at each first nominal95percent boundary. Source/bin gains are not additive. Full nuisance fit is repeated for every counterfactual.',
        weight_interpretation='Effective MC events and sumw/sumw2 are reported. These do not determine literal event entries. Effective native weight=sumw2/sumw; expected effective-event weight=variance/yield after luminosity normalization.',
        limitations=comparison['limitations'],cases=cases,scenarios=scenarios,current_boundary_source_priorities=current_priorities,
        priority_caveat='CSV ranks diagnose lost nominal boundaries. Current-boundary priorities are separately recorded for allocation after finite-MC changes the boundary. Check newer completed SM before top-ups.',
        csv_files=['sm_precision_sources.csv','sm_precision_bins.csv','sm_precision_source_bins.csv'],elapsed_seconds=time.monotonic()-started)
    save_json(destination,report)
    plt=plotting()
    with PdfPages(output/'sm_precision_priorities.pdf') as pdf:
        fig,axes=plt.subplots(2,len(masses),figsize=(12,7.3),squeeze=False)
        for column,mass in enumerate(masses):
            axis=axes[0,column]
            for factor,color in [(1.,'black'),(10.,'tab:blue'),(100.,'tab:orange')]:
                row=next(r for r in scenarios if r['mass_gev']==mass and r['sm_mc_exposure_multiplier']==factor)
                axis.plot(row['epsilon_grid'],row['q_grid'],color=color,label=f'SM MC exposure x{factor:g}')
            axis.axhline(threshold,color='tab:red',linestyle='--');axis.set(xscale='log',yscale='log',ylim=(1.e-4,1.e3),title=f'{mass:g} GeV',xlabel='Kinetic mixing epsilon',ylabel='Profile q')
            axis.grid(alpha=.2)
            if column==0:axis.legend(fontsize=7)
            axis=axes[1,column];rows=[r for r in source_records if r['mass_gev']==mass][:5]
            axis.barh([r['source'] for r in reversed(rows)],[r['q_gain_for_source_exposure_x10'] for r in reversed(rows)],color='tab:blue')
            axis.set(xlabel='q gain if this source has x10 MC exposure',title='Priority at old first95% boundary');axis.tick_params(axis='y',labelsize=8);axis.grid(axis='x',alpha=.2)
        fig.suptitle(f'SM simulation statistics: Both-Both, {config.luminosity/1000:g} fb$^{{-1}}$')
        fig.text(.5,.01,'Counterfactual forecasts: SM means and signal MC fixed; independent MC with unchanged weights assumed. Not achieved statistics.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.95));pdf.savefig(fig);plt.close(fig)
    print('SM precision diagnosis retained in',output,flush=True)



def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['snapshot','scan','native_canary','native','projections','mc_statistics','sm_statistics'],required=True);parser.add_argument('--config',default='shift_limits_current_config')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--luminosity-pb',type=float)
    parser.add_argument('--signal-inventory',type=Path)
    parser.add_argument('--channel',choices=['Both-Both','combined'])
    parser.add_argument('--native-masses',nargs='+',type=float)
    args=parser.parse_args();config=importlib.import_module(args.config)
    if args.output is not None:config.output_path=str(args.output.resolve())
    if args.luminosity_pb is not None:config.luminosity=args.luminosity_pb
    if args.signal_inventory is not None:config.signal_input_inventory=str(args.signal_inventory.resolve())
    if args.channel is not None:config.channel_names=(args.channel,);config.primary_channel=args.channel
    if args.native_masses is not None:config.native_validation_masses=args.native_masses
    {'snapshot':snapshot,'scan':scan,'native_canary':lambda c:native(c,True),'native':native,'projections':projection_inventory,'mc_statistics':mc_statistics_study,'sm_statistics':sm_statistics_study}[args.stage](config)

if __name__=='__main__':main()
