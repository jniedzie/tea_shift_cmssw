#!/usr/bin/env python3
# Code style: AI-code. AI-generated audited cut-study diagnostic for limits.
"""Read missing joint RECO variables once; retain full-GEN normalization.

This compares fixed, modest cuts, without fitting or choosing an optimum.
Stored zero background entries remain finite-MC unknowns, never a claim of
background-free sensitivity. Reconstruction failures remain in the denominator.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from shift_limits import digest
from shift_limits_yields import event_category
from shift_limits_lifetime import save_json


def cut_mask(data, minimum_p, minimum_distance):
    mask = np.ones(len(data['mass']), dtype=bool)
    if minimum_p:
        mask &= np.isfinite(data['minimum_muon_p']) & (data['minimum_muon_p'] > minimum_p)
    if minimum_distance:
        mask &= np.isfinite(data['distance_m']) & (data['distance_m'] > minimum_distance)
        # A target-produced particle travels towards decreasing z.
        mask &= data['downstream_m'] > 0
    return mask


def cache_sample(name, sample, config):
    import ROOT
    source = sample['source']
    path = Path(config.output_path).resolve()/'data'/(name+'.npz')
    receipt = path.with_suffix('.json')
    binding = dict(audited_yields_sha256=digest(config.audited_yields_path),
                   source_file_stats=source['source_file_stats'],
                   nominal_target_cm=config.nominal_target_cm,
                   algorithm_sha256=digest(__file__))
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['binding'] != json.loads(json.dumps(binding)) or saved['sha256'] != digest(path):
            raise ValueError('Cut cache binding changed: '+name)
        with np.load(path, allow_pickle=False) as archive:
            return {key:archive[key].copy() for key in archive.files}
    rows = dict(ids=[],weight=[],mass=[],minimum_muon_p=[],distance_m=[],downstream_m=[])
    event_hash = hashlib.sha256()
    totals = dict(events=0,sumw=0.,sumw2=0.,both_both=0)
    started = time.monotonic()
    for number, metadata in enumerate(source['source_file_stats']):
        source_path = Path(metadata['path'])
        before = source_path.stat()
        if (before.st_size,before.st_mtime_ns)!=(metadata['bytes'],metadata['mtime_ns']):
            raise ValueError('Audited source changed: '+name)
        file = ROOT.TFile.Open(str(source_path),'READ')
        if not file or file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
            raise ValueError('Unreadable source: '+str(source_path))
        try:
            tree = file.Get('Events')
            branches = ('run','luminosityBlock','event','nShiftDimuonVertex','nShiftMuon',
                        'ShiftDimuonVertex_topologyMin','ShiftDimuonVertex_topologyMax',
                        'ShiftDimuonVertex_muonIdx1','ShiftDimuonVertex_muonIdx2',
                        'ShiftDimuonVertex_mass','ShiftDimuonVertex_vx','ShiftDimuonVertex_vy',
                        'ShiftDimuonVertex_vz','ShiftMuon_pt','ShiftMuon_pz',source['weight_branch'])
            tree.SetBranchStatus('*',0)
            for branch in branches:
                if not tree.GetBranch(branch):raise ValueError('Missing RECO variable '+branch)
                tree.SetBranchStatus(branch,1)
            for event in tree:
                identity = (int(event.run),int(event.luminosityBlock),int(event.event))
                event_hash.update(('%d:%d:%d\n'%identity).encode())
                weight = float(getattr(event,source['weight_branch']))
                if not math.isfinite(weight) or weight<0:raise ValueError('Invalid weight')
                totals['events'] += 1
                totals['sumw'] += weight
                totals['sumw2'] += weight*weight
                if not int(event.nShiftDimuonVertex):continue
                if int(event.nShiftDimuonVertex)>1:raise ValueError('Candidate arbitration needs a new audit')
                category = event_category(int(event.ShiftDimuonVertex_topologyMin[0]),
                                          int(event.ShiftDimuonVertex_topologyMax[0]))
                if category!='Both-Both':continue
                indices = [int(event.ShiftDimuonVertex_muonIdx1[0]),int(event.ShiftDimuonVertex_muonIdx2[0])]
                if indices[0]==indices[1] or any(i<0 or i>=int(event.nShiftMuon) for i in indices):
                    raise ValueError('Invalid associated reconstructed muon')
                momenta = [math.hypot(float(event.ShiftMuon_pt[i]),float(event.ShiftMuon_pz[i])) for i in indices]
                position = [float(getattr(event,'ShiftDimuonVertex_v'+axis)[0]) for axis in 'xyz']
                displacement = [value-origin for value,origin in zip(position,config.nominal_target_cm)]
                distance = math.sqrt(sum(value*value for value in displacement))/100.
                rows['ids'].append(identity)
                rows['weight'].append(weight)
                rows['mass'].append(float(event.ShiftDimuonVertex_mass[0]))
                rows['minimum_muon_p'].append(min(momenta))
                rows['distance_m'].append(distance)
                rows['downstream_m'].append(-displacement[2]/100.)
                totals['both_both'] += 1
        finally:
            file.Close()
        after = source_path.stat()
        if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
            raise ValueError('Input changed during read')
        if number%25==0:print(name,'files',number+1,'events',totals['events'],flush=True)
    audited = sample['counts']
    if totals['events']!=audited['events'] or event_hash.hexdigest()!=audited['event_ids_sha256']:
        raise ValueError('Full audited event population differs: '+name)
    for key in ('sumw','sumw2'):
        if not math.isclose(totals[key],audited[key],rel_tol=1.e-8,abs_tol=1.e-20):
            raise ValueError('Full weight denominator differs: '+name)
    expected = audited['categories']['Both-Both']
    if totals['both_both']!=expected['events'] or not math.isclose(sum(rows['weight']),expected['sumw'],rel_tol=1.e-8,abs_tol=1.e-20):
        raise ValueError('Both-Both baseline differs: '+name)
    arrays = {key:np.asarray(value,dtype=np.uint64 if key=='ids' else float) for key,value in rows.items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    save_json(receipt,dict(binding=binding,path=str(path),sha256=digest(path),
                           totals=totals,seconds=time.monotonic()-started))
    print('Cached',name,totals,'seconds',round(time.monotonic()-started,2),flush=True)
    return arrays


def prepare(config,canary=False):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    report = json.loads(Path(config.audited_yields_path).read_text())
    output = Path(config.output_path).resolve()
    inventory = [(kind,name,row) for kind in ('backgrounds','signals') for name,row in report[kind].items()
                 if kind=='backgrounds' or row['mass_gev'] in config.study_masses]
    if canary:
        # One complete, small sample proves branch semantics and the original
        # full-population/category normalization before the background scan.
        inventory = [row for row in inventory if row[1]=='m2_prompt']
    arrays = {}
    for kind,name,row in inventory:
        arrays[name] = cache_sample(name,row,config)
    if canary:
        print('Joint-variable canary passed; full count and identity digest reproduced.',flush=True)
        return
    scenarios = []
    for label,minimum_p,minimum_distance in config.cut_scenarios:
        selected = copy.deepcopy(report)
        selected['categories'] = ['Both-Both']
        selected['selection'] = dict(category='Both-Both',minimum_each_muon_p_gev=minimum_p,
            minimum_vertex_distance_from_nominal_target_m=minimum_distance,
            downstream_required=bool(minimum_distance),nominal_target_cm=config.nominal_target_cm,
            variables='RECO only; first stored dimuon, same as audited baseline',
            source_spread_caveat='Nominal target has sigma_z=0.5 m; distance is not a flight measurement.')
        summary = dict(name=label,minimum_muon_p_gev=minimum_p,minimum_distance_m=minimum_distance,
                       backgrounds={},signals={})
        for kind in ('backgrounds','signals'):
            selected[kind] = {}
            for row_kind,name,sample in inventory:
                if row_kind!=kind:continue
                data = arrays[name]
                mask = cut_mask(data,minimum_p,minimum_distance)
                weights = data['weight'][mask]
                sumw,sumw2 = float(sum(weights)),float(np.dot(weights,weights))
                factor = config.luminosity*sample['cross_section_pb']/sample['full_gen_sumw']
                row = copy.deepcopy(sample)
                row['yields'] = {'Both-Both':factor*sumw}
                row['mc_errors'] = {'Both-Both':factor*math.sqrt(sumw2)}
                row['cut_counts'] = dict(events=int(sum(mask)),sumw=sumw,sumw2=sumw2,
                                         ess=sumw**2/sumw2 if sumw2 else 0.)
                selected[kind][name] = row
                # Empty QCD cells are particularly important here: even one
                # ordinary-weight MC event can represent billions of events.
                floor = None
                counts = sample['counts']
                if not sum(mask) and counts['events'] and counts['sumw2']:
                    uniform = math.isclose(counts['sumw']**2/counts['sumw2'],counts['events'],rel_tol=1.e-8)
                    if uniform:
                        floor = factor*counts['sumw']*(-math.expm1(math.log(.1)/counts['events']))
                summary[kind][name] = dict(yield_events=factor*sumw,mc_error=factor*math.sqrt(sumw2),
                    selected_mc=int(sum(mask)),effective_mc=row['cut_counts']['ess'],
                    full_gen_sumw=sample['full_gen_sumw'],
                    produced_events=config.luminosity*sample['cross_section_pb'],
                    acceptance=sumw/sample['full_gen_sumw'],
                    single_component_empty_mc_90_upper_events=floor)
        selected['background_totals'] = {'Both-Both':sum(row['yields']['Both-Both'] for row in selected['backgrounds'].values())}
        selected['assumptions'] += ['Fixed RECO cuts; finite MC uncertainties remain diagnostic only.',
            'Zero selected background MC is unknown, not evidence of zero background.']
        summary['total_background'] = selected['background_totals']['Both-Both']
        summary['empty_background_components'] = [name for name,row in summary['backgrounds'].items() if not row['selected_mc']]
        save_json(output/label/'yields.json',selected)
        scenarios.append(summary)
        print(label,'B',summary['total_background'],'empty MC components',len(summary['empty_background_components']),flush=True)
    save_json(output/'cut_comparison.json',dict(complete=True,luminosity_pb=config.luminosity,
        audited_yields_sha256=digest(config.audited_yields_path),scenarios=scenarios,
        no_fits_run=True,selection_is_not_optimized=True))
    plot(config,report,scenarios)


def plot(config,report,scenarios):
    from shift_limits_lifetime import plotting,finish
    plt = plotting()
    output = Path(config.output_path).resolve()
    fig,axes = plt.subplots(1,2,figsize=(11,5.5))
    labels = [row['name'] for row in scenarios]
    axes[0].bar(labels,[row['total_background'] for row in scenarios],color='#708da5')
    axes[0].set(yscale='log',ylabel='Expected background events at 3 fb$^{-1}$',title='Both–Both: nominal background after cuts')
    for mass in config.study_masses:
        name=f'm{mass:g}_prompt'
        initial=scenarios[0]['signals'][name]['yield_events']
        axes[1].plot(labels,[row['signals'][name]['yield_events']/initial if initial else np.nan for row in scenarios],marker='o',label=f'{mass:g} GeV, prompt')
        name=f'm{mass:g}_lab_100m'
        initial=scenarios[0]['signals'][name]['yield_events']
        axes[1].plot(labels,[row['signals'][name]['yield_events']/initial if initial else np.nan for row in scenarios],marker='s',linestyle='--',label=f'{mass:g} GeV, mean flight 100 m')
    axes[1].set(ylabel='Retention relative to Both–Both baseline',ylim=(0,1.08),title='Signal response depends on lifetime')
    axes[1].legend(fontsize=8)
    fig.suptitle('SHIFT: fixed momentum and displacement cuts')
    finish(fig,output,'cut_response',
        'p20: each associated RECO muon p > 20 GeV; L1/L10: RECO vertex distance > 1/10 m from nominal target, downstream.\n'
        'No classifier; source longitudinal spread 0.5 m. Nominal finite-MC yields; empty QCD cells remain unknown.')


def main():
    import importlib
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='shift_limits_cuts_config')
    parser.add_argument('--canary',action='store_true')
    args=parser.parse_args()
    prepare(importlib.import_module(args.config),args.canary)


if __name__=='__main__':main()
