#!/usr/bin/env python3
# Code style: AI-code. AI-generated sensitivity ingredient and candidate audit.
"""Diagnose background rejection using existing, unmodified Nano reconstruction.

Only already audited Both-Both candidates are read from old background Nano
files. Full-GEN rate authorities remain fixed. Truth is diagnostic only.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from shift_limits import digest
from shift_limits_lifetime import save_json


PAIR_FIELDS = ('mass','constrainedMass','constrainedValid','normalizedChi2',
    'originCompatibilityNormalizedChi2','probability','constrainedNormalizedChi2',
    'constrainedOriginCompatibilityNormalizedChi2','refitStatus','refittedMass',
    'vx','vy','vz','vzErr','muonIdx1','muonIdx2','topologyMin','topologyMax')
MUON_FIELDS = ('pt','pz','phi','charge','hitGenPartIdx','hitTruthPurity',
    'constrainedValid','constrainedPt','constrainedPz','constrainedPhi')
GEN_FIELDS = ('pt','pz','phi','mass','pdgId','genPartIdxMother','vx','vy','vz')
SELECTION = 'Original first Both-Both candidate; no truth selection'


def candidate(event, available):
    def value(prefix,field,index):
        branch=prefix+'_'+field
        if branch not in available:return None
        result=float(getattr(event,branch)[index])
        return result if math.isfinite(result) else None
    row={field:value('ShiftDimuonVertex',field,0) for field in PAIR_FIELDS}
    indices=[int(row['muonIdx1']),int(row['muonIdx2'])]
    row['muons']=[{field:value('ShiftMuon',field,i) for field in MUON_FIELDS} for i in indices]
    row['minimum_p']=min(math.hypot(m['pt'],m['pz']) for m in row['muons'])
    row['distance_m']=math.sqrt(row['vx']**2+row['vy']**2+(row['vz']-14800.)**2)/100.
    matched=[];mothers=[]
    for muon in row['muons']:
        index=muon['hitGenPartIdx']
        if index is None or index<0 or index>=int(event.nGenPart):
            matched.append(None);mothers.append([]);continue
        index=int(index)
        gen={field:value('GenPart',field,index) for field in GEN_FIELDS}
        matched.append(gen);ancestors=[];seen=set()
        mother=int(gen['genPartIdxMother'])
        while 0<=mother<int(event.nGenPart) and mother not in seen and len(ancestors)<30:
            seen.add(mother);pdg=int(event.GenPart_pdgId[mother]);ancestors.append([mother,pdg])
            mother=int(event.GenPart_genPartIdxMother[mother])
        mothers.append(ancestors)
    row['matched_gen_muons']=matched
    row['ancestor_chains']=mothers
    row['both_hit_matched']=all(g is not None for g in matched)
    row['same_truth_particle']=row['both_hit_matched'] and row['muons'][0]['hitGenPartIdx']==row['muons'][1]['hitGenPartIdx']
    row['matched_gen_mass']=None
    if row['both_hit_matched']:
        vectors=[]
        for g in matched:
            pt,pz,phi=g['pt'],g['pz'],g['phi']
            # Nano stores zero for the masses of light particles. Restore the
            # physical muon mass for this truth diagnostic only.
            mass=.1056583755 if abs(g['pdgId'])==13 else g['mass']
            vectors.append((math.sqrt(pt*pt+pz*pz+mass*mass),pt*math.cos(phi),pt*math.sin(phi),pz))
        total=[sum(v[i] for v in vectors) for i in range(4)]
        row['matched_gen_mass']=math.sqrt(max(0,total[0]**2-sum(x*x for x in total[1:])))
    return row


def collect_candidates(config):
    old=json.loads(Path(config.audited_yields_path).read_text())
    output=Path(config.output_path).resolve();cache=output/'candidates';cache.mkdir(parents=True,exist_ok=True)
    from shift_paths import base_path,limits_cut_output_path
    algorithm=digest(__file__)
    # Two complete small canaries precede the other samples in the same worker.
    for name in ['m2_prompt','qcd_0to1']+[n for n in old['backgrounds'] if n!='qcd_0to1']+['m5_prompt','m10_prompt']:
        destination=cache/(name+'.json')
        background=name in old['backgrounds'];source=old['backgrounds'][name] if background else old['signals'][name]
        if background:
            joint=Path(limits_cut_output_path).resolve()/'data'/(name+'.npz')
            receipt=json.loads(joint.with_suffix('.json').read_text())
            assert digest(joint)==receipt['sha256']
            with np.load(joint) as arrays:
                expected={tuple(map(int,i)):float(w) for i,w in zip(arrays['ids'],arrays['weight'])}
            paths=[Path(x) for x in source['source']['files']]
            sigma=source['cross_section_pb'];denominator=source['full_gen_sumw']
            authority={'original_yields_sha256':digest(config.audited_yields_path),'joint_cache_sha256':digest(joint)}
        else:
            parent=(Path(base_path)/('darkphoton_'+name)/'latest').resolve()
            rate=json.loads((parent/'cross_sections.json').read_text())
            union=json.loads((parent/'metadata/grouping/union.json').read_text())
            assert union['complete'] and union['events']==rate['full_gen']['accepted_events']
            assert union['native_sumw']==rate['full_gen']['sum_weights']
            expected=None;paths=sorted((parent/'nanoAOD_merged').glob('*.root'))
            physical=rate['physical_selected_cross_sections_pb'];assert len(physical)==1
            sigma=physical[0]['value_pb'];denominator=rate['full_gen']['sum_weights']
            authority={'parent':str(parent),'rate_sha256':digest(parent/'cross_sections.json'),
                'union_sha256':digest(parent/'metadata/grouping/union.json'),'epsilon':rate['epsilon'],
                'proper_ctau_mm':rate['proper_ctau_mm'],'full_gen':rate['full_gen']}
        files=[]
        for path in paths:
            stat=path.stat()
            files.append({'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        binding={'name':name,'kind':'background' if background else 'signal','sigma_pb':sigma,
            'full_gen_sumw':denominator,'authority':authority,'source_file_stats':files,
            'selection':SELECTION,'algorithm_sha256':algorithm}
        if destination.exists():
            saved=json.loads(destination.read_text())
            changed=[key for key,value in binding.items() if saved.get(key)!=value]
            if changed:
                raise ValueError('Stale candidate cache '+str(destination)+'; changed '+', '.join(changed)+
                    '. Choose a new output directory; existing cache was not overwritten.')
            if background:
                selected={tuple(row['id']):row['weight'] for row in saved['candidates']}
                if len(selected)!=len(saved['candidates']) or selected!=expected:
                    raise ValueError('Candidate cache selection identities/weights changed: '+str(destination))
            print('Reusing candidate cache',name,flush=True);continue
        import ROOT
        ROOT.gROOT.SetBatch(True)
        rows=[];seen=set();full_sumw=0.;full_events=0;available_all=None
        for path,record in zip(paths,files):
            before=path.stat();f=ROOT.TFile.Open(str(path))
            if not f or f.IsZombie():raise RuntimeError('Unreadable Nano '+str(path))
            try:
                tree=f.Get('Events');available={b.GetName() for b in tree.GetListOfBranches()}
                available_all=available if available_all is None else available_all&available
                tree.SetBranchStatus('*',0)
                base=['run','luminosityBlock','event']
                if not background:base+=['genWeight','nShiftDimuonVertex','ShiftDimuonVertex_topologyMin','ShiftDimuonVertex_topologyMax']
                for b in base:tree.SetBranchStatus(b,1)
                selected=[]
                for index,event in enumerate(tree):
                    identity=(int(event.run),int(event.luminosityBlock),int(event.event))
                    if background:
                        if identity not in expected:continue
                        weight=expected[identity]
                    else:
                        full_events+=1;full_sumw+=float(event.genWeight)
                        if int(event.nShiftDimuonVertex)>1:raise ValueError('Multiple signal candidates require new arbitration')
                        if not int(event.nShiftDimuonVertex):continue
                        if (int(event.ShiftDimuonVertex_topologyMin[0]),int(event.ShiftDimuonVertex_topologyMax[0]))!=(2,2):continue
                        weight=float(event.genWeight)
                    if identity in seen:raise ValueError('Repeated selected identity')
                    seen.add(identity);selected.append((index,identity,weight))
                branches=['nGenPart','nShiftMuon','nShiftDimuonVertex']
                branches += ['ShiftDimuonVertex_'+x for x in PAIR_FIELDS]
                branches += ['ShiftMuon_'+x for x in MUON_FIELDS]+['GenPart_'+x for x in GEN_FIELDS]
                for b in branches:
                    if b in available:tree.SetBranchStatus(b,1)
                for index,identity,weight in selected:
                    tree.GetEntry(index);row=candidate(tree,available);row.update(id=list(identity),weight=weight)
                    assert (row['topologyMin'],row['topologyMax'])==(2.,2.)
                    rows.append(row)
            finally:f.Close()
            after=path.stat()
            assert (record['bytes'],record['mtime_ns'])==(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        if background:assert seen==set(expected)
        else:
            assert full_events==authority['full_gen']['accepted_events']
            assert math.isclose(full_sumw,denominator,rel_tol=1.e-12)
        result=dict(binding,candidates=rows,
            available_branches=sorted(b for b in available_all if b.startswith(('ShiftMuon_','ShiftDimuonVertex_'))))
        save_json(destination,result)
        print('Cached',name,'selected',len(rows),'hit-matched',sum(r['both_hit_matched'] for r in rows),flush=True)


def main():
    import shift_limits_book_match_config as config
    collect_candidates(config)


if __name__=='__main__':main()
