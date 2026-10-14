# Code style: AI-code. AI-generated reconstructed-information extraction utility.
"""Extend frozen pair rows without truth reads, row selection, or score evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

QUALITY = ('normalizedChi2','nValidHits','nValidMuonHits','nMuonStations','nLostHits',
           'topology','recoAlgorithm','ptErr','etaErr','phiErr')
DETAIL = ('nDTHits','nCSCHits','nRPCHits','nGEMHits','nME0Hits','nHitsNearEndcap',
          'nHitsBarrel','nHitsFarEndcap','nStationsNearEndcap','nStationsBarrel',
          'nStationsFarEndcap','hitSpan','hitDeltaZ','duplicateGroupSize',
          'duplicateDSACount','duplicateTraversingCount','duplicateCosmicCount',
          'nTimingMeasurements','timingChi2','timingDeltaChi2','timingDirectionSign')
EVENT = ('nShiftMuon','nShiftDT','nShiftCSC','nShiftRPC','nShiftGEM')
FEATURE_NAMES = tuple('mu_'+leg+'_'+key for leg in ('highPt','lowPt')
                      for key in QUALITY+DETAIL+('relativePtErr',)) + tuple('event_'+key for key in EVENT)
BRANCHES = tuple(dict.fromkeys(('run','luminosityBlock','event','nShiftDimuonVertex',
    'ShiftDimuonVertex_muonIdx1','ShiftDimuonVertex_muonIdx2','ShiftMuon_pt') +
    tuple('ShiftMuon_'+key for key in QUALITY+DETAIL) + EVENT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def require(ok,message):
    if not ok: raise ValueError(message)


def extract_extension(arrays, expected):
    """Use exact original event/pair references and reconstructed-pT ordering."""
    rows=[]
    for row,(event,pair) in enumerate(zip(expected['event_index'],expected['pair_index'])):
        event,pair=int(event),int(pair)
        ids=np.asarray([int(arrays[k][event]) for k in ('run','luminosityBlock','event')],dtype=np.uint64)
        require(np.array_equal(ids,expected['ids'][row]),'Candidate physical identity mismatch')
        refs=[int(arrays['ShiftDimuonVertex_muonIdx'+str(i)][event][pair]) for i in (1,2)]
        require(np.array_equal(refs,expected['muon_indices'][row]),'Pair muon references changed')
        nmu=int(arrays['nShiftMuon'][event])
        require(refs[0]!=refs[1] and all(0<=i<nmu for i in refs),'Invalid pair muon index')
        pts=arrays['ShiftMuon_pt'][event]
        order=sorted(refs,key=lambda i:(-float(pts[i]),i))
        values=[]
        for i in order:
            for key in QUALITY+DETAIL:
                column=arrays['ShiftMuon_'+key][event]
                require(len(column)==nmu,'Muon branch collection length mismatch')
                values.append(float(column[i]))
            err=float(arrays['ShiftMuon_ptErr'][event][i]);pt=float(pts[i])
            values.append(err/pt if pt>0 and err>=0 else np.nan)
        values.extend(float(arrays[k][event]) for k in EVENT)
        rows.append(values)
    return np.asarray(rows,dtype=float).reshape(-1,len(FEATURE_NAMES))


def main():
    import awkward as ak
    import uproot
    p=argparse.ArgumentParser();p.add_argument('--old-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    require(not (out/'inputs.json').exists(),'Output already finalized')
    manifest_path=args.old_cache/'inputs.json';manifest=json.loads(manifest_path.read_text())
    old={k:np.load(args.old_cache/(k+'.npz'),allow_pickle=False) for k in ('reco','labels','events')}
    for key in old: require(sha(args.old_cache/(key+'.npz'))==manifest[key+'_sha256'],'Old cache hash mismatch')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    frozen=snapshot/Path(__file__).name
    if not frozen.exists(): shutil.copyfile(__file__,frozen)
    require(sha(frozen)==sha(__file__),'Frozen extractor differs')
    protocol={'source':{'path':str(frozen.resolve()),'sha256':sha(frozen)},
      'old_manifest':{'path':str(manifest_path.resolve()),'sha256':sha(manifest_path)},
      'branches':BRANCHES,'additional_feature_names':FEATURE_NAMES,'truth_branches_read':False,
      'full_nano_hash_policy':'Frozen SHA provenance plus exact unchanged stat signature; any changed stat triggers full SHA verification. Exact field/identity readback always.',
      'covariance_exclusion':'Stored targetPredicted/constrained covariance refers to target/prompt fit hypotheses; excluded from this first LLP-safe extension.',
      'candidate_selection_applied':False,'limit':args.limit}
    (out/'extraction_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    receipts=[];extensions=[];started=time.monotonic();r=old['reco'];ev=old['events']
    for source in manifest['inputs'][:args.limit]:
        index=source['index'];path=Path(source['path']);before=path.stat()
        sig=[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns]
        hash_rechecked = sig != source['stat_signature']
        if hash_rechecked:
            require(sha(path)==source['nano_sha256'],'Nano hash differs from frozen extraction: '+str(path))
        selected=np.flatnonzero(r['source_index']==index);parents=np.flatnonzero(ev['source_index']==index)
        with uproot.open(path,handler=uproot.source.file.MultithreadedFileSource,num_workers=1) as f:
            tree=f['Events'];missing=set(BRANCHES)-set(tree.keys())
            require(not missing,'Missing reconstructed branches: '+repr(missing))
            a=tree.arrays(BRANCHES,library='ak',how=dict)
        ids=np.column_stack([ak.to_numpy(a[k]).astype(np.uint64) for k in ('run','luminosityBlock','event')])
        require(np.array_equal(ids,ev['ids'][parents]),'Full-parent identity/order mismatch')
        require(int(ak.sum(a['nShiftDimuonVertex']))==len(selected),'Pair count mismatch')
        expected={k:r[k][selected] for k in ('ids','event_index','pair_index','muon_indices')}
        extra=extract_extension(a,expected)
        # Associated values must reduce exactly to every frozen quality min/max.
        for q,key in enumerate(QUALITY):
            cols=extra[:,[q,len(QUALITY)+len(DETAIL)+1+q]]
            for opname,fn in [('min',np.min),('max',np.max)]:
                original=r['X'][selected,list(r['feature_names']).index('mu_'+opname+'_'+key)]
                require(np.array_equal(fn(cols,axis=1),original,equal_nan=True),'Frozen quality parity failed '+key)
        after=path.stat();require(sig==[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns],'Nano changed during read')
        extensions.append(extra);receipts.append({'index':index,'path':str(path),'nano_sha256':source['nano_sha256'],
          'stat_signature':sig,'full_hash_rechecked':hash_rechecked,'events':len(ids),'pairs':len(selected),'all_parent_ids_exact':True,'quality_minmax_exact':True})
        print(json.dumps({'source':index,'pairs':len(selected),'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    x=np.concatenate(extensions);n=len(x)
    require(args.limit is not None or n==len(r['X']),'Incomplete full extraction')
    arrays={k:r[k][:n] if r[k].ndim and len(r[k])==len(r['X']) else r[k] for k in r.files}
    arrays['X']=np.column_stack((r['X'][:n],x));arrays['feature_names']=np.concatenate((r['feature_names'],np.asarray(FEATURE_NAMES)))
    np.savez_compressed(out/'reco.npz',**arrays)
    if args.limit is None:
        for key in ('labels','events'): shutil.copyfile(args.old_cache/(key+'.npz'),out/(key+'.npz'))
    audit={name:{'finite':int(np.isfinite(x[:,i]).sum()),'nonzero':int(np.sum(np.isfinite(x[:,i])&(x[:,i]!=0))),
       'unique_finite':int(len(np.unique(x[np.isfinite(x[:,i]),i]))),
       'min':float(np.min(x[np.isfinite(x[:,i]),i])) if np.isfinite(x[:,i]).any() else None,
       'max':float(np.max(x[np.isfinite(x[:,i]),i])) if np.isfinite(x[:,i]).any() else None}
       for i,name in enumerate(FEATURE_NAMES)}
    (out/'feature_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    result={**manifest,'schema':'shift-rich-reconstructed-information-v1','complete':args.limit is None,
      'inputs':receipts,'candidates':n,'sample_kind':'simulation_only','old_X50_exact':True,
      'additional_features':len(FEATURE_NAMES),'feature_count':arrays['X'].shape[1],
      'extra_feature_groups':{name:('event' if name.startswith('event_') else 'muon') for name in FEATURE_NAMES},
      'events':len(ev['ids']) if args.limit is None else sum(x['events'] for x in receipts),
      'reco_sha256':sha(out/'reco.npz'),'labels_sha256':manifest['labels_sha256'],'events_sha256':manifest['events_sha256'],
      'protocol':{'path':str((out/'extraction_protocol.json').resolve()),'sha256':sha(out/'extraction_protocol.json')},
      'source_snapshot':protocol['source'],'truth_branches_read':False,'weights_unchanged':True,
      'elapsed_seconds':time.monotonic()-started}
    (out/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'complete':result['complete'],'shape':arrays['X'].shape,'seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
