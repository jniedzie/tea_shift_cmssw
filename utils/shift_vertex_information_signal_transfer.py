# Code style: AI-code. AI-generated locked rich-feature signal transfer diagnostic.
"""Evaluate validation-selected models on existing independent signal parents.

The six points were inspected in an earlier round: they are development transfer
checks, never a new blind confirmation or a basis for model/cut selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np


def ref(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def verify(reference):
    if ref(reference['path'])['sha256']!=reference['sha256']:
        raise ValueError('Changed artifact: '+reference['path'])


def require(ok,message):
    if not ok:raise ValueError(message)


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def efficiency(passed, total):
    if total==0:return {'passed':0,'total':0,'efficiency':None,'wilson95':None}
    p=passed/total;z=1.959963984540054;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    delta=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return {'passed':int(passed),'total':int(total),'efficiency':p,'wilson95':[max(0.,center-delta),min(1.,center+delta)]}


def class_stats(values, cut, labels, weights):
    result={}
    for name,value in [('genuine',1),('accidental',0),('unknown',-1)]:
        mask=labels==value;accepted=mask&(values>=cut)
        result[name]={**efficiency(int(accepted.sum()),int(mask.sum())),
          'native_sumw':float(weights[mask].sum()),'accepted_native_sumw':float(weights[accepted].sum())}
    return result


def point_label(point):
    pieces=point.split('_')
    return pieces[0][1:]+' GeV\n'+('prompt' if pieces[1]=='prompt' else 'mean lab flight '+pieces[2])


def run(args):
    cache,training,previous,output=[p.resolve() for p in (args.cache,args.comparison,args.previous,args.output)]
    require(not output.exists(),'Refusing to overwrite signal transfer')
    completion=json.loads((training/'completion.json').read_text());require(completion['complete'],'Training evaluation incomplete')
    for key in ('protocol','selection','scores','diagnostic_inputs','parity'):verify(completion[key])
    lock=json.loads(Path(completion['selection']['path']).read_text())
    require(lock['selection_uses_test'] is False,'Selection used development test')
    for key in ('protocol','validation'):verify(lock[key])
    if 'family_validation' in lock:verify(lock['family_validation'])
    protocol=json.loads(Path(lock['protocol']['path']).read_text())
    candidates=json.loads(Path(lock['validation']['path']).read_text())
    selected=list(dict.fromkeys(lock['representative_models']+lock['paired_seed_models']))
    require(bool(selected),'No globally selected model')
    origins={}
    for name in selected:
        verify(candidates[name]['protocol'])
        origin=Path(candidates[name]['protocol']['path']).parent
        origin_protocol=json.loads((origin/'protocol.json').read_text())
        for filename,r in origin_protocol['sources'].items():
            snapshot=origin/'source_snapshot'/filename
            require(ref(snapshot)['sha256']==r['sha256'],'Changed frozen model source: '+filename)
        origins[name]=(origin,origin_protocol)
    inference_origin,first_protocol=origins[selected[0]]
    sys.path.insert(0,str(inference_origin/'source_snapshot'))
    from neural_vertex_information_study import predict
    metadata=json.loads((cache/'inputs_resolved.json').read_text())
    require(metadata['complete'] and metadata['sample_role']=='validation_only','Incomplete/wrong signal sample role')
    for k in ('reco','labels','events'):verify(metadata['cache'][k])
    verify(metadata['protocol']);verify(metadata['source_snapshot'])
    reco,labels,events=[load(cache/(k+'.npz')) for k in ('reco','labels','events')]
    from shift_bdt_signal_transfer import validate_signal_parent_alignment
    validate_signal_parent_alignment(reco,labels,events)
    require(len(events['ids'])==1200 and len(reco['ids'])==240,'Qualified six-point population changed')
    old=json.loads((previous/'transfer_summary.json').read_text());verify(old['artifacts']['scores']);verify(old['cache'])
    verify(old['global_selection']);verify(old['diagnostic_selection'])
    previous_lock=json.loads(Path(old['diagnostic_selection']['path']).read_text())
    previous_cache=Path(old['cache']['path']).parent
    old_reco=load(previous_cache/'reco.npz')
    require(np.array_equal(reco['X'][:,:50],old_reco['X'],equal_nan=True),'Old X50 feature prefix changed')
    for k in ('labels','events'):
        require(ref(cache/(k+'.npz'))['sha256']==ref(previous_cache/(k+'.npz'))['sha256'],'Old labels/events changed')
    old_scores=load(old['artifacts']['scores']['path'])
    for k in ('ids','pair_index','source_index','point'):
        require(np.array_equal(reco[k],old_scores[k]),'Old reference score row alignment changed')
    verify(first_protocol['events'])
    train_events=load(first_protocol['events']['path'])
    require(not set(map(tuple,events['ids']))&set(map(tuple,train_events['ids'])),'Signal parent overlaps training population')
    selected=list(dict.fromkeys(lock['representative_models']+lock['paired_seed_models']))
    references=['dnn_r4_l3_c3_e0_h64_s71','reference_uniform_bdt_3p0_s71','expanded_d3_s0_n400','expanded_d3_s3_n400']
    scores={};thresholds={};parity={};models={}
    for name in selected:
        entry=candidates[name]
        for k in ('model','history','protocol'):verify(entry[k])
        with Path(entry['model']['path']).open('rb') as f:bundle=pickle.load(f)
        require(bundle['requires_gen'] is False and bundle['threshold']==lock['thresholds'][name],'Model contract/cut changed')
        scores[name]=predict(bundle,reco);thresholds[name]=lock['thresholds'][name];models[name]=entry['model']
        with Path(entry['model']['path']).open('rb') as f:again=pickle.load(f)
        require(np.array_equal(scores[name],predict(again,reco)),'Reload score parity failed')
        require(scores[name].shape==labels['y'].shape and np.isfinite(scores[name]).all(),'Invalid score output')
        parity[name]={'bitwise_reload':True,'rows':len(reco['ids'])}
    for name in references:
        scores[name]=old_scores[name];thresholds[name]=old['models'][name]['threshold']
        require(thresholds[name]==previous_lock['thresholds'][name],'Old reference cut changed')
        models[name]=old['model_references'][name]
        parity[name]={'reused_exact_hash_checked_scores':True,'old_raw50_inputs_exact':True,'original_parity':old['parity'][name]}
    points=sorted(set(map(str,reco['point'])));categories=['Both-Both','Near-Both','Both-Far','Near-Far','Other']
    report={}
    for name,values in scores.items():
        cut=thresholds[name];summary={'threshold':cut,'overall':class_stats(values,cut,labels['y'],reco['native_weight']),'points':{},'categories':{}}
        for point in points:
            mask=reco['point']==point;nparent=int(np.sum(events['point']==point))
            summary['points'][point]={'all_GEN_parents':nparent,'reconstructed_pairs':int(mask.sum()),
              'classes':class_stats(values[mask],cut,labels['y'][mask],reco['native_weight'][mask]),
              'retained_genuine_per_GEN':efficiency(int(np.sum(mask&(labels['y']==1)&(values>=cut))),nparent)}
        for category in categories:
            mask=reco['topology']==category
            summary['categories'][category]=class_stats(values[mask],cut,labels['y'][mask],reco['native_weight'][mask])
        report[name]=summary
    output.mkdir(parents=True)
    np.savez_compressed(output/'scores.npz',**{k:reco[k] for k in ('ids','pair_index','source_index','point')},**scores)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    names=list(scores)
    short={name:name.replace('info_','').replace('reference_uniform_bdt_3p0_s71','Current BDT').replace('expanded_d3_s0_n400','Strong BDT').replace('expanded_d3_s3_n400','Flat BDT').replace('dnn_r4_l3_c3_e0_h64_s71','Previous strict DNN') for name in names}
    with PdfPages(output/'signal_transfer.pdf') as pdf:
        for field,title in [('points','Genuine reconstructed-pair retention'),('categories','Genuine retention by reconstructed category')]:
            columns=points if field=='points' else categories
            matrix=np.full((len(names),len(columns)),np.nan);annotations={}
            for i,name in enumerate(names):
                for j,col in enumerate(columns):
                    stats=report[name][field][col];stats=stats['classes']['genuine'] if field=='points' else stats['genuine']
                    if stats['total']:
                        matrix[i,j]=100*stats['efficiency'];annotations[i,j]=f"{stats['passed']}/{stats['total']}"
                    else:annotations[i,j]='empty'
            fig,ax=plt.subplots(figsize=(12,max(5,0.43*len(names)+2)))
            im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
            ax.set_xticks(range(len(columns)),[point_label(x) if field=='points' else x for x in columns],rotation=20,ha='right')
            ax.set_yticks(range(len(names)),[short[x] for x in names]);ax.set_title(title+' at frozen inclusive validation cuts')
            for (i,j),text in annotations.items():ax.text(j,i,text,ha='center',va='center',color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]<=60 else 'black',fontsize=9)
            fig.colorbar(im,ax=ax,label='Retained genuine pairs (%)')
            fig.text(.5,.015,'1200 independent GEN parents; 240 pairs = 173 genuine + 11 accidental + 56 unknown. Six previously inspected points; no model selection here.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,1));pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,6))
        shown=list(dict.fromkeys(lock['representative_models']+['dnn_r4_l3_c3_e0_h64_s71','expanded_d3_s3_n400']))
        for offset,name in enumerate(shown):
            stats=[report[name]['points'][point]['classes']['genuine'] for point in points]
            y=np.array([x['efficiency'] for x in stats]);bounds=np.array([x['wilson95'] for x in stats]);shift=(offset-(len(shown)-1)/2)*.065
            ax.errorbar(np.arange(len(points))+shift,100*y,yerr=100*np.maximum(0,np.array([y-bounds[:,0],bounds[:,1]-y])),fmt='o',capsize=2,label=short[name])
        ax.set_xticks(range(len(points)),[point_label(p) for p in points],rotation=20,ha='right');ax.set_ylim(0,102)
        ax.set_ylabel('Genuine reconstructed-pair retention (%)');ax.set_title('Selected models: pointwise retention with Wilson95 intervals')
        ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.2);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    result={'complete':True,'selection_uses_signal':False,'model_selection_allowed':False,'thresholds_unchanged':True,
      'selection':ref(training/'selection_locked.json'),'protocol':ref(training/'protocol.json'),'cache':ref(cache/'inputs_resolved.json'),
      'previous_signal_transfer':ref(previous/'transfer_summary.json'),'counts':{'all_GEN_parents':1200,'pairs':240,'genuine':173,'accidental':11,'unknown':56},
      'representative_models':lock['representative_models'],'paired_seed_models':lock['paired_seed_models'],'models':report,'model_references':models,'parity':parity,
      'limitations':['Six previously inspected validation-only points; no new blind confirmation.','Only24-33 genuine pairs per point; category support dominated by Both-Both.','No completed all-nine-point denominator; incomplete points excluded.','GEN denominators are retained once per point; efficiencies do not establish recording/trigger acceptance.'],
      'artifacts':{'scores':ref(output/'scores.npz'),'pdf':ref(output/'signal_transfer.pdf')},'source':ref(__file__)}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:entry['overall']['genuine'] for name,entry in report.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--comparison','--training',dest='comparison',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
