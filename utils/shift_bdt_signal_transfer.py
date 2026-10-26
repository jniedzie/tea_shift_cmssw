#!/usr/bin/env python3
# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Count sparse signal transfer at sealed validation cuts, without fitting."""
import json
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
from shift_bdt_discrimination_diagnostics import read,reference,digest,require,load_npz,clean


def summarize(e,scores,thresholds,models):
    for key in ('ids','process','heldout'):require(np.array_equal(e[key],scores[key]),'Score identity mismatch '+key)
    signal=e['process']=='signal'
    require(int(signal.sum())==39,'Expected frozen39 signal pairs')
    genuine=signal&(e['y']==1)&e['signal']
    require(not np.any(e['signal']&(e['y']!=1)),'Signal-associated flag requires genuine label')
    require(not np.any(scores['expanded_heldout'][signal]),'Signals entered expanded SM test mask')
    points=sorted(np.unique(e['point'][signal]),key=lambda name:(int(str(name).split('_')[0][1:]),{'prompt':0,'lab_5m':1,'lab_30m':2}[str(name).split('_',1)[1]]))
    require(len(points)==9,'Expected nine frozen signal points')
    rows=[]
    for point in points:
        mask=signal&(e['point']==point);positive=mask&genuine;all_common=mask&(e['y']==1)
        row=dict(point=str(point),all_pairs=int(mask.sum()),signal_genuine_pairs=int(positive.sum()),
            other_common_origin_pairs=int((all_common&~e['signal']).sum()),unknown_pairs=int((mask&(e['y']<0)).sum()),accidental_pairs=int((mask&(e['y']==0)).sum()),
            supported=False,support_reason='Only1–5 signal-associated genuine pairs; no independent-event precision or efficiency-uniformity claim.',models={})
        for model in models:
            cut=thresholds[model];values=scores[model]
            require(np.isfinite(values[mask]).all(),'Nonfinite signal scores')
            passed=int((positive&(values>=cut)).sum());n=int(positive.sum())
            row['models'][model]=dict(threshold=cut,passed=passed,total=n,retention=passed/n if n else None,
                all_common_origin_passed=int((all_common&(values>=cut)).sum()),all_common_origin_total=int(all_common.sum()))
        legacy=.4403188643890805
        row['historical_current_secondary']=dict(threshold=legacy,passed=int((positive&(e['score']>=legacy)).sum()),total=int(positive.sum()))
        rows.append(row)
    require(sum(r['signal_genuine_pairs'] for r in rows)==26,'Signal truth denominator changed')
    return rows


def legacy_main():
    root=Path.cwd();base=root/'validation/bdt_discrimination_improvement_20261012';training=base/'expanded_training';output=base/'signal_transfer';require(not output.exists(),'Refusing report overwrite')
    frozen=root/'validation/bdt_first_look_20261012/performance';report=read(frozen/'performance.json');completion=read(training/'completion.json');lock=read(training/'selection_locked.json')
    for ref in [completion['selection'],completion['development']['scores'],lock['validation'],lock['protocol']]:require(digest(ref['path'])==ref['sha256'],'Sealed candidate hash mismatch')
    require(digest(frozen/'evaluation_rows.npz')==report['artifact_sha256']['evaluation_rows.npz'],'Frozen evaluation hash mismatch')
    e=load_npz(frozen/'evaluation_rows.npz');scores=load_npz(training/'development_scores.npz');current='reference_uniform_bdt_3p0_s71'
    require(np.array_equal(scores[current],e['score']),'Current scores changed')
    models=[current,'expanded_d3_s0_n400','expanded_d3_s3_n400'];rows=summarize(e,scores,lock['thresholds'],models)
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,7),layout='constrained');xx=np.arange(9);colors=['#222222','#d95f02','#0072b2'];titles=['Current at expanded-validation cut','Rejection-selected candidate','Acceptance-selected candidate']
    for i,(model,color,title) in enumerate(zip(models,colors,titles)):
        values=[r['models'][model] for r in rows];x=xx+(i-1)*.2;y=[r['retention'] for r in values]
        ax.scatter(x,y,color=color,label=title,s=32,marker=['o','s','^'][i])
        for position,value in zip(x,values):ax.annotate(f'{value["passed"]}/{value["total"]}',(position,value['retention']),xytext=(0,8+i*11),textcoords='offset points',ha='center',fontsize=8,color=color)
    names=[]
    for row in rows:
        mass,decay=row['point'].split('_',1);decay={'prompt':'prompt','lab_5m':'5 m lab','lab_30m':'30 m lab'}[decay]
        names.append(f'{mass[1:]} GeV; {decay}\nunknown {row["unknown_pairs"]}, accidental {row["accidental_pairs"]}\nother genuine {row["other_common_origin_pairs"]}')
    ax.set_xticks(xx,names,fontsize=8);ax.set(ylim=(-.04,1.31),ylabel='Signal-associated genuine pairs retained',xlabel='Signal point and additional reconstructed-pair counts');ax.set_yticks(np.linspace(0,1,6));ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',fontsize=9)
    fig.suptitle('Signal transfer at locked expanded-validation90% cuts: exact retained / available counts\nOnly1–5 genuine signal pairs per point; no precision, independence or uniformity claim',fontsize=12)
    ax.text(.5,-.22,'Denominator: common-origin genuine AND signal-associated. One other genuine pair at50 GeV /30 m is excluded.\nSignals were never used for fitting, threshold setting or model selection; no error bars inferred from these sparse counts.',transform=ax.transAxes,ha='center',fontsize=9)
    plot=root/'tea_shift_cmssw/plots/vertex_classifier/BDT_improvement_20261012/signal_transfer_genuine_retention.pdf';require(not plot.exists(),'Refusing PDF overwrite');fig.savefig(plot,bbox_inches='tight');plt.close(fig)
    result=dict(complete=True,points=rows,models=models,counts=dict(all_pairs=39,signal_genuine=26,other_common_origin=1,unknown=11,accidental=1),
        interpretation='Sparse signal transfer only. Cuts fixed on expanded SM validation; signals never fitted or selected. No efficiency uncertainty or statistical independence claim. Historical current cut is secondary JSON only.',
        references=[reference(p) for p in [training/'completion.json',training/'selection_locked.json',training/'development_scores.npz',frozen/'performance.json',frozen/'evaluation_rows.npz',Path(__file__)]],plot=reference(plot))
    (output/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');print(json.dumps(dict(report=str(output/'summary.json'),plot=str(plot),totals={m:[sum(r['models'][m]['passed'] for r in rows),26] for m in models})))


def validate_signal_parent_alignment(data, labels, events):
    """Fail closed before calculating independent-parent efficiencies."""
    require(len(labels['y'])==len(data['ids']), 'Label rows differ')
    require(len(set(map(tuple,events['ids'])))==len(events['ids']), 'Repeated parent events')
    require(len(set(map(tuple,data['ids'])))==len(data['ids']), 'Clustered candidate intervals required')
    require(not np.any(labels['signal_associated'] & (labels['y']!=1)), 'Association requires qualified genuine label for this sample')
    for source in np.unique(events['source_index']):
        emask=events['source_index']==source;cmask=data['source_index']==source
        index=data['event_index'][cmask]
        require(np.all((index>=0)&(index<emask.sum())), 'Invalid candidate parent index')
        require(np.array_equal(data['ids'][cmask],events['ids'][emask][index]), 'Candidate parent identity differs')
        require(np.array_equal(data['native_weight'][cmask],events['native_weight'][emask][index]), 'Candidate parent weight differs')
        require(np.array_equal(data['point'][cmask],events['point'][emask][index]), 'Candidate model point differs')
    require(set(data['source_index']).issubset(set(events['source_index'])), 'Unknown candidate source')


def fresh_transfer(args):
    """Evaluate only globally locked models on an independently extracted cache."""
    from shift_bdt_category_diagnostics import interval
    workspace, cache, comparison, output = [p.resolve() for p in
        (args.workspace, args.fresh_cache, args.comparison, args.output)]
    require(output.is_relative_to(workspace/'tea_shift_cmssw/plots'), 'Output must be inside project plots')
    require(not output.exists(), 'Refusing to overwrite transfer evidence')
    completion = read(comparison/'completion.json')
    require(completion['complete'], 'Global evaluation not complete')
    checked = []
    def verify(ref):
        require(digest(ref['path']) == ref['sha256'], 'Changed sealed artifact: '+ref['path'])
        checked.append(ref)
    for key in ('protocol','selection','scores','diagnostic_inputs','parity'):
        verify(completion[key])
    diagnostic_inputs = read(completion['diagnostic_inputs']['path'])
    verify(diagnostic_inputs['selection'])
    lock = read(diagnostic_inputs['selection']['path'])
    selection = read(completion['selection']['path'])
    require(lock['selection_uses_test'] is False and selection['selection_uses_test'] is False, 'Unsafe model selection')
    require(lock['representative_models'] == selection['representative_models'], 'Changed representative models')
    require(lock['paired_seed_models'] == selection['paired_seed_models'], 'Changed paired models')
    for key in ('protocol','validation','previous_bdt_selection'): verify(lock[key])
    protocol = read(lock['protocol']['path'])
    candidates = read(lock['validation']['path'])
    metadata = read(cache/'inputs.json')
    require(metadata['complete'] and metadata['sample_role']=='validation_only', 'Incomplete or nonvalidation cache')
    require(metadata['old_event_overlap']==0 and metadata['all_parent_events_retained'], 'Unqualified parent population')
    verify(metadata['protocol']); verify(metadata['extractor'])
    for ref in metadata['sources'].values(): verify(ref)
    for ref in metadata['cache'].values(): verify(ref)
    data, labels, events = [load_npz(cache/(name+'.npz')) for name in ('reco','labels','events')]
    require(len(data['ids'])==metadata['candidates'] and len(events['ids'])==metadata['events'], 'Cache population differs')
    validate_signal_parent_alignment(data,labels,events)
    extraction_protocol = read(metadata['protocol']['path'])
    verify(extraction_protocol['old_events'])
    previous_event_ids = set(map(tuple,load_npz(extraction_protocol['old_events']['path'])['ids']))
    require(not previous_event_ids.intersection(set(map(tuple,events['ids']))), 'Signal parents overlap old development cache')
    dnns = list(dict.fromkeys(lock['representative_models']+lock['paired_seed_models']))
    bdts = ['reference_uniform_bdt_3p0_s71','expanded_d3_s0_n400','expanded_d3_s3_n400']
    old_training = Path(lock['previous_bdt_selection']['path']).parent
    old_protocol = read(old_training/'protocol.json')
    # Import inference classes from their hash-checked frozen source snapshots.
    for name in ('features.py','inference.py','uniform_bdt.py','portable_inference.py'):
        frozen = old_training/'source_snapshot'/name
        require(digest(frozen)==old_protocol['source'][name]['sha256'], 'Changed BDT source: '+name)
        checked.append(reference(frozen))
    first_round = Path(candidates[dnns[0]]['protocol']['path']).parent
    sys.path.insert(0,str(old_training/'source_snapshot'))
    sys.path.insert(0,str(first_round/'source_snapshot'))
    from features import FEATURE_NAMES
    from inference import predict as native_predict
    from portable_inference import load_model, predict as portable_predict
    require(tuple(data['feature_names'])==FEATURE_NAMES, 'Fresh feature contract differs')
    predicted, model_refs, parity = {}, {}, {}
    for name in dnns:
        entry=candidates[name]
        for key in ('model','protocol','history'): verify(entry[key])
        round_protocol=read(entry['protocol']['path'])
        for filename, ref in round_protocol['sources'].items():
            snapshot=Path(entry['protocol']['path']).parent/'source_snapshot'/filename
            require(digest(snapshot)==ref['sha256'], 'Changed neural frozen source: '+filename)
            checked.append(reference(snapshot))
        with Path(entry['model']['path']).open('rb') as stream: bundle=pickle.load(stream)
        require(tuple(bundle['feature_names'])==FEATURE_NAMES, 'Neural predictor feature order differs')
        require(np.array_equal(bundle['columns'],np.arange(len(FEATURE_NAMES))), 'Raw50 predictor wrapper required')
        require(bundle.get('requires_gen') is False, 'Model inference must be reconstructed only')
        require(bundle['threshold']==lock['thresholds'][name], 'Neural cut differs from lock')
        values=bundle['model'].predict_proba(data['X'])[:,1]
        with Path(entry['model']['path']).open('rb') as stream: reloaded=pickle.load(stream)
        require(np.array_equal(values,reloaded['model'].predict_proba(data['X'])[:,1]), 'Fresh reload parity failed')
        predicted[name]=values;model_refs[name]=entry['model']
        parity[name]={'fresh_rows':len(values),'bitwise_reload_parity':True}
    portable_parity=read(old_training/'all_cache_portable_parity.json')
    verify(portable_parity['selection'])
    for name in bdts:
        refs=portable_parity['models'][name]
        verify(refs['model']);verify(refs['pickle'])
        with Path(refs['pickle']['path']).open('rb') as stream: bundle=pickle.load(stream)
        values=portable_predict(load_model(refs['model']['path']),data['X'])
        require(np.array_equal(values,native_predict(bundle,data['X'])), 'Fresh BDT portable/native parity failed')
        predicted[name]=values;model_refs[name]={'portable':refs['model'],'native':refs['pickle']}
        parity[name]={'fresh_rows':len(values),'bitwise_portable_native_parity':True}
    for name,values in predicted.items():
        require(values.shape==labels['y'].shape and np.isfinite(values).all(), 'Invalid fresh scores: '+name)
    output.mkdir(parents=True)
    np.savez_compressed(output/'scores.npz',ids=data['ids'],pair_index=data['pair_index'],
                        source_index=data['source_index'],point=data['point'],**predicted)
    classes={'genuine':labels['y']==1,'signal_genuine':(labels['y']==1)&labels['signal_associated'],
             'accidental':labels['y']==0,'unknown':labels['y']==-1}
    def count(mask, passed):
        n=int(mask.sum());npass=int((mask&passed).sum())
        return {'total':n,'passed':npass,'retention':npass/n if n else None,'wilson95':interval(npass,n)}
    def summarize(mask, values, cut, gen_events):
        passed=values>=cut
        result={label:count(mask&truth,passed) for label,truth in classes.items()}
        result['all_pairs']=count(mask,passed)
        result['gen_events']=gen_events
        result['genuine_selected_per_gen_event']=result['signal_genuine']['passed']/gen_events
        result['all_selected_pairs_per_gen_event']=result['all_pairs']['passed']/gen_events
        return result
    point_order=sorted(np.unique(data['point']),key=lambda p:(int(str(p).split('_')[0][1:]),str(p)))
    summaries={}
    for name,values in predicted.items():
        cut=lock['thresholds'][name]
        entry={'threshold':cut,'overall':summarize(np.ones(len(values),dtype=bool),values,cut,len(events['ids'])),'points':{},'categories':{}}
        for point in point_order:
            mask=data['point']==point;nevents=int(np.sum(events['point']==point))
            entry['points'][str(point)]=summarize(mask,values,cut,nevents)
            entry['points'][str(point)]['categories']={str(cat):summarize(mask&(data['topology']==cat),values,cut,nevents) for cat in np.unique(data['topology'])}
        for cat in np.unique(data['topology']):
            entry['categories'][str(cat)]=summarize(data['topology']==cat,values,cut,len(events['ids']))
        efficiencies=[entry['points'][str(p)]['signal_genuine']['retention'] for p in point_order]
        entry['point_retention_span']=max(efficiencies)-min(efficiencies)
        summaries[name]=entry
    aliases={lock['rejection_winner']:'DNN: rejection selected',lock['flatness_winner']:'DNN: flatness selected',
             lock['acceptance_envelope_winner']:'DNN: validation envelope selected',
             bdts[0]:'Current BDT',bdts[1]:'Strong BDT',bdts[2]:'Flatness BDT'}
    main_models=lock['representative_models']+bdts
    colors=['#0072b2','#d55e00','#009e73','#222222','#cc79a7','#e69f00']
    def point_label(point):
        mass,mode=str(point).split('_',1)
        return mass[1:]+' GeV\n'+{'prompt':'prompt','lab_5m':'5 m mean lab flight','lab_30m':'30 m mean lab flight'}[mode]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output/'fresh_signal_transfer.pdf') as pdf:
        fig,axis=plt.subplots(figsize=(14,7),constrained_layout=True)
        for i,(name,color) in enumerate(zip(main_models,colors)):
            rows=[summaries[name]['points'][str(p)]['signal_genuine'] for p in point_order]
            yy=np.array([r['retention'] for r in rows]);ci=np.array([r['wilson95'] for r in rows])
            # Wilson endpoints can exceed 0/1 by roundoff for zero/all passes.
            axis.errorbar(np.arange(len(rows))+(i-2.5)*.11,yy,yerr=np.maximum(0,np.array([yy-ci[:,0],ci[:,1]-yy])),fmt='o',color=color,capsize=2,label=aliases[name])
        ns=[int(np.sum((data['point']==p)&classes['signal_genuine'])) for p in point_order]
        axis.set(xticks=np.arange(len(point_order)),xticklabels=[point_label(p)+f'\ngenuine n={n}' for p,n in zip(point_order,ns)],ylim=(0,1.05),ylabel='Signal-associated genuine pairs retained')
        axis.legend(fontsize=9,ncol=2,loc='lower left');axis.grid(axis='y',alpha=.2)
        fig.suptitle('Fresh independent signals: frozen validation cuts; Wilson 95% intervals\nSix complete points; 173 genuine pairs. Small samples do not establish mass/lifetime uniformity.',fontsize=12)
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(12,11),constrained_layout=True)
        for axis,name in zip(axes,lock['representative_models']):
            family,seed=name.rsplit('_s',1);other=family+('_s72' if seed=='71' else '_s71')
            for i,model in enumerate([name,other]):
                rows=[summaries[model]['points'][str(p)]['signal_genuine'] for p in point_order]
                yy=np.array([r['retention'] for r in rows]);ci=np.array([r['wilson95'] for r in rows])
                axis.errorbar(np.arange(len(rows))+(i-.5)*.12,yy,yerr=np.maximum(0,np.array([yy-ci[:,0],ci[:,1]-yy])),fmt='o-',capsize=2,label=('Selected ' if i==0 else 'Paired ')+model.rsplit('_',1)[1])
            axis.set(title=aliases[name],xticks=np.arange(len(point_order)),xticklabels=[point_label(p) for p in point_order],ylim=(0,1.05),ylabel='Genuine retention');axis.legend()
        fig.suptitle('Predeclared paired-seed stability; no model reselection from fresh signals',fontsize=12)
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
        for i,(name,color) in enumerate(zip(main_models,colors)):
            axes[0].plot(np.arange(len(point_order)),[summaries[name]['points'][str(p)]['genuine_selected_per_gen_event'] for p in point_order],'o-',color=color,label=aliases[name])
            cats=sorted(np.unique(data['topology']))
            rows=[summaries[name]['categories'][str(c)]['signal_genuine'] for c in cats]
            for j,row in enumerate(rows):
                if row['total']:
                    low,high=row['wilson95'];v=row['retention']
                    axes[1].errorbar(j+(i-2.5)*.1,v,yerr=np.maximum(0,[[v-low],[high-v]]),fmt='o',color=color,capsize=2)
        axes[0].plot(np.arange(len(point_order)),[np.sum((data['point']==p)&classes['signal_genuine'])/np.sum(events['point']==p) for p in point_order],'k--',label='Before classifier')
        axes[0].set(xticks=np.arange(len(point_order)),xticklabels=[point_label(p) for p in point_order],ylabel='Selected genuine reconstructed pairs / GEN events',ylim=(0,.22));axes[0].legend(fontsize=8)
        axes[1].set(xticks=np.arange(len(cats)),xticklabels=[str(c)+f'\nn={int(np.sum((data["topology"]==c)&classes["signal_genuine"]))}' for c in cats],ylabel='Genuine retention',ylim=(0,1.05))
        fig.suptitle('Denominators: 200 GEN events per point; 1,200 total; 240 retained reconstructed pairs\nProvisional detector replay, not final trigger/readout acceptance. Small categories shown with uncertainty.',fontsize=11)
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for axis,name in zip(axes.flat,main_models):
            for label,color in [('genuine','#0072b2'),('accidental','#d55e00'),('unknown','#777777')]:
                values=predicted[name][classes[label]]
                if len(values):axis.hist(values,np.linspace(0,1,41),weights=np.ones(len(values))/len(values),histtype='step',color=color,label=f'{label}: n={len(values)}')
            axis.axvline(lock['thresholds'][name],color='black',linestyle=':');axis.set(title=aliases[name],xlabel='Score',ylabel='Fraction within truth class');axis.legend(fontsize=8)
        fig.suptitle('Fresh signal score classes remain separate; unknown is not accidental',fontsize=12)
        pdf.savefig(fig);plt.close(fig)
    report={'complete':True,'schema':'shift-locked-fresh-signal-transfer-v1','selection_uses_fresh_signal':False,
            'model_choices_and_thresholds_unchanged':True,'model_selection_allowed':False,'cache':reference(cache/'inputs.json'),
            'global_selection':completion['selection'],'diagnostic_selection':diagnostic_inputs['selection'],
            'source':reference(Path(__file__)),'verified_references':checked,'model_references':model_refs,'parity':parity,
            'counts':{'gen_events':len(events['ids']),'pairs':len(data['ids']),**{k:int(v.sum()) for k,v in classes.items()}},
            'models':summaries,'representative_models':lock['representative_models'],'paired_seed_models':lock['paired_seed_models'],
            'point_order':list(map(str,point_order)),'normalization':'Raw independent parents; all native weights are1; full GEN denominator200 perpoint retained once.',
            'limitations':['Six of nine planned points, not a complete mass/lifetime grid.','Only24–33 genuine pairs/point; no precise uniformity claim.','Signals never used for fitting, thresholds, model choices or subsequent tuning.','These are fixed-cut reconstructed-candidate retention results, not final trigger/recording acceptance.','Continuous reconstructed mass/z/r and category behavior do not guarantee physical lifetime independence.'],
            'artifacts':{'scores':reference(output/'scores.npz'),'pdf':reference(output/'fresh_signal_transfer.pdf')}}
    (output/'transfer_summary.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'output':str(output),'totals':{name:entry['overall']['signal_genuine'] for name,entry in summaries.items()}}))


def main():
    if '--fresh-cache' not in sys.argv:
        legacy_main()
        return
    parser=argparse.ArgumentParser(description='Frozen-model transfer to an independent complete-point signal cache')
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--fresh-cache',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    fresh_transfer(parser.parse_args())


if __name__=='__main__':main()
