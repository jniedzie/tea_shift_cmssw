# Code style: AI-code. AI-generated diagnostic; no training or model mutation.
"""Compare fixed prefixes of a sealed forest using its original event partitions."""
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
from shift_bdt_discrimination_diagnostics import read,reference,digest,require,load_npz,clean,tree_votes
from shift_bdt_candidate_diagnostics import fit_population

PREFIXES=(50,100,150,200,300,400)


def prefix_scores(model,X,prefixes=PREFIXES):
    votes=tree_votes(model,X);coefficients=np.array([t['coefficient'] for t in model['trees']]);margin=np.zeros(len(X));output={}
    for index,coefficient in enumerate(coefficients):
        margin+=coefficient*votes[:,index]
        n=index+1
        if n in prefixes:
            normalized=margin/np.sum(np.abs(coefficients[:n]))
            output[n]=np.asarray([1/(1+math.exp(-2*value)) for value in normalized])
    require(set(output)==set(prefixes),'Requested prefix exceeds forest length')
    return output


def validation_cut(scores,y,split):
    values=scores[(split==1)&(y==1)];require(len(values)>0,'Missing validation positives')
    return float(np.quantile(values,.1,method='lower'))


def choose_prefix(rows):
    return max(rows,key=lambda n:(rows[n]['metrics']['validation']['sm']['accidental_rejection'],rows[n]['metrics']['validation']['sm']['auc'],-int(n)))


def build(workspace):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp
    base=workspace/'validation/bdt_discrimination_improvement_20261012/expanded_training';cache=workspace/'validation/bdt_acceptance_study_20261012/comparison/extraction';perf=workspace/'validation/bdt_first_look_20261012/performance'
    C=workspace/'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier';sys.path.insert(0,str(C));import bdt_performance
    completion=read(base/'completion.json');manifest=read(cache/'inputs.json');frozen=read(perf/'performance.json');modelpath=base/'portable/expanded_d3_s0_n400.json';model=read(modelpath);receipt=read(modelpath.with_suffix('.json.receipt.json'))
    require(digest(modelpath)==receipt['model_sha256'],'Model digest mismatch')
    for ref in [completion['selection'],completion['development']['scores']]:require(digest(ref['path'])==ref['sha256'],'Candidate digest mismatch')
    for name in ('reco','labels'):require(digest(cache/(name+'.npz'))==manifest[name+'_sha256'],'Cache digest mismatch')
    require(digest(perf/'evaluation_rows.npz')==frozen['artifact_sha256']['evaluation_rows.npz'],'Frozen evaluation mismatch')
    reco=load_npz(cache/'reco.npz');labels=load_npz(cache/'labels.npz');evaluation=load_npz(perf/'evaluation_rows.npz');saved=load_npz(base/'development_scores.npz')
    for key in ('ids','process'):require(np.array_equal(reco[key],evaluation[key]) and np.array_equal(saved[key],evaluation[key]),'Identity mismatch '+key)
    for key in ('pair_index','source_index'):require(np.array_equal(saved[key],reco[key]),'Pair mismatch '+key)
    require(np.array_equal(labels['y'],evaluation['y']),'Truth labels changed')
    _,full_split=fit_population(saved,evaluation,'expanded-test')
    all_scores=prefix_scores(model,reco['X']);require(np.array_equal(all_scores[400],saved['expanded_d3_s0_n400']),'Full prefix differs from sealed scores')
    keep=(evaluation['process']!='signal')&(evaluation['y']>=0);y=evaluation['y'][keep];p=evaluation['process'][keep];split=full_split[keep];nuisance=reco['nuisance'][keep];ids=evaluation['ids'][keep]
    scores={n:v[keep] for n,v in all_scores.items()};edges=[np.quantile(nuisance[(split==0)&(y==1),i],[1/3,2/3]) for i in range(3)]
    output=dict(schema='shift-bdt-overtraining-prefix-v1',complete=True,prefixes={},selection_population='Expanded validation only; maximize accidental rejection at90% genuine validation retention, then AUC; prefer smaller prefix on ties.',
        evaluation_caveat='All data are consumed development data. Prefix comparison is a new validation-based diagnostic, not a new blind test or deployment decision.',
        metric_weights='Empirical pair counts. Training used evolving class-balanced boosting weights; these diagnostic metrics do not reuse those weights or cross sections.',
        ks_interpretation='Class-conditional empirical train-versus-validation KS distance only; no p-values. Unequal training weights and previously inspected data preclude interpreting this as a calibration test.',
        prefix_score='Each prefix retains original signed tree votes and coefficients, normalized by its own coefficient sum according to the same original score formula. No production scores changed.',
        bin_edges=[edge.tolist() for edge in edges],counts={name:{str(truth):int(((split==part)&(y==truth)).sum()) for truth in (0,1)} for part,name in enumerate(('train','validation','test'))})
    coefficients=np.array([t['coefficient'] for t in model['trees']]);root272=np.array([model['feature_names'][model['columns'][t['feature'][0]]]=='mu_lowPt_energy' and abs(t['threshold'][0]-272.44314575)<.001 for t in model['trees']])
    for n,score in scores.items():
        cut=validation_cut(score,y,split);row=dict(threshold=cut,root272_count=int(root272[:n].sum()),root272_weight_fraction=float(coefficients[:n][root272[:n]].sum()/coefficients[:n].sum()),metrics={},train_validation_ks={})
        for part,name in enumerate(('train','validation','test')):
            row['metrics'][name]={}
            for process in ('sm','dy','jpsi','qcd'):
                population=(split==part)&((p!='signal') if process=='sm' else p==process);genuine=population&(y==1);accidental=population&(y==0)
                entry=dict(genuine_pairs=int(genuine.sum()),accidental_pairs=int(accidental.sum()),auc=float(roc_auc_score(y[population],score[population])) if genuine.any() and accidental.any() else None,
                    genuine_efficiency=float((score[genuine]>=cut).mean()) if genuine.any() else None,accidental_rejection=float((score[accidental]<cut).mean()) if accidental.any() else None,
                    genuine_ci95_pair_wilson=bdt_performance.wilson(int((score[genuine]>=cut).sum()),int(genuine.sum())),
                    rejection_ci95_pair_wilson=bdt_performance.wilson(int((score[accidental]<cut).sum()),int(accidental.sum())),profiles=[])
                # In this retained cache every known pair has a distinct process/event identity.
                groups={(str(proc),*map(int,event)) for proc,event in zip(p[population],ids[population])}
                require(len(groups)==int(population.sum()),'Multiple pairs per event require event-level intervals')
                spans=[]
                for dimension,edge in enumerate(edges):
                    bins=np.searchsorted(edge,nuisance[:,dimension],side='right');bins_out=[]
                    for b in range(3):
                        mask=genuine&(bins==b)&np.isfinite(nuisance[:,dimension]);total=int(mask.sum());passed=int((score[mask]>=cut).sum())
                        bins_out.append(dict(pairs=total,passed=passed,efficiency=passed/total if total else None,supported=total>=20,ci95_pair_wilson=bdt_performance.wilson(passed,total)))
                    values=[b['efficiency'] for b in bins_out if b['supported']];span=max(values)-min(values) if len(values)>=2 else None
                    entry['profiles'].append(dict(bins=bins_out,supported_span=span));spans.extend([span] if span is not None else [])
                entry['maximum_supported_mass_z_radius_span']=max(spans) if spans else None;row['metrics'][name][process]=entry
        for process in ('sm','dy','jpsi','qcd'):
            row['train_validation_ks'][process]={}
            for truth in (0,1):
                pop=(y==truth)&((p!='signal') if process=='sm' else p==process);a=score[pop&(split==0)];b=score[pop&(split==1)]
                row['train_validation_ks'][process][str(truth)]=float(ks_2samp(a,b).statistic) if len(a) and len(b) else None
        output['prefixes'][str(n)]=row
    output['validation_selected_prefix']=int(choose_prefix(output['prefixes']))
    selected=output['validation_selected_prefix'];mask=(split==1)&(y==0)
    difference=(scores[selected][mask]<output['prefixes'][str(selected)]['threshold']).astype(float)-(scores[400][mask]<output['prefixes']['400']['threshold']).astype(float)
    rng=np.random.default_rng(7113);replicates=np.mean(rng.choice(difference,size=(2000,len(difference)),replace=True),axis=1)
    output['selected_vs400_validation_rejection']=dict(difference=float(difference.mean()),selected_only_rejects=int((difference==1).sum()),full_only_rejects=int((difference==-1).sum()),
        ci95_paired_event_bootstrap=np.quantile(replicates,[.025,.975]).tolist(),interpretation='Descriptive paired interval; does not correct for choosing the prefix on this validation sample.')
    output['references']=[reference(path) for path in [base/'completion.json',base/'selection_locked.json',base/'development_scores.npz',modelpath,modelpath.with_suffix('.json.receipt.json'),cache/'inputs.json',cache/'reco.npz',cache/'labels.npz',perf/'performance.json',perf/'evaluation_rows.npz',Path(__file__)]]
    return output,dict(y=y,process=p,split=split,scores=scores)


def render(report,data,directory):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    selected=report['validation_selected_prefix'];prefixes=list(PREFIXES);styles=[('train','Training','#777777'),('validation','Validation','#0072b2'),('test','Inspected test','#d95f02')]
    def save(fig,name):
        path=directory/(name+'.pdf');fig.savefig(path,bbox_inches='tight');plt.close(fig);return reference(path)
    paths=[];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for part,title,color in styles:
        metrics=[report['prefixes'][str(n)]['metrics'][part]['sm'] for n in prefixes]
        for ax,key,ylabel in [(axes[0,0],'auc','Genuine versus accidental AUC'),(axes[0,1],'accidental_rejection','Accidental pairs rejected'),(axes[1,0],'genuine_efficiency','Genuine pairs retained'),(axes[1,1],'maximum_supported_mass_z_radius_span','Largest genuine-efficiency span across mass / z / radius bins')]:
            values=np.array([m[key] for m in metrics]);ci_key={'genuine_efficiency':'genuine_ci95_pair_wilson','accidental_rejection':'rejection_ci95_pair_wilson'}.get(key)
            if ci_key:
                ci=np.array([m[ci_key] for m in metrics]);ax.errorbar(prefixes,values,yerr=np.array([values-ci[:,0],ci[:,1]-values]),fmt='o-',capsize=2,label=title,color=color)
            else:ax.plot(prefixes,values,'o-',label=title,color=color)
            ax.set(xlabel='Number of saved trees',ylabel=ylabel);ax.axvline(selected,color='#009e73',linestyle=':');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);axes[1,0].axhline(.9,color='black',linestyle=':');fig.suptitle(f'Saved-forest prefix check; validation-selected prefix = {selected} trees\nCuts fixed at 90% validation genuine retention; fraction bars: 95% Wilson intervals\nTest already used for development; validation genuine interval is not a threshold uncertainty',fontsize=11);paths.append(save(fig,'prefix_performance_v1'))
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,(process,title) in zip(axes,[('dy','DY'),('jpsi','J/psi'),('qcd','QCD')]):
        for part,label,color in styles:ax.plot(prefixes,[report['prefixes'][str(n)]['metrics'][part][process]['accidental_rejection'] for n in prefixes],'o-',label=label,color=color)
        ax.set(title=title,xlabel='Number of saved trees',ylabel='Accidental pairs rejected',ylim=(0,1));ax.grid(alpha=.2);ax.axvline(selected,color='#009e73',linestyle=':')
    axes[0].legend(fontsize=9);fig.suptitle('Within-process accidental rejection; all cuts fixed on pooled validation genuine pairs');paths.append(save(fig,'prefix_process_rejection_v1'))
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');y=data['y'];split=data['split'];p=data['process'];bins=np.linspace(0,1,51)
    for row,n in enumerate([selected,400]):
        for ax,(process,title) in zip(axes[row],[('sm','Pooled SM'),('dy','DY'),('jpsi','J/psi')]):
            for part,partlabel,linestyle in [(0,'Training','-'),(1,'Validation','--')]:
                for truth,label,color in [(1,'Genuine','#2166ac'),(0,'Accidental','#d95f02')]:
                    mask=(split==part)&(y==truth)&((p!='signal') if process=='sm' else p==process);values=data['scores'][n][mask];ax.hist(values,bins=bins,weights=np.full(len(values),1/len(values)),histtype='step',color=color,linestyle=linestyle,label=partlabel+' '+label)
            ax.set(title=f'{title}, {n} trees',xlabel='Prefix score',ylabel='Fraction within class');ax.legend(fontsize=7)
    fig.suptitle('Class-conditional training / validation score shapes\nEmpirical pair counts; no KS p-values or physical-probability interpretation');paths.append(save(fig,'train_validation_shapes_v1'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].plot(prefixes,[report['prefixes'][str(n)]['root272_weight_fraction'] for n in prefixes],'o-');axes[0].set(xlabel='Number of saved trees',ylabel='Fraction of coefficient weight',title='Trees rooted at lower-leg energy 272.443 GeV')
    for process,color in [('sm','black'),('dy','#e69f00'),('jpsi','#0072b2')]:
        for truth,label,ls in [('0','accidental','--'),('1','genuine','-')]:axes[1].plot(prefixes,[report['prefixes'][str(n)]['train_validation_ks'][process][truth] for n in prefixes],linestyle=ls,color=color,label=process+' '+label)
    axes[1].set(xlabel='Number of saved trees',ylabel='Train / validation KS distance',title='Descriptive shape difference; no p-values');axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Repeated split weight and score-shape generalization');paths.append(save(fig,'root_weight_and_ks_v1'))
    return paths


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();require(not args.output.exists(),'Refusing previous report/plot overwrite')
    report,data=build(args.workspace.resolve());args.output.mkdir(parents=True);report['plots']=render(report,data,args.output);(args.output/'report_v1.json').write_text(json.dumps(clean(report),indent=2)+'\n')
    for n in PREFIXES:
        row=report['prefixes'][str(n)];print(n,[(k,round(row['metrics'][k]['sm']['auc'],4),round(row['metrics'][k]['sm']['accidental_rejection'],4)) for k in ('train','validation','test')],round(row['root272_weight_fraction'],4))
    print('VALIDATION_SELECTED',report['validation_selected_prefix'])


if __name__=='__main__':main()
