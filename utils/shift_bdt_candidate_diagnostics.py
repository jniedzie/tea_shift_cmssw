#!/usr/bin/env python3
# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Read sealed candidate scores; audit relative reco observables without retraining."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from shift_bdt_discrimination_diagnostics import read, reference, digest, require, load_npz, clean, single_feature_audit


def relative_features(X, names):
    """Only reconstructed quantities; never restore lost per-leg associations."""
    col = {str(name): X[:, i] for i, name in enumerate(names)}
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=np.isfinite(b) & (b > 0))
    high = np.column_stack([col['mu_highPt_'+k] for k in ('px','py','pz')])
    low = np.column_stack([col['mu_lowPt_'+k] for k in ('px','py','pz')])
    highpt, lowpt = np.linalg.norm(high[:,:2],axis=1), np.linalg.norm(low[:,:2],axis=1)
    transverse_error = np.hypot(col['pair_vxErr'],col['pair_vyErr'])
    errors = np.sqrt(sum(col['pair_'+k+'Err']**2 for k in ('vx','vy','vz')))
    values = {
        'Pair track-uncertainty scale / total transverse momentum': ratio(np.hypot(col['mu_min_ptErr'],col['mu_max_ptErr']),highpt+lowpt),
        'Hit-count balance (smaller / larger)': ratio(col['mu_min_nValidMuonHits'],col['mu_max_nValidMuonHits']),
        'Track-fit chi2 balance (smaller / larger)': ratio(col['mu_min_normalizedChi2'],col['mu_max_normalizedChi2']),
        'DCA / transverse vertex-error scale': ratio(col['pair_dca'],transverse_error),
        'DCA / total vertex-error scale': ratio(col['pair_dca'],errors),
        'Opening angle / combined angular-error scale': ratio(col['opening_angle'],np.sqrt(sum(col['mu_'+side+'_'+k]**2 for side in ('min','max') for k in ('etaErr','phiErr')))),
        'Absolute azimuthal opening cosine': abs(ratio(np.sum(high[:,:2]*low[:,:2],axis=1),highpt*lowpt)),
    }
    return np.column_stack(list(values.values())),list(values)


def build(workspace, candidate_dir):
    C=workspace/'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier';sys.path.insert(0,str(C))
    import bdt_performance
    completion=read(candidate_dir/'completion.json');lock=read(candidate_dir/'selection_locked.json')
    for ref in (completion['selection'],completion['scores'],completion['development']['scores'],lock['validation'],lock['protocol']):
        require(digest(ref['path'])==ref['sha256'],'Candidate receipt mismatch: '+ref['path'])
    scores=load_npz(candidate_dir/'development_scores.npz')
    evaluation_path=workspace/'validation/bdt_first_look_20261012/performance/evaluation_rows.npz'
    e=load_npz(evaluation_path);cache=workspace/'validation/bdt_acceptance_study_20261012/comparison/extraction'
    reco=load_npz(cache/'reco.npz')
    for name in ('ids','process','heldout'): require(np.array_equal(scores[name],e[name]),'Candidate alignment mismatch '+name)
    for name in ('pair_index','source_index'):require(np.array_equal(scores[name],reco[name]),'Candidate row mismatch '+name)
    current='reference_uniform_bdt_3p0_s71';winner=lock['portable_winner']
    require(np.array_equal(scores[current],e['score']),'Current model score changed')
    original=load_npz(candidate_dir/'candidate_scores.npz');rows={}
    for model in (current,winner):
        cut=lock['thresholds'][model];m=(original['split']==1)&(original['y']==1)
        require(int((original[model][m]>=cut).sum())==379 and int(m.sum())==420,'Validation threshold contract changed')
        rows[model]={}
        for p in ('sm','dy','jpsi'):
            mask=e['heldout']&(e['y']>=0)&((e['process']!='signal') if p=='sm' else e['process']==p)
            groups=bdt_performance._groups(e,mask)
            rows[model][p]={kind:bdt_performance.roc_summary(e['y'][mask],scores[model][mask],groups,repeats=0,
                weights=e['evaluation_weight'][mask] if kind=='weighted' else None) for kind in ('empirical','weighted')}
            rows[model][p]['threshold']=cut
            rows[model][p]['genuine_efficiency']=float(np.mean(scores[model][mask&(e['y']==1)]>=cut))
            rows[model][p]['accidental_rejection']=float(np.mean(scores[model][mask&(e['y']==0)]<cut))
    tr=C/'artifacts/sm_v10_phase2_fresh';train=load_npz(tr/'reco.npz');labels=load_npz(tr/'labels.npz');splits=load_npz(C/'artifacts/phase2_seed71_72/splits.npz')
    require(np.array_equal(train['ids'],splits['ids']) and np.array_equal(train['process'],splits['process']),'Original split mismatch')
    relative,names=relative_features(train['X'],train['feature_names'])
    features=single_feature_audit(relative,labels['y'],train['process'],splits['split'],names,names)
    report=dict(complete=True,models=rows,current=current,winner=winner,threshold_contract='Original validation 379/420 genuine retained; ties inclusive; candidate/reference primary cuts; historical cut preserved separately.',
        interpretation='Previously inspected held-out population: descriptive development evidence, not a new blind test. No production model replacement.',
        historical_current_cut=lock['original_balanced_thresholds'][current],relative_feature_validation=features,
        feature_cautions=['Per-leg pt error assignment was lost when errors were min/max sorted independently: exact per-leg relative pt errors cannot be recovered.',
            'Pair uncertainty/total-pt is a symmetric scale diagnostic, not the propagated uncertainty of pair transverse momentum.',
            'DCA/vertex-error is a scale ratio, not a calibrated DCA significance: covariance and DCA uncertainty are unavailable.',
            'Angle/angular-error is a scale ratio, not an angle significance: eta errors need metric transformation and covariances are unavailable.',
            'Lost-hit counts are all zero and valid hits equal valid muon hits: their fractions supply no extra information.',
            'Relative angles can still sculpt dimuon mass and displacement; no uniformity or deployment claim.'],
        references=[reference(p) for p in [candidate_dir/'completion.json',candidate_dir/'selection_locked.json',candidate_dir/'development_scores.npz',candidate_dir/'candidate_scores.npz',evaluation_path,cache/'reco.npz',tr/'reco.npz',tr/'labels.npz',C/'artifacts/phase2_seed71_72/splits.npz',Path(__file__)]])
    return report,e,scores


def render(report,e,scores,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=[report['current'],report['winner']];titles=['Current BDT (160 trees, depth 2)','Validation-selected candidate'];colors={1:'#2166ac',0:'#d95f02'}
    def save(fig,name):
        path=out/(name+'.pdf');require(not path.exists(),'Refusing plot overwrite');fig.savefig(path,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,(model,title) in enumerate(zip(models,titles)):
        for j,(p,ptitle) in enumerate([('sm','Pooled SM'),('dy','DY'),('jpsi','J/psi')]):
            ax=axes[i,j];mask=e['heldout']&((e['process']!='signal') if p=='sm' else e['process']==p)
            for y,label in [(1,'Genuine origin'),(0,'Accidental origin')]:
                v=scores[model][mask&(e['y']==y)];ax.hist(v,bins=np.linspace(0,1,61),weights=np.full(len(v),1/len(v)),histtype='step',color=colors[y],label=label,linewidth=1.7)
            r=report['models'][model][p];ax.axvline(r['threshold'],color='black',linestyle=':',label=f'Validation cut {r["threshold"]:.3f}')
            ax.set(title=f'{title}\n{ptitle}',xlabel='Unchanged model score',ylabel='Fraction within truth class',xlim=(0,1));ax.legend(fontsize=8)
            ax.text(.98,.54,f'Genuine retained {r["genuine_efficiency"]:.1%}\nAccidental rejected {r["accidental_rejection"]:.1%}',transform=ax.transAxes,ha='right',fontsize=8)
    fig.suptitle('Common-origin vs accidental pairs; candidate chosen on original validation only\nBoth cuts retain 379/420 validation genuine pairs; previously inspected held-out shapes',fontsize=12);save(fig,'diagnostic_candidate_truth_scores')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');palette={'sm':'black','dy':'#e69f00','jpsi':'#0072b2'}
    for i,model in enumerate(models):
        for p,ptitle in [('sm','Pooled SM'),('dy','DY'),('jpsi','J/psi')]:
            for ax,kind in zip(axes,('empirical','weighted')):
                r=report['models'][model][p][kind]
                if r['supported']:ax.plot(r['fpr'],r['tpr'],color=palette[p],linestyle='-' if i==0 else '--',label=f'{ptitle}, {"current" if i==0 else "candidate"}: {r["auc"]:.3f}')
    for ax,title in zip(axes,('Empirical pair counts','Weighted; supported populations only')):
        ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='Accidental pairs accepted',ylabel='Genuine pairs accepted',title=title);ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.2)
    axes[1].text(.02,.98,'Pooled weighted ROC omitted: too few effective accidental events.\nNo genuine held-out QCD pair; sparse signals omitted.',transform=axes[1].transAxes,va='top',fontsize=8)
    fig.suptitle('Validation-selected capacity comparison; previously inspected held-out development evidence\nImproved separation does not establish stable mass/displacement acceptance',fontsize=12);save(fig,'diagnostic_candidate_truth_rocs')
    rows=report['relative_feature_validation'];fig,ax=plt.subplots(figsize=(13,6),layout='constrained');yy=np.arange(len(rows))
    for offset,key,color,title in [(-.2,'train_auc','grey','Training'),(0,'validation_auc','black','Validation pooled')]:ax.scatter([r[key] for r in rows],yy+offset,label=title,color=color)
    for offset,p,color,title in [(.18,'dy','#e69f00','DY validation'),(.36,'jpsi','#0072b2','J/psi validation')]:ax.scatter([r['per_process_validation'][p]['auc'] for r in rows],yy+offset,label=title,color=color)
    ax.set_yticks(yy,[r['feature'] for r in rows]);ax.invert_yaxis();ax.axvline(.5,linestyle=':',color='grey');ax.set(xlabel='Single-observable AUC; polarity fixed on original training',xlim=(.3,.95));ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Relative reconstructed observables: original training / validation only\nScale ratios are diagnostics, not calibrated significances; no new feature deployed',fontsize=12);save(fig,'diagnostic_relative_feature_power')


def fit_population(scores, evaluation, population):
    """Choose evaluation rows explicitly; old heldout must never select new-fit test rows."""
    sm=evaluation['process']!='signal'
    if population=='expanded-test':
        require('expanded_split' in scores and 'expanded_heldout' in scores,'Expanded split is required; old heldout is unsafe')
        split=np.asarray(scores['expanded_split']);require(split.shape==sm.shape,'Expanded split shape mismatch')
        mask=(split==2)&sm
        require(np.array_equal(scores['expanded_heldout'],mask),'Expanded heldout does not match declared test split')
        registry={}
        for process,ids,part in zip(evaluation['process'][sm],evaluation['ids'][sm],split[sm]):
            key=(str(process),*map(int,ids));require(key not in registry or registry[key]==int(part),'Event crosses expanded splits');registry[key]=int(part)
        require(np.isin(split[sm],[0,1,2]).all(),'Unknown expanded SM split')
        return mask,split
    require(population=='original-heldout','Explicit fit population is required')
    require(np.array_equal(scores['heldout'],evaluation['heldout']),'Original heldout differs')
    return np.asarray(evaluation['heldout'])&sm,None


def threshold_counts(score,y,split,threshold):
    values=np.asarray(score)[(split==1)&(y==1)]
    require(len(values)>0,'No validation genuine pairs')
    passed=int((values>=threshold).sum());ties=int((values==threshold).sum());minimum=int(np.ceil(.9*len(values)))
    require(minimum<=passed<=minimum+ties,'Cut is not compatible with declared validation 90% target')
    return dict(genuine_pairs=len(values),passed_pairs=passed,efficiency=passed/len(values),ties_at_cut=ties,threshold=float(threshold))


def build_population(workspace,candidate_dir,population,models):
    C=workspace/'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier';sys.path.insert(0,str(C));import bdt_performance as perf
    complete=read(candidate_dir/'completion.json');lock=read(candidate_dir/'selection_locked.json')
    require(complete['complete'] and not lock['selection_uses_test'],'Candidate selection is not sealed before test inspection')
    for ref in [complete['selection'],complete['scores'],complete['development']['scores'],lock['validation'],lock['protocol']]:
        require(digest(ref['path'])==ref['sha256'],'Candidate receipt mismatch: '+ref['path'])
    scores=load_npz(candidate_dir/'development_scores.npz');original=load_npz(candidate_dir/'candidate_scores.npz')
    epath=workspace/'validation/bdt_first_look_20261012/performance/evaluation_rows.npz';frozen=read(epath.parent/'performance.json')
    require(digest(epath)==frozen['artifact_sha256']['evaluation_rows.npz'],'Frozen evaluation hash mismatch');e=load_npz(epath)
    cache=workspace/'validation/bdt_acceptance_study_20261012/comparison/extraction';reco=load_npz(cache/'reco.npz')
    cache_receipt=read(cache/'inputs.json')
    require(digest(cache/'reco.npz')==cache_receipt['reco_sha256'],'Reconstructed cache hash mismatch')
    for name in ('ids','process'):require(np.array_equal(scores[name],e[name]),'Candidate identity mismatch '+name)
    for name in ('pair_index','source_index'):require(np.array_equal(scores[name],reco[name]),'Candidate pair mismatch '+name)
    current='reference_uniform_bdt_3p0_s71';require(np.array_equal(scores[current],e['score']),'Frozen current scores changed')
    require(models and models[0]==current and len(models)==len(set(models)),'First model must be the unique current reference')
    test,split=fit_population(scores,e,population)
    if population=='expanded-test':
        train=(split==0)&(e['process']!='signal')&(e['y']==1)
    else:
        source=C/'artifacts/sm_v10_phase2_fresh';manifest=read(source/'inputs.json')
        require(digest(source/'inputs.json')==frozen['training_dataset_manifest_sha256'],'Original training manifest mismatch')
        for name in ('reco','labels'):require(digest(source/(name+'.npz'))==manifest[name+'_sha256'],'Original training array mismatch')
        require(digest(C/'artifacts/phase2_seed71_72/splits.npz')==frozen['training_split_sha256'],'Original split digest mismatch')
        tr=load_npz(source/'reco.npz');tl=load_npz(source/'labels.npz');sp=load_npz(C/'artifacts/phase2_seed71_72/splits.npz')
        require(np.array_equal(tr['ids'],sp['ids']) and np.array_equal(tr['process'],sp['process']),'Original training split mismatch')
        train=(sp['split']==0)&(tl['y']==1)
    nuisance=reco['nuisance'];training_nuisance=nuisance[train] if population=='expanded-test' else tr['nuisance'][train]
    edges=[np.unique(np.quantile(training_nuisance[np.isfinite(training_nuisance[:,i]),i],[1/3,2/3])) for i in range(3)]
    result=dict(schema='shift-bdt-explicit-population-diagnostics-v1',complete=True,population=population,
        population_description='Expanded hash-split test only; all new-fit training and validation events excluded' if population=='expanded-test' else 'Earlier adaptive negative control; original heldout only, original training-derived bins',
        interpretation='Previously inspected development evidence, not a blind test. Genuine origin is not a vertex-fit accuracy guarantee.',
        model_order=models,threshold_counts={},roc={},acceptance={},population_counts={},training_bin_edges=[x.tolist() for x in edges],
        bootstrap_replicates=200,ci_definition='95% event-resampled bootstrap; descriptive conditional acceptance. Support requires at least20 events (weighted: also20 effective events).',
        old_heldout_excluded_new_train_or_validation=int((e['heldout']&~test&(e['process']!='signal')).sum()) if population=='expanded-test' else None)
    for model in models:
        require(model in scores and model in lock['thresholds'],'Missing selected model '+model)
        threshold=float(lock['thresholds'][model])
        if population=='expanded-test':result['threshold_counts'][model]=threshold_counts(scores[model],np.where(e['process']=='signal',-1,e['y']),split,threshold)
        else:result['threshold_counts'][model]=threshold_counts(original[model],original['y'],original['split'],threshold)
        result['roc'][model]={};result['acceptance'][model]={}
        for p in ('sm','dy','jpsi'):
            pop=test&((e['process']!='signal') if p=='sm' else e['process']==p);mask=pop&(e['y']>=0)
            result['population_counts'][p]=dict(genuine=int((pop&(e['y']==1)).sum()),accidental=int((pop&(e['y']==0)).sum()),unknown=int((pop&(e['y']<0)).sum()))
            groups=perf._groups(e,mask)
            result['roc'][model][p]={kind:perf.roc_summary(e['y'][mask],scores[model][mask],groups,repeats=0,weights=e['evaluation_weight'][mask] if kind=='weighted' else None) for kind in ('empirical','weighted')}
            result['roc'][model][p]['genuine_efficiency']=perf.efficiency_summary(scores[model][pop&(e['y']==1)]>=threshold,perf._groups(e,pop&(e['y']==1)),repeats=200)
            result['roc'][model][p]['accidental_rejection']=perf.efficiency_summary(scores[model][pop&(e['y']==0)]<threshold,perf._groups(e,pop&(e['y']==0)),repeats=200)
            result['acceptance'][model][p]=[]
            for i,edge in enumerate(edges):
                bins=np.searchsorted(edge,nuisance[:,i],side='right');row=[]
                for b in range(len(edge)+1):
                    m=pop&(e['y']==1)&np.isfinite(nuisance[:,i])&(bins==b);groups=perf._groups(e,m);passed=scores[model][m]>=threshold
                    empirical=perf.efficiency_summary(passed,groups,repeats=200);weighted=perf.weighted_efficiency(passed,groups,e['evaluation_weight'][m],repeats=200)
                    empirical['supported']=empirical['events']>=20
                    weighted['supported']=weighted['available'] and weighted['events']>=20 and weighted['event_weight_effective_n']>=20
                    row.append(dict(bin=b,empirical=empirical,weighted=weighted))
                result['acceptance'][model][p].append(row)
    result['references']=[reference(p) for p in [candidate_dir/'completion.json',candidate_dir/'selection_locked.json',candidate_dir/'candidate_scores.npz',candidate_dir/'development_scores.npz',epath,epath.parent/'performance.json',cache/'inputs.json',cache/'reco.npz',C/'bdt_performance.py',Path(__file__)]]
    if population=='original-heldout':result['references'] += [reference(p) for p in [source/'inputs.json',source/'reco.npz',source/'labels.npz',C/'artifacts/phase2_seed71_72/splits.npz']]
    return result,e,scores,test


def render_population(report,e,scores,test,out,prefix):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    models=report['model_order'];colors=['#222222','#d95f02','#0072b2','#009e73'];titles=['Current reference','Rejection-selected candidate','Acceptance-selected candidate','Additional candidate']
    if len(models)==2 and models[1]=='adaptive_d2_s1_n400':titles[1]='Earlier adaptive negative control'
    model_titles=dict(zip(models,titles));report['model_titles']=model_titles
    population='Expanded test split only' if report['population']=='expanded-test' else 'Original heldout: separate negative control'
    paths=[]
    def save(fig,name):
        path=out/(prefix+'_'+name+'.pdf');require(not path.exists(),'Refusing PDF overwrite');fig.savefig(path,bbox_inches='tight');plt.close(fig);paths.append(path)
    processes=[('sm','Pooled SM'),('dy','DY'),('jpsi','J/psi')]
    fig,axes=plt.subplots(len(models),3,figsize=(14,3.4*len(models)),layout='constrained',squeeze=False)
    for i,model in enumerate(models):
        for ax,(p,title) in zip(axes[i],processes):
            pop=test&((e['process']!='signal') if p=='sm' else e['process']==p)
            for y,color,label in [(1,'#2166ac','Genuine'),(0,'#d95f02','Accidental')]:
                v=scores[model][pop&(e['y']==y)];ax.hist(v,bins=np.linspace(0,1,61),weights=np.full(len(v),1/len(v)),histtype='step',color=color,label=f'{label} (n={len(v):,})',linewidth=1.5)
            cut=report['threshold_counts'][model]['threshold'];ax.axvline(cut,color='black',linestyle=':',label=f'Validation cut {cut:.3f}')
            ax.set(title=model_titles[model]+' — '+title,xlabel='Unchanged model score',ylabel='Fraction within truth class',xlim=(0,1));ax.legend(fontsize=8)
    fig.suptitle(population+'; direct truth-class separation\nEach cut is fixed at 90% genuine retention on its declared validation population',fontsize=12);save(fig,'truth_scores')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');palette={'sm':'black','dy':'#e69f00','jpsi':'#0072b2'};styles=['-','--',':','-.']
    for i,model in enumerate(models):
        for p,title in processes:
            for ax,kind in zip(axes,('empirical','weighted')):
                r=report['roc'][model][p][kind]
                if r['supported']:ax.plot(r['fpr'],r['tpr'],linestyle=styles[i],color=palette[p],label=f'{title}, {model_titles[model]}: {r["auc"]:.3f}')
    for ax,title in zip(axes,('Empirical pair-count ROC','Weighted ROC; only supported populations')):
        ax.plot([0,1],[0,1],color='grey',linestyle=':');ax.set(title=title,xlabel='Accidental pairs accepted',ylabel='Genuine pairs accepted');ax.legend(fontsize=7,loc='lower right');ax.grid(alpha=.2)
    omitted=[p for p,title in processes if not report['roc'][models[0]][p]['weighted']['supported']]
    axes[1].text(.02,.98,'Omitted for low effective-event support: '+(', '.join(omitted) or 'none')+'\nQCD has no genuine test pair; signals excluded.',transform=axes[1].transAxes,va='top',fontsize=8)
    fig.suptitle(population+'; previously inspected development evidence',fontsize=12);save(fig,'truth_rocs')
    dimensions=[('Reconstructed mass','GeV',1.),('Fitted vertex z','m',.01),('Fitted vertex radius','m',.01)]
    for kind in ('empirical','weighted'):
        fig,axes=plt.subplots(3,3,figsize=(15,10),layout='constrained')
        for row,(p,title) in enumerate(processes):
            for dimension,(dimension_title,unit,scale) in enumerate(dimensions):
                ax=axes[row,dimension];edge=np.asarray(report['training_bin_edges'][dimension])*scale
                labels=[f'< {edge[0]:.3g}',f'{edge[0]:.3g}–{edge[1]:.3g}',f'≥ {edge[1]:.3g}']
                for i,model in enumerate(models):
                    bins=report['acceptance'][model][p][dimension]
                    for b,entry in enumerate(bins):
                        r=entry[kind]
                        if not r['supported'] or r['efficiency'] is None:continue
                        low,high=r['ci95_event_bootstrap'];value=r['efficiency']
                        ax.errorbar(b+(i-(len(models)-1)/2)*.16,value,yerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',color=colors[i],capsize=3,label=model_titles[model] if b==0 else None,markersize=4)
                base=report['acceptance'][models[0]][p][dimension]
                annotation='n = '+', '.join(str(b['empirical']['pairs']) for b in base)
                if kind=='weighted':annotation+='\nEffective events = '+', '.join(f'{b[kind].get("event_weight_effective_n",0):.1f}' for b in base)
                ax.text(.03,.04,annotation,transform=ax.transAxes,fontsize=8);ax.axhline(.9,color='grey',linestyle=':');ax.set_xticks(range(3),labels);ax.set(title=f'{title}: {dimension_title}',xlabel=unit,ylabel='Genuine pairs retained',ylim=(0,1.03));ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(fontsize=7,loc='lower left',bbox_to_anchor=(0,.13))
        fig.suptitle(population+f'; {kind} genuine acceptance with 95% event-bootstrap intervals\nBins fixed from genuine training events; unsupported (<20 effective events) bins omitted' if kind=='weighted' else population+'; empirical genuine acceptance with 95% event-bootstrap intervals\nBins fixed from genuine training events; common origin does not certify fit accuracy',fontsize=12)
        save(fig,kind+'_acceptance')
    report['plots']=[reference(p) for p in paths]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path.cwd());parser.add_argument('--candidates',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plots',type=Path,required=True)
    parser.add_argument('--population',choices=['expanded-test','original-heldout']);parser.add_argument('--models',nargs='+');parser.add_argument('--plot-prefix',default='expanded');args=parser.parse_args()
    if args.population:
        require(not args.output.exists(),'Refusing report overwrite');report,e,scores,test=build_population(args.workspace.resolve(),args.candidates.resolve(),args.population,args.models)
        args.output.mkdir(parents=True);args.plots.mkdir(parents=True,exist_ok=True);render_population(report,e,scores,test,args.plots,args.plot_prefix)
        (args.output/'diagnostics.json').write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps({'complete':True,'output':str(args.output)}));return
    require(not args.output.exists(),'Refusing report overwrite');report,e,scores=build(args.workspace.resolve(),args.candidates.resolve());args.output.mkdir(parents=True);render(report,e,scores,args.plots)
    report['plots']=[reference(args.plots/(p+'.pdf')) for p in ['diagnostic_candidate_truth_scores','diagnostic_candidate_truth_rocs','diagnostic_relative_feature_power']]
    (args.output/'diagnostics.json').write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps({'complete':True,'output':str(args.output)}))


if __name__=='__main__':main()
