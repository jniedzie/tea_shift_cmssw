#!/usr/bin/env python3
"""PDF-only origin-label diagnostics using frozen scores and original splits.

Feature directions/ranking use original training/validation only. No fitting,
relabeling, threshold change, score remapping or input ROOT scan is performed.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def require(value, message):
    if not value:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def source_function(path, name):
    """Reuse a pure frozen training helper without importing neural runtimes."""
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'np':np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def single_feature_audit(X, y, process, split, names, quality_names):
    from sklearn.metrics import roc_auc_score
    train, validation = (split == 0) & (y >= 0), (split == 1) & (y >= 0)
    rows = []
    for index, name in enumerate(names):
        finite_train = X[train, index][np.isfinite(X[train,index])]
        median = float(np.median(finite_train)) if len(finite_train) else 0.
        values = np.where(np.isfinite(X[:,index]), X[:,index], median)
        auc_train = float(roc_auc_score(y[train], values[train]))
        direction = 1 if auc_train >= .5 else -1
        row = dict(feature=str(name), index=index, quality_only=str(name) in quality_names,
                   direction_from_training=direction, training_median=median,
                   train_auc=max(auc_train,1-auc_train),
                   validation_auc=float(roc_auc_score(y[validation], direction*values[validation])),
                   per_process_validation={})
        for p in np.unique(process):
            mask = validation & (process == p)
            counts = [int((mask & (y == label)).sum()) for label in (0,1)]
            row['per_process_validation'][str(p)] = dict(accidental=counts[0], genuine=counts[1],
                supported=min(counts)>=20,
                auc=float(roc_auc_score(y[mask], direction*values[mask])) if min(counts)>0 else None)
        rows.append(row)
    return sorted(rows, key=lambda row:(-row['validation_auc'], row['feature']))


def input_audit(X, names, imputer):
    rows = []
    for index,name in enumerate(names):
        values = X[:,index]; finite = np.isfinite(values); present = values[finite]
        physical_nonnegative = (str(name).endswith('Err') or any(k in str(name) for k in
            ('Chi2','nValid','nLost','nMuonStations','topology','recoAlgorithm','probability')) or name=='pair_dca')
        rows.append(dict(feature=str(name), nonfinite=int((~finite).sum()),
            imputed_rows=int((~finite).sum()), frozen_imputation_value=float(imputer[index]),
            physical_nonnegative=physical_nonnegative,
            negative_in_nonnegative_field=int((present<0).sum()) if physical_nonnegative else None,
            minus_one=int((present==-1).sum()), zero=int((present==0).sum()),
            unique_finite_values=len(np.unique(present)),
            quantiles_0_1_50_99_100=np.quantile(present,[0,.01,.5,.99,1]).tolist() if len(present) else None))
    duplicates = [[str(names[i]),str(names[j])] for i in range(len(names)) for j in range(i)
                  if np.array_equal(X[:,i], X[:,j], equal_nan=True)]
    return dict(rows=len(X), features=len(names), total_nonfinite=int((~np.isfinite(X)).sum()),
                per_feature=rows, exact_duplicate_columns=duplicates,
                interpretation='Finite sentinels are retained by the deployed imputer. Negative coordinates/momenta are physical and are not treated as missing. Zero lost-hit counts and topology/reconstruction codes are not missing-value sentinels.')


def tree_votes(model, X):
    selected = X[:,model['columns']]
    values = np.where(np.isfinite(selected), selected, np.asarray(model['imputer_statistics'])).astype(np.float32).astype(float)
    votes = []
    for tree in model['trees']:
        left,right = np.array(tree['children_left']),np.array(tree['children_right'])
        feature,threshold = np.array(tree['feature']),np.array(tree['threshold'])
        nodes = np.zeros(len(values),int)
        while (left[nodes]>=0).any():
            active=np.flatnonzero(left[nodes]>=0);current=nodes[active]
            nodes[active]=np.where(values[active,feature[current]]<=threshold[current],left[current],right[current])
        classes=np.array([0 if v is None else v for v in tree['leaf_class']])
        votes.append(2*classes[nodes]-1)
    return np.asarray(votes).T


def build(workspace):
    from sklearn.metrics import roc_auc_score
    classifier=workspace/'CMSSW_17_0_0_pre4/src/PhysicsTools/ShiftDimuonClassifier'
    sys.path.insert(0,str(classifier))
    import bdt_performance, features, portable_inference
    cache=workspace/'validation/bdt_acceptance_study_20261012/comparison/extraction'
    performance=workspace/'validation/bdt_first_look_20261012/performance'
    training=classifier/'artifacts/sm_v10_phase2_fresh';results=classifier/'artifacts/phase2_seed71_72'
    model_path=workspace/'validation/bdt_acceptance_study_20261012/production/frozen/scoring/classifier/model.json'
    manifest, report=read(cache/'inputs.json'),read(performance/'performance.json')
    for name in ['reco','labels']:
        require(digest(cache/(name+'.npz'))==manifest[name+'_sha256'],'Changed extraction '+name)
    require(digest(performance/'evaluation_rows.npz')==report['artifact_sha256']['evaluation_rows.npz'],'Changed frozen evaluation')
    reco,labels,evaluation=load_npz(cache/'reco.npz'),load_npz(cache/'labels.npz'),load_npz(performance/'evaluation_rows.npz')
    require(np.array_equal(reco['ids'],evaluation['ids']) and np.array_equal(labels['y'],evaluation['y']) and
            np.array_equal(reco['process'],evaluation['process']),'Frozen population mismatch')
    original=read(training/'inputs.json')
    require(digest(training/'inputs.json')==report['training_dataset_manifest_sha256'],'Wrong original dataset')
    for name in ['reco','labels']:
        require(digest(training/(name+'.npz'))==original[name+'_sha256'],'Changed training '+name)
    require(digest(results/'splits.npz')==report['training_split_sha256'],'Changed training split')
    tr,tl,splits=load_npz(training/'reco.npz'),load_npz(training/'labels.npz'),load_npz(results/'splits.npz')
    require(np.array_equal(tr['ids'],splits['ids']) and np.array_equal(tr['process'],splits['process']),'Misaligned original split')
    require(tuple(reco['feature_names'])==features.FEATURE_NAMES==tuple(tr['feature_names']),'Feature contract mismatch')
    model=portable_inference.load_model(model_path)
    require(digest(model_path)==report['model_sha256'],'Changed benchmark model')
    predicted=portable_inference.predict(model,reco['X'])
    require(np.array_equal(predicted,evaluation['score']),'Frozen score parity failed')
    history=read(results/'uniform_bdt_3p0_s71_training.json')
    phase_weights=source_function(classifier/'phase2_train.py','phase_weights')
    mixture={}
    for part,label in [(0,'train'),(1,'validation')]:
        mask=(splits['split']==part)&(tl['y']>=0); y,p=tl['y'][mask],tr['process'][mask]
        w=phase_weights(y,p);mixture[label]=[]
        for process in np.unique(p):
            for truth in (0,1):
                subset=(p==process)&(y==truth)
                mixture[label].append(dict(process=str(process),truth=truth,pairs=int(subset.sum()),
                     fraction_of_training_weight=float(w[subset].sum()/w.sum()),
                     fraction_of_class_weight=float(w[subset].sum()/w[y==truth].sum())))
    audit=dict(schema='shift-bdt-discrimination-diagnostics-v1',complete=True,
        score_unchanged=True, labels_unchanged=True, benchmark_threshold=report['threshold'],
        heldout_population='Original test plus unseen events; previously inspected, so descriptive development evidence, not a new blind test.',
        feature_ranking_population='Original SM training sets polarity/imputation; original validation ranks features. No test or newly scored sample selects features.',
        input_quality=input_audit(reco['X'],reco['feature_names'],model['imputer_statistics']),
        training_input_quality=input_audit(tr['X'][splits['split']!=2],tr['feature_names'],model['imputer_statistics']),
        training_mixture=mixture,
        raw_features=single_feature_audit(tr['X'],tl['y'],tr['process'],splits['split'],tr['feature_names'],features.QUALITY_NAMES),
        score_populations={},roc={},training_history={k:v for k,v in history.items() if k!='rounds'})
    y,p,score=evaluation['y'],evaluation['process'],evaluation['score'];held=evaluation['heldout']
    for process in ['sm','dy','jpsi','qcd','signal']:
        population=held & ((p!='signal') if process=='sm' else (p==process));audit['score_populations'][process]={}
        for truth,name in [(1,'genuine'),(0,'accidental'),(-1,'unknown')]:
            mask=population&(y==truth);values=score[mask]
            audit['score_populations'][process][name]=dict(pairs=int(mask.sum()),passed=int((values>=report['threshold']).sum()),
                quantiles_0_10_50_90_100=np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else None,
                unique_scores=len(np.unique(values)))
        mask=population&(y>=0)
        groups=bdt_performance._groups(evaluation,mask)
        audit['roc'][process]={kind:bdt_performance.roc_summary(y[mask],score[mask],groups,repeats=0,minimum=20,
                    weights=evaluation['evaluation_weight'][mask] if kind=='weighted' else None)
                    for kind in ('empirical','weighted')}
    pooled=held&(p!='signal')&(y>=0)
    require(abs(audit['roc']['sm']['empirical']['auc']-report['roc']['heldout_genuine_vs_all_accidental']['auc'])<1e-14,'ROC differs from frozen result')
    votes=tree_votes(model,reco['X']);coefficient=np.array([t['coefficient'] for t in model['trees']])
    normalized_margin=np.zeros(len(score))
    for i,c in enumerate(coefficient):normalized_margin+=c*votes[:,i]
    normalized_margin/=model['sum_abs_coefficients']
    agreement=(normalized_margin+1)/2
    split_counts=Counter(features.FEATURE_NAMES[model['columns'][f]] for tree in model['trees'] for f in tree['feature'] if f>=0)
    audit['score_compression']=dict(formula=model['score_formula'],theoretical_range=[1/(1+np.exp(2)),1/(1+np.exp(-2))],
        actual_range=[float(score.min()),float(score.max())],trees=len(model['trees']),sum_abs_coefficients=model['sum_abs_coefficients'],
        uncalibrated=True,weighted_positive_vote_fraction_quantiles=np.quantile(agreement[pooled],[0,.1,.5,.9,1]).tolist(),
        score_polarity_auc=audit['roc']['sm']['empirical']['auc'],reversed_score_auc=float(roc_auc_score(y[pooled],-score[pooled])),
        feature_split_counts=dict(split_counts),final_positive_training_weight_effective_n=history['rounds'][-1]['positive_weight_effective_n'],
        first_tree_error=history['rounds'][0]['weighted_error'],last_tree_error=history['rounds'][-1]['weighted_error'],
        interpretation='The score compresses a normalized signed tree vote by design. Weak late trees and overlapping reconstructed features leave mixed votes. Stretching this score cannot improve ordering or AUC.')
    label_audit=workspace/'validation/bdt_acceptance_study_20261012/label_audit/unknown_labels.json'
    limits=workspace/'validation/bdt_acceptance_study_20261012/comparison/label_identification_limits.json'
    audit['label_audit']=read(label_audit)['details']
    audit['identification_limits']=read(limits)['models']['uniform_bdt_3p0_s71']['processes']
    audit['ceiling_interpretation']='No irreducible classification ceiling is measured. Unknown labels are excluded, origin truth does not certify vertex precision, and process composition differs between classes. Count-only unknown-label bounds assume all known labels are correct and are not physics uncertainties.'
    paths=[cache/'inputs.json',cache/'reco.npz',cache/'labels.npz',performance/'performance.json',performance/'evaluation_rows.npz',
           training/'inputs.json',training/'reco.npz',training/'labels.npz',results/'splits.npz',model_path,
           results/'uniform_bdt_3p0_s71_training.json',classifier/'phase2_train.py',classifier/'features.py',classifier/'uniform_bdt.py',
           classifier/'portable_inference.py',classifier/'bdt_performance.py',label_audit,limits,Path(__file__)]
    audit['references']=[reference(path) for path in paths]
    return audit,dict(reco=reco,evaluation=evaluation,training=tr,training_labels=tl,split=splits['split'],history=history)


def render(audit,data,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':13})
    e=data['evaluation'];held=e['heldout'];score=e['score'];y=e['y'];p=e['process'];cut=audit['benchmark_threshold']
    plain={'mu_lowPt_pz':'Lower-pt leg longitudinal momentum','mu_lowPt_energy':'Lower-pt leg energy',
           'pair_pz':'Pair longitudinal momentum','pair_vz':'Fitted vertex z',
           'mu_highPt_energy':'Higher-pt leg energy','mu_highPt_pz':'Higher-pt leg longitudinal momentum',
           'pair_normalizedChi2':'Vertex fit chi2 / degrees of freedom','pair_probability':'Vertex fit probability',
           'mu_min_nValidHits':'Smaller valid hit count','mu_min_nValidMuonHits':'Smaller valid muon-hit count',
           'mu_min_topology':'Smaller track topology code','mu_max_nValidHits':'Larger valid hit count',
           'pair_originCompatibilityNormalizedChi2':'Track-origin compatibility chi2 / degrees of freedom',
           'pair_dca':'Track distance of closest approach (cm)'}
    colors={1:'#2767a5',0:'#d55e00',-1:'#777777'};names={1:'Genuine origin',0:'Accidental origin',-1:'Unknown match'}
    processes=[('sm','Pooled SM'),('dy','DY'),('jpsi','J/psi'),('qcd','QCD'),('signal','Signal grid (sparse)')]
    bins=np.linspace(0,1,61)
    def finish(fig,name):
        path=directory/('diagnostic_'+name+'.pdf');require(not path.exists(),'Refuse plot overwrite')
        fig.savefig(path,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(process,title) in zip(axes.flat,processes):
        pop=held&((p!='signal') if process=='sm' else p==process)
        for label in [1,0,-1]:
            values=score[pop&(y==label)]
            if len(values):ax.hist(values,bins=bins,weights=np.ones(len(values))/len(values),histtype='step',linewidth=1.8,
                color=colors[label],linestyle='--' if label<0 else '-',label=f'{names[label]} (n={len(values):,})')
        ax.axvline(cut,color='black',linestyle=':',label=f'Frozen cut {cut:.3f}')
        ax.set(title=title,xlabel='Unchanged BDT score',ylabel='Fraction per score bin',xlim=(0,1));ax.legend(fontsize=8)
        if process=='qcd':ax.text(.04,.82,'No held-out genuine QCD pair',transform=ax.transAxes,fontsize=9)
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Process is not a truth class.\n\nDY and J/psi contain accidental pairs;\nQCD has no genuine held-out pair.\n\nEach truth-class curve has unit area.\nUnknowns remain a separate diagnostic.\n\nGenuine means common muon origin,\nnot an accurate reconstructed vertex.\n\nPreviously inspected held-out population;\nno new blind-test claim.',va='top',fontsize=11)
    fig.suptitle('Direct genuine-versus-accidental score comparison');finish(fig,'truth_class_scores')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for i,(process,title) in enumerate(processes[1:]):
        counts=np.array([audit['score_populations'][process][names_]['pairs'] for names_ in ['genuine','accidental','unknown']])
        bottom=0
        for label,count in zip([1,0,-1],counts):
            fraction=count/counts.sum();axes[0].bar(i,fraction,bottom=bottom,color=colors[label],label=names[label] if i==0 else None)
            if fraction>.06:axes[0].text(i,bottom+fraction/2,f'{count:,}\n{fraction:.1%}',ha='center',va='center',color='white',fontsize=9)
            bottom+=fraction
    axes[0].set_xticks(range(4),['DY','J/psi','QCD','Signal']);axes[0].set(ylabel='Fraction of retained pairs',title='Process overlays mix different truth fractions');axes[0].legend(fontsize=9)
    train=audit['training_mixture']['train'];palette={'dy':'#e69f00','jpsi':'#56b4e9','qcd':'#009e73'}
    for label in [0,1]:
        bottom=0
        for process in ['dy','jpsi','qcd']:
            row=next(r for r in train if r['truth']==label and r['process']==process);v=row['fraction_of_class_weight']
            axes[1].bar(label,v,bottom=bottom,color=palette[process],label=process if label==0 else None)
            if v:axes[1].text(label,bottom+v/2,f'{process}: {v:.1%}\n{row["pairs"]} training pairs',ha='center',va='center',fontsize=9)
            bottom+=v
    axes[1].set_xticks([0,1],['Accidental class','Genuine class']);axes[1].set(ylabel='Fraction of initial class training weight',title='Training balances classes and processes; not cross sections')
    axes[1].text(.5,-.18,'Each class has total weight 1/2. No signal training; no genuine QCD training examples.',ha='center',transform=axes[1].transAxes,fontsize=9)
    fig.suptitle('Truth-class mixture and training priors');finish(fig,'process_and_training_mixture')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');palette={'sm':'black','dy':'#e69f00','jpsi':'#0072b2'}
    weighted_notes=[]
    for process,title in processes:
        for ax,kind in zip(axes,['empirical','weighted']):
            roc=audit['roc'][process][kind]
            if process not in palette:continue
            if roc['supported']:
                label=f'{title}: AUC {roc["auc"]:.3f}'
                if kind=='weighted':label+=f' (N_eff−={roc["negative_weight_effective_events"]:.1f})'
                ax.plot(roc['fpr'],roc['tpr'],color=palette[process],label=label)
            elif kind=='weighted':weighted_notes.append(f'{title} omitted: N_eff−={roc.get("negative_weight_effective_events",0):.2f} < 20')
    for ax,title in zip(axes,['Empirical pair-count ROC','Weighted ROC: supported populations only']):
        ax.plot([0,1],[0,1],':',color='grey');ax.set(xlabel='Accidental pairs accepted',ylabel='Genuine pairs accepted',title=title,xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    axes[0].text(.02,.97,'QCD: no genuine pair; signal: only one accidental pair.\nTheir within-process ROCs are unsupported.',transform=axes[0].transAxes,va='top',fontsize=8)
    axes[1].text(.02,.97,'\n'.join(weighted_notes)+'\nWeights: sampling correction × cross section / full GEN sumw.\nCurves are descriptive; no new uncertainty fit.',transform=axes[1].transAxes,va='top',fontsize=8)
    fig.suptitle('Same-process truth separation; previously inspected held-out pairs');finish(fig,'truth_class_rocs')
    history=data['history'];rounds=history['rounds'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=np.arange(1,len(rounds)+1)
    axes[0,0].plot(x,[r['weighted_error'] for r in rounds]);axes[0,0].axhline(.5,color='grey',linestyle=':');axes[0,0].set(title='Late shallow trees are weak under updated weights',xlabel='Boosting round',ylabel='Weighted tree error')
    axes[0,1].plot(x,[r['coefficient'] for r in rounds]);axes[0,1].set(title='160 depth-two signed-vote trees',xlabel='Boosting round',ylabel='Tree coefficient')
    axes[1,0].plot(x,[r['positive_weight_effective_n'] for r in rounds]);axes[1,0].set(title='Adaptive weights concentrate on difficult positives',xlabel='Boosting round',ylabel='Effective positive training count')
    axes[1,1].axis('off');compression=audit['score_compression'];axes[1,1].text(0,.98,
        'Score = sigmoid(2 × signed tree margin / Σ|coefficient|)\n\n'
        'Theoretical range: 0.1192–0.8808, even for unanimous votes.\n'
        f'Observed full-cache range: {compression["actual_range"][0]:.3f}–{compression["actual_range"][1]:.3f}.\n\n'
        'This is an uncalibrated discriminator, not P(genuine).\n'
        'Its polarity is correct: larger means more genuine-like.\n'
        'Stretching the score would leave ordering/AUC unchanged.\n\n'
        f'Nonfinite/imputed values in 20,212 × 50 inputs: {audit["input_quality"]["total_nonfinite"]}.\n'
        'Compression is not caused by missing-value imputation.\n\n'
        'Uniformity reweighting targets genuine efficiency across\n'
        'reconstructed mass/z/radius; it trades separation for\n'
        'acceptance stability and is not a flatness guarantee.',va='top',fontsize=10)
    fig.suptitle('Why the existing BDT score occupies a narrow range');finish(fig,'score_compression')
    ranked=audit['raw_features'];selected=ranked[:8]+[row for row in ranked if row['quality_only']][:6];unique=[]
    for row in selected:
        if row['feature'] not in [r['feature'] for r in unique]:unique.append(row)
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained');yy=np.arange(len(unique))
    for offset,key,color,label in [(-.2,'train_auc','#777777','Training'),(0,'validation_auc','#222222','Validation pooled')]:
        ax.scatter([r[key] for r in unique],yy+offset,color=color,label=label,s=24)
    for offset,process,color,title in [(.16,'dy','#e69f00','DY validation'),(.32,'jpsi','#0072b2','J/psi validation')]:
        valid=[(i,r['per_process_validation'][process]) for i,r in enumerate(unique) if r['per_process_validation'][process]['supported']]
        ax.scatter([v['auc'] for i,v in valid],[i+offset for i,v in valid],color=color,label=title,s=23)
    ax.axvline(.5,color='grey',linestyle=':');ax.set_yticks(yy,[plain.get(r['feature'],r['feature'])+(' ↑' if r['direction_from_training']==1 else ' ↓') for r in unique]);ax.invert_yaxis()
    ax.set(xlabel='Single-feature AUC (direction fixed on original training)',xlim=(.35,.95),title='Which existing reconstructed observables carry separation?');ax.legend(loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Original training/validation only; no test-based feature selection\nPooled kinematic gains may depend on process, mass and displacement mixtures',fontsize=12);finish(fig,'raw_feature_power')
    selected=[next(row for row in ranked if row['feature']==name) for name in
              ['pair_normalizedChi2','mu_min_nValidMuonHits','pair_originCompatibilityNormalizedChi2','pair_dca']]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    X=data['training']['X'];ly=data['training_labels']['y'];split=data['split']
    for ax,row in zip(axes.flat,selected):
        i=row['index'];values=X[:,i];tr=(split==0)&(ly>=0);edges=np.unique(np.quantile(values[tr],np.linspace(0,1,21)))
        for label in [1,0]:
            mask=(split==1)&(ly==label);v=np.clip(values[mask],edges[0],edges[-1]);counts,_=np.histogram(v,bins=edges)
            ax.stairs(counts/counts.sum(),edges,color=colors[label],label=f'{names[label]} (n={int(mask.sum())})',linewidth=1.8)
        ax.set(title=plain.get(row['feature'],row['feature']),xlabel='Raw observable (tails folded into training range)',ylabel='Validation fraction per training bin');ax.legend(fontsize=8)
    fig.suptitle('Quality features separate partly; their distributions still overlap');finish(fig,'quality_feature_distributions')
    limits=audit['identification_limits'];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,(process,title) in enumerate(processes[:4]):
        r=limits[process]
        for ax,key,known,label in [(axes[0],'genuine_efficiency_extrema',r['known_common_passed']/r['known_common'] if r['known_common'] else None,'Genuine efficiency'),
                                   (axes[1],'accidental_rejection_extrema',r['known_accidental_rejected']/r['known_accidental'] if r['known_accidental'] else None,'Accidental rejection')]:
            lo,hi=r[key];ax.hlines(i,lo,hi,color='#777777',linewidth=6,alpha=.5)
            if known is not None:ax.plot(known,i,'o',color='#0072b2')
            ax.set(xlabel=label,xlim=(0,1),ylim=(3.5,-.5));ax.set_yticks(range(4),['Pooled SM','DY','J/psi','QCD']);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Unknown-label identification limits at the unchanged BDT cut\nBlue: known-label result. Grey: possible unknown assignments; not uncertainty intervals.',fontsize=12)
    fig.text(.5,-.04,'Assumes known labels are correct. No measured irreducible classification ceiling; common origin does not prove fitted-vertex accuracy.',ha='center',fontsize=9)
    finish(fig,'label_identification_limits')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path.cwd());parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);args.plots.mkdir(parents=True,exist_ok=True)
    audit,data=build(args.workspace.resolve());render(audit,data,args.plots)
    audit['plots']=[reference(p) for p in sorted(args.plots.glob('diagnostic_*.pdf'))]
    with (args.output/'diagnostics.json').open('x') as stream:json.dump(clean(audit),stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=True,report=str(args.output/'diagnostics.json'),pdfs=len(audit['plots']),imputed_values=audit['input_quality']['total_nonfinite'])))


if __name__=='__main__':main()
