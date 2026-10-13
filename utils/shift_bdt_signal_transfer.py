#!/usr/bin/env python3
"""Count sparse signal transfer at sealed validation cuts, without fitting."""
import json
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


def main():
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
    plot=root/'tea_shift_cmssw/plots/BDT_improvement_20261012/signal_transfer_genuine_retention.pdf';require(not plot.exists(),'Refusing PDF overwrite');fig.savefig(plot,bbox_inches='tight');plt.close(fig)
    result=dict(complete=True,points=rows,models=models,counts=dict(all_pairs=39,signal_genuine=26,other_common_origin=1,unknown=11,accidental=1),
        interpretation='Sparse signal transfer only. Cuts fixed on expanded SM validation; signals never fitted or selected. No efficiency uncertainty or statistical independence claim. Historical current cut is secondary JSON only.',
        references=[reference(p) for p in [training/'completion.json',training/'selection_locked.json',training/'development_scores.npz',frozen/'performance.json',frozen/'evaluation_rows.npz',Path(__file__)]],plot=reference(plot))
    (output/'summary.json').write_text(json.dumps(clean(result),indent=2)+'\n');print(json.dumps(dict(report=str(output/'summary.json'),plot=str(plot),totals={m:[sum(r['models'][m]['passed'] for r in rows),26] for m in models})))


if __name__=='__main__':main()
