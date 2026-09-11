#!/usr/bin/env python3
"""Reco-only distributions and independent hit-truth performance; no angular cuts.

ROOT input is read directly, so this also audits existing productions without
another reconstruction. All reco rows contribute to reco/coverage diagnostics.
Truth resolution requires a stored hit association, never a nearest direction.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)
COLORS = [ROOT.kAzure + 2, ROOT.kOrange + 7]
TOPO = ['near', 'near+barrel', 'both', 'far', 'other']


def jpsi_ancestor(index, pdg_ids, mothers):
    """Follow same-muon radiation copies; reject broken or cyclic ancestry."""
    if not 0 <= index < len(pdg_ids) or abs(int(pdg_ids[index])) != 13:
        return -1
    pdg = int(pdg_ids[index])
    visited = {index}
    mother = int(mothers[index])
    while 0 <= mother < len(pdg_ids) and int(pdg_ids[mother]) == pdg:
        if mother in visited:
            return -1
        visited.add(mother)
        mother = int(mothers[mother])
    return mother if 0 <= mother < len(pdg_ids) and int(pdg_ids[mother]) == 443 else -1


def stats(values):
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return {'n': 0}
    mean = sum(v) / len(v)
    q = lambda p: v[round(p * (len(v) - 1))]
    return dict(n=len(v), mean=mean, rms=math.sqrt(sum((x - mean)**2 for x in v) / len(v)),
                q16=q(.16), median=q(.5), q84=q(.84),
                abs90=sorted(map(abs, v))[round(.9 * (len(v) - 1))])


def analyze(path, label, ordinal, max_events):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f'Unreadable input: {path}')
    tree = f.Get('Events')
    if not tree or not tree.GetBranch('ShiftMuon_hitGenPartIdx'):
        raise RuntimeError('MC audit requires Events and ShiftMuon_hitGenPartIdx; no angular fallback')
    tree.SetBranchStatus('*', False)
    for branch in ('nShiftMuon', 'ShiftMuon_*', 'nGenPart', 'GenPart_*', 'nShiftDimuonVertex', 'ShiftDimuonVertex_*'):
        tree.SetBranchStatus(branch, True)
    branches = {b.GetName() for b in tree.GetListOfBranches()}
    hist, values, counts = {}, defaultdict(list), Counter({k: 0 for k in (
        'events', 'reco', 'hitMatched', 'legacyMissedHitMatch', 'reverseCloser',
        'hitMatchedDaughters', 'genDaughters', 'hitMatchedDimuons', 'recoDimuons')})
    def h1(name, bins, lo, hi):
        hist[name] = ROOT.TH1D(f'{ordinal}_{name}', name, bins, lo, hi)
    for name in ('eta', 'constrainedEta', 'axisEta'):
        h1(name, 480, -12., 12.)
    for name in ('phi', 'constrainedPhi', 'axisPhi'):
        h1(name, 200, -math.pi, math.pi)
    for name in ('etaPull', 'phiPull', 'targetPullX', 'targetPullY'):
        h1(name, 200, -20., 20.)
    for name in ('etaErr', 'phiErr'):
        h1(name, 100, 0., 1.)
    for name in ('massScale', 'constrainedMassScale', 'vertexRefitMassScale'):
        h1(name, 160, 0., 4.)
    for name in ('topology', 'matchedTopology', 'reverseTopology'):
        h1(name, 5, -.5, 4.5)
    for name in ('recoP',):
        h1(name, 100, 0., 500.)
    for name in ('recoEta',):
        h1(name, 120, -6., 6.)
    for name in ('recoPhi',):
        h1(name, 100, -math.pi, math.pi)
    maps = {'etaP': (30, -6., -3., 25, 0., 250.),
            'etaPt': (30, -6., -3., 25, 0., 5.),
            'etaPhi': (30, -6., -3., 24, -math.pi, math.pi),
            'chargePhi': (2, -2., 2., 24, -math.pi, math.pi)}
    for name, bins in maps.items():
        for suffix in ('pass', 'total'):
            hist[name + '_' + suffix] = ROOT.TH2D(f'{ordinal}_{name}_{suffix}', name, *bins)
    for name, bins in {'phiVsP': (25, 0., 250., 100, -math.pi, math.pi),
                       'constrainedPhiVsP': (25, 0., 250., 100, -math.pi, math.pi),
                       'constrainedEtaVsP': (25, 0., 250., 120, -3., 3.),
                       'reverseVsAlgorithm': (3, -.5, 2.5, 2, -.5, 1.5),
                       'reverseVsTiming': (3, -1.5, 1.5, 2, -.5, 1.5)}.items():
        hist[name] = ROOT.TH2D(f'{ordinal}_{name}', name, *bins)
    def fill(name, value):
        if math.isfinite(value):
            values[name].append(value)
            hist[name].Fill(value)
    n = min(tree.GetEntries(), max_events) if max_events >= 0 else tree.GetEntries()
    for entry in range(n):
        tree.GetEntry(entry)
        counts['events'] += 1
        matched = set()
        for i in range(int(tree.nShiftMuon)):
            get = lambda name: getattr(tree, 'ShiftMuon_' + name)[i]
            counts['reco'] += 1
            top = int(get('topology'))
            hist['topology'].Fill(top)
            fill('recoP', float(get('p')))
            fill('recoEta', float(get('eta')))
            fill('recoPhi', float(get('phi')))
            valid = bool(get('constrainedValid'))
            counts['constraintValid'] += valid
            idx, legacy = int(get('hitGenPartIdx')), int(get('genPartIdx'))
            counts['legacyMatched'] += legacy >= 0
            if valid and 'ShiftMuon_constrainedEtaErr' in branches:
                for a in ('Eta', 'Phi'):
                    fill(a.lower() + 'Err', float(get('constrained' + a + 'Err')))
                for a in ('X', 'Y'):
                    fill('targetPull' + a, float(get('targetPull' + a)))
            if idx < 0 or idx >= int(tree.nGenPart):
                counts['hitUnmatched'] += 1
                continue
            counts['hitMatched'] += 1
            matched.add(idx)
            hist['matchedTopology'].Fill(top)
            counts['legacyMissedHitMatch'] += legacy < 0
            counts['legacyDisagrees'] += legacy >= 0 and legacy != idx
            ge, gp = float(tree.GenPart_eta[idx]), float(tree.GenPart_phi[idx])
            p = math.hypot(float(tree.GenPart_pt[idx]), float(tree.GenPart_pz[idx]))
            de, dp = float(get('eta')) - ge, math.remainder(float(get('phi')) - gp, 2*math.pi)
            re, rp = -float(get('eta')) - ge, math.remainder(float(get('phi')) + math.pi - gp, 2*math.pi)
            reverse = math.hypot(re, rp) < math.hypot(de, dp)
            counts['reverseCloser'] += reverse
            counts[f'reverse_sourceSide{int(get("inferredSourceSide"))}'] += reverse
            counts[f'reverse_refitValid{int(get("directionalRefitValid"))}'] += reverse
            counts[f'reverse_positivePz{int(float(get("pz")) > 0)}'] += reverse
            if reverse:
                hist['reverseTopology'].Fill(top)
            hist['reverseVsAlgorithm'].Fill(float(get('recoAlgorithm')), float(reverse))
            hist['reverseVsTiming'].Fill(float(get('timingDirectionSign')), float(reverse))
            fill('eta', de); fill('phi', dp)
            fill('axisEta', re if reverse else de); fill('axisPhi', rp if reverse else dp)
            hist['phiVsP'].Fill(p, dp)
            if valid:
                ce = float(get('constrainedEta')) - ge
                cp = math.remainder(float(get('constrainedPhi')) - gp, 2*math.pi)
                fill('constrainedEta', ce); fill('constrainedPhi', cp)
                hist['constrainedPhiVsP'].Fill(p, cp)
                hist['constrainedEtaVsP'].Fill(p, ce)
                if 'ShiftMuon_constrainedEtaErr' in branches:
                    for a, residual in (('Eta', ce), ('Phi', cp)):
                        err = float(get('constrained' + a + 'Err'))
                        if err > 0.:
                            fill(a.lower() + 'Pull', residual / err)
        counts['uniqueHitMatched'] += len(matched)
        for j in range(int(tree.nGenPart)):
            mother = jpsi_ancestor(j, tree.GenPart_pdgId, tree.GenPart_genPartIdxMother)
            if abs(int(tree.GenPart_pdgId[j])) != 13 or int(tree.GenPart_status[j]) != 1 or mother < 0:
                continue
            counts['genDaughters'] += 1
            passed = j in matched
            counts['hitMatchedDaughters'] += passed
            eta, pt, pz, phi = [float(getattr(tree, 'GenPart_' + b)[j]) for b in ('eta', 'pt', 'pz', 'phi')]
            charge = -1 if int(tree.GenPart_pdgId[j]) > 0 else 1
            for name, x, y in (('etaP', eta, math.hypot(pt, pz)), ('etaPt', eta, pt),
                               ('etaPhi', eta, phi), ('chargePhi', charge, phi)):
                hist[name + '_total'].Fill(x, y)
                if passed: hist[name + '_pass'].Fill(x, y)
        for i in range(int(tree.nShiftDimuonVertex)):
            counts['recoDimuons'] += 1
            get = lambda name: getattr(tree, 'ShiftDimuonVertex_' + name)[i]
            a, b = int(get('muonIdx1')), int(get('muonIdx2'))
            if min(a,b) < 0 or max(a,b) >= int(tree.nShiftMuon) or a == b: continue
            a, b = int(tree.ShiftMuon_hitGenPartIdx[a]), int(tree.ShiftMuon_hitGenPartIdx[b])
            if min(a,b) < 0 or max(a,b) >= int(tree.nGenPart) or a == b: continue
            ma, mb = [jpsi_ancestor(j, tree.GenPart_pdgId, tree.GenPart_genPartIdxMother) for j in (a,b)]
            if ma < 0 or ma != mb or int(tree.GenPart_pdgId[a]) != -int(tree.GenPart_pdgId[b]): continue
            counts['hitMatchedDimuons'] += 1
            vectors = []
            for j in (a,b):
                v = ROOT.TLorentzVector()
                v.SetPtEtaPhiM(float(tree.GenPart_pt[j]),float(tree.GenPart_eta[j]),float(tree.GenPart_phi[j]),.1056583745)
                vectors.append(v)
            mass = (vectors[0]+vectors[1]).M()
            if mass <= 0: continue
            fill('massScale',float(get('mass'))/mass)
            if bool(get('constrainedValid')): fill('constrainedMassScale',float(get('constrainedMass'))/mass)
            if 'ShiftDimuonVertex_refittedMass' in branches and int(get('refitStatus')) == 1:
                fill('vertexRefitMassScale',float(get('refittedMass'))/mass)
    f.Close()
    result = dict(label=label, input=path, counts=dict(counts), statistics={k:stats(v) for k,v in values.items()})
    result['histogramFlows'] = {k: {'underflow':h.GetBinContent(0), 'overflow':h.GetBinContent(h.GetNbinsX()+1)}
                              for k,h in hist.items() if h.GetDimension() == 1}
    return result, hist


def plot(results, output):
    canvas = ROOT.TCanvas('audit', 'SHIFT reconstruction audit', 1300, 900)
    pdf = str(output)
    keep = []
    canvas.Print(pdf + '[')
    def title(text):
        t = ROOT.TLatex(); t.SetNDC(); t.SetTextSize(.025); t.DrawLatex(.02,.975,text); keep.append(t)
    def overlay(names, heading):
        canvas.Clear(); canvas.Divide(2,2)
        for panel,name in enumerate(names,1):
            canvas.cd(panel); ROOT.gPad.SetLogy()
            leg = ROOT.TLegend(.45,.72,.88,.88); leg.SetBorderSize(0)
            for i,(result,hist) in enumerate(results):
                h = hist[name].Clone(f'plot_{name}_{i}'); keep.append(h)
                integral=h.Integral(0,h.GetNbinsX()+1)
                if integral: h.Scale(1./integral)
                h.SetLineColor(COLORS[i]); h.SetLineWidth(2)
                h.SetTitle(f'{name};{name};fraction of entries / bin')
                h.SetMinimum(1.e-5); h.SetMaximum(1.)
                h.Draw('hist' if i==0 else 'hist same')
                leg.AddEntry(h,result['label'],'l')
            leg.Draw(); keep.append(leg)
        canvas.cd(); title(heading); canvas.Print(pdf)
    overlay(['recoP','recoEta','recoPhi','topology'],'All reconstructed rows; no truth requirement')
    overlay(['eta','phi','constrainedEta','constrainedPhi'],'Hit-associated residuals; no direction or residual cuts; failures retained in coverage')
    overlay(['axisEta','axisPhi','matchedTopology','reverseTopology'],'Axis-only diagnostic folds orientation; canonical residuals above remain unfolded')
    overlay(['massScale','constrainedMassScale','etaErr','phiErr'],'Independent pair identity; target uncertainties where stored')
    overlay(['etaPull','phiPull','targetPullX','targetPullY'],'Target-state pulls; target X/Y use innovation uncertainties')
    overlay(['massScale','constrainedMassScale','vertexRefitMassScale','topology'],'Same independently identified pair; separate unconstrained, target and vertex hypotheses')
    for names, heading in (
        (('phiVsP','constrainedPhiVsP'), 'Unfolded #Delta#phi versus true momentum; no residual selection'),
        (('constrainedEtaVsP','reverseVsAlgorithm'), 'Target #Delta#eta versus momentum; orientation by reconstruction algorithm'),
        (('reverseVsTiming',), 'Orientation diagnostic versus measured timing direction')):
        canvas.Clear(); canvas.Divide(2,len(names))
        for row,name in enumerate(names):
            for i,(result,hist) in enumerate(results):
                canvas.cd(2*row+i+1); ROOT.gPad.SetRightMargin(.16); ROOT.gPad.SetTopMargin(.15); ROOT.gPad.SetBottomMargin(.15)
                h=hist[name]
                axes={'phiVsP':('p_{gen} [GeV]','#Delta#phi [rad]'),
                      'constrainedPhiVsP':('p_{gen} [GeV]','target #Delta#phi [rad]'),
                      'constrainedEtaVsP':('p_{gen} [GeV]','target #Delta#eta'),
                      'reverseVsAlgorithm':('reconstruction algorithm','reversed direction closer'),
                      'reverseVsTiming':('measured timing sign','reversed direction closer')}
                x,y=axes[name]; h.SetTitle(result['label'] + ';' + x + ';' + y)
                if name.startswith('reverseVs'):
                    h.GetYaxis().SetBinLabel(1,'no'); h.GetYaxis().SetBinLabel(2,'yes')
                if name=='reverseVsAlgorithm':
                    for b,label in enumerate(('DSA','traversing','cosmic'),1): h.GetXaxis().SetBinLabel(b,label)
                h.Draw('colz')
        canvas.cd(); title(heading); canvas.Print(pdf)
    for mapname in ('etaP','etaPt','etaPhi','chargePhi'):
        canvas.Clear(); canvas.Divide(2,1)
        for i,(result,hist) in enumerate(results):
            canvas.cd(i+1); ROOT.gPad.SetRightMargin(.16)
            h=hist[mapname+'_pass'].Clone(f'eff_{i}_{mapname}'); keep.append(h)
            h.Divide(hist[mapname+'_pass'],hist[mapname+'_total'],1.,1.,'B')
            h.SetMinimum(0.); h.SetMaximum(1.)
            x,y={'etaP':('#eta_{gen}','p_{gen} [GeV]'),'etaPt':('#eta_{gen}','p_{T,gen} [GeV]'),
                 'etaPhi':('#eta_{gen}','#phi_{gen} [rad]'),'chargePhi':('q_{gen}','#phi_{gen} [rad]')}[mapname]
            h.SetTitle(f"{result['label']};{x};{y};hit-associated efficiency")
            h.Draw('colz')
            denominator=hist[mapname+'_total']
            for bx in range(1,h.GetNbinsX()+1):
                for by in range(1,h.GetNbinsY()+1):
                    if denominator.GetBinContent(bx,by): continue
                    box=ROOT.TBox(h.GetXaxis().GetBinLowEdge(bx),h.GetYaxis().GetBinLowEdge(by),
                                  h.GetXaxis().GetBinUpEdge(bx),h.GetYaxis().GetBinUpEdge(by))
                    box.SetFillColor(ROOT.kGray); box.SetLineColor(ROOT.kGray); box.Draw(); keep.append(box)
        canvas.cd(); title('J/#psi daughters: every generator daughter is in the denominator; gray bins have no denominator')
        canvas.Print(pdf)
    canvas.Clear(); canvas.cd()
    text = ROOT.TLatex(); text.SetNDC(); text.SetTextSize(.028)
    lines=['Coverage and tails (raw counts; no angular selection)']
    for result,hist in results:
        c=result['counts']; lines += [result['label'] + ': ' + f"{c['events']} events, {c['reco']} muons, {c['hitMatched']} hit-associated",
          f"Legacy match misses {c['legacyMissedHitMatch']} hit-associated rows; reverse direction closer: {c['reverseCloser']}",
          f"J/#psi daughters: {c['hitMatchedDaughters']}/{c['genDaughters']}; dimuons: {c['hitMatchedDimuons']}/{c['recoDimuons']}"]
        for name in ('phi','constrainedPhi','etaPull','phiPull','massScale','constrainedMassScale'):
            s=result['statistics'].get(name,{})
            if s.get('n',0): lines.append(f"{name}: n={s['n']}, median={s['median']:.3g}, 68%=[{s['q16']:.3g},{s['q84']:.3g}], RMS={s['rms']:.3g}")
    for i,line in enumerate(lines): text.DrawLatex(.04,.95-i*.044,line)
    canvas.Print(pdf); canvas.Print(pdf+']')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs=2)
    parser.add_argument('--labels',nargs=2,default=['control','material + field'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--summary')
    parser.add_argument('--max-events',type=int,default=-1)
    args=parser.parse_args()
    results=[analyze(path,label,i,args.max_events) for i,(path,label) in enumerate(zip(args.inputs,args.labels))]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    plot(results,Path(args.output))
    report=[r[0] for r in results]
    if args.summary: Path(args.summary).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','counts','statistics')} for r in report],indent=2))
if __name__=='__main__': main()
