#!/usr/bin/env python3
# Code style: AI-code.
"""Reco-only distributions and independent hit-truth performance; no angular cuts.

ROOT input is read directly, so this also audits existing productions without
another reconstruction. All reco rows contribute to reco/coverage diagnostics.
Truth resolution requires a stored hit association, never a nearest direction.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.TH1.AddDirectory(False)
COLORS = [ROOT.kAzure + 2, ROOT.kOrange + 7]
TOPO = ['near', 'near+barrel', 'both', 'far', 'other']

PAIRED_POLICIES = ('canonical', 'stable', 'joint', 'joint_sig2', 'joint_sig3')
MUON_MASS = .105658


def stable_pair_mass(first, second, mass=MUON_MASS):
    """Positive radial/angular terms avoid subtracting two large boosted energies."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    p1 = np.linalg.norm(first, axis=-1)
    p2 = np.linalg.norm(second, axis=-1)
    e1, e2 = np.hypot(p1, mass), np.hypot(p2, mass)
    u1 = np.divide(first, p1[..., None], out=np.zeros_like(first), where=p1[..., None] > 0.)
    u2 = np.divide(second, p2[..., None], out=np.zeros_like(second), where=p2[..., None] > 0.)
    radial = mass * mass * (p1 * p1 + p2 * p2 + mass * mass) / (e1 * e2 + p1 * p2)
    angular = p1 * p2 * np.sum((u1-u2)**2, axis=-1)
    return np.sqrt(2. * mass * mass + 2. * radial + angular)


def momentum_vector(pt, phi, pz):
    return np.column_stack((pt*np.cos(phi), pt*np.sin(phi), pz))


def paired_policy_states(measured, policies=PAIRED_POLICIES):
    """Inference uses only measured branches; every failed alternative falls back.

    Sample labels, truth, event identity and split assignments never enter this
    function. Numerical validity checks only reject an unusable alternative state.
    """
    pair_fields = ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')
    canonical = {name: np.asarray(measured['pair_'+name], dtype=float) for name in pair_fields}
    for leg in ('mu1', 'mu2'):
        for field in ('pt', 'pz', 'phi'):
            canonical[leg+'_'+field] = np.asarray(measured[leg+'_'+field], dtype=float)
    first = np.column_stack([measured['mu1_'+x] for x in ('px', 'py', 'pz')])
    second = np.column_stack([measured['mu2_'+x] for x in ('px', 'py', 'pz')])
    stable = stable_pair_mass(first, second)
    joint = {name: np.asarray(measured['pair_refitted'+name[0].upper()+name[1:]], dtype=float)
             for name in pair_fields}
    for leg in ('mu1', 'mu2'):
        for field in ('pt', 'pz', 'phi'):
            joint[leg+'_'+field] = np.asarray(measured[leg+'_vertexRefitted'+field[0].upper()+field[1:]], dtype=float)
    native_valid = np.asarray(measured['pair_refitStatus']) == 1
    numerical_valid = np.ones(len(stable), dtype=bool)
    association_valid = np.ones(len(stable), dtype=bool)
    for value in joint.values():
        numerical_valid &= np.isfinite(value)
    numerical_valid &= (joint['mass'] > 0.) & (joint['pt'] >= 0.)
    for leg in ('mu1', 'mu2'):
        numerical_valid &= joint[leg+'_pt'] >= 0.
        required = (leg+'_vertexRefitStatus', leg+'_vertexRefitIdx', 'pair_index')
        if not all(name in measured for name in required):
            association_valid[:] = False
            continue
        association_valid &= np.asarray(measured[leg+'_vertexRefitStatus']) == 1
        association_valid &= np.asarray(measured[leg+'_vertexRefitIdx']) == np.asarray(measured['pair_index'])
    significance = np.asarray(measured['pair_refittedMinQoverPSignificance'], dtype=float)
    result, choices = {}, {}
    for policy in policies:
        if policy not in PAIRED_POLICIES:
            raise ValueError('Unknown reconstruction policy: '+policy)
        use_joint = native_valid & numerical_valid & association_valid
        if policy.startswith('joint_sig'):
            use_joint &= np.isfinite(significance) & (significance >= int(policy[-1]))
        if not policy.startswith('joint'):
            use_joint = np.zeros(len(stable), dtype=bool)
        state = {name: np.where(use_joint, joint[name], value) for name, value in canonical.items()}
        if policy == 'stable':
            # Persisted px/py/pz are the strict arithmetic comparison; deriving
            # px/py through rounded pT/phi would also change the input momenta.
            state['mass'] = np.where(np.isfinite(stable), stable, canonical['mass'])
        result[policy] = state
        choices[policy] = dict(use_joint=use_joint, native_valid=native_valid,
                               numerical_valid=numerical_valid,
                               association_valid=association_valid,
                               failed_alternative=native_valid & ~(numerical_valid & association_valid))
    return result, choices


def reconstruction_residuals(state, truth):
    """Truth appears only here, after all measured-only states have been fixed."""
    result = {}
    def relative(name, reco, reference):
        result[name] = np.divide(reco-reference, reference, out=np.full_like(reference, np.nan), where=reference != 0.)
    relative('mass', state['mass'], truth['truth_pair_mass'])
    relative('pair_pt', state['pt'], truth['truth_pair_pt'])
    relative('pair_pz', state['pz'], truth['truth_pair_pz'])
    relative('pair_p', np.hypot(state['pt'], state['pz']), truth['truth_pair_p'])
    shared = np.asarray(truth['truth_pair_shared_vertex']) == 1
    for axis in ('x', 'y', 'z'):
        result['vertex_'+axis] = np.where(shared, state['v'+axis]-truth['truth_pair_v'+axis], np.nan)
    reco_vectors, truth_vectors = [], []
    muon_residuals = defaultdict(list)
    for leg in ('mu1', 'mu2'):
        reco = momentum_vector(state[leg+'_pt'], state[leg+'_phi'], state[leg+'_pz'])
        gen = momentum_vector(truth['truth_'+leg+'_pt'], truth['truth_'+leg+'_phi'], truth['truth_'+leg+'_pz'])
        reco_vectors.append(reco)
        truth_vectors.append(gen)
        for quantity, value, reference in (
                ('muon_pt', state[leg+'_pt'], truth['truth_'+leg+'_pt']),
                ('muon_pz', state[leg+'_pz'], truth['truth_'+leg+'_pz']),
                ('muon_p', np.linalg.norm(reco, axis=1), np.linalg.norm(gen, axis=1))):
            muon_residuals[quantity].append(np.divide(value-reference, reference, out=np.full_like(reference, np.nan), where=reference != 0.))
    for name, values in muon_residuals.items():
        result[name] = np.concatenate(values)
    first, second = reco_vectors
    angle = np.arctan2(np.linalg.norm(np.cross(first, second), axis=1), np.sum(first*second, axis=1))
    result['opening_angle'] = angle-truth['truth_pair_openingAngle']
    reco_norms = [np.linalg.norm(v, axis=1) for v in reco_vectors]
    true_norms = [np.linalg.norm(v, axis=1) for v in truth_vectors]
    # Counterfactuals separate magnitude and direction errors for diagnosis.
    # They are never used to construct, choose, or filter a reconstruction state.
    true_angles = [np.divide(v, n[:, None], out=np.zeros_like(v), where=n[:, None] > 0.) * r[:, None]
                   for v, n, r in zip(truth_vectors, true_norms, reco_norms)]
    true_magnitudes = [np.divide(v, n[:, None], out=np.zeros_like(v), where=n[:, None] > 0.) * r[:, None]
                       for v, n, r in zip(reco_vectors, reco_norms, true_norms)]
    relative('mc_only_true_angles_mass', stable_pair_mass(*true_angles), truth['truth_pair_mass'])
    relative('mc_only_true_magnitudes_mass', stable_pair_mass(*true_magnitudes), truth['truth_pair_mass'])
    return result


def residual_statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    low, median, high = np.quantile(values, [.158655, .5, .841345])
    return dict(sigma68=float(.5*(high-low)), median=float(median),
                tail100=float(np.mean(np.abs(values) > 1.)),
                tail50=float(np.mean(np.abs(values) > .5)),
                abs90=float(np.quantile(np.abs(values), .9)),
                rms_zero=float(np.sqrt(np.mean(values*values))))


def paired_parent_statistics(values, parents, bootstrap=200, seed=190019):
    """Resample physical GEN parents once per draw and reuse draws for each policy."""
    parents = np.asarray(parents)
    arrays = {name: np.asarray(value, dtype=float) for name, value in values.items()}
    common = np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    arrays = {name: value[common] for name, value in arrays.items()}
    parents = parents[common]
    unique, inverse = np.unique(parents, return_inverse=True)
    result = dict(entries=int(common.sum()), omitted_nonfinite=int((~common).sum()), physical_parents=len(unique),
                  quantile_resolved=bool(common.sum() >= 20 and len(unique) >= 2), policies={})
    for policy, value in arrays.items():
        result['policies'][policy] = dict(estimate=residual_statistics(value))
    # A single parent cannot identify a population uncertainty. Its bootstrap
    # would repeat the identical event and give a misleading zero-width band.
    if len(unique) < 2:
        return result
    indices = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    generator = np.random.default_rng(seed)
    draws = {name: defaultdict(list) for name in arrays}
    for _ in range(bootstrap):
        chosen = generator.integers(0, len(unique), size=len(unique))
        sample = np.concatenate([indices[i] for i in chosen])
        for policy, value in arrays.items():
            metrics = residual_statistics(value[sample])
            for statistic, estimate in metrics.items():
                draws[policy][statistic].append(estimate)
    for policy in arrays:
        intervals, differences = {}, {}
        for statistic, samples in draws[policy].items():
            low, high = np.quantile(samples, [.158655, .841345])
            estimate = result['policies'][policy]['estimate'][statistic]
            intervals[statistic] = dict(low=float(low), high=float(high), error_down=float(max(0., estimate-low)), error_up=float(max(0., high-estimate)))
            if 'canonical' in draws:
                delta = np.asarray(samples)-np.asarray(draws['canonical'][statistic])
                lo, hi = np.quantile(delta, [.158655, .841345])
                differences[statistic] = dict(value=float(estimate-result['policies']['canonical']['estimate'][statistic]), low=float(lo), high=float(hi))
        result['policies'][policy].update(interval68=intervals, paired_delta_vs_canonical=differences)
    return result


def paired_development_masks(data, sources):
    """Frozen point/event split, independent of outcomes or the tested policy."""
    source_index = np.asarray(data['source_index'], dtype=int)
    buckets = np.asarray(data['split_bucket'], dtype=int)
    development = np.zeros(len(source_index), dtype=bool)
    point_holdout = np.zeros(len(source_index), dtype=bool)
    process_holdout = np.zeros(len(source_index), dtype=bool)
    transfer_grid = np.zeros(len(source_index), dtype=bool)
    for name in ('initial_canary_peeked', 'prior_development_peeked'):
        if name not in data:
            raise ValueError('Paired extraction requires explicit physical-parent exposure marker: '+name)
    canary = np.asarray(data['initial_canary_peeked'], dtype=bool)
    previous_development = np.asarray(data['prior_development_peeked'], dtype=bool)
    for source in sources:
        mask = source_index == source['index']
        process, sample = source['process'], source['sample']
        if process in ('DY', 'QCD'):
            process_holdout |= mask
        elif process == 'JPsi':
            development |= mask
        elif process == 'DarkPhoton':
            mass_label, lifetime = sample.removeprefix('darkphoton_m').split('_', 1)
            mass = float(mass_label.replace('p', '.'))
            held = mass in (1.5, 2., 10., 15., 70.) or lifetime in ('lab_5m', 'lab_1000m')
            point_holdout |= mask & held
            development |= mask & (mass in (5., 40.)) & (not held)
            transfer_grid |= mask & (not held) & (mass not in (5., 40.))
    return dict(development=(development & (buckets <= 5)) | canary,
                internal_check=development & np.isin(buckets, (6, 7)) & ~canary,
                event_holdout=development & np.isin(buckets, (8, 9)) & ~canary & ~previous_development,
                point_holdout=point_holdout & ~canary,
                process_holdout=process_holdout & ~canary,
                transfer_grid=transfer_grid & ~canary & ~previous_development)


def native_alternative_state(canonical, proposed, status):
    """Use one complete measured state or its original state, retaining every pair."""
    required = ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')
    valid = status == 1 and all(math.isfinite(proposed.get(k, math.nan)) for k in required)
    valid = valid and proposed['mass'] > 0. and proposed['pt'] >= 0.
    legs = tuple(f'mu{i}_{field}' for i in (1, 2) for field in ('pt', 'pz', 'phi'))
    if any(k in proposed for k in legs):
        valid = valid and all(math.isfinite(proposed.get(k, math.nan)) for k in legs)
        valid = valid and proposed.get('mu1_pt', -1.) >= 0. and proposed.get('mu2_pt', -1.) >= 0.
    return (dict(proposed) if valid else dict(canonical)), bool(valid)


def native_file_checksum(path):
    """Hash the original file bytes, including remote ROOT files on EOS."""
    digest = hashlib.sha256()
    if not str(path).startswith('root://'):
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        return digest.hexdigest()
    import ctypes
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise RuntimeError('Cannot checksum remote native file: '+str(path))
    try:
        size = int(source.GetSize())
        if size <= 0:
            raise RuntimeError('Remote native file has no bytes: '+str(path))
        buffer = ctypes.create_string_buffer(1024*1024)
        for offset in range(0, size, len(buffer)):
            count = min(len(buffer), size-offset)
            if source.ReadBuffer(buffer, offset, count):
                raise RuntimeError('Remote native checksum read failed: '+str(path))
            digest.update(buffer.raw[:count])
    finally:
        source.Close()
    return digest.hexdigest()


def read_native_reconstruction(path):
    """Read every event/pair. Stored hit associations are used only for closure."""
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError('Unreadable native replay: '+str(path))
    tree = source.Get('Events')
    if not tree:
        raise RuntimeError('Missing Events in '+str(path))
    branches = {b.GetName() for b in tree.GetListOfBranches()}
    required = {'run', 'luminosityBlock', 'event', 'nShiftMuon', 'nShiftDimuonVertex', 'nGenPart'}
    required.update('ShiftMuon_'+k for k in ('hitGenPartIdx', 'pt', 'pz', 'phi', 'topology'))
    required.update('ShiftDimuonVertex_'+k for k in ('mass', 'pt', 'pz', 'vx', 'vy', 'vz',
                    'muonIdx1', 'muonIdx2', 'topologyMin', 'topologyMax'))
    gen_fields = ('pdgId', 'genPartIdxMother', 'pt', 'pz', 'phi', 'mass', 'vx', 'vy', 'vz')
    required.update('GenPart_'+k for k in gen_fields)
    if required-branches:
        raise RuntimeError('Native replay missing branches: '+str(sorted(required-branches)))
    tree.SetBranchStatus('*', False)
    for name in required | {n for n in branches if n.startswith(('ShiftDimuonVertex_', 'ShiftMuon_', 'GenPart_')) or n in
            ('genWeight', 'shiftSamplingGenWeight', 'shiftOriginalGenWeight', 'shiftSamplingWeight',
             'shiftSamplingProbability')}:
        tree.SetBranchStatus(name, True)
    for name in branches:
        if name.startswith('ShiftMuon_vertexRefit'):
            tree.SetBranchStatus(name, True)
    events, pairs = {}, {}
    pair_fields = sorted(n.removeprefix('ShiftDimuonVertex_') for n in branches if n.startswith('ShiftDimuonVertex_'))
    for entry in range(int(tree.GetEntries())):
        tree.GetEntry(entry)
        identity = tuple(int(getattr(tree, n)) for n in ('run', 'luminosityBlock', 'event'))
        if identity in events:
            raise RuntimeError('Duplicate native event '+str(identity))
        gen = {n: list(getattr(tree, 'GenPart_'+n)) for n in gen_fields}
        full_gen = {n: list(getattr(tree, n)) for n in sorted(branches) if n.startswith('GenPart_')}
        gen_digest = hashlib.sha256(repr(full_gen).encode()).hexdigest()
        protected_muons = {n: list(getattr(tree, n)) for n in sorted(branches)
                           if n.startswith('ShiftMuon_') and not n.startswith('ShiftMuon_vertexRefit')}
        muon_digest = hashlib.sha256(repr(protected_muons).encode()).hexdigest()
        weights = {n: float(getattr(tree, n)) for n in branches if n in
                   ('genWeight', 'shiftSamplingGenWeight', 'shiftOriginalGenWeight',
                    'shiftSamplingWeight', 'shiftSamplingProbability')}
        events[identity] = dict(muons=int(tree.nShiftMuon), pairs=int(tree.nShiftDimuonVertex),
                                gen_sha256=gen_digest, protected_muons_sha256=muon_digest, weights=weights)
        for index in range(int(tree.nShiftDimuonVertex)):
            measured = {n: float(getattr(tree, 'ShiftDimuonVertex_'+n)[index]) for n in pair_fields}
            legs = tuple(int(measured[n]) for n in ('muonIdx1', 'muonIdx2'))
            if legs[0] == legs[1] or not all(0 <= i < int(tree.nShiftMuon) for i in legs):
                raise RuntimeError('Invalid native leg reference')
            canonical = {n: measured[n] for n in ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')}
            for leg, i in enumerate(legs, 1):
                for field in ('pt', 'pz', 'phi'):
                    canonical[f'mu{leg}_{field}'] = float(getattr(tree, 'ShiftMuon_'+field)[i])
                for branch in branches:
                    if branch.startswith('ShiftMuon_vertexRefit'):
                        measured[f'nativeMuon{leg}_'+branch.removeprefix('ShiftMuon_')] = float(getattr(tree, branch)[i])
            matched = [int(tree.ShiftMuon_hitGenPartIdx[i]) for i in legs]
            valid = len(set(matched)) == 2 and all(0 <= g < int(tree.nGenPart) and
                    abs(gen['pdgId'][g]) == 13 for g in matched)
            truth = {}
            if valid:
                vectors = [momentum_vector(np.array([gen['pt'][g]]), np.array([gen['phi'][g]]),
                                           np.array([gen['pz'][g]]))[0] for g in matched]
                total = vectors[0]+vectors[1]
                truth.update(mass=float(stable_pair_mass(*vectors)), pt=float(np.hypot(*total[:2])),
                             pz=float(total[2]), p=float(np.linalg.norm(total)),
                             opening_angle=float(np.arctan2(np.linalg.norm(np.cross(*vectors)), np.dot(*vectors))))
                for leg, g in enumerate(matched, 1):
                    for field in ('pt', 'pz', 'phi'):
                        truth[f'mu{leg}_{field}'] = float(gen[field][g])
                positions = [np.array([gen[n][g] for n in ('vx', 'vy', 'vz')]) for g in matched]
                shared = bool(np.max(np.abs(positions[0]-positions[1])) < 1.e-4)
                truth['shared_vertex'] = shared
                for axis, value in zip(('vx', 'vy', 'vz'), positions[0]):
                    truth[axis] = float(value) if shared else math.nan
                ancestors = []
                for g in matched:
                    ancestor, visited = g, set()
                    while 0 <= ancestor < int(tree.nGenPart) and abs(gen['pdgId'][ancestor]) == 13:
                        if ancestor in visited:
                            ancestor = -1
                            break
                        visited.add(ancestor)
                        ancestor = int(gen['genPartIdxMother'][ancestor])
                    ancestors.append(ancestor)
                same_parent = ancestors[0] == ancestors[1] and 0 <= ancestors[0] < int(tree.nGenPart)
                truth['parent_pdgId'] = int(gen['pdgId'][ancestors[0]]) if same_parent else 0
                truth['flight'] = math.nan
                if same_parent and shared:
                    ancestor = ancestors[0]
                    origin = np.array([gen[n][ancestor] for n in ('vx', 'vy', 'vz')])
                    truth['flight'] = float(np.linalg.norm(positions[0]-origin))
            pairs[identity+(index,)] = dict(canonical=canonical, measured=measured, truth=truth,
                signature=tuple(measured[k] for k in ('muonIdx1', 'muonIdx2', 'topologyMin', 'topologyMax'))+tuple(matched),
                category=tuple(int(measured[k]) for k in ('topologyMin', 'topologyMax')))
    stamp = dict(path=str(path), bytes=int(source.GetSize()), root_uuid=source.GetUUID().AsString(),
                 events=len(events), pairs=len(pairs), pair_branches=pair_fields)
    source.Close()
    return events, pairs, stamp


def compare_native_reconstruction(manifest_path, bootstrap=200, seed=190019):
    """Paired native replay comparison, including failed fits and zero-pair events.

    The manifest fixes source membership and method branch mappings. It never
    supplies an event-level choice, true mass, true position or a fit parameter.
    """
    payload = Path(manifest_path).read_bytes()
    manifest = json.loads(payload)
    definitions = manifest['hypotheses']
    policies = ['canonical']+list(definitions)
    records, sources, coverage = [], [], []
    seen_parents = set()
    for source in manifest['sources']:
        if source.get('baseline_sha256'):
            if native_file_checksum(source['baseline']) != source['baseline_sha256']:
                raise RuntimeError('Native baseline checksum differs from manifest: '+source['baseline'])
        events, pairs, stamp = read_native_reconstruction(source['baseline'])
        if 'expected_events' in source and len(events) != source['expected_events']:
            raise RuntimeError('Native baseline event count differs from manifest: '+source['baseline'])
        sources.append(dict(source=source, baseline=stamp))
        modes = {}
        loaded = {source['baseline']: (events, pairs, stamp)}
        for policy, definition in definitions.items():
            path = source.get('alternatives', {}).get(policy, source['baseline'])
            if path not in loaded:
                loaded[path] = read_native_reconstruction(path)
            alt_events, alt_pairs, alt_stamp = loaded[path]
            expected = {definition['status']} | {definition['prefix']+n[0].upper()+n[1:]
                       for n in ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')}
            if expected-set(alt_stamp['pair_branches']):
                raise RuntimeError('Native alternative branches missing for '+policy+': '+
                                   str(sorted(expected-set(alt_stamp['pair_branches']))))
            if events != alt_events or pairs.keys() != alt_pairs.keys():
                raise RuntimeError('Native replay changes event identities, counts, weights, GEN or pair membership: '+policy)
            for key, row in pairs.items():
                alternative = alt_pairs[key]
                if row['signature'] != alternative['signature'] or row['canonical'] != alternative['canonical']:
                    raise RuntimeError('Native replay changes protected canonical pair: '+str(key))
            modes[policy] = alt_pairs
            sources[-1][policy] = alt_stamp
        for identity in events:
            parent = source['source_context']+':'+':'.join(map(str, identity))
            if parent in seen_parents:
                raise RuntimeError('Native manifest repeats a physical parent: '+parent)
            seen_parents.add(parent)
        counts = Counter(events=len(events), reconstructed_pairs=len(pairs))
        for key, row in pairs.items():
            states = {'canonical': row['canonical']}
            use = {'canonical': False}
            for policy, definition in definitions.items():
                measured = modes[policy][key]['measured']
                proposed = {name: measured.get(definition['prefix']+name[0].upper()+name[1:], math.nan)
                            for name in ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')}
                native_status = int(measured.get(definition['status'], 0))
                status = native_status
                leg_fields = {}
                for leg in (1, 2):
                    for field in ('pt', 'pz', 'phi'):
                        branch = definition['prefix']+f'Muon{leg}'+field[0].upper()+field[1:]
                        if definition.get('muon_prefix'):
                            branch = f'nativeMuon{leg}_'+definition['muon_prefix']+field[0].upper()+field[1:]
                        if branch in measured:
                            leg_fields[f'mu{leg}_{field}'] = measured[branch]
                if leg_fields:
                    proposed.update(leg_fields)
                if native_status == 1 and definition.get('muon_prefix') and any(
                        measured.get(f'nativeMuon{leg}_vertexRefitStatus') != 1 or
                        measured.get(f'nativeMuon{leg}_vertexRefitIdx') != key[3] for leg in (1, 2)):
                    status = -98
                    counts[policy+'_association_failure'] += 1
                states[policy], use[policy] = native_alternative_state(row['canonical'], proposed, status)
                counts[policy+'_status_'+str(native_status)] += 1
                counts[policy+'_used'] += use[policy]
            records.append(dict(sample=source['sample'], process=source['process'],
                cohort=source.get('cohort', 'native_development'), category=row['category'],
                parent=source['source_context']+':'+':'.join(map(str, key[:3])),
                states=states, use=use, truth=row['truth']))
        coverage.append(dict(sample=source['sample'], source_context=source['source_context'], counts=dict(counts)))
    reports = {}
    categories = sorted({r['category'] for r in records})
    labels = {0: 'Near', 1: 'Near-Barrel', 2: 'Both', 3: 'Far', 4: 'Other'}
    for category in categories:
        selected = [r for r in records if r['category'] == category]
        name = '-'.join(labels.get(k, str(k)) for k in category)
        report = dict(schema='shift-native-paired-reconstruction-v1', category=name, mode='native',
            policies=policies, policy_labels=dict(canonical='Original', **{k: v.get('label', k) for k, v in definitions.items()}),
            extraction_complete=True, sources=sources, coverage=coverage,
            coverage_scope='Source event/status coverage includes every measured category; group choices and performance are category-specific.',
            manifest_sha256=hashlib.sha256(payload).hexdigest(), bootstrap_draws=bootstrap,
            audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            validation_note=manifest.get('validation_note', 'Native development comparison; no blind validation claim.'),
            weighting='Unweighted reconstruction diagnostic. Full GEN exposure and native weights remain in source provenance.', groups=[])
        for cohort in sorted({r['cohort'] for r in selected}):
            scope = [r for r in selected if r['cohort'] == cohort]
            groups = [('all', scope)]
            groups += [('process/'+p, [r for r in scope if r['process'] == p]) for p in sorted({r['process'] for r in scope})]
            groups += [('point/'+p, [r for r in scope if r['sample'] == p]) for p in sorted({r['sample'] for r in scope})]
            groups += [('signal_point/'+p, [r for r in scope if r['sample'] == p and
                       abs(r['truth'].get('parent_pdgId', 0)) == (32 if p.startswith('darkphoton_') else 443)])
                       for p in sorted({r['sample'] for r in scope}) if p.startswith(('darkphoton_', 'jpsi_'))]
            scan_definitions = (
                ('mass', 'True dimuon mass [GeV]', (.2, .5, 1., 2., 5., 10., 20., 40., 70., 100.)),
                ('pt', 'True dimuon p_{T} [GeV]', (0., 1., 2., 5., 10., 20., 50., 100., 500.)),
                ('abs_pz', 'True dimuon |p_{z}| [GeV]', (0., 20., 50., 100., 200., 500., 1000., 5000., 10000.)),
                ('p', 'True dimuon |p| [GeV]', (0., 20., 50., 100., 200., 500., 1000., 5000., 10000.)),
                ('vz', 'True production z [cm]', (-20000., -15000., -10000., -5000., 0., 5000.,
                    10000., 12000., 13000., 14000., 14500., 14750., 15000., 20000.)),
                ('flight', 'True lab flight [cm]', (0., .1, 1., 10., 100., 1000., 3000., 10000., 30000.)),
                ('opening_angle', 'True opening angle [rad]', (0., .001, .003, .01, .03, .1, .3, 1., math.pi)),
            )
            report['scan_definitions'] = {key: label for key, label, _ in scan_definitions}
            for quantity, _, edges in scan_definitions:
                for low, high in zip(edges, edges[1:]):
                    population = []
                    for row in scope:
                        truth = row['truth']
                        value = abs(truth.get('pz', math.nan)) if quantity == 'abs_pz' else truth.get(quantity, math.nan)
                        if low <= value < high:
                            population.append(row)
                    groups.append((f'scan/{quantity}/{low:g}..{high:g}', population))
            for group_name, group in groups:
                matched = [r for r in group if r['truth'] and r['truth']['mass'] > 0.]
                if not matched:
                    continue
                item = dict(stage=cohort, group=group_name, reconstructed_pairs=len(group), matched_pairs=len(matched),
                            choices={p: dict(used=sum(r['use'][p] for r in group)) for p in policies}, metrics={})
                parents = [r['parent'] for r in matched]
                metrics = ['mass', 'pair_pt', 'pair_pz', 'pair_p', 'vertex_x', 'vertex_y', 'vertex_z']
                if all(all(f'mu{leg}_{field}' in r['states'][p] for leg in (1, 2) for field in ('pt', 'pz', 'phi'))
                       for r in matched for p in policies):
                    metrics += ['muon_pt', 'muon_pz', 'muon_p', 'opening_angle',
                                'mc_only_true_angles_mass', 'mc_only_true_magnitudes_mass']
                for metric in metrics:
                    residuals = {}
                    for policy in policies:
                        values = []
                        for row in matched:
                            state, truth = row['states'][policy], row['truth']
                            if metric.startswith('muon_'):
                                field = metric.removeprefix('muon_')
                                for leg in (1, 2):
                                    prefix = f'mu{leg}_'
                                    value = math.hypot(state[prefix+'pt'], state[prefix+'pz']) if field == 'p' else state[prefix+field]
                                    reference = math.hypot(truth[prefix+'pt'], truth[prefix+'pz']) if field == 'p' else truth[prefix+field]
                                    values.append((value-reference)/reference if reference != 0. else math.nan)
                            elif metric == 'opening_angle' or metric.startswith('mc_only_'):
                                vectors = [momentum_vector(np.array([state[f'mu{leg}_pt']]), np.array([state[f'mu{leg}_phi']]),
                                                           np.array([state[f'mu{leg}_pz']]))[0] for leg in (1, 2)]
                                if metric == 'opening_angle':
                                    values.append(float(np.arctan2(np.linalg.norm(np.cross(*vectors)), np.dot(*vectors)))-truth['opening_angle'])
                                else:
                                    truth_vectors = [momentum_vector(np.array([truth[f'mu{leg}_pt']]), np.array([truth[f'mu{leg}_phi']]),
                                                                     np.array([truth[f'mu{leg}_pz']]))[0] for leg in (1, 2)]
                                    replacements = []
                                    for reco, gen in zip(vectors, truth_vectors):
                                        direction, magnitude = (gen, reco) if metric == 'mc_only_true_angles_mass' else (reco, gen)
                                        norm = np.linalg.norm(direction)
                                        replacements.append(direction/norm*np.linalg.norm(magnitude) if norm > 0. else np.zeros(3))
                                    values.append((float(stable_pair_mass(*replacements))-truth['mass'])/truth['mass'])
                            elif metric.startswith('vertex_'):
                                axis = 'v'+metric[-1]
                                values.append(state[axis]-truth[axis] if truth['shared_vertex'] else math.nan)
                            else:
                                field = metric.removeprefix('pair_')
                                value = math.hypot(state['pt'], state['pz']) if field == 'p' else state[field]
                                values.append((value-truth[field])/truth[field] if truth[field] != 0. else math.nan)
                        residuals[policy] = values
                    item['metrics'][metric] = paired_parent_statistics(residuals,
                        np.repeat(parents, 2) if metric.startswith('muon_') else parents, bootstrap, seed)
                report['groups'].append(item)
            # Individual-muon scans select each leg by its own true momentum,
            # while retaining the shared physical parent in bootstrap draws.
            leg_complete = all(all(f'mu{leg}_{field}' in r['states'][p] for leg in (1, 2)
                               for field in ('pt', 'pz', 'phi')) for r in scope if r['truth'] for p in policies)
            if not leg_complete:
                continue
            muon_scans = (
                ('pt', 'True muon p_{T} [GeV]', (0., .5, 1., 2., 5., 10., 20., 50., 100., 500.)),
                ('abs_pz', 'True muon |p_{z}| [GeV]', (0., 20., 50., 100., 200., 500., 1000., 5000., 10000.)),
                ('p', 'True muon |p| [GeV]', (0., 20., 50., 100., 200., 500., 1000., 5000., 10000.)),
            )
            report['muon_scan_definitions'] = {key: label for key, label, _ in muon_scans}
            for quantity, _, edges in muon_scans:
                for low, high in zip(edges, edges[1:]):
                    population = []
                    for row in scope:
                        if not row['truth']:
                            continue
                        for leg in (1, 2):
                            pt, pz = (row['truth'][f'mu{leg}_'+q] for q in ('pt', 'pz'))
                            reference = pt if quantity == 'pt' else abs(pz) if quantity == 'abs_pz' else math.hypot(pt, pz)
                            if low <= reference < high:
                                population.append((row, leg))
                    if not population:
                        continue
                    item = dict(stage=cohort, group=f'muon_scan/{quantity}/{low:g}..{high:g}',
                                reconstructed_muons=len(population), metrics={})
                    parents = [row['parent'] for row, _ in population]
                    for metric in ('pt', 'pz', 'p'):
                        residuals = {}
                        for policy in policies:
                            values = []
                            for row, leg in population:
                                prefix = f'mu{leg}_'
                                state, truth = row['states'][policy], row['truth']
                                value = math.hypot(state[prefix+'pt'], state[prefix+'pz']) if metric == 'p' else state[prefix+metric]
                                reference = math.hypot(truth[prefix+'pt'], truth[prefix+'pz']) if metric == 'p' else truth[prefix+metric]
                                values.append((value-reference)/reference if reference != 0. else math.nan)
                            residuals[policy] = values
                        item['metrics']['muon_'+metric] = paired_parent_statistics(residuals, parents, bootstrap, seed)
                    report['groups'].append(item)
        reports[name] = report
    return reports


def validate_detector_replay(events, pairs, alternative_events, alternative_pairs):
    """Allow fitted states to change, preserving the complete measured population.

    GEN and stored hit-truth checks are file-integrity checks only. They never
    choose an alternative state or remove an individual event from the report.
    """
    protected = ('muons', 'pairs', 'gen_sha256', 'weights')
    if events.keys() != alternative_events.keys() or pairs.keys() != alternative_pairs.keys():
        raise RuntimeError('Detector replay changes event or pair membership')
    for identity, event in events.items():
        if any(event[k] != alternative_events[identity][k] for k in protected):
            raise RuntimeError('Detector replay changes counts, GEN or native weights')
    for key, pair in pairs.items():
        if pair['signature'] != alternative_pairs[key]['signature']:
            raise RuntimeError('Detector replay changes pair identities, topology or hit association')


def prepare_native_detector_overlay(baseline, alternative, output):
    """Append separate detector-fit hypotheses to an unchanged baseline copy.

    This diagnostic adapter supports a detector refit that legitimately changes
    the canonical state in its own file. The normal paired audit can then keep
    its exact canonical-state protection. Every original pair remains present;
    failed detector fits keep the entire baseline state during the paired audit.
    """
    from array import array
    import shutil
    import subprocess
    events, pairs, _ = read_native_reconstruction(baseline)
    alt_events, alt_pairs, _ = read_native_reconstruction(alternative)
    validate_detector_replay(events, pairs, alt_events, alt_pairs)
    output = Path(output)
    if output.exists():
        raise RuntimeError('Refusing to overwrite detector overlay: '+str(output))
    definitions = {
        'detailed_detector': dict(prefix='detectorDetailed', status='detectorDetailedStatus', label='Detailed detector'),
        'detailed_common_plane': dict(prefix='detectorCommonPlane', status='detectorCommonPlaneStatus', label='Detailed + plane'),
        'detailed_position': dict(prefix='detectorPositionOnly', status='detectorPositionOnlyStatus', label='Detailed + position')}
    raw_identity_fields = ('recoAlgorithm', 'sourceIndex', 'topology', 'nValidHits', 'trackPt', 'preRefitPt', 'preRefitPz')
    originals = ROOT.TFile.Open(str(baseline)); refitted = ROOT.TFile.Open(str(alternative))
    original_tree, refitted_tree = originals.Get('Events'), refitted.Get('Events')
    detector_valid = {}
    try:
        for field in raw_identity_fields+('directionalRefitValid',):
            if not original_tree.GetBranch('ShiftMuon_'+field) or not refitted_tree.GetBranch('ShiftMuon_'+field):
                raise RuntimeError('Missing detector identity/validity branch '+field)
        for entry in range(int(original_tree.GetEntries())):
            original_tree.GetEntry(entry); refitted_tree.GetEntry(entry)
            identity = tuple(int(getattr(original_tree, k)) for k in ('run', 'luminosityBlock', 'event'))
            other = tuple(int(getattr(refitted_tree, k)) for k in ('run', 'luminosityBlock', 'event'))
            if identity != other:
                raise RuntimeError('Detector overlay requires unchanged event ordering')
            for field in raw_identity_fields:
                if list(getattr(original_tree, 'ShiftMuon_'+field)) != list(getattr(refitted_tree, 'ShiftMuon_'+field)):
                    raise RuntimeError('Detector replay changes original measured identity/input '+field)
            detector_valid[identity] = list(refitted_tree.ShiftMuon_directionalRefitValid)
    finally:
        originals.Close(); refitted.Close()
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(baseline).startswith('root://'):
        subprocess.run(['xrdcp', '--nopbar', str(baseline), str(output)], check=True)
    else:
        shutil.copy2(baseline, output)
    fields = ('mass', 'pt', 'pz', 'vx', 'vy', 'vz')
    leg_fields = tuple(f'mu{i}_{field}' for i in (1, 2) for field in ('pt', 'pz', 'phi'))
    buffers, branches, counts = {}, {}, Counter()
    maximum = max(1, max(event['pairs'] for event in events.values()))
    target = ROOT.TFile.Open(str(output), 'UPDATE')
    tree = target.Get('Events')
    try:
        for definition in definitions.values():
            names = {definition['status']: 'i'}
            names.update({definition['prefix']+k[0].upper()+k[1:]: 'f' for k in fields})
            names.update({definition['prefix']+f'Muon{i}'+k[0].upper()+k[1:]: 'f'
                          for i in (1, 2) for k in ('pt', 'pz', 'phi')})
            for name, kind in names.items():
                branch = 'ShiftDimuonVertex_'+name
                if tree.GetBranch(branch):
                    raise RuntimeError('Detector overlay branch already exists '+branch)
                buffers[name] = array(kind, [0]*maximum)
                branches[name] = tree.Branch(branch, buffers[name], branch+'[nShiftDimuonVertex]/'+kind.upper())
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            identity = tuple(int(getattr(tree, k)) for k in ('run', 'luminosityBlock', 'event'))
            for index in range(int(tree.nShiftDimuonVertex)):
                key = identity+(index,); row = alt_pairs[key]
                legs = [int(row['measured'][k]) for k in ('muonIdx1', 'muonIdx2')]
                tracks_valid = all(detector_valid[identity][i] == 1 for i in legs)
                for policy, definition in definitions.items():
                    prefix = definition['prefix']
                    if policy == 'detailed_detector':
                        state, status = row['canonical'], 1
                    else:
                        native_prefix, native_status = (('materialTransported', 'materialTransportStatus')
                            if policy == 'detailed_common_plane' else ('positionOnly', 'positionOnlyStatus'))
                        state = {k: row['measured'][native_prefix+k[0].upper()+k[1:]] for k in fields}
                        state.update({f'mu{i}_{k}': row['measured'][native_prefix+f'Muon{i}'+k[0].upper()+k[1:]]
                                      for i in (1, 2) for k in ('pt', 'pz', 'phi')})
                        status = int(row['measured'][native_status])
                    status = status if tracks_valid else -10
                    buffers[definition['status']][index] = status
                    for field in fields:
                        buffers[prefix+field[0].upper()+field[1:]][index] = state[field]
                    for leg in leg_fields:
                        muon, field = leg.split('_')
                        buffers[prefix+'Muon'+muon[-1]+field[0].upper()+field[1:]][index] = state[leg]
                    counts[policy+'_status_'+str(status)] += 1
            for branch in branches.values():
                branch.Fill()
        tree.Write('', ROOT.TObject.kOverwrite)
    finally:
        target.Close()
    restored_events, restored_pairs, _ = read_native_reconstruction(output)
    if restored_events != events or any(restored_pairs[k]['canonical'] != row['canonical'] for k, row in pairs.items()):
        raise RuntimeError('Detector overlay changed protected baseline content')
    return dict(schema='shift-native-detector-overlay-v1', baseline=str(baseline),
        baseline_sha256=native_file_checksum(baseline), alternative=str(alternative),
        alternative_sha256=native_file_checksum(alternative), output=str(output),
        output_sha256=native_file_checksum(output), hypotheses=definitions, events=len(events),
        original_pairs=len(pairs), raw_identity_fields=list(raw_identity_fields), counts=dict(counts),
        inference='Both native detector fits valid plus complete physical alternative; otherwise entire original state.',
        truth_scope='Whole-file GEN/hit integrity validation and later closure only; no event or state selection.')


def run_paired_audit(npz_path, extraction_path, protocol_path, mode='development',
                     frozen_policy_path=None, bootstrap=200, seed=190019):
    """Compare fixed policies without opening validation outcomes during tuning."""
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes)
    if not protocol.get('created_before_full_outcome_analysis'):
        raise ValueError('Expected a predeclared validation protocol')
    extraction_bytes = Path(extraction_path).read_bytes()
    extraction = json.loads(extraction_bytes)
    input_stamp = Path(npz_path).stat()
    with np.load(npz_path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    sources = extraction['sources']
    masks = paired_development_masks(data, sources)
    policies = PAIRED_POLICIES
    frozen = None
    if mode == 'validation':
        if not frozen_policy_path:
            raise ValueError('Formal validation requires a frozen policy decision file')
        frozen = json.loads(Path(frozen_policy_path).read_text())
        winner = frozen['policy']
        if winner not in PAIRED_POLICIES:
            raise ValueError('Frozen policy is not one of the predeclared candidates')
        policies = tuple(dict.fromkeys(('canonical', 'stable', winner)))
        stages = ('event_holdout', 'point_holdout', 'process_holdout', 'transfer_grid')
    elif mode == 'development':
        stages = ('development', 'internal_check')
    else:
        raise ValueError('Unknown audit mode')
    # Restrict the arrays before evaluating any outcome. Development mode never
    # computes a hidden metric on the formal event, point, or process holdouts.
    allowed = np.logical_or.reduce([masks[stage] for stage in stages])
    data = {name: value[allowed] for name, value in data.items()}
    masks = {name: value[allowed] for name, value in masks.items()}
    measured = {name: value for name, value in data.items() if name.startswith(('pair_', 'mu1_', 'mu2_'))}
    states, choices = paired_policy_states(measured, policies)
    residuals = {policy: reconstruction_residuals(state, data) for policy, state in states.items()}
    both = (data['pair_topologyMin'] == 2) & (data['pair_topologyMax'] == 2)
    matched = (data['truth_pair_matched'] == 1) & np.isfinite(data['truth_pair_mass']) & (data['truth_pair_mass'] > 0.)
    report = dict(schema='shift-paired-reconstruction-policy-audit-v1', mode=mode,
                  inference_inputs='Measured track and native fit branches only; truth is evaluation only.',
                  joint_fit_prior='Native common-vertex fit has no target-z, mass or lifetime constraint, but can retain a transverse beamline Gaussian prior; verify its width in each frozen campaign before calling a result prior-free.',
                  counterfactuals='mc_only quantities replace truth directions or magnitudes solely after inference; they are not reconstructable estimators.',
                  weighting='Unweighted available reconstructed pairs; no rate or GEN efficiency claim.',
                  confidence='Paired physical-parent bootstrap; equal parent resampling preserves rows and muon correlations within a physical parent.',
                  event_holdout_excludes='Explicit initial_canary_peeked and prior_development_peeked physical-parent markers; source reindexing or rehashed buckets cannot turn a known parent into an unseen outcome.',
                  tail_units='tail100/tail50 mean absolute residual >1/>0.5 in the stated residual units; for relative mass/momentum these are100%/50%. Positions are cm; opening angle radians.',
                  bootstrap_draws=bootstrap, seed=seed, policies=list(policies), frozen_policy=frozen,
                  input_npz=str(Path(npz_path).resolve()),
                  input_npz_sha256=hashlib.sha256(Path(npz_path).read_bytes()).hexdigest(),
                  extraction_sha256=hashlib.sha256(extraction_bytes).hexdigest(),
                  protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  extraction_complete=extraction.get('complete', False), source_count=len(sources), groups=[])
    for stage in stages:
        scope = masks[stage]
        groups = [('all', scope)]
        for process in sorted({row['process'] for row in sources}):
            indices = [row['index'] for row in sources if row['process'] == process]
            groups.append(('process/'+process, scope & np.isin(data['source_index'], indices)))
        for sample in sorted({row['sample'] for row in sources}):
            indices = [row['index'] for row in sources if row['sample'] == sample]
            groups.append(('point/'+sample, scope & np.isin(data['source_index'], indices)))
            # Direct resonance daughters are an evaluation population only.
            # They never enter paired_policy_states or change reconstructed pairs.
            parent_pdg = 32 if sample.startswith('darkphoton_') else 443 if sample.startswith('jpsi_') else None
            if parent_pdg is not None:
                direct = (data['truth_pair_same_parent'] == 1) & (np.abs(data['truth_pair_parent_pdgId']) == parent_pdg)
                groups.append(('signal_point/'+sample, scope & np.isin(data['source_index'], indices) & direct))
        radius = np.hypot(data['truth_pair_vx'], data['truth_pair_vy'])
        common_vertex = data['truth_pair_shared_vertex'] == 1
        for low, high in ((0., 1.), (1., 10.), (10., 100.), (100., 300.), (300., 1000.), (1000., math.inf)):
            label = f'true_radius_cm/{low:g}..{high:g}'
            groups.append((label, scope & common_vertex & (radius >= low) & (radius < high)))
        for label, group in groups:
            evaluation = group & both & matched
            if not evaluation.any():
                continue
            item = dict(stage=stage, group=label, reconstructed_pairs=int(group.sum()),
                        reconstructed_both_both=int((group & both).sum()), matched_both_both=int(evaluation.sum()),
                        choices={}, metrics={})
            for policy in policies:
                item['choices'][policy] = {name: int((mask & evaluation).sum()) for name, mask in choices[policy].items()}
            for metric in residuals['canonical']:
                muon = metric.startswith('muon_')
                selected = np.concatenate((evaluation, evaluation)) if muon else evaluation
                parent = np.concatenate((data['physical_parent_key'], data['physical_parent_key'])) if muon else data['physical_parent_key']
                samples = {policy: residuals[policy][metric][selected] for policy in policies}
                item['metrics'][metric] = paired_parent_statistics(samples, parent[selected], bootstrap, seed)
            report['groups'].append(item)
    report['balanced_point_comparison'] = []
    for stage in stages:
        points = [row for row in report['groups'] if row['stage'] == stage and row['group'].startswith('point/')
                  and row['metrics']['mass']['quantile_resolved']]
        for policy in policies:
            ratios, bias_changes, tail_changes = [], [], []
            for point in points:
                baseline = point['metrics']['mass']['policies']['canonical']['estimate']
                candidate = point['metrics']['mass']['policies'][policy]['estimate']
                if baseline['sigma68'] <= 0.:
                    continue
                ratios.append(candidate['sigma68']/baseline['sigma68'])
                bias_changes.append(abs(candidate['median'])-abs(baseline['median']))
                tail_changes.append(candidate['tail100']-baseline['tail100'])
            if ratios:
                report['balanced_point_comparison'].append(dict(
                    stage=stage, policy=policy, points=len(ratios),
                    median_width_ratio=float(np.median(ratios)), worst_width_ratio=float(max(ratios)),
                    points_narrower=int(np.sum(np.asarray(ratios) < 1.)),
                    mean_absolute_bias_change=float(np.mean(bias_changes)),
                    mean_tail100_change=float(np.mean(tail_changes)),
                    point_weights='Equal weight per point with at least20 matched pairs and2 physical parents; no winner selected.'))
    final_stamp = Path(npz_path).stat()
    if (input_stamp.st_size, input_stamp.st_mtime_ns) != (final_stamp.st_size, final_stamp.st_mtime_ns):
        raise RuntimeError('The branch extraction changed during this audit; wait for its completed checkpoint and rerun')
    return report


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
    if not f or f.IsZombie() or f.TestBit(ROOT.TFile.kRecovered):
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
    h1('constrainedStatus', 16, -11.5, 4.5)
    h1('targetForwardStatus', 12, -10.5, 1.5)
    h1('targetForwardIterations', 66, -1.5, 64.5)
    h1('constrainedTargetChi2', 102, -1.5, 100.5)
    for name in ('massScale', 'constrainedMassScale', 'vertexRefitMassScale'):
        h1(name, 160, 0., 4.)
    h1('vertexRefitMassPull', 200, -20., 20.)
    h1('vertexRefitMinQoverPSignificance', 100, 0., 20.)
    h1('vertexRefitMassRelativeErr', 100, 0., 5.)
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
                       'vertexMassScaleVsCurvature': (40,0.,20.,100,0.,5.),
                       'vertexMassPullVsCurvature': (40,0.,20.,200,-20.,20.),
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
            for name in ('constrainedStatus', 'targetForwardStatus', 'targetForwardIterations', 'constrainedTargetChi2'):
                if 'ShiftMuon_' + name in branches:
                    value = float(get(name))
                    fill(name, value)
                    if name.endswith('Status'):
                        counts[f'{name}_{int(value)}'] += 1
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
                if 'ShiftDimuonVertex_refittedMassErr' in branches:
                    error = float(get('refittedMassErr'))
                    if error > 0.: fill('vertexRefitMassPull',(float(get('refittedMass'))-mass)/error)
                    if error >= 0. and float(get('refittedMass')) > 0.:
                        fill('vertexRefitMassRelativeErr',error/float(get('refittedMass')))
                    fill('vertexRefitMinQoverPSignificance',float(get('refittedMinQoverPSignificance')))
                    significance=float(get('refittedMinQoverPSignificance'))
                    hist['vertexMassScaleVsCurvature'].Fill(significance,float(get('refittedMass'))/mass)
                    if error>0.: hist['vertexMassPullVsCurvature'].Fill(significance,(float(get('refittedMass'))-mass)/error)
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
    overlay(['constrainedStatus','targetForwardStatus','targetForwardIterations','constrainedTargetChi2'],
            'Fit status and convergence: every reconstructed row, including failed and unattempted fits')
    overlay(['eta','phi','constrainedEta','constrainedPhi'],'Hit-associated residuals; no direction or residual cuts; failures retained in coverage')
    overlay(['axisEta','axisPhi','matchedTopology','reverseTopology'],'Axis-only diagnostic folds orientation; canonical residuals above remain unfolded')
    overlay(['massScale','constrainedMassScale','etaErr','phiErr'],'Independent pair identity; target uncertainties where stored')
    overlay(['etaPull','phiPull','targetPullX','targetPullY'],'Target-state pulls; target X/Y use innovation uncertainties')
    overlay(['massScale','constrainedMassScale','vertexRefitMassScale','topology'],'Same independently identified pair; separate unconstrained, target and vertex hypotheses')
    overlay(['vertexRefitMassScale','vertexRefitMassPull','vertexRefitMassRelativeErr','vertexRefitMinQoverPSignificance'],
            'Joint vertex refit: correlated mass uncertainty and curvature significance; no quality or truth-residual cuts')
    for names, heading in (
        (('phiVsP','constrainedPhiVsP'), 'Unfolded #Delta#phi versus true momentum; no residual selection'),
        (('constrainedEtaVsP','reverseVsAlgorithm'), 'Target #Delta#eta versus momentum; orientation by reconstruction algorithm'),
        (('vertexMassScaleVsCurvature','vertexMassPullVsCurvature'), 'Vertex-refit mass versus measured curvature significance; no quality cuts'),
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
                      'vertexMassScaleVsCurvature':('minimum |q/p| / #sigma(q/p)','refitted m / m_{gen}'),
                      'vertexMassPullVsCurvature':('minimum |q/p| / #sigma(q/p)','refitted mass pull'),
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


def plot_paired_report(report, output):
    """Category-specific fit diagnostics from already evaluated paired metrics."""
    ROOT.gStyle.SetPaperSize(28., 19.5)
    ROOT.gStyle.SetOptStat(0)
    category = report.get('category', 'Both-Both')
    canvas = ROOT.TCanvas('paired_reco_report', category+' reconstruction', 1500, 1040)
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.Print(output+'[')
    policies = [p for p in report['policies'] if p != 'stable']
    labels = {'canonical': 'Original', 'joint': 'Common vertex', 'joint_sig2': 'Common vertex, |q/p|/#sigma#geq2',
              'joint_sig3': 'Common vertex, |q/p|/#sigma#geq3'}
    labels.update(report.get('policy_labels', {}))
    colors = [ROOT.kGray+2, ROOT.kAzure+2, ROOT.kOrange+7, ROOT.kGreen+2, ROOT.kMagenta+1]
    page_number = 0

    counterfactual_curves = [('mass', 'Original measured state'),
                            ('mc_only_true_angles_mass', 'MC: replace directions'),
                            ('mc_only_true_magnitudes_mass', 'MC: replace magnitudes')]

    def page(title, counterfactual=False):
        nonlocal page_number
        page_number += 1
        canvas.cd(); canvas.Clear()
        text = ROOT.TLatex(); text.SetNDC(True); text.SetTextFont(42); text.SetTextSize(.03)
        text.DrawLatex(.035, .962, category+' | '+title)
        legend = ROOT.TLegend(.19, .905, .95, .945)
        legend.SetNColumns(3 if counterfactual else len(policies)); legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextSize(.021)
        retained = [text, legend]
        entries = counterfactual_curves if counterfactual else [(p, labels[p]) for p in policies]
        for index, (policy, label) in enumerate(entries):
            graph = ROOT.TGraphAsymmErrors()
            graph.SetLineColor(colors[index]); graph.SetMarkerColor(colors[index]); graph.SetMarkerStyle(20+index)
            graph.SetLineWidth(2); legend.AddEntry(graph, label, 'lep'); retained.append(graph)
        legend.Draw()
        text.SetTextSize(.018)
        footer = ('MC replacements are correlated diagnostics, not estimators or resolution floors | bars: parent-bootstrap 68% intervals'
                  if counterfactual else 'Same pairs + explicit fallback | unweighted diagnostic draw | bars: paired parent-bootstrap 68% intervals')
        text.DrawLatex(.035, .02, footer)
        text.DrawLatex(.94, .02, str(page_number))
        panel = ROOT.TPad('paired_panels', '', 0., .05, 1., .90)
        panel.Draw(); panel.Divide(2, 2, .012, .016); retained.append(panel)
        return panel, retained

    def draw(pad, rows, metric, statistic, title, x_title, y_title, factor=1., logarithmic_x=False, categorical=False,
             counterfactual=False):
        pad.SetLeftMargin(.18); pad.SetRightMargin(.055); pad.SetBottomMargin(.23); pad.SetTopMargin(.12)
        pad.SetTicks(1, 1)
        multigraph = ROOT.TMultiGraph(); retained = [multigraph]
        lower, upper, xvalues = [], [], []
        entries = [(name, 'canonical') for name, _ in counterfactual_curves] if counterfactual else [(metric, p) for p in policies]
        for index, (curve_metric, policy) in enumerate(entries):
            graph = None
            for x, row, label in rows:
                summary = row['metrics'].get(curve_metric)
                if not summary or not summary['quantile_resolved']:
                    graph = None
                    continue
                result = summary['policies'][policy]
                if 'interval68' not in result:
                    graph = None
                    continue
                if graph is None:
                    graph = ROOT.TGraphAsymmErrors()
                    graph.SetLineColor(colors[index]); graph.SetMarkerColor(colors[index]); graph.SetMarkerStyle(20+index)
                    graph.SetLineWidth(2); graph.SetMarkerSize(.85); multigraph.Add(graph, 'LP'); retained.append(graph)
                value = factor*result['estimate'][statistic]
                interval = result['interval68'][statistic]
                down, up = factor*interval['error_down'], factor*interval['error_up']
                n = graph.GetN(); graph.SetPoint(n, x, value); graph.SetPointError(n, 0., 0., down, up)
                xvalues.append(x); lower.append(value-down); upper.append(value+up)
        multigraph.SetTitle(title+';'+x_title+';'+y_title)
        if xvalues:
            multigraph.Draw('A')
            for axis in (multigraph.GetXaxis(), multigraph.GetYaxis()):
                axis.SetTitleSize(.052); axis.SetLabelSize(.043); axis.SetNdivisions(505)
            multigraph.GetYaxis().SetTitleOffset(1.4)
            lo, hi = min(lower), max(upper)
            span = max(hi-lo, .1*max(abs(lo), abs(hi)), 1.e-5)
            multigraph.SetMinimum(min(0., lo-.1*span)); multigraph.SetMaximum(max(0., hi)+.2*span)
            if statistic == 'sigma68' and lo > 0. and hi/lo > 20.:
                pad.SetLogy(True); multigraph.SetMinimum(lo*.7); multigraph.SetMaximum(hi*1.4)
            if logarithmic_x:
                pad.SetLogx(True); multigraph.GetXaxis().SetLimits(min(xvalues)*.75, max(xvalues)*1.3)
                multigraph.GetXaxis().SetMoreLogLabels(True)
                multigraph.GetXaxis().SetNoExponent(True)
            if categorical:
                multigraph.GetXaxis().SetLimits(.5, len(rows)+.5)
                axis = multigraph.GetXaxis(); axis.SetLabelSize(.065)
                for x, _, label in rows:
                    axis.SetBinLabel(axis.FindBin(x), label)
                axis.LabelsOption('h')
            if statistic == 'median':
                line = ROOT.TLine(multigraph.GetXaxis().GetXmin(), 0., multigraph.GetXaxis().GetXmax(), 0.)
                line.SetLineStyle(2); line.SetLineColor(ROOT.kGray+1); line.Draw(); retained.append(line)
        else:
            text = ROOT.TLatex(); text.SetNDC(True); text.SetTextSize(.05)
            text.DrawLatex(.12, .72, title); text.DrawLatex(.12, .5, 'Fewer than 20 matched observations per point')
            retained.append(text)
        return retained

    summary_rows = [row for row in report['groups'] if row['group'] == 'all']
    for start in range(0, len(summary_rows), 2):
        panels, retained = page('Mass and position | all reconstruction options')
        canvas.cd()
        text = ROOT.TLatex(); text.SetNDC(True); text.SetTextFont(42); text.SetTextSize(.029)
        text.DrawLatex(.055, .83, 'Compare the mass width, mass bias and all three position widths together.')
        text.SetTextSize(.022)
        y = .76
        for row in summary_rows[start:start+2]:
            summary = row['metrics']['mass']
            vertices = row['metrics']['vertex_z']['entries']
            if not summary['quantile_resolved']:
                text.DrawLatex(.055, y, row['stage'].replace('_', ' ')+
                               f': mass N={summary["entries"]}, vertex N={vertices}; too few pairs for widths.')
                y -= .08
                continue
            title = row['stage'].replace('_', ' ')
            text.DrawLatex(.055, y, f'{title}: mass N={summary["entries"]}, vertex N={vertices}; '
                           f'{summary["physical_parents"]} physical events')
            y -= .045
            for x, label in ((.055, 'Option'), (.34, 'Mass width [%]'), (.51, 'Mass bias [%]'),
                             (.68, 'x / y width [cm]'), (.86, 'z width [m]')):
                text.DrawLatex(x, y, label)
            y -= .04
            for policy in policies:
                mass = summary['policies'][policy]['estimate']
                widths = []
                for axis in 'xyz':
                    position = row['metrics']['vertex_'+axis]
                    widths.append(position['policies'][policy]['estimate']['sigma68']
                                  if position['quantile_resolved'] else math.nan)
                text.DrawLatex(.055, y, labels[policy])
                text.DrawLatex(.37, y, f'{100*mass["sigma68"]:.2f}')
                text.DrawLatex(.54, y, f'{100*mass["median"]:+.2f}')
                text.DrawLatex(.70, y, f'{widths[0]:.2f} / {widths[1]:.2f}')
                text.DrawLatex(.88, y, f'{.01*widths[2]:.2f}')
                y -= .038
            y -= .04
        text.SetTextSize(.024)
        text.DrawLatex(.055, .17, 'Where statistics permit, intervals are shown in plots and the numerical report.')
        text.DrawLatex(.055, .12, 'All alternatives retain the same pairs through fallback. Sparse checks cannot establish precision.')
        retained.append(text); canvas.Print(output, 'Title:Results and limits')

    panels, retained = page('How to read this comparison')
    canvas.cd()
    text = ROOT.TLatex(); text.SetNDC(True); text.SetTextFont(42); text.SetTextSize(.026)
    notes = [
        'Inference uses measured track states and fit diagnostics only.',
        'No sample mass, lifetime, process, truth match or response enters the choice.',
        'Common-vertex fits have free z and a transverse beam-line prior; not a prompt-target constraint.',
        'Failed or unsuitable alternatives keep the original state. No difficult pair is removed.',
        'Resolution is the central 68.27% half-width; bias is the median residual.',
        'Candidates keep their original measured '+category+' topology. Truth is used only for closure.',
        'Mass scans use independently hit-matched daughters of the same dark-photon parent.',
        'The original truth-independent event selection, identities and native weights are retained.',
        'These bounded samples are not a physical cross-process rate mixture or a GEN denominator.',
        'QCD pair-mass closure is not resonance resolution; a common production point is not guaranteed.',
        'Development, event checks and whole-point/process transfer tests appear separately.',
        'Sparse points stay blank. Connecting lines guide the eye and stop at unresolved bins.',
        'A smaller mass width does not establish better vertex resolution or absence of bias.',
    ]
    if report.get('mode') == 'native':
        notes[2] = 'Common-plane position-only fit: free z, no curvature update; prior settings are in the frozen configs.'
        notes[6] = 'Mass scans use hit-matched daughters of one resonance parent; all-pair closure also stays in the JSON.'
        notes[10] = 'Native cohorts are predeclared; previously viewed outcomes are labelled as development.'
        if report.get('fit_uncertainty_note'):
            notes[12] = report['fit_uncertainty_note']
    if report.get('vertex_prior_note'):
        notes[2] = report['vertex_prior_note']
    for index, note in enumerate(notes): text.DrawLatex(.055, .83-.055*index, note)
    retained.append(text); canvas.Print(output, 'Title:Guide')
    stages = list(dict.fromkeys(row['stage'] for row in report['groups']))
    stage_labels = {'development': 'Development', 'internal_check': 'Internal check',
                    'event_holdout': 'Held-out events', 'point_holdout': 'Held-out masses / lifetimes',
                    'process_holdout': 'DY / QCD transfer', 'transfer_grid': 'Remaining DP points',
                    'native_development': 'Native development', 'native_validation': 'Native validation',
                    'native_jpsi_control': 'J/#psi detector control'}
    for stage in stages:
        groups = [row for row in report['groups'] if row['stage'] == stage]
        processes = [row for row in groups if row['group'].startswith('process/')]
        if not any(summary['quantile_resolved'] for row in processes for summary in row['metrics'].values()):
            continue
        categorical = [(i+1., row, row['group'].split('/')[-1].replace('DarkPhoton', "A'")) for i, row in enumerate(processes)]
        global_metrics = (
            ('mass', 'Relative mass', 'Residual [%]', 100.),
            ('vertex_x' if report.get('mode') == 'native' else 'muon_pt',
             'Production x' if report.get('mode') == 'native' else 'Relative muon p_{T}',
             'Residual [cm]' if report.get('mode') == 'native' else 'Residual [%]',
             1. if report.get('mode') == 'native' else 100.),
            ('vertex_y' if report.get('mode') == 'native' else 'muon_p',
             'Production y' if report.get('mode') == 'native' else 'Relative muon |p|',
             'Residual [cm]' if report.get('mode') == 'native' else 'Residual [%]',
             1. if report.get('mode') == 'native' else 100.),
            ('vertex_z', 'Production z', 'Residual [m]', .01))
        for statistic in ('sigma68', 'median'):
            quantity = '68% half-width' if statistic == 'sigma68' else 'Median'
            if any(row['metrics'][metric]['quantile_resolved']
                   for row in processes for metric, _, _, _ in global_metrics):
                panels, retained = page(stage_labels.get(stage, stage)+' | '+('resolution' if statistic == 'sigma68' else 'bias'))
                for index, (metric, title, ytitle, factor) in enumerate(global_metrics):
                    retained += draw(panels.cd(index+1), categorical, metric, statistic, title, 'Process', quantity+' '+ytitle, factor, categorical=True)
                canvas.cd(); canvas.Print(output, 'Title:'+stage+' '+statistic)
            panels, retained = page(stage_labels.get(stage, stage)+' | '+('momentum resolution' if statistic == 'sigma68' else 'momentum bias'))
            for index, (metric, title) in enumerate((('muon_pz', 'Relative muon p_{z}'),
                    ('pair_pt', 'Relative dimuon p_{T}'), ('pair_pz', 'Relative dimuon p_{z}'),
                    ('pair_p', 'Relative dimuon |p|'))):
                retained += draw(panels.cd(index+1), categorical, metric, statistic, title, 'Process',
                                 quantity+' residual [%]', 100., categorical=True)
            canvas.cd(); canvas.Print(output, 'Title:'+stage+' momentum '+statistic)
            if report.get('mode') == 'native' and any('muon_pt' in row['metrics'] for row in processes):
                panels, retained = page(stage_labels.get(stage, stage)+' | muon / opening-angle '+statistic)
                for index, (metric, title, factor, y_title) in enumerate((
                        ('muon_pt', 'Relative muon p_{T}', 100., 'p_{T} residual [%]'),
                        ('muon_pz', 'Relative muon p_{z}', 100., 'p_{z} residual [%]'),
                        ('muon_p', 'Relative muon |p|', 100., '|p| residual [%]'),
                        ('opening_angle', 'Dimuon opening angle', 1000., 'Angle residual [mrad]'))):
                    retained += draw(panels.cd(index+1), categorical, metric, statistic, title,
                                     'Process', y_title, factor, categorical=True)
                canvas.cd(); canvas.Print(output, 'Title:'+stage+' muon '+statistic)
        families = {}
        for row in groups:
            prefix = 'signal_point/darkphoton_m'
            if not row['group'].startswith(prefix): continue
            mass, family = row['group'].removeprefix(prefix).split('_', 1)
            families.setdefault(family, []).append((float(mass.replace('p', '.')), row, ''))
        anchors = ('prompt', 'lab_5m', 'lab_30m', 'lab_100m', 'lab_1000m')
        families = {name: rows for name, rows in families.items()
                    if (report.get('mode') != 'native' and name in anchors) or
                    any(row['metrics']['mass']['quantile_resolved'] for _, row, _ in rows)}
        for family in families: families[family].sort(key=lambda item: item[0])
        for statistic in ('sigma68', 'median'):
            ordered = sorted(families, key=lambda name: (name != 'prompt', name))
            for start in range(0, len(ordered), 4):
                panels, retained = page(stage_labels.get(stage, stage)+' | mass '+('resolution' if statistic == 'sigma68' else 'bias'))
                for index, family in enumerate(ordered[start:start+4]):
                    title = 'Prompt' if family == 'prompt' else 'Mean LAB '+family.removeprefix('lab_').replace('em', 'e-').replace('p', '.')
                    retained += draw(panels.cd(index+1), families[family], 'mass', statistic, title,
                                     "Nominal A' mass [GeV]", 'Relative mass residual [%]', 100., logarithmic_x=True)
                canvas.cd(); canvas.Print(output, 'Title:'+stage+' mass '+statistic+' '+str(start//4+1))
        for quantity, x_title in report.get('scan_definitions', {}).items():
            rows = []
            for row in groups:
                if not row['group'].startswith('scan/'+quantity+'/'):
                    continue
                low, high = map(float, row['group'].split('/')[-1].split('..'))
                rows.append((.5*(low+high), row, ''))
            rows.sort(key=lambda item: item[0])
            if not any(row['metrics']['mass']['quantile_resolved'] for _, row, _ in rows):
                continue
            for statistic in ('sigma68', 'median'):
                panels, retained = page(stage_labels.get(stage, stage)+' | '+quantity+' | '+
                                       ('resolution' if statistic == 'sigma68' else 'bias'))
                for index, (metric, title, y_title, factor) in enumerate((
                        ('mass', 'Relative mass', 'Mass residual [%]', 100.),
                        ('pair_pt', 'Relative dimuon p_{T}', 'p_{T} residual [%]', 100.),
                        ('vertex_z', 'Production z', 'z residual [m]', .01),
                        ('vertex_x', 'Production x', 'x residual [cm]', 1.))):
                    retained += draw(panels.cd(index+1), rows, metric, statistic, title, x_title, y_title, factor,
                                     logarithmic_x=quantity != 'vz')
                canvas.cd(); canvas.Print(output, 'Title:'+stage+' scan '+quantity+' '+statistic)
                if report.get('mode') == 'native':
                    panels, retained = page(stage_labels.get(stage, stage)+' | '+quantity+' | momentum / position '+
                                           ('resolution' if statistic == 'sigma68' else 'bias'))
                    for index, (metric, title, y_title, factor) in enumerate((
                            ('pair_pz', 'Relative dimuon p_{z}', 'p_{z} residual [%]', 100.),
                            ('pair_p', 'Relative dimuon |p|', '|p| residual [%]', 100.),
                            ('vertex_y', 'Production y', 'y residual [cm]', 1.),
                            ('opening_angle', 'Dimuon opening angle', 'Angle residual [mrad]', 1000.))):
                        retained += draw(panels.cd(index+1), rows, metric, statistic, title, x_title,
                                         y_title, factor, logarithmic_x=quantity != 'vz')
                    canvas.cd(); canvas.Print(output, 'Title:'+stage+' mixed '+quantity+' '+statistic)
        for quantity, x_title in report.get('muon_scan_definitions', {}).items():
            rows = []
            for row in groups:
                if row['group'].startswith('muon_scan/'+quantity+'/'):
                    low, high = map(float, row['group'].split('/')[-1].split('..'))
                    rows.append((.5*(low+high), row, f'{low:g}..{high:g}'))
            rows.sort(key=lambda item: item[0])
            if not any(row['metrics']['muon_pt']['quantile_resolved'] for _, row, _ in rows):
                continue
            for statistic in ('sigma68', 'median'):
                panels, retained = page(stage_labels.get(stage, stage)+' | muon '+quantity+' | '+statistic)
                for index, (metric, title) in enumerate((('muon_pt', 'Relative muon p_{T}'),
                        ('muon_pz', 'Relative muon p_{z}'), ('muon_p', 'Relative muon |p|'))):
                    retained += draw(panels.cd(index+1), rows, metric, statistic, title, x_title, 'Residual [%]', 100.,
                                     logarithmic_x=True)
                panels.cd(4)
                text = ROOT.TLatex(); text.SetNDC(True); text.SetTextSize(.04)
                text.DrawLatex(.12, .88, 'Matched muons / physical events per bin')
                for index, (_, row, label) in enumerate(rows):
                    summary = row['metrics']['muon_pt']
                    text.DrawLatex(.12, .78-.07*index,
                        label+f': {summary["entries"]} / {summary["physical_parents"]}')
                retained.append(text)
                canvas.cd(); canvas.Print(output, 'Title:'+stage+' muon scan '+quantity+' '+statistic)
        radial = [row for row in groups if row['group'].startswith('true_radius_cm/')]
        if radial:
            rows = [(i+1., row, row['group'].split('/')[-1]) for i, row in enumerate(radial)]
            panels, retained = page(stage_labels.get(stage, stage)+' | displacement checks')
            for index, (metric, statistic, title, ytitle, factor) in enumerate((
                    ('mass', 'sigma68', 'Mass width vs true radius', 'Relative mass width [%]', 100.),
                    ('mass', 'median', 'Mass bias vs true radius', 'Relative mass median [%]', 100.),
                    ('opening_angle', 'sigma68', 'Opening-angle width vs true radius', 'Width [mrad]', 1000.),
                    ('vertex_z', 'sigma68', 'z width vs true radius', 'Width [m]', .01))):
                retained += draw(panels.cd(index+1), rows, metric, statistic, title, 'True radius bin [cm]', ytitle, factor, categorical=True)
            canvas.cd(); canvas.Print(output, 'Title:'+stage+' displacement')
    # These disjoint held-out point families diagnose which measured quantity
    # limits mass reconstruction. Truth replacements never enter inference.
    families = {}
    for row in report['groups']:
        prefix = 'signal_point/darkphoton_m'
        if row['stage'] not in ('point_holdout', 'transfer_grid', 'native_development', 'native_validation') or not row['group'].startswith(prefix):
            continue
        mass, family = row['group'].removeprefix(prefix).split('_', 1)
        families.setdefault(family, []).append((float(mass.replace('p', '.')), row, ''))
    families = {name: rows for name, rows in families.items()
                if any(row['metrics']['mass']['quantile_resolved'] for _, row, _ in rows)}
    if families:
        panels, retained = page('MC diagnostic only | directions versus momentum magnitudes', counterfactual=True)
        ordered = sorted(families, key=lambda name: (name != 'prompt', name))[:4]
        for index, family in enumerate(ordered):
            rows = sorted(families.get(family, []), key=lambda item: item[0])
            title = 'Prompt' if family == 'prompt' else 'Mean LAB '+family.removeprefix('lab_')
            retained += draw(panels.cd(index+1), rows, 'mass', 'sigma68', title,
                             "Nominal A' mass [GeV]", 'Relative mass 68% half-width [%]', 100.,
                             logarithmic_x=True, counterfactual=True)
        canvas.cd(); canvas.Print(output, 'Title:MC diagnosis')
    canvas.cd(); canvas.Print(output+']'); canvas.Close()
    return page_number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='*')
    parser.add_argument('--labels',nargs=2,default=['control','material + field'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--summary')
    parser.add_argument('--max-events',type=int,default=-1)
    parser.add_argument('--paired-plot-output', help='Category PDF for the evaluated paired-fit metrics')
    parser.add_argument('--paired-report-json', help='Render existing paired metrics without rereading or reevaluating samples')
    parser.add_argument('--paired-npz', help='Bounded branch extraction from the existing NanoAOD samples')
    parser.add_argument('--native-manifest', help='Paired native replay sources and measured alternative branch mappings')
    parser.add_argument('--native-detector-overlay', nargs=2, metavar=('BASELINE', 'REFIT'),
                        help='Prepare separate detector-refit branches while preserving every baseline event/pair')
    parser.add_argument('--extraction', help='Exact source inventory accompanying --paired-npz')
    parser.add_argument('--protocol', help='Predeclared development and validation protocol JSON')
    parser.add_argument('--mode', choices=('development', 'validation'), default='development')
    parser.add_argument('--frozen-policy', help='Required policy decision JSON before formal validation')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=190019)
    args=parser.parse_args()
    if args.native_detector_overlay:
        if args.inputs or args.native_manifest or args.paired_npz or args.paired_report_json:
            parser.error('Detector overlay requires only baseline/refit files and an output ROOT path')
        receipt = prepare_native_detector_overlay(*args.native_detector_overlay, args.output)
        Path(args.output).with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=args.output, events=receipt['events'], pairs=receipt['original_pairs'])))
        return
    if args.native_manifest:
        if args.inputs or args.paired_npz or args.paired_report_json or args.bootstrap < 20:
            parser.error('Native comparison requires a manifest, no other event inputs and at least20 bootstrap draws')
        reports = compare_native_reconstruction(args.native_manifest, args.bootstrap, args.seed)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for category, report in reports.items():
            (output/(category+'.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            if args.paired_plot_output:
                plot_paired_report(report, Path(args.paired_plot_output)/(category+'.pdf'))
        print(json.dumps(dict(output=str(output), categories=list(reports))))
        return
    if args.paired_report_json:
        if args.inputs or args.paired_npz:
            parser.error('--paired-report-json cannot be combined with event inputs')
        report = json.loads(Path(args.paired_report_json).read_text())
        plot_paired_report(report, args.output)
        return
    if args.paired_npz:
        if args.inputs or not args.extraction or not args.protocol or args.bootstrap < 20:
            parser.error('Paired mode requires --extraction, --protocol and at least20 bootstrap draws, with no positional ROOT inputs')
        report=run_paired_audit(args.paired_npz, args.extraction, args.protocol, args.mode,
                                args.frozen_policy, args.bootstrap, args.seed)
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if args.paired_plot_output:
            plot_paired_report(report, args.paired_plot_output)
        print(json.dumps(dict(output=str(output), mode=args.mode, groups=len(report['groups']),
                             policies=report['policies'], extraction_complete=report['extraction_complete'])))
        return
    if len(args.inputs) != 2:
        parser.error('The ROOT comparison requires exactly two positional input files')
    results=[analyze(path,label,i,args.max_events) for i,(path,label) in enumerate(zip(args.inputs,args.labels))]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    plot(results,Path(args.output))
    report=[r[0] for r in results]
    if args.summary: Path(args.summary).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','counts','statistics')} for r in report],indent=2))
if __name__=='__main__': main()
