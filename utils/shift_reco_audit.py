#!/usr/bin/env python3
# Code style: AI-code.
"""Reco-only distributions and independent hit-truth performance; no angular cuts.

ROOT input is read directly, so this also audits existing productions without
another reconstruction. All reco rows contribute to reco/coverage diagnostics.
Truth resolution requires a stored hit association, never a nearest direction.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import tarfile
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


ACTUAL_BIRTH_POSITION_TOLERANCE_CM = 1.e-9
ACTUAL_BIRTH_TIME_TOLERANCE_NS = 1.e-9


def read_actual_birth_reference(record, source_context):
    """Index every saved SimTrack birth, independently of GEN/CSC support.

    SimTrack IDs reset per event. The source context must identify the exact
    source/case; a GEN barcode is ancestry information, never a Nano index.
    """
    if not source_context or record.get('schema') != 'shift-material-scout-SIM-audit-v1':
        raise ValueError('Actual birth reference needs a source and supported audit schema')
    if record.get('passed') is not True or not record.get('checks') or not all(record['checks'].values()):
        raise ValueError('Actual birth SIM audit did not pass its source/identity checks')
    events = {}
    for event in record['events']:
        identity = tuple(event['event_id'])
        if len(identity) != 3 or any(type(n) is not int or n < 1 for n in identity):
            raise ValueError('Invalid actual birth event identity')
        if identity in events:
            raise ValueError('Duplicate actual birth event identity')
        weight = float(event['native_weight'])
        if not math.isfinite(weight):
            raise ValueError('Nonfinite actual birth native weight')
        if event['actual_saved_muons'] != len(event['muons']):
            raise ValueError('Actual birth saved-muon count mismatch')
        tracks = {}
        for track in event['muons']:
            number = track['track_id']
            if type(number) is not int or number < 1 or number in tracks:
                raise ValueError('Duplicate or invalid actual SimTrack ID')
            pdg = int(track['pdg'])
            momentum = np.asarray(track['momentum_GeV'], dtype=float)
            energy = float(track['energy_GeV'])
            valid = momentum.shape == (3,) and np.all(np.isfinite(momentum))
            p = float(np.linalg.norm(momentum)) if valid else 0.
            # This is a serialization/mass-shell check, not a fitted-quality cut.
            energy_valid = (math.isfinite(energy) and energy > 0.
                            and math.isclose(energy, math.hypot(p, MUON_MASS), rel_tol=1.e-8, abs_tol=1.e-6))
            vertex = track.get('vertex')
            vertex_status = 'missing'
            if vertex is not None:
                position = np.asarray(vertex['position_cm'], dtype=float)
                vertex_status = ('available' if position.shape == (3,) and np.all(np.isfinite(position))
                                 and math.isfinite(float(vertex['time_ns'])) else 'invalid')
                if vertex_status == 'invalid':
                    vertex = None
            tracks[number] = dict(context=source_context, event_id=identity, track_id=number,
                pdg=pdg, charge=-1 if pdg > 0 else 1, momentum=momentum,
                energy_gev=energy, valid=bool(valid and energy_valid and abs(pdg) == 13),
                vertex=vertex, vertex_status=vertex_status, GEN_barcode=track.get('GEN_barcode'),
                saved_parent_chain=track.get('saved_parent_chain', []),
                missing_saved_parent_track_id=track.get('missing_saved_parent_track_id'))
        events[identity] = dict(context=source_context, native_weight=weight, tracks=tracks)
    return events


def actual_birth_muon_reference(selected, event_reference):
    """Exact stored hit identity only; wrong measured charge remains matched."""
    number = int(selected['hitSimTrackId'])
    if number < 1:
        return dict(status='missing_hit_identity', reference=None, wrong_charge=None)
    reference = event_reference['tracks'].get(number)
    if reference is None:
        return dict(status='no_saved_actual_muon', reference=None, wrong_charge=None)
    if abs(reference['pdg']) != 13:
        return dict(status='non_muon_reference', reference=None, wrong_charge=None)
    if not reference['valid']:
        return dict(status='invalid_actual_birth_state', reference=None, wrong_charge=None)
    return dict(status='matched', reference=reference,
                wrong_charge=int(selected['charge']) != reference['charge'])


def actual_birth_pair_reference(first, second):
    """Keep two-leg birth kinematics separate from a common production vertex."""
    if first is None or second is None:
        return dict(status='unmatched_leg', shared_vertex=False)
    if (first['context'], first['event_id']) != (second['context'], second['event_id']):
        raise ValueError('Actual birth pair crosses source/event context')
    if first['track_id'] == second['track_id']:
        return dict(status='same_simtrack_split', shared_vertex=False)
    if not first['valid'] or not second['valid']:
        return dict(status='invalid_actual_birth_state', shared_vertex=False)
    a, b = first['momentum'], second['momentum']
    total = a+b
    result = dict(status='distinct_actual_muons',
        actual_is_os=first['pdg']*second['pdg'] < 0,
        kinematic_birth_mass_gev=float(stable_pair_mass(a, b)),
        pt_gev=float(np.hypot(*total[:2])), pz_gev=float(total[2]), p_gev=float(np.linalg.norm(total)),
        opening_angle=(float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))
                       if np.linalg.norm(a) > 0. and np.linalg.norm(b) > 0. else None),
        shared_vertex=False, common_vertex_reason='no_established_common_birth_vertex')
    av, bv = first['vertex'], second['vertex']
    if av is None or bv is None:
        return result
    positions = [np.asarray(v['position_cm'], dtype=float) for v in (av, bv)]
    times = [float(v['time_ns']) for v in (av, bv)]
    valid = all(v.shape == (3,) and np.all(np.isfinite(v)) for v in positions)
    valid = valid and all(math.isfinite(t) for t in times)
    same_position_time = (valid and np.allclose(*positions, rtol=0., atol=ACTUAL_BIRTH_POSITION_TOLERANCE_CM)
                          and abs(times[0]-times[1]) <= ACTUAL_BIRTH_TIME_TOLERANCE_NS)
    same_index = av['index'] >= 0 and av['index'] == bv['index']
    same_parent = av['parent_G4_track_id'] > 0 and av['parent_G4_track_id'] == bv['parent_G4_track_id']
    if same_position_time and (same_index or same_parent):
        result.update(shared_vertex=True, vertex_cm=positions[0].tolist(), time_ns=times[0],
                      common_vertex_reason='same_actual_vertex' if same_index else 'same_actual_parent_and_birth')
    return result


def actual_birth_case_inputs(case):
    """Require terminal joint publication before opening the paired evidence."""
    terminal_path = Path(case['terminal_result_path'])
    request_path = Path(case['request_path'])
    for path, expected in ((terminal_path, case['terminal_result_sha256']),
                           (request_path, case['request_sha256'])):
        if native_file_checksum(path) != expected:
            raise RuntimeError('Actual birth terminal/request hash mismatch')
    terminal = json.loads(terminal_path.read_text())
    request = json.loads(request_path.read_text())
    if (terminal.get('complete') is not True or terminal.get('exit_code') != 0
            or terminal.get('source_id') != case['source_id']
            or terminal.get('original_source_sha256') != case['source']['gen_sha256']
            or terminal.get('request_sha256') != case['request_sha256']):
        raise RuntimeError('Actual birth case is not a qualified terminal publication')
    sample = request['sample']
    if (request['source']['source_id'] != case['source_id']
            or request['source']['gen_sha256'] != case['source']['gen_sha256']
            or sample['id'] != case['case_id'] or sample['event_ids'] != case['expected_event_ids']
            or sample['native_weights'] != case['expected_native_EDM_double_weights']
            or sample['probability'] != case['sample']['probability']):
        raise RuntimeError('Actual birth case/request source, membership or weight binding mismatch')
    published = {row['url']: row for row in terminal['published']}
    for name, path in (('nano.root', case['nano_path']), ('evidence.tar.gz', case['material_evidence_path'])):
        expected = case['publications'][name]
        if published.get(expected['url']) != expected:
            raise RuntimeError('Actual birth Nano/evidence do not share the terminal publication')
        if Path(path).stat().st_size != expected['bytes'] or native_file_checksum(path) != expected['sha256']:
            raise RuntimeError('Actual birth published payload hash/size mismatch')
    with tarfile.open(case['material_evidence_path'], 'r:gz') as archive:
        member = archive.getmember('material_audit.json.gz')
        if member.size > 64*1024*1024:
            raise RuntimeError('Actual birth audit exceeds bounded payload size')
        with gzip.GzipFile(fileobj=io.BytesIO(archive.extractfile(member).read())) as stream:
            raw = stream.read(64*1024*1024+1)
    if len(raw) > 64*1024*1024:
        raise RuntimeError('Actual birth audit exceeds bounded decoded size')
    record = json.loads(raw)
    # The scout audits the timed/sliced GEN sent to SIM, not the full raw source.
    # Its two file hashes must bind to the same final worker receipt as Nano.
    scratch = {Path(row['path']).name: row['sha256'] for row in terminal.get('removed_scratch_inputs', [])}
    if (record['complete_GEN']['sha256'] != scratch.get('timed_gen.root')
            or record['SIM']['sha256'] != scratch.get('step1.root')):
        raise RuntimeError('Actual birth timed GEN/SIM hash binding mismatch')
    context = case['source']['gen_sha256']+':'+case['case_id']
    references = read_actual_birth_reference(record, context)
    expected = [tuple(event) for event in case['expected_event_ids']]
    if list(references) != expected or len(set(expected)) != len(expected):
        raise RuntimeError('Actual birth source event membership/order mismatch')
    weights = case['expected_native_EDM_double_weights']
    if len(weights) != len(expected) or any(references[event]['native_weight'] != weight
                                         for event, weight in zip(expected, weights)):
        raise RuntimeError('Actual birth source native double weights mismatch')
    return references


def actual_birth_check_weights(persisted, native, probability, native_type='Float_t'):
    """Use the declared scalar type, retaining the original EDM double separately."""
    if native_type not in ('Float_t', 'Double_t') or not math.isfinite(native):
        raise RuntimeError('Actual birth native weight has invalid type/value')
    if not math.isfinite(probability) or not 0. < probability <= 1.:
        raise RuntimeError('Actual birth invalid benchmark inclusion probability')
    expected = float(np.float32(native)) if native_type == 'Float_t' else native
    if not all(math.isfinite(w) for w in persisted.values()) or persisted['genWeight'] != expected:
        raise RuntimeError('Actual birth Nano native serialized weight mismatch')
    checked = dict(shiftOriginalGenWeight=native, shiftSamplingProbability=probability,
                   shiftSamplingWeight=1./probability, shiftSamplingGenWeight=native/probability,
                   shiftDecayProposalDensity=1., shiftDecayProposalWeight=1.,
                   shiftProposalGenWeight=native/probability)
    for name, value in checked.items():
        if name not in persisted or persisted[name] != value:
            raise RuntimeError('Actual birth Nano exact accounting weight mismatch: '+name)


def actual_birth_muon_metrics(measured, reference):
    """Target-side reconstructed momentum versus this actual muon's birth."""
    if not all(math.isfinite(measured[n]) for n in ('pt', 'phi', 'pz')) or measured['pt'] < 0.:
        return {}
    fitted = momentum_vector(np.array([measured['pt']]), np.array([measured['phi']]),
                             np.array([measured['pz']]))[0]
    truth = reference['momentum']
    metrics = {}
    for name, value, target in (('p_relative', np.linalg.norm(fitted), np.linalg.norm(truth)),
                                 ('pt_relative', measured['pt'], np.hypot(*truth[:2])),
                                 ('pz_relative', measured['pz'], truth[2])):
        if target != 0.:
            metrics[name] = float((value-target)/target)
    if np.linalg.norm(fitted) > 0. and np.linalg.norm(truth) > 0.:
        metrics['angle_error_rad'] = float(np.arctan2(np.linalg.norm(np.cross(fitted, truth)),
                                                    np.dot(fitted, truth)))
    return metrics


def read_actual_birth_nano(case, references):
    """Read all selected rows; the hit-ID join never changes their membership."""
    source = ROOT.TFile.Open(str(case['nano_path']))
    if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError('Unreadable actual birth Nano input')
    muon_fields = ('recoAlgorithm', 'sourceIndex', 'charge', 'topology', 'hitSimTrackId', 'pt', 'phi', 'pz')
    pair_fields = ('muonIdx1', 'muonIdx2', 'topologyMin', 'topologyMax', 'isOS', 'mass', 'pt', 'pz', 'vx', 'vy', 'vz')
    from shift_event_weights import DETECTOR_BRANCHES, PROPOSAL_BRANCHES
    weights = ('genWeight',)+DETECTOR_BRANCHES+PROPOSAL_BRANCHES
    try:
        tree = source.Get('Events')
        if not tree:
            raise RuntimeError('Missing actual birth Events')
        branches = {b.GetName() for b in tree.GetListOfBranches()}
        required = {'run', 'luminosityBlock', 'event', 'genWeight', 'nShiftMuon', 'nShiftDimuonVertex'}
        required.update('ShiftMuon_'+n for n in muon_fields)
        required.update('ShiftDimuonVertex_'+n for n in pair_fields)
        required.update(DETECTOR_BRANCHES+PROPOSAL_BRANCHES)
        if required-branches:
            raise RuntimeError('Actual birth Nano missing branches: '+str(sorted(required-branches)))
        weight_types = {name: tree.GetLeaf(name).GetTypeName() for name in weights if name in branches}
        if any(kind != 'Double_t' for name, kind in weight_types.items() if name != 'genWeight'):
            raise RuntimeError('Actual birth annotated accounting weights must be Double_t')
        tree.SetBranchStatus('*', False)
        for name in required | (set(weights)&branches):
            tree.SetBranchStatus(name, True)
        events, muons, pairs = [], [], []
        seen = set()
        for entry in range(int(tree.GetEntries())):
            if tree.GetEntry(entry) <= 0:
                raise RuntimeError('Actual birth Nano entry read failed')
            identity = tuple(int(getattr(tree, n)) for n in ('run', 'luminosityBlock', 'event'))
            if identity in seen or identity not in references:
                raise RuntimeError('Actual birth Nano duplicate/unrecognized event')
            seen.add(identity)
            reference = references[identity]
            expected_weight = reference['native_weight']
            saved_weights = {n: float(getattr(tree, n)) for n in weights if n in branches}
            probability = case['sample']['probability']
            actual_birth_check_weights(saved_weights, expected_weight, probability, weight_types['genWeight'])
            parent = reference['context']+':'+':'.join(map(str, identity))
            events.append(dict(identity=list(identity), parent=parent, native_EDM_double_weight=expected_weight,
                               persisted_weights=saved_weights, weight_types=weight_types, selected_muons=int(tree.nShiftMuon),
                               selected_pairs=int(tree.nShiftDimuonVertex),
                               actual_saved_muon_rows=len(reference['tracks']),
                               valid_actual_muon_births=sum(r['valid'] for r in reference['tracks'].values())))
            local, matches = [], []
            for index in range(int(tree.nShiftMuon)):
                measured = {n: float(getattr(tree, 'ShiftMuon_'+n)[index]) for n in muon_fields}
                matched = actual_birth_muon_reference(measured, reference)
                truth = matched['reference']
                row = dict(parent=parent, index=index, algorithm=int(measured['recoAlgorithm']),
                    source_index=int(measured['sourceIndex']), topology=int(measured['topology']),
                    hit_simtrack_id=int(measured['hitSimTrackId']), match_status=matched['status'],
                    wrong_charge=matched['wrong_charge'], metrics=actual_birth_muon_metrics(measured, truth) if truth else {})
                if truth:
                    row.update(actual_simtrack_id=truth['track_id'], actual_pdg=truth['pdg'],
                               actual_birth_momentum_gev=truth['momentum'].tolist(), actual_birth_vertex=truth['vertex'],
                               actual_birth_vertex_status=truth['vertex_status'])
                local.append(row);matches.append(truth);muons.append(row)
            for index in range(int(tree.nShiftDimuonVertex)):
                measured = {n: float(getattr(tree, 'ShiftDimuonVertex_'+n)[index]) for n in pair_fields}
                a, b = (int(measured[n]) for n in ('muonIdx1', 'muonIdx2'))
                if a == b or not all(0 <= i < len(local) for i in (a, b)):
                    raise RuntimeError('Actual birth pair has invalid selected leg indices')
                truth = actual_birth_pair_reference(matches[a], matches[b])
                row = dict(parent=parent, index=index, legs=[a, b],
                    category=str(int(measured['topologyMin']))+':'+str(int(measured['topologyMax'])),
                    is_os=bool(measured['isOS']), reference=truth, metrics={})
                if truth['status'] == 'distinct_actual_muons':
                    for name, field, target in (
                            ('two_birth_mass_relative', 'mass', truth['kinematic_birth_mass_gev']),
                            ('two_birth_pair_pt_relative', 'pt', truth['pt_gev']),
                            ('two_birth_pair_pz_relative', 'pz', truth['pz_gev'])):
                        value = measured[field]
                        if math.isfinite(value) and target != 0.:
                            row['metrics'][name] = float((value-target)/target)
                    if truth['shared_vertex']:
                        for axis, target in zip('xyz', truth['vertex_cm']):
                            if math.isfinite(measured['v'+axis]):
                                row['metrics']['shared_vertex_'+axis+'_cm'] = measured['v'+axis]-target
                pairs.append(row)
        if seen != set(references) or [tuple(e['identity']) for e in events] != list(references):
            raise RuntimeError('Actual birth Nano/audit event membership/order mismatch')
        return dict(events=events, muons=muons, pairs=pairs)
    finally:
        source.Close()


def actual_birth_group_statistics(rows, bootstrap=200, seed=190019):
    """Do not turn a few selected muons into a population resolution result."""
    result = {}
    metrics = sorted({name for row in rows for name in row['metrics']})
    for name in metrics:
        selected = [row for row in rows if name in row['metrics'] and math.isfinite(row['metrics'][name])]
        parents = [row['parent'] for row in selected]
        count = len(set(parents))
        item = dict(observations=len(selected), independent_originals=count,
                    population_summary_available=count >= 20, statistics=None)
        if count >= 20:
            item['statistics'] = paired_parent_statistics({'canonical': [r['metrics'][name] for r in selected]},
                                                         parents, bootstrap, seed)
        else:
            item['reason'] = 'Fewer than20 independent originals; ledger/coverage only.'
        result[name] = item
    return result


def run_actual_birth_audit(manifest_path, release_path, bootstrap=200, seed=190019):
    """Bounded closure from a fixed, jointly published material-response cohort."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    release = json.loads(Path(release_path).read_text())
    if (release.get('outcome_analysis_released') is not True
            or release.get('cohort_sha256') != native_file_checksum(manifest_path)
            or release.get('analysis_source_sha256') != native_file_checksum(__file__)):
        raise ValueError('Actual birth analysis requires release of this exact frozen cohort')
    if manifest.get('schema') != 'shift-corrected-QCD-offline-birth-join-frozen-cohort-v1':
        raise ValueError('Unsupported actual birth frozen cohort schema')
    if manifest['kernel']['pileup'] != 'off':
        raise ValueError('Actual birth hit-ID join needs no pileup until EncodedEventId is saved')
    cases = manifest['cases']
    if not cases or len(cases) > 9 or sum(len(c['expected_event_ids']) for c in cases) > 450:
        raise ValueError('Actual birth cohort exceeds the bounded source/event plan')
    for key in ('plan', 'deployment'):
        record = manifest[key]
        if native_file_checksum(record['path']) != record['sha256']:
            raise RuntimeError('Actual birth frozen plan/deployment hash mismatch')
    deployment = json.loads(Path(manifest['deployment']['path']).read_text())
    tools = Path(manifest['deployment']['path']).with_name('frozen_tools.tar.gz')
    if (deployment['frozen_tools_sha256'] != manifest['frozen_tools_sha256']
            or native_file_checksum(tools) != manifest['frozen_tools_sha256']):
        raise RuntimeError('Actual birth frozen runtime-tools binding mismatch')
    with tarfile.open(tools, 'r:gz') as archive:
        overlay = archive.extractfile('plugin/libShiftCompleteIncomingBaseline.so').read()
    if hashlib.sha256(overlay).hexdigest() != manifest['kernel']['incoming_overlay_sha256']:
        raise RuntimeError('Actual birth incoming-parent transport overlay mismatch')
    events, muons, pairs, exposure, seen = [], [], [], {}, set()
    for case in cases:
        references = actual_birth_case_inputs(case)
        for identity in references:
            key = (case['source']['gen_sha256'], identity)
            if key in seen:
                raise RuntimeError('Actual birth original appears in multiple cases')
            seen.add(key)
        rows = read_actual_birth_nano(case, references)
        for row in rows['events']:
            row.update(case_id=case['case_id'], source_id=case['source_id'])
        events.extend(rows['events']);muons.extend(rows['muons']);pairs.extend(rows['pairs'])
        source = case['source']
        key = source['gen_sha256']
        if key in exposure and exposure[key] != source:
            raise RuntimeError('Actual birth source exposure changed between cases')
        exposure[key] = source
    groups = {}
    categories = sorted({row['category'] for row in pairs})
    for category in ['all']+categories:
        selected = pairs if category == 'all' else [p for p in pairs if p['category'] == category]
        groups[category] = dict(selected_pairs=len(selected),
            reference_status=dict(Counter(r['reference']['status'] for r in selected)),
            shared_birth_vertex=sum(r['reference'].get('shared_vertex', False) for r in selected),
            statistics=actual_birth_group_statistics(selected, bootstrap, seed))
    muon_groups = {}
    for topology in sorted({r['topology'] for r in muons}):
        selected = [r for r in muons if r['topology'] == topology]
        muon_groups[str(topology)] = dict(selected_muons=len(selected),
            reference_status=dict(Counter(r['match_status'] for r in selected)),
            wrong_charge=sum(r['wrong_charge'] is True for r in selected),
            statistics=actual_birth_group_statistics(selected, bootstrap, seed))
    return dict(schema='shift-actual-SimTrack-birth-closure-v1', complete=True,
        analysis_source_sha256=native_file_checksum(__file__),
        manifest_sha256=native_file_checksum(manifest_path), release_sha256=native_file_checksum(release_path),
        kernel=manifest['kernel'], counts=dict(cases=len(cases), original_events=len(events),
            actual_saved_muon_rows=sum(e['actual_saved_muon_rows'] for e in events),
            valid_actual_muon_births=sum(e['valid_actual_muon_births'] for e in events),
            selected_muons=len(muons), selected_pairs=len(pairs),
            muon_reference_status=dict(Counter(r['match_status'] for r in muons)),
            matched_birth_vertex_status=dict(Counter(r['actual_birth_vertex_status'] for r in muons
                                                      if r['match_status'] == 'matched')),
            matched_muon_undefined_metrics=dict(Counter(n for r in muons if r['match_status'] == 'matched'
                for n in ('p_relative', 'pt_relative', 'pz_relative', 'angle_error_rad') if n not in r['metrics'])),
            wrong_reconstructed_charge=sum(r['wrong_charge'] is True for r in muons)),
        groups=groups, muon_groups=muon_groups, muon_statistics=actual_birth_group_statistics(muons, bootstrap, seed),
        events=events, selected_muon_ledger=muons, selected_pair_ledger=pairs,
        full_source_exposure=list(exposure.values()), missing_sources=manifest['missing_sources'],
        reference='All saved actual PDG13 SimTracks at birth; exact source/case/event/trackID, no GEN or angular fallback.',
        interpretation='Momentum closure includes detector reconstruction and material return to the target. Two-individual-birth-vector mass is kinematic, not a common-parent resonance or common-vertex claim.',
        mass_definition=dict(muon_mass_gev=MUON_MASS, stable_momentum_arithmetic=True,
            actual_energy_consistency_absolute_tolerance_gev=1.e-6, actual_energy_consistency_relative_tolerance=1.e-8),
        shared_vertex_definition=dict(position_tolerance_cm=ACTUAL_BIRTH_POSITION_TOLERANCE_CM,
            time_tolerance_ns=ACTUAL_BIRTH_TIME_TOLERANCE_NS,
            links='Same actual vertex index, or same explicit positive actual parent; both also need equal physical position/time.'),
        native_weight_definition='Original EDM double versus the declared Nano scalar serialization; all7 annotated Double accounting fields required and checked, source fraction r only, proposal density/correction1 and final W/r. Candidate probability p never enters these benchmark weights. No rate or inclusive efficiency.',
        population_summary_minimum_independent_originals=20, physics_normalization_ready=False)


def detector_pair_identity(first, second):
    """Raw source identities include the physical event, algorithm and source index."""
    first, second = tuple(first), tuple(second)
    if first == second or first[:-2] != second[:-2]:
        raise ValueError('Detector pair needs two distinct sources in the same event')
    return tuple(sorted((first, second)))


def validate_detector_audit_inputs(modes):
    """Protect input events while allowing fitted states and selections to migrate."""
    if not modes:
        raise ValueError('Detector audit requires at least one mode')
    baseline = next(iter(modes.values()))['events']
    protected = ('gen_sha256', 'weights', 'raw_input_sha256')
    for name, data in modes.items():
        if data['events'].keys() != baseline.keys():
            raise RuntimeError('Detector audit event membership differs: '+name)
        for identity, event in data['events'].items():
            if any(k not in event or k not in baseline[identity] for k in protected):
                raise RuntimeError('Detector audit input fingerprint missing: '+name)
            if any(event[k] != baseline[identity][k] for k in protected):
                raise RuntimeError('Detector audit raw inputs, GEN or native weights differ: '+name)
        if any(identity[:-2] not in data['events'] for identity in data['muons']):
            raise RuntimeError('Detector audit muon references an absent event: '+name)
        for pair in data['pairs']:
            if pair != detector_pair_identity(*pair) or any(leg not in data['muons'] for leg in pair):
                raise RuntimeError('Detector audit pair references absent/invalid source identity: '+name)


def detector_population_union(modes):
    """Account for selection changes; never force a new mode onto old pair membership."""
    validate_detector_audit_inputs(modes)
    reference = next(iter(modes.values()))
    muon_sets = [set(data['muons']) for data in modes.values()]
    pair_sets = [set(data['pairs']) for data in modes.values()]
    result = dict(muon_union=sorted(set.union(*muon_sets)),
                  muon_common_all=sorted(set.intersection(*muon_sets)),
                  pair_union=sorted(set.union(*pair_sets)),
                  pair_common_all=sorted(set.intersection(*pair_sets)), modes={})
    for name, data in modes.items():
        counts = {}
        for kind in ('muons', 'pairs'):
            old, new = set(reference[kind]), set(data[kind])
            transitions = defaultdict(Counter)
            fields = ('charge', 'topology') if kind == 'muons' else ('is_os', 'category')
            for identity in old & new:
                for field in fields:
                    before, after = reference[kind][identity][field], data[kind][identity][field]
                    transitions[field][str(before)+' -> '+str(after)] += 1
            counts[kind] = dict(retained=len(new), common_with_reference=len(old & new),
                                appeared=sorted(new-old), disappeared=sorted(old-new),
                                transitions={k: dict(v) for k, v in transitions.items()})
        result['modes'][name] = counts
    return result


def detector_momentum_decomposition(fitted, surface_truth, target, production_truth):
    """Separate fit error and material return only with a matched detector reference."""
    names = ('detector_error_gev', 'material_return_error_gev', 'total_error_gev',
             'detector_relative', 'detector_contribution_relative', 'material_return_relative',
             'total_relative', 'detector_log_ratio', 'material_return_log_ratio', 'total_log_ratio')
    values = (fitted, surface_truth, target, production_truth)
    if not all(math.isfinite(x) and x > 0. for x in values):
        return dict(valid=False, **{name: math.nan for name in names})
    detector = fitted-surface_truth
    returned = (target-fitted)-(production_truth-surface_truth)
    return dict(valid=True, detector_error_gev=detector, material_return_error_gev=returned,
                total_error_gev=target-production_truth, detector_relative=detector/surface_truth,
                detector_contribution_relative=detector/production_truth,
                material_return_relative=returned/production_truth,
                total_relative=(target-production_truth)/production_truth,
                detector_log_ratio=math.log(fitted/surface_truth),
                material_return_log_ratio=math.log(target/fitted)-math.log(production_truth/surface_truth),
                total_log_ratio=math.log(target/production_truth))


def detector_baseline_statistics(values, parents, bootstrap=200, seed=190019):
    """Pair each mode with the reference separately; a failed control shrinks only itself."""
    if not values:
        raise ValueError('Detector statistics require a reference mode')
    reference = next(iter(values.values()))
    return dict(all_retained={name: paired_parent_statistics({name: value}, parents, bootstrap, seed)
                              for name, value in values.items()},
                paired_vs_reference={name: paired_parent_statistics(
                    {'canonical': reference, name: value}, parents, bootstrap, seed)
                    for name, value in list(values.items())[1:]})


def detector_surface_residuals(fitted, truth):
    """Evaluation in one native detector chart; original-track errors are not used."""
    same_surface = fitted.get('surface') is not None and fitted.get('surface') == truth.get('surface')
    result = dict(measured_valid=False, truth_valid=False, same_surface=same_surface,
                  covariance_valid=False, pull_reference_valid=False, metrics={})
    result['orientation_agrees'] = (fitted.get('local_pz_sign') in (-1, 1)
        and fitted.get('local_pz_sign') == truth.get('local_pz_sign'))
    momentum = np.asarray(fitted.get('momentum', ()), dtype=float)
    parameters = np.asarray(fitted.get('parameters', ()), dtype=float)
    result['measured_valid'] = bool(fitted.get('valid', False) and momentum.shape == (3,)
        and parameters.shape == (5,) and np.isfinite(momentum).all() and np.isfinite(parameters).all()
        and np.linalg.norm(momentum) > 0. and parameters[0] != 0.)
    reference = np.asarray(truth.get('momentum', ()), dtype=float)
    local_truth = np.asarray(truth.get('parameters', ()), dtype=float)
    result['truth_valid'] = bool(same_surface and truth.get('valid', False) and reference.shape == (3,)
        and local_truth.shape == (5,) and np.isfinite(reference).all() and np.isfinite(local_truth).all()
        and np.linalg.norm(reference) > 0. and local_truth[0] != 0.)
    covariance = np.asarray(fitted.get('covariance', ()), dtype=float)
    if covariance.shape == (5, 5) and np.isfinite(covariance).all() and np.allclose(
            covariance, covariance.T, rtol=1.e-10, atol=1.e-12):
        try:
            np.linalg.cholesky(covariance)
            result['covariance_valid'] = True
        except np.linalg.LinAlgError:
            pass
    if not (result['measured_valid'] and result['truth_valid']):
        return result
    result['pull_reference_valid'] = result['covariance_valid'] and result['orientation_agrees']
    p, true_p = np.linalg.norm(momentum), np.linalg.norm(reference)
    difference = parameters-local_truth
    metrics = dict(p_relative=float(p/true_p-1.), qoverp_error=float(difference[0]),
                   qoverp_relative=float(difference[0]/abs(local_truth[0])),
                   angle_error=float(np.arctan2(np.linalg.norm(np.cross(momentum, reference)),
                                               np.dot(momentum, reference))))
    if np.hypot(*momentum[:2]) > 0. and np.hypot(*reference[:2]) > 0.:
        metrics['phi_error'] = math.remainder(math.atan2(momentum[1], momentum[0])-
                                             math.atan2(reference[1], reference[0]), 2.*math.pi)
    for index, label in enumerate(('qoverp', 'slope_x', 'slope_y', 'x', 'y')):
        metrics[label+'_error'] = float(difference[index])
        if result['pull_reference_valid']:
            metrics[label+'_pull'] = float(difference[index]/math.sqrt(covariance[index, index]))
    if result['pull_reference_valid']:
        metrics['mahalanobis2'] = float(difference @ np.linalg.solve(covariance, difference))
    result['metrics'] = metrics
    return result


def detector_pair_surface_closure(first, second, first_truth, second_truth):
    """Pair closure against corresponding detector surfaces, never a nominal mass."""
    checked = [detector_surface_residuals(a, b)
               for a, b in ((first, first_truth), (second, second_truth))]
    if not all(x['measured_valid'] and x['truth_valid'] for x in checked):
        return dict(valid=False)
    momenta = [np.asarray(x['momentum'], dtype=float) for x in (first, second)]
    truth = [np.asarray(x['momentum'], dtype=float) for x in (first_truth, second_truth)]
    def angle(vectors):
        return float(np.arctan2(np.linalg.norm(np.cross(*vectors)), np.dot(*vectors)))
    mass, true_mass = float(stable_pair_mass(*momenta)), float(stable_pair_mass(*truth))
    return dict(valid=True, mass_error_gev=mass-true_mass, mass_relative=mass/true_mass-1.,
                opening_error=angle(momenta)-angle(truth))


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


def read_detector_audit(path, context, raw_input_sha256, event_sources=None):
    """Selected-row detector snapshots plus all-candidate counters, without fallback."""
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie() or source.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError('Unreadable detector audit: '+str(path))
    counters = ('inputCandidates', 'attempted', 'stateAvailable', 'fitAccepted', 'targetAttempted',
                'targetValid', 'canonicalAccepted', 'erased', 'retained')
    audit_fields = ('Available', 'FitAccepted', 'TargetAttempted', 'TargetStatus', 'CanonicalAccepted',
                    'Variant', 'Iteration', 'SnapshotKind', 'NativeStatus', 'FirstStatus', 'SecondStatus',
                    'SurfaceId', 'LocalPzSign', 'Charge', 'X', 'Y', 'Z', 'Px', 'Py', 'Pz', 'Chi2', 'Ndof',
                    'UsedHits', 'InputHashLo', 'InputHashHi', 'UsedHashLo', 'UsedHashHi')
    truth_fields = ('Status', 'SimTrackId', 'Charge', 'LocalPzSign', 'X', 'Y', 'Z', 'Px', 'Py', 'Pz',
                    'LowerX', 'LowerY', 'LowerZ', 'LowerPx', 'LowerPy', 'LowerPz',
                    'UpperX', 'UpperY', 'UpperZ', 'UpperPx', 'UpperPy', 'UpperPz',
                    'LowerLocalZ', 'UpperLocalZ', 'Span', 'Weight', 'LowerKind', 'UpperKind')
    raw_fields = ('trackPt', 'preRefitPt', 'preRefitPz', 'nValidHits', 'nValidMuonHits',
                  'nCSCHits', 'nDTHits', 'nRPCHits', 'nGEMHits', 'nME0Hits')
    muon_fields = ('recoAlgorithm', 'sourceIndex', 'charge', 'topology', 'px', 'py', 'pz',
                   'hitGenPartIdx', 'hitSimTrackId', 'genPartIdx')+raw_fields
    pair_fields = ('mass', 'pt', 'pz', 'vx', 'vy', 'vz', 'muonIdx1', 'muonIdx2', 'isOS',
                   'topologyMin', 'topologyMax')
    lower_cov = [(i, j) for i in range(5) for j in range(i+1)]
    audit_names = list(audit_fields)+['Par'+str(i) for i in range(5)]
    audit_names += ['Cov'+str(i)+str(j) for i, j in lower_cov]
    truth_names = list(truth_fields)+['Par'+str(i) for i in range(5)]
    try:
        tree = source.Get('Events')
        if not tree:
            raise RuntimeError('Missing Events in detector audit: '+str(path))
        branches = {b.GetName() for b in tree.GetListOfBranches()}
        required = {'run', 'luminosityBlock', 'event', 'genWeight', 'nShiftMuon', 'nShiftDimuonVertex', 'nGenPart'}
        required.update('ShiftDetectorFitAudit'+name for name in counters)
        required.update('ShiftMuon_'+name for name in muon_fields)
        required.update('ShiftMuon_detectorAudit'+name for name in audit_names)
        required.update('ShiftMuon_detectorAuditTruth'+name for name in truth_names)
        required.update('ShiftDimuonVertex_'+name for name in pair_fields)
        gen_fields = ('pdgId', 'genPartIdxMother', 'pt', 'phi', 'pz', 'vx', 'vy', 'vz')
        required.update('GenPart_'+name for name in gen_fields)
        if required-branches:
            raise RuntimeError('Detector audit branches missing: '+str(sorted(required-branches)))
        audit_ints = set(audit_fields)-{'SurfaceId', 'X', 'Y', 'Z', 'Px', 'Py', 'Pz', 'Chi2', 'Ndof',
                                      'InputHashLo', 'InputHashHi', 'UsedHashLo', 'UsedHashHi'}
        truth_ints = {'Status', 'SimTrackId', 'Charge', 'LocalPzSign', 'LowerKind', 'UpperKind'}
        types = {'ShiftDetectorFitAudit'+name: 'Int_t' for name in counters}
        for name in audit_names:
            types['ShiftMuon_detectorAudit'+name] = ('Int_t' if name in audit_ints else
                'UInt_t' if name == 'SurfaceId' or 'Hash' in name else 'Double_t')
        for name in truth_names:
            types['ShiftMuon_detectorAuditTruth'+name] = 'Int_t' if name in truth_ints else 'Double_t'
        for name, expected in types.items():
            leaf = tree.GetBranch(name).GetLeaf(name)
            if not leaf or leaf.GetTypeName() != expected:
                raise RuntimeError('Detector audit branch type differs: '+name+' expected '+expected)
        weight_fields = sorted(n for n in branches if n == 'genWeight' or
                               (n.startswith('shift') and ('Weight' in n or 'Probability' in n)))
        tree.SetBranchStatus('*', False)
        for name in required | set(weight_fields) | {n for n in branches if n.startswith('GenPart_')}:
            tree.SetBranchStatus(name, True)
        events, muons, pairs = {}, {}, {}
        labels = {0: 'Near', 1: 'Near-Barrel', 2: 'Both', 3: 'Far', 4: 'Other'}
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            native_id = tuple(int(getattr(tree, n)) for n in ('run', 'luminosityBlock', 'event'))
            event_token = ':'.join(map(str, native_id))
            if event_sources is not None and event_token not in event_sources:
                raise RuntimeError('Detector event lacks physical-source metadata: '+event_token)
            event_info = (event_sources or {}).get(event_token, {})
            if event_sources is not None and any(not event_info.get(k) for k in
                    ('physical_parent_context', 'raw_input_sha256', 'sample', 'process')):
                raise RuntimeError('Detector physical-source metadata incomplete: '+event_token)
            physical_context = event_info.get('physical_parent_context', context)
            identity = (physical_context,)+native_id
            if identity in events:
                raise RuntimeError('Duplicate detector audit event: '+str(identity))
            full_gen = {n: list(getattr(tree, n)) for n in sorted(branches) if n.startswith('GenPart_')}
            gen = {n: list(getattr(tree, 'GenPart_'+n)) for n in gen_fields}
            weights = {n: float(getattr(tree, n)) for n in weight_fields}
            if not all(math.isfinite(x) for x in weights.values()):
                raise RuntimeError('Nonfinite native weight in detector audit: '+str(identity))
            event_counts = {n: int(getattr(tree, 'ShiftDetectorFitAudit'+n)) for n in counters}
            if event_counts['retained'] != int(tree.nShiftMuon):
                raise RuntimeError('Detector audit retained counter disagrees with ShiftMuon')
            events[identity] = dict(gen_sha256=hashlib.sha256(repr(full_gen).encode()).hexdigest(),
                weights=weights, raw_input_sha256=event_info.get('raw_input_sha256', raw_input_sha256),
                counters=event_counts, sample=event_info.get('sample'), process=event_info.get('process'))
            row_keys = []
            for index in range(int(tree.nShiftMuon)):
                row = {n: float(getattr(tree, 'ShiftMuon_'+n)[index]) for n in muon_fields}
                key = identity+(int(row['recoAlgorithm']), int(row['sourceIndex']))
                if key in muons:
                    raise RuntimeError('Duplicate selected detector raw source: '+str(key))
                row_keys.append(key)
                measured = {n: float(getattr(tree, 'ShiftMuon_detectorAudit'+n)[index]) for n in audit_names}
                reference = {n: float(getattr(tree, 'ShiftMuon_detectorAuditTruth'+n)[index]) for n in truth_names}
                covariance = np.zeros((5, 5))
                for i, j in lower_cov:
                    covariance[i, j] = covariance[j, i] = measured['Cov'+str(i)+str(j)]
                surface = (int(measured['SurfaceId']), 'native')
                fitted = dict(valid=measured['Available'] == 1 and surface[0] != 0, surface=surface,
                    local_pz_sign=int(measured['LocalPzSign']),
                    momentum=[measured[n] for n in ('Px', 'Py', 'Pz')],
                    parameters=[measured['Par'+str(i)] for i in range(5)], covariance=covariance)
                bracket = all(math.isfinite(reference[n]) for n in ('LowerLocalZ', 'UpperLocalZ', 'Weight', 'Span'))
                bracket = bracket and reference['LowerLocalZ'] <= 0. <= reference['UpperLocalZ']
                bracket = bracket and 0. <= reference['Weight'] <= 1. and reference['Span'] >= 0.
                matched = int(row['hitGenPartIdx'])
                gen_valid = 0 <= matched < len(gen['pdgId']) and abs(gen['pdgId'][matched]) == 13
                truth_valid = reference['Status'] == 1 and bracket and reference['SimTrackId'] >= 0
                truth_valid = truth_valid and reference['SimTrackId'] == row['hitSimTrackId']
                truth = dict(valid=truth_valid, surface=surface,
                    local_pz_sign=int(reference['LocalPzSign']),
                    momentum=[reference[n] for n in ('Px', 'Py', 'Pz')],
                    parameters=[reference['Par'+str(i)] for i in range(5)])
                evaluated = detector_surface_residuals(fitted, truth)
                production = {}
                if gen_valid:
                    production = {n: gen[n][matched] for n in gen_fields}
                    production['momentum'] = momentum_vector(np.array([production['pt']]),
                        np.array([production['phi']]), np.array([production['pz']]))[0]
                metrics = dict(evaluated['metrics'])
                returns_from_fit = (measured['SnapshotKind'] == 1 and measured['FitAccepted'] == 1 and measured['CanonicalAccepted'] == 1
                                    and measured['TargetAttempted'] == 1 and measured['TargetStatus'] == 1)
                if evaluated['measured_valid'] and evaluated['truth_valid'] and returns_from_fit and production:
                    decomposition = detector_momentum_decomposition(np.linalg.norm(fitted['momentum']),
                        np.linalg.norm(truth['momentum']), np.linalg.norm([row[n] for n in ('px', 'py', 'pz')]),
                        np.linalg.norm(production['momentum']))
                    metrics.update({k: v for k, v in decomposition.items() if k != 'valid'})
                muons[key] = dict(charge=int(row['charge']), topology=int(row['topology']),
                    raw_signature=tuple(row[n] for n in raw_fields), fitted=fitted, truth=truth, audit=measured,
                    truth_audit=reference, evaluation=evaluated, metrics=metrics, production=production,
                    hit_gen_index=matched, canonical_momentum=[row[n] for n in ('px', 'py', 'pz')],
                    original_fallback=not returns_from_fit,
                    legacy_hit_association_agrees=int(row['genPartIdx']) == matched if gen_valid else None)
            for index in range(int(tree.nShiftDimuonVertex)):
                row = {n: float(getattr(tree, 'ShiftDimuonVertex_'+n)[index]) for n in pair_fields}
                a, b = int(row['muonIdx1']), int(row['muonIdx2'])
                if not (0 <= a < len(row_keys) and 0 <= b < len(row_keys)):
                    raise RuntimeError('Detector pair references invalid selected muon index')
                key = detector_pair_identity(row_keys[a], row_keys[b])
                if key in pairs:
                    raise RuntimeError('Duplicate detector source-pair identity: '+str(key))
                first, second = (muons[row_keys[i]] for i in (a, b))
                metrics = {}
                closure = detector_pair_surface_closure(first['fitted'], second['fitted'], first['truth'], second['truth'])
                if closure['valid']:
                    metrics.update({'detector_pair_'+k: v for k, v in closure.items() if k != 'valid'})
                distinct_truth = first['hit_gen_index'] != second['hit_gen_index']
                if first['production'] and second['production'] and distinct_truth:
                    vectors = [m['production']['momentum'] for m in (first, second)]
                    true_mass = float(stable_pair_mass(*vectors))
                    metrics['production_mass_relative'] = row['mass']/true_mass-1.
                    vertices = [np.array([m['production']['v'+axis] for axis in 'xyz']) for m in (first, second)]
                    if np.max(np.abs(vertices[0]-vertices[1])) < 1.e-4:
                        for i, axis in enumerate('xyz'):
                            metrics['production_vertex_'+axis] = row['v'+axis]-.5*(vertices[0][i]+vertices[1][i])
                category = '-'.join(labels.get(int(row[n]), str(int(row[n]))) for n in ('topologyMin', 'topologyMax'))
                pairs[key] = dict(is_os=bool(row['isOS']), category=category, metrics=metrics, canonical=row)
        return dict(events=events, muons=muons, pairs=pairs), dict(path=str(path), entries=int(tree.GetEntries()),
            bytes=int(source.GetSize()), root_uuid=source.GetUUID().AsString(), branches=sorted(required))
    finally:
        source.Close()


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


def upstream_muon_residuals(momentum, charge, production):
    """Evaluate a fixed selected state; wrong charge/direction never remove it."""
    measured = np.asarray(momentum, dtype=float)
    truth = np.asarray(production.get('momentum', ()), dtype=float)
    pdg = production.get('pdgId', 0)
    if abs(pdg) != 13:
        return dict(metrics={}, wrong_charge=None, direction_disagrees=None)
    expected_charge = -1 if pdg == 13 else 1
    wrong_charge = int(charge) != expected_charge
    if (measured.shape != (3,) or truth.shape != (3,) or not np.isfinite(measured).all()
            or not np.isfinite(truth).all()):
        return dict(metrics={}, wrong_charge=wrong_charge, direction_disagrees=None)
    p, reference = float(np.linalg.norm(measured)), float(np.linalg.norm(truth))
    metrics = {}
    for name, value, target in (('p', p, reference), ('pt', np.hypot(*measured[:2]), np.hypot(*truth[:2])),
                                ('pz', measured[2], truth[2])):
        if target != 0.:
            metrics['production_'+name+'_relative'] = float(value/target-1.)
    direction = None
    if p > 0. and reference > 0.:
        metrics['production_qoverp_relative'] = float((charge/p-expected_charge/reference)/(1./reference))
        metrics['production_angle_rad'] = float(np.arctan2(np.linalg.norm(np.cross(measured, truth)),
                                                          np.dot(measured, truth)))
        direction = bool(np.dot(measured, truth) < 0.)
    return dict(metrics=metrics, wrong_charge=wrong_charge,
                direction_disagrees=direction)


def upstream_scalar_statistics(values, metric):
    """Use existing estimators with explicit relative, centimetre or radian tails."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    result = residual_statistics(finite)
    if result is None:
        return None
    result.pop('tail100'); result.pop('tail50')
    thresholds = ((('tail_abs_gt_50pct', .5), ('tail_abs_gt_100pct', 1.)) if metric.endswith('_relative')
                  else (('tail_abs_gt_1m', 100.), ('tail_abs_gt_5m', 500.)) if metric.endswith('_cm')
                  else (('tail_abs_gt_10mrad', .01), ('tail_abs_gt_100mrad', .1)) if metric.endswith('_rad')
                  else ())
    for name, threshold in thresholds:
        result[name] = float(np.mean(np.abs(finite) > threshold))
    return result


def upstream_weight_accounting(weights):
    """Original-event weight concentration only; never turn signed weights into a CDF."""
    values = [float(weight) for weight in weights]
    if not all(math.isfinite(weight) for weight in values):
        raise RuntimeError('Nonfinite upstream original weight')
    total = math.fsum(values)
    squared = math.fsum(weight*weight for weight in values)
    absolute = math.fsum(abs(weight) for weight in values)
    nonzero = [abs(weight) for weight in values if weight != 0.]
    return dict(originals=len(values), positive=sum(weight > 0. for weight in values),
        negative=sum(weight < 0. for weight in values), zero=sum(weight == 0. for weight in values),
        minimum=min(values) if values else None, maximum=max(values) if values else None,
        nonzero_absolute_range=max(nonzero)/min(nonzero) if nonzero else None,
        sumw=total, sumw2=squared, sum_absw=absolute,
        effective_originals_signed=total*total/squared if squared else None,
        effective_originals_absolute=absolute*absolute/squared if squared else None,
        positive_weight_CDF_allowed=bool(values and min(values) >= 0. and total > 0.),
        interpretation='Weight concentration of original events; no rate, weighted width or adoption estimate.')


def upstream_event_statistics(modes, kind, metric, bootstrap=200, seed=190019):
    """Same original-event draws, each arm's own rows, stratified by source file.

    Zero-response originals remain sampling units. No row intersection, missing
    state replacement or matching by sourceIndex enters this comparison.
    """
    if not modes:
        raise ValueError('Upstream statistics need a reference arm')
    baseline = next(iter(modes.values()))['events']
    for name, data in modes.items():
        if set(data['events']) != set(baseline):
            raise RuntimeError('Upstream statistics original event membership differs: '+name)
    parents = sorted(baseline)
    strata = defaultdict(list)
    for parent in parents:
        event = baseline[parent]
        strata[event.get('bootstrap_stratum', event['source_context'])].append(parent)
    values = {}
    result = dict(originals=len(parents), source_strata={name: len(rows) for name, rows in strata.items()},
                  sampling_unit='original event; shared draws within each source/file', arms={},
                  paired_delta_vs_reference={})
    for name, data in modes.items():
        grouped = {parent: [] for parent in parents}
        total_rows = Counter()
        for row in data[kind]:
            parent = tuple(row['event'])
            if parent not in grouped:
                raise RuntimeError('Upstream selected row references an absent original')
            total_rows[parent] += 1
            value = row['metrics'].get(metric, math.nan)
            if math.isfinite(value):
                grouped[parent].append(float(value))
        values[name] = grouped
        finite = [value for parent in parents for value in grouped[parent]]
        contributing = sum(bool(grouped[parent]) for parent in parents)
        resampleable = sum(bool(grouped[parent]) for rows in strata.values() if len(rows) > 1 for parent in rows)
        result['arms'][name] = dict(selected_rows=sum(total_rows.values()), finite_rows=len(finite),
            omitted_or_undefined_rows=sum(total_rows.values())-len(finite), contributing_originals=contributing,
            resampleable_contributing_originals=resampleable,
            zero_metric_originals=len(parents)-contributing,
            quantile_resolved=bool(len(finite) >= 20 and contributing >= 20),
            estimate=upstream_scalar_statistics(finite, metric))
        if all('native_EDM_double_weight' in event for event in data['events'].values()):
            result['arms'][name]['contributing_original_weights_by_source'] = {
                source: upstream_weight_accounting([data['events'][parent]['native_EDM_double_weight']
                    for parent in rows if grouped[parent]]) for source, rows in strata.items()}
    generator = np.random.default_rng(seed)
    draws = {name: [] for name in modes}
    for _ in range(bootstrap):
        chosen = []
        for group in strata.values():
            chosen.extend(group[i] for i in generator.integers(0, len(group), size=len(group)))
        for name in modes:
            sample = [value for parent in chosen for value in values[name][parent]]
            draws[name].append(upstream_scalar_statistics(sample, metric))
    reference = next(iter(modes))
    for name, arm in result['arms'].items():
        valid = [row for row in draws[name] if row is not None]
        arm['bootstrap_nonempty_draws'] = len(valid)
        arm['bootstrap_empty_draws'] = bootstrap-len(valid)
        if arm['resampleable_contributing_originals'] < 2:
            arm['uncertainty_status'] = 'undefined: fewer than two contributing originals in resampleable source strata'
        elif len(valid) < 2:
            arm['uncertainty_status'] = 'undefined: fewer than two nonempty bootstrap draws'
        else:
            arm['uncertainty_status'] = 'diagnostic original-event bootstrap; sparse widths remain unresolved'
            arm['interval68'] = {}
            for statistic, estimate in arm['estimate'].items():
                low, high = np.quantile([row[statistic] for row in valid], [.158655, .841345])
                arm['interval68'][statistic] = dict(low=float(low), high=float(high),
                    error_down=float(max(0., estimate-low)), error_up=float(max(0., high-estimate)))
        if name == reference:
            continue
        comparison = dict(contributing_originals_reference=result['arms'][reference]['contributing_originals'],
                          contributing_originals_arm=arm['contributing_originals'])
        paired = [(a, b) for a, b in zip(draws[reference], draws[name]) if a is not None and b is not None]
        comparison['jointly_nonempty_draws'] = len(paired)
        comparison['omitted_empty_draws'] = bootstrap-len(paired)
        if result['arms'][reference]['estimate'] is not None and arm['estimate'] is not None:
            comparison['estimate'] = {key: value-result['arms'][reference]['estimate'][key]
                                      for key, value in arm['estimate'].items()}
        if min(result['arms'][reference]['resampleable_contributing_originals'],
               arm['resampleable_contributing_originals']) >= 2 and len(paired) >= 2:
            comparison['interval68'] = {}
            for statistic, value in comparison['estimate'].items():
                low, high = np.quantile([b[statistic]-a[statistic] for a, b in paired], [.158655, .841345])
                comparison['interval68'][statistic] = dict(value=float(value), low=float(low), high=float(high))
        else:
            comparison['uncertainty_status'] = 'undefined: fewer than two contributing originals in an arm'
        result['paired_delta_vs_reference'][name] = comparison
    return result


def upstream_report_values(value):
    """Retain failed/nonfinite case states as JSON null, never invented zeroes."""
    if isinstance(value, dict):
        return {key: upstream_report_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [upstream_report_values(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def upstream_pair_reference_status(first, second):
    """Two saved hit-GEN indices alone must not turn a split SimTrack into a pair."""
    if not first['matched_production'] or not second['matched_production']:
        return 'missing_or_nonmuon_hit_reference'
    ids = [row['hit_simtrack_id'] for row in (first, second)]
    if min(ids) <= 0:
        return 'missing_hit_simtrack_identity'
    if ids[0] == ids[1]:
        return 'same_simtrack_split'
    if first['hit_gen_index'] == second['hit_gen_index']:
        return 'same_hit_associated_GEN_muon'
    return 'distinct_hit_associated_muons'


def read_upstream_reconstruction(path, context):
    """Reuse native pair/GEN extraction and retain every arm's selected muons."""
    events, native_pairs, stamp = read_native_reconstruction(path)
    source = ROOT.TFile.Open(str(path))
    try:
        tree = source.Get('Events')
        branches = {b.GetName() for b in tree.GetListOfBranches()}
        fields = ('recoAlgorithm', 'sourceIndex', 'charge', 'topology', 'pt', 'pz', 'phi', 'hitGenPartIdx', 'hitSimTrackId')
        required = {'run', 'luminosityBlock', 'event', 'nShiftMuon', 'nGenPart', 'genWeight'}
        required.update('ShiftMuon_'+name for name in fields)
        gen_fields = ('pdgId', 'pt', 'pz', 'phi', 'vx', 'vy', 'vz')
        required.update('GenPart_'+name for name in gen_fields)
        if required-branches:
            raise RuntimeError('Upstream selected-muon branches missing: '+str(sorted(required-branches)))
        weight_names = sorted(name for name in branches if name == 'genWeight' or
                              (name.startswith('shift') and ('Weight' in name or 'Probability' in name)))
        weight_types = {name: tree.GetBranch(name).GetLeaf(name).GetTypeName() for name in weight_names}
        if weight_types.get('genWeight') != 'Float_t':
            raise RuntimeError('Upstream genWeight must retain standard Float_t serialization')
        tree.SetBranchStatus('*', False)
        for name in required | set(weight_names):
            tree.SetBranchStatus(name, True)
        output = dict(events={}, muons=[], pairs=[])
        labels = {0: 'Near', 1: 'Near-Barrel', 2: 'Both', 3: 'Far', 4: 'Other'}
        for entry in range(int(tree.GetEntries())):
            tree.GetEntry(entry)
            identity = tuple(int(getattr(tree, name)) for name in ('run', 'luminosityBlock', 'event'))
            key = (context,)+identity
            event = dict(events[identity], source_context=context,
                         weights={name: float(getattr(tree, name)) for name in weight_names}, weight_types=weight_types)
            if not all(math.isfinite(value) for value in event['weights'].values()):
                raise RuntimeError('Nonfinite upstream native/sampling weight')
            output['events'][key] = event
            gen = {name: list(getattr(tree, 'GenPart_'+name)) for name in gen_fields}
            selected = []
            for index in range(int(tree.nShiftMuon)):
                measured = {name: float(getattr(tree, 'ShiftMuon_'+name)[index]) for name in fields}
                vector = momentum_vector(np.array([measured['pt']]), np.array([measured['phi']]),
                                         np.array([measured['pz']]))[0]
                matched = int(measured['hitGenPartIdx'])
                production = {}
                if 0 <= matched < int(tree.nGenPart) and abs(gen['pdgId'][matched]) == 13:
                    production = {name: gen[name][matched] for name in gen_fields}
                    production['momentum'] = momentum_vector(np.array([production['pt']]),
                        np.array([production['phi']]), np.array([production['pz']]))[0].tolist()
                evaluated = upstream_muon_residuals(vector, int(measured['charge']), production)
                row = dict(event=key, row_index=index, algorithm=int(measured['recoAlgorithm']),
                    source_index=int(measured['sourceIndex']), charge=int(measured['charge']),
                    topology=int(measured['topology']), category=labels.get(int(measured['topology']), 'Other'),
                    canonical_momentum=vector.tolist(), production=production, hit_gen_index=matched,
                    hit_simtrack_id=int(measured['hitSimTrackId']),
                    matched_production=bool(production), **evaluated)
                selected.append(row); output['muons'].append(row)
            for native_key, pair in native_pairs.items():
                if native_key[:3] != identity:
                    continue
                canonical, truth = pair['canonical'], pair['truth']
                legs = [int(pair['measured'][name]) for name in ('muonIdx1', 'muonIdx2')]
                reference_status = upstream_pair_reference_status(*(selected[i] for i in legs))
                truth_valid = bool(truth) and reference_status == 'distinct_hit_associated_muons'
                metrics = {}
                if truth_valid:
                    for name in ('mass', 'pt', 'pz', 'p'):
                        value = np.hypot(canonical['pt'], canonical['pz']) if name == 'p' else canonical[name]
                        if math.isfinite(value) and math.isfinite(truth[name]) and truth[name] != 0.:
                            metrics['production_'+name+'_relative'] = float(value/truth[name]-1.)
                    vectors = [np.asarray(selected[i]['canonical_momentum']) for i in legs]
                    if (all(np.isfinite(v).all() and np.linalg.norm(v) > 0. for v in vectors)
                            and math.isfinite(truth['opening_angle'])):
                        metrics['production_angle_rad'] = float(np.arctan2(np.linalg.norm(np.cross(*vectors)),
                            np.dot(*vectors))-truth['opening_angle'])
                    if truth['shared_vertex']:
                        for axis in 'xyz':
                            if math.isfinite(canonical['v'+axis]) and math.isfinite(truth['v'+axis]):
                                metrics['production_vertex_'+axis+'_cm'] = float(canonical['v'+axis]-truth['v'+axis])
                output['pairs'].append(dict(event=key, row_index=native_key[3], legs=legs,
                    category='-'.join(labels.get(value, 'Other') for value in pair['category']),
                    is_os=bool(pair['measured']['isOS']), canonical=canonical, metrics=metrics,
                    distinct_hit_associated_muons=truth_valid, reference_status=reference_status,
                    shared_production_vertex=bool(truth_valid and truth.get('shared_vertex', False))))
        stamp['upstream_selected_branches'] = sorted(required | set(weight_names))
        return output, stamp
    finally:
        source.Close()


def validate_upstream_populations(modes):
    """Validate shared originals/GEN/accounting; fitted track fields may change."""
    baseline = next(iter(modes.values()))['events']
    for name, data in modes.items():
        if data['events'].keys() != baseline.keys():
            raise RuntimeError('Upstream original membership differs: '+name)
        for key, event in data['events'].items():
            for field in ('gen_sha256', 'weights', 'weight_types', 'raw_input_sha256', 'source_context'):
                if field not in event or event[field] != baseline[key].get(field):
                    raise RuntimeError('Upstream original GEN/source/native weights differ: '+name+' '+field)
        for kind in ('muons', 'pairs'):
            seen = set()
            for row in data[kind]:
                identity = (tuple(row['event']), row['row_index'])
                if identity[0] not in baseline or identity in seen:
                    raise RuntimeError('Duplicate or absent upstream original/selected row')
                seen.add(identity)
        muons = {(tuple(row['event']), row['row_index']) for row in data['muons']}
        for pair in data['pairs']:
            if (len(pair['legs']) != 2 or pair['legs'][0] == pair['legs'][1]
                    or any((tuple(pair['event']), index) not in muons for index in pair['legs'])):
                raise RuntimeError('Upstream chosen pair references absent or repeated muon legs')


def upstream_population_report(modes):
    """Count event-level migrations without claiming track or pair identity."""
    validate_upstream_populations(modes)
    reference = next(iter(modes))
    result = dict(arms={}, event_cases=[], transitions_vs_reference={})
    categories = sorted({row['category'] for data in modes.values() for row in data['pairs']})
    for name, data in modes.items():
        weight_names = sorted({field for event in data['events'].values() for field in event['weights']})
        result['arms'][name] = dict(originals=len(data['events']), selected_muons=len(data['muons']),
            selected_pairs=len(data['pairs']), matched_production_muons=sum(r['matched_production'] for r in data['muons']),
            missing_production_muons=sum(not r['matched_production'] for r in data['muons']),
            wrong_charge=sum(r['wrong_charge'] is True for r in data['muons']),
            wrong_direction=sum(r['direction_disagrees'] is True for r in data['muons']),
            distinct_hit_associated_pairs=sum(r['distinct_hit_associated_muons'] for r in data['pairs']),
            unknown_pair_reference=sum(not r['distinct_hit_associated_muons'] for r in data['pairs']),
            shared_vertex_pairs=sum(r['shared_production_vertex'] for r in data['pairs']),
            muon_charge_counts=dict(Counter(str(r['charge']) for r in data['muons'])),
            muon_source_counts=dict(Counter(str(r['algorithm']) for r in data['muons'])),
            muon_category_counts=dict(Counter(r['category'] for r in data['muons'])),
            pair_category_counts=dict(Counter(r['category'] for r in data['pairs'])),
            pair_reference_counts=dict(Counter(r['reference_status'] for r in data['pairs'])),
            undefined_direction=sum(r['direction_disagrees'] is None for r in data['muons']),
            zero_selected_muon_originals=sum(not any(tuple(row['event']) == key for row in data['muons'])
                                              for key in data['events']),
            zero_selected_pair_originals=sum(not any(tuple(row['event']) == key for row in data['pairs'])
                                              for key in data['events']),
            native_EDM_double_sumw=math.fsum(event['native_EDM_double_weight'] for event in data['events'].values()),
            persisted_weights={field: dict(events_present=sum(field in event['weights'] for event in data['events'].values()),
                sumw=math.fsum(event['weights'].get(field, 0.) for event in data['events'].values()),
                sumw2=math.fsum(event['weights'].get(field, 0.)**2 for event in data['events'].values()))
                for field in weight_names})
        source_weights = defaultdict(list)
        for event in data['events'].values():
            source_weights[event.get('bootstrap_stratum', event['source_context'])].append(event['native_EDM_double_weight'])
        result['arms'][name]['original_native_weights_by_source'] = {
            source: upstream_weight_accounting(weights) for source, weights in source_weights.items()}
    for event in sorted(modes[reference]['events']):
        case = dict(event=event, arms={})
        for name, data in modes.items():
            muons = [row for row in data['muons'] if tuple(row['event']) == event]
            pairs = [row for row in data['pairs'] if tuple(row['event']) == event]
            case['arms'][name] = dict(muons=muons, pairs=pairs, weights=data['events'][event]['weights'],
                                     native_EDM_double_weight=data['events'][event]['native_EDM_double_weight'])
        result['event_cases'].append(case)
    for name in list(modes)[1:]:
        transitions = defaultdict(Counter)
        for case in result['event_cases']:
            for kind in ('muons', 'pairs'):
                a, b = (len(case['arms'][arm][kind]) for arm in (reference, name))
                transitions[kind][str(a)+' -> '+str(b)] += 1
            for category in categories:
                a, b = (sum(r['category'] == category for r in case['arms'][arm]['pairs']) for arm in (reference, name))
                transitions['category/'+category][str(a)+' -> '+str(b)] += 1
        result['transitions_vs_reference'][name] = {key: dict(value) for key, value in transitions.items()}
    result['identity_qualification'] = 'Event/count migrations only. Equal sourceIndex or row number never proves a common track/pair; native detector-lineage evidence is separate.'
    return result


def run_upstream_replay(manifest_path, release_path, bootstrap=200, seed=190019):
    """Explicit upstream route; existing same-raw-track audit gates stay intact."""
    payload = Path(manifest_path).read_bytes()
    manifest = json.loads(payload)
    release = json.loads(Path(release_path).read_text())
    source_sha = native_file_checksum(__file__)
    if (release.get('released_for_analysis') is not True
            or release.get('manifest_sha256') != hashlib.sha256(payload).hexdigest()
            or release.get('analysis_source_sha256') != source_sha):
        raise RuntimeError('Upstream outcomes are not released for this manifest/source')
    cohort_ref, gate_ref = manifest['cohort'], manifest['native_gate']
    for record in (cohort_ref, gate_ref):
        if native_file_checksum(record['path']) != record['sha256']:
            raise RuntimeError('Upstream frozen cohort/native gate hash mismatch')
    if release.get('cohort_sha256') != cohort_ref['sha256']:
        raise RuntimeError('Upstream release/cohort differs')
    cohort = json.loads(Path(cohort_ref['path']).read_text())
    native_gate = json.loads(Path(gate_ref['path']).read_text())
    if native_gate.get('passed') is not True or manifest['mode_order'] != ['etaOff', 'etaOn']:
        raise RuntimeError('Upstream native gate/order is not qualified')
    original_equivalence = native_gate.get('original_equivalence')
    qualification = None
    if original_equivalence is not True:
        record = manifest.get('comparison_qualification')
        if (original_equivalence is not False or not record or native_gate.get('comparison_qualification') != record
                or release.get('comparison_qualification_sha256') != record['sha256']
                or native_file_checksum(record['path']) != record['sha256']):
            raise RuntimeError('Upstream historical baseline mismatch lacks exact explicit engineering qualification')
        qualification = json.loads(Path(record['path']).read_text())
        reference_directory = Path(manifest['reference_directory'])
        def reference_path(reference):
            path = Path(reference['path'])
            return path if path.is_absolute() else reference_directory/path
        same_cohort = any(reference['sha256'] == cohort_ref['sha256'] and
                          reference_path(reference).resolve() == Path(cohort_ref['path']).resolve()
                          for reference in qualification.get('refs', []))
        if (qualification.get('schema') != 'shift-eta45-engineering-baseline-qualification-v1'
                or qualification.get('released_for_downstream') is not True
                or qualification.get('passed_for_scope') is not True
                or qualification.get('original_full_raw_equality_passed') is not False
                or not same_cohort):
            raise RuntimeError('Upstream engineering qualification/cohort is not explicit')
        for reference in qualification['refs']:
            if native_file_checksum(reference_path(reference)) != reference['sha256']:
                raise RuntimeError('Upstream engineering qualification historical reference changed')
        historical = json.loads(reference_path(qualification['refs'][0]).read_text())
        if historical.get('passed') is not False:
            raise RuntimeError('Upstream historical failed raw gate was not preserved')
    fixed = {row['source']: row for row in cohort['sources']}
    if (len(fixed) != len(cohort['sources']) or len(manifest['sources']) != len(fixed)
            or {row['source_name'] for row in manifest['sources']} != set(fixed)):
        raise RuntimeError('Upstream source membership differs from frozen cohort')
    bindings = {(row['source'], row['mode'], row['path'], row['sha256'], row['bytes'])
                for row in native_gate['outputs']}
    # Hash every declared output before any outcome reader is called.
    for source in manifest['sources']:
        if set(source['modes']) != set(manifest['mode_order']):
            raise RuntimeError('Upstream source lacks an arm')
        for name, record in source['modes'].items():
            if ((source['source_name'], name, record['path'], record['sha256'], record['bytes']) not in bindings
                    or native_file_checksum(record['path']) != record['sha256']):
                raise RuntimeError('Upstream output lacks native qualification/hash binding')
    modes = {name: dict(events={}, muons=[], pairs=[]) for name in manifest['mode_order']}
    provenance = []
    physical_originals = set()
    expected_sources = {}
    for source in manifest['sources']:
        declared = fixed[source['source_name']]
        expected = [tuple(row['event_id']) for row in declared['selected_for_replay']]
        weights = [row['native_EDM_double_weight'] for row in declared['selected_for_replay']]
        expected_process = ('DarkPhoton' if declared['sample'].startswith('darkphoton_') else
                            'JPsi' if declared['sample'].startswith('jpsi_') else None)
        if (len(set(expected)) != len(expected) or any(not math.isfinite(weight) for weight in weights)
                or source['expected_ids'] != [list(key) for key in expected]
                or source['expected_native_EDM_double_weights'] != weights
                or source['raw_input_sha256'] != declared['original_AOD_reference']['sha256']
                or source['sample'] != declared['sample'] or source['process'] != expected_process
                or expected_process is None or source['kernel'] != declared['kernel']
                or declared['full_GEN_accounting']['parent_gen_sha256'] != declared['parent_GEN_sha256']
                or source['reference'] != 'saved_hit_associated_GEN_engineering_only'):
            raise RuntimeError('Upstream source/weight/reference differs from frozen inputs')
        for identity in expected:
            physical = (declared['parent_GEN_sha256'],)+identity
            if physical in physical_originals:
                raise RuntimeError('Upstream manifest duplicates a physical GEN original across source files')
            physical_originals.add(physical)
        expected_sources[source['source_name']] = (expected, weights)
    for source in manifest['sources']:
        expected, weights = expected_sources[source['source_name']]
        for name, record in source['modes'].items():
            data, stamp = read_upstream_reconstruction(record['path'], source['source_context'])
            ids = [key[1:] for key in data['events']]
            if ids != expected or stamp['bytes'] != record['bytes']:
                raise RuntimeError('Upstream output original order/count/bytes differs')
            for key, weight in zip(data['events'], weights):
                event = data['events'][key]
                if event['weights']['genWeight'] != float(np.float32(weight)):
                    raise RuntimeError('Upstream native Float32 weight differs from original EDM double')
                event.update(raw_input_sha256=source['raw_input_sha256'], native_EDM_double_weight=weight,
                             sample=source['sample'], process=source['process'],
                             bootstrap_stratum=source['source_name']+':'+source['raw_input_sha256'])
            if set(modes[name]['events']) & set(data['events']):
                raise RuntimeError('Upstream manifest duplicates physical originals')
            modes[name]['events'].update(data['events'])
            for kind in ('muons', 'pairs'):
                modes[name][kind].extend(data[kind])
            provenance.append(dict(source=source, mode=name, native=stamp))
    validate_upstream_populations(modes)
    report = dict(schema='shift-upstream-event-paired-comparison-v1', manifest_sha256=hashlib.sha256(payload).hexdigest(),
        analysis_source_sha256=source_sha, cohort=cohort_ref, native_gate=gate_ref, sources=provenance,
        original_equivalence=original_equivalence,
        comparison_qualification=dict(reference=manifest['comparison_qualification'], record=qualification) if qualification else None,
        full_GEN_sources={name: row['full_GEN_accounting'] for name, row in fixed.items()},
        declared_cohort_counts=cohort.get('counts', {}),
        mode_order=manifest['mode_order'], bootstrap_draws=bootstrap, seed=seed,
        population=upstream_population_report(modes), groups=[],
        statistics_scope='Unweighted conditional mechanism diagnostics of each arm native populations; same original-event bootstrap draws within each source/file, including every zero/failed event. No forced track/pair intersection or fallback. Source weight ranges/concentration are separate; these are not physical weighted resolutions or adoption proof.',
        reference_scope='Declared prior-exposed JPsi/DP engineering kernels: distinct positive hitSimTrack IDs and distinct stored hit-associated saved GEN muons only. Wrong charge/direction retained. No angular match, nominal resonance mass or corrected-QCD actual-birth substitution.',
        sparse_scope='Finite selected rows and independent contributing originals are separate. Fewer than two contributing originals in resampleable source strata gives no interval. Widths need20 contributing originals to be labelled resolved; sparse baseline-BB cases do not establish resolution.',
        physics_normalization_ready=False)
    baseline = modes[manifest['mode_order'][0]]['events']
    groups = [('all_conditioned_controls', set(baseline))]
    for field in ('process', 'sample'):
        groups += [(field+'/'+value, {key for key, row in baseline.items() if row[field] == value})
                   for value in sorted({row[field] for row in baseline.values()})]
    for label, keys in groups:
        group_modes = {name: dict(events={key: row for key, row in data['events'].items() if key in keys},
            **{kind: [row for row in data[kind] if tuple(row['event']) in keys] for kind in ('muons', 'pairs')})
            for name, data in modes.items()}
        group = dict(name=label, originals=len(keys), metrics={}, categories={})
        for kind in ('muons', 'pairs'):
            metrics = sorted({metric for data in group_modes.values() for row in data[kind] for metric in row['metrics']})
            group['metrics'][kind] = {metric: upstream_event_statistics(group_modes, kind, metric, bootstrap, seed)
                                      for metric in metrics}
        categories = sorted({row['category'] for data in group_modes.values() for row in data['pairs']})
        for category in categories:
            selected_modes = {}
            for name, data in group_modes.items():
                pairs = [row for row in data['pairs'] if row['category'] == category]
                legs = {(tuple(row['event']), index) for row in pairs for index in row['legs']}
                selected_modes[name] = dict(events=data['events'], pairs=pairs,
                    muons=[row for row in data['muons'] if (tuple(row['event']), row['row_index']) in legs])
            counts = {name: dict(pairs=len(data['pairs']), muons=len(data['muons'])) for name, data in selected_modes.items()}
            statistics = {}
            for kind in ('muons', 'pairs'):
                metrics = sorted({metric for data in selected_modes.values() for row in data[kind] for metric in row['metrics']})
                statistics[kind] = {metric: upstream_event_statistics(selected_modes, kind, metric, bootstrap, seed)
                                   for metric in metrics}
            group['categories'][category] = dict(native_counts=counts, metrics=statistics)
        report['groups'].append(group)
    return upstream_report_values(report)


def upstream_plot_summary_points(report, category, kind, metric):
    """Only display median summaries with identifiable original-event intervals."""
    order = list(dict.fromkeys(row['source']['sample'] for row in report['sources']))
    process = {row['source']['sample']: row['source']['process'] for row in report['sources']}
    points = []
    for group in report['groups']:
        if not group['name'].startswith('sample/'):
            continue
        sample = group['name'].removeprefix('sample/')
        summary = group.get('categories', {}).get(category, {}).get('metrics', {}).get(kind, {}).get(metric)
        if summary is None:
            continue
        for mode, arm in summary['arms'].items():
            interval = arm.get('interval68', {}).get('median')
            if arm['contributing_originals'] < 2 or not interval or arm['estimate'] is None:
                continue
            points.append(dict(mode=mode, sample=sample, process=process[sample], x=order.index(sample)+1,
                value=arm['estimate']['median'], error_down=interval['error_down'], error_up=interval['error_up'],
                contributing_originals=arm['contributing_originals']))
    return points


def upstream_event_case_cells(case, category, order):
    """A table compares each event's chosen outputs, without inventing pair lineage."""
    cells = {}
    for mode in order:
        pairs = [row for row in case['arms'][mode]['pairs'] if row['category'] == category]
        row = pairs[0] if len(pairs) == 1 else {}
        metrics = row.get('metrics', {})
        cells[mode] = dict(pairs=len(pairs), mass_gev=row.get('canonical', {}).get('mass'),
            mass_error_percent=100.*metrics['production_mass_relative'] if 'production_mass_relative' in metrics else None,
            z_error_m=.01*metrics['production_vertex_z_cm'] if 'production_vertex_z_cm' in metrics else None)
    first, second = (cells[name] for name in order)
    for field, label in (('mass_error_percent', 'delta_abs_mass_error_pp'), ('z_error_m', 'delta_abs_z_error_m')):
        cells[label] = (abs(second[field])-abs(first[field])
                        if first[field] is not None and second[field] is not None else None)
    return cells


def plot_upstream_report(report, directory):
    """Compact category PDFs: native populations, event cases and supported medians."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    order = report['mode_order']
    cases = report['population']['event_cases']
    categories = sorted({pair['category'] for case in cases for arm in case['arms'].values() for pair in arm['pairs']})
    sample_order = list(dict.fromkeys(row['source']['sample'] for row in report['sources']))
    source_names = {row['source']['source_context']: row['source']['source_name'] for row in report['sources']}
    figures = {}
    for category in categories:
        path = directory/(category+'.pdf')
        canvas = ROOT.TCanvas('upstream_'+category, '', 1900, 1150)
        canvas.Print(str(path)+'[')
        pages = 0
        def text_page(title, lines, size=.025):
            nonlocal pages
            canvas.Clear(); canvas.cd()
            text = ROOT.TText(); text.SetNDC(); text.SetTextFont(82); text.SetTextSize(size)
            retained = [text.DrawText(.035, .94, title)]
            y = .88
            for line in lines:
                retained.append(text.DrawText(.035, y, line)); y -= .043
            canvas.Print(str(path), 'Title:'+title); pages += 1
        lines = ['Each arm keeps its own selected muons/pairs; all original events remain in accounting.',
                 'Prior-exposed source/file-defined engineering controls; no physical rate or adoption proof.',
                 'Unweighted diagnostic summaries. Native weights and full GEN exposure remain separate.',
                 'No forced track/sourceIndex match or analysis fallback. Zero/failed cases remain in JSON.', '']
        for mode in order:
            native = report['population']['arms'][mode]
            count = sum(pair['category'] == category for case in cases for pair in case['arms'][mode]['pairs'])
            occupied = sum(any(pair['category'] == category for pair in case['arms'][mode]['pairs']) for case in cases)
            lines.append(f'{mode}: {native["originals"]} originals, {count} {category} pairs in {occupied} originals')
        transitions = report['population']['transitions_vs_reference'][order[1]].get('category/'+category, {})
        lines += ['Event pair-count transitions off -> on: '+', '.join(f'{key}: {value}' for key, value in transitions.items()), '',
                  'Original native-weight concentration (not weighted resolution):']
        native = report['population']['arms'][order[0]]
        for source, weights in native['original_native_weights_by_source'].items():
            label = source.split(':', 1)[0]
            ess = weights['effective_originals_signed']
            ess_label = '--' if ess is None else f'{ess:.3g}'
            lines.append(f'{label}: N={weights["originals"]}, W=[{weights["minimum"]:.3g},{weights["maximum"]:.3g}], '
                         f'effective N={ess_label}, negative={weights["negative"]}')
        lines += ['', 'Error bars: shared original-event bootstrap within source/file, finite contributors reported.',
                  'No interval from one contributing original or singleton source strata.',
                  'No unresolved resolution widths are drawn. Sparse baseline-BB cases do not measure resolution.']
        if report.get('original_equivalence') is False:
            mismatch = report['comparison_qualification']['record']['mismatching_original']
            lines += ['Historical original raw equality FAILED for '+':'.join(map(str, mismatch))+'. All originals retained.',
                      'Only identical replay-runtime etaOff/etaOn is qualified; no production equivalence.']
        text_page(category+' | population and interpretation', lines, .022)
        selected = [case for case in cases if any(any(pair['category'] == category for pair in arm['pairs'])
                                                for arm in case['arms'].values())]
        def number(value):
            return '--' if value is None or not math.isfinite(value) else f'{value:.4g}'
        def pair_values(cells, key):
            return number(cells[order[0]][key])+'/'+number(cells[order[1]][key])
        for start in range(0, len(selected), 12):
            lines = ['Event-wise chosen outputs, off/on. Changes do not assert the same physical track/pair.',
                     'Negative Delta|error| means a smaller absolute error for that event selected output.',
                     'Missing reference/no pair/multiple pairs: --. Shared-vertex z only; full cases are in JSON.', '',
                     'source                run:lumi:event        Noff/on  mass GeV(off/on)  mass err %(off/on)  D|mass| pp  z err m(off/on)  D|z| m']
            for case in selected[start:start+12]:
                cells = upstream_event_case_cells(case, category, order)
                label = source_names[case['event'][0]][:20]
                identity = ':'.join(map(str, case['event'][1:]))
                counts = str(cells[order[0]]['pairs'])+'/'+str(cells[order[1]]['pairs'])
                lines.append(f'{label:<21} {identity:<21} {counts:<7} {pair_values(cells,"mass_gev"):<18} '
                    f'{pair_values(cells,"mass_error_percent"):<19} {number(cells["delta_abs_mass_error_pp"]):<11} '
                    f'{pair_values(cells,"z_error_m"):<17} {number(cells["delta_abs_z_error_m"])}')
            no_pair = len(cases)-len(selected)
            lines += ['', f'{no_pair} originals have no {category} pair in either arm; all are retained in accounting.',
                      'Charge/orientation, missing/split references and all selected muons remain in the JSON ledger.']
            text_page(category+f' | original-event cases {start+1}-{min(start+12,len(selected))}', lines, .0165)
        panels = [('pairs', 'production_mass_relative', 'Chosen mass error median', '%', 100.),
                  ('pairs', 'production_vertex_z_cm', 'Shared-vertex z error median', 'm', .01),
                  ('muons', 'production_p_relative', 'Muon p error median', '%', 100.),
                  ('muons', 'production_pt_relative', 'Muon pT error median', '%', 100.),
                  ('muons', 'production_pz_relative', 'Muon pz error median', '%', 100.),
                  ('muons', 'production_qoverp_relative', 'Muon q/p error median', '%', 100.)]
        summaries = [upstream_plot_summary_points(report, category, kind, metric) for kind, metric, *_ in panels]
        if any(summaries):
            canvas.Clear(); canvas.Divide(3, 2); retained = []
            for index, ((kind, metric, title, unit, scale), points) in enumerate(zip(panels, summaries)):
                pad = canvas.cd(index+1); pad.SetLeftMargin(.15); pad.SetBottomMargin(.23); pad.SetTopMargin(.12)
                if not points:
                    text = ROOT.TLatex(); text.SetNDC(); text.SetTextSize(.045)
                    retained += [text, text.DrawLatex(.12,.6,title), text.DrawLatex(.12,.5,'No identifiable bootstrap interval')]
                    continue
                low = min((point['value']-point['error_down'])*scale for point in points)
                high = max((point['value']+point['error_up'])*scale for point in points)
                span = max(high-low, .1*max(abs(low),abs(high)), .01)
                frame = ROOT.TH1F('upstream_frame_'+category+str(index), title, len(sample_order), .5, len(sample_order)+.5)
                frame.SetDirectory(0); frame.SetMinimum(low-.2*span); frame.SetMaximum(high+.35*span)
                frame.GetYaxis().SetTitle('Median error ['+unit+']'); frame.GetYaxis().SetTitleOffset(1.7)
                for i, sample in enumerate(sample_order):
                    label = sample.replace('jpsi_pThat','JPsi ').replace('darkphoton_','').replace('GeV','').replace('_',' ')
                    frame.GetXaxis().SetBinLabel(i+1,label)
                frame.GetXaxis().SetLabelSize(.04); frame.Draw('AXIS'); retained.append(frame)
                heading = ROOT.TLatex(); heading.SetNDC(); heading.SetTextSize(.045)
                retained += [heading, heading.DrawLatex(.15,.94,title)]
                legend = ROOT.TLegend(.16,.76,.63,.88); legend.SetBorderSize(0); legend.SetTextSize(.037)
                for mode_index, mode in enumerate(order):
                    legend_added = False
                    for process in sorted({point['process'] for point in points}):
                        selected_points = sorted((point for point in points if point['mode'] == mode and
                                                  point['process'] == process), key=lambda point: point['x'])
                        if not selected_points:
                            continue
                        graph = ROOT.TGraphAsymmErrors(len(selected_points))
                        for i, point in enumerate(selected_points):
                            graph.SetPoint(i,point['x'],point['value']*scale)
                            graph.SetPointError(i,0.,0.,point['error_down']*scale,point['error_up']*scale)
                        graph.SetMarkerStyle(20+4*mode_index); graph.SetMarkerColor(COLORS[mode_index])
                        graph.SetLineColor(COLORS[mode_index]); graph.SetLineWidth(2)
                        graph.Draw('LP SAME' if len(selected_points)>1 else 'P SAME'); retained.append(graph)
                        if not legend_added:
                            legend.AddEntry(graph,mode,'lp'); legend_added=True
                legend.Draw(); retained.append(legend)
            canvas.cd(); canvas.Print(str(path),'Title:Supported median diagnostics, original-event bootstrap'); pages += 1
        canvas.Print(str(path)+']'); canvas.Close()
        figures[category] = dict(path=str(path.resolve()), bytes=path.stat().st_size,
                                  sha256=native_file_checksum(path), pages=pages)
    return figures


def detector_pull_coverage(values, parents, bootstrap=200, seed=190019):
    """Marginal local-pull coverage; resample physical parents, never individual legs."""
    values, parents = np.asarray(values, dtype=float), np.asarray(parents)
    finite = np.isfinite(values)
    values, parents = values[finite], parents[finite]
    unique = np.unique(parents)
    result = dict(entries=len(values), physical_parents=len(unique), missing=int((~finite).sum()))
    for limit, label in ((1., 'one_sigma'), (1.96, 'one_point_96_sigma')):
        covered = np.abs(values) < limit
        item = dict(covered=int(covered.sum()), fraction=float(covered.mean()) if len(values) else None)
        if len(unique) >= 2:
            generator = np.random.default_rng(seed)
            indices = [np.flatnonzero(parents == parent) for parent in unique]
            draws = []
            for _ in range(bootstrap):
                sample = np.concatenate([indices[i] for i in generator.integers(0, len(unique), len(unique))])
                draws.append(float(covered[sample].mean()))
            item['interval68'] = dict(zip(('low', 'high'), map(float, np.quantile(draws, [.158655, .841345]))))
        result[label] = item
    return result


def detector_category_statistics(modes, bootstrap=200, seed=190019):
    """Pair fixed reference-category identities; native category populations stay separate."""
    reference = next(iter(modes.values()))
    categories = sorted({row['category'] for data in modes.values() for row in data['pairs'].values()})
    result = {}
    for category in categories:
        native_pairs = {name: {key for key, row in data['pairs'].items() if row['category'] == category}
                        for name, data in modes.items()}
        reference_pairs = native_pairs[next(iter(modes))]
        reference_legs = {leg for key in reference_pairs for leg in key}
        native_legs = {name: {leg for key in keys for leg in key} for name, keys in native_pairs.items()}
        output = dict(reference_pairs=len(reference_pairs), reference_legs=len(reference_legs),
                      native_counts={name: dict(pairs=len(native_pairs[name]), legs=len(native_legs[name]))
                                     for name in modes},
                      membership={name: dict(common_reference_pairs=len(reference_pairs & data['pairs'].keys()),
                          lost_reference_pairs=len(reference_pairs-data['pairs'].keys()),
                          new_category_pairs=len(native_pairs[name]-reference['pairs'].keys()),
                          recategorized_reference_pairs=sum(data['pairs'][key]['category'] != category
                              for key in reference_pairs & data['pairs'].keys())) for name, data in modes.items()},
                      pair_metrics={}, muon_metrics={})
        for kind, primary, native, key_name in (
                ('pairs', reference_pairs, native_pairs, 'pair_metrics'),
                ('muons', reference_legs, native_legs, 'muon_metrics')):
            identities = sorted(primary | set().union(*native.values()))
            parents = [repr(key[0][:-2] if kind == 'pairs' else key[:-2]) for key in identities]
            metrics = sorted({metric for name, data in modes.items() for key in native[name]
                              for metric in data[kind][key]['metrics']})
            # Also retain errors of a reference pair that migrates out of this category.
            metrics = sorted(set(metrics) | {metric for key in primary for data in modes.values()
                             if key in data[kind] for metric in data[kind][key]['metrics']})
            for metric in metrics:
                paired_values = {name: [data[kind].get(key, {}).get('metrics', {}).get(metric, math.nan)
                                         if key in primary else math.nan for key in identities]
                                 for name, data in modes.items()}
                paired = detector_baseline_statistics(paired_values, parents, bootstrap, seed)['paired_vs_reference']
                descriptive = {}
                for name, data in modes.items():
                    values = [data[kind][key]['metrics'].get(metric, math.nan) if key in native[name]
                              else math.nan for key in identities]
                    descriptive[name] = paired_parent_statistics({name: values}, parents, bootstrap, seed)
                output[key_name][metric] = dict(paired_vs_reference=paired, all_retained=descriptive)
        result[category] = output
    return result


def detector_selected_case_ledger(modes):
    """Export already evaluated selected rows; this never changes a cohort or metric."""
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not math.isfinite(value) else value
    result = {}
    for name, data in modes.items():
        muons, pairs = [], []
        categories = defaultdict(set)
        for identity, row in data['pairs'].items():
            for leg in identity:
                categories[leg].add(row['category'])
            relation = 'unknown'
            if 'detector_pair_mass_relative' in row['metrics']:
                tracks = [data['muons'][leg]['truth_audit']['SimTrackId'] for leg in identity]
                relation = 'same' if tracks[0] == tracks[1] else 'distinct'
            pairs.append(dict(identity=identity, category=row['category'], is_os=row['is_os'],
                              canonical=row.get('canonical', {}), metrics=row['metrics'],
                              surface_proxy_simtrack_relation=relation))
        for identity, row in data['muons'].items():
            fitted, truth = row.get('fitted', {}), row.get('truth', {})
            muons.append(dict(identity=identity, categories=sorted(categories[identity]),
                charge=row['charge'], topology=row['topology'],
                original_fallback=row['original_fallback'], audit=row['audit'],
                truth_audit=row['truth_audit'], evaluation=row['evaluation'],
                detector_momentum=fitted.get('momentum'), surface_truth_momentum=truth.get('momentum'),
                detector_p=float(np.linalg.norm(fitted['momentum'])) if 'momentum' in fitted else None,
                surface_truth_p=float(np.linalg.norm(truth['momentum'])) if 'momentum' in truth else None,
                canonical_momentum=row.get('canonical_momentum'), hit_gen_index=row.get('hit_gen_index'),
                legacy_hit_association_agrees=row['legacy_hit_association_agrees'], metrics=row['metrics']))
        result[name] = plain(dict(muons=muons, pairs=pairs))
    return result


def run_detector_audit(manifest_path, bootstrap=200, seed=190019):
    """Read only a released fixed milestone, allowing measured population changes."""
    payload = Path(manifest_path).read_bytes()
    manifest = json.loads(payload)
    if manifest.get('released_for_analysis') is not True:
        raise RuntimeError('Detector audit outcomes have not been released for analysis')
    protocol_ref = manifest['protocol']
    protocol_bytes = Path(protocol_ref['path']).read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_ref['sha256']:
        raise RuntimeError('Detector milestone protocol checksum differs')
    protocol = json.loads(protocol_bytes)
    order = manifest['mode_order']
    declared = [x['name'] for x in protocol['ordered_modes']]
    if not order or order[0] != declared[0] or order != [name for name in declared if name in order]:
        raise RuntimeError('Detector modes differ from frozen order')
    # Complete-file hashes and release are checked before any outcome reader.
    for source in manifest['sources']:
        if set(source['modes']) != set(order):
            raise RuntimeError('Detector source does not cover every declared mode')
        for record in source['modes'].values():
            if native_file_checksum(record['path']) != record['sha256']:
                raise RuntimeError('Detector output checksum differs: '+record['path'])
    combined = {name: dict(events={}, muons={}, pairs={}) for name in order}
    provenance = []
    for source in manifest['sources']:
        for name, record in source['modes'].items():
            data, stamp = read_detector_audit(record['path'], source.get('source_context', 'combined'),
                source['raw_input_sha256'], source.get('event_sources'))
            ids = {key[1:] for key in data['events']}
            if ids != {tuple(int(x) for x in key) for key in source['expected_ids']}:
                raise RuntimeError('Detector file does not match every expected event ID')
            for event in data['events'].values():
                event['sample'] = event['sample'] or source['sample']
                event['process'] = event['process'] or source['process']
            for kind in ('events', 'muons', 'pairs'):
                if combined[name][kind].keys() & data[kind].keys():
                    raise RuntimeError('Detector manifest repeats physical '+kind+' identities')
                combined[name][kind].update(data[kind])
            provenance.append(dict(mode=name, source=source, native=stamp))
    union = detector_population_union(combined)
    baseline = combined[order[0]]
    for name, data in combined.items():
        for key in baseline['muons'].keys() & data['muons'].keys():
            if baseline['muons'][key]['raw_signature'] != data['muons'][key]['raw_signature']:
                raise RuntimeError('Original measured track fields differ for common source: '+str(key))
    report = dict(schema='shift-detector-audit-comparison-v1', mode_order=order,
        manifest_sha256=hashlib.sha256(payload).hexdigest(), protocol=protocol_ref,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_draws=bootstrap, seed=seed, sources=provenance, population=union, groups=[],
        scope='Selected-row detector closure; singleton counters retain all-attempt totals. Erased means geometric/invalid removal before cleaning; duplicate-cleaning losses are inputCandidates-erased-retained. No all-attempt identity or pulse/channel completeness claim.',
        statistics_scope='Unweighted diagnostic summaries; native weights preserved for accounting. Each mode pairs separately with reference; common-all is supplemental only. All-retained per-mode counts can differ. No analysis fallback or fabricated missing pair.',
        truth_scope='Independent hit-associated same-surface bracket/vector proxy only for evaluation. Production mass is chosen-pair closure with distinct GEN muons. Detector-surface candidate-pair closure includes same-SimTrack duplicates and is not physical dimuon/resonance resolution. Pair relation is unknown when either valid surface closure is unavailable.',
        covariance_scope='Local5D fitted covariance, same native surface/chart and local-pz sign. Wrong-orientation states retain global-angle errors/counts but no ordinary local-pull coverage. No empirical rescaling. SimHit exits reuse entry momentum; bracket interpolation is a proxy, not exact surface truth.')
    for sample in sorted({x['sample'] for x in baseline['events'].values()}):
        keys = {key for key, event in baseline['events'].items() if event['sample'] == sample}
        modes = {name: {kind: {key: value for key, value in data[kind].items()
                    if (key if kind == 'events' else key[:-2] if kind == 'muons' else key[0][:-2]) in keys}
                    for kind in ('events', 'muons', 'pairs')} for name, data in combined.items()}
        group = dict(sample=sample, events=len(keys), population=detector_population_union(modes),
                     telemetry={}, muon_metrics={}, pair_metrics={}, pull_coverage={})
        for name, data in modes.items():
            counts = Counter()
            for event in data['events'].values():
                counts.update({'event_'+k: v for k, v in event['counters'].items()})
                c = event['counters']
                counts['event_duplicate_cleaning_losses'] += c.get('inputCandidates', 0)-c.get('erased', 0)-c.get('retained', 0)
            statuses = defaultdict(Counter)
            for row in data['muons'].values():
                counts['selected'] += 1
                for field in ('measured_valid', 'truth_valid', 'covariance_valid', 'pull_reference_valid'):
                    counts[field] += row['evaluation'][field]
                counts['original_fallback'] += row['original_fallback']
                counts['decomposed'] += 'material_return_error_gev' in row['metrics']
                if row['production']:
                    counts['hit_gen_matched'] += 1
                    counts['canonical_wrong_charge'] += row['charge'] != (-1 if row['production']['pdgId'] == 13 else 1)
                if row['truth']['valid'] and row['evaluation']['measured_valid']:
                    counts['matched_detector_states'] += 1
                    counts['detector_wrong_charge'] += row['audit']['Charge'] != row['truth_audit']['Charge']
                    counts['local_direction_disagrees'] += row['audit']['LocalPzSign'] != row['truth_audit']['LocalPzSign']
                if row['legacy_hit_association_agrees'] is not None:
                    counts['legacy_hit_agree'] += row['legacy_hit_association_agrees']
                    counts['legacy_hit_disagree'] += not row['legacy_hit_association_agrees']
                statuses['TruthStatus'][str(int(row['truth_audit']['Status']))] += 1
                if row['truth_audit']['Status'] == 1:
                    for field in ('LowerKind', 'UpperKind'):
                        statuses['Truth'+field][str(int(row['truth_audit'][field]))] += 1
                for field in ('NativeStatus', 'TargetStatus', 'SnapshotKind', 'Variant', 'Iteration',
                              'Available', 'FitAccepted', 'CanonicalAccepted'):
                    statuses[field][str(int(row['audit'][field]))] += 1
            for key in modes[order[0]]['muons'].keys() & data['muons'].keys():
                for prefix in ('Input', 'Used'):
                    before = tuple(modes[order[0]]['muons'][key]['audit'][prefix+'Hash'+half] for half in ('Lo', 'Hi'))
                    after = tuple(data['muons'][key]['audit'][prefix+'Hash'+half] for half in ('Lo', 'Hi'))
                    counts[prefix.lower()+'_hash_changed'] += before != after
            spans = [row['truth_audit']['Span'] for row in data['muons'].values()
                     if row['truth_audit']['Status'] == 1 and math.isfinite(row['truth_audit']['Span'])]
            group['telemetry'][name] = dict(counts=dict(counts), statuses={k: dict(v) for k, v in statuses.items()},
                truth_bracket_span=dict(entries=len(spans), minimum=min(spans), median=float(np.median(spans)),
                                        maximum=max(spans)) if spans else dict(entries=0))
        for kind, output_name in (('muons', 'muon_metrics'), ('pairs', 'pair_metrics')):
            identities = group['population']['muon_union' if kind == 'muons' else 'pair_union']
            parents = [repr(key[:-2] if kind == 'muons' else key[0][:-2]) for key in identities]
            metric_names = sorted({metric for data in modes.values() for row in data[kind].values() for metric in row['metrics']})
            for metric in metric_names:
                values = {name: [data[kind].get(key, {}).get('metrics', {}).get(metric, math.nan)
                                 for key in identities] for name, data in modes.items()}
                group[output_name][metric] = detector_baseline_statistics(values, parents, bootstrap, seed)
                if kind == 'muons' and metric.endswith('_pull'):
                    group['pull_coverage'][metric] = {name: detector_pull_coverage(v, parents, bootstrap, seed)
                                                    for name, v in values.items()}
        group['categories'] = detector_category_statistics(modes, bootstrap, seed)
        group['selected_case_ledger'] = detector_selected_case_ledger(modes)
        for name, cases in group['selected_case_ledger'].items():
            counts = Counter(row['surface_proxy_simtrack_relation'] for row in cases['pairs'])
            group['telemetry'][name]['surface_proxy_simtrack_pairs'] = {key: counts[key]
                for key in ('distinct', 'same', 'unknown')}
        report['groups'].append(group)
    return report


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


def plot_detector_audit_report(report, output_directory):
    """Category PDFs for migrations and pairwise detector-mode diagnostics."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(28., 19.5)
    modes = report['mode_order']
    short = {'RKreference': 'RK', 'legacyG4none_diagnostic': 'G4 old', 'G4_B': 'B',
             'G4_BE': 'BE', 'G4_BEF': 'BEF', 'G4_BEFV': 'BEFV',
             'RKdedup': 'RK unique hits', 'BEFVreference': 'BEFV', 'BEFVdedup': 'BEFV unique hits'}
    reference_label = short.get(modes[0], modes[0])
    categories = sorted({name for group in report['groups'] for name in group['categories']})
    pages = {}
    for category in categories:
        output = str(output_directory/(category+'.pdf'))
        canvas = ROOT.TCanvas('detector_milestone', category, 1500, 1040)
        canvas.Print(output+'[')
        page_number = 0
        def page(title):
            nonlocal page_number
            page_number += 1
            canvas.Clear(); canvas.cd()
            text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.029)
            text.DrawLatex(.04, .955, category+' | '+title)
            text.SetTextSize(.017)
            text.DrawLatex(.04, .025, 'Prior-peeked development | selected-row closure + all-candidate counters | parent-bootstrap 68% intervals')
            text.DrawLatex(.94, .025, str(page_number))
            return [text]
        retained = page('How to read this mechanism check')
        text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.026)
        notes = [
            f'Each alternative pairs separately with {reference_label} on its own common valid source set.',
            'A failed diagnostic mode does not shrink another mode. Missing states and pairs stay missing.',
            'Native category counts may change. Paired curves use fixed reference-category identities.',
            'Detector snapshot closure can include failed/unapplied fits; accepted production ancestry is separate.',
            f'Graphs show changes from {reference_label}; negative width change alone does not establish improvement.',
            'No width point below 20 finite observations. No population error bar from one physical parent.',
            'Surface truth is a measured-plane SimHit bracket proxy; exits reuse entry momentum.',
            'Detector-surface candidate-pair closure includes duplicate SimTracks; it is not dimuon resolution.',
            'Same/distinct surface SimTrack flags and counts are explicit; unavailable surface references are unknown.',
            'Local pulls require valid covariance, matching native surface and local-pz orientation.',
            'Wrong charge/orientation, transport failures, erased candidates and pair migrations remain counted.',
            'No truth, sample mass or lifetime enters reconstruction or selection; no empirical error scaling.',
            'Summaries are unweighted diagnostics; native event weights remain preserved for accounting.',
            'B: backward covariance; E: loss/interface Jacobian; F: field gradient; V: ionization variance.',
        ]
        for i, line in enumerate(notes): text.DrawLatex(.055, .84-.057*i, line)
        retained.append(text); canvas.Print(output, 'Title:Guide')
        for group in report['groups']:
            if category not in group['categories']:
                continue
            selected = group['categories'][category]
            retained = page(group['sample'].replace('_', ' ')+' | counts and limits')
            text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.024)
            text.DrawLatex(.05, .85, f'{group["events"]} frozen events; category reference: {selected["reference_pairs"]} pairs / {selected["reference_legs"]} distinct legs')
            text.SetTextSize(.021)
            text.DrawLatex(.05, .79, 'All candidates: input / available / detector-accepted / target-valid / geometric-erased / cleaned / retained')
            for i, mode in enumerate(modes):
                c = group['telemetry'][mode]['counts']
                values = [c.get('event_'+key, 0) for key in ('inputCandidates', 'stateAvailable', 'fitAccepted',
                          'targetValid', 'erased', 'duplicate_cleaning_losses', 'retained')]
                text.DrawLatex(.06, .74-.041*i, short.get(mode, mode)+': '+ ' / '.join(map(str, values)))
            y = .43
            text.DrawLatex(.05, y, 'Category pairs: native / common reference IDs / lost / new / recategorized')
            for i, mode in enumerate(modes):
                m = selected['membership'][mode]
                values = [selected['native_counts'][mode]['pairs'], m['common_reference_pairs'],
                          m['lost_reference_pairs'], m['new_category_pairs'], m['recategorized_reference_pairs']]
                text.DrawLatex(.06, y-.048-.038*i, short.get(mode, mode)+': '+' / '.join(map(str, values)))
            text.SetTextSize(.019)
            text.DrawLatex(.05, .105, 'Candidate state/closure is retained-row only; counters preserve erased totals, not their missing identities.')
            text.DrawLatex(.05, .065, 'Native medians follow; graphs require paired uncertainty support; widths need at least 20 observations.')
            retained.append(text); canvas.Print(output, 'Title:'+group['sample']+' counts')
            retained = page(group['sample'].replace('_', ' ')+' | native medians and counts')
            text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.024)
            text.DrawLatex(.05, .85, 'Each mode uses its own native category and valid reference set; populations can differ.')
            columns = ((.26, 'muon_metrics', 'p_relative', 'Detector p [%]', 100.),
                       (.44, 'muon_metrics', 'angle_error', 'Detector angle [mrad]', 1000.),
                       (.64, 'pair_metrics', 'production_mass_relative', 'Production mass [%]', 100.),
                       (.83, 'pair_metrics', 'production_vertex_z', 'Production z [m]', .01))
            text.SetTextSize(.019); text.DrawLatex(.05, .77, 'Mode')
            for x, _, _, title, _ in columns: text.DrawLatex(x, .77, title)
            for i, mode in enumerate(modes):
                y = .70-.067*i
                text.DrawLatex(.05, y, short.get(mode, mode))
                for x, container, metric, _, factor in columns:
                    item = selected[container].get(metric, {}).get('all_retained', {}).get(mode)
                    policy = item['policies'][mode] if item else {}
                    estimate = policy.get('estimate')
                    value = '--; N=0' if not estimate else f'{factor*estimate["median"]:+.3g}; N={item["entries"]}'
                    text.DrawLatex(x, y, value)
                    if 'interval68' in policy:
                        bounds = policy['interval68']['median']
                        text.DrawLatex(x, y-.025, f'[{factor*bounds["low"]:+.3g}, {factor*bounds["high"]:+.3g}]')
            text.SetTextSize(.022)
            text.DrawLatex(.05, .20, 'Brackets: parent-bootstrap 68% interval of the median, where at least two physical parents exist.')
            text.DrawLatex(.05, .15, 'One-parent values are raw mechanism observations; they establish no population precision.')
            text.DrawLatex(.05, .10, 'All missing/error counts and conditional pull coverage remain in the numerical report.')
            retained.append(text); canvas.Print(output, 'Title:'+group['sample']+' native medians')
            panels = [
                ('Detector and material return', (
                    ('muon_metrics', 'p_relative', 'Detector momentum relative error', '%', 100.),
                    ('muon_metrics', 'angle_error', 'Detector angular distance', 'mrad', 1000.),
                    ('muon_metrics', 'qoverp_pull', 'Detector q/p pull', '', 1.),
                    ('muon_metrics', 'material_return_relative', 'Material-return error / production p', '%', 100.))),
                ('Production pair closure', (
                    ('pair_metrics', 'production_mass_relative', 'Chosen-pair production mass error', '%', 100.),
                    ('pair_metrics', 'production_vertex_x', 'Shared production x error', 'cm', 1.),
                    ('pair_metrics', 'production_vertex_y', 'Shared production y error', 'cm', 1.),
                    ('pair_metrics', 'production_vertex_z', 'Shared production z error', 'm', .01))),
                ('Detector-surface candidate-pair closure', (
                    ('pair_metrics', 'detector_pair_mass_relative', 'Candidate-pair surface mass closure', '%', 100.),
                    ('pair_metrics', 'detector_pair_opening_error', 'Candidate-pair surface opening closure', 'mrad', 1000.))),
            ]
            for statistic, label in (('median', 'median'), ('sigma68', '68% half-width')):
                for heading, definitions in panels:
                    def supported(container, metric):
                        item = selected[container].get(metric, {})
                        return any('interval68' in row['policies'].get(mode, {}) and
                                   (statistic != 'sigma68' or row['quantile_resolved'])
                                   for mode, row in item.get('paired_vs_reference', {}).items())
                    if not any(supported(a, b) for a, b, *_ in definitions):
                        continue
                    retained = page(group['sample'].replace('_', ' ')+' | '+heading+' | '+label+' change')
                    pad = ROOT.TPad('detector_panels', '', 0., .065, 1., .90)
                    pad.Draw(); pad.Divide(2, 2, .014, .018); retained.append(pad)
                    for panel, (container, metric, title, unit, factor) in enumerate(definitions, 1):
                        axis_pad = pad.cd(panel)
                        axis_pad.SetLeftMargin(.18); axis_pad.SetBottomMargin(.18)
                        axis_pad.SetRightMargin(.05); axis_pad.SetTopMargin(.12)
                        multigraph = ROOT.TMultiGraph(); retained.append(multigraph)
                        lows, highs, points = [], [], []
                        graph = None
                        for index, mode in enumerate(modes[1:], 1):
                            item = selected[container].get(metric, {}).get('paired_vs_reference', {}).get(mode)
                            policy = item['policies'].get(mode, {}) if item else {}
                            if 'interval68' not in policy or (statistic == 'sigma68' and not item['quantile_resolved']):
                                graph = None
                                continue
                            delta = policy['paired_delta_vs_canonical'][statistic]
                            if graph is None:
                                graph = ROOT.TGraphAsymmErrors(); graph.SetMarkerStyle(20)
                                graph.SetMarkerColor(ROOT.kAzure+2); graph.SetLineColor(ROOT.kAzure+2)
                                graph.SetLineWidth(2); multigraph.Add(graph, 'LP'); retained.append(graph)
                            value, lo, hi = (factor*delta[k] for k in ('value', 'low', 'high'))
                            n = graph.GetN(); graph.SetPoint(n, index, value)
                            graph.SetPointError(n, 0., 0., max(0., value-lo), max(0., hi-value))
                            lows.append(lo); highs.append(hi); points.append(index)
                        if points:
                            multigraph.SetTitle(title+';Mode paired with '+reference_label+';Change in '+label+(' ['+unit+']' if unit else ''))
                            multigraph.Draw('A'); multigraph.GetXaxis().SetLimits(.5, len(modes)-.5)
                            xaxis = multigraph.GetXaxis(); xaxis.SetLabelSize(.06)
                            for index, mode in enumerate(modes[1:], 1):
                                xaxis.SetBinLabel(xaxis.FindBin(index), short.get(mode, mode))
                            xaxis.LabelsOption('h'); xaxis.SetTitleSize(.045); xaxis.SetTitleOffset(1.35)
                            lo, hi = min(lows+[0.]), max(highs+[0.]); span = max(hi-lo, .1*max(abs(lo), abs(hi)), 1.e-6)
                            multigraph.SetMinimum(lo-.15*span); multigraph.SetMaximum(hi+.2*span)
                            multigraph.GetYaxis().SetTitleSize(.05); multigraph.GetYaxis().SetLabelSize(.045)
                            multigraph.GetYaxis().SetTitleOffset(1.5)
                            zero = ROOT.TLine(.5, 0., len(modes)-.5, 0.); zero.SetLineStyle(2)
                            zero.Draw(); retained.append(zero)
                        else:
                            text = ROOT.TLatex(); text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.041)
                            text.DrawLatex(.12, .78, title)
                            text.DrawLatex(.12, .52, 'No paired population interval; see counts / JSON')
                            retained.append(text)
                    canvas.cd(); canvas.Print(output, 'Title:'+group['sample']+' '+heading+' '+statistic)
        canvas.Print(output+']'); canvas.Close(); pages[category] = page_number
    return pages


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
    parser.add_argument('--actual-birth-manifest', help='Frozen jointly published Nano/material-audit cohort')
    parser.add_argument('--actual-birth-release', help='Explicit analysis release bound to the actual-birth cohort SHA')
    parser.add_argument('--upstream-replay-manifest', help='Frozen upstream replay sources and native arm qualification')
    parser.add_argument('--upstream-replay-release', help='Explicit release bound to the upstream cohort, manifest and reader')
    parser.add_argument('--upstream-plot-output', help='Optional final PDF directory, one file per observed upstream category')
    parser.add_argument('--detector-audit-manifest', help='Released detector-fit milestone with explicit population migrations')
    parser.add_argument('--detector-plot-output', help='Output directory for detector-milestone category PDFs')
    parser.add_argument('--native-detector-overlay', nargs=2, metavar=('BASELINE', 'REFIT'),
                        help='Prepare separate detector-refit branches while preserving every baseline event/pair')
    parser.add_argument('--extraction', help='Exact source inventory accompanying --paired-npz')
    parser.add_argument('--protocol', help='Predeclared development and validation protocol JSON')
    parser.add_argument('--mode', choices=('development', 'validation'), default='development')
    parser.add_argument('--frozen-policy', help='Required policy decision JSON before formal validation')
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=190019)
    args=parser.parse_args()
    if args.upstream_replay_manifest:
        if (not args.upstream_replay_release or args.inputs or args.actual_birth_manifest or args.actual_birth_release
                or args.native_manifest or args.native_detector_overlay or args.detector_audit_manifest
                or args.paired_npz or args.paired_report_json or args.paired_plot_output or args.detector_plot_output):
            parser.error('Upstream replay requires only its manifest, explicit release and output JSON')
        report = run_upstream_replay(args.upstream_replay_manifest, args.upstream_replay_release,
                                     args.bootstrap, args.seed)
        if args.upstream_plot_output:
            report['figures'] = plot_upstream_report(report, args.upstream_plot_output)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output), arms=report['mode_order'], groups=len(report['groups']))))
        return
    if args.upstream_replay_release or args.upstream_plot_output:
        parser.error('Upstream release/plots require --upstream-replay-manifest')
    if args.actual_birth_manifest:
        if (not args.actual_birth_release or args.inputs or args.native_manifest or args.native_detector_overlay
                or args.detector_audit_manifest or args.paired_npz or args.paired_report_json or args.bootstrap < 20):
            parser.error('Actual-birth audit requires its exact release, no other event inputs, and at least20 draws')
        report = run_actual_birth_audit(args.actual_birth_manifest, args.actual_birth_release, args.bootstrap, args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(output), counts=report['counts'])))
        return
    if args.actual_birth_release:
        parser.error('--actual-birth-release requires --actual-birth-manifest')
    if args.detector_audit_manifest:
        if (args.inputs or args.native_manifest or args.native_detector_overlay or args.paired_npz
                or args.paired_report_json or args.bootstrap < 20):
            parser.error('Detector audit requires only its released manifest and at least20 bootstrap draws')
        report = run_detector_audit(args.detector_audit_manifest, args.bootstrap, args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        pages = plot_detector_audit_report(report, args.detector_plot_output) if args.detector_plot_output else {}
        print(json.dumps(dict(output=str(output), samples=len(report['groups']), category_pdf_pages=pages)))
        return
    if args.detector_plot_output:
        parser.error('--detector-plot-output requires --detector-audit-manifest')
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
