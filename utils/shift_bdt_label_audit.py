#!/usr/bin/env python3
# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Trace a dominant genuine QCD pair and audit retained-pair truth coverage.

This consumes completed evaluation evidence and the shared reconstructed-pair
cache. It does not select events, alter labels, train, or rerun reconstruction.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def locate_source(sources, row, total_pairs):
    counts = [s['pairs'] for s in sources]
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != total_pairs:
        raise ValueError('Source counts do not partition evaluated pairs')
    if not 0 <= row < total_pairs:
        raise ValueError('Pair row outside evaluated population')
    offset = 0
    for index, count in enumerate(counts):
        if offset <= row < offset + count:
            return index, row - offset
        offset += count
    raise AssertionError('Unreachable pair row')


def ancestor_indices(mothers, start):
    result, seen = [], set()
    index = int(start)
    while index >= 0:
        if index >= len(mothers) or index in seen:
            raise ValueError('Malformed recorded generator ancestry')
        seen.add(index)
        result.append(index)
        index = int(mothers[index])
    return result


def corrected_weight(native, probability, stored, cross_section, denominator):
    values = (native, probability, stored, cross_section, denominator)
    if not all(math.isfinite(v) for v in values) or not 0 < probability <= 1 or denominator <= 0 or cross_section < 0:
        raise ValueError('Invalid sampling or full-GEN normalization')
    expected = native / probability
    if not math.isclose(stored, expected, rel_tol=1.e-12, abs_tol=1.e-12):
        raise ValueError('Stored corrected weight is not native weight / sampling probability')
    return stored * cross_section / denominator


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def residual_summary(reconstructed, truth, labels, process, topology=None, z_errors=None, scores=None, threshold=None):
    """Truth residuals for nominal common-origin pairs; never a selection."""
    if reconstructed.shape != (len(labels), 3) or truth.shape != (len(labels), 2, 3):
        raise ValueError('Unaligned reconstructed/truth position arrays')
    nominal = labels == 1
    valid = nominal & np.isfinite(reconstructed).all(axis=1) & np.isfinite(truth).all(axis=(1, 2))
    # A pair may be labelled common within 0.01 cm without identical positions.
    # Use the midpoint rather than privileging either reconstructed leg.
    delta = reconstructed - truth.mean(axis=1)
    def summarize(mask):
        values = delta[mask & valid]
        result = dict(nominal_genuine_pairs=int((mask & nominal).sum()), finite_position_pairs=len(values),
                    signed_z_cm_quantiles_10_50_90=np.quantile(values[:, 2], [.1, .5, .9]).tolist() if len(values) else None,
                    absolute_z_cm_quantiles_10_50_90=np.quantile(np.abs(values[:, 2]), [.1, .5, .9]).tolist() if len(values) else None,
                    distance_cm_quantiles_10_50_90=np.quantile(np.linalg.norm(values, axis=1), [.1, .5, .9]).tolist() if len(values) else None,
                    transverse_distance_cm_quantiles_10_50_90=np.quantile(np.linalg.norm(values[:, :2], axis=1), [.1, .5, .9]).tolist() if len(values) else None)
        if z_errors is not None:
            good = mask & valid & np.isfinite(z_errors) & (z_errors > 0)
            pulls = delta[good, 2] / z_errors[good]
            result['quoted_z_uncertainty'] = dict(finite_positive_pairs=int(good.sum()),
                invalid_error_pairs=int((mask & valid & ~(np.isfinite(z_errors) & (z_errors > 0))).sum()),
                error_cm_quantiles_10_50_90=np.quantile(z_errors[good], [.1, .5, .9]).tolist() if good.any() else None,
                signed_pull_quantiles_10_50_90=np.quantile(pulls, [.1, .5, .9]).tolist() if good.any() else None,
                absolute_pull_quantiles_10_50_90=np.quantile(np.abs(pulls), [.1, .5, .9]).tolist() if good.any() else None,
                within_one_quoted_sigma=int((np.abs(pulls) <= 1).sum()), within_two_quoted_sigma=int((np.abs(pulls) <= 2).sum()))
        return result
    result = dict(available=True, units='cm', reference='Midpoint of the two matched generator production positions',
                  selection_applied=False, interpretation='A common-origin truth label is independent of fit accuracy. Residuals are descriptive only.',
                  all=summarize(np.ones(len(labels), dtype=bool)),
                  per_process={str(p):summarize(process == p) for p in np.unique(process)})
    if topology is not None:
        result['per_topology'] = {str(p):summarize(topology == p) for p in np.unique(topology)}
    if scores is not None and threshold is not None:
        passed = scores >= threshold
        result['unchanged_benchmark_cut'] = dict(threshold=float(threshold), source='Completed evaluator frozen threshold',
            before=summarize(np.ones(len(labels), dtype=bool)), after=summarize(passed),
            per_process_after={str(p):summarize((process == p) & passed) for p in np.unique(process)})
    result['uncertainty_interpretation'] = 'Spatial residuals and quoted errors are reported together. A large residual with a large quoted uncertainty does not alone demonstrate a fit defect or uncertainty miscalibration.'
    return result


def optional_match_audit(labels, process):
    """Separate stored association failure evidence without inventing matches."""
    n = len(labels['y'])
    fields = ('hitTruthLayers', 'hitTruthAmbiguousLayers', 'genPartIdx', 'genPartDeltaR')
    availability = {name:dict(present=name in labels,
        finite_legs=int(np.isfinite(labels[name]).sum()) if name in labels else 0) for name in fields}
    invalid = labels['gen_index'] < 0
    layers, purity, sim = labels['matched_layers'], labels['match_purity'], labels['sim_track_id']
    failure = np.full((n, 2), 'stored_gen_link', dtype='U48')
    failure[invalid] = 'unresolved_gen_link'
    failure[invalid & (sim >= 0)] = 'accepted_simtrack_without_gen_link'
    failure[invalid & (sim < 0) & ((layers < 3) | ~np.isfinite(purity) | (purity < .75))] = 'insufficient_layer_or_purity_gate'
    failure[invalid & (sim < 0) & (layers >= 3) & np.isfinite(purity) & (purity >= .75)] = 'rejected_identity_despite_summary_gate'
    def summary(mask):
        unknown = mask & (labels['y'] < 0)
        item = dict(unknown_pairs=int(unknown.sum()), leg_states=dict(Counter(failure[unknown].ravel().tolist())))
        if 'genPartIdx' in labels:
            angle = labels['genPartIdx']
            present = np.isfinite(angle).all(axis=1)
            item['unknown_pairs_with_both_angle_matches'] = int((unknown & present & (angle >= 0).all(axis=1)).sum())
            item['unknown_pairs_with_any_angle_match'] = int((unknown & present & (angle >= 0).any(axis=1)).sum())
        return item
    return dict(optional_branch_availability=availability,
                interpretation='Angular matches remain diagnostic and never replace hit-based labels. A missing unique truth link cannot be resolved from summary counts alone.',
                overall=summary(np.ones(n, dtype=bool)), per_process={str(p):summary(process == p) for p in np.unique(process)})


def geometry_figure(report, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    processes = [('dy', 'DY', '#e68613'), ('jpsi', 'J/psi', '#2869a6'), ('qcd', 'QCD', '#378b45'), ('signal', 'Signal grid', '#a445ab')]
    for axis, key, title in zip(axes,
            ('reconstructed_mass', 'reconstructed_vertex_z', 'reconstructed_vertex_radius'),
            ('Reconstructed mass [GeV]', 'Reconstructed vertex z [cm]', 'Reconstructed radius [cm]')):
        bins = report['continuous'][key]['bins']
        for offset, (process, label, color) in enumerate(processes):
            first = True
            for index, entry in enumerate(bins):
                item = entry['per_process'].get(process)
                if not item or item['unknown_fraction'] is None:
                    continue
                value, interval = item['unknown_fraction'], item['unknown_ci95_pair_wilson']
                axis.errorbar(index + (offset - 1.5) * .13, 100 * value,
                    yerr=[[100 * (value - interval[0])], [100 * (interval[1] - value)]],
                    color=color, marker='o', linestyle='none', capsize=2, label=label if first else None)
                first = False
        ticks = []
        for entry in bins:
            low, high = entry['lower'], entry['upper']
            ticks.append(f'< {high:.3g}' if low is None and high is not None else
                         f'>= {low:.3g}' if high is None and low is not None else
                         f'{low:.3g} to {high:.3g}' if low is not None else 'all finite')
        axis.set_xticks(range(len(bins)), ticks, rotation=25, ha='right')
        axis.set(xlabel=title, ylabel='Unknown-label fraction [%]', ylim=(0, 100))
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Truth-match coverage before any BDT cut\nAll-retained-pair quartiles; unweighted fractions and 95% pair-Wilson intervals')
    fig.savefig(output / 'unknown_geometry.pdf')
    plt.close(fig)


def dominant_qcd(manifest_path, report_path, evaluation_path, inventory_path, output):
    import awkward as ak
    import uproot
    from labels import pair_label, label_diagnostics, TRUTH_BRANCHES
    manifest, report, inventory = map(read, (manifest_path, report_path, inventory_path))
    if not manifest.get('inputs_complete') or not report.get('complete'):
        raise ValueError('Completed input/evaluation evidence is required')
    if report['input_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Evaluation input manifest changed')
    if report['artifact_sha256'].get(Path(evaluation_path).name) != digest(evaluation_path):
        raise ValueError('Evaluation rows changed')
    with np.load(evaluation_path, allow_pickle=False) as data:
        data = {key: data[key] for key in data.files}
    candidates = np.flatnonzero((data['process'] == 'qcd') & (data['y'] == 1))
    if len(candidates) == 0:
        raise ValueError('No genuine QCD pair to trace')
    row_index = int(candidates[np.argmax(np.abs(data['evaluation_weight'][candidates]))])
    source_index, pair_offset = locate_source(report['sources'], row_index, len(data['y']))
    row, source = manifest['inputs'][source_index], report['sources'][source_index]
    if Path(row['path']).resolve() != Path(source['path']).resolve() or row['process'] != 'qcd':
        raise ValueError('Source ordering differs from evaluator')
    source_path = Path(row['path'])
    if digest(source_path) != row['nano_sha256']:
        raise ValueError('Dominant pair ROOT source changed')
    identity = list(map(int, data['ids'][row_index]))
    with uproot.open(source_path, handler=uproot.source.file.MultithreadedFileSource, num_workers=1) as root:
        tree = root['Events']
        ids = tree.arrays(['run', 'luminosityBlock', 'event', 'nShiftDimuonVertex'], library='np')
        matches = np.flatnonzero(np.logical_and.reduce([ids[name] == value for name, value in zip(('run', 'luminosityBlock', 'event'), identity)]))
        if len(matches) != 1:
            raise ValueError('Target physical event is absent or repeated')
        entry = int(matches[0])
        local_pair = pair_offset - int(ids['nShiftDimuonVertex'][:entry].sum())
        if not 0 <= local_pair < int(ids['nShiftDimuonVertex'][entry]):
            raise ValueError('Pair offset does not identify the target event')
        branches = [k for k in tree.keys() if k.startswith(('GenPart_', 'ShiftMuon_', 'ShiftDimuonVertex_', 'shiftSampling')) or k in ('run', 'luminosityBlock', 'event', 'genWeight', 'nShiftMuon', 'nShiftDimuonVertex')]
        arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library='ak', how=dict)
    event = {key: ak.to_list(values[0]) for key, values in arrays.items()}
    legs = [int(event['ShiftDimuonVertex_muonIdx1'][local_pair]), int(event['ShiftDimuonVertex_muonIdx2'][local_pair])]
    label, reason = pair_label(event, *legs)
    score = float(event['ShiftDimuonVertex_bdtScore'][local_pair])
    if label != 1 or reason != data['reason'][row_index] or score != data['score'][row_index]:
        raise ValueError('Reopened pair disagrees with evaluated label/score')
    diagnostics = label_diagnostics({key: arrays[key] for key in TRUTH_BRANCHES}, dict(event_index=[0], muon_indices=[legs]))
    matched = [int(event['ShiftMuon_hitGenPartIdx'][leg]) for leg in legs]
    ancestors = [ancestor_indices(event['GenPart_genPartIdxMother'], index) for index in matched]
    generator = {str(index): {key.removeprefix('GenPart_'): values[index] for key, values in event.items() if key.startswith('GenPart_')} for index in sorted(set(sum(ancestors, [])))}
    truth_positions = [[float(event['GenPart_' + key][index]) for key in ('vx', 'vy', 'vz')] for index in matched]
    reco_position = [float(event['ShiftDimuonVertex_' + key][local_pair]) for key in ('vx', 'vy', 'vz')]
    normalization = inventory['bins'][row['stratum']]
    if any(row[key] != normalization[key] for key in ('cross_section_pb', 'full_parent_gen_sumw')):
        raise ValueError('Source normalization differs from sealed inventory')
    weight = corrected_weight(float(event['genWeight']), float(event['shiftSamplingProbability']), float(event['shiftSamplingGenWeight']), row['cross_section_pb'], row['full_parent_gen_sumw'])
    if not math.isclose(weight, float(data['evaluation_weight'][row_index]), rel_tol=1.e-14):
        raise ValueError('Evaluation weight differs from W/p * sigma/full GEN sumw')
    authorities = {}
    for key in ('sampling_plan', 'sampling_ledger'):
        item = normalization[key]
        if digest(item['path']) != item['sha256']:
            raise ValueError('Sealed normalization authority changed: ' + key)
        authorities[key] = reference(item['path'])
    ledger_matches = []
    with gzip.open(normalization['sampling_ledger']['path'], 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            if item['stratum'] == row['stratum'] and item['event_id'] == identity:
                ledger_matches.append(item)
    if len(ledger_matches) != 1:
        raise ValueError('Sampling ledger does not identify exactly one physical event')
    ledger = ledger_matches[0]
    if (ledger['corrected_weight'] != event['shiftSamplingGenWeight'] or ledger['sampling_probability'] != event['shiftSamplingProbability'] or ledger['native_weight'] != event['genWeight']):
        raise ValueError('Sampling ledger differs from stored event weights')
    plan = read(normalization['sampling_plan']['path'])
    plan_bin = plan['strata'][row['stratum']]
    if plan_bin['original_sumw'] != row['full_parent_gen_sumw']:
        raise ValueError('Full GEN denominator differs from original sampling population')
    full_genuine = (data['y'] == 1) & (data['process'] != 'signal')
    heldout_genuine = full_genuine & data['heldout']
    dominance = {}
    for name, mask in (('full_sm_genuine', full_genuine), ('heldout_sm_genuine', heldout_genuine), ('full_qcd_genuine', (data['process'] == 'qcd') & (data['y'] == 1))):
        total = float(data['evaluation_weight'][mask].sum())
        dominance[name] = dict(pairs=int(mask.sum()), sumw=total, target_included=bool(mask[row_index]), target_fraction=weight / total if mask[row_index] and total else 0.)
    result = dict(schema='shift-bdt-dominant-qcd-audit-v1', complete=True, source=reference(source_path), source_index=source_index,
                  evaluation_row=row_index, source_pair_offset=pair_offset, event_entry=entry, event_id=identity, pair_index=local_pair,
                  muon_indices=legs, matched_gen_indices=matched, ancestry_indices=ancestors, generator_ancestry=generator,
                  match_diagnostics={key: value[0] for key, value in diagnostics.items()}, label=label, reason=reason,
                  score=score, threshold=report['threshold'], passes_frozen_cut=score >= report['threshold'], heldout=bool(data['heldout'][row_index]),
                  reconstructed_pair={key.removeprefix('ShiftDimuonVertex_'): values[local_pair] for key, values in event.items() if key.startswith('ShiftDimuonVertex_')},
                  reconstructed_legs=[{key.removeprefix('ShiftMuon_'): values[leg] for key, values in event.items() if key.startswith('ShiftMuon_')} for leg in legs],
                  truth_positions_cm=truth_positions, reconstructed_position_cm=reco_position,
                  reconstructed_minus_truth_cm=(np.asarray(reco_position) - np.asarray(truth_positions[0])),
                  position_distance_cm=float(np.linalg.norm(np.asarray(reco_position) - truth_positions[0])),
                  normalization=dict(stratum=row['stratum'], native_weight=event['genWeight'], sampling_probability=event['shiftSamplingProbability'], inverse_probability=event['shiftSamplingWeight'], corrected_weight=event['shiftSamplingGenWeight'], cross_section_pb=row['cross_section_pb'], full_parent_gen_sumw=row['full_parent_gen_sumw'], plotted_pair_weight=weight, ledger_event=ledger, sampling_bin={k:v for k,v in plan_bin.items() if k != 'original_normalization_records'}, normalization_authorities=authorities),
                  dominance=dominance, fullsample_genuine_qcd_pairs=len(candidates),
                  interpretation='Nominal label establishes two independently hit-associated generator muons at a common production position. It does not establish accurate reconstructed mass or vertex. One event cannot establish QCD genuine-pair acceptance.',
                  unavailable_checks={'individual_CSC_layer_votes':'Nano stores counts and winning identity, not layer-by-layer measurement/SimHit associations.', 'independent_SimTrack_to_GenPart_rematch':'Raw SimTracks/SimVertices are absent from Nano; stored identities and generator ancestry are verified.', 'unretained_reconstruction_candidates':'Cleaning and vertex failures are outside this retained-pair denominator.'},
                  input_evidence={name:reference(path) for name,path in (('manifest',manifest_path),('performance',report_path),('evaluation_rows',evaluation_path),('source_inventory',inventory_path),('audit_code',__file__))})
    write_new(output / 'dominant_qcd.json', clean(result))
    write_new(output / 'dominant_qcd_event.json', clean(event))
    return result


def unknown_audit(dataset, evaluation_path, output):
    import label_audit
    manifest, reco, labels = label_audit.load_dataset(dataset)
    if manifest.get('complete') is not True or manifest.get('full_population') is not True or manifest.get('frozen_population_parity') is not True:
        raise ValueError('Unknown-label audit requires the complete parity-checked shared population')
    if manifest.get('evaluation_rows', {}).get('sha256') != digest(evaluation_path):
        raise ValueError('Shared cache belongs to another evaluated population')
    with np.load(evaluation_path, allow_pickle=False) as archive:
        evaluation = {key: archive[key] for key in archive.files}
    for key in ('ids', 'process', 'nuisance'):
        equal = (np.array_equal(reco[key], evaluation[key], equal_nan=True)
                 if reco[key].dtype.kind in 'fc' else np.array_equal(reco[key], evaluation[key]))
        if not equal:
            raise ValueError('Shared audit cache differs from evaluated population: ' + key)
    if not np.array_equal(labels['y'], evaluation['y']) or not np.array_equal(labels['reason'], evaluation['reason']):
        raise ValueError('Shared audit cache labels differ from completed evaluation')
    report = label_audit.build_report(dataset, manifest, reco, labels)
    y, weights = labels['y'], evaluation['evaluation_weight']
    report['per_process_weighted_completeness'] = {}
    for process in np.unique(reco['process']):
        mask = reco['process'] == process
        total = float(weights[mask].sum())
        unknown = float(weights[mask & (y < 0)].sum())
        report['per_process_weighted_completeness'][str(process)] = dict(sumw=total, unknown_sumw=unknown, unknown_signed_weight_fraction=unknown / total if total else None,
            negative_weight_pairs=int((weights[mask] < 0).sum()), weight_effective_pairs=float(total**2 / np.square(weights[mask]).sum()) if np.square(weights[mask]).sum() else None)
    report['per_process_heldout'] = {str(p):label_audit.completeness(y, (reco['process'] == p) & evaluation['heldout']) for p in np.unique(reco['process'])}
    report['per_signal_point'] = {str(p):label_audit.completeness(y, evaluation['point'] == p) for p in np.unique(evaluation['point']) if p}
    report['optional_match_audit'] = optional_match_audit(labels, reco['process'])
    events_path = Path(dataset) / 'events.npz'
    if events_path.exists() and 'source_index' in reco and 'event_index' in reco:
        if manifest.get('events_sha256') != digest(events_path):
            raise ValueError('All-event cache lacks a matching sealed digest')
        with np.load(events_path, allow_pickle=False) as archive:
            events = {key:archive[key] for key in archive.files}
        event_keys = np.column_stack((reco['source_index'], reco['event_index']))
        baseline = {}
        for process in np.unique(events['process']):
            pair_mask = reco['process'] == process
            total = int((events['process'] == process).sum())
            retained = len(np.unique(event_keys[pair_mask], axis=0))
            baseline[str(process)] = dict(recorded_nano_events=total,
                events_with_retained_pair=retained, retained_pairs=int(pair_mask.sum()),
                retained_pair_event_fraction=retained / total if total else None,
                events_with_known_pair=len(np.unique(event_keys[pair_mask & (y >= 0)], axis=0)),
                events_with_unknown_pair=len(np.unique(event_keys[pair_mask & (y < 0)], axis=0)))
        report['recorded_event_baseline'] = dict(per_process=baseline, events_cache=reference(events_path),
            caveat='Known-pair and unknown-pair event categories may overlap. These are recorded, sampled Nano events; events failing production or absent from this frozen subset are outside the denominator.')
    if 'truth_position_cm' in labels and 'vertex_position_cm' in reco:
        feature_names = list(reco['feature_names'])
        topology = np.array(['/'.join(format(float(v), '.6g') for v in pair) for pair in
            reco['X'][:, [feature_names.index('mu_min_topology'), feature_names.index('mu_max_topology')]]])
        performance_path = Path(evaluation_path).with_name('performance.json')
        performance = read(performance_path)
        if performance['artifact_sha256'].get(Path(evaluation_path).name) != digest(evaluation_path):
            raise ValueError('Evaluation NPZ no longer matches the completed report')
        report['common_origin_fit_residuals'] = residual_summary(reco['vertex_position_cm'], labels['truth_position_cm'], y, reco['process'], topology,
            z_errors=reco['X'][:, feature_names.index('pair_vzErr')], scores=evaluation['score'], threshold=performance['threshold'])
        report['common_origin_fit_residuals']['population'] = 'All retained pairs, including original training and validation events; descriptive label/fit audit only.'
        heldout = evaluation['heldout']
        report['heldout_common_origin_fit_residuals'] = residual_summary(reco['vertex_position_cm'][heldout], labels['truth_position_cm'][heldout], y[heldout], reco['process'][heldout], topology[heldout],
            z_errors=reco['X'][heldout, feature_names.index('pair_vzErr')], scores=evaluation['score'][heldout], threshold=performance['threshold'])
        report['heldout_common_origin_fit_residuals']['population'] = 'Saved test and unseen events only, using the completed evaluator heldout mask; original training and validation events excluded.'
        report['heldout_common_origin_fit_residuals']['excluded_training_or_validation_pairs'] = int((~heldout).sum())
    else:
        report['common_origin_fit_residuals'] = dict(available=False, reason='Shared cache lacks generator and reconstructed position vectors')
    report['recorded_denominator'] = 'All reconstructed ShiftDimuonVertex rows, before any BDT cut. This is conditional on upstream reconstruction and pair creation, not all generated pairs.'
    report['upstream_pair_selection'] = 'The producer forms pairs from cleaned reconstructed muons, requires compatible same-side origins and a valid line approach/fit, applies configured charge and compatibility requirements, then assigns non-sharing pairs by reconstructed quality. Rejected candidates and failed pair hypotheses are absent from Nano; their efficiency cannot be reconstructed from this retained population.'
    report['production_matching_floor'] = dict(minimum_leading_CSC_layers=3, minimum_purity=.75, layer_chi2_max=25., sim_to_gen_relative_momentum_max=1.e-5, sim_to_gen_vertex_distance_max_cm=.001,
        caveat='The producer already discards winning truth identities below three layers or purity 0.75. Relaxing the downstream label convention cannot recover absent identities; raw layer votes and SimTracks are unavailable in Nano.')
    report['topology_codes'] = {'0':'near endcap only','1':'near endcap and barrel','2':'both endcaps','3':'far endcap only','4':'unclassified'}
    report['missing_information'] = ['Raw layer-by-layer votes and SimHit/SimTrack collections are not stored in Nano.', 'Unretained muon candidates and failed pair hypotheses are not in the retained-pair cache.', 'The invalid hitGenPartIdx sentinel does not distinguish absent SimTracks, rejected hit association, unmatched truth, or multiple truth links without optional diagnostics.']
    report['audit_code'] = reference(__file__)
    classifier = Path(label_audit.__file__).resolve().parent
    report['semantic_source_references'] = {name:reference(path) for name,path in (
        ('label_convention', classifier / 'labels.py'),
        ('label_sensitivity', classifier / 'label_audit.py'),
        ('hit_association_policy', classifier.parent / 'ShiftMuonSegments/interface/HitTruthAssociation.h'),
        ('producer_branch_semantics', classifier.parent / 'ShiftMuonSegments/plugins/ShiftMuonTableProducer.cc'))}
    report['evaluation_rows'] = reference(evaluation_path)
    code_snapshot = output / ('audit_code_' + digest(__file__) + '.py')
    if not code_snapshot.exists():
        with code_snapshot.open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
    write_new(output / 'unknown_labels.json', clean(report))
    label_audit.make_figure(report, output)
    geometry_figure(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier-directory', type=Path, required=True)
    parser.add_argument('--evaluation-rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--performance-report', type=Path)
    parser.add_argument('--source-inventory', type=Path)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.classifier_directory.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        if not args.performance_report or not args.source_inventory:
            parser.error('--manifest needs --performance-report and --source-inventory')
        result = dominant_qcd(args.manifest, args.performance_report, args.evaluation_rows, args.source_inventory, args.output)
        print(json.dumps(dict(complete=True, event_id=result['event_id'], nominal_label=result['reason'], weight=result['normalization']['plotted_pair_weight'])), flush=True)
    if args.dataset:
        report = unknown_audit(args.dataset, args.evaluation_rows, args.output)
        print(json.dumps(dict(complete=True, unknown=report['overall'], per_process=report['per_process'])), flush=True)
    if not args.manifest and not args.dataset:
        parser.error('At least one of --manifest or --dataset is required')


if __name__ == '__main__':
    main()
