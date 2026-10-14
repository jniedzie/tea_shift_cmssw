# Code style: AI-code. AI-generated complete-signal cache adapter.
"""Extract RECO119 and separate truth labels; never load models or select pairs.

CLI: --manifest INPUT.json --output NEW_DIRECTORY [--validate-only]
Manifest schema: shift-complete-signal-cache-inputs-v1, sample_kind:
simulation_only, sample_role: validation_only, sources: {filename: {path,
sha256}}, exclude_events: [{path, sha256}], points: [{point, receipt,
nano, gen_inventory, gen_audit}]. Every reference has path and sha256;
nano also has bytes. Sources must include features.py, labels.py,
portable_inference.py, bdt_performance.py, shift_vertex_reco_information.py.
Use frozen source snapshots. Exclusions must cover all earlier inspected
physical parent populations. Local/mounted EOS paths are supported.

Only shift-independent-signal-point-complete-v1 finalizer receipts are
supported. Partial production, missing GEN parents, changed files and reused
physical parents fail closed. --validate-only checks metadata without Nano
reads or output creation. It does not certify ROOT content.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

SOURCE_ORDER = ('features.py', 'labels.py', 'portable_inference.py',
                'bdt_performance.py', 'shift_vertex_reco_information.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verified(reference):
    path = Path(reference['path'])
    require(path.is_absolute(), 'Input paths must be absolute')
    require(digest(path) == reference['sha256'], 'Input hash mismatch: ' + str(path))
    if 'bytes' in reference:
        require(path.stat().st_size == reference['bytes'], 'Input size mismatch')
    return path


def read_json(reference):
    return json.loads(verified(reference).read_text())


def identities(values):
    """Reject float coercion and preserve event numbers above 2**53."""
    array = np.asarray(values)
    require(array.ndim == 2 and array.shape[1] == 3 and array.dtype.kind in 'iu',
            'Event identities must be an integer Nx3 array')
    require(np.all(array >= 0), 'Negative event identity')
    array = array.astype(np.uint64)
    require(len(set(map(tuple, array))) == len(array), 'Duplicate parent identity')
    return array


def validate_point(source, receipt, inventory, audit):
    point = source['point']
    require(receipt.get('schema') == 'shift-independent-signal-point-complete-v1',
            'Unsupported complete-point receipt schema')
    require(receipt.get('complete') is True and not receipt.get('partial', False)
            and receipt.get('production_complete', True) is True, 'Incomplete point')
    require(receipt['point'] == point and receipt['sample_role'] == 'validation_only',
            'Point or sample role mismatch')
    require(inventory.get('complete') is True and inventory['each_parent_ledger_counted_once']
            and inventory['sample_role'] == 'validation_only', 'Incomplete GEN inventory')
    parent = inventory['points'][point]
    union = receipt['union']
    require(union['complete'] is True and union['parent_gen_ledger_count'] == 1,
            'Incomplete or repeated parent ledger')
    require(union['point'] == point and union['parent_gen'] == parent,
            'Point receipt differs from GEN inventory')
    require(source['gen_audit']['sha256'] == parent['validation_sha256'], 'GEN audit not pinned by inventory')
    expected = identities(parent['event_ids'])
    require(np.array_equal(expected, identities(union['event_ids'])), 'GEN union identity mismatch')
    require(len(expected) == parent['events'] == receipt['events'] == union['events']
            == parent['full_gen_denominator']['accepted_events'] == audit['events'],
            'Incomplete full GEN denominator')
    require(audit.get('runtime_validated') is True, 'GEN runtime audit not qualified')
    require(np.array_equal(expected, identities(audit['event_ids'])), 'GEN audit identities differ')
    require(np.array_equal(expected, identities([row['id'] for row in audit['event_rows']])),
            'GEN audit row identities differ')
    weights = np.asarray([row['weight'] for row in audit['event_rows']], dtype=float)
    require(np.isfinite(weights).all(), 'Nonfinite GEN weight')
    require(np.isclose(weights.sum(), parent['native_sumw'], rtol=1e-12, atol=0)
            and np.isclose(np.square(weights).sum(), parent['native_sumw2'], rtol=1e-12, atol=0),
            'GEN weight sums differ')
    marker = receipt['score_histograms']['marker']
    require(marker['complete'] is True and marker['selection_applied'] is False
            and marker['events'] == len(expected), 'Incomplete or selected Nano')
    require(source['nano'] == marker['published']['nano'], 'Nano not linked to complete receipt')
    return parent, expected, weights, int(marker['pairs'])


def prepare_manifest(manifest):
    require(manifest.get('schema') == 'shift-complete-signal-cache-inputs-v1'
            and manifest.get('sample_kind') == 'simulation_only'
            and manifest.get('sample_role') == 'validation_only', 'Wrong simulation manifest contract')
    require(set(manifest['sources']) == set(SOURCE_ORDER), 'Wrong frozen source set')
    for reference in manifest['sources'].values():
        verified(reference)
    require(manifest.get('exclude_events'), 'Explicit previous-parent exclusion caches required')
    excluded = set()
    for reference in manifest['exclude_events']:
        with np.load(verified(reference), allow_pickle=False) as cache:
            excluded.update(map(tuple, identities(cache['ids'])))
    require(manifest['points'], 'No complete signal points')
    prepared, seen, points = [], set(), set()
    for source in manifest['points']:
        require(source['point'] not in points, 'Repeated signal point')
        points.add(source['point'])
        result = validate_point(source, read_json(source['receipt']),
                                read_json(source['gen_inventory']), read_json(source['gen_audit']))
        ids = set(map(tuple, result[1]))
        require(not ids.intersection(excluded), 'Signal overlaps previously inspected parents')
        require(not ids.intersection(seen), 'Physical parent reused across points')
        seen.update(ids)
        prepared.append((source, *result))
    return prepared


@contextmanager
def frozen_readers(sources):
    """Pin every project dependency without retaining global import overrides."""
    previous, loaded = {}, {}
    try:
        for filename in SOURCE_ORDER:
            name = Path(filename).stem
            previous[name] = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, verified(sources[filename]))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def validate_event_readback(ids, weights, counts, expected, gen_weights, expected_pairs):
    ids = identities(ids)
    require(np.array_equal(ids, expected), 'Nano parent identities/order differ')
    require(np.array_equal(weights, gen_weights), 'Nano native weights differ from GEN')
    counts = np.asarray(counts)
    require(counts.shape == (len(ids),) and counts.dtype.kind in 'iu'
            and np.all(counts >= 0) and int(counts.sum()) == expected_pairs,
            'Nano candidate count differs')
    return ids


def extract_point(source, parent, expected, gen_weights, expected_pairs, modules, index):
    import awkward as ak
    import uproot
    features, labels = modules['features'], modules['labels']
    rich, performance = modules['shift_vertex_reco_information'], modules['bdt_performance']
    metadata = ('genWeight', 'ShiftDimuonVertex_topologyMin', 'ShiftDimuonVertex_topologyMax',
                'ShiftDimuonVertex_isOS', 'ShiftDimuonVertex_dcaValid')
    reco_branches = tuple(dict.fromkeys(features.RECO_BRANCHES + rich.BRANCHES + metadata))
    require(not any('score' in name.lower() or 'genpart' in name.lower() or 'truth' in name.lower()
                    for name in reco_branches), 'Non-RECO branch in feature read')
    path = verified(source['nano'])
    before = path.stat()
    with uproot.open(path, handler=uproot.source.file.MultithreadedFileSource, num_workers=1) as root:
        tree = root['Events']
        require(tree.num_entries == len(expected), 'Nano entry count differs')
        reco_arrays = tree.arrays(reco_branches, library='ak', how=dict)
        # Truth is read in a separate call and never passed to feature extractors.
        truth_arrays = tree.arrays(labels.TRUTH_BRANCHES, library='ak', how=dict)
    ids = np.column_stack([ak.to_numpy(reco_arrays[k]) for k in ('run', 'luminosityBlock', 'event')])
    weights = ak.to_numpy(reco_arrays['genWeight']).astype(float)
    counts = ak.to_numpy(reco_arrays['nShiftDimuonVertex'])
    ids = validate_event_readback(ids, weights, counts, expected, gen_weights, expected_pairs)
    reco = features.extract_reco({k: reco_arrays[k] for k in features.RECO_BRANCHES})
    extra = rich.extract_extension({k: reco_arrays[k] for k in rich.BRANCHES}, reco)
    require(reco['X'].shape == (expected_pairs, 50) and extra.shape == (expected_pairs, 69),
            'Frozen feature contract is not 50+69')
    for q, key in enumerate(rich.QUALITY):
        columns = extra[:, [q, len(rich.QUALITY) + len(rich.DETAIL) + 1 + q]]
        for operation, function in [('min', np.min), ('max', np.max)]:
            column = list(features.FEATURE_NAMES).index('mu_' + operation + '_' + key)
            require(np.array_equal(function(columns, axis=1), reco['X'][:, column], equal_nan=True),
                    'Associated quality differs from original features')
    require(np.array_equal(reco['ids'], ids[reco['event_index']]), 'Pair parent alignment differs')
    reco['X'] = np.column_stack((reco['X'], extra))
    reco['feature_names'] = np.asarray(tuple(features.FEATURE_NAMES) + tuple(rich.FEATURE_NAMES))
    y, reason = labels.extract_labels(truth_arrays, reco)
    associated = performance.signal_pairs(truth_arrays, reco)
    label = dict(y=y, reason=reason, **labels.label_diagnostics(truth_arrays, reco),
                 signal_associated=associated, genuine_signal_associated=(y == 1) & associated)
    flatten = lambda name: ak.to_numpy(ak.flatten(reco_arrays['ShiftDimuonVertex_' + name]))
    topology = np.full(expected_pairs, 'Other', dtype='U10')
    for low, high, name in [(0, 2, 'Near-Both'), (0, 3, 'Near-Far'), (2, 2, 'Both-Both'), (2, 3, 'Both-Far')]:
        topology[(flatten('topologyMin') == low) & (flatten('topologyMax') == high)] = name
    reco.update(source_index=np.full(expected_pairs, index, dtype=np.int32),
                process=np.full(expected_pairs, 'signal'), point=np.full(expected_pairs, source['point']),
                stratum=np.full(expected_pairs, source['point']), native_weight=weights[reco['event_index']],
                evaluation_weight=weights[reco['event_index']], topology=topology,
                vertex_position_cm=np.column_stack([flatten(axis) for axis in ('vx', 'vy', 'vz')]),
                good=(flatten('isOS') == 1) & (flatten('dcaValid') == 1))
    events = dict(ids=ids, native_weight=weights, pair_count=counts.astype(np.int32),
                  source_index=np.full(len(ids), index, dtype=np.int32),
                  process=np.full(len(ids), 'signal'), point=np.full(len(ids), source['point']))
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Nano changed during read')
    verified(source['nano'])
    receipt = dict(source, source_index=index, events=len(ids), pairs=expected_pairs,
                   zero_pair_events=int(np.sum(counts == 0)), parent_gen=parent,
                   parent_gen_ledger_count=1, native_sumw=float(weights.sum()),
                   native_sumw2=float(np.square(weights).sum()),
                   labels={name: int(np.sum(y == value)) for name, value in
                           [('genuine', 1), ('accidental', 0), ('unknown', -1)]})
    return reco, label, events, receipt


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def extract_cache(manifest_path, output, validate_only=False):
    manifest = json.loads(manifest_path.read_text())
    prepared = prepare_manifest(manifest)
    if validate_only:
        return {'metadata_validated': True, 'nano_readback_performed': False,
                'points': len(prepared), 'events': sum(len(row[2]) for row in prepared)}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    sources = {}
    for name, reference in manifest['sources'].items():
        target = snapshot / name
        shutil.copyfile(verified(reference), target)
        sources[name] = {'path': str(target.resolve()), 'sha256': digest(target)}
        require(sources[name]['sha256'] == reference['sha256'], 'Source snapshot changed')
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    protocol = dict(manifest, complete_point_inputs=[dict(source, source_receipt=source['receipt'])
                                                   for source in manifest['points']])
    write_json(output / 'extraction_protocol.json', protocol)
    parts = [defaultdict(list) for _ in range(3)]
    receipts, names = [], None
    with frozen_readers(sources) as modules:
        for index, row in enumerate(prepared):
            result = extract_point(*row, modules, index)
            for target, arrays in zip(parts, result[:3]):
                for name, value in arrays.items():
                    if name == 'feature_names':
                        require(names is None or np.array_equal(names, value), 'Feature schema changed')
                        names = value
                    else:
                        target[name].append(value)
            receipts.append(result[3])
    refs = {}
    for kind, part in zip(('reco', 'labels', 'events'), parts):
        arrays = {key: np.concatenate(value) for key, value in part.items()}
        if kind == 'reco':
            arrays['feature_names'] = names
            audit = {str(name): {'finite': int(np.isfinite(arrays['X'][:, i]).sum()),
                                'unique_finite': int(len(np.unique(arrays['X'][np.isfinite(arrays['X'][:, i]), i])))}
                     for i, name in enumerate(names)}
            write_json(output / 'feature_audit.json', audit)
        path = output / (kind + '.npz')
        np.savez_compressed(path, **arrays)
        refs[kind] = {'path': str(path.resolve()), 'sha256': digest(path), 'bytes': path.stat().st_size}
    result = dict(schema='shift-complete-signal-rich-cache-v1', complete=True,
                  sample_kind='simulation_only', sample_role='validation_only',
                  model_selection_allowed=False, scoring_performed=False, score_branches_read=False,
                  all_parent_events_retained=True, candidate_selection_applied=False,
                  feature_count=119, old_X50_exact=True, inputs=receipts, cache=refs,
                  events=sum(row['events'] for row in receipts), candidates=sum(row['pairs'] for row in receipts),
                  source_snapshot=sources, excluded_parent_overlap=0,
                  event_index_convention='Per source; join physical parents by exact uint64 ids',
                  normalization='Native weights once; full GEN ledger once per point; no physical-rate claim',
                  protocol={'path': str((output / 'extraction_protocol.json').resolve()),
                            'sha256': digest(output / 'extraction_protocol.json')},
                  extractor={'path': str((snapshot / Path(__file__).name).resolve()),
                             'sha256': digest(snapshot / Path(__file__).name)})
    result.update({kind + '_sha256': ref['sha256'] for kind, ref in refs.items()})
    result['extra_feature_groups'] = {str(name): 'event' if str(name).startswith('event_') else 'muon'
                                    for name in names[50:]}
    write_json(output / 'inputs.json', result)
    write_json(output / 'inputs_resolved.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    result = extract_cache(args.manifest, args.output, args.validate_only)
    print(json.dumps({key: result[key] for key in ('events', 'candidates', 'complete',
                                                 'metadata_validated', 'nano_readback_performed') if key in result}))


if __name__ == '__main__':
    main()
