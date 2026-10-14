# Code style: AI-code. AI-generated historical-domain reconstructed cache extractor.
"""Cache complete historical binned Nano separately from current CMS IR5.

Reads pinned RECO119 features and truth labels in separate calls. No scoring,
fitting, event selection, cross-section rescaling or generator predictors.
Raw run/lumi/event IDs are local to each declared historical source context;
physical_parent_key is the split/group identity, including zero-pair parents.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from shift_vertex_signal_cache import digest, verified, read_json, require, frozen_readers, identities


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def parent_keys(ids, context):
    ids = identities(ids)
    return np.asarray([hashlib.sha256((context + ':' + ':'.join(str(int(i)) for i in row)).encode()).hexdigest()
                       for row in ids], dtype='U64')


def source_context(source, checkpoint):
    contract = checkpoint['contract']
    require(contract['SHIFT_LSS_FIELD_MODE'] == 'ir1_atlas_proxy', 'Unexpected historical geometry context')
    require(contract['PROCESS'] == source['generator_process'], 'Generator process context mismatch')
    # Stable physical source namespace; file relocation and chunking do not alter it.
    context = dict(process=source['process'], stratum=source['stratum'],
                   original_campaign=source['original_campaign'],
                   checkpoint_sha256=source['checkpoint']['sha256'])
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def extract_chunk(reco_arrays, truth_arrays, modules, source, index, offset, context):
    import awkward as ak
    features, labels = modules['features'], modules['labels']
    rich = modules['shift_vertex_reco_information']
    ids = identities(np.column_stack([ak.to_numpy(reco_arrays[name])
                                     for name in ('run', 'luminosityBlock', 'event')]))
    weights = ak.to_numpy(reco_arrays['genWeight']).astype(float)
    require(np.isfinite(weights).all() and np.all(weights == 1.), 'Historical unit GEN weight contract differs')
    counts = ak.to_numpy(reco_arrays['nShiftDimuonVertex']).astype(np.int32)
    require(np.all(counts >= 0), 'Negative pair count')
    reco = features.extract_reco({name: reco_arrays[name] for name in features.RECO_BRANCHES})
    extra = rich.extract_extension({name: reco_arrays[name] for name in rich.BRANCHES}, reco)
    require(reco['X'].shape == (int(counts.sum()), 50) and extra.shape == (len(reco['X']), 69),
            'Pair retention or frozen feature dimensions differ')
    for q, key in enumerate(rich.QUALITY):
        columns = extra[:, [q, len(rich.QUALITY) + len(rich.DETAIL) + 1 + q]]
        for operation, function in [('min', np.min), ('max', np.max)]:
            original = reco['X'][:, list(features.FEATURE_NAMES).index('mu_' + operation + '_' + key)]
            require(np.array_equal(function(columns, axis=1), original, equal_nan=True), 'Quality parity differs')
    event_index = reco['event_index'].copy()
    require(np.array_equal(reco['ids'], ids[event_index]), 'Pair identity alignment differs')
    y, reason = labels.extract_labels(truth_arrays, reco)
    label = dict(y=y, reason=reason, **labels.label_diagnostics(truth_arrays, reco))
    reco['X'] = np.column_stack((reco['X'], extra))
    reco['feature_names'] = np.asarray(tuple(features.FEATURE_NAMES) + tuple(rich.FEATURE_NAMES))
    flat = lambda name: ak.to_numpy(ak.flatten(reco_arrays['ShiftDimuonVertex_' + name]))
    topology = np.full(len(y), 'Other', dtype='U10')
    for lo, hi, name in [(0, 2, 'Near-Both'), (0, 3, 'Near-Far'), (2, 2, 'Both-Both'), (2, 3, 'Both-Far')]:
        topology[(flat('topologyMin') == lo) & (flat('topologyMax') == hi)] = name
    groups = parent_keys(ids, context)
    events = dict(ids=ids, native_weight=weights, pair_count=counts, physical_parent_key=groups,
                  event_index=np.arange(offset, offset + len(ids), dtype=np.int64))
    for key, value in [('process', source['process']), ('stratum', source['stratum']),
                       ('identity_context', source['identity_context'])]:
        events[key] = np.full(len(ids), value)
        reco[key] = np.full(len(y), value)
    events['source_index'] = np.full(len(ids), index, dtype=np.int32)
    reco.update(event_index=event_index + offset, source_index=np.full(len(y), index, dtype=np.int32),
                physical_parent_key=groups[event_index], native_weight=weights[event_index],
                evaluation_weight=weights[event_index], topology=topology,
                vertex_position_cm=np.column_stack([flat(axis) for axis in ('vx', 'vy', 'vz')]),
                good=(flat('isOS') == 1) & (flat('dcaValid') == 1))
    return reco, label, events


def extract(manifest_path, output, chunk_events=5000):
    import uproot
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema'] == 'shift-historical-binned-cache-inputs-v1'
            and manifest['sample_kind'] == 'simulation_only'
            and manifest['domain'] == 'historical_ir1_atlas_proxy', 'Wrong historical domain manifest')
    require(chunk_events > 0 and manifest['inputs'], 'No inputs or invalid chunk size')
    contexts, qualified = set(), []
    for source in manifest['inputs']:
        sidecar, checkpoint = read_json(source['sidecar']), read_json(source['checkpoint'])
        require(sidecar['complete'] is True and sidecar['status'] == 'validated'
                and not sidecar['missing_chunks'], 'Incomplete historical Nano merge')
        require(sidecar['sha256'] == source['nano']['sha256']
                and sidecar['output_bytes'] == source['nano']['bytes']
                and sidecar['campaign'] == source['original_campaign'], 'Nano/sidecar mismatch')
        context = source_context(source, checkpoint)
        require(context not in contexts, 'Repeated historical population')
        contexts.add(context)
        qualified.append((source, sidecar, context))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'source_snapshot'; snapshot.mkdir()
    sources = {}
    for name, ref in manifest['sources'].items():
        target = snapshot / name; shutil.copyfile(verified(ref), target)
        sources[name] = reference(target)
        require(sources[name]['sha256'] == ref['sha256'], 'Frozen reader copy mismatch')
    for path in (Path(__file__), Path(__file__).with_name('shift_vertex_signal_cache.py')):
        shutil.copyfile(path, snapshot / path.name)
    protocol = dict(manifest, chunk_events=chunk_events, truth_read_separately=True,
                    no_current_domain_pooling=True, predictor_count=119,
                    identity_policy='physical_parent_key includes historical process/stratum/campaign/checkpoint context and exact uint64 IDs')
    write(output / 'extraction_protocol.json', protocol)
    parts = [defaultdict(list) for _ in range(3)]; receipts = []; names = None
    started = time.monotonic(); benchmark_written = False
    with frozen_readers(sources) as modules:
        features, rich, labels = modules['features'], modules['shift_vertex_reco_information'], modules['labels']
        metadata = ('genWeight', 'ShiftDimuonVertex_topologyMin', 'ShiftDimuonVertex_topologyMax',
                    'ShiftDimuonVertex_isOS', 'ShiftDimuonVertex_dcaValid')
        branches = tuple(dict.fromkeys(features.RECO_BRANCHES + rich.BRANCHES + metadata))
        require(not any('score' in key.lower() or key in labels.TRUTH_BRANCHES for key in branches), 'Non-RECO feature read')
        for index, (source, sidecar, context) in enumerate(qualified):
            path = verified(source['nano']); before = path.stat(); event_seen = set()
            source_events = source_pairs = source_empty = 0; class_counts = defaultdict(int)
            with uproot.open(path, handler=uproot.source.file.MultithreadedFileSource, num_workers=1) as root:
                tree = root['Events']; require(tree.num_entries == sidecar['events'], 'Nano entries differ')
                for offset in range(0, tree.num_entries, chunk_events):
                    begin = time.monotonic(); stop = min(offset + chunk_events, tree.num_entries)
                    kwargs = dict(entry_start=offset, entry_stop=stop, library='ak', how=dict)
                    reco_arrays = tree.arrays(branches, **kwargs)
                    truth_arrays = tree.arrays(labels.TRUTH_BRANCHES, **kwargs)
                    reco, label, events = extract_chunk(reco_arrays, truth_arrays, modules, source, index, offset, context)
                    keys = set(map(tuple, events['ids']))
                    require(not keys.intersection(event_seen), 'Repeated parent within historical source')
                    event_seen.update(keys)
                    for part, arrays in zip(parts, (reco, label, events)):
                        for key, value in arrays.items():
                            if key == 'feature_names':
                                require(names is None or np.array_equal(names, value), 'Feature schema changed')
                                names = value
                            else: part[key].append(value)
                    source_events += len(events['ids']); source_pairs += len(label['y'])
                    source_empty += int(np.sum(events['pair_count'] == 0))
                    for name, value in [('genuine', 1), ('fake', 0), ('unknown', -1)]:
                        class_counts[name] += int(np.sum(label['y'] == value))
                    if not benchmark_written:
                        benchmark = dict(events=len(events['ids']), pairs=len(label['y']),
                                         seconds=time.monotonic()-begin, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                                         source_index=index, all119_features=True, truth_read_separately=True)
                        write(output/'first_chunk_benchmark.json', benchmark)
                        print(json.dumps(dict(benchmark=benchmark)), flush=True); benchmark_written = True
            require(source_events == sidecar['events'] and source_pairs == sidecar['counts']['vertices'], 'Full historical population differs')
            after = path.stat()
            require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino), 'Nano changed during read')
            verified(source['nano'])
            row = dict(source, index=index, identity_context_sha256=context, events=source_events,
                       pairs=source_pairs, zero_pair_events=source_empty, labels=dict(class_counts),
                       native_sumw=source_events, native_sumw2=source_events, full_parent_count_once=True)
            receipts.append(row); write(output/f'source_{index:02d}_readback.json', row)
            print(json.dumps(dict(point=source['stratum'],events=source_events,pairs=source_pairs,labels=dict(class_counts))),flush=True)
    refs = {}
    for kind, part in zip(('reco','labels','events'),parts):
        arrays = {key: np.concatenate(value) for key,value in part.items()}
        if kind == 'reco': arrays['feature_names'] = names
        if kind == 'events': require(len(set(arrays['physical_parent_key'])) == len(arrays['ids']), 'Historical group identity collision')
        path = output/(kind+'.npz'); np.savez_compressed(path, **arrays); refs[kind] = reference(path)
    total_events = sum(row['events'] for row in receipts); total_pairs = sum(row['pairs'] for row in receipts)
    require(total_events == manifest['expected_events'] and total_pairs == manifest['expected_pairs'], 'Manifest population totals differ')
    result = dict(schema='shift-historical-rich-cache-v1', complete=True, sample_kind='simulation_only',
                  domain='historical_ir1_atlas_proxy', sample_role='separate_historical_stress_study',
                  all_parent_events_retained=True, candidate_selection_applied=False,
                  no_current_domain_pooling=True, current_domain_independence_claim=False,
                  scoring_performed=False, feature_count=119, old_X50_exact=True,
                  weights='Native unit genWeight; evaluation_weight is identical. No cross-section mixture normalization.',
                  split_group_key='physical_parent_key', events=total_events,candidates=total_pairs,
                  inputs=receipts,cache=refs,source_snapshot=sources,
                  protocol=reference(output/'extraction_protocol.json'), extractor=reference(snapshot/Path(__file__).name),
                  adapter_dependency=reference(snapshot/'shift_vertex_signal_cache.py'),
                  elapsed_seconds=time.monotonic()-started,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    result.update({kind+'_sha256':ref['sha256'] for kind,ref in refs.items()})
    write(output/'inputs.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunk-events',type=int,default=5000)
    args=parser.parse_args(); result=extract(args.manifest,args.output,args.chunk_events)
    print(json.dumps({key:result[key] for key in ('complete','events','candidates','elapsed_seconds','peak_rss_mb')}))
