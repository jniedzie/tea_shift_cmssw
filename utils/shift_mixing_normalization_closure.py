# Code style: AI-code. AI-generated source-event bootstrap of SS-normalized pair sums.
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

import shift_mixing as mixing


def plot_result(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    directory.mkdir(parents=True, exist_ok=True)
    output = directory / 'ss_normalized_Both_Both_closure.pdf'
    with PdfPages(output) as pdf:
        for graph, result in report['results'].items():
            edges = np.array(result['mass_edges'])
            figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True,
                gridspec_kw={'height_ratios': [3, 1]})
            for key, color, label in [('target', 'black', 'Real opposite-sign Both-Both'),
                                       ('prediction', 'tab:blue', 'Mixed, normalized with same-sign pairs')]:
                value = np.array(result[key])
                quantiles = np.array(result[key + '_quantiles'])
                axes[0].stairs(value, edges, color=color, label=label, linewidth=1.8)
                axes[0].fill_between(edges, np.r_[quantiles[1], quantiles[1, -1]],
                    np.r_[quantiles[3], quantiles[3, -1]], step='post', color=color, alpha=.18)
            target = np.array(result['target'])
            value = np.divide(result['prediction'], target, out=np.full(len(target), np.nan), where=target > 0)
            axes[1].stairs(value, edges, color='tab:blue')
            axes[1].axhline(1., color='black', linewidth=.8)
            axes[1].set(xlabel='Dimuon mass [GeV]', ylabel='Mixed / real', xlim=(0., 80.))
            axes[1].set_xscale('symlog', linthresh=4.)
            axes[0].set(ylabel='Cross section per bin [pb]', yscale='log')
            axes[0].legend(fontsize=9)
            axes[0].set_title('QCD Both-Both: same-sign normalization check; ' + graph)
            ratio = result['nominal_prediction_over_real']
            low, _, _, _, high = result['ratio_quantiles_025_16_50_84_975']
            axes[0].text(.02, .025,
                f'Total mixed / real = {ratio:.2f}; 95% bootstrap interval [{low:.2f}, {high:.2f}]\n'
                f'Same-sign control covers {100*result["coverage"]:.1f}% of real pair weight.\n'
                'Generator bin and event fold calibrated separately; 68% source-event bands.\n'
                'All-pairs diagnostic; real mass-tail statistics remain absent.',
                transform=axes[0].transAxes, fontsize=8)
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)
    (directory / 'publication.json').write_text(json.dumps(dict(path=str(output.resolve()),
        sha256=mixing.digest(output), numerical_summary_sha256=mixing.digest(Path(report['output']) / 'summary.json')), indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Check same-sign-normalized physical mixed-pair sums.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', type=Path)
    parser.add_argument('--graph', action='append', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('At least two bootstrap repetitions are required')
    INPUT = args.input
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'code').mkdir()
    events = list(mixing.iter_pool(INPUT / 'qcd_events.jsonl.gz'))
    provenance = mixing.validate_inventory(INPUT / 'qcd_events.jsonl.gz', INPUT / 'input_inventory.json', events)
    pools = mixing.pools_for(events)
    pool_index = np.zeros(len(events), dtype=int)
    for index, indices in enumerate(pools.values()):
        pool_index[indices] = index
    source_weight = np.array([row['detector_weight'] for row in events])
    pool_total = np.bincount(pool_index, weights=source_weight, minlength=len(pools))
    group_keys = sorted({(row['stratum'], row['partition']) for row in events})
    group_index = {key: index for index, key in enumerate(group_keys)}
    group_count = len(group_keys)
    edges = np.array([0., 2., 4., 10., 20., 40., 80., 200., 1000.])


    def read_rows(path):
        result = []
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if not row['both_endcaps']:
                    continue
                first, second = row['anchor_event'], row['donor_event']
                assert row['source_events'] == [events[first]['event_key'], events[second]['event_key']]
                assert pool_index[first] == pool_index[second]
                group = group_index[row['stratum'], row['partition']]
                result.append((first, second, row['weight'], group, row['kind'], row['pair']['mass']))
        return result


    def arrays(rows, kind, linearize=False):
        selected = [row for row in rows if row[4] == kind]
        first = np.array([row[0] for row in selected], dtype=int)
        second = np.array([row[1] for row in selected], dtype=int)
        weight = np.array([row[2] for row in selected])
        group = np.array([row[3] for row in selected], dtype=int)
        bins = np.searchsorted(edges, [row[5] for row in selected], side='right') - 1
        assert np.all((bins >= 0) & (bins < len(edges) - 1))
        donor_sum = pool_total[pool_index[first]] - source_weight[first]
        if linearize:
            assert np.all(first != second) and np.all(donor_sum > 0)
            scale = np.array([events[index]['physical_weight_pb'] / events[index]['detector_weight'] for index in first])
            # Ordered anchor/donor muon pairs occur twice; cross sections occur once.
            weight = weight * scale / (2. * donor_sum)
        return first, second, weight, group, bins, donor_sum


    def sums(data, weights):
        return np.bincount(data[3], weights=weights, minlength=group_count)


    def hist(data, weights):
        return np.bincount(data[4], weights=weights, minlength=len(edges) - 1)


    all_results = {}
    for graph in args.graph:
        folder = INPUT / graph
        assert json.loads((folder / 'report.json').read_text())['pool_sha256'] == provenance['pool_sha256']
        same, rows = read_rows(folder / 'same_event.jsonl.gz'), read_rows(folder / 'row_augmentation.jsonl.gz')
        real_os, real_ss = arrays(same, 'same_os'), arrays(same, 'same_ss')
        mixed_os, mixed_ss = arrays(rows, 'row_os', True), arrays(rows, 'row_ss', True)
        control, denominator = sums(real_ss, real_ss[2]), sums(mixed_ss, mixed_ss[2])
        available = (control > 0) & (denominator > 0)
        nominal_scale = np.divide(control, denominator, out=np.zeros(group_count), where=available)
        nominal_target = hist(real_os, real_os[2] * available[real_os[3]])
        nominal_prediction = hist(mixed_os, mixed_os[2] * nominal_scale[mixed_os[3]])
        predictions, targets = [], []
        lost = 0
        rng = np.random.default_rng(2026101507)
        for repetition in range(args.bootstrap):
            counts = rng.poisson(1., len(events))
            total = np.bincount(pool_index, weights=source_weight * counts, minlength=len(pools))
            def mixed_weights(data):
                donor_sum = total[pool_index[data[0]]] - counts[data[0]] * source_weight[data[0]]
                correction = np.divide(data[5], donor_sum, out=np.zeros(len(data[0])), where=donor_sum > 0)
                return data[2] * counts[data[0]] * counts[data[1]] * correction
            os_weights, ss_weights = mixed_weights(mixed_os), mixed_weights(mixed_ss)
            ss_control = sums(real_ss, real_ss[2] * counts[real_ss[0]])
            ss_denominator = sums(mixed_ss, ss_weights)
            if np.any(available & (ss_denominator <= 0)):
                lost += 1
                continue
            scale = np.divide(ss_control, ss_denominator, out=np.zeros(group_count), where=available)
            predictions.append(hist(mixed_os, os_weights * scale[mixed_os[3]]))
            targets.append(hist(real_os, real_os[2] * counts[real_os[0]] * available[real_os[3]]))
        prediction, target = np.asarray(predictions), np.asarray(targets)
        assert len(prediction) > 1
        difference = prediction - target
        nominal_ratio = nominal_prediction.sum() / nominal_target.sum()
        valid_ratio = target.sum(axis=1) > 0
        ratios = prediction.sum(axis=1)[valid_ratio] / target.sum(axis=1)[valid_ratio]
        result = dict(real_OS_rows=len(real_os[0]), real_SS_rows=len(real_ss[0]),
            mixed_OS_rows=len(mixed_os[0]), mixed_SS_rows=len(mixed_ss[0]),
            coverage=float((real_os[2] * available[real_os[3]]).sum() / real_os[2].sum()),
            available_controls=int(available.sum()), total_controls=group_count,
            nominal_predicted_OS_pb=float(nominal_prediction.sum()), nominal_real_OS_pb=float(nominal_target.sum()),
            nominal_prediction_over_real=nominal_ratio,
            ratio_quantiles_025_16_50_84_975=np.quantile(ratios, [.025, .16, .5, .84, .975]).tolist(),
            repetitions=args.bootstrap, valid_repetitions=len(predictions), lost_support_repetitions=lost,
            mass_edges=edges.tolist(), prediction=nominal_prediction.tolist(), target=nominal_target.tolist(),
            prediction_quantiles=np.quantile(prediction, [.025, .16, .5, .84, .975], axis=0).tolist(),
            target_quantiles=np.quantile(target, [.025, .16, .5, .84, .975], axis=0).tolist(),
            difference_quantiles=np.quantile(difference, [.025, .16, .5, .84, .975], axis=0).tolist(),
            controls={'/'.join(key): dict(available=bool(available[index]), real_SS_pb=float(control[index]),
                mixed_SS_pb=float(denominator[index])) for index, key in enumerate(group_keys)},
            input_sha256={name: mixing.digest(folder / name) for name in ('same_event.jsonl.gz', 'row_augmentation.jsonl.gz')})
        all_results[graph] = result
        print(graph, 'OS prediction/real', nominal_ratio, 'coverage', result['coverage'],
              'ratio intervals', result['ratio_quantiles_025_16_50_84_975'], 'lost', lost, flush=True)
    report = dict(schema='shift-SS-normalized-physical-pair-sum-diagnostic-v1', source=provenance,
        source_sha256=mixing.digest(Path(__file__)), output=str(OUT.resolve()), results=all_results,
        calibration='Separate SS normalization in each generator bin AND event fold; never mix source folds.',
        uncertainty='Joint source-event Poisson resamples, including physical donor-event normalization.',
        weight_conversion='W_i W_j N_eligible/K * (sigma/S_GEN)/(2*sum_{other donor originals in actual pool} W)',
        limitations=['All-pairs row model differs from the main best-candidate event-replacement model.',
                     'Calibration pools have positive support only when real SS and mixed SS exist; uncovered OS remains explicit.',
                     'The old 60080-event graph excludes the 78 baseline repairs; this is separate from the complete main shape study.',
                     'Intervals are conditional when a bootstrap loses mixed-SS support.',
                     'Sparse real tails cannot establish high-mass shape or rate closure.'],
        physics_valid=False, rate_validated=False, analysis_release=False)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copy2(__file__, OUT / 'code/shift_mixing_normalization_closure.py')
    if args.plots is not None:
        plot_result(report, args.plots)


if __name__ == '__main__':
    main()
