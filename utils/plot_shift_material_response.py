#!/usr/bin/env python3
# Code style: AI-code. AI-generated material-selection closure diagnostic.
"""Plot weighted OS Both-Both shapes and the conditional precision/CPU test."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap


def plot(summary, output):
    mechanism = summary.get('mechanism_study', False)
    reference_label = summary.get('reference_label', 'Reconstruction reference, weighted')
    scope = summary.get('source_population_scope',
        'Saved mu-enriched GEN; unchanged decays. Inclusive material rate unvalidated.')
    if 'declared_source_count' in summary and 'sampled_source_count' in summary:
        scope = (f"{summary['sampled_source_count']} sampled of {summary['declared_source_count']} "
                 f'declared source exposures. {scope}')
    design = summary.get('sampling_design', 'Sampling design not recorded in this older summary.')
    rate_scope = summary.get('reconstructed_tail_rate_scope',
        'Conditional test; inclusive material normalization remains unvalidated.')
    footer = '\n'.join(textwrap.fill(line, 145) for line in (scope, design, rate_scope))
    # Scheduler jobs should not create a shared AFS font cache.
    with tempfile.TemporaryDirectory(prefix='shift-material-plot-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(output) as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(11, 6.1))
            for ax, field, title in zip(axes, ('mass', 'pt'), ('Dimuon mass [GeV]', 'Dimuon pT [GeV]')):
                hist = summary['shapes'][field]
                edges = np.array(hist['edges'])
                centres, widths = (edges[1:]+edges[:-1])/2., np.diff(edges)
                for key, label, colour in [('reference', reference_label, '.35'),
                                            ('candidate', 'Frozen selection, weighted', '#1774b8')]:
                    values = np.array(hist[key])/widths
                    errors = np.sqrt(hist[key+'_variance'])/widths
                    ax.errorbar(centres, values, yerr=errors, xerr=widths/2., fmt='.',
                                color=colour, label=label, capsize=2)
                ylabel = ('GEN-mass-selected contribution [pb / GeV]' if mechanism
                          else 'Conditional cross section [pb / GeV]')
                ax.set(xlabel=title, ylabel=ylabel, xlim=(edges[0], edges[-1]))
                if any(value > 0 for key in ('reference', 'candidate') for value in hist[key]):
                    ax.set_yscale('log')
                else:
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
                    ax.text(.5, .5, 'No reconstructed OS Both-Both pairs',
                            ha='center', transform=ax.transAxes, fontsize=9)
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=8)
            fig.suptitle('OS Both-Both: GEN-mass-selected mechanism study' if mechanism
                         else 'OS Both-Both: material-aware reconstruction selection')
            fig.text(.5, .015, footer,
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .20, 1, .93))
            pdf.savefig(fig)
            plt.close(fig)
            result = summary['all_bins']
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.axis('off')
            lines = [f"Declared-cohort originals reconstructed: {result['originals']:,}"]
            if mechanism:
                lines.extend(['GEN-mass-selected mechanism study.',
                              'These cases test where true GEN-mass tail events are lost.',
                              'They cannot establish inclusive reconstructed-tail rates or a selection gain.', ''])
            else:
                lines.extend(['Gain > 1 means smaller weighted uncertainty for the same CPU time.',
                              f"{100*result['bootstrap'].get('confidence', .95):.3g}% intervals resample original events within pThat bins.", ''])
            for name, row in result['targets'].items():
                if mechanism:
                    lines.append(f"{name}: {row['positive_originals']} retained OS Both-Both tail originals")
                    continue
                gain = row['precision_per_CPU_gain_estimate']
                interval = row.get('precision_per_CPU_gain_interval', row.get('precision_per_CPU_gain_95pct_bootstrap'))
                value = 'undefined (no tail entries)' if gain is None else f'{gain:.3g}'
                bounds = '' if interval is None else f'  [{interval[0]:.3g}, {interval[1]:.3g}]'
                lines.extend([f'{name}: gain {value}{bounds}',
                              f"  {row['positive_originals']} tail originals; effective count {row['baseline_effective_count']:.2f}; "
                              + ('enough statistics' if row['statistics_sufficient'] else 'more statistics needed')])
            if not mechanism:
                lines.extend(['', 'Conditional gain supported in both primary tails: '
                              + ('yes' if result['tail_improvement_proven'] else 'no'),
                              'CPU uses measured batch averages and charges the full feature scan.'])
            lines.extend(['', textwrap.fill(rate_scope, 100),
                          'This does not qualify inclusive material normalization or a new reconstruction.'])
            ax.text(.02, .97, '\n'.join(lines), va='top', fontsize=10, linespacing=1.55)
            fig.suptitle('GEN-mass mechanism study: reconstructed outcomes' if mechanism
                         else 'Tail precision per CPU time')
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
