#!/usr/bin/env python3
# Code style: AI-code. AI-generated material-selection closure diagnostic.
"""Plot weighted OS Both-Both shapes and the conditional precision/CPU test."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def plot(summary, output):
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
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
            for ax, field, title in zip(axes, ('mass', 'pt'), ('Dimuon mass [GeV]', 'Dimuon pT [GeV]')):
                hist = summary['shapes'][field]
                edges = np.array(hist['edges'])
                centres, widths = (edges[1:]+edges[:-1])/2., np.diff(edges)
                for key, label, colour in [('reference', 'Random reconstruction reference', '.35'),
                                            ('candidate', 'Frozen smart selection, weighted', '#1774b8')]:
                    values = np.array(hist[key])/widths
                    errors = np.sqrt(hist[key+'_variance'])/widths
                    ax.errorbar(centres, values, yerr=errors, xerr=widths/2., fmt='.',
                                color=colour, label=label, capsize=2)
                ax.set(xlabel=title, ylabel='Conditional cross section [pb / GeV]', yscale='log')
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=8)
            fig.suptitle('OS Both-Both: material-aware reconstruction selection')
            fig.text(.5, .015, 'Saved mu-enriched GEN; unchanged decays. Conditional test, inclusive material rate unvalidated.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, .93))
            pdf.savefig(fig)
            plt.close(fig)
            result = summary['all_bins']
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.axis('off')
            lines = [f"Independent originals reconstructed: {result['originals']:,}",
                     'Gain > 1 means smaller weighted uncertainty for the same CPU time.',
                     f"{100*result['bootstrap'].get('confidence', .95):.3g}% intervals resample original events within pThat bins.", '']
            for name, row in result['targets'].items():
                gain = row['precision_per_CPU_gain_estimate']
                interval = row.get('precision_per_CPU_gain_interval', row.get('precision_per_CPU_gain_95pct_bootstrap'))
                value = 'undefined (no tail entries)' if gain is None else f'{gain:.3g}'
                bounds = '' if interval is None else f'  [{interval[0]:.3g}, {interval[1]:.3g}]'
                lines.extend([f'{name}: gain {value}{bounds}',
                              f"  {row['positive_originals']} tail originals; effective count {row['baseline_effective_count']:.2f}; "
                              + ('enough statistics' if row['statistics_sufficient'] else 'more statistics needed')])
            lines.extend(['', 'Conditional gain supported in both primary tails: '
                          + ('yes' if result['tail_improvement_proven'] else 'no'),
                          'CPU uses measured batch averages and charges the full feature scan.',
                          'This does not qualify inclusive material normalization or a new reconstruction.'])
            ax.text(.02, .97, '\n'.join(lines), va='top', fontsize=10, linespacing=1.55)
            fig.suptitle('Tail precision per CPU time')
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
