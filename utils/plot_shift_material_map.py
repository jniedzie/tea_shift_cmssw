#!/usr/bin/env python3
# Code style: AI-code. AI-generated geometry diagnostic; no event selection or rate weights.
"""Plot the shared material scorer's angular grid, masking untraced directions."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.pdf':
        parser.error('Final geometry diagnostics must be PDF')
    opener = gzip.open if args.input.suffix == '.gz' else open
    with opener(args.input, 'rt') as stream:
        data = json.load(stream)
    grid = data['angular_grid']
    eta = np.asarray(grid['eta_edges'])
    phi = np.asarray(grid['phi_edges'])/np.pi
    fields = [('interaction_depth_proxy', r'Nuclear interaction lengths: $\log_{10}(1+L/\lambda_I)$'),
              ('radiation_depth_proxy', r'Radiation lengths: $\log_{10}(1+L/X_0)$'),
              ('column_density_g_cm2', r'Material column: $\log_{10}(1+\Sigma/\mathrm{g\,cm^{-2}})$')]
    fig, axes = plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    for axis,(field,label) in zip(axes,fields):
        values = np.full((len(phi)-1,len(eta)-1),np.nan)
        for row in data['paths']:
            if row['complete']:
                value = row[field]
                if not np.isfinite(value) or value < 0:
                    raise ValueError('A complete ray has an invalid material integral')
                values[row['phi_index'],row['eta_index']] = np.log10(1.+value)
        mesh = axis.pcolormesh(eta,phi,np.ma.masked_invalid(values),cmap=cmap,rasterized=True)
        axis.set_xlabel(r'Direction $\eta$')
        axis.set_ylabel(r'Direction $\phi/\pi$')
        axis.set_yticks([-1.,-.5,0.,.5,1.])
        fig.colorbar(mesh,ax=axis,label=label,orientation='horizontal',pad=.12)
    start = grid['origin_cm'][2]/100.
    end = grid['target_z_cm']/100.
    fig.suptitle(f'IR5 material along straight rays: source z={start:g} m to z={end:g} m\n'
                 'Gray: incomplete geometry coverage. Magnetic bending and central CMS are excluded.',fontsize=13)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__ == '__main__':
    main()
