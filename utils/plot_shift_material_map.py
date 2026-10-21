#!/usr/bin/env python3
# Code style: AI-code. AI-generated geometry diagnostic; no event selection or rate weights.
"""Plot the shared material scorer's rays in angle or on their CMS endpoint plane."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def xy_coordinates(grid):
    """Project angular cell boundaries to the same plane already integrated.

    Each material value stays with its original ray; this changes coordinates,
    not the path endpoint, material integral or angular sampling resolution.
    """
    origin = np.asarray(grid['origin_cm'], dtype=float)
    eta = np.asarray(grid['eta_edges'], dtype=float)
    phi = np.asarray(grid['phi_edges'], dtype=float)
    if origin.shape != (3,):
        raise ValueError('XY map needs a three-coordinate source in cm')
    dz = float(grid['target_z_cm'])-origin[2]
    if (not np.all(np.isfinite(origin)) or not np.isfinite(dz)
            or dz == 0 or eta.ndim != 1 or phi.ndim != 1 or len(eta) < 2 or len(phi) < 2
            or not np.all(np.isfinite(eta)) or not np.all(np.isfinite(phi))
            or np.any(np.diff(eta) <= 0) or np.any(np.diff(phi) <= 0)
            or np.any(np.abs(eta) > 20) or np.any(eta == 0) or np.any(dz/np.tanh(eta) <= 0)):
        raise ValueError('XY map needs finite angular edges pointing to the integrated endpoint plane')
    radius = dz/np.sinh(eta)
    x = (origin[0]+np.cos(phi[:, None])*radius[None, :])/100.
    y = (origin[1]+np.sin(phi[:, None])*radius[None, :])/100.
    return x, y


def plot(data, output, view='eta-phi', quantity='all', xy_limit_m=10.):
    grid = data['angular_grid']
    eta = np.asarray(grid['eta_edges'])
    phi = np.asarray(grid['phi_edges'])/np.pi
    fields = [('interaction', 'interaction_depth_proxy', r'Nuclear interaction lengths: $\log_{10}(1+L/\lambda_I)$'),
              ('radiation', 'radiation_depth_proxy', r'Radiation lengths: $\log_{10}(1+L/X_0)$'),
              ('column', 'column_density_g_cm2', r'Material column: $\log_{10}(1+\Sigma/\mathrm{g\,cm^{-2}})$')]
    if quantity != 'all':
        fields = [entry for entry in fields if entry[0] == quantity]
    if not fields or view not in ('eta-phi', 'xy'):
        raise ValueError('Unknown material quantity or view')
    if not np.isfinite(xy_limit_m) or xy_limit_m <= 0:
        raise ValueError('XY display range must be finite and positive')
    coordinates = xy_coordinates(grid) if view == 'xy' else (eta, phi)
    figsize = (16,5) if view == 'eta-phi' and quantity == 'all' else (6*len(fields),6)
    fig, axes = plt.subplots(1,len(fields),figsize=figsize,squeeze=False,constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    for axis,(_,field,label) in zip(axes[0],fields):
        values = np.full((len(phi)-1,len(eta)-1),np.nan)
        for row in data['paths']:
            if row['complete']:
                value = row[field]
                if not np.isfinite(value) or value < 0:
                    raise ValueError('A complete ray has an invalid material integral')
                values[row['phi_index'],row['eta_index']] = np.log10(1.+value)
        mesh = axis.pcolormesh(*coordinates,np.ma.masked_invalid(values),cmap=cmap,rasterized=True)
        if view == 'xy':
            axis.set_xlabel('CMS x at entrance [m]')
            axis.set_ylabel('CMS y at entrance [m]')
            axis.set_aspect('equal')
            axis.set_xlim(-xy_limit_m,xy_limit_m)
            axis.set_ylim(-xy_limit_m,xy_limit_m)
            # Outside the sampled angular annulus, including its central hole,
            # there is no material measurement. Never draw it as vacuum.
            axis.set_facecolor('lightgray')
            axis.plot(0.,0.,marker='+',color='black',markersize=6)
        else:
            axis.set_xlabel(r'Direction $\eta$')
            axis.set_ylabel(r'Direction $\phi/\pi$')
            axis.set_yticks([-1.,-.5,0.,.5,1.])
        bar = fig.colorbar(mesh,ax=axis,label=label,orientation='horizontal',pad=.12)
        if view == 'xy' and field == 'radiation_depth_proxy':
            ticks = [value for value in (0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000)
                     if mesh.get_clim()[0] <= np.log10(1.+value) <= mesh.get_clim()[1]]
            bar.set_ticks(np.log10(1.+np.asarray(ticks)))
            bar.set_ticklabels([str(value) for value in ticks])
            bar.set_label(r'Integrated radiation lengths $X/X_0$ (logarithmic color scale)')
    start = grid['origin_cm'][2]/100.
    end = grid['target_z_cm']/100.
    if view == 'xy':
        title = (f'Material from collimator z={start:g} m to CMS entrance z={end:g} m\n'
                 'Straight rays; central CMS and magnetic bending excluded.\n'
                 'Gray: untraced directions.')
    else:
        title = (f'IR5 material along straight rays: source z={start:g} m to z={end:g} m\n'
                 'Gray: incomplete geometry coverage. Magnetic bending and central CMS are excluded.')
    fig.suptitle(title,fontsize=13)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--view', choices=('eta-phi', 'xy'), default='eta-phi')
    parser.add_argument('--quantity', choices=('all', 'radiation', 'interaction', 'column'), default='all')
    parser.add_argument('--xy-limit-m', type=float, default=10., help='Displayed CMS x/y range is +/- this distance')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.pdf':
        parser.error('Final geometry diagnostics must be PDF')
    opener = gzip.open if args.input.suffix == '.gz' else open
    with opener(args.input, 'rt') as stream:
        data = json.load(stream)
    plot(data, args.output, args.view, args.quantity, args.xy_limit_m)


if __name__ == '__main__':
    main()
