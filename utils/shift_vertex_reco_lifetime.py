# Code style: AI-code. AI-generated reconstructed displacement diagnostics.
"""Use fitted vertex coordinates and the fixed nominal production point.

The laboratory distance uses their Euclidean separation. The optional signed
proper-length proxy additionally uses fitted mass and momentum. Target spread
and fit uncertainties remain: neither quantity is a GEN lifetime measurement.
Negative proper-length estimates remain negative. Invalid inputs are NaN.
"""
import numpy as np


def reconstructed_vertex_distance_m(X, feature_names, target_cm):
    """Euclidean fitted-vertex distance from a fixed nominal target, in metres.

    This is a laboratory distance, without a momentum projection or boost
    correction. Invalid fitted coordinates remain NaN; distances are not clipped.
    """
    x, target = np.asarray(X), np.asarray(target_cm)
    names = list(feature_names)
    if x.ndim != 2 or x.shape[1] != len(names) or target.shape != (3,):
        raise ValueError('Misaligned reconstructed vertex-distance inputs')
    if not np.isfinite(target).all():
        raise ValueError('Invalid nominal target')
    position = x[:, [names.index('pair_' + axis) for axis in ('vx', 'vy', 'vz')]]
    displacement = position - target
    valid = np.isfinite(displacement).all(axis=1)
    result = np.full(len(x), np.nan)
    result[valid] = np.linalg.norm(displacement[valid], axis=1) / 100.
    return result


def reconstructed_proper_length_mm(X, feature_names, mass_gev, target_cm):
    x, mass, target = np.asarray(X), np.asarray(mass_gev), np.asarray(target_cm)
    names = list(feature_names)
    if x.ndim != 2 or x.shape[1] != len(names) or mass.shape != (len(x),) or target.shape != (3,):
        raise ValueError('Misaligned reconstructed proper-length inputs')
    if not np.isfinite(target).all() or np.any(mass < 0):
        raise ValueError('Invalid nominal target or reconstructed mass')
    get = lambda name: x[:, names.index('pair_' + name)]
    pt, phi, pz = get('pt'), get('phi'), get('pz')
    momentum = np.column_stack((pt*np.cos(phi), pt*np.sin(phi), pz))
    displacement = np.column_stack((get('vx'), get('vy'), get('vz'))) - target
    norm2 = np.square(momentum).sum(axis=1)
    valid = (norm2 > 0) & np.isfinite(momentum).all(axis=1) & np.isfinite(displacement).all(axis=1) & np.isfinite(mass)
    result = np.full(len(x), np.nan)
    # cm to mm; m*(L dot p)/|p|^2 is the signed projected proper length.
    result[valid] = 10*mass[valid]*np.sum(displacement[valid]*momentum[valid], axis=1)/norm2[valid]
    return result
