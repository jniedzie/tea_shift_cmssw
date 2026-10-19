# Code style: AI-code. AI-generated independent high-count likelihood audit.
"""Profile exact Asimov Poisson counts with two Gaussian lognormal nuisances."""
import math

import numpy as np

from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

LOG_KAPPA = math.log(1.30)


def poisson_deviance_half(delta):
    if delta <= -1:
        return math.inf
    if abs(delta) < .001:
        # Avoid subtracting nearly equal numbers when s/b is around 1e-6.
        return math.fsum((-1) ** order * delta ** order / order for order in range(2, 10))
    return delta - math.log1p(delta)


def poisson_deviance_array(delta):
    result=np.empty_like(delta)
    small=np.abs(delta)<.001
    x=delta[small]
    result[small]=sum((-1)**order*x**order/order for order in range(2,10))
    result[~small]=delta[~small]-np.log1p(delta[~small])
    return result


def profile_q(cross_section_pb, signal_per_pb, backgrounds):
    backgrounds = np.asarray(backgrounds,dtype=float)
    signals = cross_section_pb*np.asarray(signal_per_pb,dtype=float)
    if (signals.shape != backgrounds.shape or np.any(backgrounds < 0)
            or np.any(signals < 0) or not np.all(np.isfinite(backgrounds + signals))):
        raise ValueError('Expected finite nonnegative signal/background cells')
    # A raw-MC zero is an explicit model assumption. Its Asimov observation is
    # zero and its exact Poisson deviance is 2*s, without any background ratio.
    positive = backgrounds > 0
    empty_signal = float(np.sum(signals[~positive]))
    backgrounds = backgrounds[positive]
    signal_ratios = signals[positive]/backgrounds

    def q_given_signal_nuisance(theta_signal):
        signal_scale = math.exp(LOG_KAPPA * theta_signal)

        def background_score(theta_background):
            background_shift = math.expm1(LOG_KAPPA * theta_background)
            deltas = background_shift + signal_ratios * signal_scale
            if np.any(deltas <= -1):
                return -math.inf
            score = math.fsum((backgrounds*deltas/(1.+deltas)).tolist())
            return LOG_KAPPA * math.exp(LOG_KAPPA * theta_background) * score + theta_background

        lower = -10.
        while background_score(lower) >= 0.:
            lower *= 2.
        theta_background = brentq(background_score, lower, 50., xtol=5.e-16, rtol=1.e-14)
        background_shift = math.expm1(LOG_KAPPA * theta_background)
        deltas = background_shift + signal_ratios * signal_scale
        return (2. * float(np.sum(backgrounds * poisson_deviance_array(deltas)))
                + 2. * empty_signal * signal_scale
                + theta_signal ** 2 + theta_background ** 2)

    profile = minimize_scalar(q_given_signal_nuisance, bounds=(-20., 0.), method='bounded',
                              options={'xatol': 1.e-10, 'maxiter': 500})
    if not profile.success:
        raise RuntimeError('Signal-nuisance profile did not converge')
    return min(profile.fun, q_given_signal_nuisance(0.))


def expected_limit(signal_per_pb, backgrounds, confidence_level):
    target = norm.ppf((1. + confidence_level) / 2.) ** 2
    upper = 1.
    while profile_q(upper, signal_per_pb, backgrounds) < target:
        upper *= 10.
        if upper > 1.e30:
            raise RuntimeError('Could not bracket expected cross-section limit')
    value = brentq(lambda sigma: profile_q(sigma, signal_per_pb, backgrounds) - target,
                   0., upper, xtol=1.e-8, rtol=1.e-10)
    return value, target
