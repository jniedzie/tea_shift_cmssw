# Code style: AI-code. AI-generated independent high-count likelihood audit.
"""Profile exact Asimov Poisson counts with two Gaussian lognormal nuisances."""
import math

import numpy as np

from scipy.optimize import brentq, minimize, minimize_scalar
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


def profile_q_mc(signal, signal_variance, background_components, background_variances,
                 *, allow_zero_background=False):
    """Profile positive-weight, per-source effective-Poisson MC constraints.

    A yield y with MC variance v has n_eff=y*y/v and an independent multiplier
    beta, constrained by 2*n_eff*(beta-1-log(beta)). This preserves the first
    two MC moments; it is not an exact likelihood for arbitrary weighted events.
    Variance zero means a known template. Empty background MC cells are unknown
    and must be removed by the caller unless a literal zero-background scenario
    is explicitly requested. The two existing 30% global nuisances are retained.
    """
    signal=np.asarray(signal,dtype=float).ravel()
    signal_variance=np.asarray(signal_variance,dtype=float).ravel()
    background_components=np.atleast_2d(np.asarray(background_components,dtype=float))
    background_variances=np.atleast_2d(np.asarray(background_variances,dtype=float))
    if (signal.shape!=signal_variance.shape or background_components.shape!=background_variances.shape
            or background_components.shape[1:]!=signal.shape):
        raise ValueError('MC yields and variances must have matching bin shapes')
    yields=np.vstack((signal,background_components))
    variances=np.vstack((signal_variance,background_variances))
    if (np.any(yields<0) or np.any(variances<0) or not np.all(np.isfinite(yields+variances))
            or np.any((yields==0)&(variances>0))):
        raise ValueError('Expected finite nonnegative positive-weight MC moments')
    observed=np.sum(background_components,axis=0)
    if not allow_zero_background and np.any((observed==0)&(signal>0)):
        raise ValueError('Empty background MC is unknown, not a known zero rate')
    if not np.any(variances):
        return profile_q(1.,signal,observed)
    effective=np.divide(yields*yields,variances,out=np.full_like(yields,np.inf),where=variances>0)

    def objective(theta):
        scales=np.full(yields.shape[0],math.exp(LOG_KAPPA*theta[1]))
        scales[0]=math.exp(LOG_KAPPA*theta[0])
        current=yields*scales[:,None]
        a=np.divide(current,effective,out=np.zeros_like(current),where=np.isfinite(effective))
        total=current.sum(axis=0)
        # t=1-observed/lambda. Stationarity gives beta_i=1/(1+a_i*t).
        # The remaining one-dimensional equation is monotone in each bin.
        maximum=a.max(axis=0)
        low=np.divide(-1.+1.e-14,maximum,out=np.full_like(maximum,-1.e30),where=maximum>0)
        high=np.ones_like(low)
        denominator=total+np.sum(current*a,axis=0)
        t=np.divide(total-observed,denominator,out=np.zeros_like(total),where=denominator>0)
        t=np.minimum(high,np.maximum(low,t))
        deterministic=maximum==0
        single=((a>0).sum(axis=0)==1)&(observed>0)
        t[deterministic]=np.divide(total[deterministic]-observed[deterministic],total[deterministic],
                                  out=np.zeros_like(total[deterministic]),where=total[deterministic]>0)
        t[observed==0]=1.
        active=(observed>0)&~deterministic&~single
        for iteration in range(80):
            factors=1.+a*t
            prediction=np.sum(current/factors,axis=0)
            residual=(1.-t)*prediction-observed
            converged=np.abs(residual)<=2.e-12*np.maximum(observed,1.)
            remaining=active&~converged
            if not np.any(remaining):break
            low=np.where(remaining&(residual>0),t,low)
            high=np.where(remaining&(residual<=0),t,high)
            derivative=-np.sum(current*(1.+a)/(factors*factors),axis=0)
            newton=t-np.divide(residual,derivative,out=np.zeros_like(t),where=derivative!=0)
            proposal=np.where((newton>low)&(newton<high),newton,(low+high)/2.)
            t=np.where(remaining,proposal,t)
        else:
            raise RuntimeError('Per-bin MC likelihood profile did not converge')
        factors=1.+a*t
        beta=1./factors
        if np.any(single):
            # With one uncertain source, use its positive quadratic root.
            # This stays accurate when a small source would otherwise need
            # an enormous beta near the pole in the t parameterisation.
            columns=np.flatnonzero(single);indices=np.argmax(a[:,single],axis=0)
            c=current[indices,columns];n=effective[indices,columns]
            fixed=total[single]-c
            aa=c*(c+n);bb=(c+n)*fixed-c*observed[single]-n*c
            discriminant=np.hypot(bb,2.*np.sqrt(aa*n*fixed))
            root=np.empty_like(c);stable=bb>=0
            root[stable]=2.*n[stable]*fixed[stable]/(bb[stable]+discriminant[stable])
            root[~stable]=(-bb[~stable]+discriminant[~stable])/(2.*aa[~stable])
            beta[indices,columns]=root
        fitted=current*beta
        t[single]=1.-observed[single]/fitted[:,single].sum(axis=0)
        positive=observed>0
        delta=t[positive]/(1.-t[positive])
        data=2.*float(np.sum(observed[positive]*poisson_deviance_array(delta)))
        data+=2.*float(fitted[:,~positive].sum())
        uncertain=np.isfinite(effective)
        mc=2.*float(np.sum(effective[uncertain]*poisson_deviance_array((beta-1.)[uncertain])))
        value=data+mc+float(np.dot(theta,theta))
        gradient=np.array([2.*LOG_KAPPA*float(np.sum(fitted[0]*t))+2.*theta[0],
                           2.*LOG_KAPPA*float(np.sum(fitted[1:]*t))+2.*theta[1]])
        return value,gradient

    # Approximate local information rescales the two optimizer coordinates.
    # Large, well-known control yields can make the background-normalization
    # curvature many orders larger than the signal curvature. Unit steps in
    # theta would then send trial fits near a numerically singular gamma pole.
    # This changes coordinates only: objective, nuisance bounds and priors stay
    # identical. The scale uses the Poisson+MC variance at the nominal point.
    denominator=observed+variances.sum(axis=0)
    information=np.array([np.sum(np.divide(signal*signal,denominator,out=np.zeros_like(signal),where=denominator>0)),
                          np.sum(np.divide(observed*observed,denominator,out=np.zeros_like(observed),where=denominator>0))])
    information[0]+=signal[denominator==0].sum()
    coordinate_scale=np.sqrt(1.+LOG_KAPPA**2*information)

    def scaled_objective(coordinates):
        value,gradient=objective(coordinates/coordinate_scale)
        return value,gradient/coordinate_scale

    bounds=[(-20.*scale,20.*scale) for scale in coordinate_scale]
    fit=minimize(scaled_objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=bounds,
                 options={'ftol':1.e-13,'gtol':1.e-8,'maxiter':500,'maxls':40})
    if not fit.success and np.linalg.norm(fit.jac,np.inf)>1.e-5:
        raise RuntimeError('Global MC likelihood profile did not converge: '+str(fit.message))
    return max(0.,min(float(fit.fun),float(objective(np.zeros(2))[0])))
