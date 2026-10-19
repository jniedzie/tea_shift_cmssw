# Code style: AI-code. AI-generated multi-target QCD sampling-design mathematics.
"""Allocate detector work against separate tail-precision constraints.

This is an offline design tool. It does not change production probabilities.
The constraints apply to the supplied conditional second moments; predicting
those moments does not establish material survival or detector acceptance.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import gammaln


def constrained_probabilities(second_moments, costs, maximum_inflation=2.,
                              minimum_probability=.1, protected=None):
    """Minimize expected remaining cost with a bound for every target.

    ``second_moments[target][i]`` is the predicted squared contribution of
    original collision i, including its physical source weight squared. Sum
    pair contributions within each collision before taking that square.
    Each target receives its own constraint ``sum(A_i/p_i) <= R*sum(A_i)``;
    a large inclusive rate cannot hide a poorly sampled tail. Positive floors
    and explicit certainty events protect uncertain/unmodelled categories.

    The returned bound concerns the model inputs, not unobserved physics.
    Existing source-inclusion probabilities are not changed by this function.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not len(costs) or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Remaining costs must be finite, positive and one-dimensional')
    if not isinstance(second_moments, dict) or not second_moments:
        raise ValueError('Name at least one target second moment')
    if not math.isfinite(maximum_inflation) or maximum_inflation < 1:
        raise ValueError('Every target variance bound must be finite and at least one')
    floor = np.broadcast_to(np.asarray(minimum_probability, dtype=float), costs.shape).copy()
    if not np.isfinite(floor).all() or np.any(floor <= 0) or np.any(floor > 1):
        raise ValueError('Every original event needs a finite probability in (0, 1]')
    if protected is not None:
        protected = np.asarray(protected)
        if protected.shape != costs.shape or protected.dtype != bool:
            raise ValueError('Certainty-event mask must be Boolean and match the costs')
        floor[protected] = 1.

    normalized, names, unsupported = [], [], []
    for name, values in second_moments.items():
        values = np.asarray(values, dtype=float)
        if (not isinstance(name, str) or not name or values.shape != costs.shape
                or not np.isfinite(values).all() or np.any(values < 0)):
            raise ValueError('Each named target needs finite, nonnegative matching moments')
        largest = float(values.max())
        if largest == 0:
            unsupported.append(name)
            continue
        # Normalize before summing to retain very small physical weights.
        scaled = values/largest
        normalized.append(scaled/scaled.sum())
        names.append(name)
    cost = costs/costs.max()
    cost = cost/cost.sum()
    matrix = np.asarray(normalized)
    solver_success = True
    feasibility_adjustment = 0.
    if not names:
        probability = floor
    elif maximum_inflation == 1:
        probability = np.where(np.any(matrix > 0, axis=0), 1., floor)
    elif np.all(matrix @ (1./floor) <= maximum_inflation):
        probability = floor
    else:
        def from_multipliers(multipliers):
            moment = multipliers @ matrix
            return np.minimum(1., np.maximum(floor, np.sqrt(moment/cost)))

        def negative_dual(multipliers):
            moment = multipliers @ matrix
            probability = from_multipliers(multipliers)
            value = np.sum(cost*probability + moment/probability)
            value -= maximum_inflation*np.sum(multipliers)
            gradient = maximum_inflation - matrix @ (1./probability)
            return -float(value), gradient

        solution = minimize(negative_dual, np.full(len(names), .1), jac=True,
                            method='L-BFGS-B', bounds=[(0., None)]*len(names),
                            options=dict(maxiter=500, ftol=1e-13, gtol=1e-10))
        solver_success = bool(solution.success)
        probability = from_multipliers(solution.x)
        # Numerical stopping must never leave an advertised constraint violated.
        # Move toward certainty only if needed; report the repair explicitly.
        if np.any(matrix @ (1./probability) > maximum_inflation):
            low, high = 0., 1.
            for _ in range(70):
                middle = (low+high)/2
                proposed = probability + middle*(1.-probability)
                if np.any(matrix @ (1./proposed) > maximum_inflation):
                    low = middle
                else:
                    high = middle
            feasibility_adjustment = high
            probability = probability + high*(1.-probability)

    predicted_inflation = (matrix @ (1./probability)) if names else []
    return probability.tolist(), dict(
        schema='shift-multitarget-tail-sampling-design-v1',
        target_second_moment_inflation=dict(zip(names, map(float, predicted_inflation))),
        maximum_inflation=float(maximum_inflation),
        unsupported_zero_moment_targets=unsupported,
        expected_remaining_cost_fraction=float(cost @ probability),
        minimum_probability=float(probability.min()), maximum_probability=float(probability.max()),
        solver_success=solver_success, feasibility_adjustment=float(feasibility_adjustment),
        bound_applies_to_supplied_model_only=True,
        material_survival_or_detector_gain_validated=False)


def sampled_design_moments(native_contributions, previous_probabilities, new_probabilities,
                           remaining_costs=None):
    """Evaluate a fixed new design on a previous probability sample.

    The old inclusion probability enters once in the Horvitz--Thompson second
    moment. Squaring the old W/p again would incorrectly enter it twice.
    Contributions must already be grouped by original independent collision.
    """
    values = np.asarray(native_contributions, dtype=float)
    previous = np.asarray(previous_probabilities, dtype=float)
    new = np.asarray(new_probabilities, dtype=float)
    if (values.ndim != 1 or previous.shape != values.shape or new.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(previous).all()
            or not np.isfinite(new).all() or np.any(previous <= 0) or np.any(previous > 1)
            or np.any(new <= 0) or np.any(new > 1)):
        raise ValueError('Need finite grouped contributions and aligned positive probabilities')
    costs = np.ones_like(values) if remaining_costs is None else np.asarray(remaining_costs, dtype=float)
    if costs.shape != values.shape or not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('Remaining costs must be finite, positive and aligned')
    return dict(estimated_native_sum=float(np.sum(values/previous)),
                estimated_new_second_moment=float(np.sum(values*values/(previous*new))),
                estimated_new_selection_variance=float(np.sum(values*values*(1./new-1.)/previous)),
                estimated_new_remaining_cost=float(np.sum(costs*new/previous)))


def two_stage_cost(simulation_cost, downstream_cost, retained_probability):
    """Cost of a material-aware scout, then optional digitization/reconstruction."""
    if (not all(math.isfinite(x) for x in (simulation_cost, downstream_cost, retained_probability))
            or simulation_cost < 0 or downstream_cost < 0 or not 0 <= retained_probability <= 1
            or simulation_cost+downstream_cost == 0):
        raise ValueError('Need nonnegative stage costs and a probability in [0, 1]')
    expected = simulation_cost + retained_probability*downstream_cost
    return dict(expected_cost=expected,
                baseline_cost=simulation_cost+downstream_cost,
                event_count_speedup=(simulation_cost+downstream_cost)/expected if expected else None,
                downstream_only_rejection_cannot_remove_simulation_cost=True)


def lifetime_mixture_log_weight(lengths, natural_mean, proposal_mean, ordinary_fraction=.1):
    """Exact f/q before transport for one whole-event lifetime mixture.

    A single recorded coin chooses either all natural exponential lifetimes
    or all proposal exponentials. Interactions remain physical and are not
    part of this weight. An interaction that removes a proposed daughter does
    not authorize redrawing it or conditioning on the daughter surviving.
    ``lengths`` are decay path lengths (or consistently scaled proper times).
    """
    lengths = np.asarray(lengths, dtype=float)
    if (lengths.ndim != 1 or not len(lengths) or not np.isfinite(lengths).all()
            or np.any(lengths < 0) or not math.isfinite(natural_mean) or natural_mean <= 0
            or not math.isfinite(proposal_mean) or proposal_mean <= 0
            or not math.isfinite(ordinary_fraction) or not 0 <= ordinary_fraction <= 1):
        raise ValueError('Need positive exponential means, finite decay lengths and a mixture in [0, 1]')
    total_length = float(lengths.sum())
    if not math.isfinite(total_length):
        raise ValueError('Total proposed lifetime exceeds finite numerical support')
    natural_log = -len(lengths)*math.log(natural_mean)-total_length/natural_mean
    proposal_log = -len(lengths)*math.log(proposal_mean)-total_length/proposal_mean
    if ordinary_fraction == 1:
        return 0.
    if ordinary_fraction == 0:
        return natural_log-proposal_log
    mixture_log = np.logaddexp(math.log(ordinary_fraction)+natural_log,
                               math.log1p(-ordinary_fraction)+proposal_log)
    return float(natural_log-mixture_log)


def competing_exponential_benchmark(decay_mean, interaction_mean, ordinary_fraction=.1, parents=2):
    """Analytic rare-decay benchmark, not a material calibration or QCD rate.

    Independent exponential interactions with their physical mean compete
    with each parent's decay. A daughter counts only if its parent decays
    first. The optimal lifetime-only proposal is q proportional to f*sqrt(S),
    where S(l)=exp(-l/interaction_mean). This changes sampling before transport;
    it never multiplies surviving events by an estimated material efficiency.
    """
    if (not math.isfinite(decay_mean) or decay_mean <= 0
            or not math.isfinite(interaction_mean) or interaction_mean <= 0
            or not math.isfinite(ordinary_fraction) or not 0 <= ordinary_fraction <= 1
            or type(parents) is not int or parents < 1):
        raise ValueError('Need positive means, a mixture in [0, 1], and a positive parent count')
    rate = interaction_mean/decay_mean
    proposal_rate = rate+.5
    proposal_mean = interaction_mean/proposal_rate
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(proposal_mean) or proposal_mean <= 0:
        raise ValueError('Mean ratio exceeds finite benchmark support')
    target = (rate/(rate+1.))**parents
    optimal_second = (rate/proposal_rate)**(2*parents)
    if target == 0 or optimal_second == 0:
        raise ValueError('Mean ratio exceeds representable benchmark probabilities')
    if ordinary_fraction == 1:
        second = target
        integration_error = 0.
    elif ordinary_fraction == 0:
        second = optimal_second
        integration_error = 0.
    else:
        # The sum of identical independent exponentials is Gamma(parents).
        # Its conditional partition among parents is identical in both mixture
        # components, so a one-dimensional integral gives the exact joint f/q.
        log_optimal_second = 2*parents*math.log(rate/proposal_rate)

        def integrand(total_length):
            if total_length == 0:
                return rate**(2*parents)/(ordinary_fraction*rate**parents +
                    (1.-ordinary_fraction)*proposal_rate**parents)/optimal_second if parents == 1 else 0.
            common = (parents-1)*math.log(total_length)-float(gammaln(parents))
            natural_log = parents*math.log(rate)+common-rate*total_length
            proposal_log = parents*math.log(proposal_rate)+common-proposal_rate*total_length
            mixture_log = np.logaddexp(math.log(ordinary_fraction)+natural_log,
                                       math.log1p(-ordinary_fraction)+proposal_log)
            return math.exp(2*natural_log-total_length-mixture_log-log_optimal_second)

        scaled_second, scaled_error = quad(integrand, 0., np.inf, epsabs=1e-11, epsrel=1e-10)
        second = optimal_second*scaled_second
        integration_error = optimal_second*scaled_error
    raw_proposal = (ordinary_fraction*target + (1.-ordinary_fraction)*
                    (proposal_rate/(proposal_rate+1.))**parents)
    variance = second-target*target
    return dict(natural_decay_mean=decay_mean, physical_interaction_mean=interaction_mean,
                proposal_decay_mean=proposal_mean, parents=parents,
                ordinary_fraction=ordinary_fraction, mixture_scope='one coin for the whole parent group',
                target_daughter_probability=target, raw_proposal_daughter_probability=raw_proposal,
                ordinary_second_moment=target, proposal_second_moment=second,
                optimal_zero_ordinary_second_moment=optimal_second,
                effective_events_per_independent_proposal=target*target/second,
                raw_daughter_gain=raw_proposal/target,
                second_moment_gain=target/second,
                estimator_variance_gain=(target-target*target)/variance if variance > 0 else None,
                quadrature_absolute_error=integration_error,
                physics_material_calibration=False, kinematics_branching_or_reconstruction_included=False)


def fixed_angle_mixture(scores, ordinary_fraction=.1):
    """Probabilities and exact weights on a fixed global rotation grid.

    Scores may depend on the original event and fixed geometry. Grid rotations
    themselves must be fixed independently of the original phase. The reference
    is a uniform draw on that grid, so each output weight is (1/M)/q_j. Its
    physical validity still requires global source rotational invariance.
    Do not reuse this discrete formula for point angles aligned separately to
    each event, or rotate only selected particles from a correlated collision.
    """
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores < 0) or not math.isfinite(ordinary_fraction)
            or not 0 < ordinary_fraction <= 1):
        raise ValueError('Need nonnegative geometry scores and a positive ordinary fraction')
    uniform = np.full(len(scores), 1./len(scores))
    if scores.max() == 0:
        probability = uniform
    else:
        scaled = scores/scores.max()
        probability = ordinary_fraction*uniform + (1.-ordinary_fraction)*scaled/scaled.sum()
    return probability.tolist(), (uniform/probability).tolist()
