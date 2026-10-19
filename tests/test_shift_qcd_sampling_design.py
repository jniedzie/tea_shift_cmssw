# Code style: AI-code. AI-generated independent exact sampling-design checks.
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

spec = importlib.util.spec_from_file_location(
    'design', Path(__file__).parents[1]/'utils'/'shift_qcd_sampling_design.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


class MultiTargetDesignTests(unittest.TestCase):
    def test_separate_targets_prevent_starving_the_rare_tail(self):
        probabilities, report = design.constrained_probabilities(
            {'large_rate': [10000., 0., 0.], 'rare_tail': [0., 1., 0.]}, [1., 1., 1.])
        self.assertAlmostEqual(probabilities[0], .5, places=6)
        self.assertAlmostEqual(probabilities[1], .5, places=6)
        self.assertAlmostEqual(probabilities[2], .1, places=6)
        self.assertTrue(all(v <= 2.+1e-9 for v in report['target_second_moment_inflation'].values()))
        self.assertFalse(report['material_survival_or_detector_gain_validated'])

    def test_analytic_one_target_solution_and_cost(self):
        probabilities, report = design.constrained_probabilities({'tail': [1., 4.]}, [1., 1.])
        self.assertAlmostEqual(probabilities[0], .3, places=6)
        self.assertAlmostEqual(probabilities[1], .6, places=6)
        self.assertAlmostEqual(report['expected_remaining_cost_fraction'], .45, places=6)
        feasible_costs = [p+q for p in [i/500 for i in range(50, 501)]
                          for q in [i/500 for i in range(50, 501)] if 1/p+4/q <= 10]
        self.assertLessEqual(sum(probabilities), min(feasible_costs)+1e-6)

    def test_certainty_and_uncertainty_floors_are_preserved(self):
        probabilities, report = design.constrained_probabilities(
            {'tail': [1., 1., 0.]}, [1., 1., 1.], minimum_probability=[.1, .4, .2],
            protected=[True, False, False])
        self.assertEqual(probabilities[0], 1.)
        self.assertGreaterEqual(probabilities[1], .4)
        self.assertEqual(probabilities[2], .2)
        self.assertLessEqual(report['target_second_moment_inflation']['tail'], 2.+1e-9)

    def test_zero_targets_are_reported_as_unsupported(self):
        probabilities, report = design.constrained_probabilities({'mass50': [0., 0.]}, [1., 2.])
        self.assertEqual(probabilities, [.1, .1])
        self.assertEqual(report['unsupported_zero_moment_targets'], ['mass50'])
        self.assertFalse(report['target_second_moment_inflation'])

    def test_tiny_moments_have_the_same_allocation(self):
        expected, _ = design.constrained_probabilities({'tail': [1., 4.]}, [1., 1.])
        actual, _ = design.constrained_probabilities({'tail': [1e-250, 4e-250]}, [1., 1.])
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=10)

    def test_multi_target_dual_matches_an_independent_small_primal(self):
        rng = np.random.default_rng(190619)
        matrix = rng.uniform(.01, 2., size=(3, 8))
        costs = rng.uniform(.2, 3., size=8)
        probabilities, report = design.constrained_probabilities(
            dict(zip(('pt10', 'pt20', 'mass20'), matrix)), costs,
            maximum_inflation=2., protected=[True]+[False]*7)
        constraints = [{'type': 'ineq', 'fun': lambda p, a=a: 2.*a.sum()-np.sum(a/p)}
                       for a in matrix]
        primal = minimize(lambda p: float(costs @ p), np.ones(8), method='SLSQP',
                          bounds=[(1., 1.)]+[(.1, 1.)]*7, constraints=constraints,
                          options=dict(ftol=1e-11, maxiter=300))
        self.assertTrue(primal.success, primal.message)
        self.assertAlmostEqual(float(costs @ probabilities), primal.fun, places=5)
        self.assertTrue(all(v <= 2.+1e-9 for v in report['target_second_moment_inflation'].values()))

    def test_unit_bound_keeps_every_contributing_original(self):
        probabilities, _ = design.constrained_probabilities(
            {'tail': [1., 0., 1.]}, [1., 1., 1.], maximum_inflation=1.)
        self.assertEqual(probabilities, [1., .1, 1.])

    def test_invalid_or_zero_support_inputs_fail(self):
        for arguments in (
            ({'tail': [1.]}, [0.]), ({'tail': [math.nan]}, [1.]),
            ({'tail': [-1.]}, [1.]), ({'tail': [1., 2.]}, [1.]),
        ):
            with self.assertRaises(ValueError):
                design.constrained_probabilities(*arguments)
        for floor in (0., -1., math.nan, 1.1):
            with self.assertRaises(ValueError):
                design.constrained_probabilities({'tail': [1.]}, [1.], minimum_probability=floor)

    def test_previous_inclusion_enters_the_second_moment_once(self):
        values, old, new = [2., 10.], [.25, .5], [.5, .2]
        expected = design.sampled_design_moments(values, old, new)
        total = dict.fromkeys(expected, 0.)
        for bits in itertools.product((0, 1), repeat=2):
            chance = math.prod(p if bit else 1-p for bit, p in zip(bits, old))
            selected = [i for i, bit in enumerate(bits) if bit]
            measured = design.sampled_design_moments(
                [values[i] for i in selected], [old[i] for i in selected], [new[i] for i in selected])
            for key in total:
                total[key] += chance*measured[key]
        self.assertAlmostEqual(total['estimated_native_sum'], sum(values))
        self.assertAlmostEqual(total['estimated_new_second_moment'], sum(v*v/p for v, p in zip(values, new)))
        self.assertAlmostEqual(total['estimated_new_remaining_cost'], sum(new))
        self.assertAlmostEqual(expected['estimated_new_second_moment'], 1032.)

    def test_scout_cpu_bounds_the_gain_from_downstream_rejection(self):
        result = design.two_stage_cost(200., 400., .1)
        self.assertEqual(result['expected_cost'], 240.)
        self.assertEqual(result['event_count_speedup'], 2.5)
        self.assertEqual(design.two_stage_cost(200., 400., 0.)['event_count_speedup'], 3.)

    def test_competing_decays_have_the_exact_ordinary_pair_probability(self):
        result = design.competing_exponential_benchmark(100., 1., ordinary_fraction=1.)
        target = (1./101.)**2
        self.assertAlmostEqual(result['target_daughter_probability'], target)
        self.assertEqual(result['ordinary_second_moment'], result['proposal_second_moment'])
        self.assertEqual(result['second_moment_gain'], 1.)
        self.assertEqual(design.lifetime_mixture_log_weight([2., 4.], 100., 2., 1.), 0.)

    def test_optimal_lifetime_proposal_has_the_closed_second_moment(self):
        for parents in (1, 2):
            result = design.competing_exponential_benchmark(100., 1., ordinary_fraction=0., parents=parents)
            self.assertAlmostEqual(result['proposal_decay_mean'], 1./.51)
            self.assertAlmostEqual(result['proposal_second_moment'], (.01/.51)**(2*parents))
        self.assertGreater(result['second_moment_gain'], 600.)

    def test_whole_event_mixture_keeps_full_support_and_bounded_weights(self):
        for lengths in ([0., 0.], [1., 2.], [1000., 1000.]):
            log_weight = design.lifetime_mixture_log_weight(lengths, 100., 2., .1)
            self.assertTrue(math.isfinite(log_weight))
            self.assertLessEqual(log_weight, math.log(10.)+1e-10)
        result = design.competing_exponential_benchmark(1000., 1., ordinary_fraction=.1)
        self.assertGreater(result['second_moment_gain'], 50000.)
        self.assertFalse(result['physics_material_calibration'])

    def test_importance_draws_still_allow_physical_interaction_losses(self):
        # Independent check of the analytic integral with actual unchanged
        # exponential interaction draws, including events where parents die.
        rng = np.random.default_rng(190619)
        result = design.competing_exponential_benchmark(100., 1., ordinary_fraction=.1)
        count = 120000
        ordinary = rng.random(count) < .1
        decay = rng.exponential(size=(count, 2))
        decay *= np.where(ordinary, 100., result['proposal_decay_mean'])[:, None]
        interaction = rng.exponential(1., size=(count, 2))
        survived = np.all(decay < interaction, axis=1)
        total = decay.sum(axis=1)
        natural_log = -2*math.log(100.)-total/100.
        proposal_log = -2*math.log(result['proposal_decay_mean'])-total/result['proposal_decay_mean']
        log_weight = natural_log-np.logaddexp(math.log(.1)+natural_log, math.log(.9)+proposal_log)
        values = np.exp(log_weight)*survived
        error = math.sqrt(float(np.var(values, ddof=1))/count)
        self.assertLess(abs(float(values.mean())-result['target_daughter_probability']), 5.*error)
        self.assertLess(survived.mean(), .2)  # Most biased parents still fail.
        self.assertGreater(survived.mean(), .08)

    def test_invalid_lifetime_proposals_are_rejected(self):
        for means in ((0., 1.), (1., -1.), (math.nan, 1.), (1., math.inf)):
            with self.assertRaises(ValueError):
                design.competing_exponential_benchmark(*means)
        for lengths in ([], [-1., 2.], [math.nan, 2.]):
            with self.assertRaises(ValueError):
                design.lifetime_mixture_log_weight(lengths, 100., 2.)

    def test_original_phase_dependent_scores_keep_the_augmented_reference(self):
        # Original physical phase is uniform. The fixed global grid changes
        # every phase; only its probabilities depend on the original event.
        bins, accepted = 40, 2
        total = second = raw = 0.
        for original in range(bins):
            outcomes = [int((original+rotation) % bins < accepted) for rotation in range(bins)]
            probability, correction = design.fixed_angle_mixture(outcomes, .1)
            self.assertAlmostEqual(sum(probability), 1.)
            self.assertLessEqual(max(correction), 10.)
            for q, w, y in zip(probability, correction, outcomes):
                total += q*w*y/bins
                second += q*w*w*y/bins
                raw += q*y/bins
        target = accepted/bins
        self.assertAlmostEqual(total, target)
        self.assertAlmostEqual(raw, .9+.1*target)
        self.assertAlmostEqual(second, target*target/(.9+.1*target))

    def test_geometry_false_negatives_keep_positive_rotation_support(self):
        probability, correction = design.fixed_angle_mixture([1., 0., 0.], .1)
        self.assertGreater(probability[2], 0.)
        self.assertEqual(correction[2], 10.)
        self.assertAlmostEqual(sum(q*w for q, w in zip(probability, correction)), 1.)
        self.assertEqual(design.fixed_angle_mixture([0., 0.], .1), ([.5, .5], [1., 1.]))


class LifetimeComponentTests(unittest.TestCase):
    def test_unequal_single_and_pair_components_use_the_full_mixture(self):
        lengths = {'a': .4, 'b': 1.7, 'unbiased': 8.}
        means = {'a': 10., 'b': 20., 'unbiased': 30.}
        components = [dict(probability=.1, proposal_means={}),
                      dict(probability=.2, proposal_means={'a': 2.}),
                      dict(probability=.3, proposal_means={'b': 3.}),
                      dict(probability=.4, proposal_means={'a': 2., 'b': 3.})]
        density = lambda length, mean: math.exp(-length/mean)/mean
        natural = math.prod(density(lengths[p], means[p]) for p in means)
        proposal = sum(c['probability']*math.prod(
            density(lengths[p], c['proposal_means'].get(p, means[p])) for p in means)
            for c in components)
        actual = design.lifetime_components_log_weight(lengths, means, components)
        self.assertAlmostEqual(math.exp(actual), natural/proposal, places=14)
        # The ordinary latent third parent remains required, but cancels from
        # the density ratio because every component keeps its natural kernel.
        lengths['unbiased'] = 1000000.
        self.assertEqual(actual, design.lifetime_components_log_weight(lengths, means, components))

    def test_equal_mean_case_matches_the_existing_analytic_benchmark(self):
        for ordinary in (0., .1, 1.):
            for lengths in ([0., 0.], [1., 3.], [1000., 2000.]):
                value = design.lifetime_components_log_weight(
                    dict(enumerate(lengths)), {0: 100., 1: 100.},
                    [dict(probability=ordinary, proposal_means={}),
                     dict(probability=1.-ordinary, proposal_means={0: 2., 1: 2.})])
                expected = design.lifetime_mixture_log_weight(lengths, 100., 2., ordinary)
                self.assertAlmostEqual(value, expected, places=11)

    def test_absorbed_parent_lifetime_is_required_and_ordinary_weights_can_exceed_one(self):
        means = {'a': 100., 'absorbed': 100.}
        components = [dict(probability=.1, proposal_means={}),
                      dict(probability=.3, proposal_means={'a': 2.}),
                      dict(probability=.6, proposal_means={'a': 2., 'absorbed': 4.})]
        value = math.exp(design.lifetime_components_log_weight(
            {'a': 1000., 'absorbed': 4000.}, means, components))
        self.assertGreater(value, 1.)
        self.assertLessEqual(value, 10.*(1.+1e-14))
        with self.assertRaisesRegex(ValueError, 'every latent parent'):
            design.lifetime_components_log_weight({'a': 1000.}, means, components)
        with self.assertRaises(ValueError):
            design.lifetime_components_log_weight({'a': 1000., 'absorbed': None}, means, components)

    def test_frozen_components_normalize_with_physical_interactions_and_exact_second_moment(self):
        # Generate the whole-event component first, then all latent lifetimes.
        # Physical interactions are independent and may kill either parent.
        means, proposed, interactions = [2., 5.], [.7, 1.1], [.8, 1.4]
        components = [dict(probability=.2, proposal_means={}),
                      dict(probability=.25, proposal_means={0: proposed[0]}),
                      dict(probability=.15, proposal_means={1: proposed[1]}),
                      dict(probability=.4, proposal_means=dict(enumerate(proposed)))]
        rng = np.random.default_rng(20261019)
        count = 25000
        choices = rng.choice(len(components), count, p=[x['probability'] for x in components])
        scales = np.array([[c['proposal_means'].get(i, means[i]) for i in range(2)]
                           for c in components])
        lengths = rng.exponential(size=(count, 2))*scales[choices]
        weights = np.array([math.exp(design.lifetime_components_log_weight(
            dict(enumerate(row)), dict(enumerate(means)), components)) for row in lengths])
        def within_mc_error(values, expected):
            error = math.sqrt(float(np.var(values, ddof=1))/count)
            self.assertLess(abs(float(values.mean())-expected), 5.*error)
        within_mc_error(weights, 1.)
        survived = np.all(lengths < rng.exponential(interactions, size=(count, 2)), axis=1)
        physical_target = math.prod(i/(i+d) for i, d in zip(interactions, means))
        within_mc_error(weights*survived, physical_target)
        self.assertTrue(np.any(~survived))
        # Independent direct density integral, with no use of the helper under
        # test. This is E_q[(f/q)^2 I(physical decays before interactions)].
        def integrand(b, a):
            values = [a, b]
            natural = math.prod(math.exp(-x/m)/m for x, m in zip(values, means))
            proposal = sum(c['probability']*math.prod(
                math.exp(-values[i]/c['proposal_means'].get(i, means[i])) /
                c['proposal_means'].get(i, means[i]) for i in range(2)) for c in components)
            survival = math.exp(-sum(x/i for x, i in zip(values, interactions)))
            return (natural/proposal)*natural*survival if proposal else 0.
        second = quad(lambda a: quad(lambda b: integrand(b, a), 0., np.inf,
                                     epsabs=1e-10, epsrel=1e-8)[0],
                      0., np.inf, epsabs=1e-10, epsrel=1e-8)[0]
        within_mc_error(weights*weights*survived, second)
        self.assertLess(second, physical_target/.2)

    def test_missing_extra_invalid_or_observed_only_lifetimes_fail(self):
        good = [dict(probability=1., proposal_means={})]
        for lifetimes, means, components in (
                ({}, {}, good), ({'a': 1., 'extra': 1.}, {'a': 2.}, good),
                ({'a': math.nan}, {'a': 2.}, good), ({'a': -1.}, {'a': 2.}, good),
                ({'a': 1.}, {'a': 0.}, good), ({'a': True}, {'a': 2.}, good),
                ({'a': 1.}, {'a': 2.}, [dict(probability=1., proposal_means={'missing': 1.})]),
                ({'a': 1.}, {'a': 2.}, [dict(probability=1., proposal_means={'a': 0.})]),
                ({'a': 1.}, {'a': 2.}, [dict(probability=.5, proposal_means={})]),
                ({'a': 1.}, {'a': 2.}, [dict(probability=-1., proposal_means={})])):
            with self.subTest(lifetimes=lifetimes, means=means, components=components), self.assertRaises(ValueError):
                design.lifetime_components_log_weight(lifetimes, means, components)


if __name__ == '__main__':
    unittest.main()
