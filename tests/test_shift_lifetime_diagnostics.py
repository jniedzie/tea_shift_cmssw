# Code style: AI-code. AI-generated numerical tests for lifetime diagnostics.
"""Check the physical length/coupling conversion without external packages."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "utils/shift_lifetime_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("shift_lifetime_diagnostics", SOURCE)
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)


class CouplingLengthTest(unittest.TestCase):
    def setUp(self):
        self.spectral = SimpleNamespace(width=lambda state, mass, epsilon: mass * epsilon**2)
        self.electroweak = SimpleNamespace(
            PERTURBATIVE_MIN_MASS_GEV=12.,
            proper_decay_length_mm=lambda width: 1.973269804e-13 / width,
            # An epsilon-dependent correction checks that inversion uses the
            # full model instead of a blindly squared coupling ratio.
            fermion_widths_lo=lambda mass, epsilon: {
                "e": mass * epsilon**2 * (1 + 1.e8 * epsilon**2)})

    def test_spectral_width_has_correct_length_units(self):
        expected = 1.973269804e-16 / (1. * 1.e-12)
        actual = DIAGNOSTICS.coupling_proper_length_m(1., 1.e-6, self.spectral, self.electroweak)
        self.assertAlmostEqual(actual, expected)

    def test_inversion_closes_both_models(self):
        for mass in [1., 50., 90.]:
            for epsilon in [1.e-9, 1.e-7, 5.e-4]:
                length = DIAGNOSTICS.coupling_proper_length_m(mass, epsilon, self.spectral, self.electroweak)
                actual = DIAGNOSTICS.coupling_for_proper_length(mass, length, self.spectral, self.electroweak)
                self.assertTrue(math.isclose(actual, epsilon, rel_tol=1.e-12))

    def test_invalid_lengths_and_parameters_fail(self):
        for value in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                DIAGNOSTICS.coupling_for_proper_length(15., value, self.spectral, self.electroweak)
            with self.assertRaises(ValueError):
                DIAGNOSTICS.coupling_proper_length_m(15., value, self.spectral, self.electroweak)

    def test_production_mean_lab_flight_keeps_boost(self):
        reference = {"references": [dict(mass_gev=15., epsilon=1.e-7, mean_beta_gamma=80.,
                                       width_authority=dict(ctau_mm=100., width_scope="native-pilot"))]}
        rows = DIAGNOSTICS.reference_coupling_rows(reference, [5., 100., 1000.])
        for row in rows:
            self.assertEqual(row["proper_ctau_m"] * 80., row["requested_mean_lab_flight_m"])
            self.assertTrue(math.isclose(.1 * (1.e-7 / row["epsilon"])**2,
                                         row["proper_ctau_m"], rel_tol=1.e-12))
            self.assertEqual(row["proper_lifetime_s"], row["proper_ctau_m"] / 299792458.)

    def test_additional_mass_references_keep_their_physical_epsilon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model.py'
            model.write_text('# test model is only loaded, not evaluated\n')
            def reference(mass):
                return dict(schema='shift-dark-photon-grid-reference-v1',common=[],cp5=[],references=[
                    dict(mass_gev=mass,epsilon=1.e-7,mean_beta_gamma=80.,
                         width_authority=dict(mass_gev=mass,epsilon=1.e-7,
                                              ctau_mm=1.,total_width_gev=1.973269804e-13))])
            first, second = root/'first.json', root/'second.json'
            first.write_text(json.dumps(reference(15.)))
            second.write_text(json.dumps(reference(2.)))
            fake = SimpleNamespace(Model=lambda name:None)
            with patch.dict('sys.modules',{'darkcast':fake}):
                merged,_,_ = DIAGNOSTICS.load_coupling_widths(first,root,model,[second])
            self.assertEqual([row['mass_gev'] for row in merged['references']],[2.,15.])
            mismatched = reference(2.)
            mismatched['references'][0]['width_authority']['epsilon'] = 2.e-7
            second.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(ValueError,'another mass or epsilon'):
                DIAGNOSTICS.load_coupling_widths(first,root,model,[second])


if __name__ == "__main__":
    unittest.main()
