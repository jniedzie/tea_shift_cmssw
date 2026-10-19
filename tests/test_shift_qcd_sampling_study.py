# Code style: AI-code. AI-generated tests for the GEN sampling study.
import importlib.util
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile


MODULE = Path(__file__).resolve().parents[1] / "utils" / "shift_qcd_sampling_study.py"
SPEC = importlib.util.spec_from_file_location("qcd_sampling_study", MODULE)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class SamplingStudyTest(unittest.TestCase):
    def test_parent_species_distinguishes_heavy_hadrons_and_quarkonia(self):
        expected = {211: "pion", -321: "kaon", 130: "kaon", 411: "charm",
                    4122: "charm", 511: "bottom", 5122: "bottom", 443: "resonance",
                    553: "resonance", 22: "other", 0: "unknown"}
        self.assertEqual({pdg: study.parent_kind(pdg) for pdg in expected}, expected)

    def test_parent_vertex_and_flight_are_read_rather_than_assumed_at_target(self):
        tree = SimpleNamespace(nGenPart=3, GenPart_pdgId=[211, -13, -13],
            GenPart_status=[2, 2, 1], GenPart_pt=[3., 2., 1.],
            GenPart_phi=[0., 0., 0.], GenPart_pz=[-4., -4., -4.],
            GenPart_mass=[.14, .105, .105], GenPart_vx=[1., 4., 4.],
            GenPart_vy=[2., 6., 6.], GenPart_vz=[300., 288., 288.],
            GenPart_genPartIdxMother=[-1, 0, 1])
        muons = study.extract_muons(tree)
        self.assertEqual(len(muons), 1)
        self.assertEqual(muons[0]["parent_index"], 0)
        self.assertEqual(muons[0]["parent_start_cm"], [1., 2., 300.])
        self.assertEqual(muons[0]["parent_flight_cm"], 13.)
        self.assertEqual(muons[0]["parent_p_GeV"], 5.)

    def test_cyclic_ancestry_rejected(self):
        tree = SimpleNamespace(nGenPart=2, GenPart_pdgId=[13, 13],
            GenPart_status=[1, 2], GenPart_pt=[1., 1.], GenPart_phi=[0., 0.],
            GenPart_pz=[-4., -4.], GenPart_mass=[.105, .105],
            GenPart_vx=[0., 0.], GenPart_vy=[0., 0.], GenPart_vz=[1000., 1000.],
            GenPart_genPartIdxMother=[1, 0])
        with self.assertRaisesRegex(ValueError, "Cycle"):
            study.extract_muons(tree)

    def test_backward_line_is_not_counted_as_forward_detector_support(self):
        muon = dict(x_cm=0., y_cm=0., z_cm=1000., px=1., py=0., pz=2.)
        self.assertIsNone(study.line_radius(muon, 700.))
        muon["pz"] = -2.
        self.assertEqual(study.line_radius(muon, 700.), 150.)

    def test_reconstructed_tail_not_inferred_from_generator_mass(self):
        first = dict(index=0, pdg_id=13, p=30., pt=.1, pz=-math.sqrt(900. - .01),
            px=.1, py=0., energy=math.hypot(30., .105), x_cm=0., y_cm=0., z_cm=14000.,
            parent_flight_cm=800., loose_two_plane=False)
        second = dict(first, index=1, pdg_id=-13)
        summary = study.summarize_event([first, second], [dict(mass=75., pt=40.)])
        self.assertLess(summary["max_gen_OS_mass"], 1.)
        self.assertEqual(summary["reco_BB_mass_ge50"], 1)
        self.assertEqual(summary["reco_BB_pt_ge20"], 1)
        self.assertTrue(summary["gen_OS_p20"])
        self.assertFalse(summary["gen_OS_p20_line"])
        self.assertFalse(summary["gen_OS_p10_flight100"])
        self.assertTrue(summary["gen_OS_p10_flight1000"])

    def test_full_source_trials_are_counted_once_per_independent_source(self):
        rows = [dict(source_id="a", stratum="qcd", corrected_weight=4.,
                     normalization=dict(full_source_trials=50, sigma_nd_pb=100.)),
                dict(source_id="a", stratum="qcd", corrected_weight=2.,
                     normalization=dict(full_source_trials=50, sigma_nd_pb=100.)),
                dict(source_id="b", stratum="qcd", corrected_weight=1.,
                     normalization=dict(full_source_trials=150, sigma_nd_pb=100.))]
        self.assertEqual(study.physics_weights(rows), [2., 1., .5])

    def test_multiple_muons_from_one_event_do_not_increase_independent_count(self):
        row = dict(reco_BB_pairs=[dict(hit_gen_indices=[0, 1])],
                   muons=[dict(index=0, p=20.), dict(index=1, p=20.)])
        table = study.muon_table([row], [3.], "p", edges=[0., 30.])
        self.assertEqual(table[0]["raw_muons"], 2)
        self.assertEqual(table[0]["raw_BB_hit_matched_muons"], 2)
        self.assertEqual(table[0]["denominator_effective_original_event_count"], 1.)

    def test_same_sign_generator_pair_can_be_retained_for_reco_charge_tails(self):
        muon = dict(index=0, pdg_id=13, p=30., pz=-29., x_cm=0., y_cm=0.,
                    z_cm=14000., loose_two_plane=True)
        row = dict(muons=[muon, dict(muon, index=1)], features=dict(gen_OS_p20=False))
        self.assertTrue(study.keep_rule(row, "gen_two_p20"))
        self.assertTrue(study.keep_rule(row, "gen_two_p20_line"))
        self.assertFalse(study.keep_rule(row, "gen_OS_p20"))

    def test_input_rejects_repeated_originals_and_double_sampling_correction(self):
        row = dict(event_id=[1, 2, 3], sampling_probability=.5, native_weight=2., corrected_weight=4.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                study.read_events(path)
            row["corrected_weight"] = 8.
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly once"):
                study.read_events(path)

    def test_material_control_missing_reconstruction_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.json"
            path.write_text(json.dumps(dict(schema="shift-material-scout-SIM-audit-v1", passed=True,
                minimum_CSC_stations_each_endcap=2, events=[dict(event_id=[1, 2, 3], GEN_muons=10,
                    actual_saved_muons=8, muons_with_required_CSC_stations_each_endcap=2)])))
            case = dict(event_id=[1, 2, 3], label="ordinary", variant="ordinary", sim_audit=str(path))
            rows, inputs, pending = study.material_control_rows(dict(schema="shift-material-aware-control-plot-inputs-v1", cases=[case]))
            self.assertEqual(rows[0]["GEN_muons"], 10)
            self.assertEqual(rows[0]["true_CSC_muons"], 2)
            self.assertIsNone(rows[0]["reco_Both_muons"])
            self.assertIsNone(rows[0]["reco_BothBoth_pairs"])
            self.assertFalse(pending)
            self.assertEqual(len(inputs), 1)

    def test_material_control_counts_exact_Both_topology_and_preserves_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.json"
            path.write_text(json.dumps(dict(schema="shift-material-scout-SIM-audit-v1", passed=True,
                minimum_CSC_stations_each_endcap=2, events=[dict(event_id=[1, 2, 3], GEN_muons=10,
                    muons_with_required_CSC_stations_each_endcap=2)])))
            response = Path(directory) / "response.json"
            response.write_text(json.dumps(dict(schema="shift-material-scout-reconstruction-response-v1", passed=True,
                events=[dict(event_id=[1, 2, 3], muons=[dict(topology=2), dict(topology=3)],
                    pairs=[dict(topologyMin=2, topologyMax=2, isOS=0, mass=16.75), dict(topologyMin=2, topologyMax=3, isOS=1, mass=5.)])])))
            case = dict(event_id=[1, 2, 3], label="ordinary", variant="ordinary", sim_audit=str(path), response_audit=str(response))
            rows, _, _ = study.material_control_rows(dict(schema="shift-material-aware-control-plot-inputs-v1", cases=[case]))
            self.assertEqual(rows[0]["reco_Both_muons"], 1)
            self.assertEqual(rows[0]["reco_BothBoth_pairs"], 1)
            self.assertEqual(rows[0]["reco_BothBoth_pair_isOS"], [0])
            self.assertEqual(rows[0]["reco_BothBoth_mass_GeV"], [16.75])


if __name__ == "__main__":
    unittest.main()
