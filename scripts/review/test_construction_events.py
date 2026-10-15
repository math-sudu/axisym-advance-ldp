"""Check actual activation references and readout beyond the original 12a face."""
import unittest
from unittest.mock import patch

import numpy as np

import extract_lining_installation_events as events
import extract_lining_band_histories as bands


class ConstructionEventTests(unittest.TestCase):
    def test_nonintegral_final_round_is_rejected(self):
        with self.assertRaises(ValueError):
            events.round_count(14.0, 0.3)
        self.assertEqual(events.round_count(14.4, 0.3), 48)

    def test_fine_grid_events_and_late_convergence_are_retained(self):
        source = events.ROOT / "results/synthetic-events"
        edges = np.round(np.arange(3.0, 6.001, 0.05), 10)
        lo, hi = np.r_[-1.05, edges[:-1]], np.r_[-1.0, edges[1:]]

        def state(path):
            step = int(path.stem.split("-")[-1])
            face = step * 0.2
            return {
                "face_over_a": face,
                "liner_e_z0": lo * events.A, "liner_e_z1": hi * events.A,
                "liner_e_rho0": np.full(len(lo), 1.8),
                "liner_e_rho1": np.full(len(lo), 2.0),
                "liner_active": hi <= face - (1.0 if step == 0 else 0.95) + 1e-9,
                "liner_sig": np.zeros((len(lo), 4, 4)),
                "wall_nodes": np.array([0, 1]), "z_lines_m": np.array([6.0, 12.0]),
                "displacement_m": np.array([-face / 1000, 0, -face / 1000, 0]),
            }

        with patch.object(events, "read_state", side_effect=state), \
                patch.object(bands, "read_state", side_effect=state):
            run = events.extract_run(source, "B", 0.95, 1, 0.2, final_face=14.4)
            result = bands.extract_run(run, np.array([[3.0, 6.0]]))
        self.assertEqual(len(run["events"]), 60)
        event = next(item for item in run["events"]
                     if np.isclose(item["band_over_a"][1], 3.05))
        self.assertAlmostEqual(event["installation_edge_distance_over_a"], 0.75)
        self.assertAlmostEqual(event["installation_face_over_a"], 3.8)
        self.assertAlmostEqual(event["first_loaded_face_over_a"], 4.0)
        self.assertEqual(result["construction_rounds"], 72)
        self.assertAlmostEqual(result["summary"]["convergence_envelope"]["convergence_mm"], 14.4)
        self.assertAlmostEqual(result["summary"]["convergence_envelope"]["face_over_a"], 14.4)


if __name__ == "__main__":
    unittest.main()
