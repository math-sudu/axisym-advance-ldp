"""Check activation-relative strain and same-batch section selection."""
import unittest

import numpy as np

from extract_paired_section_strains import AxiMesh, control_pair, elastic_stress, integrate_components


class PairedSectionStrainTests(unittest.TestCase):
    def test_affine_loading_after_nonuniform_activation_reference(self):
        mesh = AxiMesh([1.8, 1.92, 2.0], [0.0, .16]).build_B()
        r, z = mesh.node_rho, mesh.node_z
        reference = np.column_stack((.01 * r * z, .02 * r**2 - .003 * z**2 + .2)).ravel()
        increment = np.column_stack((-.0004 * r, .0002 * z + .0001 * r)).ravel()
        elements = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
        gp = np.array([[0, 1, 0, 1], [2, 3, 2, 3]])
        weights = np.tile([.06, .06, .04, .04], (2, 1))
        strain = np.einsum("sgij,sgj->sgi", mesh.B_all[elements, gp],
                           (reference + increment - reference)[mesh.dof_all[elements]])
        expected_strain = np.broadcast_to([-.0004, .0002, -.0004, .0001], strain.shape)
        np.testing.assert_allclose(strain, expected_strain, atol=1e-14)
        # E=18000 MPa and nu=.2 give lambda=5000 MPa and mu=7500 MPa.
        expected_stress = np.broadcast_to([-9.0, 0.0, -9.0, .75], strain.shape)
        np.testing.assert_allclose(elastic_stress(strain), expected_stress, atol=1e-9)
        np.testing.assert_allclose(integrate_components(strain, weights),
                                   [[400, -200, 1600], [400, -200, 1600]], atol=1e-8)

    def test_low_section_uses_controlling_batch_and_state(self):
        run = {
            "summary": {"demand_envelope": {"section_index": 1, "step": 0, "face_over_a": 2}},
            "sections": [{"installation_batch_over_a": [0, .4]},
                         {"installation_batch_over_a": [0, .4]},
                         {"installation_batch_over_a": [.4, .8]}],
            "history": [{"hoop_thrust_kN_per_m": [5, 9, 1]},
                        {"hoop_thrust_kN_per_m": [7, 3, 2]}],
        }
        pair = control_pair(run)
        self.assertEqual(pair["low_index"], 0)
        self.assertEqual(pair["high_index"], 1)
        self.assertEqual(pair["batch_indices"], [0, 1])
        self.assertEqual(pair["selection_face_over_a"], 2)


if __name__ == "__main__":
    unittest.main()
