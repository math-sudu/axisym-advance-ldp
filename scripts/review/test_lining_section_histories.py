"""Check section quadrature against an analytic stress field and transient peak."""
import unittest

import numpy as np

from extract_lining_section_histories import integrate_thrust, peak_record, section_quadrature


class SectionHistoryTests(unittest.TestCase):
    def test_thickness_integral_preserves_axial_variation_and_unequal_layers(self):
        r0, r1 = np.array([1.8, 1.88]), np.array([1.88, 2.0])
        z0, z1 = 6.0, 6.2
        q = 1 / np.sqrt(3)
        r = (r0[:, None] + r1[:, None]) / 2 + (r1 - r0)[:, None] / 2 * np.array([-q, q, -q, q])
        z = np.tile((z0 + z1) / 2 + (z1 - z0) / 2 * np.array([-q, -q, q, q]), (2, 1))
        # Reorder the four points so the section grouping must use coordinates.
        order = [2, 0, 3, 1]
        r, z = r[:, order], z[:, order]
        state = dict(liner_e_rho0=r0, liner_e_rho1=r1,
                     liner_e_z0=np.full(2, z0), liner_e_z1=np.full(2, z1), liner_gp_z=z)
        quadrature = section_quadrature(state, np.array([[3.0, 3.1]]))
        hoop = -(2 + 3 * r**2 + 7 * z + 11 * z * r)
        thrust = integrate_thrust(hoop, quadrature)
        def exact(z):
            return 1000 * (2 * .2 + (2**3 - 1.8**3) + 7 * z * .2
                           + 11 * z * (2**2 - 1.8**2) / 2)
        np.testing.assert_allclose(thrust, [exact(item[0] * 2) for item in quadrature])
        np.testing.assert_allclose(thrust.mean(), exact((z0 + z1) / 2))
        self.assertGreater(thrust[1], thrust[0])

    def test_history_peak_can_precede_the_final_state(self):
        result = peak_record(np.array([[np.nan, np.nan], [0, np.nan], [9, 4], [6, 7]]),
                             [0, 1, 2, 3], [{"z_over_a": 3.02}, {"z_over_a": 3.08}])
        self.assertEqual(result["step"], 2)
        self.assertEqual(result["z_over_a"], 3.02)
        self.assertEqual(result["hoop_thrust_kN_per_m"], 9)


if __name__ == "__main__":
    unittest.main()
