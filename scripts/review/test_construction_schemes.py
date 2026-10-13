"""Check feasibility, allowance boundaries and construction-round ties."""
from math import inf, nan, nextafter
import unittest

from compare_construction_schemes import allowance_map, select_runs, select_under_allowances


class ConstructionChoiceTests(unittest.TestCase):
    def setUp(self):
        self.schemes = [dict(zip(("convergence_envelope_mm", "demand_envelope_kN_per_m",
                                 "construction_rounds"), values, strict=True))
                        for values in ((11, 2200, 30), (10, 2000, 60),
                                       (10.5, 1800, 60), (12, 1600, 60))]

    def test_empty_unique_tied_and_fewer_round_choices(self):
        for u, thrust, feasible, selected, rounds in (
            (11, 1700, [], [], None),
            (11, 1900, [2], [2], 60),
            (11, 2100, [1, 2], [1, 2], 60),
            (11, 2200, [0, 1, 2], [0], 30),
            (12, 1700, [3], [3], 60),
        ):
            with self.subTest(u=u, thrust=thrust):
                self.assertEqual(select_under_allowances(self.schemes, u, thrust), {
                    "feasible_indices": feasible, "selected_indices": selected,
                    "minimum_rounds": rounds,
                })

    def test_exact_boundaries_are_inclusive_without_rounding(self):
        for scheme in self.schemes:
            u, thrust = scheme["convergence_envelope_mm"], scheme["demand_envelope_kN_per_m"]
            self.assertEqual(select_under_allowances([scheme], u, thrust)["selected_indices"], [0])
            self.assertEqual(select_under_allowances([scheme], nextafter(u, -inf), thrust)
                             ["feasible_indices"], [])
            self.assertEqual(select_under_allowances([scheme], u, nextafter(thrust, -inf))
                             ["feasible_indices"], [])

    def test_map_covers_each_feasible_quadrant_and_retains_all_ties(self):
        result = allowance_map(self.schemes)
        self.assertEqual(result["convergence_intervals_mm"],
                         [[0, 10], [10, 10.5], [10.5, 11], [11, 12], [12, None]])
        self.assertEqual(result["thrust_intervals_kN_per_m"],
                         [[0, 1600], [1600, 1800], [1800, 2000], [2000, 2200], [2200, None]])
        self.assertEqual([[cell["feasible_indices"] for cell in row] for row in result["cells"]],
                         [[[], [], [], [], []],
                          [[], [], [], [], [3]],
                          [[], [], [2], [2], [2, 3]],
                          [[], [1], [1, 2], [1, 2], [1, 2, 3]],
                          [[], [1], [1, 2], [0, 1, 2], [0, 1, 2, 3]]])
        self.assertEqual(result["cells"][-1][-1]["selected_indices"], [0])
        self.assertEqual(result["cells"][-2][-1]["selected_indices"], [1, 2, 3])

    def test_duplicate_thresholds_do_not_create_zero_width_regions(self):
        result = allowance_map([self.schemes[0], self.schemes[0]])
        self.assertEqual(result["convergence_intervals_mm"], [[0, 11], [11, None]])
        self.assertEqual(result["cells"][-1][-1]["selected_indices"], [0, 1])

    def test_invalid_allowances_fail_explicitly(self):
        for value in (-1, inf, nan):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                select_under_allowances(self.schemes, value, 2100)
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                select_under_allowances(self.schemes, 11, value)

    def test_different_spinups_cannot_form_a_construction_comparison(self):
        runs = [dict(case="C", mesh_level=1, installation_target_over_a=0.6,
                     round_length_over_a=step, initial_state=f"spinup-{step}")
                for step in (0.4, 0.2)]
        with self.assertRaisesRegex(ValueError, "different initial states"):
            select_runs({"runs": runs}, "C")


if __name__ == "__main__":
    unittest.main()
