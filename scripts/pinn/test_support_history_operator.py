"""Physical element and event identities for the installation-history layer.

These checks evaluate the SAME trial field through the original quadrature;
they do not solve or compare a second tunnel model. Tolerances below measure
floating-point algebra, not acceptable error in construction predictions.
"""
import copy
import json
import unittest

import numpy as np

from run_construction import initial_model, begin_round, corrected_kernel, base
from activation_pinn import axifem, lining_sections, tensor
from extract_lining_section_histories import section_quadrature, integrate_thrust
from support_history_operator import (
    InstallationReference, SupportHistory, SupportHistoryOperator,
    extend_installation_field,
)


def assemble_stress(fe, elements, stress):
    force = np.zeros(fe.ndof)
    values = np.einsum("egij,egi,eg->ej", fe.m.B_all[elements], stress, fe.m.W_all[elements])
    np.add.at(force, fe.m.dof_all[elements], values)
    return force


def original_lining_trial(fe, displacement):
    block = fe.blocks["liner"]
    elements, strain, previous = fe._block_state(block, displacement)
    stress = axifem.integrate_axisym(
        tensor(strain.reshape(-1, 4)), tensor(previous.reshape(-1, 4)),
        block["E"], block["nu"], block["mat"], block["kind"])[0].numpy().reshape(-1, 4, 4)
    return assemble_stress(fe, elements, stress), stress


class SupportHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = corrected_kernel()
        cls.context.__enter__()
        cls.models = {mesh: initial_model(mesh=mesh)[0] for mesh in ("coarse", "production")}

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def assert_roundoff(self, actual, expected, scale=None):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if scale is None:
            scale = max(np.linalg.norm(actual), np.linalg.norm(expected), 1.0)
        # Sparse solves, element sums and cancellation of absolute K*u and h
        # accumulate rounding. This allows ~4k double-precision rounding units.
        tolerance = 4096 * np.finfo(float).eps * scale
        self.assertLessEqual(np.linalg.norm(actual - expected), tolerance)

    def first_event(self, original):
        model = copy.deepcopy(original)
        history = SupportHistory.from_initial(original.fe)
        born = begin_round(model, 1)
        elements = model.fe.blocks["liner"]["elems"][born]
        installation = extend_installation_field(original.fe, elements)
        history = history.install(original.fe, elements, installation, event_id=1)
        return model, history, installation, born

    def test_initial_reference_and_arbitrary_contact_recovery(self):
        for mesh, model in self.models.items():
            with self.subTest(mesh=mesh):
                fe = model.fe
                history = SupportHistory.from_initial(fe)
                operator = SupportHistoryOperator(fe, history)
                self.assertEqual(history.records[0].kind, "equivalent_initial")
                self.assert_roundoff(operator.recover_field(fe.u_n), fe.u_n)
                self.assert_roundoff(operator.recover_stress(fe.u_n),
                                     fe.blocks["liner"]["sig"][fe.blocks["liner"]["active"]])
                trial = fe.u_n.copy()
                trial[operator.contact_dofs] += np.random.default_rng(4).normal(
                    scale=2e-4, size=len(operator.c))
                trial = operator.recover_field(trial)
                direct, stress = original_lining_trial(fe, trial)
                self.assert_roundoff(operator.recover_stress(trial), stress)
                scale = np.linalg.norm(abs(operator.K) @ abs(trial[operator.dofs])
                                       + abs(operator.h))
                self.assert_roundoff(direct[operator.internal_dofs], 0, scale)
                self.assert_roundoff(operator.interface_force(trial[operator.contact_dofs]),
                                     direct[operator.contact_dofs], scale)
                self.assert_roundoff(operator.K @ trial[operator.dofs] - operator.h,
                                     direct[operator.dofs], scale)
                print(json.dumps({"check": "arbitrary_contact", "mesh": mesh,
                                  "contact_dofs": len(operator.c), "internal_dofs": len(operator.i),
                                  "internal_force_norm": float(np.linalg.norm(direct[operator.internal_dofs])),
                                  "stress_difference_norm": float(np.linalg.norm(operator.recover_stress(trial) - stress))}))

    def test_nonzero_birth_and_lambda_zero_preserve_the_previous_residual(self):
        for mesh, original in self.models.items():
            with self.subTest(mesh=mesh):
                model, history, installation, born = self.first_event(original)
                fe, before = model.fe, original.fe
                operator = SupportHistoryOperator(fe, history)
                active_before = before.active_dof_mask()
                np.testing.assert_array_equal(installation[active_before], before.u_n[active_before])
                newborn = np.isin(operator.elements, fe.blocks["liner"]["elems"][born])
                self.assertGreater(np.linalg.norm(installation[operator.element_dofs[newborn]]), 0)
                # Dormant values explicitly follow the wall; they are not the
                # zero/reference values left behind by an earlier excavation.
                new_nodes = np.unique(fe.m.conn[operator.elements[newborn]])
                for node in new_nodes[~active_before.reshape(-1, 2).any(axis=1)[new_nodes]]:
                    wall = model.wall_nodes[fe.m.node_z[model.wall_nodes] == fe.m.node_z[node]][0]
                    np.testing.assert_array_equal(installation.reshape(-1, 2)[node],
                                                  before.u_n.reshape(-1, 2)[wall])
                born_stress = operator.recover_stress(installation)[newborn]
                self.assert_roundoff(born_stress, 0)
                old_force, _ = original_lining_trial(before, before.u_n)
                new_force = assemble_stress(fe, operator.elements, operator.recover_stress(installation))
                self.assert_roundoff(new_force, old_force)
                self.assert_roundoff(operator.recover_field(installation), installation)

                # The actual I-load components, with the residual's sign convention.
                for load in (fe.f_ext_fix, fe.f_release, model.solver.F0,
                             model.solver._maps["T"] @ model.solver.c):
                    np.testing.assert_array_equal(load[operator.internal_dofs], 0)
                self.assertFalse(fe.dirichlet[operator.internal_dofs].any())
                self.assertEqual(len(np.intersect1d(operator.internal_dofs, model.solver.ext.if_dofs)), 0)
                old_internal = sum(assemble_stress(before, b["elems"][b["active"]],
                                                   b["sig"][b["active"]])
                                   for b in before.blocks.values())
                rock = fe.blocks["rock"]
                after_internal = new_force + assemble_stress(
                    fe, rock["elems"][rock["active"]], rock["sig"][rock["active"]])
                residual_difference = after_internal + fe.f_release - old_internal
                self.assert_roundoff(residual_difference, 0, np.linalg.norm(old_internal))
                # Neither reference construction nor operator calls commit FE state.
                np.testing.assert_array_equal(fe.u_n, before.u_n)
                np.testing.assert_array_equal(fe.blocks["liner"]["sig"][born], 0)
                print(json.dumps({"check": "first_event", "mesh": mesh,
                                  "born_elements": int(born.sum()),
                                  "birth_stress_norm": float(np.linalg.norm(born_stress)),
                                  "lambda_zero_difference_norm": float(np.linalg.norm(residual_difference))}))

    def test_split_registration_retains_shared_nodes_and_whole_stiffness(self):
        original = self.models["production"]
        model, history, installation, born = self.first_event(original)
        fe = model.fe
        elements = fe.blocks["liner"]["elems"][born]
        split = SupportHistory.from_initial(original.fe)
        for part in (elements[::2], elements[1::2]):
            split = split.install(original.fe, part, installation, event_id=1)
        whole, divided = (SupportHistoryOperator(fe, h) for h in (history, split))
        self.assert_roundoff((whole.K - divided.K).data, 0)
        self.assert_roundoff(whole.h, divided.h)
        trial = installation + np.random.default_rng(7).normal(scale=1e-4, size=fe.ndof)
        self.assert_roundoff(whole.recover_field(trial), divided.recover_field(trial))
        old_nodes = np.unique(fe.m.conn[history.records[0].elements])
        shared = np.intersect1d(old_nodes, np.unique(fe.m.conn[elements]))
        self.assertGreater(len(shared), 0)
        self.assertEqual(len(whole.dofs), 2 * len(np.union1d(old_nodes, np.unique(fe.m.conn[elements]))))
        # Original element tangent includes all radial and axial connections.
        lining_fe = copy.copy(fe)
        lining_fe.blocks = {"liner": fe.blocks["liner"]}
        _, direct_k = lining_fe.internal_force(fe.u_n, want_tangent=True)
        self.assert_roundoff((whole.K - direct_k[whole.dofs][:, whole.dofs]).data, 0,
                             np.linalg.norm(whole.K.data))
        with self.assertRaisesRegex(ValueError, "second reference"):
            split.install(original.fe, elements, installation, event_id=2)

    def test_same_geometry_keeps_different_complete_references(self):
        fe = self.models["production"].fe
        elements = fe.blocks["liner"]["elems"][fe.blocks["liner"]["active"]]
        birth_a = fe.u_n.copy()
        birth_b = birth_a.copy().reshape(-1, 2)
        birth_b[:, 0] += 2e-4 * fe.m.node_rho * np.cos(fe.m.node_z)
        birth_b[:, 1] += 3e-4 * np.sin(fe.m.node_z) * fe.m.node_rho
        birth_b = birth_b.ravel()
        a, b = (SupportHistoryOperator(fe, SupportHistory().install(fe, elements, u, 1))
                for u in (birth_a, birth_b))
        self.assert_roundoff((a.K - b.K).data, 0)
        self.assert_roundoff(b.h - a.h, a.K @ (birth_b - birth_a)[a.dofs], np.linalg.norm(a.h))
        force_a = a.interface_force(birth_a[a.contact_dofs])
        force_b = b.interface_force(birth_a[b.contact_dofs])
        expected = a.apply_tangent((birth_b - birth_a)[a.contact_dofs])
        self.assert_roundoff(force_b - force_a, -expected, np.linalg.norm(a.h))
        self.assertGreater(np.linalg.norm(force_b - force_a), 1e-3)
        ua, ub = a.recover_field(birth_a), b.recover_field(birth_a)
        self.assertGreater(np.linalg.norm(b.recover_stress(ub) - a.recover_stress(ua)), 1e-3)

    def test_full_reference_retains_stress_lost_in_nodal_force_assembly(self):
        fe = self.models["coarse"].fe
        history = SupportHistory.from_initial(fe)
        reference = history.records[0]
        element = reference.elements[0]
        mapping = np.concatenate([fe.m.B_all[element, q].T * fe.m.W_all[element, q]
                                  for q in range(4)], axis=1)
        _, _, vectors = np.linalg.svd(mapping, full_matrices=True)
        self_equilibrated = vectors[-1].reshape(4, 4)
        bias = reference.stress_bias.copy()
        bias[0] += self_equilibrated
        changed = SupportHistory((InstallationReference(reference.elements, bias, 0,
                                                        "equivalent_initial"),))
        a, b = (SupportHistoryOperator(fe, h) for h in (history, changed))
        self.assert_roundoff(a.h, b.h)
        self.assert_roundoff(a.recover_field(fe.u_n), b.recover_field(fe.u_n))
        stress_difference = b.recover_stress(fe.u_n) - a.recover_stress(fe.u_n)
        self.assert_roundoff(stress_difference[0], -self_equilibrated)
        self.assertGreater(np.linalg.norm(stress_difference), 0.9)

    def test_directions_transposes_and_known_internal_load(self):
        for mesh, model in self.models.items():
            with self.subTest(mesh=mesh):
                fe = model.fe
                operator = SupportHistoryOperator(fe, SupportHistory.from_initial(fe))
                rng = np.random.default_rng(13)
                direction, dual_c = rng.normal(size=(2, len(operator.c)))
                dual_i = rng.normal(size=len(operator.i))
                self.assert_roundoff(dual_c @ operator.apply_tangent(direction),
                                     direction @ operator.apply_tangent_transpose(dual_c))
                self.assert_roundoff(dual_i @ operator.recover_direction(direction),
                                     direction @ operator.recover_transpose(dual_i))
                p_i = 0.1 * rng.normal(size=len(operator.i))
                u_c = fe.u_n[operator.contact_dofs]
                trial = operator.recover_field(fe.u_n, p_i)
                direct, _ = original_lining_trial(fe, trial)
                scale = np.linalg.norm(abs(operator.K) @ abs(trial[operator.dofs])
                                       + abs(operator.h) + np.linalg.norm(p_i))
                self.assert_roundoff(direct[operator.internal_dofs], p_i, scale)
                self.assert_roundoff(direct[operator.contact_dofs],
                                     operator.interface_force(u_c, p_i), scale)
                errors = []
                for step in (1e-4, 1e-6, 1e-8):
                    finite = (operator.interface_force(u_c + step * direction, p_i)
                              - operator.interface_force(u_c - step * direction, p_i)) / (2 * step)
                    derivative = operator.apply_tangent(direction)
                    errors.append(np.linalg.norm(finite - derivative) / np.linalg.norm(derivative))
                    finite_i = (operator.recover_internal(u_c + step * direction, p_i)
                                - operator.recover_internal(u_c - step * direction, p_i)) / (2 * step)
                    np.testing.assert_allclose(finite_i, operator.recover_direction(direction),
                                               rtol=1e-7, atol=1e-7)
                self.assertLess(min(errors), 1e-9)
                print(json.dumps({"check": "directional_derivative", "mesh": mesh,
                                  "steps": [1e-4, 1e-6, 1e-8], "relative_errors": errors}))

    def test_recovered_stress_preserves_the_complete_thrust_profile(self):
        for mesh, original in self.models.items():
            with self.subTest(mesh=mesh):
                model, history, installation, born = self.first_event(original)
                fe = model.fe
                operator = SupportHistoryOperator(fe, history)
                # Independent incremental reading starts from the extended
                # birth field, with old stresses preserved and new stresses zero.
                fe.u_n = installation.copy()
                trial = fe.u_n + np.random.default_rng(15).normal(scale=1e-4, size=fe.ndof)
                trial = operator.recover_field(trial)
                direct, direct_stress = original_lining_trial(fe, trial)
                self.assert_roundoff(operator.recover_stress(trial), direct_stress)
                scale = np.linalg.norm(abs(operator.K) @ abs(trial[operator.dofs])
                                       + abs(operator.h))
                self.assert_roundoff(direct[operator.internal_dofs], 0, scale)
                self.assert_roundoff(operator.interface_force(trial[operator.contact_dofs]),
                                     direct[operator.contact_dofs], scale)
                expected_z, expected_thrust = lining_sections(model, trial)
                state = base.state_arrays(model)
                active = fe.blocks["liner"]["active"]
                bands = np.unique(np.column_stack([state["liner_e_z0"][active],
                                                   state["liner_e_z1"][active]]) / base.scen.A, axis=0)
                quadrature = section_quadrature(state, bands)
                stress = np.zeros_like(state["liner_sig"])
                stress[active] = operator.recover_stress(trial)
                thrust = integrate_thrust(stress[:, :, 2], quadrature)
                np.testing.assert_allclose([q[0] for q in quadrature], expected_z, atol=1e-12)
                self.assert_roundoff(thrust, expected_thrust)
                print(json.dumps({"check": "thrust_profile", "mesh": mesh,
                                  "sections": len(thrust), "new_elements": int(born.sum()),
                                  "maximum_difference_kN_per_m": float(np.max(np.abs(thrust - expected_thrust)))}))

    def test_empty_disconnected_and_unattached_support(self):
        fe = copy.deepcopy(self.models["coarse"].fe)
        block = fe.blocks["liner"]
        initial = block["active"].copy()
        block["active"][:] = False
        empty = SupportHistoryOperator(fe, SupportHistory())
        np.testing.assert_array_equal(empty.recover_field(fe.u_n), fe.u_n)
        self.assertEqual(empty.recover_stress(fe.u_n).shape, (0, 4, 4))
        self.assertEqual(empty.interface_force(np.empty(0)).size, 0)
        block["active"] = initial
        ends = fe.m.e_z1[block["elems"]]
        cut = np.median(ends[initial])
        block["active"] &= ~np.isclose(ends, cut)
        split = SupportHistoryOperator(fe, SupportHistory.from_initial(fe))
        field = split.recover_field(fe.u_n)
        direct, _ = original_lining_trial(fe, field)
        scale = np.linalg.norm(abs(split.K) @ abs(field[split.dofs]) + abs(split.h))
        self.assert_roundoff(direct[split.internal_dofs], 0, scale)
        fe.blocks["rock"]["active"][:] = False
        with self.assertRaisesRegex(ValueError, "share nodes"):
            SupportHistoryOperator(fe, SupportHistory.from_initial(fe))

    def test_trials_do_not_refresh_or_mutate_installation_memory(self):
        model, history, field, _ = self.first_event(self.models["coarse"])
        fe = model.fe
        operator = SupportHistoryOperator(fe, history)
        saved_u = fe.u_n.copy()
        saved_sig = {k: b["sig"].copy() for k, b in fe.blocks.items()}
        saved_bias = [r.stress_bias.copy() for r in history.records]
        trials = [field + scale * np.sin(np.arange(fe.ndof)) for scale in (1e-4, -2e-4)]
        first = [operator.recover_stress(operator.recover_field(u)) for u in trials]
        second = [operator.recover_stress(operator.recover_field(u)) for u in reversed(trials)]
        for a, b in zip(first, reversed(second)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(saved_u, fe.u_n)
        for name, block in fe.blocks.items():
            np.testing.assert_array_equal(saved_sig[name], block["sig"])
        for reference, expected in zip(history.records, saved_bias):
            np.testing.assert_array_equal(reference.stress_bias, expected)
            self.assertFalse(reference.stress_bias.flags.writeable)

    def test_unbalanced_old_field_requires_an_explicit_internal_correction(self):
        fe = copy.deepcopy(self.models["coarse"].fe)
        initial = SupportHistoryOperator(fe, SupportHistory.from_initial(fe))
        trial = fe.u_n.copy()
        trial[initial.internal_dofs] += 1e-4 * np.sin(np.arange(len(initial.i)))
        direct, stress = original_lining_trial(fe, trial)
        # Construct a saved, unbalanced elastic trial without treating it as
        # a new-method equilibrium or changing the contact displacement.
        fe.u_n = trial.copy()
        fe.blocks["liner"]["sig"][fe.blocks["liner"]["active"]] = stress
        operator = SupportHistoryOperator(fe, SupportHistory.from_initial(fe))
        recovered = operator.recover_field(fe.u_n)
        correction = recovered[operator.internal_dofs] - trial[operator.internal_dofs]
        self.assertGreater(np.linalg.norm(correction), 1e-4)
        self.assert_roundoff(correction, -operator.factor.solve(direct[operator.internal_dofs]))
        np.testing.assert_array_equal(recovered[operator.contact_dofs], trial[operator.contact_dofs])
        np.testing.assert_array_equal(fe.u_n, trial)


if __name__ == "__main__":
    unittest.main()
