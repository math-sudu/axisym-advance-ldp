"""Same-field physics, derivatives and restart checks for the coupled method."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from run_construction import initial_model, corrected_kernel
from activation_pinn import tensor, axifem, lining_sections, train_increment
from support_history_operator import SupportHistory, SupportHistoryOperator
from history_pinn import GroundNetwork, HistoryProblem, SupportRecovery, prepare_event
from neural_optimization import head_basis, head_system, head_residual_jacobian, neural_linearization, refine_head
from run_history_construction import (snapshot, restore_snapshot, weights, save_checkpoint,
                                      build_trial, batch_response, material_spec, extend_network, network_from_controls, run)


class HistoryPinnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.context = corrected_kernel()
        cls.context.__enter__()
        cls.initial = initial_model()[0]
        cls.initial_history = SupportHistory.from_initial(cls.initial.fe)

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def problem(self, release=0.5, device="cpu"):
        model, history, born, event = prepare_event(self.initial, self.initial_history, 1)
        network = GroundNetwork(width=6).to(device)
        return HistoryProblem(model, history, release, network), born, event

    def perturb(self, network):
        with torch.no_grad():
            for branch in network.branches:
                branch[-1].weight.normal_(std=0.004)
                branch[-1].bias.normal_(std=0.002)

    def test_ground_topology_and_actual_fronts(self):
        for mesh in ("coarse", "production"):
            model = self.initial if mesh == "coarse" else initial_model(mesh=mesh)[0]
            stage, history, _, _ = prepare_event(model, SupportHistory.from_initial(model.fe), 1)
            problem = HistoryProblem(stage, history, 0.5, GroundNetwork(width=6))
            axis = np.flatnonzero((stage.mesh.node_rho == 0) & (stage.mesh.node_z > stage.dstep))
            available = axis[~stage.fe.dirichlet[2 * axis + 1]]
            retained = np.union1d(problem.interior, problem.if_dofs)
            self.assertTrue(np.isin(2 * available + 1, retained).all())
            self.assertFalse(np.isin(2 * axis, problem.interior).any())
            self.assertFalse(np.isin(problem.support.internal_dofs, problem.interior).any())
            np.testing.assert_allclose(problem.B.T.cpu() @ problem.B.cpu(),
                                       np.eye(problem.B.shape[1]), atol=2e-14)
            np.testing.assert_allclose(problem.fronts, (stage.dstep, stage.dstep - stage.liner_d), atol=1e-14)
            np.testing.assert_array_equal(problem.internal_load, 0)
            u, _ = problem.field()
            np.testing.assert_allclose(u.detach().cpu(), stage.fe.u_n, atol=2e-14, rtol=0)

    def test_zero_release_event_preserves_remaining_residual(self):
        before = self.initial.fe.internal_force(self.initial.fe.u_n)[0]
        before -= self.initial.fe.f_ext_fix + self.initial.solver.F0
        before -= self.initial.solver._maps["T"] @ self.initial.solver.c
        problem, born, _ = self.problem(release=0)
        residual = problem.residual()[0].detach().numpy()
        np.testing.assert_allclose(residual, problem.P.T @ before, atol=4e-11, rtol=0)
        np.testing.assert_allclose(problem.model.fe.blocks["liner"]["sig"][born], 0, atol=2e-14)
        # Staging did not change the prior accepted arrays or references.
        original = SupportHistoryOperator(self.initial.fe, self.initial_history)
        np.testing.assert_allclose(original.recover_stress(self.initial.fe.u_n),
                                   self.initial.fe.blocks["liner"]["sig"][self.initial.fe.blocks["liner"]["active"]],
                                   atol=2e-14)

    def test_residual_matches_direct_full_element_assembly(self):
        problem, _, _ = self.problem()
        self.perturb(problem.network)
        residual, u, c = problem.residual()
        full = problem.model.fe.internal_force(u.detach().numpy())[0]
        full += problem.force.numpy() - problem.T.numpy() @ c.detach().numpy()
        np.testing.assert_allclose(residual.detach(), problem.P.T @ full, atol=3e-11, rtol=1e-9)
        np.testing.assert_allclose(full[problem.support.internal_dofs], 0, atol=3e-11)
        liner = problem.model.fe.blocks["liner"]
        for row, reference in zip(batch_response(problem, u.detach().numpy()), problem.history.records):
            z, thrust = lining_sections(problem.model, u.detach().numpy(),
                                        np.isin(liner["elems"], reference.elements))
            np.testing.assert_array_equal(row["section_z_over_a"], z)
            np.testing.assert_allclose(row["section_thrust_kN_per_m"], thrust, atol=2e-10, rtol=1e-10)

    def test_loss_gradient_tangent_and_transpose(self):
        problem, _, _ = self.problem()
        self.perturb(problem.network)
        normal, gradient, _, _, _, diagonal = neural_linearization(problem)
        automatic = torch.cat([value.ravel() for value in torch.autograd.grad(
            problem.objective()[0], problem.network.parameters())])
        torch.testing.assert_close(gradient, automatic, rtol=2e-8, atol=2e-10)
        a, b = torch.randn_like(gradient), torch.randn_like(gradient)
        torch.testing.assert_close(a @ normal(b), b @ normal(a), rtol=1e-9, atol=1e-9)
        self.assertGreater(float(a @ normal(a)), 0)
        self.assertTrue(torch.all(diagonal > 0))
        original = [p.detach().clone() for p in problem.network.parameters()]
        a /= a.norm()
        losses = []
        for sign in (-1, 1):
            offset = 0
            with torch.no_grad():
                for p, old in zip(problem.network.parameters(), original):
                    count = p.numel()
                    p.copy_(old + sign * 1e-5 * a[offset:offset + count].reshape_as(p))
                    offset += count
                losses.append(float(problem.objective()[0]))
        self.assertAlmostEqual((losses[1] - losses[0]) / 2e-5, float(gradient @ a), delta=2e-7)
        u = problem.field()[0].detach().numpy()
        tangent = problem.tangent(u)
        rng = np.random.default_rng(3)
        v, w = rng.normal(size=(2, problem.size))
        self.assertAlmostEqual(float(w @ (tangent @ v)), float(v @ tangent.rmatvec(w)), delta=1e-6)

    def test_output_layer_jacobian_gradient_and_update(self):
        problem, _, _ = self.problem()
        self.perturb(problem.network)
        basis, parameters = head_basis(problem)
        before = problem.field()[0].detach().numpy()
        reduced_before = problem.P.T @ before
        delta = np.random.default_rng(7).normal(size=basis.shape[1]) * 1e-4
        original = [parameter.detach().clone() for parameter in parameters]
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                size = parameter.numel()
                parameter.add_(torch.as_tensor(delta[offset:offset + size]).reshape_as(parameter))
                offset += size
        after = problem.P.T @ problem.field()[0].detach().numpy()
        np.testing.assert_allclose(after - reduced_before, basis @ delta, atol=2e-15, rtol=1e-9)
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)
        _, gradient, loss, _ = head_system(problem, basis)
        automatic = torch.cat([value.ravel() for value in torch.autograd.grad(problem.objective()[0], parameters)])
        torch.testing.assert_close(gradient, automatic, rtol=2e-8, atol=2e-10)
        frozen = problem.model.fe.u_n.copy()
        refine_head(problem, 1, 1e-10)
        self.assertLess(float(problem.objective()[0].detach()), loss)
        np.testing.assert_array_equal(problem.model.fe.u_n, frozen)

    def test_face_patch_preserves_trial_and_supplies_local_directions(self):
        problem, born, event = self.problem()
        anchor = weights(problem.network)
        self.perturb(problem.network)
        before = problem.field()[0].detach().numpy()
        frozen = snapshot(problem.model, problem.history, 1, 0, born, event)
        before_controls = {"width": 6, "seed": 17}
        after_controls = {**before_controls, "face_patch_span_over_a": 0.2}
        pending = {"state": frozen, "target": 0.5,
                   "anchor": extend_network(anchor, before_controls, after_controls),
                   "network": extend_network(weights(problem.network), before_controls, after_controls)}
        network = GroundNetwork(6, 17, 0.2)
        expanded = build_trial(problem.model, network, pending, 0.5)
        np.testing.assert_allclose(expanded.field()[0].detach(), before, atol=2e-14, rtol=0)
        basis, parameters = head_basis(expanded)
        count = sum(p.numel() for p in network.branches[-1][-1].parameters())
        self.assertGreater(np.linalg.norm(basis[:, -count:].data), 0)
        delta = np.zeros(basis.shape[1])
        delta[-count:] = np.random.default_rng(9).normal(size=count) * 1e-4
        original = expanded.P.T @ before
        with torch.no_grad():
            for parameter, increment in zip(parameters[-2:], np.split(delta[-count:], [12])):
                parameter.add_(tensor(increment.reshape(parameter.shape)))
        np.testing.assert_allclose(expanded.P.T @ expanded.field()[0].detach().numpy() - original,
                                   basis @ delta, atol=2e-15, rtol=1e-9)
        frozen_after = snapshot(problem.model, problem.history, 1, 0, born, event)
        for key in frozen["arrays"]:
            torch.testing.assert_close(frozen_after["arrays"][key], frozen["arrays"][key], rtol=0, atol=0)

    def test_local_front_extension_and_svd_training(self):
        problem, born, event = self.problem()
        anchor = weights(problem.network)
        self.perturb(problem.network)
        before = problem.field()[0].detach().clone()
        source = {"width": 6, "seed": 17}
        target = {**source, "front_local": True, "support_patch_span_over_a": 0.4, "support_local": True}
        frozen = snapshot(problem.model, problem.history, 1, 0, born, event)
        pending = {"state": frozen, "target": 0.5,
                   "anchor": extend_network(anchor, source, target),
                   "network": extend_network(weights(problem.network), source, target)}
        expanded = build_trial(problem.model, network_from_controls(target), pending, 0.5)
        torch.testing.assert_close(expanded.field()[0], before, rtol=0, atol=0)
        basis, parameters = head_basis(expanded)
        matrix, error, loss, _ = head_residual_jacobian(expanded, basis)
        automatic = torch.cat([v.ravel() for v in torch.autograd.grad(expanded.objective()[0], parameters)])
        np.testing.assert_allclose(matrix.T @ error, automatic, rtol=2e-8, atol=2e-10)
        refine_head(expanded, 1, 1e-10, solver="svd")
        self.assertLess(float(expanded.objective()[0].detach()), loss)
        np.testing.assert_array_equal(problem.model.fe.u_n, frozen["arrays"]["u"].numpy())
        # The same architecture and anchor reconstruct the trained field exactly.
        pending["network"] = weights(expanded.network)
        restored = build_trial(copy.deepcopy(problem.model), network_from_controls(target), pending, 0.5)
        torch.testing.assert_close(restored.field()[0], expanded.field()[0], rtol=0, atol=0)

    def test_recovery_autograd_and_condensed_elastic_scale(self):
        problem, _, _ = self.problem()
        operator = problem.support
        contact = tensor(problem.model.fe.u_n[operator.contact_dofs]).requires_grad_(True)
        field = SupportRecovery.apply(contact, operator, problem.internal_load)
        dual = torch.randn_like(field)
        derivative = torch.autograd.grad(field @ dual, contact)[0]
        np.testing.assert_allclose(derivative, operator.recover_transpose(dual.numpy()), atol=1e-14)
        vector = np.random.default_rng(4).normal(size=problem.size)
        correction = problem.factor.solve(vector)
        fe = problem.model.fe
        previous = fe.blocks["rock"]["kind"]
        threads = torch.get_num_threads()
        try:
            fe.blocks["rock"]["kind"] = "elastic"
            elastic = problem.tangent(fe.u_n)
            np.testing.assert_allclose(elastic @ correction, vector, atol=2e-8, rtol=1e-8)
            directions = np.random.default_rng(6).normal(size=(problem.size, 65))
            for trans in ("N", "T"):
                torch.set_num_threads(1)
                batch = problem.factor.solve(directions, trans=trans)
                torch.set_num_threads(4)
                parallel = problem.factor.solve(directions, trans=trans)
                # BLAS threading can change summation order at roundoff level.
                np.testing.assert_allclose(parallel, batch, rtol=1e-12, atol=1e-14)
                action = elastic.matmat if trans == "N" else elastic.rmatmat
                np.testing.assert_allclose(action(batch), directions, atol=2e-8, rtol=1e-8)
        finally:
            fe.blocks["rock"]["kind"] = previous
            torch.set_num_threads(threads)
        dual = np.random.default_rng(5).normal(size=problem.size)
        np.testing.assert_allclose(dual @ correction, vector @ problem.factor.solve(dual, trans="T"),
                                   atol=1e-9, rtol=1e-10)

    def test_scaled_parameter_training_decreases_loss_without_committing(self):
        problem, _, _ = self.problem()
        self.perturb(problem.network)
        before = float(problem.objective()[0].detach())
        frozen = problem.model.fe.u_n.copy()
        scale = neural_linearization(problem)[-1].rsqrt()
        u, _, record = train_increment(problem, 8, 1e-10, progress_every=8, parameter_scale=scale)
        self.assertLess(record["loss"], before)
        np.testing.assert_array_equal(problem.model.fe.u_n, frozen)
        np.testing.assert_allclose(u, problem.field()[0].detach().numpy(), atol=0, rtol=0)

    def test_trial_is_frozen_and_commit_uses_one_ground_update(self):
        problem, _, _ = self.problem()
        fe = problem.model.fe
        saved = {name: {key: b[key].copy() for key in ("sig", "plast", "active")}
                 for name, b in fe.blocks.items()}
        initial_u = fe.u_n.copy()
        self.perturb(problem.network)
        for _ in range(3):
            problem.objective()[0].backward()
        for name, block in fe.blocks.items():
            for key in saved[name]:
                np.testing.assert_array_equal(block[key], saved[name][key])
        np.testing.assert_array_equal(fe.u_n, initial_u)
        u, c = (value.detach().numpy() for value in problem.field())
        ground = fe.blocks["rock"]
        _, strain, stress = fe._block_state(ground, u)
        expected = axifem.integrate_axisym(tensor(strain.reshape(-1, 4)), tensor(stress.reshape(-1, 4)),
                                           ground["E"], ground["nu"], ground["mat"], ground["kind"])[0]
        lining = problem.support.recover_stress(u)
        problem.commit(u, c)
        np.testing.assert_allclose(ground["sig"][ground["active"]], expected.reshape(-1, 4, 4), atol=1e-13)
        np.testing.assert_array_equal(fe.blocks["liner"]["sig"][fe.blocks["liner"]["active"]], lining)
        # Resetting the neural reference in the same or changed event coordinates
        # does not reinterpret the newly committed physical field.
        reset = HistoryProblem(problem.model, problem.history, 0.75, problem.network)
        np.testing.assert_allclose(reset.field()[0].detach(), u, atol=2e-14, rtol=0)
        stage, history, _, _ = prepare_event(problem.model, problem.history, 2)
        moved = HistoryProblem(stage, history, 0, problem.network)
        np.testing.assert_allclose(moved.field()[0].detach(), stage.fe.u_n, atol=2e-14, rtol=0)
        np.testing.assert_array_equal(history.records[1].stress_bias, problem.history.records[1].stress_bias)

    def test_failed_trial_checkpoint_restores_anchor_references_and_field(self):
        problem, born, event = self.problem()
        accepted = snapshot(self.initial, self.initial_history, 0, 1, np.zeros_like(born),
                            {"face_m": 0.0, "support_front_m": -2.0})
        anchor = weights(problem.network)
        frozen = snapshot(problem.model, problem.history, 1, 0, born, event)
        self.perturb(problem.network)
        pending = {"state": frozen, "target": 0.5, "anchor": anchor, "network": weights(problem.network)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, accepted, anchor, pending,
                            {"elapsed_s": 0.0, "materials": material_spec(problem.model)})
            saved = torch.load(path, weights_only=True)
            restored = copy.deepcopy(self.initial)
            restore_snapshot(restored, saved["accepted"])
            np.testing.assert_array_equal(restored.fe.u_n, self.initial.fe.u_n)
            trial = build_trial(restored, GroundNetwork(width=6), saved["pending"], 0.5)
            for first, second in zip(problem.objective(), trial.objective()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertEqual(batch_response(trial, trial.field()[0].detach().numpy()),
                             batch_response(problem, problem.field()[0].detach().numpy()))
            changed_target = build_trial(restored, GroundNetwork(width=6), saved["pending"], 0.25)
            np.testing.assert_allclose(changed_target.field()[0].detach(),
                                       frozen["arrays"]["u"], atol=2e-14, rtol=0)
            self.assertEqual(len(changed_target.history.records), 2)

    def test_explicit_budget_acceptance_and_parallel_training_restart(self):
        with tempfile.TemporaryDirectory(prefix="history-budget-") as directory:
            for accept in (False, True):
                args = SimpleNamespace(
                    mesh="coarse", round_length=0.4, gap=0.6, rounds=1,
                    width=6, seed=17, face_patch_span=None, front_local=False,
                    support_patch_span=None, support_local=False, initial_network=None,
                    device="cpu", cpu_threads=2, resume=False, resume_from=None,
                    head_steps=0, head_solver="svd", head_rcond=1e-9, scaled_lbfgs=False,
                    gauss_newton_steps=0, cg_iterations=8, iterations=1, tolerance=1e-12,
                    release_step=1.0, archive_rounds=[], accept_training_limit=accept,
                    output=Path(directory) / str(accept))
                self.assertEqual(run(args), 0 if accept else 2)
                saved = torch.load(args.output / "checkpoint.pt", weights_only=True)
                record = saved["run"]["increments"][-1]
                self.assertFalse(record["converged"])
                self.assertEqual(record["material_committed"], accept)
                self.assertEqual(record["acceptance_reason"], "training_limit" if accept else "not_accepted")
                self.assertEqual(saved["accepted"]["round"], 1 if accept else 0)
                self.assertEqual(saved["pending"] is None, accept)
                if accept:
                    args.resume = True
                    self.assertEqual(run(args), 0)
                    restored = torch.load(args.output / "checkpoint.pt", weights_only=True)
                    torch.testing.assert_close(restored["accepted"]["arrays"]["u"],
                                               saved["accepted"]["arrays"]["u"], rtol=0, atol=0)
                    for first, second in zip(saved["accepted"]["references"], restored["accepted"]["references"]):
                        torch.testing.assert_close(first["stress_bias"], second["stress_bias"], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_same_field_and_parameter_gradient(self):
        cpu, _, _ = self.problem()
        self.perturb(cpu.network)
        gpu, _, _ = self.problem(device="cuda")
        gpu.network.load_state_dict(cpu.network.state_dict())
        for first, second in zip(cpu.objective(), gpu.objective()):
            torch.testing.assert_close(first, second.cpu(), rtol=1e-9, atol=2e-11)
        gradients = [torch.cat([x.ravel().cpu() for x in torch.autograd.grad(p.objective()[0], p.network.parameters())])
                     for p in (cpu, gpu)]
        torch.testing.assert_close(*gradients, rtol=2e-8, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
