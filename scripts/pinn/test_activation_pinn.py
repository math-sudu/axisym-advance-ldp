"""Checks for neural gradients and construction-state handling."""
import unittest
import copy
from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

torch.set_num_threads(1)

from activation_pinn import (ElasticScale, IncrementNetwork, LocalIncrementNetwork, IncrementProblem,
                             axifem, lining_sections, prepare_first_round, corrected_kernel, tensor, train_increment)
from run_construction import initial_model, load_checkpoint, save_checkpoint
from neural_optimization import head_basis, head_system, neural_linearization


class ActivationPinnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material_context = corrected_kernel()
        cls.material_context.__enter__()
        cls.model, _, cls.born = prepare_first_round()

    @classmethod
    def tearDownClass(cls):
        cls.material_context.__exit__(None, None, None)

    def test_residual_scaling_uses_transpose_gradient(self):
        matrix = np.array([[3., -2.], [1., 4.]])
        factor = spla.splu(sp.csc_matrix(matrix))
        residual = tensor([0.7, -0.2]).requires_grad_()
        direction = tensor([0.3, 0.9])
        value = (ElasticScale.apply(residual, factor) * direction).sum()
        value.backward()
        np.testing.assert_allclose(residual.grad, np.linalg.solve(matrix.T, direction),
                                   rtol=1e-12, atol=1e-12)

    def test_activation_and_trial_leave_committed_history_unchanged(self):
        fe = self.model.fe
        saved_u = fe.u_n.copy()
        saved = {name: block["sig"].copy() for name, block in fe.blocks.items()}
        network = IncrementNetwork(width=12)
        problem = IncrementProblem(self.model, 1 / 16, network)
        loss, _, u, _ = problem.objective()
        loss.backward()
        np.testing.assert_allclose(u.detach(), saved_u, atol=1e-14)
        np.testing.assert_array_equal(fe.u_n, saved_u)
        for name, block in fe.blocks.items():
            np.testing.assert_array_equal(block["sig"], saved[name])
        lining = fe.blocks["liner"]
        elements = lining["elems"][self.born]
        self.assertGreater(np.max(np.abs(saved_u[fe.m.dof_all[elements]])), 0)
        stress, _, _ = axifem.integrate_axisym(
            torch.zeros((len(elements) * 4, 4), dtype=torch.float64),
            tensor(lining["sig"][self.born].reshape(-1, 4)),
            lining["E"], lining["nu"], lining["mat"], lining["kind"])
        np.testing.assert_array_equal(stress, 0)

    def test_zero_lbfgs_budget_returns_the_field_and_actual_force_status(self):
        network = IncrementNetwork(width=12)
        problem = IncrementProblem(self.model, 0.125, network)
        with torch.no_grad():
            network.layers[-1].weight.fill_(0.005)
            expected_u, expected_c = problem.field()
        u, c, record = train_increment(problem, 0, 1e-12)
        np.testing.assert_array_equal(u, expected_u.numpy())
        np.testing.assert_array_equal(c, expected_c.numpy())
        self.assertFalse(record["converged"])
        self.assertEqual(record["evaluations"], 0)

    def test_neural_equilibrium_directional_derivative(self):
        network = LocalIncrementNetwork(width=12)
        problem = IncrementProblem(self.model, 1 / 16, network)
        parameter = network.branches[4].layers[-1].weight
        torch.manual_seed(8)
        direction = torch.randn_like(parameter)
        direction /= direction.norm()
        with torch.no_grad():
            parameter.add_(0.04 * torch.randn_like(parameter))
        loss = problem.objective()[0]
        gradient = torch.autograd.grad(loss, parameter)[0]
        h = 1e-5
        original = parameter.detach().clone()
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                parameter.copy_(original + sign * h * direction)
                values.append(float(problem.objective()[0]))
            parameter.copy_(original)
        finite_difference = (values[0] - values[1]) / (2 * h)
        derivative = float((gradient * direction).sum())
        np.testing.assert_allclose(derivative, finite_difference, rtol=2e-4, atol=1e-5)

    def test_thrust_integrates_radial_gauss_weights_without_radius_factor(self):
        block = self.model.fe.blocks["liner"]
        original = block["sig"].copy()
        try:
            block["sig"][self.born, :, 2] = -4.0
            _, thrust = lining_sections(self.model, self.model.fe.u_n, self.born)
            np.testing.assert_allclose(thrust, 4.0 * 0.2 * 1000, rtol=1e-12)
        finally:
            block["sig"] = original

    def test_checkpoint_preserves_neural_and_material_continuation(self):
        network = LocalIncrementNetwork(width=12)
        with torch.no_grad():
            network.branches[4].layers[-1].weight.fill_(0.02)
        problem = IncrementProblem(self.model, 0.125, network)
        expected = problem.objective()[1].detach().numpy()
        restored, _ = initial_model()
        restored_network = LocalIncrementNetwork(width=12)
        with tempfile.TemporaryDirectory(prefix="pinn-checkpoint-") as directory:
            path = Path(directory) / "checkpoint.pt"
            expected_predictor = {name: torch.ones_like(value) * 0.001
                                  for name, value in network.named_parameters()}
            save_checkpoint(path, self.model, network, {"elapsed_s": 0.0}, 1, 0.0625,
                            self.born, expected_predictor)
            _, index, release, born, predictor, _ = load_checkpoint(path, restored, restored_network)
        for name in expected_predictor:
            torch.testing.assert_close(predictor[name], expected_predictor[name], rtol=0, atol=0)
        self.assertEqual((index, release), (1, 0.0625))
        np.testing.assert_array_equal(born, self.born)
        np.testing.assert_array_equal(restored.fe.f_release, self.model.fe.f_release)
        np.testing.assert_array_equal(restored.fe.u_n, self.model.fe.u_n)
        for name, block in self.model.fe.blocks.items():
            for field in ("sig", "active", "plast"):
                np.testing.assert_array_equal(restored.fe.blocks[name][field], block[field])
        for key, value in network.state_dict().items():
            torch.testing.assert_close(value, restored_network.state_dict()[key], rtol=0, atol=0)
        actual = IncrementProblem(restored, 0.125, restored_network).objective()[1]
        np.testing.assert_array_equal(actual.detach().numpy(), expected)

    def test_transferred_increment_keeps_material_normalization_and_restarts(self):
        anchor = LocalIncrementNetwork(width=12)
        network = copy.deepcopy(anchor)
        baseline = IncrementProblem(self.model, 0.125, anchor)
        with torch.no_grad():
            network.branches[4].layers[-1].weight.fill_(0.02)
        problem = IncrementProblem(self.model, 0.125, network, anchor_network=anchor)
        self.assertEqual(problem.displacement_scale, baseline.displacement_scale)
        self.assertEqual(problem.initial_force_norm, baseline.initial_force_norm)
        self.assertGreater(float((problem.field()[0] - problem.u_committed).abs().max()), 0)
        with tempfile.TemporaryDirectory(prefix="pinn-reference-") as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, network, {"elapsed_s": 0.0}, 1, 0.0625,
                            self.born, field_anchor=anchor.state_dict())
            restored, _ = initial_model()
            restored_network = LocalIncrementNetwork(width=12)
            _, _, _, _, _, saved_anchor = load_checkpoint(path, restored, restored_network)
        restored_anchor = LocalIncrementNetwork(width=12)
        restored_anchor.load_state_dict(saved_anchor)
        actual = IncrementProblem(restored, 0.125, restored_network, anchor_network=restored_anchor)
        torch.testing.assert_close(problem.field()[0], actual.field()[0], rtol=0, atol=0)
        torch.testing.assert_close(problem.residual()[0], actual.residual()[0], rtol=0, atol=0)

    def test_output_layer_gradient_matches_the_physical_loss(self):
        network = LocalIncrementNetwork(width=12, face_patch=0.4)
        problem = IncrementProblem(self.model, 0.125, network)
        with torch.no_grad():
            network.branches[4].layers[-1].weight.fill_(0.005)
        for face_only in (True, False):
            with self.subTest(face_only=face_only):
                basis, parameters = head_basis(problem, face_only=face_only)
                _, gradient, _, _ = head_system(problem, basis)
                automatic = torch.cat([value.reshape(-1) for value in
                                       torch.autograd.grad(problem.objective()[0], parameters)])
                torch.testing.assert_close(gradient, automatic, rtol=1e-9, atol=1e-9)

    def test_zero_face_patch_preserves_the_existing_neural_field(self):
        network = LocalIncrementNetwork(width=12, moving=True)
        with torch.no_grad():
            for branch in network.branches:
                branch.layers[-1].weight.fill_(0.01)
        refined = LocalIncrementNetwork(width=12, moving=True, face_patch=0.4)
        loaded = refined.load_state_dict(network.state_dict(), strict=False)
        self.assertFalse(loaded.unexpected_keys)
        self.assertTrue(all(key.startswith(("patch_", "branches.11.")) for key in loaded.missing_keys))
        coordinates = IncrementProblem(self.model, 0.125, network).coordinates
        torch.testing.assert_close(network(coordinates), refined(coordinates), rtol=0, atol=0)

    def test_matrix_free_neural_gradient_and_normal_products(self):
        network = LocalIncrementNetwork(width=6, face_patch=0.4)
        problem = IncrementProblem(self.model, 0.125, network)
        with torch.no_grad():
            network.branches[4].layers[-1].weight.fill_(0.005)
        normal, gradient, _, _, _, diagonal = neural_linearization(problem)
        automatic = torch.cat([value.reshape(-1) for value in
                               torch.autograd.grad(problem.objective()[0], network.parameters())])
        torch.testing.assert_close(gradient, automatic, rtol=1e-9, atol=1e-9)
        a, b = torch.randn_like(gradient), torch.randn_like(gradient)
        torch.testing.assert_close(a @ normal(b), b @ normal(a), rtol=1e-10, atol=1e-10)
        self.assertGreater(float(a @ normal(a)), 0)
        self.assertTrue(torch.all(diagonal > 0))

    def test_moving_local_supports_cover_the_complete_construction_domain(self):
        network = LocalIncrementNetwork(width=12, moving=True)
        coordinates = tensor(np.column_stack([
            self.model.mesh.node_rho / 2,
            self.model.mesh.node_z / 2 - 14.4,
            np.zeros(self.model.mesh.n_nodes)]))
        self.assertTrue(torch.isfinite(network(coordinates)).all())

    def test_batched_local_network_preserves_fields_and_parameter_gradients(self):
        network = LocalIncrementNetwork(width=12, moving=True, face_patch=0.4)
        with torch.no_grad():
            for branch in network.branches:
                branch.layers[-1].weight.normal_(std=0.01)
        problem = IncrementProblem(self.model, 0.125, network)
        coordinates, branches, _ = problem.prepared
        reference = coordinates.new_zeros((len(coordinates), 2))
        for branch, (indices, inputs, weights) in zip(network.branches, branches):
            reference = reference.index_add(0, indices, branch.evaluate(inputs).reshape(-1, 2) * weights)
        actual = network.evaluate(problem.prepared)
        torch.testing.assert_close(actual, reference.reshape(-1), rtol=1e-12, atol=1e-14)
        direction = torch.randn_like(actual)
        gradients = [torch.autograd.grad((field.reshape(-1) * direction).sum(), network.parameters())
                     for field in (reference, actual)]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_matches_cpu_residual_and_parameter_gradient(self):
        cpu = LocalIncrementNetwork(width=12)
        gpu = LocalIncrementNetwork(width=12).cuda()
        problems = [IncrementProblem(self.model, 0.125, net) for net in (cpu, gpu)]
        with torch.no_grad():
            cpu.branches[4].layers[-1].weight.fill_(0.005)
        gpu.load_state_dict(cpu.state_dict())
        values, gradients = [], []
        for problem, net in zip(problems, (cpu, gpu)):
            loss, residual, displacement, _ = problem.objective()
            loss.backward()
            values.append([value.detach().cpu() for value in (loss, residual, displacement)])
            gradients.append(torch.cat([p.grad.reshape(-1).cpu() for p in net.parameters()]))
        for a, b in zip(*values):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(*gradients, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
