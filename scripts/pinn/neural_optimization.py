"""Damped least-squares training of the current neural displacement field.

The committed constitutive history stays fixed. The Jacobian composes the
material tangent with the neural displacement Jacobian. Both output-layer
and matrix-free full-network updates minimize the same weak-equilibrium
objective used by L-BFGS, without displacement or stress targets.
"""
from __future__ import annotations

import numpy as np
import json
import scipy.sparse as sp
import torch


class ReducedNeuralField(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.network = problem.network
        self.problem = problem

    def forward(self):
        problem = self.problem
        displacement = problem.displacement_scale * self.network.evaluate(problem.prepared)
        return torch.cat([displacement[problem.interior],
                          problem.B.T @ displacement[problem.if_dofs]])


def neural_linearization(problem):
    """Matrix-free Gauss--Newton products for all neural parameters."""
    module = ReducedNeuralField(problem)
    names, parameters = zip(*module.named_parameters())
    parameters = tuple(value.detach() for value in parameters)
    sizes = [value.numel() for value in parameters]

    def unflatten(vector):
        return tuple(value.reshape_as(parameter)
                     for value, parameter in zip(vector.split(sizes), parameters))

    def field(values):
        return torch.func.functional_call(module, dict(zip(names, values)), ())

    _, pullback = torch.func.vjp(field, parameters)
    with torch.no_grad():
        loss, residual, displacement, _ = problem.objective()
    tangent = reduced_tangent(problem, displacement.cpu().numpy()).tocsr()
    device = residual.device
    matrix = torch.sparse_csr_tensor(tangent.indptr, tangent.indices, tangent.data,
                                    size=tangent.shape, device=device)
    transpose = matrix.transpose(0, 1).to_sparse_csr()

    def metric(vector):
        correction = problem.factor.solve(vector.detach().cpu().numpy())
        scaled = problem.factor.solve(correction, trans="T") / problem.initial_correction_norm ** 2
        return vector / problem.initial_force_norm ** 2 + torch.as_tensor(scaled, device=device)

    def adjoint(vector):
        cotangent = torch.sparse.mm(transpose, metric(vector)[:, None]).ravel()
        values = pullback(cotangent)[0]
        return torch.cat([value.reshape(-1) for value in values]).detach()

    def normal(vector):
        _, direction = torch.func.jvp(field, (parameters,), (unflatten(vector),))
        force = torch.sparse.mm(matrix, direction[:, None]).ravel()
        return adjoint(force)

    # The squared adjoints of independent Rademacher vectors estimate the
    # diagonal of J.T @ J without forming the neural Jacobian. This only
    # scales the Krylov solve; each matrix-vector product remains exact.
    generator = torch.Generator(device=device).manual_seed(31)
    diagonal = torch.zeros(sum(sizes), device=device, dtype=residual.dtype)
    for _ in range(8):
        signs = torch.randint(0, 2, (2, len(residual)), generator=generator,
                              device=device).to(residual.dtype) * 2 - 1
        scaled = problem.factor.solve(signs[1].cpu().numpy(), trans="T")
        metric_signs = (signs[0] / problem.initial_force_norm
                        + torch.as_tensor(scaled, device=device) / problem.initial_correction_norm)
        cotangent = torch.sparse.mm(transpose, metric_signs[:, None]).ravel()
        values = pullback(cotangent)[0]
        sample = torch.cat([value.reshape(-1) for value in values]).detach()
        diagonal += sample.square() / 8
    diagonal.clamp_min_(torch.finfo(diagonal.dtype).eps * diagonal.max())
    return normal, adjoint(residual), float(loss), parameters, unflatten, diagonal


def refine_network(problem, steps, tolerance, observe=None, cg_iterations=40):
    """Damped matrix-free Gauss--Newton training of the existing network."""
    history, damping = [], 1e-3
    for iteration in range(steps):
        with torch.no_grad():
            if float(problem.residual()[0].norm() / problem.force_scale) <= tolerance:
                break
        normal, gradient, loss_before, parameters, unflatten, diagonal = neural_linearization(problem)
        delta = torch.zeros_like(gradient)
        residual = -gradient
        preconditioned = residual / ((1 + damping) * diagonal)
        direction = preconditioned.clone()
        square = residual @ preconditioned
        initial_square = square.clone()
        for inner in range(cg_iterations):
            action = normal(direction) + damping * diagonal * direction
            denominator = direction @ action
            if not torch.isfinite(denominator) or denominator <= 0:
                raise FloatingPointError("Invalid damped neural Gauss--Newton direction")
            alpha = square / denominator
            delta += alpha * direction
            residual -= alpha * action
            preconditioned = residual / ((1 + damping) * diagonal)
            next_square = residual @ preconditioned
            if next_square <= 1e-4 * initial_square:
                break
            direction = preconditioned + (next_square / square) * direction
            square = next_square
        original = tuple(value.clone() for value in parameters)
        update = unflatten(delta)
        accepted = False
        with torch.no_grad():
            for backtrack in range(16):
                fraction = 0.5 ** backtrack
                for parameter, value, change in zip(problem.network.parameters(), original, update):
                    parameter.copy_(value + fraction * change)
                loss, force, u, _ = problem.objective()
                if torch.isfinite(loss) and float(loss) < loss_before:
                    accepted = True
                    break
            if not accepted:
                for parameter, value in zip(problem.network.parameters(), original):
                    parameter.copy_(value)
                damping *= 10
                continue
        row = {"iteration": iteration + 1, "loss": float(loss),
               "relative_force": float(force.norm() / problem.force_scale),
               "step_fraction": fraction, "damping": damping, "cg_iterations": inner + 1}
        if observe is not None:
            row.update(observe(u.cpu().numpy()))
        history.append(row)
        print(json.dumps({key: row[key] for key in (
            "iteration", "loss", "relative_force", "face_convergence_mm",
            "new_lining_peak_thrust_kN_per_m") if key in row}), flush=True)
        damping = max(damping / 3, 1e-8) if backtrack == 0 else damping * 3
        if row["relative_force"] <= tolerance:
            break
    return history


def head_basis(problem, face_only=False):
    """Reduced displacement Jacobian in output weight/bias storage order."""
    network = problem.network
    if hasattr(network, "branches"):
        _, prepared, _ = problem.prepared
        branches = network.branches
    else:
        coordinates = problem.coordinates
        prepared = [(torch.arange(len(coordinates), device=coordinates.device),
                     problem.prepared, coordinates.new_ones((len(coordinates), 1)))]
        branches = [network]
    if face_only:
        if not hasattr(network, "patch_centre"):
            raise ValueError("Face-only output updates require a local face patch")
        branches, prepared = branches[-1:], prepared[-1:]
    rows, cols, data, parameters = [], [], [], []
    offset = 0
    for branch, (indices, (features, enrichment), weights) in zip(branches, prepared):
        head = branch.layers[-1]
        with torch.no_grad():
            hidden = branch.layers[:-1](features).cpu().numpy()
        n, width = hidden.shape
        indices = indices.cpu().numpy()
        weight = weights.detach().cpu().numpy().ravel()
        enrichment = enrichment.detach().cpu().numpy().ravel()
        values = np.column_stack([hidden, np.ones(n)])
        for output in range(4):
            coefficient = weight * (enrichment if output >= 2 else 1.0)
            selected = np.flatnonzero(coefficient != 0)
            row = np.repeat(2 * indices[selected] + output % 2, width + 1)
            column = np.r_[offset + output * width + np.arange(width),
                           offset + 4 * width + output]
            rows.append(row)
            cols.append(np.tile(column, len(selected)))
            data.append((values[selected] * coefficient[selected, None]).ravel())
        parameters.extend([head.weight, head.bias])
        offset += head.weight.numel() + head.bias.numel()
    basis = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(problem.u_committed), offset)).tocsr()
    basis *= problem.displacement_scale
    trace = sp.csr_matrix(problem.model.solver._maps["B"])
    return sp.vstack([basis[problem.interior], trace.T @ basis[problem.if_dofs]],
                     format="csc"), parameters


def reduced_tangent(problem, displacement):
    fe = problem.model.fe
    floor = fe.tang_floor
    try:
        # The output-weight damping supplies regularization. Retain the exact
        # constitutive derivative here, including the perfectly plastic apex.
        fe.tang_floor = 0.0
        _, tangent = fe.internal_force(displacement, want_tangent=True)
    finally:
        fe.tang_floor = floor
    maps = problem.model.solver._maps
    trace = sp.csr_matrix(maps["B"])
    interior, interface = problem.interior, problem.if_dofs
    return sp.bmat([
        [tangent[interior][:, interior], tangent[interior][:, interface] @ trace],
        [trace.T @ tangent[interface][:, interior],
         trace.T @ tangent[interface][:, interface] @ trace
         - trace.T @ sp.csr_matrix(maps["T"][interface])],
    ], format="csc")


def head_system(problem, basis):
    """Normal matrix and gradient of the existing two-term physical loss."""
    with torch.no_grad():
        loss, residual, displacement, _ = problem.objective()
    jacobian = (reduced_tangent(problem, displacement.cpu().numpy()) @ basis).toarray()
    scaled = problem.factor.solve(jacobian)
    correction = problem.factor.solve(residual.cpu().numpy())
    device = residual.device
    jacobian = torch.as_tensor(jacobian, device=device) / problem.initial_force_norm
    scaled = torch.as_tensor(scaled, device=device) / problem.initial_correction_norm
    error = torch.as_tensor(correction, device=device) / problem.initial_correction_norm
    normal = jacobian.T @ jacobian + scaled.T @ scaled
    gradient = (jacobian.T @ (residual / problem.initial_force_norm) + scaled.T @ error)
    return normal, gradient, float(loss), float(residual.norm() / problem.force_scale)


def refine_head(problem, steps, tolerance, observe=None, face_only=False):
    basis, parameters = head_basis(problem, face_only)
    damping, history = 1e-4, []
    for iteration in range(steps):
        normal, gradient, loss_before, relative_before = head_system(problem, basis)
        if relative_before <= tolerance:
            break
        original = torch.cat([p.detach().reshape(-1) for p in parameters])
        diagonal = torch.arange(len(gradient), device=gradient.device)
        normal[diagonal, diagonal] += damping * normal.diagonal().max()
        delta = torch.linalg.solve(normal, -gradient)
        accepted = False
        with torch.no_grad():
            for backtrack in range(16):
                trial = original + (0.5 ** backtrack) * delta
                offset = 0
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.copy_(trial[offset:offset + size].reshape_as(parameter))
                    offset += size
                loss, residual, u, _ = problem.objective()
                if torch.isfinite(loss) and float(loss) < loss_before:
                    accepted = True
                    break
            if not accepted:
                offset = 0
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.copy_(original[offset:offset + size].reshape_as(parameter))
                    offset += size
                break
        row = {"iteration": iteration + 1, "loss": float(loss),
               "relative_force": float(residual.norm() / problem.force_scale),
               "step_fraction": 0.5 ** backtrack, "damping": damping}
        if observe is not None:
            row.update(observe(u.cpu().numpy()))
        history.append(row)
        print(json.dumps({key: row[key] for key in ("iteration", "loss", "relative_force")}), flush=True)
        damping = max(damping / 3, 1e-10) if backtrack == 0 else damping * 3
        if row["relative_force"] <= tolerance:
            break
    return history
