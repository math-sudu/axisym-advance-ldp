"""Incremental neural equilibrium with stress-free lining activation.

The coordinate network predicts displacement increments. Quadrature, the
non-associated Mohr--Coulomb return, and the exterior trace map define the
discrete weak equilibrium residual. Training uses no target displacement or
stress fields. Committed stresses are held fixed throughout an increment.

The neural histories continue case B from its saved initial equilibrium.
The construction runner records joint neural/material states for the manuscript.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from neural_model import ROOT, initial_model, corrected_kernel
import runtime as base
from solver import restore

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn
from wave2_src import axifem


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


class ElasticScale(torch.autograd.Function):
    """Fixed residual scaling, including the transpose solve in its gradient."""

    @staticmethod
    def forward(ctx, residual, factor):
        ctx.factor = factor
        return tensor(factor.solve(residual.detach().cpu().numpy())).to(residual.device)

    @staticmethod
    def backward(ctx, gradient):
        return tensor(ctx.factor.solve(gradient.detach().cpu().numpy(), trans="T")).to(gradient.device), None


class IncrementNetwork(nn.Module):
    def __init__(self, width=64, seed=17):
        super().__init__()
        torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(10, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 4),
        ).double()
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def prepare(self, coordinates):
        r, z, lining = coordinates.unbind(dim=1)
        features = torch.stack([
            r / 4, z / 26, torch.tanh(r - 1), torch.tanh(z),
            torch.tanh(2 * z), torch.tanh(z + 0.6),
            torch.exp(-(r - 1).square()),
            torch.exp(-z.square()), torch.sin(z), torch.cos(z),
        ], dim=1)
        return features, (lining * (1 - r))[:, None]

    def evaluate(self, prepared):
        features, enrichment = prepared
        values = self.layers(features)
        # The enrichment vanishes at the bonded rock--lining interface.
        # Its independent slope represents the material strain jump while
        # preserving displacement compatibility exactly.
        values = values[:, :2] + enrichment * values[:, 2:]
        return values.reshape(-1)

    def forward(self, coordinates):
        return self.evaluate(self.prepare(coordinates))


class LocalIncrementNetwork(nn.Module):
    """Overlapping neural fields resolve the face and installation front."""

    def __init__(self, width=32, seed=17, moving=False, face_patch=None):
        super().__init__()
        centres = [-10, -5, -2, -1, 0, 1, 3, 7, 14, 26]
        spans = [5.5, 5.5, 3.3, 1.1, 1.1, 2.2, 4.4, 7.7, 13.2, 13.2]
        if moving:
            centres = [-26, -14, -7, -3, -1, 0, 1, 3, 7, 14, 26]
            spans = [13.2, 13.2, 7.7, 4.4, 1.1, 1.1, 2.2, 4.4, 7.7, 13.2, 13.2]
        self.register_buffer("centres", tensor(centres))
        self.register_buffer("spans", tensor(spans))
        self.branches = nn.ModuleList([
            IncrementNetwork(width=width, seed=seed + j) for j in range(len(centres))])
        if face_patch is not None:
            self.register_buffer("patch_centre", tensor(face_patch))
            self.register_buffer("patch_span", tensor(0.2))
            self.branches.append(IncrementNetwork(width=width, seed=seed + len(centres)))

    def prepare(self, coordinates):
        local_z = (coordinates[:, 1, None] - self.centres) / self.spans
        inside = local_z.abs() < 1
        weights = torch.where(inside, torch.cos(torch.pi / 2 * local_z).square(), 0)
        total = weights.sum(dim=1, keepdim=True)
        if (total <= 0).any():
            raise ValueError("Local neural supports do not cover the model coordinates")
        weights = weights / total
        prepared = []
        for j, branch in enumerate(self.branches[:len(self.centres)]):
            indices = torch.nonzero(inside[:, j], as_tuple=True)[0]
            local = torch.stack([coordinates[indices, 0], local_z[indices, j],
                                 coordinates[indices, 2]], dim=1)
            prepared.append((indices, branch.prepare(local), weights[indices, j, None]))
        if hasattr(self, "patch_centre"):
            local_z = (coordinates[:, 1] - self.patch_centre) / self.patch_span
            indices = torch.nonzero(local_z.abs() < 1, as_tuple=True)[0]
            local = torch.stack([coordinates[indices, 0], local_z[indices],
                                 coordinates[indices, 2]], dim=1)
            patch_weight = torch.cos(torch.pi / 2 * local_z[indices]).square()[:, None]
            prepared.append((indices, self.branches[-1].prepare(local), patch_weight))
        count = max(len(indices) for indices, _, _ in prepared)
        indices_batch = torch.zeros((len(prepared), count), dtype=torch.long, device=coordinates.device)
        features_batch = coordinates.new_zeros((len(prepared), count, 10))
        enrichment_batch = coordinates.new_zeros((len(prepared), count, 1))
        weights_batch = coordinates.new_zeros((len(prepared), count, 1))
        for j, (indices, (features, enrichment), weight) in enumerate(prepared):
            size = len(indices)
            indices_batch[j, :size] = indices
            features_batch[j, :size] = features
            enrichment_batch[j, :size] = enrichment
            weights_batch[j, :size] = weight
        return coordinates, prepared, (indices_batch, features_batch, enrichment_batch, weights_batch)

    def evaluate(self, prepared):
        coordinates, _, (indices, values, enrichment, weights) = prepared
        # Evaluate the independent local networks in one batch. This preserves
        # their parameters and supports while avoiding repeated device launches
        # in every neural Jacobian-vector product.
        for layer in (0, 2, 4, 6):
            matrix = torch.stack([branch.layers[layer].weight for branch in self.branches])
            bias = torch.stack([branch.layers[layer].bias for branch in self.branches])
            values = torch.bmm(values, matrix.transpose(1, 2)) + bias[:, None, :]
            if layer != 6:
                values = torch.tanh(values)
        values = (values[:, :, :2] + enrichment * values[:, :, 2:]) * weights
        output = coordinates.new_zeros((len(coordinates), 2)).index_add(
            0, indices.reshape(-1), values.reshape(-1, 2))
        return output.reshape(-1)

    def forward(self, coordinates):
        return self.evaluate(self.prepare(coordinates))


class IncrementProblem:
    """One frozen material history and active geometry, with a trainable field."""

    def __init__(self, model, release, network, coordinate_origin=0.0, anchor_network=None):
        self.model, self.network = model, network
        device = next(network.parameters()).device
        def on_device(value):
            return tensor(value).to(device)
        fe, solver = model.fe, model.solver
        self.if_dofs = solver.ext.if_dofs
        active = fe.active_dof_mask()
        active[self.if_dofs] = True
        free = active & ~fe.dirichlet
        free[self.if_dofs] = False
        self.interior = np.flatnonzero(free)
        self.B = on_device(solver._maps["B"])
        self.T = on_device(solver._maps["T"])
        self.u_committed = on_device(fe.u_n.copy())
        self.c_committed = on_device(solver.c.copy())
        self.force = on_device(-fe.f_ext_fix + (1 - release) * fe.f_release - solver.F0)
        self.force_scale = max(np.linalg.norm(fe.f_release), np.linalg.norm(solver.F0), fe.p0)
        lining_nodes = np.zeros(model.mesh.n_nodes)
        lining = fe.blocks["liner"]
        lining_nodes[model.mesh.conn[lining["elems"][lining["active"]]].ravel()] = 1
        self.coordinates = on_device(np.column_stack([
            model.mesh.node_rho / base.scen.A,
            (model.mesh.node_z - coordinate_origin) / base.scen.A,
            lining_nodes]))
        self.prepared = network.prepare(self.coordinates)
        with torch.no_grad():
            self.anchor = (network if anchor_network is None else anchor_network).evaluate(self.prepared)
        self.blocks = []
        for block in fe.blocks.values():
            elements = block["elems"][block["active"]]
            if elements.size:
                self.blocks.append({
                    "dofs": torch.as_tensor(fe.m.dof_all[elements], device=device),
                    "B": on_device(fe.m.B_all[elements]),
                    "W": on_device(fe.m.W_all[elements]),
                    "stress": on_device(block["sig"][block["active"]].copy()),
                    "E": block["E"], "nu": block["nu"],
                    "mat": block["mat"], "kind": block["kind"],
                })
        kinds = {name: block["kind"] for name, block in fe.blocks.items()}
        try:
            for block in fe.blocks.values():
                block["kind"] = "elastic"
            _, stiffness = fe.internal_force(fe.u_n, want_tangent=True)
        finally:
            for name, block in fe.blocks.items():
                block["kind"] = kinds[name]
        B = sp.csr_matrix(solver._maps["B"])
        i, g = self.interior, self.if_dofs
        Kgg = B.T @ (stiffness[g][:, g] @ B) - B.T @ sp.csr_matrix(solver._maps["T"][g])
        elastic = sp.bmat([
            [stiffness[i][:, i], stiffness[i][:, g] @ B],
            [B.T @ stiffness[g][:, i], Kgg],
        ], format="csc")
        self.factor = spla.splu(elastic)
        # Normalize from the committed material state, independently of a
        # transferred neural prediction for the next displacement increment.
        self.displacement_scale = 0.0
        with torch.no_grad():
            residual = self.residual()[0]
        correction = self.factor.solve(residual.cpu().numpy())
        self.displacement_scale = max(np.max(np.abs(correction)), np.finfo(float).eps)
        self.initial_correction_norm = max(np.linalg.norm(correction), np.finfo(float).eps)
        self.initial_force_norm = max(float(residual.norm()), np.finfo(float).eps)

    def field(self):
        increment = self.displacement_scale * (self.network.evaluate(self.prepared) - self.anchor)
        c = self.c_committed + self.B.T @ increment[self.if_dofs]
        u = self.u_committed.clone()
        u[self.interior] = u[self.interior] + increment[self.interior]
        u[self.if_dofs] = self.B @ c
        return u, c

    def residual(self):
        u, c = self.field()
        force = torch.zeros_like(u)
        for block in self.blocks:
            du = (u - self.u_committed)[block["dofs"]]
            strain = torch.einsum("egij,ej->egi", block["B"], du)
            stress = axifem.integrate_axisym(
                strain.reshape(-1, 4), block["stress"].reshape(-1, 4),
                block["E"], block["nu"], block["mat"], block["kind"])[0]
            element_force = torch.einsum(
                "egij,egi,eg->ej", block["B"], stress.reshape_as(strain), block["W"])
            force = force.index_add(0, block["dofs"].reshape(-1), element_force.reshape(-1))
        force = force + self.force - self.T @ c
        reduced = torch.cat([force[self.interior], self.B.T @ force[self.if_dofs]])
        return reduced, u, c

    def objective(self):
        residual, u, c = self.residual()
        correction = ElasticScale.apply(residual, self.factor)
        loss = 0.5 * ((correction / self.initial_correction_norm).square().sum()
                      + (residual / self.initial_force_norm).square().sum())
        return loss, residual, u, c


def lining_sections(model, displacement, selected=None):
    """Trial hoop thrust at axial Gauss sections, integrated through thickness."""
    block = model.fe.blocks["liner"]
    active = block["active"].copy()
    if selected is not None:
        active &= selected
    elements = block["elems"][active]
    if not elements.size:
        return np.empty(0), np.empty(0)
    du = (np.asarray(displacement) - model.fe.u_n)[model.mesh.dof_all[elements]]
    strain = np.einsum("egij,ej->egi", model.mesh.B_all[elements], du)
    stress = axifem.integrate_axisym(
        tensor(strain.reshape(-1, 4)), tensor(block["sig"][active].reshape(-1, 4)),
        block["E"], block["nu"], block["mat"], block["kind"])[0].numpy()
    z, inverse = np.unique(np.round(model.mesh.gp_z[elements].ravel() / base.scen.A, 12),
                           return_inverse=True)
    thickness = model.mesh.e_rho1[elements] - model.mesh.e_rho0[elements]
    contributions = -1000 * stress[:, 2] * np.repeat(thickness / 2, 4)
    thrust = np.bincount(inverse, weights=contributions, minlength=len(z))
    return z, thrust


def physical_quantities(model, displacement, face, born):
    """Observables affected by the current neural construction increment."""
    wall_mm = -1000 * np.asarray(displacement)[2 * model.wall_nodes]
    z, thrust = lining_sections(model, displacement, born)
    peak = int(thrust.argmax()) if thrust.size else None
    mean = None
    if thrust.size:
        block = model.fe.blocks["liner"]
        elements = block["elems"][block["active"] & born]
        _, first = np.unique(np.round(model.mesh.gp_z[elements].ravel() / base.scen.A, 12),
                             return_index=True)
        weights = (model.mesh.e_z1[elements] - model.mesh.e_z0[elements])[first // 4] / 2
        mean = float(np.average(thrust, weights=weights))
    return {
        "face_convergence_mm": float(np.interp(face, model.wall_z, wall_mm)),
        "new_lining_mean_thrust_kN_per_m": mean,
        "new_lining_peak_thrust_kN_per_m": float(thrust[peak]) if peak is not None else None,
        "new_lining_peak_z_over_a": float(z[peak]) if peak is not None else None,
        "new_lining_section_z_over_a": z.tolist(),
        "new_lining_section_thrust_kN_per_m": thrust.tolist(),
    }


def train_increment(problem, iterations, tolerance, progress_every=200, observe=None, parameter_scale=None):
    with torch.no_grad():
        loss, residual, u, c = problem.objective()
    relative = float(residual.norm() / problem.force_scale)
    if relative <= tolerance or iterations == 0:
        return u.cpu().numpy(), c.cpu().numpy(), {
            "evaluations": 0, "loss": float(loss), "relative_force": relative,
            "converged": relative <= tolerance, "history": [], "response_history": [],
        }
    parameters = list(problem.network.parameters())
    if parameter_scale is not None:
        original = torch.cat([value.detach().ravel() for value in parameters])
        coordinate = torch.nn.Parameter(torch.zeros_like(original))

        def update_parameters():
            offset = 0
            with torch.no_grad():
                value = original + parameter_scale * coordinate
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.copy_(value[offset:offset + size].reshape_as(parameter))
                    offset += size

    optimizer = torch.optim.LBFGS(
        parameters if parameter_scale is None else [coordinate], lr=1, max_iter=min(progress_every, iterations),
        max_eval=progress_every * 2,
        tolerance_grad=1e-12, tolerance_change=1e-15,
        history_size=50, line_search_fn="strong_wolfe")
    evaluations, history, response_history = 0, [], []

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        if parameter_scale is not None:
            update_parameters()
            problem.network.zero_grad(set_to_none=True)
        loss, residual, _, _ = problem.objective()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite neural equilibrium loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in problem.network.parameters()):
            raise FloatingPointError("Nonfinite gradient through the material return")
        if parameter_scale is not None:
            coordinate.grad = parameter_scale * torch.cat([p.grad.ravel() for p in parameters])
        if evaluations % progress_every == 0:
            row = {"evaluation": evaluations, "loss": float(loss.detach()),
                   "relative_force": float(residual.detach().norm() / problem.force_scale)}
            history.append(row)
            print(json.dumps(row), flush=True)
        evaluations += 1
        return loss

    for first in range(0, iterations, progress_every):
        optimizer.param_groups[0]["max_iter"] = min(progress_every, iterations - first)
        optimizer.step(closure)
        if parameter_scale is not None:
            update_parameters()
        with torch.no_grad():
            loss, residual, u, c = problem.objective()
        relative = float(residual.norm() / problem.force_scale)
        if observe is not None:
            response_history.append({"evaluation": evaluations, "relative_force": relative,
                                     **observe(u.cpu().numpy())})
        if relative <= tolerance:
            break
    return u.cpu().numpy(), c.cpu().numpy(), {
        "evaluations": evaluations, "loss": float(loss), "relative_force": relative,
        "converged": relative <= tolerance, "history": history,
        "response_history": response_history,
    }


def prepare_first_round():
    model, source = initial_model(mesh="coarse")
    elements = model._rings(0, model.dstep)
    model.fe.f_release = model.fe.removal_force("rock", elements)
    model.fe.set_active("rock", elements, False)
    before = model.fe.blocks["liner"]["active"].copy()
    model._liner_activation(model.dstep)
    born = model.fe.blocks["liner"]["active"] & ~before
    if not born.any():
        raise ValueError("The prototype must contain newly activated lining")
    if np.any(model.fe.blocks["liner"]["sig"][born] != 0):
        raise ValueError("New lining must start stress-free")
    return model, source, born


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--architecture", choices=["global", "local"], default="local")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--release-step", type=float, default=1 / 16)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an unaccepted first increment from this output directory")
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis/pinn_activation")
    args = parser.parse_args()
    if not (1 <= args.substeps and 0 < args.release_step <= 1
            and args.substeps * args.release_step <= 1):
        parser.error("The requested release fractions must lie in (0, 1]")
    started = time.perf_counter()
    with corrected_kernel():
        model, source, born = prepare_first_round()
        network = LocalIncrementNetwork() if args.architecture == "local" else IncrementNetwork()
        previous = None
        if args.resume:
            previous = json.loads((args.output / "prototype.json").read_text(encoding="utf-8"))
            if (args.substeps != 1 or len(previous["increments"]) != 1
                    or previous["increments"][0]["converged"]
                    or previous["increments"][0]["release_fraction"] != args.release_step
                    or previous["architecture"] != args.architecture + "_displacement_with_lining_enrichment"
                    or previous["initial_state"] != source.relative_to(ROOT).as_posix()):
                raise ValueError("Resume requires the same unaccepted first increment and architecture")
        result = {"method": "incremental_discrete_weak_residual_PINN",
                  "architecture": args.architecture + "_displacement_with_lining_enrichment",
                  "initial_state": source.relative_to(ROOT).as_posix(),
                  "field_supervision": False, "new_lining_elements": int(born.sum()),
                  "nodes": model.mesh.n_nodes, "elements": model.mesh.n_elems,
                  "settings": {k: v for k, v in vars(args).items() if k != "output"},
                  "increments": []}
        args.output.mkdir(parents=True, exist_ok=True)
        for step in range(1, args.substeps + 1):
            release = step * args.release_step
            problem = IncrementProblem(model, release, network)
            if previous is not None:
                # Construct the zero-output anchor and its scales before
                # restoring weights: the committed material state is unchanged.
                network.load_state_dict(torch.load(args.output / "network.pt", weights_only=True))
            u, c, record = train_increment(problem, args.iterations, args.tolerance)
            if previous is not None:
                prior = previous["increments"][0]
                for row in record["history"]:
                    row["evaluation"] += prior["evaluations"]
                record["history"] = prior["history"] + record["history"]
                record["evaluations"] += prior["evaluations"]
                result["optimizer_restart_from_evaluation"] = prior["evaluations"]
            record["release_fraction"] = release
            result["increments"].append(record)
            np.savez_compressed(args.output / f"trial-{step:03d}.npz", displacement_m=u, c_reduced=c)
            if not record["converged"]:
                break
            model.fe.commit(u)
            model.solver.c = c
            wall = -model.fe.u_n[2 * model.wall_nodes] / model.u_max
            model.history = [{"k": 1, "x_f": model.dstep, "wall": wall,
                              "wall_s": time.perf_counter() - started, "info": record}]
            record["new_lining_peak_compressive_hoop_stress_MPa"] = float(
                -model.fe.blocks["liner"]["sig"][born, :, 2].min())
            record["plastic_quadrature_points"] = int(model.fe.blocks["rock"]["plast"].sum())
            arrays = base.state_arrays(model)
            arrays["c_reduced"] = c
            np.savez_compressed(args.output / f"accepted-{step:03d}.npz", **arrays)
        torch.save(network.state_dict(), args.output / "network.pt")
        result["elapsed_s"] = time.perf_counter() - started + (
            previous["elapsed_s"] if previous is not None else 0)
        result["completed_requested_increments"] = (
            len(result["increments"]) == args.substeps and result["increments"][-1]["converged"])
        result["first_round_completed"] = (result["completed_requested_increments"]
                                             and args.substeps * args.release_step == 1)
        (args.output / "prototype.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "increments"}), flush=True)
        return 0 if result["completed_requested_increments"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
