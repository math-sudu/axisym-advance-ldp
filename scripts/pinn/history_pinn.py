"""Ground neural increments with an installation-history support recovery layer.

The material history is fixed within a problem. Only active ground nodes are
neural unknowns; lining-only displacements and stresses follow from the complete
installation references. The elastic metric uses an augmented sparse solve,
so neither a dense Schur matrix nor a dense inverse is needed.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from activation_pinn import base, axifem, tensor, ElasticScale
from support_history_operator import SupportHistoryOperator, extend_installation_field


def actual_fronts(model):
    """Read axial boundaries from the active core and installed lining."""
    mesh, fe = model.mesh, model.fe
    rock, liner = fe.blocks["rock"], fe.blocks["liner"]
    core = rock["elems"][rock["active"]]
    core = core[mesh.e_rho0[core] < base.scen.A - 1e-9]
    support = liner["elems"][liner["active"]]
    return (float(mesh.e_z0[core].min()),
            float(mesh.e_z1[support].max()) if len(support) else None)


def prepare_event(model, history, index):
    """Stage deletion, birth field and references without changing the input.

    Installation uses the accepted field before any release. The new geometry
    is fixed for all release fractions of this event.
    """
    stage = copy.deepcopy(model)
    face = index * stage.dstep
    for position in (face - stage.dstep, face, face - stage.liner_d):
        if not np.isclose(stage.z_lines, position, rtol=0, atol=1e-9).any():
            raise ValueError("Construction endpoint is not a mesh boundary")
    removed = stage._rings(face - stage.dstep, face)
    stage.fe.f_release = stage.fe.removal_force("rock", removed)
    stage.fe.set_active("rock", removed, False)
    liner = stage.fe.blocks["liner"]
    before = liner["active"].copy()
    stage._liner_activation(face)
    born = liner["active"] & ~before
    elements = liner["elems"][born]
    # The dormant-node mask must be read before the new lining is active.
    liner["active"] = before
    installation = extend_installation_field(stage.fe, elements)
    updated = history.install(stage.fe, elements, installation, index) if len(elements) else history
    liner["active"] |= born
    stage.fe.u_n = installation
    operator = SupportHistoryOperator(stage.fe, updated)
    liner["sig"][liner["active"]] = operator.recover_stress(installation)
    fronts = actual_fronts(stage)
    return stage, updated, born, {"event_id": index, "face_m": fronts[0],
                                "support_front_m": fronts[1],
                                "born_elements": elements.tolist()}


class GroundNetwork(nn.Module):
    """Two displacement outputs with background, face and support-front fields."""

    front_windows = tuple((r, z, 0.55, 0.65)
                          for r in (0.25, 0.75, 1.25, 1.75) for z in (0, 0.5)) + ((1, 0, 0.2, 0.2),)
    support_windows = tuple((r, z, 0.55, 0.65)
                            for r in (1.25, 1.75) for z in (0, -0.5))

    def __init__(self, width=32, seed=17, face_patch_span=None, front_local=False,
                 support_patch_span=None, support_local=False):
        super().__init__()
        self.width, self.seed = width, seed
        self.face_patch_span = face_patch_span
        self.front_local = front_local
        self.support_patch_span = support_patch_span
        self.support_local = support_local
        self.register_buffer("centres", tensor([-26, -14, -7, -3, -1, 0, 1, 3, 7, 14, 26]))
        self.register_buffer("spans", tensor([13.2, 13.2, 7.7, 4.4, 1.1, 1.1, 2.2, 4.4, 7.7, 13.2, 13.2]))
        self.branches = nn.ModuleList()
        for j in range(len(self.centres) + 2 + int(face_patch_span is not None)
                       + len(self.front_windows) * int(front_local) + int(support_patch_span is not None)
                       + len(self.support_windows) * int(support_local)):
            torch.manual_seed(seed + j)
            layers = nn.Sequential(nn.Linear(8, width), nn.Tanh(),
                                   nn.Linear(width, width), nn.Tanh(),
                                   nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 2)).double()
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            self.branches.append(layers)

    @staticmethod
    def features(r, z):
        return torch.stack([r / 4, z, torch.tanh(r - 1), torch.tanh(z),
                            torch.exp(-(r - 1).square()), torch.exp(-z.square()),
                            torch.sin(z), torch.cos(z)], dim=1)

    @staticmethod
    def local_features(r, z):
        return torch.stack([r, z, torch.tanh(r), torch.tanh(z),
                            torch.exp(-r.square()), torch.exp(-z.square()),
                            torch.sin(r), torch.sin(z)], dim=1)

    def prepare(self, model, nodes, fronts):
        device = next(self.parameters()).device
        coordinates = tensor(np.column_stack([model.mesh.node_rho[nodes],
                                               model.mesh.node_z[nodes]]) / base.scen.A).to(device)
        r, z = coordinates.unbind(1)
        local = (z[:, None] - self.centres) / self.spans
        weights = torch.where(local.abs() < 1, torch.cos(torch.pi / 2 * local).square(), 0)
        if (weights.sum(1) <= 0).any():
            raise ValueError("Background neural supports do not cover the active ground")
        weights = weights / weights.sum(1, keepdim=True)
        prepared = [(r, local[:, j], weights[:, j], False) for j in range(len(self.centres))]
        # Each window spans several axial cells on both supported meshes.
        windows = [(front, 0.8) for front in fronts]
        if self.face_patch_span is not None:
            windows.append((fronts[0], self.face_patch_span))
        for front, span in windows:
            relative = (z - front / base.scen.A) / span if front is not None else z * 0
            weight = (torch.where(relative.abs() < 1,
                                  torch.cos(torch.pi / 2 * relative).square(), 0)
                      if front is not None else torch.zeros_like(z))
            prepared.append((r, relative, weight, False))
        local_windows = [(r_centre, fronts[0] / base.scen.A + z_offset, r_span, z_span)
                         for r_centre, z_offset, r_span, z_span in self.front_windows] if self.front_local else []
        if self.support_patch_span is not None:
            local_windows.append((1, fronts[1] / base.scen.A if fronts[1] is not None else None,
                                  self.support_patch_span, self.support_patch_span))
        if self.support_local:
            local_windows.extend((r_centre, fronts[1] / base.scen.A + z_offset if fronts[1] is not None else None,
                                  r_span, z_span) for r_centre, z_offset, r_span, z_span in self.support_windows)
        for r_centre, z_centre, r_span, z_span in local_windows:
            radial = (r - r_centre) / r_span
            axial = (z - z_centre) / z_span if z_centre is not None else z * 0
            weight = (torch.where((radial.abs() < 1) & (axial.abs() < 1),
                                  torch.cos(torch.pi / 2 * radial).square()
                                  * torch.cos(torch.pi / 2 * axial).square(), 0)
                      if z_centre is not None else torch.zeros_like(z))
            prepared.append((radial, axial, weight, True))
        sizes = [int(torch.count_nonzero(weight)) for _, _, weight, _ in prepared]
        count = max(sizes)
        indices = torch.zeros((len(prepared), count), dtype=torch.long, device=device)
        features = coordinates.new_zeros((len(prepared), count, 8))
        weight_batch = coordinates.new_zeros((len(prepared), count, 1))
        for j, ((radial, relative, weight, is_local), size) in enumerate(zip(prepared, sizes)):
            selected = torch.nonzero(weight, as_tuple=True)[0]
            indices[j, :size] = torch.as_tensor(nodes, device=device)[selected]
            feature_map = self.local_features if is_local else self.features
            features[j, :size] = feature_map(radial[selected], relative[selected])
            weight_batch[j, :size, 0] = weight[selected]
        return model.mesh.n_nodes, indices, features, weight_batch

    def evaluate(self, prepared):
        count, indices, values, weights = prepared
        for layer in (0, 2, 4, 6):
            matrix = torch.stack([branch[layer].weight for branch in self.branches])
            bias = torch.stack([branch[layer].bias for branch in self.branches])
            values = torch.bmm(values, matrix.transpose(1, 2)) + bias[:, None, :]
            if layer != 6:
                values = torch.tanh(values)
        return values.new_zeros((count, 2)).index_add(
            0, indices.ravel(), (values * weights).reshape(-1, 2)).ravel()

    def output_basis(self, prepared):
        """Exact nodal Jacobian for all output weights and biases."""
        count, indices, features, weights = prepared
        rows, columns, values, parameters = [], [], [], []
        offset = 0
        for branch, nodes, inputs, window in zip(self.branches, indices, features, weights):
            head = branch[-1]
            with torch.no_grad():
                hidden = branch[:-1](inputs).cpu().numpy()
            window = window.cpu().numpy().ravel()
            selected = np.flatnonzero(window)
            nodes = nodes.cpu().numpy()[selected]
            data = np.column_stack([hidden[selected], np.ones(len(selected))]) * window[selected, None]
            for component in range(2):
                rows.append(np.repeat(2 * nodes + component, self.width + 1))
                columns.append(np.tile(np.r_[offset + component * self.width + np.arange(self.width),
                                              offset + 2 * self.width + component], len(selected)))
                values.append(data.ravel())
            parameters.extend((head.weight, head.bias))
            offset += head.weight.numel() + head.bias.numel()
        basis = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                              shape=(2 * count, offset)).tocsr()
        return basis, parameters


class SupportRecovery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contact, operator, load):
        ctx.operator = operator
        return contact.new_tensor(operator.recover_internal(contact.detach().cpu().numpy(), load))

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_tensor(ctx.operator.recover_transpose(gradient.detach().cpu().numpy())), None, None


class SupportForce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, contact, operator, load):
        ctx.operator = operator
        return contact.new_tensor(operator.interface_force(contact.detach().cpu().numpy(), load))

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_tensor(ctx.operator.apply_tangent_transpose(gradient.detach().cpu().numpy())), None, None


class CondensedScale:
    """Top block of the augmented elastic inverse, including transpose solves."""

    def __init__(self, matrix, size):
        self.factor = spla.splu(matrix.tocsc())
        self.size, self.total = size, matrix.shape[0]

    def solve(self, residual, trans="N"):
        residual = np.asarray(residual)
        if residual.ndim == 2:
            # Bound the dense working set for the many output-layer directions.
            result = np.empty_like(residual, dtype=float)
            starts = range(0, residual.shape[1], 32)

            def solve_block(first):
                last = min(first + 32, residual.shape[1])
                rhs = np.zeros((self.total, last - first), order="F")
                rhs[:self.size] = residual[:, first:last]
                result[:, first:last] = self.factor.solve(rhs, trans=trans)[:self.size]

            workers = min(torch.get_num_threads(), len(starts))
            if workers > 1:
                # Factors are read-only; each solve owns its RHS and output slice.
                with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(solve_block, starts))
            else:
                for first in starts:
                    solve_block(first)
            return result
        rhs = np.zeros((self.total,) + residual.shape[1:])
        rhs[:self.size] = residual
        return self.factor.solve(rhs, trans=trans)[:self.size]


class HistoryProblem:
    """Frozen ground material history and affine support response for one release."""

    def __init__(self, model, history, release, network, operator=None):
        self.model, self.history, self.network = model, history, network
        self.release = release
        fe, solver, mesh = model.fe, model.solver, model.mesh
        device = next(network.parameters()).device
        def on_device(value):
            return tensor(value).to(device)
        self.support = operator if operator is not None else SupportHistoryOperator(fe, history)
        self.if_dofs = solver.ext.if_dofs
        ground = fe.blocks["rock"]
        elements = ground["elems"][ground["active"]]
        nodes = np.unique(mesh.conn[elements])
        active = np.zeros(fe.ndof, dtype=bool)
        active[mesh.dof_all[elements].ravel()] = True
        active[self.if_dofs] = True
        active &= ~fe.dirichlet
        active[self.if_dofs] = False
        self.interior = np.flatnonzero(active)
        self.B = on_device(solver._maps["B"])
        self.T = on_device(solver._maps["T"])
        if np.intersect1d(self.if_dofs, self.support.internal_dofs).size or np.any(
                solver._maps["T"][self.support.internal_dofs]):
            raise ValueError("Lining-only DOFs cannot carry the exterior map")
        self.u_committed = on_device(fe.u_n.copy())
        self.c_committed = on_device(solver.c.copy())
        self.force = on_device(-fe.f_ext_fix + (1 - release) * fe.f_release - solver.F0)
        self.internal_load = -self.force[self.support.internal_dofs].cpu().numpy()
        self.force_scale = max(np.linalg.norm(fe.f_release), np.linalg.norm(solver.F0), fe.p0)
        self.fronts = actual_fronts(model)
        self.prepared = network.prepare(model, nodes, self.fronts)
        with torch.no_grad():
            self.anchor = network.evaluate(self.prepared).clone()
        self.block = {"dofs": torch.as_tensor(mesh.dof_all[elements], device=device),
                      "B": on_device(mesh.B_all[elements]), "W": on_device(mesh.W_all[elements]),
                      "stress": on_device(ground["sig"][ground["active"]].copy()),
                      "E": ground["E"], "nu": ground["nu"], "mat": ground["mat"],
                      "kind": ground["kind"]}
        # P retains the original orthonormal exterior displacement space.
        trace = sp.csr_matrix(solver._maps["B"])
        selector = sp.eye(fe.ndof, format="csc")
        self.P = sp.hstack([selector[:, self.interior], selector[:, self.if_dofs] @ trace], format="csr")
        self.size = self.P.shape[1]
        extended = sp.hstack([self.P, selector[:, self.support.internal_dofs]], format="csc")
        kinds = {name: block["kind"] for name, block in fe.blocks.items()}
        try:
            for block in fe.blocks.values():
                block["kind"] = "elastic"
            _, stiffness = fe.internal_force(fe.u_n, want_tangent=True)
        finally:
            for name, block in fe.blocks.items():
                block["kind"] = kinds[name]
        exterior = sp.hstack([sp.csr_matrix((fe.ndof, len(self.interior))),
                              sp.csr_matrix(solver._maps["T"]),
                              sp.csr_matrix((fe.ndof, len(self.support.i)))], format="csc")
        self.factor = CondensedScale(extended.T @ (stiffness @ extended - exterior), self.size)
        self.displacement_scale = 0.0
        with torch.no_grad():
            residual = self.residual()[0]
        correction = self.factor.solve(residual.cpu().numpy())
        # Roundoff floors keep an unloaded anchor finite without amplifying
        # cancellation into an artificial objective. Loaded increments use
        # their actual initial residual/correction norms.
        eps = 128 * np.finfo(float).eps
        self.displacement_scale = max(np.max(np.abs(correction)), eps * base.scen.A)
        self.initial_correction_norm = max(np.linalg.norm(correction), eps * base.scen.A)
        self.initial_force_norm = max(float(residual.norm()), eps * self.force_scale)

    def field(self):
        increment = self.displacement_scale * (self.network.evaluate(self.prepared) - self.anchor)
        c = self.c_committed + self.B.T @ increment[self.if_dofs]
        u = self.u_committed.clone()
        u[self.interior] = u[self.interior] + increment[self.interior]
        u[self.if_dofs] = self.B @ c
        u[self.support.internal_dofs] = SupportRecovery.apply(
            u[self.support.contact_dofs], self.support, self.internal_load)
        return u, c

    def residual(self):
        u, c = self.field()
        block = self.block
        strain = torch.einsum("egij,ej->egi", block["B"], (u - self.u_committed)[block["dofs"]])
        stress = axifem.integrate_axisym(
            strain.reshape(-1, 4), block["stress"].reshape(-1, 4),
            block["E"], block["nu"], block["mat"], block["kind"])[0].reshape_as(strain)
        values = torch.einsum("egij,egi,eg->ej", block["B"], stress, block["W"])
        force = torch.zeros_like(u).index_add(0, block["dofs"].ravel(), values.ravel())
        contact = torch.as_tensor(self.support.contact_dofs, device=u.device)
        force = force.index_add(0, contact, SupportForce.apply(u[contact], self.support, self.internal_load))
        force = force + self.force - self.T @ c
        return torch.cat([force[self.interior], self.B.T @ force[self.if_dofs]]), u, c

    def objective(self):
        residual, u, c = self.residual()
        correction = ElasticScale.apply(residual, self.factor)
        loss = 0.5 * ((residual / self.initial_force_norm).square().sum()
                      + (correction / self.initial_correction_norm).square().sum())
        return loss, residual, u, c

    def tangent(self, displacement):
        fe = self.model.fe
        floor = fe.tang_floor
        try:
            fe.tang_floor = 0.0
            _, stiffness = fe.internal_force(displacement, want_tangent=True)
        finally:
            fe.tang_floor = floor
        P, support = self.P, self.support
        T = self.model.solver._maps["T"]

        def action(direction):
            lifted = P @ direction
            lifted[support.internal_dofs] = support.recover_direction(lifted[support.contact_dofs])
            return P.T @ (stiffness @ lifted - T @ direction[len(self.interior):])

        def transpose(cotangent):
            lifted = P @ cotangent
            value = stiffness.T @ lifted
            value[support.contact_dofs] += support.recover_transpose(value[support.internal_dofs])
            value[support.internal_dofs] = 0
            reduced = P.T @ value
            reduced[len(self.interior):] -= T.T @ lifted
            return reduced

        return spla.LinearOperator((self.size, self.size), matvec=action, rmatvec=transpose,
                                   matmat=action, rmatmat=transpose, dtype=float)

    def commit(self, displacement, exterior):
        """Compute all new states before writing; never reintegrate the lining."""
        fe, block = self.model.fe, self.block
        u = tensor(displacement).to(self.u_committed.device)
        strain = torch.einsum("egij,ej->egi", block["B"], (u - self.u_committed)[block["dofs"]])
        stress, plastic, _ = axifem.integrate_axisym(
            strain.reshape(-1, 4), block["stress"].reshape(-1, 4),
            block["E"], block["nu"], block["mat"], block["kind"])
        rock_stress = stress.detach().cpu().numpy().reshape(-1, 4, 4)
        liner_stress = self.support.recover_stress(displacement)
        if not all(np.isfinite(value).all() for value in (displacement, exterior, rock_stress, liner_stress)):
            raise FloatingPointError("Nonfinite accepted physical state")
        rock, liner = fe.blocks["rock"], fe.blocks["liner"]
        rock["sig"][rock["active"]] = rock_stress
        rock["plast"][rock["active"]] |= plastic.detach().cpu().numpy().reshape(-1, 4) > 1e-14
        liner["sig"][liner["active"]] = liner_stress
        fe.u_n = np.array(displacement, copy=True)
        self.model.solver.c = np.array(exterior, copy=True)
