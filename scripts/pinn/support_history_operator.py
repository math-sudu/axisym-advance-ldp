"""Linear lining recovery with full installation references.

All active lining elements are assembled together, including shared batch
nodes. C denotes DOFs shared with active ground; I denotes lining-only DOFs.
The returned interface force is the lining's INTERNAL force contribution.
Known external nodal forces p_I enter K_II u_I = h_I + p_I - K_IC u_C.
Geometry, material and references stay fixed for the lifetime of an operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import torch

from wave2_src.axifem import elastic_trial_axisym


def elastic_matrix(block):
    """Use the original rr/zz/tt/engineering-rz constitutive map."""
    if block["kind"] != "elastic":
        raise ValueError("Support recovery requires a linear elastic lining")
    basis = torch.eye(4, dtype=torch.float64)
    return elastic_trial_axisym(
        basis, torch.zeros_like(basis), block["E"], block["nu"]).numpy().T


@dataclass(frozen=True)
class InstallationReference:
    elements: np.ndarray
    stress_bias: np.ndarray
    event_id: int
    kind: str

    def __post_init__(self):
        elements = np.array(self.elements, dtype=np.int64, copy=True)
        bias = np.array(self.stress_bias, dtype=float, copy=True)
        if elements.ndim != 1 or len(np.unique(elements)) != len(elements):
            raise ValueError("Installation elements must be unique")
        if bias.shape != (len(elements), 4, 4) or not np.isfinite(bias).all():
            raise ValueError("Each element needs four finite four-component references")
        if self.kind not in ("equivalent_initial", "actual_event"):
            raise ValueError("Unknown installation-reference source")
        elements.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "stress_bias", bias)


@dataclass(frozen=True)
class SupportHistory:
    records: tuple[InstallationReference, ...] = ()

    @classmethod
    def from_initial(cls, fe):
        """Reproduce saved u0/sigma0; do not infer an unavailable birth field."""
        block = fe.blocks["liner"]
        elements = block["elems"][block["active"]]
        strain = np.einsum("egij,ej->egi", fe.m.B_all[elements],
                           fe.u_n[fe.m.dof_all[elements]])
        bias = strain @ elastic_matrix(block).T - block["sig"][block["active"]]
        return cls((InstallationReference(elements, bias, 0, "equivalent_initial"),))

    def install(self, fe, elements, displacement, event_id):
        """Return new history at an explicitly supplied physical birth field.

        This changes neither FE active sets nor accepted displacements. Call
        extend_installation_field BEFORE activation to supply dormant values.
        Multiple records may share an event ID, but never an element.
        """
        elements = np.asarray(elements, dtype=np.int64)
        previous = np.concatenate([r.elements for r in self.records]) if self.records else []
        if np.isin(elements, previous).any():
            raise ValueError("An installed element cannot acquire a second reference")
        if not np.isin(elements, fe.blocks["liner"]["elems"]).all():
            raise ValueError("Installation contains an element outside the lining block")
        rock = fe.blocks["rock"]
        if np.isin(elements, rock["elems"][rock["active"]]).any():
            raise ValueError("New lining overlaps active ground")
        strain = np.einsum("egij,ej->egi", fe.m.B_all[elements],
                           np.asarray(displacement)[fe.m.dof_all[elements]])
        reference = InstallationReference(
            elements, strain @ elastic_matrix(fe.blocks["liner"]).T,
            event_id, "actual_event")
        return SupportHistory(self.records + (reference,))


def extend_installation_field(fe, elements):
    """Extend the accepted field radially into newly live lining nodes.

    For this structured bonded tunnel mesh, a dormant node takes both
    displacement components from the active ground contact at the same z.
    Already active nodes, including shared older lining nodes, are unchanged.
    Run before toggling the new lining active: no physical state is mutated.
    """
    mesh = fe.m
    nodes = np.unique(mesh.conn[np.asarray(elements, dtype=np.int64)])
    active = fe.active_dof_mask().reshape(-1, 2).any(axis=1)
    rock = fe.blocks["rock"]
    rock_nodes = np.unique(mesh.conn[rock["elems"][rock["active"]]])
    contacts = np.intersect1d(nodes, rock_nodes)
    field = fe.u_n.copy().reshape(-1, 2)
    for node in nodes[~active[nodes]]:
        candidates = contacts[mesh.node_z[contacts] == mesh.node_z[node]]
        if not len(candidates):
            raise ValueError("New lining node has no same-z ground contact for extension")
        source = candidates[np.argmin(mesh.node_rho[candidates])]
        field[node] = field[source]
    return field.ravel()


class SupportHistoryOperator:
    """Sparse whole-support Schur action, recovery and their adjoints.

    No dense inverse or Schur matrix is stored. The full Gauss-point bias is
    retained because assembled h alone cannot recover self-equilibrated stress.
    An operator must be rebuilt after an active-set or reference change.
    """

    def __init__(self, fe, history):
        block, mesh = fe.blocks["liner"], fe.m
        self.C = elastic_matrix(block)
        self.elements = block["elems"][block["active"]].copy()
        records = history.records
        recorded = np.concatenate([r.elements for r in records]) if records else np.empty(0, int)
        if not np.array_equal(np.sort(recorded), np.sort(self.elements)):
            raise ValueError("Active lining must have exactly one reference per element")
        biases = (np.concatenate([r.stress_bias for r in records]) if records
                  else np.empty((0, 4, 4)))
        order = np.argsort(recorded)
        self.stress_bias = biases[order[np.searchsorted(recorded[order], self.elements)]].copy()
        self.B = mesh.B_all[self.elements].copy()
        self.W = mesh.W_all[self.elements].copy()
        self.element_dofs = mesh.dof_all[self.elements].copy()
        self.dofs = np.unique(self.element_dofs)
        local = np.searchsorted(self.dofs, self.element_dofs)
        rock = fe.blocks["rock"]
        rock_dofs = np.unique(mesh.dof_all[rock["elems"][rock["active"]]])
        contact = np.isin(self.dofs, rock_dofs)
        self.c = np.flatnonzero(contact)
        self.i = np.flatnonzero(~contact)
        self.contact_dofs, self.internal_dofs = self.dofs[self.c], self.dofs[self.i]
        if fe.dirichlet[self.internal_dofs].any():
            raise ValueError("Lining-only Dirichlet DOFs require an explicit constraint lift")
        # An unattached component has an unconstrained axial rigid translation.
        # Diagnose it rather than accepting a near-singular LU or adding springs.
        if len(self.dofs):
            nodes = local[:, ::2] // 2
            adjacency = sp.coo_matrix((np.ones(nodes.size * 4),
                                       (np.repeat(nodes, 4, axis=1).ravel(),
                                        np.tile(nodes, (1, 4)).ravel())),
                                      shape=(len(self.dofs) // 2,) * 2).tocsr()
            count, labels = connected_components(adjacency)
            if len(np.unique(labels[self.c // 2])) != count:
                raise ValueError("Every lining component must share nodes with active ground")

        element_k = np.einsum("egki,kl,eglj,eg->eij", self.B, self.C, self.B, self.W)
        self.K = sp.coo_matrix((element_k.ravel(),
                               (np.repeat(local, 8, axis=1).ravel(),
                                np.tile(local, (1, 8)).ravel())),
                              shape=(len(self.dofs),) * 2).tocsr()
        self.h = np.zeros(len(self.dofs))
        element_h = np.einsum("egij,egi,eg->ej", self.B, self.stress_bias, self.W)
        np.add.at(self.h, local, element_h)
        self.Kcc = self.K[self.c][:, self.c]
        self.Kci = self.K[self.c][:, self.i]
        self.Kic = self.K[self.i][:, self.c]
        self.Kii = self.K[self.i][:, self.i].tocsc()
        self.factor = spla.splu(self.Kii) if len(self.i) else None

    def _solve(self, rhs, transpose=False):
        if self.factor is None:
            return np.zeros_like(rhs)
        return self.factor.solve(np.asarray(rhs), trans="T" if transpose else "N")

    def recover_internal(self, contact_displacement, internal_load=None):
        """Absolute lining-only displacement, including history and p_I."""
        rhs = self.h[self.i] - self.Kic @ contact_displacement
        if internal_load is not None:
            rhs = rhs + internal_load
        return self._solve(rhs)

    def recover_field(self, displacement, internal_load=None):
        field = np.array(displacement, dtype=float, copy=True)
        field[self.internal_dofs] = self.recover_internal(
            field[self.contact_dofs], internal_load)
        return field

    def interface_force(self, contact_displacement, internal_load=None):
        return (self.Kcc @ contact_displacement - self.h[self.c]
                + self.Kci @ self.recover_internal(contact_displacement, internal_load))

    def recover_direction(self, direction):
        return -self._solve(self.Kic @ direction)

    def recover_transpose(self, cotangent):
        return -self.Kic.T @ self._solve(cotangent, transpose=True)

    def apply_tangent(self, direction):
        return self.Kcc @ direction + self.Kci @ self.recover_direction(direction)

    def apply_tangent_transpose(self, cotangent):
        return (self.Kcc.T @ cotangent
                - self.Kic.T @ self._solve(self.Kci.T @ cotangent, transpose=True))

    def recover_stress(self, displacement):
        strain = np.einsum("egij,ej->egi", self.B,
                           np.asarray(displacement)[self.element_dofs])
        return strain @ self.C.T - self.stress_bias
