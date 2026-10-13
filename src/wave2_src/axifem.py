# -*- coding: utf-8 -*-
"""Axisymmetric small-strain FE core for EXEC-18 (E-C / E-D near field).

Frame: TENSION-POSITIVE, cylindrical (rho, z), axisymmetric (no twist).
Stress/strain component layout matches fe_dev's plane-strain convention
with the HOOP direction in the out-of-plane slot so that fe_dev's
principal-space Mohr-Coulomb return map (_principal4 -> _sort3 ->
_mc_sorted -> _unsort3 -> _recompose_stress) is reused READ-ONLY:

    sig4  = (s_rr, s_zz, s_tt, s_rz)      # in-plane pair (rr,zz),
    deps4 = (e_rr, e_zz, e_tt, g_rz)      # hoop in slot 2, shear slot 3

Unlike plane strain the hoop strain is ACTIVE (e_tt = u_rho/rho enters
the elastic trial).  Volume weights are PER RADIAN (the 2*pi factor is
dropped uniformly from element volumes, edge tractions and the exterior
coupling loads -- freeze S2 consistency convention).

Element: 4-node bilinear quad (CAX4-type) on a structured axis-aligned
grid, 2x2 Gauss.  Excavation: element removal with the released
internal-force vector ramped down adaptively (freeze S2-3: in-step ramp
is a solver aid, same class as ABAQUS *MODEL CHANGE ramp).  Liner
activation: elements enter the active set with sig = 0 and strains
measured incrementally from the activation-time displacement (the
incremental formulation gives strain-free activation for free).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


from fedev_src.material import (  # read-only import (freeze S2-2)
    _principal4, _sort3, _unsort3, _recompose_stress, _recompose_strain,
    _mc_sorted, lame,
)

DTYPE = torch.float64

__all__ = ["elastic_trial_axisym", "integrate_axisym", "tangent_axisym",
           "AxiMesh", "AxiFE"]


# ------------------------------------------------------------ material

def elastic_trial_axisym(deps4, sig4, E, nu):
    """sig_trial = sig + C : deps with ACTIVE hoop strain."""
    lam, G = lame(E, nu)
    err, ezz, ett, grz = (deps4[..., 0], deps4[..., 1],
                          deps4[..., 2], deps4[..., 3])
    tr = err + ezz + ett
    return torch.stack([
        sig4[..., 0] + lam * tr + 2.0 * G * err,
        sig4[..., 1] + lam * tr + 2.0 * G * ezz,
        sig4[..., 2] + lam * tr + 2.0 * G * ett,
        sig4[..., 3] + G * grz,
    ], dim=-1)


def integrate_axisym(deps4, sig4, E, nu, mat, kind):
    """Backward-Euler step; returns (sig_new4, dlam, region).

    kind in {"elastic", "mc"}.  Mirrors fe_dev _integrate_perfect with
    the axisymmetric elastic trial; the principal-space return map is
    fe_dev's bit-for-bit.
    """
    E = torch.as_tensor(E, dtype=DTYPE)
    nu = torch.as_tensor(nu, dtype=DTYPE)
    sig_tr = elastic_trial_axisym(deps4, sig4, E, nu)
    if kind == "elastic":
        z = torch.zeros_like(sig_tr[..., 0])
        return sig_tr, z, torch.zeros_like(z, dtype=torch.int64)
    if kind != "mc":
        raise ValueError(f"unknown kind {kind!r}")
    matt = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in mat.items()}
    sA, sB, sZ, c2, s2 = _principal4(sig_tr)
    s1, s2s, s3, m0, m2 = _sort3(sA, sB, sZ)
    t1, t2, t3, e1, e2, e3, dlam, region = _mc_sorted(
        s1, s2s, s3, E, nu, matt["c"], matt["phi"], matt["psi"])
    vA, vB, vZ = _unsort3(t1, t2, t3, m0, m2)
    sig_pl = _recompose_stress(vA, vB, vZ, c2, s2)
    plast4 = (region > 0).unsqueeze(-1)
    sig_new = torch.where(plast4, sig_pl, sig_tr)
    return sig_new, dlam, region


def tangent_axisym(deps4_qp, sig4_qp, E, nu, mat, kind):
    """Exact algorithmic tangent d(sig4)/d(deps4), shape (nq, 4, 4)."""
    def f(d, s):
        return integrate_axisym(d, s, E, nu, mat, kind)[0]
    return torch.func.vmap(torch.func.jacrev(f, argnums=0))(
        deps4_qp, sig4_qp)


# ---------------------------------------------------------------- mesh

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


class AxiMesh:
    """Structured axis-aligned quad grid from rho_lines x z_lines.

    Node id = i_z * n_rho1 + i_rho.  Element (i_rho, i_z) has ccw nodes
    [(i,j), (i+1,j), (i+1,j+1), (i,j+1)] in (rho, z) index space.
    """

    def __init__(self, rho_lines, z_lines):
        self.rho = np.asarray(rho_lines, dtype=float)
        self.z = np.asarray(z_lines, dtype=float)
        self.nr1 = self.rho.size
        self.nz1 = self.z.size
        self.n_nodes = self.nr1 * self.nz1
        self.ne_r = self.nr1 - 1
        self.ne_z = self.nz1 - 1
        self.n_elems = self.ne_r * self.ne_z
        ir = np.arange(self.ne_r)
        iz = np.arange(self.ne_z)
        IR, IZ = np.meshgrid(ir, iz, indexing="ij")
        self.elem_ir = IR.ravel()
        self.elem_iz = IZ.ravel()
        n00 = IZ * self.nr1 + IR
        self.conn = np.stack(
            [n00, n00 + 1, n00 + 1 + self.nr1, n00 + self.nr1],
            axis=-1).reshape(-1, 4)
        # node coordinates
        RR, ZZ = np.meshgrid(self.rho, self.z, indexing="xy")
        self.node_rho = RR.ravel()      # id = iz*nr1 + ir -> rho[ir]
        self.node_z = ZZ.ravel()
        # element centers / sizes
        self.e_rho0 = self.rho[self.elem_ir]
        self.e_rho1 = self.rho[self.elem_ir + 1]
        self.e_z0 = self.z[self.elem_iz]
        self.e_z1 = self.z[self.elem_iz + 1]

    def elems_in_box(self, r0, r1, z0, z1, tol=1e-9):
        """Element ids fully inside [r0,r1]x[z0,z1]."""
        return np.where(
            (self.e_rho0 >= r0 - tol) & (self.e_rho1 <= r1 + tol)
            & (self.e_z0 >= z0 - tol) & (self.e_z1 <= z1 + tol))[0]

    def build_B(self, bbar=True):
        """Vectorized B matrices: B_all (ne, 4gp, 4, 8), w_all (ne, 4gp)
        with per-radian weights rho_gp * detJ * w_gp.

        bbar=True applies the mean-dilatation (B-bar) treatment to the
        three normal rows -- the standard cure for volumetric locking
        of bilinear quads under (near-)isochoric plastic flow (psi = 0
        MC; EXEC-16 fe_dev used the F-bar analogue).  Constant-
        volumetric fields are reproduced exactly (patch preserved)."""
        ne = self.n_elems
        dr = (self.e_rho1 - self.e_rho0)
        dz = (self.e_z1 - self.e_z0)
        detJ = dr * dz / 4.0
        B = np.zeros((ne, 4, 4, 8))
        W = np.zeros((ne, 4))
        gp_list = [(xi, eta) for eta in _GP for xi in _GP]
        for g, (xi, eta) in enumerate(gp_list):
            N = 0.25 * np.array([(1 - xi) * (1 - eta),
                                 (1 + xi) * (1 - eta),
                                 (1 + xi) * (1 + eta),
                                 (1 - xi) * (1 + eta)])
            dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta),
                                      (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                       (1 + xi), (1 - xi)])
            rho_g = 0.5 * (self.e_rho0 + self.e_rho1) + 0.5 * xi * dr
            dNdr = dN_dxi[None, :] * (2.0 / dr)[:, None]     # (ne,4)
            dNdz = dN_deta[None, :] * (2.0 / dz)[:, None]
            Nr = N[None, :] / rho_g[:, None]
            for a4 in range(4):
                B[:, g, 0, 2 * a4] = dNdr[:, a4]          # e_rr <- u_rho
                B[:, g, 1, 2 * a4 + 1] = dNdz[:, a4]      # e_zz <- u_z
                B[:, g, 2, 2 * a4] = Nr[:, a4]            # e_tt <- u_rho
                B[:, g, 3, 2 * a4] = dNdz[:, a4]          # g_rz <- u_rho
                B[:, g, 3, 2 * a4 + 1] = dNdr[:, a4]      # g_rz <- u_z
            W[:, g] = rho_g * detJ
        if bbar:
            vol = B[:, :, 0, :] + B[:, :, 1, :] + B[:, :, 2, :]
            wsum = W.sum(axis=1)
            vbar = (np.einsum("eg,ega->ea", W, vol)
                    / wsum[:, None])                       # (ne, 8)
            corr = (vbar[:, None, :] - vol) / 3.0          # (ne,4gp,8)
            for i in range(3):
                B[:, :, i, :] += corr
        self.B_all = B
        self.W_all = W
        # gauss point coordinates (ne, 4, 2)
        gp_rho = np.zeros((ne, 4))
        gp_z = np.zeros((ne, 4))
        for g, (xi, eta) in enumerate(gp_list):
            gp_rho[:, g] = 0.5 * (self.e_rho0 + self.e_rho1) \
                + 0.5 * xi * dr
            gp_z[:, g] = 0.5 * (self.e_z0 + self.e_z1) + 0.5 * eta * dz
        self.gp_rho = gp_rho
        self.gp_z = gp_z
        # dof map (ne, 8)
        dof = np.empty((ne, 8), dtype=np.int64)
        dof[:, 0::2] = 2 * self.conn
        dof[:, 1::2] = 2 * self.conn + 1
        self.dof_all = dof
        return self


# ------------------------------------------------------------ FE state

class AxiFE:
    """Incremental axisymmetric FE with element birth/death.

    Element blocks are (name -> dict) with keys: elems (ids), kind
    ("elastic"|"mc"), mat (dict), E, nu, active (bool array).  Stress
    state sig_n is stored per element-block gp array; u_n is the last
    converged displacement.
    """

    def __init__(self, mesh, p0, tang_floor=1e-8):
        self.m = mesh.build_B()
        self.p0 = float(p0)
        self.ndof = 2 * mesh.n_nodes
        self.u_n = np.zeros(self.ndof)
        self.blocks = {}
        self.dirichlet = np.zeros(self.ndof, dtype=bool)
        self.f_release = np.zeros(self.ndof)   # ramped removal force
        self.f_ext_fix = np.zeros(self.ndof)   # e.g. back-disk traction
        # small elastic floor on the algorithmic tangent: regularizes
        # the exactly-singular apex tangent of perfect MC during
        # intermediate Newton iterates (solver aid only -- the residual
        # and hence the converged solution are unaffected)
        self.tang_floor = float(tang_floor)

    # ---------------------------------------------------------- blocks
    def add_block(self, name, elems, kind, E, nu, mat=None,
                  active=True, sig0=None):
        elems = np.asarray(elems, dtype=np.int64)
        blk = {
            "elems": elems, "kind": kind, "E": float(E), "nu": float(nu),
            "mat": dict(mat or {}),
            "active": np.full(elems.size, bool(active)),
            "sig": np.zeros((elems.size, 4, 4)),
            "plast": np.zeros((elems.size, 4), dtype=bool),
        }
        if sig0 is not None:
            blk["sig"][:] = np.asarray(sig0, dtype=float)
        self.blocks[name] = blk
        return blk

    def set_active(self, name, elem_ids, active):
        blk = self.blocks[name]
        sel = np.isin(blk["elems"], np.asarray(elem_ids, dtype=np.int64))
        blk["active"][sel] = bool(active)
        return sel

    def active_dof_mask(self):
        touched = np.zeros(self.ndof, dtype=bool)
        m = self.m
        for blk in self.blocks.values():
            e = blk["elems"][blk["active"]]
            touched[m.dof_all[e].ravel()] = True
        return touched

    # ------------------------------------------------------- integrals
    def _block_state(self, blk, u):
        """Trial strains/stresses for current iterate u (active only)."""
        m = self.m
        e = blk["elems"][blk["active"]]
        if e.size == 0:
            return e, None, None
        du_e = (u - self.u_n)[m.dof_all[e]]                # (na, 8)
        deps = np.einsum("egij,ej->egi", m.B_all[e], du_e)  # (na,4gp,4)
        sig_n = blk["sig"][blk["active"]]                   # (na,4,4)
        return e, deps, sig_n

    def internal_force(self, u, want_tangent=False):
        """Assemble F_int(u) and (optionally) tangent K."""
        m = self.m
        F = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        for blk in self.blocks.values():
            e, deps, sig_n = self._block_state(blk, u)
            if e.size == 0:
                continue
            dt = torch.as_tensor(deps.reshape(-1, 4), dtype=DTYPE)
            st = torch.as_tensor(sig_n.reshape(-1, 4), dtype=DTYPE)
            sig_new, dlam, region = integrate_axisym(
                dt, st, blk["E"], blk["nu"], blk["mat"], blk["kind"])
            sig_np = sig_new.numpy().reshape(e.size, 4, 4)
            Fe = np.einsum("egij,egi,eg->ej", m.B_all[e], sig_np,
                           m.W_all[e])
            np.add.at(F, m.dof_all[e], Fe)
            if want_tangent:
                lam, G = lame(blk["E"], blk["nu"])
                D1 = np.array([
                    [lam + 2 * G, lam, lam, 0.0],
                    [lam, lam + 2 * G, lam, 0.0],
                    [lam, lam, lam + 2 * G, 0.0],
                    [0.0, 0.0, 0.0, G]])
                if blk["kind"] == "elastic":
                    D = np.broadcast_to(D1, (e.size, 4, 4, 4))
                else:
                    D = tangent_axisym(
                        dt, st, blk["E"], blk["nu"], blk["mat"],
                        blk["kind"]).numpy().reshape(e.size, 4, 4, 4)
                    if self.tang_floor > 0.0:
                        D = D + self.tang_floor * D1[None, None]
                Ke = np.einsum("egki,egkl,eglj,eg->eij",
                               m.B_all[e], D, m.B_all[e], m.W_all[e])
                dofs = m.dof_all[e]
                rows.append(np.repeat(dofs, 8, axis=1).ravel())
                cols.append(np.tile(dofs, (1, 8)).ravel())
                vals.append(Ke.ravel())
        K = None
        if want_tangent:
            K = sp.coo_matrix(
                (np.concatenate(vals) if vals else np.zeros(0),
                 (np.concatenate(rows) if rows else np.zeros(0, int),
                  np.concatenate(cols) if cols else np.zeros(0, int))),
                shape=(self.ndof, self.ndof)).tocsr()
        return F, K

    def commit(self, u):
        """Commit converged state: update sig_n from u, set u_n = u."""
        m = self.m
        for blk in self.blocks.values():
            e_loc = np.where(blk["active"])[0]
            e = blk["elems"][e_loc]
            if e.size == 0:
                continue
            du_e = (u - self.u_n)[m.dof_all[e]]
            deps = np.einsum("egij,ej->egi", m.B_all[e], du_e)
            dt = torch.as_tensor(deps.reshape(-1, 4), dtype=DTYPE)
            st = torch.as_tensor(
                blk["sig"][e_loc].reshape(-1, 4), dtype=DTYPE)
            sig_new, dlam, region = integrate_axisym(
                dt, st, blk["E"], blk["nu"], blk["mat"], blk["kind"])
            blk["sig"][e_loc] = sig_new.numpy().reshape(e.size, 4, 4)
            if blk["kind"] == "mc":
                blk["plast"][e_loc] |= (
                    dlam.numpy().reshape(e.size, 4) > 1e-14)
        self.u_n = u.copy()

    def removal_force(self, name, elem_ids):
        """Internal-force vector of the elements about to be removed at
        their committed stress (the release load to be ramped down)."""
        m = self.m
        blk = self.blocks[name]
        sel = np.isin(blk["elems"], np.asarray(elem_ids, dtype=np.int64))
        e = blk["elems"][sel]
        F = np.zeros(self.ndof)
        if e.size:
            Fe = np.einsum("egij,egi,eg->ej", m.B_all[e],
                           blk["sig"][sel], m.W_all[e])
            np.add.at(F, m.dof_all[e], Fe)
        return F

    # ------------------------------------------------------- edge loads
    def edge_load(self, edges, t_fun):
        """Consistent nodal load of traction t(rho, z) -> (t_rho, t_z)
        on straight axis-aligned edges.

        edges: list of (n1, n2) node-id pairs; per-radian weights
        rho_gp * (L/2) * w_gp.
        """
        F = np.zeros(self.ndof)
        m = self.m
        for n1, n2 in edges:
            x1 = np.array([m.node_rho[n1], m.node_z[n1]])
            x2 = np.array([m.node_rho[n2], m.node_z[n2]])
            L = np.linalg.norm(x2 - x1)
            for xi in _GP:
                N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi)
                xg = N1 * x1 + N2 * x2
                tr, tz = t_fun(xg[0], xg[1])
                w = xg[0] * (L / 2.0)
                F[2 * n1] += w * N1 * tr
                F[2 * n1 + 1] += w * N1 * tz
                F[2 * n2] += w * N2 * tr
                F[2 * n2 + 1] += w * N2 * tz
        return F

    # ---------------------------------------------------------- newton
    def newton(self, u0=None, lam_release=1.0, extra=None,
               tol_rel=1e-10, max_iter=30):
        """Solve equilibrium at release factor lam_release.

        Residual R(u) = F_int(u) - f_ext_fix + (1-lam)*f_release
        (f_release = internal-force vector of the removed elements at
        their committed stress; at lam = 0 this restores the intact
        equilibrium F_int_active = F_ext - F_removed exactly, at
        lam = 1 the support is fully released).
        extra: optional object with methods contributing to residual and
        tangent (used by coupling.CondensedSystem); None = plain solve
        with Dirichlet mask only.
        """
        u = self.u_n.copy() if u0 is None else u0.copy()
        act = self.active_dof_mask()
        free = act & ~self.dirichlet
        fscale = max(np.linalg.norm(self.f_release),
                     np.linalg.norm(self.f_ext_fix), self.p0)
        info = {"iters": 0, "res_rel": np.inf, "converged": False}
        for it in range(max_iter):
            F, K = self.internal_force(u, want_tangent=True)
            R = F - self.f_ext_fix + (1.0 - lam_release) * self.f_release
            res = np.linalg.norm(R[free]) / fscale
            info["iters"] = it
            info["res_rel"] = float(res)
            if not np.isfinite(res):
                return u, info
            if res < tol_rel:
                info["converged"] = True
                return u, info
            idx = np.where(free)[0]
            Kff = K[idx][:, idx]
            du = spla.spsolve(Kff.tocsc(), -R[idx])
            if not np.all(np.isfinite(du)):
                return u, info
            u[idx] += du
        return u, info

    def solve_step(self, tol_rel=1e-10):
        """Adaptive release-ramp Newton to lam = 1, then commit."""
        lam_done = 0.0
        u = self.u_n.copy()
        attempts = 0
        d_lam = 1.0
        substeps = 0
        while lam_done < 1.0 - 1e-12:
            lam_try = min(1.0, lam_done + d_lam)
            u_try, info = self.newton(u0=u, lam_release=lam_try,
                                      tol_rel=tol_rel)
            attempts += 1
            if info["converged"]:
                u = u_try
                # intermediate commit keeps the incremental strains
                # small along the ramp (backward-Euler subincrements)
                self.commit(u)
                lam_done = lam_try
                substeps += 1
                d_lam = min(1.0, d_lam * 2.0)
            else:
                d_lam *= 0.5
                if d_lam < 1e-4 or attempts > 60:
                    raise RuntimeError(
                        f"release ramp failed at lam={lam_done}"
                        f" (d_lam={d_lam}, res={info['res_rel']:.3e})")
        return u, {"substeps": substeps, "attempts": attempts}
