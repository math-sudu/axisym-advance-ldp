# -*- coding: utf-8 -*-
"""Semi-analytic interface-fitting condensation (EXEC-18 E-C freeze S2-4).

Exterior of the near-field FE box is carried by the EXEC-17 production
dictionary (line source + on-axis multi-center two-potential families,
face-covariant).  On the interface Gamma = lateral cylinder rho = R_i
U front disk z = z_front:

  * interface nodal displacements are CONSTRAINED into the analytic
    trace space  u_Gamma = B_c c   (basis expansion + fitting),
  * the exterior traction (sigma_0 + sigma_A(c)) . n enters the FE
    weak form as a consistent load  F0 + T_A c,
  * interface equilibrium rows are Galerkin-reduced by B_c^T
    (virtual interface motions delta u = B_c delta c),

giving a square nonsymmetric (u_I, c) Newton system; the c block is
linear (exterior elastic).  Conditioning: dictionary columns are
equilibrated by their interface trace norm and a small ridge
eps_ridge is added to the scaled c block (freeze: conditioning
treatment inside the frozen candidate; eps sensitivity is a reading).

Non-closed-form by construction -- this is the R3 (Godoy) corrected
form of candidate three: basis expansion + interface fitting, no
closed-form DtN claimed.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


from axisym_src.potentials import TwoPotentialBasis  # exec17 read-only

__all__ = ["make_face_dictionary", "InterfaceExterior", "CoupledSolver"]

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def make_face_dictionary(x_f, a, G, nu, n_ord=8,
                         pool=(1.0, 3.0, 6.0, 12.0), z0_over_a=0.5):
    """Face-covariant EXEC-17 production dictionary at face x_f.

    Line source endpoint x_f - z0_over_a*a, centers x_f - pool*a
    (all inside the cavity / non-material region).
    """
    centers = [x_f - p * a for p in pool]
    return TwoPotentialBasis(centers, n_ord, n_ord, a, G, nu,
                             line_sources=[x_f - z0_over_a * a])


class InterfaceExterior:
    """Interface geometry + exterior maps for one mesh (static edges).

    Parameters
    ----------
    mesh : AxiMesh (built)
    p0 : float
    lateral : bool -- include rho = R_i cylinder
    front : bool -- include z = z_front disk
    """

    def __init__(self, mesh, p0):
        self.m = mesh
        self.p0 = float(p0)
        nr1, nz1 = mesh.nr1, mesh.nz1
        # interface nodes: rho = R_i (all z) + z = z_front (all rho)
        ids = set()
        for iz in range(nz1):
            ids.add(iz * nr1 + (nr1 - 1))
        for ir in range(nr1):
            ids.add((nz1 - 1) * nr1 + ir)
        self.if_nodes = np.array(sorted(ids), dtype=np.int64)
        self.if_dofs = np.stack([2 * self.if_nodes,
                                 2 * self.if_nodes + 1], -1).ravel()
        # edge lists (node pairs) on Gamma
        edges = []
        for e in range(mesh.n_elems):
            if mesh.elem_ir[e] == mesh.ne_r - 1:      # right edge
                edges.append((mesh.conn[e][1], mesh.conn[e][2],
                              "lat"))
            if mesh.elem_iz[e] == mesh.ne_z - 1:      # top edge
                edges.append((mesh.conn[e][3], mesh.conn[e][2],
                              "top"))
        self.edges = edges

    # ------------------------------------------------- exterior loads
    def sigma0_load(self, ndof):
        """F0: consistent load of sigma_0 . n on Gamma (per-radian)."""
        F = np.zeros(ndof)
        m = self.m
        p0 = self.p0
        for n1, n2, kind in self.edges:
            x1 = np.array([m.node_rho[n1], m.node_z[n1]])
            x2 = np.array([m.node_rho[n2], m.node_z[n2]])
            L = np.linalg.norm(x2 - x1)
            for xi in _GP:
                N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi)
                xg = N1 * x1 + N2 * x2
                w = xg[0] * (L / 2.0)
                if kind == "lat":       # n = (+1, 0): t = (s_rr, s_rz)
                    tr, tz = -p0, 0.0
                else:                    # n = (0, +1): t = (s_rz, s_zz)
                    tr, tz = 0.0, -p0
                F[2 * n1] += w * N1 * tr
                F[2 * n1 + 1] += w * N1 * tz
                F[2 * n2] += w * N2 * tr
                F[2 * n2 + 1] += w * N2 * tz
        return F

    def basis_maps(self, basis, ndof):
        """(B_c, T_A): trace matrix at interface nodes (2n_if x n_c)
        and global traction-load matrix (ndof x n_c) of unit-coefficient
        exterior corrections on Gamma."""
        m = self.m
        rho_n = m.node_rho[self.if_nodes]
        z_n = m.node_z[self.if_nodes]
        cols = basis.columns(rho_n, z_n)
        n_c = basis.n_columns
        n_if = self.if_nodes.size
        B_c = np.zeros((2 * n_if, n_c))
        B_c[0::2, :] = cols["u_rho"].T
        B_c[1::2, :] = cols["u_z"].T
        T_A = np.zeros((ndof, n_c))
        for n1, n2, kind in self.edges:
            x1 = np.array([m.node_rho[n1], m.node_z[n1]])
            x2 = np.array([m.node_rho[n2], m.node_z[n2]])
            L = np.linalg.norm(x2 - x1)
            for xi in _GP:
                N1, N2 = 0.5 * (1 - xi), 0.5 * (1 + xi)
                xg = N1 * x1 + N2 * x2
                w = xg[0] * (L / 2.0)
                cg = basis.columns(np.array([xg[0]]),
                                   np.array([xg[1]]))
                if kind == "lat":
                    tr = cg["s_rr"][:, 0]
                    tz = cg["s_rz"][:, 0]
                else:
                    tr = cg["s_rz"][:, 0]
                    tz = cg["s_zz"][:, 0]
                T_A[2 * n1, :] += w * N1 * tr
                T_A[2 * n1 + 1, :] += w * N1 * tz
                T_A[2 * n2, :] += w * N2 * tr
                T_A[2 * n2 + 1, :] += w * N2 * tz
        return B_c, T_A


class CoupledSolver:
    """Monolithic Newton on (u_I, c) with interface condensation.

    fe : AxiFE (owns state, blocks, f_release / f_ext_fix WITHOUT the
         exterior Gamma parts -- those live here)
    ext : InterfaceExterior
    basis : current-face dictionary (rebuild per advancement step via
            set_basis)
    eps_ridge : ridge on the equilibrated c block (freeze S2-4).
    """

    def __init__(self, fe, ext, eps_ridge=1e-10):
        self.fe = fe
        self.ext = ext
        self.eps_ridge = float(eps_ridge)
        self.F0 = ext.sigma0_load(fe.ndof)
        self.basis = None
        self.c = None
        self._maps = None

    def set_basis(self, basis, keep_c=True, svd_tol=1e-6,
                  t_norm_guard=100.0):
        """Build interface maps with SVD-reduced trace coordinates.

        The far-surface trace of the multi-center dictionary is
        massively redundant (measured cond ~ 9e16, effective rank
        ~ half the columns) -- the trace matrix is SVD-truncated at
        svd_tol * sigma_max and the Newton unknowns are the orthonormal
        trace coordinates y (u_Gamma = U_k y).  Physical coefficients
        are recovered as the min-norm representative
        c = V_k diag(1/s_k) y.  Modes whose reduced traction-column
        norm exceeds t_norm_guard * median are numerically-null junk
        (near-zero trace but noisy traction) and are dropped; counts
        are recorded in self.svd_info (conditioning treatment inside
        the frozen candidate, freeze S2-4)."""
        u_if_old = None
        if keep_c and self.c is not None and self._maps is not None:
            u_if_old = self._maps["B"] @ self.c   # current trace field
        B_c, T_A = self.ext.basis_maps(basis, self.fe.ndof)
        scale = np.linalg.norm(B_c, axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        Bs = B_c / scale[None, :]
        Ts = T_A / scale[None, :]
        U, s, Vt = np.linalg.svd(Bs, full_matrices=False)
        keep = s > svd_tol * s[0]
        U_k, s_k, V_k = U[:, keep], s[keep], Vt[keep].T
        T_y = Ts @ (V_k / s_k[None, :])
        tn = np.linalg.norm(T_y, axis=0)
        med = np.median(tn)
        ok = tn <= t_norm_guard * max(med, 1e-30)
        self.svd_info = {"n_cols": Bs.shape[1],
                         "k_svd": int(keep.sum()),
                         "k_final": int(ok.sum()),
                         "cond_B": float(s[0] / s[-1]) if s[-1] > 0
                         else np.inf}
        self._maps = {
            "B": U_k[:, ok], "T": T_y[:, ok],
            "V": V_k[:, ok], "s": s_k[ok], "scale": scale,
        }
        if u_if_old is not None:
            # continue from the same physical trace field
            self.c = self._maps["B"].T @ u_if_old
        else:
            self.c = np.zeros(self._maps["B"].shape[1])
        self.basis = basis

    # physical dictionary coefficients (min-norm representative)
    @property
    def c_phys(self):
        mp = self._maps
        return (mp["V"] @ (self.c / mp["s"])) / mp["scale"]

    def newton(self, u0=None, lam_release=1.0, tol_rel=1e-10,
               max_iter=60):
        """Damped Newton with backtracking line search on the reduced
        residual norm (perfect-plasticity return-region cycling and
        the per-step dictionary re-anchoring kick both need it --
        step-2 probe reading: plain Newton diverges after a covariant
        basis switch, line search converges)."""
        fe = self.fe
        mp = self._maps
        B, T = mp["B"], mp["T"]
        if_dofs = self.ext.if_dofs
        act = fe.active_dof_mask()
        act[if_dofs] = True          # interface dofs always carried
        free = act & ~fe.dirichlet
        free_o = free.copy()
        free_o[if_dofs] = False       # non-interface unknowns
        idx_o = np.where(free_o)[0]
        n_o = idx_o.size
        n_c = B.shape[1]
        u = fe.u_n.copy() if u0 is None else u0.copy()
        c = self.c.copy()
        u[if_dofs] = B @ c
        fscale = max(np.linalg.norm(fe.f_release),
                     np.linalg.norm(self.F0), fe.p0)
        info = {"iters": 0, "res_rel": np.inf, "converged": False}
        T_if = T[if_dofs, :]

        def resid(u_, c_):
            F, _ = fe.internal_force(u_, want_tangent=False)
            R = (F - fe.f_ext_fix
                 + (1.0 - lam_release) * fe.f_release
                 - self.F0 - T @ c_)
            r = np.sqrt(np.linalg.norm(R[idx_o]) ** 2
                        + np.linalg.norm(B.T @ R[if_dofs]) ** 2)
            return R, r

        R, rnorm = resid(u, c)
        for it in range(max_iter):
            res = rnorm / fscale
            info["iters"] = it
            info["res_rel"] = float(res)
            if not np.isfinite(res):
                return u, c, info
            if res < tol_rel:
                info["converged"] = True
                self.c = c
                return u, c, info
            _, K = fe.internal_force(u, want_tangent=True)
            K_oo = K[idx_o][:, idx_o]
            K_oi = K[idx_o][:, if_dofs] @ sp.csr_matrix(B)
            K_io = sp.csr_matrix(B.T) @ K[if_dofs][:, idx_o]
            K_cc = (B.T @ (K[if_dofs][:, if_dofs] @ B)
                    - B.T @ T_if)
            K_cc[np.diag_indices(n_c)] += (
                self.eps_ridge * np.abs(K_cc).max())
            Kfull = sp.bmat(
                [[K_oo, K_oi], [K_io, sp.csr_matrix(K_cc)]],
                format="csc")
            rhs = -np.concatenate([R[idx_o], B.T @ R[if_dofs]])
            dw = spla.spsolve(Kfull, rhs)
            if not np.all(np.isfinite(dw)):
                return u, c, info
            alpha = 1.0
            accepted = False
            for _ in range(14):
                u_t = u.copy()
                u_t[idx_o] += alpha * dw[:n_o]
                c_t = c + alpha * dw[n_o:]
                u_t[if_dofs] = B @ c_t
                R_t, r_t = resid(u_t, c_t)
                if (r_t < rnorm * (1.0 - 1e-4 * alpha)
                        or r_t / fscale < tol_rel):
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return u, c, info      # stalled -> caller ramps
            u, c, R, rnorm = u_t, c_t, R_t, r_t
        return u, c, info

    def solve_step(self, tol_rel=1e-10):
        """Adaptive release-ramp Newton to lam = 1, then commit."""
        fe = self.fe
        lam_done = 0.0
        u = fe.u_n.copy()
        d_lam = 1.0
        attempts = 0
        substeps = 0
        while lam_done < 1.0 - 1e-12:
            lam_try = min(1.0, lam_done + d_lam)
            u_try, c_try, info = self.newton(
                u0=u, lam_release=lam_try, tol_rel=tol_rel)
            attempts += 1
            if info["converged"]:
                u = u_try
                fe.commit(u)
                lam_done = lam_try
                substeps += 1
                d_lam = min(1.0, d_lam * 2.0)
            else:
                d_lam *= 0.5
                if d_lam < 1e-4 or attempts > 60:
                    raise RuntimeError(
                        f"coupled release ramp failed at lam={lam_done}"
                        f" (d_lam={d_lam}, res={info['res_rel']:.3e})")
        return u, {"substeps": substeps, "attempts": attempts,
                   "c": self.c_phys.copy()}
