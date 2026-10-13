# -*- coding: utf-8 -*-
"""E-C / E-D advancement scenarios (EXEC-18 freeze S1/S2/S4).

Geometry (freeze S2-1): near box rho in [0, R_i], z in [z_back=-10a,
z_front=+26a]; spin-up removes the core {z in [z_back, 0)}; then
N_adv steps of Delta = 0.4a (=0.2D, V&D-comparable primary step) to
x_f = 12a.  Face frame zeta = z - x_f.

Per-radian force convention throughout (axifem).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .axifem import AxiMesh, AxiFE
from .coupling import (InterfaceExterior, CoupledSolver,
                       make_face_dictionary)
from .ccm import Deep2D, ring_stiffness_hat, ccm_equilibrium, \
    grc_function

A = 2.0
P0 = 5.0
E_ROCK = 1800.0
NU_ROCK = 0.30
G_ROCK = E_ROCK / (2.0 * (1.0 + NU_ROCK))
PHI = np.deg2rad(30.0)
PSI = 0.0
E_L, NU_L = 18000.0, 0.20          # ch02 case-2 liner
R_IL = 0.9 * A                      # liner inner radius 1.8
Z_BACK = -10.0 * A
L_ADV = 12.0 * A
Z_FRONT = 26.0 * A
DSTEP = 0.4 * A


@dataclass
class ArmSpec:
    name: str
    kind: str                    # "elastic" | "mc"
    c: float = 0.0
    phi: float = PHI
    psi: float = PSI
    R_i_over_a: float = 4.0

    @property
    def mat(self):
        if self.kind == "elastic":
            return {}
        return {"c": self.c, "phi": self.phi, "psi": self.psi}

    def deep2d(self, p_i=0.0):
        return Deep2D(self.kind, P0, A, E_ROCK, NU_ROCK,
                      mat=self.mat, p_i=p_i)


ARMS = {
    "EL": ArmSpec("EL", "elastic", R_i_over_a=4.0),
    "A": ArmSpec("A", "mc", c=1.00, R_i_over_a=4.0),
    "B": ArmSpec("B", "mc", c=0.50, R_i_over_a=4.0),
    "C": ArmSpec("C", "mc", c=0.25, R_i_over_a=5.0),
    "D": ArmSpec("D", "mc", c=0.12, R_i_over_a=6.5),
    "B10": ArmSpec("B10", "mc", c=0.50, psi=np.deg2rad(10.0),
                   R_i_over_a=4.0),
}


# ------------------------------------------------------------- meshing

def radial_lines(R_i, R_band, h_band=0.15, h_scale=1.0):
    """Core (with liner band 0.9/0.95/1.0) + fine annulus to R_band +
    geometric coarsening to R_i.  All in units of a, returned in m."""
    hb = h_band * h_scale
    core = [0.0, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0]
    if h_scale < 1.0:                      # fine arm: split liner band
        core = [0.0, 0.25, 0.5, 0.7, 0.8, 0.875, 0.9, 0.925,
                0.95, 0.975, 1.0]
    lines = list(core)
    x = 1.0
    while x < R_band - 1e-9:
        x = min(R_band, x + hb)
        lines.append(x)
    h = hb
    while lines[-1] < R_i - 1e-9:
        h = min(h * 1.45, 0.35 * (R_i - 1.0))
        x = lines[-1] + h
        if x > R_i - 0.25 * h:
            x = R_i
        lines.append(x)
    return A * np.asarray(lines)


def axial_lines(h_track=0.2, h_scale=1.0, z_back_over_a=-10.0):
    ht = h_track * h_scale
    if z_back_over_a <= -10.0 + 1e-9:
        back = [-10.0, -7.4, -5.4, -4.0, -3.0, -2.4]
    else:                          # L_b comparison tier (freeze S2-4)
        back = [z_back_over_a, -4.6, -3.4, -2.6]
    front = [15.0, 15.6, 16.6, 18.0, 19.6, 21.4, 23.6, 26.0]
    track0, track1 = -2.0, 14.6
    n_tr = int(round((track1 - track0) / ht))
    track = track0 + ht * np.arange(n_tr + 1)
    lines = np.concatenate([np.asarray(back), track,
                            np.asarray(front)])
    lines = np.unique(np.round(lines, 10))
    return A * lines


# ------------------------------------------------------- scenario core

class AdvanceScenario:
    """One arm's spin-up + advancement run (coupled by default)."""

    def __init__(self, arm, liner_d=None, dstep=DSTEP, L_adv=L_ADV,
                 n_ord=8, h_scale=1.0, eps_ridge=1e-10,
                 record_every=1, tol_rel=1e-10, anchor_x_f=None,
                 z_back_over_a=-10.0):
        self.arm = arm
        self.liner_d = liner_d
        self.dstep = float(dstep)
        self.L_adv = float(L_adv)
        self.n_steps = int(round(self.L_adv / self.dstep))
        self.n_ord = n_ord
        self.tol_rel = tol_rel
        self.z_back = z_back_over_a * A
        self.record_every = record_every

        d2 = arm.deep2d()
        self.deep2d = d2
        self.R_p2d = d2.R_p
        # deep-limit / normalization source = own-FE 2D full-return
        # profile (anchor amendment: build_reference u / sigma_z are
        # in-plane-flow constructions, invalid where the z-intermediacy
        # margin is negative; sigma_r / R_p remain exact there)
        self.prof2d = own_2d_profiles(arm)
        self.u_max = -self.prof2d["u_wall"]     # positive magnitude
        R_band = min(self.R_p2d / A + 0.5, arm.R_i_over_a - 0.5)
        self.rho_lines = radial_lines(arm.R_i_over_a,
                                      max(R_band, 1.5),
                                      h_scale=h_scale)
        self.z_lines = axial_lines(h_scale=h_scale,
                                   z_back_over_a=z_back_over_a)
        self.mesh = AxiMesh(self.rho_lines, self.z_lines)
        self.fe = AxiFE(self.mesh, P0)

        sig0 = np.array([-P0, -P0, -P0, 0.0])
        self.fe.add_block("rock", np.arange(self.mesh.n_elems),
                          arm.kind, E_ROCK, NU_ROCK, mat=arm.mat,
                          active=True, sig0=sig0)
        if liner_d is not None:
            lin = self.mesh.elems_in_box(R_IL, A, self.z_back,
                                         self.L_adv)
            self.fe.add_block("liner", lin, "elastic", E_L, NU_L,
                              active=False, sig0=0.0)
            self.K_hat = ring_stiffness_hat(E_L, NU_L, R_IL, A)
            # wished-in-place composite state for the back BC
            grc = grc_function(arm.kind, P0, A, E_ROCK, NU_ROCK,
                               mat=arm.mat)
            self.p_s_wished, _ = ccm_equilibrium(
                grc, self.K_hat, 0.0, P0)
        else:
            self.K_hat = None
            self.p_s_wished = 0.0

        # axis Dirichlet u_rho = 0 (except condensed top axis node)
        self.ext = InterfaceExterior(self.mesh, P0)
        nr1 = self.mesh.nr1
        axis_nodes = np.arange(self.mesh.nz1) * nr1
        cond = np.isin(axis_nodes, self.ext.if_nodes)
        for nid in axis_nodes[~cond]:
            self.fe.dirichlet[2 * nid] = True

        self.solver = CoupledSolver(self.fe, self.ext,
                                    eps_ridge=eps_ridge)
        # FIXED dictionary anchored at the run's final face (protocol
        # deviation note, EXEC-9b channel: the freeze's face-covariant
        # per-step re-anchoring injects a trace-projection kick that
        # stalls Newton from step 3 on (probe reading: no-switch
        # converges in 8 iters, switch stalls at any lam).  A fixed
        # dictionary is legal -- the singular supports lie inside the
        # FE box and the analytic field is only used on Gamma and the
        # exterior; the elastic control arm carries the translation
        # baseline of c(x_f), so the family reading separates history
        # as the difference from that control.
        self.anchor_x_f = (self.L_adv if anchor_x_f is None
                           else float(anchor_x_f))
        self.solver.set_basis(
            make_face_dictionary(self.anchor_x_f, A, G_ROCK, NU_ROCK,
                                 n_ord=self.n_ord))

        # wall node line (rho = a)
        self.ir_a = int(np.argmin(np.abs(self.rho_lines - A)))
        assert abs(self.rho_lines[self.ir_a] - A) < 1e-9
        self.wall_nodes = (np.arange(self.mesh.nz1) * nr1 + self.ir_a)
        self.wall_z = self.z_lines.copy()

        # back-disk edges (annulus rho >= a, active rock bottom)
        m = self.mesh
        self.back_edges = [
            (m.conn[e][0], m.conn[e][1])
            for e in range(m.n_elems)
            if m.elem_iz[e] == 0 and m.e_rho0[e] >= A - 1e-9]
        self._back_loads()
        self.history = []
        self.cost = {"steps_wall_s": [], "total_wall_s": 0.0}

    # -------------------------------------------------------- back BC
    def _back_loads(self):
        """f_ext_fix = post-state 2D traction on the back annulus;
        the sigma_0 -> 2D transition rides the spin-up release ramp."""
        if self.liner_d is not None:
            prof = own_2d_profiles(self.arm, p_i=self.p_s_wished)
            # liner sigma_z at back plane: plane-strain Lame ring under
            # external pressure p_s_wished, inner free
            Mm = np.array([[1.0, -1.0 / R_IL ** 2],
                           [1.0, -1.0 / A ** 2]])
            A_l, B_l = np.linalg.solve(
                Mm, np.array([0.0, -self.p_s_wished]))

            def t_back(rho, z):
                if rho >= A:
                    sz = float(prof["sigma_z"](rho))
                else:
                    sz = NU_L * 2.0 * A_l
                return 0.0, -sz
        else:
            prof = self.prof2d

            def t_back(rho, z):
                return 0.0, -float(prof["sigma_z"](rho))

        self.F_back_2d = self.fe.edge_load(self.back_edges, t_back)
        self.F_back_s0 = self.fe.edge_load(
            self.back_edges, lambda r, z: (0.0, P0))
        self.fe.f_ext_fix = self.F_back_2d.copy()

    # ------------------------------------------------------------ run
    def _rings(self, z0, z1):
        return self.mesh.elems_in_box(0.0, A, z0, z1)

    def _liner_activation(self, x_f):
        if self.liner_d is None:
            return
        blk = self.fe.blocks["liner"]
        zc1 = self.mesh.e_z1[blk["elems"]]
        newly = (~blk["active"]) & (zc1 <= x_f - self.liner_d + 1e-9)
        blk["active"][newly] = True

    def step(self, k):
        """k = 0 spin-up; k >= 1 advancement ring k."""
        t0 = time.perf_counter()
        fe = self.fe
        if k == 0:
            rings = self._rings(self.z_back, 0.0)
            x_f = 0.0
        else:
            x_f = k * self.dstep
            rings = self._rings(x_f - self.dstep, x_f)
        F_rem = fe.removal_force("rock", rings)
        fe.set_active("rock", rings, False)
        fe.f_release = F_rem
        if k == 0:
            # ride the sigma_0 -> 2D back-BC transition on the same
            # ramp: F_ext_eff(lam) = F_back_2d + (1-lam)(F_s0 - F_2d)
            # with R = F_int - f_ext_fix + (1-lam) f_release
            fe.f_release = fe.f_release + (self.F_back_2d
                                           - self.F_back_s0)
        self._liner_activation(x_f)
        u, info = self.solver.solve_step(tol_rel=self.tol_rel)
        fe.f_release = np.zeros(fe.ndof)
        wall = -fe.u_n[2 * self.wall_nodes] / self.u_max
        rec = {"k": k, "x_f": x_f, "c": self.solver.c_phys.copy(),
               "u0": float(np.interp(x_f, self.wall_z, wall)),
               "wall": wall, "info": info,
               "wall_s": time.perf_counter() - t0}
        self.history.append(rec)
        self.cost["steps_wall_s"].append(rec["wall_s"])
        return rec

    def run(self):
        t0 = time.perf_counter()
        for k in range(self.n_steps + 1):
            self.step(k)
        self.cost["total_wall_s"] = time.perf_counter() - t0
        return self.history

    # ------------------------------------------------------------ QoI
    def ldp(self, rec=None, smooth=False):
        """(zeta/a, u*) at wall nodes for a history record.

        smooth=True returns the ring-averaged profile away from the
        face (centered +-0.21a window for |zeta| > 1a): the B-bar
        quad under isochoric MC flow carries a two-node checkerboard
        sawtooth along the excavated wall (period = one ring; EXEC-16
        ran the same checkerboard falsification on fe_dev's F-bar);
        the ring average is the standard post-processing and matches
        the ABAQUS anchor.  The raw sawtooth amplitude is reported
        separately (sawtooth_amp)."""
        rec = rec or self.history[-1]
        zeta = (self.wall_z - rec["x_f"]) / A
        wall = rec["wall"]
        if not smooth:
            return zeta, wall
        out = wall.copy()
        for i, z in enumerate(zeta):
            if abs(z) <= 1.0:
                continue
            sel = np.abs(zeta - z) <= 0.21
            out[i] = wall[sel].mean()
        return zeta, out

    def sawtooth_amp(self, band=(-10.0, -6.0)):
        zeta, raw = self.ldp()
        _, sm = self.ldp(smooth=True)
        sel = (zeta >= band[0]) & (zeta <= band[1])
        if not sel.any():
            return float("nan")
        return float(np.max(np.abs(raw[sel] - sm[sel])))

    def plastic_map(self, bin_a=0.4):
        """R_p(zeta) from cumulative plastic gauss flags."""
        blk = self.fe.blocks["rock"]
        e = blk["elems"]
        gpr = self.mesh.gp_rho[e].ravel()
        gpz = self.mesh.gp_z[e].ravel()
        fl = blk["plast"].ravel()
        x_f = self.history[-1]["x_f"]
        zeta = (gpz - x_f) / A
        bins = np.arange(zeta.min(), zeta.max() + bin_a, bin_a)
        out = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = (zeta >= b0) & (zeta < b1) & fl
            out.append((0.5 * (b0 + b1),
                        gpr[sel].max() / A if sel.any() else np.nan))
        return np.array(out)

    def deep_profile(self, zeta_a=-8.0):
        """Radial node-u and gauss-sigma profiles near zeta for the
        deep-limit check V4b."""
        x_f = self.history[-1]["x_f"]
        z_t = x_f + zeta_a * A
        iz = int(np.argmin(np.abs(self.z_lines - z_t)))
        nodes = (iz * self.mesh.nr1
                 + np.arange(self.ir_a, self.mesh.nr1))
        rho = self.mesh.node_rho[nodes]
        u_rho = self.fe.u_n[2 * nodes]
        blk = self.fe.blocks["rock"]
        act = blk["active"]
        e = blk["elems"][act]
        gpr = self.mesh.gp_rho[e].ravel()
        gpz = self.mesh.gp_z[e].ravel()
        sig = blk["sig"][act].reshape(-1, 4)
        sel = np.abs(gpz - z_t) < 0.30 * A
        order = np.argsort(gpr[sel])
        return {"rho_nodes": rho, "u_rho": u_rho,
                "z_used": float(self.z_lines[iz]),
                "gp_rho": gpr[sel][order], "sig": sig[sel][order]}

    def liner_pressure(self):
        """p_s(zeta).  PRIMARY estimator = liner hoop-thrust ring
        equilibrium p = -sum(mean_stt * drho) / R_o over the radial
        liner stack; the rock-side first-cell sigma_rr reading sits at
        gp radius ~1.07a and is biased by the a^2/rho^2 gradient
        (smoke reading: the analytic value at the gp radius reproduces
        the bias exactly) -- reported with its radius as an
        offset-radius note only."""
        if self.liner_d is None:
            return None
        x_f = self.history[-1]["x_f"]
        blk = self.fe.blocks["rock"]
        e = blk["elems"]
        first = (np.abs(self.mesh.e_rho0[e] - A) < 1e-9)
        gz = self.mesh.gp_z[e][first]
        gr = self.mesh.gp_rho[e][first]
        srr = blk["sig"][first][:, :, 0]
        zeta = (gz.mean(axis=1) - x_f) / A
        p_rock = -srr.mean(axis=1)
        rock_gp_radius = float(gr.mean())
        lb = self.fe.blocks["liner"]
        act_l = lb["active"]
        le = lb["elems"][act_l]
        z_l, p_hoop = np.zeros(0), np.zeros(0)
        if le.size:
            stt = lb["sig"][act_l][:, :, 2].mean(axis=1)
            drho = self.mesh.e_rho1[le] - self.mesh.e_rho0[le]
            gzl = self.mesh.gp_z[le].mean(axis=1)
            zl_u = np.unique(np.round(gzl, 9))
            vals = []
            for zv in zl_u:
                sel = np.abs(gzl - zv) < 1e-8
                vals.append(-(stt[sel] * drho[sel]).sum() / A)
            z_l = (zl_u - x_f) / A
            p_hoop = np.asarray(vals)
        order = np.argsort(zeta)
        return {"zeta": zeta[order], "p_rock": p_rock[order],
                "rock_gp_radius": rock_gp_radius,
                "zeta_l": z_l, "p_hoop": p_hoop}


# ------------------------------------------------- V4c (2D-equivalent)

def solve_2d(arm, h_over_a, p_i=0.0, R_out_over_a=None,
             tol_rel=1e-10):
    """Plane-strain-equivalent axisymmetric slab (own FE, full MC
    return incl. edge/apex regimes), exact 2D stress-anchored lateral
    traction at rho = R_out (build_reference sigma_r is exact for all
    arms -- the face yield f13 = 0 stays active in the edge regime, so
    the stress ODE and R_p remain valid; only u / sigma_z carry the
    in-plane-flow assumption)."""
    d2 = arm.deep2d(p_i=p_i)
    R_out = (R_out_over_a or max(2.5 * d2.R_p / A, 5.0)) * A
    h = h_over_a * A
    core = np.array([0.0, 0.3, 0.6, 0.85, 1.0]) * A
    ann = np.arange(A + h, R_out + 0.5 * h, h)
    if abs(ann[-1] - R_out) > 1e-9:
        ann = np.append(ann, R_out)
    rho_lines = np.concatenate([core, ann])
    z_lines = np.array([0.0, 0.5 * A, 1.0 * A])
    mesh = AxiMesh(rho_lines, z_lines)
    fe = AxiFE(mesh, P0)
    sig0 = np.array([-P0, -P0, -P0, 0.0])
    fe.add_block("rock", np.arange(mesh.n_elems), arm.kind,
                 E_ROCK, NU_ROCK, mat=arm.mat, active=True, sig0=sig0)
    fe.dirichlet[1::2] = True                       # u_z = 0 everywhere
    for iz in range(mesh.nz1):                      # axis
        fe.dirichlet[2 * (iz * mesh.nr1)] = True
    sr_out = float(d2.sigma(np.array([R_out]))[0][0])
    lat = [(mesh.conn[e][1], mesh.conn[e][2])
           for e in range(mesh.n_elems)
           if mesh.elem_ir[e] == mesh.ne_r - 1]
    fe.f_ext_fix = fe.edge_load(lat, lambda r, z: (sr_out, 0.0))
    ir_a = int(np.argmin(np.abs(rho_lines - A)))
    if p_i != 0.0:
        wall = [(mesh.conn[e][0], mesh.conn[e][3])
                for e in range(mesh.n_elems)
                if mesh.elem_ir[e] == ir_a]
        # internal support pressure on the wall: n = -rho,
        # t = sigma.n = (+p_i, 0) tension-positive
        fe.f_ext_fix += fe.edge_load(wall, lambda r, z: (p_i, 0.0))
    core_e = mesh.elems_in_box(0.0, A, z_lines[0], z_lines[-1])
    fe.f_release = fe.removal_force("rock", core_e)
    fe.set_active("rock", core_e, False)
    u, info = fe.solve_step(tol_rel=tol_rel)
    return {"arm": arm, "d2": d2, "mesh": mesh, "fe": fe, "u": u,
            "info": info, "ir_a": ir_a, "rho_lines": rho_lines}


def own_2d_profiles(arm, h_over_a=0.05, p_i=0.0):
    """Own-FE fine-h 2D radial profiles: u_r(rho), sigma_z(rho) interp
    + u_wall + R_p.  Anchor amendment (report S-deviations): for arms
    with negative z-intermediacy margin the build_reference u/sigma_z
    are outside their validity domain; the own-FE full-return profile
    is the consistent deep-limit / back-BC source."""
    if arm.kind == "elastic":
        d2 = arm.deep2d(p_i=p_i)
        return {"u_r": d2.u_r,
                "sigma_z": lambda r: d2.sigma(np.asarray(r))[2],
                "u_wall": d2.u_wall(), "R_p": A, "d2": d2}
    out = solve_2d(arm, h_over_a, p_i=p_i)
    mesh, fe, u = out["mesh"], out["fe"], out["u"]
    ir_a = out["ir_a"]
    nodes = np.arange(ir_a, mesh.nr1)
    rho_n = mesh.rho[ir_a:]
    u_n = u[2 * nodes]
    blk = fe.blocks["rock"]
    act = blk["active"]
    e = blk["elems"][act]
    gpr = mesh.gp_rho[e].ravel()
    sz = blk["sig"][act].reshape(-1, 4)[:, 1]
    order = np.argsort(gpr)
    gpr_s, sz_s = gpr[order], sz[order]
    fl = blk["plast"][act]
    gpr2 = mesh.gp_rho[e]
    R_p = float(gpr2[fl].max()) if fl.any() else A

    def u_r(r):
        return np.interp(np.asarray(r, dtype=float), rho_n, u_n)

    def sigma_z(r):
        return np.interp(np.asarray(r, dtype=float), gpr_s, sz_s)

    return {"u_r": u_r, "sigma_z": sigma_z,
            "u_wall": float(u_n[0]), "R_p": R_p, "d2": out["d2"]}


def run_v4c(arm, h_over_a, R_out_over_a=None, tol_rel=1e-10):
    """V4c readings: u vs build_reference (validity-domain aware),
    sigma_r/sigma_theta profile vs reference (exact anchor, all arms),
    R_p, z-intermediacy margin, edge-regime gauss count."""
    out = solve_2d(arm, h_over_a, R_out_over_a=R_out_over_a,
                   tol_rel=tol_rel)
    mesh, fe, u, d2 = out["mesh"], out["fe"], out["u"], out["d2"]
    ir_a = out["ir_a"]
    nodes = np.arange(ir_a, mesh.nr1)
    rho = mesh.rho[ir_a:]
    u_fe = u[2 * nodes]
    u_ref = d2.u_r(rho)
    blk = fe.blocks["rock"]
    act = blk["active"]
    e = blk["elems"][act]
    gpr = mesh.gp_rho[e]
    fl = blk["plast"][act]
    R_p_fe = gpr[fl].max() if fl.any() else A
    sig = blk["sig"][act].reshape(-1, 4)
    gprf = gpr.ravel()
    sr_ref, st_ref, sz_ref = d2.sigma(gprf)
    rel_sr = np.max(np.abs(sig[:, 0] - sr_ref)) / P0
    rel_st = np.max(np.abs(sig[:, 2] - st_ref)) / P0
    # edge-regime detection: yielded gps with sigma_zz ~ sigma_theta
    from fedev_src.material import yield_values
    import torch as _t
    if arm.kind == "mc":
        f13 = yield_values(_t.as_tensor(sig), arm.mat, "mc").numpy()
        on_y = f13 > -1e-8
        n_edge = int((on_y & (np.abs(sig[:, 1] - sig[:, 2])
                              < 5e-3 * P0)).sum())
        margin = float(d2.ref.self_verify(n=400)
                       ["z_intermediate_margin"])
    else:
        on_y = np.zeros(sig.shape[0], dtype=bool)
        n_edge = 0
        margin = np.nan
    return {
        "h_over_a": h_over_a,
        "u_wall_fe": float(u_fe[0]), "u_wall_ref": float(u_ref[0]),
        "rel_u_wall": float(abs(u_fe[0] - u_ref[0]) / abs(u_ref[0])),
        "rel_u_max": float(np.max(np.abs(u_fe - u_ref)
                                  / np.max(np.abs(u_ref)))),
        "rel_sr_max": float(rel_sr), "rel_st_max": float(rel_st),
        "R_p_fe": float(R_p_fe), "R_p_ref": float(d2.R_p),
        "z_margin": margin, "n_yield": int(on_y.sum()),
        "n_edge": n_edge,
        "info": out["info"],
    }
