"""Material return, release ramp and Newton solve used in the paper."""
import json
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from contextlib import contextmanager
from fedev_src.material import lame, _compliance
from wave2_src import axifem, scenario as scen
from wave2_src.coupling import CoupledSolver

REFINEMENT_MAX_FAILED_ATTEMPTS = 60


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def subdivide(lines, level):
    n = 2 ** level
    lines = np.asarray(lines)
    if n == 1:
        return lines.copy()
    return np.concatenate([np.concatenate([
        np.linspace(lo, hi, n, endpoint=False) for lo, hi in zip(lines[:-1], lines[1:])
    ]), lines[-1:]])


def construction_axial_lines(lines, spacing_over_a, radius, track_end_over_a=14.6,
                             event_positions_over_a=()):
    """Refine the construction track while retaining the exterior mesh lines."""
    if spacing_over_a is None:
        return lines
    if not np.isfinite(spacing_over_a) or spacing_over_a <= 0:
        raise ValueError("Axial spacing must be finite and positive")
    count = round((track_end_over_a + 2.0) / spacing_over_a)
    if not np.isclose(count * spacing_over_a, track_end_over_a + 2.0):
        raise ValueError("Axial spacing must divide the construction track")
    track = np.round(np.linspace(-2.0, track_end_over_a, count + 1), 10) * radius
    outside = lines[(lines < -2 * radius - 1e-9) | (lines > track_end_over_a * radius + 1e-9)]
    events = np.asarray(event_positions_over_a, dtype=float)
    if not np.isfinite(events).all() or np.any(events < -2.0) or np.any(events > track_end_over_a):
        raise ValueError("Construction events must lie inside the refined axial track")
    return np.unique(np.concatenate((outside, track, events * radius)))


def save_npz(path, meta, arrays):
    """Exclusive, completed-file publication; previous observations survive."""
    tmp = path.with_suffix(".npz.part")
    with tmp.open("wb") as out:
        np.savez_compressed(out, meta=np.asarray(json.dumps(jsonable(meta),
                            sort_keys=True, allow_nan=False)), **arrays)
    if path.exists():
        raise FileExistsError(path)
    tmp.rename(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["meta"])), {
            k: data[k].copy() for k in data.files if k != "meta"}


def geometry(scn, k):
    m = scn.mesh
    face = k * scn.dstep
    z0 = scn.z_back if k == 0 else face - scn.dstep
    removed = scn._rings(z0, face)
    row = {"k": k, "face_m": face, "removal_z0_m": z0,
           "removal_z1_m": face, "removed_count": int(removed.size)}
    if removed.size:
        row.update(actual_removal_z0_m=float(m.e_z0[removed].min()),
                   actual_removal_z1_m=float(m.e_z1[removed].max()))
    if scn.liner_d is not None:
        block = scn.fe.blocks["liner"]
        eligible = m.e_z1[block["elems"]] <= face - scn.liner_d + 1e-9
        eligible_edges = m.e_z1[block["elems"][eligible]]
        edge = float(eligible_edges.max()) if eligible_edges.size else None
        row.update(nominal_activation_front_m=face - scn.liner_d,
                   actual_activation_front_m=edge,
                   activation_front_lag_over_a=(face - scn.liner_d - edge) / scen.A
                   if edge is not None else None,
                   newly_eligible_count=int((eligible & ~block["active"]).sum()))
    return row


def state_arrays(scn):
    arrays = {"rho_lines_m": scn.rho_lines, "z_lines_m": scn.z_lines,
              "displacement_m": scn.fe.u_n, "c_phys": scn.solver.c_phys,
              "F_back_2d": scn.F_back_2d, "F_back_s0": scn.F_back_s0,
              "wall_nodes": scn.wall_nodes}
    if scn.history:
        zeta, raw = scn.ldp()
        arrays.update(zeta_wall=zeta, u_wall=raw)
    for name, block in scn.fe.blocks.items():
        for field in ("elems", "sig", "active", "plast"):
            if field in block:
                arrays[name + "_" + field] = block[field]
    if scn.liner_d is not None and scn.history:
        pressure = scn.liner_pressure()
        arrays.update(zeta_liner=pressure["zeta_l"], p_hoop=pressure["p_hoop"])
        le = scn.fe.blocks["liner"]["elems"]
        for field in ("e_rho0", "e_rho1", "e_z0", "e_z1", "gp_z"):
            arrays["liner_" + field] = getattr(scn.mesh, field)[le]
    return arrays

def mc_sorted_roundoff(s1, s2, s3, E, nu, c, phi, psi, visc_H=0.0):
    """The pinned closed-form returns with scale-covariant branch predicates.

    gamma_64 supplies a conservative FP64 roundoff rule for these controls,
    not a formal error proof for all parameters or the full operation graph.
    The envelope is in stress units, independent of the Newton residual target.
    Edge multiplier bounds propagate this stress envelope through the exact
    infinity norm of the symmetric two-surface system's inverse. Only branch
    decisions change; finite face/edge/apex return equations are retained.
    """
    lam, G = lame(E, nu)
    sphi, cphi = torch.sin(phi), torch.cos(phi)
    spsi = torch.sin(psi)
    cb1 = lam * spsi + G * (1.0 + spsi)
    cb2 = lam * spsi
    cb3 = lam * spsi - G * (1.0 - spsi)
    d11 = lam * sphi * spsi + G * (1.0 + sphi * spsi)
    d11v = d11 + visc_H

    def f(sa, sb):
        return 0.5 * (sa - sb) + 0.5 * (sa + sb) * sphi - c * cphi

    unit_roundoff = torch.finfo(s1.dtype).eps / 2.0
    gamma = 64.0 * unit_roundoff / (1.0 - 64.0 * unit_roundoff)
    scale = s1.abs() + s2.abs() + s3.abs() + 2.0 * (c * cphi).abs()
    stress_roundoff = gamma * scale
    f13 = f(s1, s3)
    plastic = f13 > stress_roundoff
    dl_f = f13 / d11v
    s1f, s2f, s3f = s1 - dl_f * cb1, s2 - dl_f * cb2, s3 - dl_f * cb3
    face_ok = (s1f >= s2f - stress_roundoff) & (s2f >= s3f - stress_roundoff)

    d12r = lam * sphi * spsi + 0.5 * G * (1.0 - sphi) * (1.0 - spsi)
    f23 = f(s2, s3)
    det_r = d11v * d11v - d12r * d12r
    x1r = (d11v * f13 - d12r * f23) / det_r
    x2r = (d11v * f23 - d12r * f13) / det_r
    s1r = s1 - x1r * cb1 - x2r * cb2
    s2r = s2 - x1r * cb2 - x2r * cb1
    s3r = s3 - (x1r + x2r) * cb3
    right_roundoff = stress_roundoff * (d11v.abs() + d12r.abs()) / det_r.abs()
    right_ok = ((x1r >= -right_roundoff) & (x2r >= -right_roundoff)
                & (s2r >= s3r - stress_roundoff))

    d12l = lam * sphi * spsi + 0.5 * G * (1.0 + sphi) * (1.0 + spsi)
    f12 = f(s1, s2)
    det_l = d11v * d11v - d12l * d12l
    det_l = torch.where(det_l.abs() > 1e-30, det_l, torch.ones_like(det_l))
    x1l = (d11v * f13 - d12l * f12) / det_l
    x2l = (d11v * f12 - d12l * f13) / det_l
    s1l = s1 - (x1l + x2l) * cb1
    s2l = s2 - x1l * cb2 - x2l * cb3
    s3l = s3 - x1l * cb3 - x2l * cb2
    left_roundoff = stress_roundoff * (d11v.abs() + d12l.abs()) / det_l.abs()
    left_ok = ((x1l >= -left_roundoff) & (x2l >= -left_roundoff)
               & (s1l >= s2l - stress_roundoff))

    s_apex = c * cphi / torch.clamp_min(sphi, 1e-12)
    ea1_a, ea2_a, ea3_a = _compliance(s1 - s_apex, s2 - s_apex, s3 - s_apex, E, nu)
    L_sq = ea1_a * ea1_a + ea2_a * ea2_a + ea3_a * ea3_a
    positive = L_sq > 0.0
    L_ap = torch.where(positive, torch.sqrt(torch.where(positive, L_sq, torch.ones_like(L_sq))),
                       torch.zeros_like(L_sq))
    den_ap = f13 + visc_H * L_ap
    den_ap = torch.where(den_ap.abs() > 1e-30, den_ap, torch.ones_like(den_ap))
    t_ap = torch.clamp(visc_H * L_ap / den_ap, 0.0, 1.0)
    r_ap = 1.0 - t_ap
    a1, a2, a3 = (s_apex + t_ap * (s - s_apex) for s in (s1, s2, s3))
    ea1, ea2, ea3 = r_ap * ea1_a, r_ap * ea2_a, r_ap * ea3_a
    sel_face = plastic & face_ok
    need_edge = plastic & ~face_ok
    try_right = need_edge & (s1f < s2f - stress_roundoff)
    sel_right = try_right & right_ok
    sel_left = need_edge & ~try_right & left_ok
    sel_apex = plastic & ~sel_face & ~sel_right & ~sel_left
    bp, bm = 0.5 * (1.0 + spsi), -0.5 * (1.0 - spsi)
    zero = torch.zeros_like(s1)

    def pick(vf, vr, vl, va, velse):
        return torch.where(sel_face, vf, torch.where(sel_right, vr,
            torch.where(sel_left, vl, torch.where(sel_apex, va, velse))))

    t1 = pick(s1f, s1r, s1l, a1, s1)
    t2 = pick(s2f, s2r, s2l, a2, s2)
    t3 = pick(s3f, s3r, s3l, a3, s3)
    e1 = pick(dl_f * bp, x1r * bp, (x1l + x2l) * bp, ea1, zero)
    e2 = pick(zero, x2r * bp, x2l * bm, ea2, zero)
    e3 = pick(dl_f * bm, (x1r + x2r) * bm, x1l * bm, ea3, zero)
    dlam = pick(dl_f, x1r + x2r, x1l + x2l, r_ap * L_ap, zero)
    region = (sel_face.to(torch.int64) + 2 * sel_right.to(torch.int64)
              + 3 * sel_left.to(torch.int64) + 4 * sel_apex.to(torch.int64))
    return t1, t2, t3, e1, e2, e3, dlam, region

@contextmanager
def corrected_kernel():
    old = axifem._mc_sorted
    axifem._mc_sorted = mc_sorted_roundoff
    try:
        yield
    finally:
        axifem._mc_sorted = old

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

class RefinementCoupled(CoupledSolver):
    """Traced pinned solver with refinement-appropriate ramp guards.

    ``TracedCoupled.solve_step`` inherits the production ``attempts > 60``
    guard.  Once ``d_lam_cap < 1/60``, any later failed trial sees more than 60
    total attempts even though the mandatory successful substeps are not a
    failure.  Here the budget applies to failed Newton trials only, and the
    production ``1e-4`` minimum increment is replaced by the actual floating-
    point forward-progress boundary.  The Newton implementation, convergence
    tolerance, state commits, cap recovery, and finite-element equations are
    unchanged.
    """

    max_failed_attempts = REFINEMENT_MAX_FAILED_ATTEMPTS

    def solve_step(self, tol_rel=1e-10):
        fe = self.fe
        lam_done = 0.0
        u = fe.u_n.copy()
        d_lam = self.d_lam_cap
        attempts = 0
        failed_attempts = 0
        failed_trials = []
        substeps = 0
        seqs = []
        self.last_ramp_failure = None
        while lam_done < 1.0 - 1e-12:
            lam_try = min(1.0, lam_done + d_lam)
            u_try, _c_try, info = self.newton(
                u0=u, lam_release=lam_try, tol_rel=tol_rel)
            attempts += 1
            if info["converged"]:
                u = u_try
                fe.commit(u)
                lam_done = lam_try
                substeps += 1
                seqs.append(info["seq"])
                d_lam = min(self.d_lam_cap, d_lam * 2.0)
            else:
                failed_attempts += 1
                res_rel = float(info["res_rel"])
                residual_sequence = [
                    float(v) if math.isfinite(float(v)) else None
                    for v in info.get("seq", [])]
                failed_trials.append({
                    "lam_done": float(lam_done),
                    "lam_try": float(lam_try),
                    "d_lam_attempted": float(d_lam),
                    "res_rel": res_rel if math.isfinite(res_rel) else None,
                    "res_rel_is_finite": bool(math.isfinite(res_rel)),
                    "newton_iterations": int(info["iters"]),
                    "residual_sequence": residual_sequence,
                })
                next_d_lam = d_lam * 0.5
                guard_reason = None
                if failed_attempts > self.max_failed_attempts:
                    guard_reason = "failed_newton_attempt_limit"
                elif min(1.0, lam_done + next_d_lam) <= lam_done:
                    guard_reason = "floating_point_no_forward_progress"
                if guard_reason is not None:
                    self.last_ramp_failure = {
                        "guard_reason": guard_reason,
                        "lam_done": float(lam_done),
                        "next_d_lam": float(next_d_lam),
                        "failed_attempts": int(failed_attempts),
                        "failed_trials": failed_trials,
                    }
                    raise RuntimeError(
                        f"coupled release ramp failed at lam={lam_done} "
                        f"(next_d_lam={next_d_lam}, res={res_rel:.3e}, "
                        f"failed_attempts={failed_attempts}, "
                        f"guard={guard_reason})")
                d_lam = next_d_lam
        return u, {
            "substeps": substeps,
            "attempts": attempts,
            "failed_attempts": failed_attempts,
            "failed_trials": failed_trials,
            "c": self.c_phys.copy(),
            "seqs": seqs,
        }

def _newton(self, u0=None, lam_release=1.0, tol_rel=1e-10, max_iter=60,
            c0=None, progress=None, pseudo_transient=False):
    fe, mp = self.fe, self._maps
    B, T = mp["B"], mp["T"]
    if_dofs = self.ext.if_dofs
    act = fe.active_dof_mask()
    act[if_dofs] = True
    free = act & ~fe.dirichlet
    free[if_dofs] = False
    idx_o = np.where(free)[0]
    n_o, n_c = idx_o.size, B.shape[1]
    u = fe.u_n.copy() if u0 is None else u0.copy()
    c = self.c.copy() if c0 is None else c0.copy()
    u[if_dofs] = B @ c
    fscale = max(np.linalg.norm(fe.f_release), np.linalg.norm(self.F0), fe.p0)
    seq, searches = [], []
    kinds = {name: block["kind"] for name, block in fe.blocks.items()}
    try:
        for block in fe.blocks.values():
            block["kind"] = "elastic"
        _, E = fe.internal_force(u, want_tangent=True)
    finally:
        for name, block in fe.blocks.items():
            block["kind"] = kinds[name]
    Bs = sp.csr_matrix(B)
    elastic_cc = B.T @ (E[if_dofs][:, if_dofs] @ B)
    elastic_cc -= B.T @ T[if_dofs, :]
    elastic_cc[np.diag_indices(n_c)] += self.eps_ridge * np.abs(elastic_cc).max()
    reduced_elastic = sp.bmat([
        [E[idx_o][:, idx_o], E[idx_o][:, if_dofs] @ Bs],
        [Bs.T @ E[if_dofs][:, idx_o], sp.csr_matrix(elastic_cc)]], format="csc")
    merit_lu = spla.splu(reduced_elastic)

    def merit(full):
        reduced = np.concatenate([full[idx_o], B.T @ full[if_dofs]])
        return np.linalg.norm(merit_lu.solve(reduced))

    def resid(uu, cc):
        F, _ = fe.internal_force(uu, want_tangent=False)
        R = F - fe.f_ext_fix + (1.0 - lam_release) * fe.f_release - self.F0 - T @ cc
        return R, np.sqrt(np.linalg.norm(R[idx_o]) ** 2 + np.linalg.norm(B.T @ R[if_dofs]) ** 2)

    def done(converged, reason):
        if converged:
            self.c = c
        info = {"iters": len(seq) - 1, "res_rel": seq[-1], "converged": converged,
                "seq": seq, "termination": reason, "line_searches": searches,
                "search_method": "pseudo_transient" if pseudo_transient else "coupled_elastic_merit"}
        self.spatial_trials.append({"lam_release": float(lam_release), "tol_rel": float(tol_rel),
                                    **jsonable(info)})
        return u, c, info

    R, rnorm = resid(u, c)
    current_merit = merit(R)
    # The last update gets a residual check too. max_iter still bounds the
    # number of Newton corrections, not the number of residual observations.
    for it in range(max_iter + 1):
        res = float(rnorm / fscale)
        seq.append(res)
        if progress is not None:
            progress({"iteration": it, "res_rel": res, "merit": float(current_merit),
                      "pseudo_transient": pseudo_transient})
        if not np.isfinite(res) or not np.isfinite(current_merit):
            return done(False, "nonfinite_residual_or_merit")
        if res < tol_rel:
            return done(True, "residual_tolerance")
        if it == max_iter:
            return done(False, "newton_correction_limit")
        _, K = fe.internal_force(u, want_tangent=True)
        K_oo = K[idx_o][:, idx_o]
        K_oi = K[idx_o][:, if_dofs] @ sp.csr_matrix(B)
        K_io = sp.csr_matrix(B.T) @ K[if_dofs][:, idx_o]
        K_cc = B.T @ (K[if_dofs][:, if_dofs] @ B) - B.T @ T[if_dofs, :]
        K_cc[np.diag_indices(n_c)] += self.eps_ridge * np.abs(K_cc).max()
        matrix = sp.bmat([[K_oo, K_oi], [K_io, sp.csr_matrix(K_cc)]], format="csc")
        if pseudo_transient:
            inverse_dt = res / seq[0]
            matrix = matrix + inverse_dt * reduced_elastic
        direction = spla.spsolve(matrix, -np.concatenate([R[idx_o], B.T @ R[if_dofs]]))
        if not np.isfinite(direction).all():
            return done(False, "nonfinite_newton_direction")
        if pseudo_transient:
            u[idx_o] += direction[:n_o]
            c += direction[n_o:]
            u[if_dofs] = B @ c
            R, rnorm = resid(u, c)
            current_merit = merit(R)
            searches.append({"direction": "pseudo_transient", "inverse_dt": inverse_dt,
                             "res_rel": float(rnorm/fscale), "accepted": True})
            continue
        alpha, trials = 1.0, 0
        while True:
            bound = current_merit * (1 - 1e-4 * alpha)
            uu, cc = u.copy(), c + alpha * direction[n_o:]
            uu[idx_o] += alpha * direction[:n_o]
            uu[if_dofs] = B @ cc
            # A rounded-away Armijo decrease cannot establish descent.
            if bound == current_merit or (np.array_equal(uu, u) and np.array_equal(cc, c)):
                searches.append({"trials": trials, "alpha": alpha, "accepted": False})
                return done(False, "no_representable_line_search_update")
            rr, nn = resid(uu, cc)
            trials += 1
            trial_merit = merit(rr)
            if trial_merit < bound or nn / fscale < tol_rel:
                searches.append({"trials": trials, "alpha": alpha, "accepted": True,
                    "reference_merit": float(bound), "merit": float(trial_merit),
                    "metric": "coupled_elastic_correction", "direction": "newton"})
                u, c, R, rnorm, current_merit = uu, cc, rr, nn, trial_merit
                break
            alpha *= 0.5


def newton(self, u0=None, lam_release=1.0, tol_rel=1e-10, max_iter=60,
           c0=None, progress=None):
    """Use pseudo-transient continuation to leave a stalled Newton trajectory.

    No constitutive commit is made within the search phases. The transient
    iterate is only an initial guess for the original equilibrium equation;
    the ramp commits it only after the same physical force check passes.
    """
    result = _newton(self, u0, lam_release, tol_rel, max_iter, c0, progress)
    info = result[2]
    if not info["converged"] and info["termination"] in (
            "no_representable_line_search_update", "newton_correction_limit"):
        first = self.spatial_trials.pop()
        result = _newton(self, u0, lam_release, tol_rel, max_iter, c0, progress,
                         pseudo_transient=True)
        if not result[2]["converged"] and result[2]["termination"] == "newton_correction_limit":
            transient = self.spatial_trials.pop()
            result = _newton(self, result[0], lam_release, tol_rel, max_iter, result[1], progress)
            result[2]["pseudo_transient_start"] = transient
            self.spatial_trials[-1]["pseudo_transient_start"] = transient
        result[2]["search_restart"] = first
        self.spatial_trials[-1]["search_restart"] = first
    return result
