"""Independent axisymmetric elastoplastic reference (MC / HB), K0 = 1.

Deep circular tunnel, plane strain, hydrostatic in-situ stress p0
(tension-positive sigma_r0 = sigma_th0 = sigma_z0 = -p0, matching the
package's sigma_z0 = K0*sigma_v choice at K0 = 1), inner support
pressure p_i = (1 - lambda) * p0 (full release: p_i = 0).

Construction (first principles, no FE ingredients):

* Plastic annulus a <= r <= R_p: radial equilibrium ODE
      d sigma_r / d r + (sigma_r - sigma_th) / r = 0
  with the hoop stress eliminated through the EXACT yield condition
  (sigma_1 = sigma_r, sigma_3 = sigma_th; the out-of-plane stress must
  stay intermediate -- verified):
      MC:  sigma_th = k_phi*sigma_r - sigma_cm,
           k_phi = (1+sin phi)/(1-sin phi),
           sigma_cm = 2 c cos phi/(1-sin phi)
      HB:  sigma_th = sigma_r - sigma_ci*(s - m_b*sigma_r/sigma_ci)**a
  integrated by scipy solve_ivp (tight tolerances, dense output) in
  t = ln(r/a) from sigma_r(a) = -p_i.
* Elastic zone r >= R_p (Lame): sigma_r = -p0 + C/r^2,
  sigma_th = -p0 - C/r^2, u = -C/(2 G r).
* Interface: sigma_r continuity + yield exactly reached from the
  elastic side give the scalar equation
      sigma_th_yield(x) = -2 p0 - x  at  x = sigma_r(R_p)
  (root by brentq), then R_p from the ODE event sigma_r = x, and
  C = (x + p0) R_p^2.
* Out-of-plane: flow is in-plane (MC / MC-type HB potential), so
  eps_z^p = 0 and sigma_z = -p0 + nu*(D sigma_r + D sigma_th)
  (elastic evolution); continuous at R_p by construction.
* Displacements in the plastic zone: total strains split
  eps = eps_e + eps_p with the non-associated in-plane flow
  eps_r^p = L*(1+sin psi)/2, eps_th^p = -L*(1-sin psi)/2, so
      eps_r + k_psi * eps_th = eps_r^e + k_psi * eps_th^e =: h(r),
      k_psi = (1+sin psi)/(1-sin psi)
  giving  du/dr + k_psi u/r = h(r)  integrated inward from
  u(R_p) = -C/(2 G R_p) (plane-strain compliance on stress CHANGES).

``self_verify`` substitutes the built solution back into radial
equilibrium (high-order finite differences on the dense output), the
yield condition, the elastic relations, interface continuity, the wall
boundary condition and the intermediate-stress ordering, and (for MC)
cross-checks the closed-form plastic sigma_r.  All residuals are
returned for the test suite to assert integrator/machine levels.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass
class AxisymRef:
    kind: str
    p0: float
    p_i: float
    a: float
    E: float
    nu: float
    mat: dict
    R_p: float
    x_star: float
    C: float
    _sol_sig: object
    _sol_u: object

    # ------------------------------------------------------------ fields
    def _sig_th_of_sig_r(self, sr):
        return _sig_th_of_sig_r(self.kind, self.mat, np.asarray(sr))

    def sigma(self, r):
        """(sigma_r, sigma_th, sigma_z) tension-positive at radii r."""
        r = np.asarray(r, dtype=float)
        sr = np.empty_like(r)
        st = np.empty_like(r)
        pl = r < self.R_p
        if pl.any():
            t = np.log(np.clip(r[pl], self.a, None) / self.a)
            sr[pl] = self._sol_sig.sol(t)[0]
            st[pl] = self._sig_th_of_sig_r(sr[pl])
        el = ~pl
        sr[el] = -self.p0 + self.C / r[el] ** 2
        st[el] = -self.p0 - self.C / r[el] ** 2
        sz = -self.p0 + self.nu * ((sr + self.p0) + (st + self.p0))
        return sr, st, sz

    def u_r(self, r):
        """Radial perturbation displacement (tension-positive frame:
        negative = inward convergence)."""
        r = np.asarray(r, dtype=float)
        u = np.empty_like(r)
        pl = r < self.R_p
        if pl.any():
            u[pl] = self._sol_u.sol(np.clip(r[pl], self.a, None))[0]
        u[~pl] = -self.C / (2.0 * self._G() * r[~pl])
        return u

    def _G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    # ------------------------------------------------------ verification
    def self_verify(self, n=4000):
        res = {}
        G = self._G()
        eps = 1e-6
        # ---- plastic zone: equilibrium via 8th-order central FD
        r = np.linspace(self.a * (1 + eps), self.R_p * (1 - eps), n)
        h = r[1] - r[0]
        sr, st, _ = self.sigma(r)
        c8 = np.array([1/280, -4/105, 1/5, -4/5, 0.0,
                       4/5, -1/5, 4/105, -1/280])
        dsr = np.convolve(sr, c8[::-1], mode="valid") / h
        rc, src, stc = r[4:-4], sr[4:-4], st[4:-4]
        res["eq_plastic"] = float(np.max(np.abs(
            dsr + (src - stc) / rc)) / (self.p0 / self.a))
        # ---- plastic zone: yield residual (exact substitution)
        res["yield_plastic"] = float(np.max(np.abs(
            _yield_f(self.kind, self.mat, sr, st))) / self.p0)
        # ---- elastic zone: equilibrium (closed form, sampled FD) + yield < 0
        re = np.linspace(self.R_p * (1 + eps), 10 * self.a, n)
        he = re[1] - re[0]
        sre, ste, _ = self.sigma(re)
        dsre = np.convolve(sre, c8[::-1], mode="valid") / he
        res["eq_elastic"] = float(np.max(np.abs(
            dsre + (sre[4:-4] - ste[4:-4]) / re[4:-4]))
            / (self.p0 / self.a))
        f_el = _yield_f(self.kind, self.mat, sre, ste)
        res["yield_elastic_max"] = float(np.max(f_el) / self.p0)  # < 0 req.
        # ---- elastic zone: Hooke consistency of u (closed form)
        ue = -self.C / (2.0 * G * re)
        eps_th = ue / re
        d_sr, d_st = sre + self.p0, ste + self.p0
        eps_th_hooke = ((1 - self.nu**2) * d_st
                        - self.nu * (1 + self.nu) * d_sr) / self.E
        res["hooke_elastic"] = float(np.max(np.abs(eps_th - eps_th_hooke))
                                     / np.max(np.abs(eps_th)))
        # ---- interface continuity + wall BC
        sr_pl_Rp = float(self._sol_sig.sol(np.log(self.R_p / self.a))[0])
        res["cont_sr"] = abs(sr_pl_Rp - (-self.p0 + self.C / self.R_p**2)) \
            / self.p0
        st_pl_Rp = float(self._sig_th_of_sig_r(sr_pl_Rp))
        res["cont_st"] = abs(st_pl_Rp - (-self.p0 - self.C / self.R_p**2)) \
            / self.p0
        u_pl_Rp = float(self._sol_u.sol(self.R_p)[0])
        res["cont_u"] = abs(u_pl_Rp - (-self.C / (2 * G * self.R_p))) \
            / (self.p0 * self.a / (2 * G))
        sr_a = float(self._sol_sig.sol(0.0)[0])
        res["wall_bc"] = abs(sr_a + self.p_i) / self.p0
        # ---- intermediate-stress ordering margins in the plastic zone
        srp, stp, szp = self.sigma(r)
        res["z_intermediate_margin"] = float(
            min(np.min(srp - szp), np.min(szp - stp)) / self.p0)  # > 0 req.
        # ---- displacement ODE residual (FD check)
        up = self.u_r(r)
        dup = np.convolve(up, c8[::-1], mode="valid") / h
        spsi = np.sin(self.mat["psi"])
        k_psi = (1 + spsi) / (1 - spsi)
        d_srp, d_stp = src + self.p0, stc + self.p0
        er_e = ((1 - self.nu**2) * d_srp
                - self.nu * (1 + self.nu) * d_stp) / self.E
        et_e = ((1 - self.nu**2) * d_stp
                - self.nu * (1 + self.nu) * d_srp) / self.E
        rhs = er_e + k_psi * et_e
        res["u_ode"] = float(np.max(np.abs(
            dup + k_psi * up[4:-4] / rc - rhs))
            / np.max(np.abs(rhs)))
        # ---- MC closed-form cross-check
        if self.kind == "mc":
            sphi = np.sin(self.mat["phi"])
            k = (1 + sphi) / (1 - sphi)
            scm = 2 * self.mat["c"] * np.cos(self.mat["phi"]) / (1 - sphi)
            src_cf = -((self.p_i + scm / (k - 1))
                       * (r / self.a) ** (k - 1) - scm / (k - 1))
            res["mc_closed_form"] = float(
                np.max(np.abs(sr - src_cf)) / self.p0)
        return res


def _yield_f(kind, mat, sr, st):
    """Tension-positive yield value with sigma_1 = sr, sigma_3 = st."""
    if kind == "mc":
        sphi, cphi = np.sin(mat["phi"]), np.cos(mat["phi"])
        return (0.5 * (sr - st) + 0.5 * (sr + st) * sphi
                - mat["c"] * cphi)
    if kind == "hb":
        X = np.clip(mat["s"] - mat["m_b"] * sr / mat["sigma_ci"],
                    0.0, None)
        return (sr - st) - mat["sigma_ci"] * X ** mat["a_hb"]
    raise ValueError(kind)


def _sig_th_of_sig_r(kind, mat, sr):
    if kind == "mc":
        sphi = np.sin(mat["phi"])
        k = (1 + sphi) / (1 - sphi)
        scm = 2 * mat["c"] * np.cos(mat["phi"]) / (1 - sphi)
        return k * sr - scm
    if kind == "hb":
        X = np.clip(mat["s"] - mat["m_b"] * sr / mat["sigma_ci"],
                    0.0, None)
        return sr - mat["sigma_ci"] * X ** mat["a_hb"]
    raise ValueError(kind)


def build_reference(kind, mat, p0, a, E, nu, p_i=0.0):
    """Construct the reference; raises if no plastic annulus forms."""
    G = E / (2.0 * (1.0 + nu))
    s_a = -float(p_i)

    def h_root(x):
        return float(_sig_th_of_sig_r(kind, mat, x) + 2.0 * p0 + x)

    if not h_root(s_a) > 0.0:
        raise ValueError("strength too high: wall state is elastic "
                         "(no plastic annulus)")
    if not h_root(-p0 * (1 - 1e-13)) < 0.0:
        raise ValueError("initial hydrostatic state violates yield")
    x_star = brentq(h_root, -p0 * (1 - 1e-13), s_a, xtol=1e-15,
                    rtol=8.9e-16, maxiter=200)

    def rhs_sig(t, y):
        return [float(_sig_th_of_sig_r(kind, mat, y[0]) - y[0])]

    def hit(t, y):
        return y[0] - x_star
    hit.terminal = True
    hit.direction = -1.0

    sol_sig = solve_ivp(rhs_sig, [0.0, 60.0], [s_a], events=hit,
                        method="DOP853", rtol=1e-13, atol=1e-14,
                        dense_output=True, max_step=0.05)
    if not sol_sig.t_events[0].size:
        raise RuntimeError("plastic ODE never reached x_star")
    t_p = float(sol_sig.t_events[0][0])
    R_p = a * np.exp(t_p)
    C = (x_star + p0) * R_p ** 2

    spsi = np.sin(mat["psi"])
    k_psi = (1 + spsi) / (1 - spsi)

    def rhs_u(r, y):
        t = np.log(r / a)
        sr = sol_sig.sol(t)[0]
        st = float(_sig_th_of_sig_r(kind, mat, sr))
        d_sr, d_st = sr + p0, st + p0
        er_e = ((1 - nu**2) * d_sr - nu * (1 + nu) * d_st) / E
        et_e = ((1 - nu**2) * d_st - nu * (1 + nu) * d_sr) / E
        return [er_e + k_psi * et_e - k_psi * y[0] / r]

    u_Rp = -C / (2.0 * G * R_p)
    sol_u = solve_ivp(rhs_u, [R_p, a], [u_Rp], method="DOP853",
                      rtol=1e-13, atol=1e-16, dense_output=True,
                      max_step=(R_p - a) / 50.0)

    return AxisymRef(kind=kind, p0=float(p0), p_i=float(p_i), a=float(a),
                     E=float(E), nu=float(nu), mat=dict(mat),
                     R_p=R_p, x_star=float(x_star), C=float(C),
                     _sol_sig=sol_sig, _sol_u=sol_u)
