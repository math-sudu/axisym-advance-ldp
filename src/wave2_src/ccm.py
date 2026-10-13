# -*- coding: utf-8 -*-
"""2D deep-limit references + CCM support algebra (EXEC-18 freeze S4-2).

All tension-positive; radial displacement negative = inward (fe_dev
benchmark_ref convention).  Per freeze S1: elastic branch closed-form,
MC branch via fe_dev build_reference (read-only ODE reference).
"""

import numpy as np
from scipy.optimize import brentq


from fedev_src.benchmark_ref import build_reference  # read-only

__all__ = ["Deep2D", "ring_stiffness_hat", "ccm_equilibrium"]


class Deep2D:
    """2D plane-strain deep-limit fields for one arm.

    kind "elastic": closed-form Lame (no plastic annulus).
    kind "mc": fe_dev build_reference (raises if wall stays elastic).
    """

    def __init__(self, kind, p0, a, E, nu, mat=None, p_i=0.0):
        self.kind = kind
        self.p0, self.a = float(p0), float(a)
        self.E, self.nu = float(E), float(nu)
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.p_i = float(p_i)
        self.mat = dict(mat or {})
        if kind == "elastic":
            self.C = (self.p0 - self.p_i) * self.a ** 2
            self.R_p = self.a          # no plastic annulus
            self.ref = None
        elif kind == "mc":
            try:
                self.ref = build_reference(
                    "mc", self.mat, self.p0, self.a,
                    self.E, self.nu, p_i=self.p_i)
                self.R_p = self.ref.R_p
                self.C = self.ref.C
            except ValueError:
                # wall elastic at this p_i (e.g. wished-in-place
                # composite support pressure): elastic closed form is
                # the exact solution
                self.ref = None
                self.C = (self.p0 - self.p_i) * self.a ** 2
                self.R_p = self.a
        else:
            raise ValueError(kind)

    def u_r(self, r):
        r = np.asarray(r, dtype=float)
        if self.ref is None:
            return -self.C / (2.0 * self.G * r)
        return self.ref.u_r(r)

    def sigma(self, r):
        r = np.asarray(r, dtype=float)
        if self.ref is None:
            sr = -self.p0 + self.C / r ** 2
            st = -self.p0 - self.C / r ** 2
            sz = np.full_like(r, -self.p0)
            return sr, st, sz
        return self.ref.sigma(r)

    def u_wall(self):
        return float(self.u_r(np.array([self.a]))[0])


def ring_stiffness_hat(E_l, nu_l, R_il, R_o):
    """Exact plane-strain Lame ring radial stiffness K_hat [MPa/m]:

    external pressure p on rho = R_o, inner surface rho = R_il free;
    K_hat = p / (inward displacement at R_o).  Solved from the two-
    parameter Lame ring (sigma_r = A - B/r^2) -- no formula recall.
    """
    # BCs: sigma_r(R_il) = 0 ; sigma_r(R_o) = -p  (take p = 1)
    M = np.array([[1.0, -1.0 / R_il ** 2],
                  [1.0, -1.0 / R_o ** 2]])
    A, B = np.linalg.solve(M, np.array([0.0, -1.0]))
    # plane strain: u(r) = r/E [ (1-nu^2) sig_th - nu (1+nu) sig_r ]
    sig_r = A - B / R_o ** 2
    sig_t = A + B / R_o ** 2
    u_Ro = R_o / E_l * ((1.0 - nu_l ** 2) * sig_t
                        - nu_l * (1.0 + nu_l) * sig_r)
    return 1.0 / (-u_Ro)          # p=1 over inward displacement


def ccm_equilibrium(grc_u, K_hat, u_d, p0):
    """Deep-limit support pressure: solve p in [0, p0) with

        grc_u(p) = u_d - p / K_hat

    grc_u: callable p -> wall displacement (negative inward).
    Returns (p_s, u_inf).  p_s = 0 if the support never engages
    (grc_u(0) >= u_d, i.e. wall stops before reaching installation
    displacement -- cannot happen for u_d from the same GRC)."""
    def g(p):
        return grc_u(p) - (u_d - p / K_hat)
    if g(0.0) >= 0.0:
        return 0.0, grc_u(0.0)
    p_s = brentq(g, 0.0, p0 * (1 - 1e-12), xtol=1e-14, rtol=8.9e-16)
    return float(p_s), float(grc_u(p_s))


def grc_function(kind, p0, a, E, nu, mat=None):
    """Wall displacement as a function of internal pressure p_i.

    MC branch falls back to the elastic closed form when the wall
    state at p_i is elastic (build_reference raises)."""
    G = E / (2.0 * (1.0 + nu))

    def u_of_p(p_i):
        if kind == "elastic":
            return -(p0 - p_i) * a / (2.0 * G)
        try:
            ref = build_reference("mc", mat, p0, a, E, nu, p_i=p_i)
            return float(ref.u_r(np.array([a]))[0])
        except ValueError:
            return -(p0 - p_i) * a / (2.0 * G)
    return u_of_p
