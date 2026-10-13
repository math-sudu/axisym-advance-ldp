# -*- coding: utf-8 -*-
"""Axisymmetric two-potential (Boussinesq/PN) synthesis for EXEC-17.

Representation (interface finalization ruling 2026-07-17, Godoy eq. 22
form; TENSION-POSITIVE stresses, displacements positive along +rho/+z):

  2 mu u = grad(Phi + dz * Psi) - 4 (1 - nu) Psi zhat

with Phi, Psi harmonic and dz the LOCAL axial coordinate of the source
center (the global-z form differs by absorbing z_c*Psi into Phi, so the
local form is the same representation, numerically better scaled).

Stress closure (derived for this batch, unit-tested against central
differences and torch AD in tests/test_potentials.py):

  sigma_ij = Phi,ij + dz Psi,ij - (1-2nu)(Psi,i d_jz + Psi,j d_iz)
             - 2 nu Psi,z delta_ij

Cylindrical axisymmetric components for one basis element:

Phi-element of order n  (Phi = a^{n+3} S_n, Psi = 0; the a-power makes
every stress column dimensionless O(1) at r ~ a, coefficient carries
stress dimension):

  2mu u_rho = a^{n+3} T_n
  2mu u_z   = -(n+1) a^{n+3} S_{n+1}
  s_rr = a^{n+3} (-(n+1)(n+2) S_{n+2} - W_n)
  s_tt = a^{n+3} W_n
  s_zz = a^{n+3} (n+1)(n+2) S_{n+2}
  s_rz = -(n+1) a^{n+3} T_{n+1}

Psi-element of order n  (Psi = a^{n+2} S_n, Phi = 0):

  2mu u_rho = a^{n+2} dz T_n
  2mu u_z   = a^{n+2} (-(3-4nu) S_n - (n+1) dz S_{n+1})
  s_rr = a^{n+2} (dz (-(n+1)(n+2) S_{n+2} - W_n) + 2 nu (n+1) S_{n+1})
  s_tt = a^{n+2} (dz W_n + 2 nu (n+1) S_{n+1})
  s_zz = a^{n+2} (dz (n+1)(n+2) S_{n+2} + 2(1-nu)(n+1) S_{n+1})
  s_rz = a^{n+2} (-(n+1) dz T_{n+1} - (1-2nu) T_n)

Trace check: Phi-element traceless; Psi-element trace
= -2(1+nu) Psi,z = 2(1+nu)(n+1) a^{n+2} S_{n+1}  (both verified in
tests).  The Psi_0 element IS the axial Kelvin point force (u ~ 1/r,
net axial force through any enclosing sphere nonzero) -- the frozen
slow-decay completeness element of the basis dictionary (freeze S2).

Basis dictionary = [Phi_0..Phi_N] + [Psi_0..Psi_M] per source center;
centers sit on the axis INSIDE the cavity (non-material region), so
every basis field is smooth on the whole material domain.

Line-source elements (E-B second-round construction, replacing the
non-equilibrated H(z)*u2D pointwise carry): the semi-infinite line
distribution of dilatation centers along the cavity axis,

  Phi_L(rho, z; z_0) = -a^2 ln((z - z_0) + R),
  R = sqrt(rho^2 + (z - z_0)^2)

(harmonic; singular support = the half-line {rho = 0, z <= z_0},
inside the cavity for z_0 < x_f).  Asymptotics: deep behind the face
d(Phi_L)/drho -> -2 a^2/rho -- the 2D plane-strain Lame release
recovered EXACTLY as a field-theoretic limit (no ad-hoc transition
function, equilibrium built in); ahead of the face the field decays
algebraically.  This is the Boussinesq second-kind (logarithmic)
potential of Godoy 2017 in this scene's role; a pool of end points
z_0 acts as an equilibrated transition-shape dictionary for the LS.
Closed derivatives (zeta = z - z_0):

  Phi,rho = -a^2 rho / (R (zeta + R)),   Phi,z  = -a^2 / R
  Phi,zz  =  a^2 zeta / R^3,             Phi,rhoz = a^2 rho / R^3
  Phi,rhorho = -a^2 zeta / R^3 + a^2 / (R (zeta + R))

with the cancellation-free form zeta + R = rho^2 / (R - zeta) used
for zeta < 0.
"""

import numpy as np

from .harmonics import HarmonicFamily

__all__ = ["TwoPotentialBasis"]

_FIELDS = ("u_rho", "u_z", "s_rr", "s_tt", "s_zz", "s_rz")


class TwoPotentialBasis:
    """Dictionary of Phi/Psi exterior-harmonic elements, multi-center.

    Parameters
    ----------
    centers : sequence of float
        Axial source positions z_c (GLOBAL frame); each must lie in a
        non-material region of the problem (caller's responsibility).
    n_phi, n_psi : int
        Highest order of the Phi resp. Psi family per center
        (inclusive; families run n = 0..n_phi / 0..n_psi).
    a : float
        Length scale of the dimensionless column powers (tunnel /
        cavity radius).
    mu, nu : float
        Shear modulus and Poisson ratio.

    Attributes
    ----------
    labels : list of (str, float, int)
        Per-column ("phi"|"psi", z_c, n) in the fixed column order:
        centers outer loop, Phi block then Psi block inner.
    n_columns : int
    """

    def __init__(self, centers, n_phi, n_psi, a, mu, nu,
                 line_sources=()):
        self.centers = [float(c) for c in centers]
        if not self.centers:
            raise ValueError("at least one source center required")
        if n_phi < 0 or n_psi < 0:
            raise ValueError("n_phi and n_psi must be >= 0")
        self.n_phi = int(n_phi)
        self.n_psi = int(n_psi)
        self.a = float(a)
        self.mu = float(mu)
        self.nu = float(nu)
        self.line_sources = [float(z0) for z0 in line_sources]
        self._fam = HarmonicFamily(max(self.n_phi, self.n_psi))
        self.labels = []
        for z0 in self.line_sources:
            self.labels.append(("line", z0, -1))
        for z_c in self.centers:
            for n in range(self.n_phi + 1):
                self.labels.append(("phi", z_c, n))
            for n in range(self.n_psi + 1):
                self.labels.append(("psi", z_c, n))
        self.n_columns = len(self.labels)

    def columns(self, rho, z):
        """Unit-coefficient fields of every basis element.

        Parameters
        ----------
        rho, z : array_like
            GLOBAL cylindrical coordinates (broadcast together, then
            flattened).

        Returns
        -------
        dict with keys u_rho, u_z, s_rr, s_tt, s_zz, s_rz
            Each an ndarray of shape (n_columns, n_points): the field
            of basis element k with unit (stress-dimension)
            coefficient.  Displacements are TRUE displacements (the
            1/(2 mu) factor is applied here).
        """
        rho, z = np.broadcast_arrays(
            np.asarray(rho, dtype=float), np.asarray(z, dtype=float)
        )
        rho = rho.ravel()
        z = z.ravel()
        npts = rho.size
        out = {f: np.empty((self.n_columns, npts)) for f in _FIELDS}
        a = self.a
        nu = self.nu
        inv2mu = 0.5 / self.mu
        col = 0
        a2 = a * a
        for z0 in self.line_sources:
            zeta = z - z0
            R = np.hypot(rho, zeta)
            # cancellation-free zeta + R (zeta < 0 branch)
            zpr = np.where(zeta >= 0.0, zeta + R,
                           rho ** 2 / (R - zeta))
            out["u_rho"][col] = inv2mu * (-a2 * rho / (R * zpr))
            out["u_z"][col] = inv2mu * (-a2 / R)
            out["s_rr"][col] = -a2 * zeta / R ** 3 + a2 / (R * zpr)
            out["s_tt"][col] = -a2 / (R * zpr)
            out["s_zz"][col] = a2 * zeta / R ** 3
            out["s_rz"][col] = a2 * rho / R ** 3
            col += 1
        for z_c in self.centers:
            dz = z - z_c
            fam = self._fam.evaluate(rho, dz)
            S, T, W = fam["S"], fam["T"], fam["W"]
            # a-power ladders: ap3[n] = a^{n+3}, ap2[n] = a^{n+2}
            for n in range(self.n_phi + 1):
                ap = a ** (n + 3)
                c1 = (n + 1.0) * (n + 2.0)
                out["u_rho"][col] = inv2mu * ap * T[n]
                out["u_z"][col] = -inv2mu * ap * (n + 1.0) * S[n + 1]
                out["s_rr"][col] = ap * (-c1 * S[n + 2] - W[n])
                out["s_tt"][col] = ap * W[n]
                out["s_zz"][col] = ap * c1 * S[n + 2]
                out["s_rz"][col] = -ap * (n + 1.0) * T[n + 1]
                col += 1
            for n in range(self.n_psi + 1):
                ap = a ** (n + 2)
                c1 = (n + 1.0) * (n + 2.0)
                out["u_rho"][col] = inv2mu * ap * dz * T[n]
                out["u_z"][col] = inv2mu * ap * (
                    -(3.0 - 4.0 * nu) * S[n]
                    - (n + 1.0) * dz * S[n + 1]
                )
                out["s_rr"][col] = ap * (
                    dz * (-c1 * S[n + 2] - W[n])
                    + 2.0 * nu * (n + 1.0) * S[n + 1]
                )
                out["s_tt"][col] = ap * (
                    dz * W[n] + 2.0 * nu * (n + 1.0) * S[n + 1]
                )
                out["s_zz"][col] = ap * (
                    dz * c1 * S[n + 2]
                    + 2.0 * (1.0 - nu) * (n + 1.0) * S[n + 1]
                )
                out["s_rz"][col] = ap * (
                    -(n + 1.0) * dz * T[n + 1]
                    - (1.0 - 2.0 * nu) * T[n]
                )
                col += 1
        return out

    def synthesize(self, coefs, rho, z):
        """Fields of one coefficient vector (shape (n_columns,))."""
        coefs = np.asarray(coefs, dtype=float)
        if coefs.shape != (self.n_columns,):
            raise ValueError(
                f"coefs must have shape ({self.n_columns},),"
                f" got {coefs.shape}"
            )
        shape = np.broadcast_shapes(
            np.asarray(rho, dtype=float).shape,
            np.asarray(z, dtype=float).shape,
        )
        cols = self.columns(rho, z)
        return {f: (coefs @ cols[f]).reshape(shape) for f in _FIELDS}
