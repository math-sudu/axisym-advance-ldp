# -*- coding: utf-8 -*-
"""Axisymmetric exterior solid harmonics for the EXEC-17 two-potential basis.

Exterior (decaying) axisymmetric solid harmonics about one source
center on the axis, in the LOCAL frame (rho, dz = z - z_c)::

  S_n(rho, dz) = r^{-(n+1)} P_n(x),   r = sqrt(rho^2 + dz^2),  x = dz / r

with P_n the Legendre polynomial.  The whole derivative closure used
by the two-potential synthesis (potentials.py) lives in three stable
identities (no autodiff, no sympy):

  d/dz  S_n = -(n+1) S_{n+1}                     (descent identity)
  d/drho S_n = T_n = -rho r^{-(n+3)} P'_{n+1}(x)  (associated-Legendre
               identity P^1_m = sin(phi) P'_m with the PHASE-FREE
               convention P^1_1 = sin(phi); the explicit rho factor
               makes T_n vanish on the axis exactly)
  W_n = T_n / rho = -r^{-(n+3)} P'_{n+1}(x)       (finite on the axis)

Second derivatives follow by re-applying the identities and the
harmonicity relation  d2/drho2 S_n = -d2/dz2 S_n - W_n.

Recurrences (upward, stable for |x| <= 1):

  (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
  P'_{n+1} = (2n+1) P_n + P'_{n-1},  P'_0 = 0, P'_1 = 1

Everything is float64 numpy; evaluation is vectorized over point
arrays and returns families stacked along axis 0 (order index n).
"""

import numpy as np

__all__ = ["legendre_all", "HarmonicFamily"]


def legendre_all(nmax, x):
    """P_n(x) and P'_n(x) for n = 0..nmax, shape (nmax+1,) + x.shape."""
    x = np.asarray(x, dtype=float)
    P = np.empty((nmax + 1,) + x.shape)
    dP = np.empty_like(P)
    P[0] = 1.0
    dP[0] = 0.0
    if nmax >= 1:
        P[1] = x
        dP[1] = 1.0
    for n in range(1, nmax):
        P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
        dP[n + 1] = (2 * n + 1) * P[n] + dP[n - 1]
    return P, dP


class HarmonicFamily:
    """S/T/W families about one axis center, orders n = 0..nmax.

    Parameters
    ----------
    nmax : int
        Highest order n of the family (the evaluation internally
        builds Legendre tables to nmax+3 so that the synthesis can
        reach S_{n+2} and T_{n+1} for every basis order).
    """

    def __init__(self, nmax):
        if nmax < 0:
            raise ValueError("nmax must be >= 0")
        self.nmax = int(nmax)

    def evaluate(self, rho, dz):
        """Evaluate the S, T, W families at points (rho, dz).

        Parameters
        ----------
        rho, dz : array_like
            Cylindrical coordinates in the LOCAL frame of the center
            (broadcast together).  r = 0 is a true singularity of the
            exterior family and raises.

        Returns
        -------
        dict of ndarray, each of shape (nmax+3,) + pts.shape
            ``"S"``  : S_n,          n = 0..nmax+2
            ``"T"``  : T_n = dS/drho, n = 0..nmax+2
            ``"W"``  : W_n = T_n/rho, n = 0..nmax+2
            plus ``"r"``, ``"x"`` (point geometry, pts.shape).
        """
        rho, dz = np.broadcast_arrays(
            np.asarray(rho, dtype=float), np.asarray(dz, dtype=float)
        )
        r = np.hypot(rho, dz)
        if np.any(r == 0.0):
            raise ValueError("exterior harmonics singular at r = 0")
        x = dz / r
        ntab = self.nmax + 3  # need P'_{n+1} up to n = nmax+2
        P, dP = legendre_all(ntab, x)
        inv_r = 1.0 / r
        # rp[k] = r^{-(k+1)} for k = 0..nmax+2
        rp = np.empty((self.nmax + 3,) + r.shape)
        rp[0] = inv_r
        for k in range(1, self.nmax + 3):
            rp[k] = rp[k - 1] * inv_r
        S = rp * P[: self.nmax + 3]
        # W_n = -r^{-(n+3)} P'_{n+1};  r^{-(n+3)} = rp[n] * inv_r^2
        W = np.empty_like(S)
        for n in range(self.nmax + 3):
            W[n] = -(rp[n] * inv_r * inv_r) * dP[n + 1]
        T = rho * W
        return {"S": S, "T": T, "W": W, "r": r, "x": x}
