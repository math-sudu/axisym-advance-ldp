"""Vectorized differentiable return mapping (perfect MC / HB), torch f64.

Stress state layout (TOTAL stress, tension-positive):
    sig4 = (s_xx, s_yy, s_zz, s_xy)          last-dim length 4
Strain increment layout (perturbation, plane strain -> deps_zz = 0):
    deps3 = (deps_xx, deps_yy, dgamma_xy)    engineering shear
Plastic-strain increments are returned in the layout
    (eps_xx, eps_yy, eps_zz, gamma_xy)       engineering shear.

Return mapping runs in principal-stress space (the out-of-plane axis is
a principal direction; the in-plane 2x2 block is diagonalized
analytically).  Backward-Euler elastic predictor + multi-surface
(Koiter) plastic corrector with EXACT surfaces -- no smoothing:

* Mohr-Coulomb (tension-positive, sorted s1 >= s2 >= s3):
      f_ab = 0.5*(s_a - s_b) + 0.5*(s_a + s_b)*sin(phi) - c*cos(phi)
  main-plane return (closed form), two-vector edge returns (closed
  form), apex return (s_i = c*cot(phi)).  Non-associated flow with the
  same expression at dilation psi.
* Generalized Hoek-Brown (tension-positive):
      f_ab = (s_a - s_b) - sigma_ci*(s - m_b*s_a/sigma_ci)**a_hb
  MC-type plastic potential with constant dilation psi_hb (documented
  package choice); main-plane return by safeguarded 1D Newton (analytic
  bracket [0, f_tr/(2G)]), edge returns by damped 2x2 Newton, apex
  return to the hydrostatic tension cutoff s_i = s*sigma_ci/m_b.
* CWFS strain-softening Mohr-Coulomb (tier B, kind "cwfs"): the MC
  surfaces above with kappa-dependent strength
      c(k)   = c_peak + (c_res - c_peak)*min(k/gamma_c, 1)
      phi(k) = phi_peak + (phi_res - phi_peak)*min(k/gamma_phi, 1)
  and constant dilation psi.  Internal variable kappa = equivalent
  plastic shear strain; per increment
      d_kappa = deps_p_max - deps_p_min
  over the returned principal plastic-strain increments (equals the
  reference kernel's |deps_p1 - deps_p3|; on the main plane, with the
  0.5-normalized potential, d_kappa = dlam exactly -- a property, not
  an implementation shortcut: d_kappa is always computed from the
  plastic-strain triple).  Fully implicit backward Euler: the return
  satisfies f(sig_new, kappa_old + d_kappa(dlam)) = 0 with c, phi at
  kappa_new.  Face: bracketed safeguarded Newton (F can lose
  monotonicity under softening, so the bracket is grown by doubling
  from the perfect-plasticity seed until F <= 0, then bisection
  safeguards every Newton step); edges: 2x2 Newton (perfect-MC linear
  part plus a rank-one softening coupling through d_kappa); apex:
  CLOSED FORM -- d_kappa = e1 - e3 = (1+nu)*(s1-s3)/E is independent
  of the (kappa-dependent) apex position, so kappa_new and the apex
  s_i = c(k)*cos(phi(k))/sin(phi(k)) are explicit.  All iterative
  solves run DETACHED with ONE differentiable implicit-function polish
  step from the converged multipliers (the HB pattern).

* Perzyna/Bingham regularization of the MC return (tier C, consumed
  by material_visc): ``_mc_sorted`` accepts an optional scalar
  visc_H = eta_vp/dt (default 0.0 == the exact rate-independent
  behavior, bit-for-bit).  Backward-Euler Perzyna consistency
      f_j(sig_new) = visc_H * dlam_j
  turns every multiplier equation into its linear-hardening analogue:
  the face closed form divides by (d11 + visc_H), the two-surface edge
  systems gain visc_H on the diagonal, and the apex uses the documented
  Koiter-Perzyna RAY treatment -- sig_new stays on the trial->apex ray
  (where every f is linear, f13(apex + t*(sig_tr - apex)) = t*f13_tr)
  and the scalar consistency f13(sig_new) = visc_H * dlam with the
  apex multiplier measure dlam = (1 - t) * L,
  L = ||C^-1 : (sig_tr - apex)||, gives
      t = visc_H*L / (f13_tr + visc_H*L)
  (t = 0: the exact apex projection; t -> 1: no return as
  visc_H -> inf; NaN-free and branchless in unselected lanes by a
  guarded denominator plus a [0, 1] clamp).  The compressive tunnel
  regime never exercises the apex; the treatment is smoke-tested in
  tests/test_nishihara_material.py.

* Perzyna regularization of the CWFS return (softening path; THESIS-
  NEW construction): ``_cwfs_sorted`` accepts the same optional scalar
  visc_H (consumed through the mat-dict key "visc_H", see
  ``integrate_full``; default 0.0 / absent key == the exact rate-
  independent softening return, bit-for-bit, regression-guarded by
  tests/test_cwfs_perzyna.py hex goldens).  The reference CWFS line
  this package's form is locked to (the sha256-pinned kernel behind
  tests/data/gen_cwfs_golden.py) carries NO viscous regularization
  anywhere -- there is no reference form to defer to -- so the
  construction is documented here: the SAME backward-Euler Perzyna
  consistency f_j(sig_new, kappa_new) = visc_H * dlam_j is appended
  to the fully implicit softening equations (laws still evaluated at
  the UPDATED kappa).  Face: the scalar residual gains -visc_H*x
  (F' gains -visc_H, strengthening the bracket/Newton monotonicity
  that softening erodes); edges: the 2x2 systems gain -visc_H on the
  Jacobian diagonal (residuals f_i - visc_H*x_i); apex: the Koiter-
  Perzyna ray of the MC path with the laws FROZEN at the exact-apex
  kappa update kap + dk_a -- a frozen-strength linearization of the
  ray consistency, exact at visc_H = 0 (t_ap = 0 reproduces the
  closed-form apex bit-for-bit; the compressive tunnel regime never
  selects the apex).  Regularization semantics: visc_H = eta_vp/dt of
  a Perzyna overstress law at a FIXED ratio -- the standard viscous
  regularization of softening localization; it adds a positive rate
  term to every active multiplier equation (restoring uniqueness of
  the return where the softening modulus overwhelms d11) and
  stiffens the consistent tangent on the plastic diagonal.  Driven
  by a lambda schedule, a constant visc_H defines a FIXED regularized
  pseudo-time problem (viscoplastic continuation); it is NOT a
  physical time integration (that is the tier-C Nishihara path with
  its strict dt/times contract).

Region selection (standard sequential multi-surface logic, e.g.
de Souza Neto/Peric/Owen): elastic if f_tr <= 0; main-plane return if
the returned principal ordering is preserved; otherwise the edge picked
by the violated ordering side; otherwise apex.  Implemented branchless
(torch.where masks) so the same code path is vmap/jacrev-compatible;
every branch is sanitized to stay NaN-free even where unselected.

Consistent algorithmic tangents are exact reverse-mode Jacobians of
this return map (torch.func.jacrev + vmap), so tangent and stress
update can never drift apart.

Nonsmooth caveats (documented): repeated in-plane trial eigenvalues,
region-boundary trial states, and (for psi = 0) the MC apex are
measure-zero switching configurations; derivative-based tests must not
sit exactly on them.  The apex return places the stress exactly at the
apex; for psi = 0 the associated volumetric flow inconsistency at the
apex regime is the standard engineering treatment and is outside this
project's compressive tunnel regime.  CWFS adds two nonsmooth loci:
(i) the law kinks kappa = gamma_c / gamma_phi (clamp subgradient is
the documented choice at the kink -- gradient-based tests must place
base points off the kinks), and (ii) the edge-multiplier tie x1 = x2
where d_kappa involves max(x1, x2) (subgradient picks the x1 side;
values stay continuous, so vmap/jacrev remain NaN-free).
"""

import torch

DTYPE = torch.float64
_TINY_RAD = 1e-150     # eigen-radius clamp (gradient-safe sqrt)
_X_EPS = 1e-16         # HB yield-domain clamp (dimensionless)
_N_HB_FACE = 24        # safeguarded-Newton iterations (face)
_N_HB_EDGE = 24        # damped-Newton iterations (edges)
_N_CWFS_GROW = 24      # CWFS face bracket-expansion doublings
_N_CWFS_FACE = 48      # CWFS safeguarded-Newton iterations (face)
_N_CWFS_EDGE = 48      # CWFS 2x2-Newton iterations (edges)

IN_PLANE_IDX = (0, 1, 3)


def lame(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return lam, G


def elastic_trial(deps3, sig4, E, nu):
    """sig_trial = sig + C : deps (plane strain, deps_zz = 0)."""
    lam, G = lame(E, nu)
    exx, eyy, gxy = deps3[..., 0], deps3[..., 1], deps3[..., 2]
    tr = exx + eyy
    return torch.stack([
        sig4[..., 0] + lam * tr + 2.0 * G * exx,
        sig4[..., 1] + lam * tr + 2.0 * G * eyy,
        sig4[..., 2] + lam * tr,
        sig4[..., 3] + G * gxy,
    ], dim=-1)


# ------------------------------------------------ principal decomposition

def _principal4(sig4):
    """In-plane eigenvalues sA >= sB, out-of-plane sZ, direction (c2, s2)."""
    sxx, syy, szz, sxy = (sig4[..., 0], sig4[..., 1],
                          sig4[..., 2], sig4[..., 3])
    mean = 0.5 * (sxx + syy)
    half = 0.5 * (sxx - syy)
    rad = torch.sqrt(torch.clamp_min(half * half + sxy * sxy,
                                     _TINY_RAD * _TINY_RAD))
    degen = rad <= _TINY_RAD
    c2 = torch.where(degen, torch.ones_like(rad), half / rad)
    s2 = torch.where(degen, torch.zeros_like(rad), sxy / rad)
    return mean + rad, mean - rad, szz, c2, s2


def _sort3(sA, sB, sZ):
    """Sort (sA >= sB, sZ) into s1 >= s2 >= s3; masks identify the case."""
    m0 = sZ >= sA                       # sZ largest
    m2 = (~m0) & (sZ <= sB)             # sZ smallest
    s1 = torch.where(m0, sZ, sA)
    s2 = torch.where(m0, sA, torch.where(m2, sB, sZ))
    s3 = torch.where(m0, sB, torch.where(m2, sZ, sB))
    return s1, s2, s3, m0, m2


def _unsort3(t1, t2, t3, m0, m2):
    """Inverse of :func:`_sort3` for values returned in sorted slots."""
    vZ = torch.where(m0, t1, torch.where(m2, t3, t2))
    vA = torch.where(m0, t2, t1)
    vB = torch.where(m0, t3, torch.where(m2, t2, t3))
    return vA, vB, vZ


def _recompose_stress(vA, vB, vZ, c2, s2):
    mean = 0.5 * (vA + vB)
    half = 0.5 * (vA - vB)
    return torch.stack([mean + half * c2, mean - half * c2,
                        vZ, half * s2], dim=-1)


def _recompose_strain(eA, eB, eZ, c2, s2):
    """Principal strains -> (eps_xx, eps_yy, eps_zz, gamma_xy)."""
    mean = 0.5 * (eA + eB)
    half = 0.5 * (eA - eB)
    return torch.stack([mean + half * c2, mean - half * c2,
                        eZ, 2.0 * half * s2], dim=-1)


def _compliance(d1, d2, d3, E, nu):
    """Principal elastic compliance eps_i = (ds_i - nu*(ds_j + ds_k))/E."""
    tr = d1 + d2 + d3
    return (((1.0 + nu) * d1 - nu * tr) / E,
            ((1.0 + nu) * d2 - nu * tr) / E,
            ((1.0 + nu) * d3 - nu * tr) / E)


# --------------------------------------------------------- Mohr-Coulomb

def _mc_sorted(s1, s2, s3, E, nu, c, phi, psi, visc_H=0.0):
    """MC return on sorted principals; optional Perzyna term ``visc_H``.

    ``visc_H = eta_vp/dt`` adds the backward-Euler Perzyna consistency
    f_j(sig_new) = visc_H * dlam_j to every multiplier equation (linear
    hardening analogue; module docstring).  The default 0.0 reproduces
    the rate-independent return BIT-FOR-BIT (regression-guarded by
    tests/test_nishihara_material.py).
    """
    lam, G = lame(E, nu)
    sphi, cphi = torch.sin(phi), torch.cos(phi)
    spsi = torch.sin(psi)
    # C_p action on the potential gradient b_13 = (bp, 0, bm)
    cb1 = lam * spsi + G * (1.0 + spsi)
    cb2 = lam * spsi
    cb3 = lam * spsi - G * (1.0 - spsi)
    d11 = lam * sphi * spsi + G * (1.0 + sphi * spsi)
    d11v = d11 + visc_H

    def f(sa, sb):
        return 0.5 * (sa - sb) + 0.5 * (sa + sb) * sphi - c * cphi

    f13 = f(s1, s3)
    plastic = f13 > 0.0

    # --- main-plane return (closed form, f13(sig_new) = visc_H*dlam)
    dl_f = f13 / d11v
    s1f, s2f, s3f = s1 - dl_f * cb1, s2 - dl_f * cb2, s3 - dl_f * cb3
    face_ok = (s1f >= s2f) & (s2f >= s3f)

    # --- right edge (s1 = s2): surfaces f13, f23 (visc_H on the diag)
    d12r = lam * sphi * spsi + 0.5 * G * (1.0 - sphi) * (1.0 - spsi)
    f23 = f(s2, s3)
    det_r = d11v * d11v - d12r * d12r
    x1r = (d11v * f13 - d12r * f23) / det_r
    x2r = (d11v * f23 - d12r * f13) / det_r
    s1r = s1 - x1r * cb1 - x2r * cb2
    s2r = s2 - x1r * cb2 - x2r * cb1
    s3r = s3 - (x1r + x2r) * cb3
    right_ok = (x1r >= 0.0) & (x2r >= 0.0) & (s2r >= s3r)

    # --- left edge (s2 = s3): surfaces f13, f12 (visc_H on the diag)
    d12l = lam * sphi * spsi + 0.5 * G * (1.0 + sphi) * (1.0 + spsi)
    f12 = f(s1, s2)
    det_l = d11v * d11v - d12l * d12l
    det_l = torch.where(det_l.abs() > 1e-30, det_l, torch.ones_like(det_l))
    x1l = (d11v * f13 - d12l * f12) / det_l
    x2l = (d11v * f12 - d12l * f13) / det_l
    s1l = s1 - (x1l + x2l) * cb1
    s2l = s2 - x1l * cb2 - x2l * cb3
    s3l = s3 - x1l * cb3 - x2l * cb2
    left_ok = (x1l >= 0.0) & (x2l >= 0.0) & (s1l >= s2l)

    # --- apex: Koiter-Perzyna ray return (module docstring).  sig_new
    # stays on the trial->apex ray at fraction t_ap from the apex;
    # visc_H = 0 gives t_ap = 0 (exact apex projection, bit-for-bit).
    s_apex = c * cphi / torch.clamp_min(sphi, 1e-12)
    ea1_a, ea2_a, ea3_a = _compliance(s1 - s_apex, s2 - s_apex,
                                      s3 - s_apex, E, nu)
    L_ap = torch.sqrt(ea1_a * ea1_a + ea2_a * ea2_a + ea3_a * ea3_a)
    den_ap = f13 + visc_H * L_ap
    den_ap = torch.where(den_ap.abs() > 1e-30, den_ap,
                         torch.ones_like(den_ap))
    t_ap = torch.clamp(visc_H * L_ap / den_ap, 0.0, 1.0)
    r_ap = 1.0 - t_ap
    a1 = s_apex + t_ap * (s1 - s_apex)
    a2 = s_apex + t_ap * (s2 - s_apex)
    a3 = s_apex + t_ap * (s3 - s_apex)
    ea1, ea2, ea3 = r_ap * ea1_a, r_ap * ea2_a, r_ap * ea3_a

    # --- region selection
    sel_face = plastic & face_ok
    need_edge = plastic & ~face_ok
    try_right = need_edge & (s1f < s2f)
    sel_right = try_right & right_ok
    try_left = need_edge & ~try_right
    sel_left = try_left & left_ok
    sel_apex = plastic & ~sel_face & ~sel_right & ~sel_left

    bp = 0.5 * (1.0 + spsi)
    bm = -0.5 * (1.0 - spsi)
    zero = torch.zeros_like(s1)

    def pick(vf, vr, vl, va, velse):
        return torch.where(
            sel_face, vf, torch.where(
                sel_right, vr, torch.where(
                    sel_left, vl, torch.where(sel_apex, va, velse))))

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


# ---------------------------------------------------------- Hoek-Brown

def _hb_S(x, sci, mb, s_hb, a_hb):
    X = torch.clamp_min(s_hb - mb * x / sci, _X_EPS)
    return sci * torch.pow(X, a_hb)


def _hb_Sp(x, sci, mb, s_hb, a_hb):
    X = torch.clamp_min(s_hb - mb * x / sci, _X_EPS)
    return -a_hb * mb * torch.pow(X, a_hb - 1.0)


def _hb_sorted(s1, s2, s3, E, nu, sci, mb, s_hb, a_hb, psi):
    lam, G = lame(E, nu)
    spsi = torch.sin(psi)
    cb1 = lam * spsi + G * (1.0 + spsi)
    cb2 = lam * spsi
    cb3 = lam * spsi - G * (1.0 - spsi)
    # cb1 - cb3 = 2 G exactly

    def S(x):
        return _hb_S(x, sci, mb, s_hb, a_hb)

    def Sp(x):
        return _hb_Sp(x, sci, mb, s_hb, a_hb)

    f13 = (s1 - s3) - S(s1)
    plastic = f13 > 0.0
    scale = sci + s1.abs() + s3.abs()
    ftol = 1e-11 * scale
    hi0 = torch.clamp_min(f13, 0.0) / (2.0 * G) + 1e-300

    def F_face(xv, s1v, s3v):
        sig1 = s1v - xv * cb1
        return (sig1 - (s3v - xv * cb3)) - S(sig1)

    # --- main-plane return: safeguarded Newton on F(x), F' <= -2G < 0.
    # The iteration runs on DETACHED tensors (no autograd graph is
    # recorded); ONE differentiable implicit-function polish step from
    # the converged multiplier then carries the exact derivative
    # (Newton step from the solution: d x / d theta = -F_theta / F_x).
    s1_d, s3_d = s1.detach(), s3.detach()
    lo = torch.zeros_like(s1_d)
    hi = hi0.detach()
    x = 0.5 * hi
    for _ in range(_N_HB_FACE):
        Fx = F_face(x, s1_d, s3_d).detach()
        lo = torch.where(Fx > 0.0, x, lo)
        hi = torch.where(Fx <= 0.0, x, hi)
        Fp = (-(2.0 * G) + cb1 * Sp(s1_d - x * cb1)).detach()
        xn = x - Fx / Fp
        inside = (xn > lo) & (xn < hi)
        x = (torch.where(inside, xn, 0.5 * (lo + hi))).detach()
    # differentiable polish (exact implicit derivative at F = 0)
    Fp_live = -(2.0 * G) + cb1 * Sp(s1 - x * cb1)
    xf = x - F_face(x, s1, s3) / Fp_live
    s1f, s2f, s3f = s1 - xf * cb1, s2 - xf * cb2, s3 - xf * cb3
    face_ok = ((s1f >= s2f) & (s2f >= s3f)
               & (F_face(xf, s1, s3).detach().abs()
                  <= torch.clamp_min(ftol.detach(), 1e-13)))

    # --- edge returns: damped 2x2 Newton from (0, 0), DETACHED loop +
    #     one differentiable polish step (exact implicit derivative).
    xcap = (6.0 * hi0 + 1e-300).detach()

    def edge_sys(x1, x2, s1v, s2v, s3v, right):
        """Residuals + Jacobian of the two-surface edge system."""
        if right:
            sig1 = s1v - x1 * cb1 - x2 * cb2
            sig2 = s2v - x1 * cb2 - x2 * cb1
            sig3 = s3v - (x1 + x2) * cb3
            F1 = (sig1 - sig3) - S(sig1)
            F2 = (sig2 - sig3) - S(sig2)
            J11 = -2.0 * G + cb1 * Sp(sig1)
            J12 = -(cb2 - cb3) + cb2 * Sp(sig1)
            J21 = -(cb2 - cb3) + cb2 * Sp(sig2)
            J22 = -2.0 * G + cb1 * Sp(sig2)
        else:
            sig1 = s1v - (x1 + x2) * cb1
            sig2 = s2v - x1 * cb2 - x2 * cb3
            sig3 = s3v - x1 * cb3 - x2 * cb2
            F1 = (sig1 - sig3) - S(sig1)
            F2 = (sig1 - sig2) - S(sig1)
            J11 = -2.0 * G + cb1 * Sp(sig1)
            J12 = -cb1 + cb2 + cb1 * Sp(sig1)
            J21 = J12
            J22 = J11
        return sig1, sig2, sig3, F1, F2, J11, J12, J21, J22

    def newton_step(x1, x2, F1, F2, J11, J12, J21, J22):
        det = J11 * J22 - J12 * J21
        det = torch.where(det.abs() > 1e-30, det, torch.ones_like(det))
        dx1 = (J22 * F1 - J12 * F2) / det
        dx2 = (J11 * F2 - J21 * F1) / det
        return x1 - dx1, x2 - dx2

    s2_d = s2.detach()

    def edge_solve(right):
        x1 = torch.zeros_like(s1_d)
        x2 = torch.zeros_like(s1_d)
        for _ in range(_N_HB_EDGE):
            out = edge_sys(x1, x2, s1_d, s2_d, s3_d, right)
            _, _, _, F1, F2, J11, J12, J21, J22 = [t.detach() for t in out]
            x1, x2 = newton_step(x1, x2, F1, F2, J11, J12, J21, J22)
            x1 = torch.clamp(x1, -2.0 * xcap, xcap).detach()
            x2 = torch.clamp(x2, -2.0 * xcap, xcap).detach()
        # differentiable polish with live tensors
        out = edge_sys(x1, x2, s1, s2, s3, right)
        _, _, _, F1, F2, J11, J12, J21, J22 = out
        x1, x2 = newton_step(x1, x2, F1, F2, J11, J12, J21, J22)
        sig1, sig2, sig3, F1c, F2c, *_ = edge_sys(x1, x2, s1, s2, s3, right)
        dom = (s_hb - mb * sig1 / sci > 2.0 * _X_EPS).detach()
        x1_d, x2_d = x1.detach(), x2.detach()
        ok = ((F1c.detach().abs() <= ftol.detach())
              & (F2c.detach().abs() <= ftol.detach())
              & (x1_d >= 0.0) & (x2_d >= 0.0) & dom
              & ((sig2 >= sig3) if right else (sig1 >= sig2)).detach())
        return x1, x2, sig1, sig2, sig3, ok

    x1r, x2r, s1r, s2r, s3r, right_ok = edge_solve(True)
    x1l, x2l, s1l, s2l, s3l, left_ok = edge_solve(False)

    # --- apex (hydrostatic tension cutoff)
    s_apex = s_hb * sci / mb

    sel_face = plastic & face_ok
    need_edge = plastic & ~face_ok
    try_right = need_edge & (s1f < s2f)
    sel_right = try_right & right_ok
    try_left = need_edge & ~try_right
    sel_left = try_left & left_ok
    sel_apex = plastic & ~sel_face & ~sel_right & ~sel_left

    bp = 0.5 * (1.0 + spsi)
    bm = -0.5 * (1.0 - spsi)
    zero = torch.zeros_like(s1)

    def pick(vf, vr, vl, va, velse):
        return torch.where(
            sel_face, vf, torch.where(
                sel_right, vr, torch.where(
                    sel_left, vl, torch.where(sel_apex, va, velse))))

    ap = s_apex * torch.ones_like(s1)
    t1 = pick(s1f, s1r, s1l, ap, s1)
    t2 = pick(s2f, s2r, s2l, ap, s2)
    t3 = pick(s3f, s3r, s3l, ap, s3)

    ea1, ea2, ea3 = _compliance(s1 - s_apex, s2 - s_apex, s3 - s_apex, E, nu)
    e1 = pick(xf * bp, x1r * bp, (x1l + x2l) * bp, ea1, zero)
    e2 = pick(zero, x2r * bp, x2l * bm, ea2, zero)
    e3 = pick(xf * bm, (x1r + x2r) * bm, x1l * bm, ea3, zero)

    dlam = pick(xf, x1r + x2r, x1l + x2l,
                torch.sqrt(ea1 * ea1 + ea2 * ea2 + ea3 * ea3), zero)
    region = (sel_face.to(torch.int64) + 2 * sel_right.to(torch.int64)
              + 3 * sel_left.to(torch.int64) + 4 * sel_apex.to(torch.int64))
    return t1, t2, t3, e1, e2, e3, dlam, region


# ------------------------------------------ CWFS Mohr-Coulomb (tier B)

def _cwfs_c(kap, c_peak, c_res, gamma_c):
    """Piecewise-linear cohesion weakening, saturating at gamma_c."""
    t = torch.clamp(kap / gamma_c, 0.0, 1.0)
    return c_peak + (c_res - c_peak) * t


def _cwfs_phi(kap, phi_peak, phi_res, gamma_phi):
    """Piecewise-linear friction strengthening, saturating at gamma_phi."""
    t = torch.clamp(kap / gamma_phi, 0.0, 1.0)
    return phi_peak + (phi_res - phi_peak) * t


def _cwfs_sorted(s1, s2, s3, kap, E, nu, c_pk, c_rs, g_c, p_pk, p_rs,
                 g_p, psi, visc_H=0.0):
    """CWFS return map on sorted principals; extra output dkappa.

    Fully implicit: every candidate return satisfies
    f(sig_new, kap + d_kappa) = 0 with the laws at the UPDATED kappa,
    d_kappa = deps_p_max - deps_p_min of that candidate's plastic
    triple (face: x*(bp - bm); edges: coupled through max(x1, x2);
    apex: e1 - e3, independent of the apex position).

    ``visc_H = eta_vp/dt`` appends the backward-Euler Perzyna
    consistency f_j(sig_new, kappa_new) = visc_H * dlam_j to every
    multiplier equation (module docstring, CWFS-Perzyna item).  The
    default 0.0 reproduces the rate-independent softening return
    BIT-FOR-BIT (regression-guarded by tests/test_cwfs_perzyna.py).
    """
    lam, G = lame(E, nu)
    spsi = torch.sin(psi)
    cb1 = lam * spsi + G * (1.0 + spsi)
    cb2 = lam * spsi
    cb3 = lam * spsi - G * (1.0 - spsi)
    bp = 0.5 * (1.0 + spsi)
    bm = -0.5 * (1.0 - spsi)

    def strength(kn):
        """c, sin/cos(phi) and law slopes (clamp subgradient) at kn."""
        c = _cwfs_c(kn, c_pk, c_rs, g_c)
        phi = _cwfs_phi(kn, p_pk, p_rs, g_p)
        sphi, cphi = torch.sin(phi), torch.cos(phi)
        dc = (c_rs - c_pk) / g_c * ((kn >= 0.0) & (kn < g_c)).to(s1.dtype)
        dphi = ((p_rs - p_pk) / g_p
                * ((kn >= 0.0) & (kn < g_p)).to(s1.dtype))
        return c, sphi, cphi, dc, dphi

    def fval(sa, sb, c, sphi, cphi):
        return 0.5 * (sa - sb) + 0.5 * (sa + sb) * sphi - c * cphi

    c0, sphi0, cphi0, _, _ = strength(kap)
    f13 = fval(s1, s3, c0, sphi0, cphi0)
    plastic = f13 > 0.0
    scale = c_pk + s1.abs() + s3.abs()
    ftol = 1e-11 * scale

    def F_face(x, s1v, s3v, kv):
        kn = kv + (x * bp - x * bm)
        c, sphi, cphi, _, _ = strength(kn)
        return (fval(s1v - x * cb1, s3v - x * cb3, c, sphi, cphi)
                - visc_H * x)

    def Fp_face(x, s1v, s3v, kv):
        kn = kv + (x * bp - x * bm)
        sig1 = s1v - x * cb1
        sig3 = s3v - x * cb3
        c, sphi, cphi, dc, dphi = strength(kn)
        d11 = lam * sphi * spsi + G * (1.0 + sphi * spsi)
        fk = (0.5 * (sig1 + sig3) * cphi + c * sphi) * dphi - dc * cphi
        return -d11 + fk * (bp - bm) - visc_H

    # --- main-plane return: bracketed safeguarded Newton, DETACHED.
    # Softening can break monotonicity of F, so first GROW the bracket
    # from the perfect-plasticity seed until F(hi) <= 0 (F -> -inf once
    # the laws saturate), then bisection safeguards every Newton step.
    s1_d, s3_d, kap_d = s1.detach(), s3.detach(), kap.detach()
    d11_0 = (lam * sphi0 * spsi + G * (1.0 + sphi0 * spsi)).detach()
    lo = torch.zeros_like(s1_d)
    hi = (torch.clamp_min(f13, 0.0) / d11_0 + 1e-300).detach()
    for _ in range(_N_CWFS_GROW):
        grow = F_face(hi, s1_d, s3_d, kap_d).detach() > 0.0
        lo = torch.where(grow, hi, lo)
        hi = torch.where(grow, 2.0 * hi, hi)
    x = 0.5 * (lo + hi)
    for _ in range(_N_CWFS_FACE):
        Fx = F_face(x, s1_d, s3_d, kap_d).detach()
        lo = torch.where(Fx > 0.0, x, lo)
        hi = torch.where(Fx <= 0.0, x, hi)
        Fp = Fp_face(x, s1_d, s3_d, kap_d).detach()
        Fp = torch.where(Fp.abs() > 1e-30, Fp, -torch.ones_like(Fp))
        xn = x - Fx / Fp
        inside = (xn > lo) & (xn < hi)
        x = torch.where(inside, xn, 0.5 * (lo + hi)).detach()
    # ONE differentiable implicit polish: dx/dtheta = -F_theta/F_x at
    # F = 0 (detached F_x is exact there and skips second-order terms).
    Fp_pol = Fp_face(x, s1_d, s3_d, kap_d).detach()
    Fp_pol = torch.where(Fp_pol.abs() > 1e-30, Fp_pol,
                         -torch.ones_like(Fp_pol))
    xf = x - F_face(x, s1, s3, kap) / Fp_pol
    s1f, s2f, s3f = s1 - xf * cb1, s2 - xf * cb2, s3 - xf * cb3
    dk_f = xf * bp - xf * bm
    face_ok = ((s1f >= s2f) & (s2f >= s3f)
               & (F_face(xf, s1, s3, kap).detach().abs()
                  <= torch.clamp_min(ftol.detach(), 1e-13)))

    # --- edge returns: 2x2 Newton (perfect-MC linear part + rank-one
    #     softening coupling through d_kappa), DETACHED + one polish.
    xcap = (8.0 * hi + 1e-300).detach()

    def newton_step(x1, x2, F1, F2, J11, J12, J21, J22):
        det = J11 * J22 - J12 * J21
        det = torch.where(det.abs() > 1e-30, det, torch.ones_like(det))
        dx1 = (J22 * F1 - J12 * F2) / det
        dx2 = (J11 * F2 - J21 * F1) / det
        return x1 - dx1, x2 - dx2

    def edge_sys(x1, x2, s1v, s2v, s3v, kv, right):
        """Residuals + Jacobian of the softening two-surface system."""
        if right:
            sig1 = s1v - x1 * cb1 - x2 * cb2
            sig2 = s2v - x1 * cb2 - x2 * cb1
            sig3 = s3v - (x1 + x2) * cb3
            dk = bp * torch.maximum(x1, x2) - bm * (x1 + x2)
        else:
            sig1 = s1v - (x1 + x2) * cb1
            sig2 = s2v - x1 * cb2 - x2 * cb3
            sig3 = s3v - x1 * cb3 - x2 * cb2
            dk = bp * (x1 + x2) - bm * torch.maximum(x1, x2)
        kn = kv + dk
        c, sphi, cphi, dc, dphi = strength(kn)
        d11 = lam * sphi * spsi + G * (1.0 + sphi * spsi)
        m1 = (x1 >= x2).to(s1.dtype)          # subgradient at the tie
        if right:
            F1v = fval(sig1, sig3, c, sphi, cphi)
            F2v = fval(sig2, sig3, c, sphi, cphi)
            d12 = lam * sphi * spsi + 0.5 * G * (1.0 - sphi) * (1.0 - spsi)
            fk1 = ((0.5 * (sig1 + sig3) * cphi + c * sphi) * dphi
                   - dc * cphi)
            fk2 = ((0.5 * (sig2 + sig3) * cphi + c * sphi) * dphi
                   - dc * cphi)
            dk1 = bp * m1 - bm
            dk2 = bp * (1.0 - m1) - bm
        else:
            F1v = fval(sig1, sig3, c, sphi, cphi)
            F2v = fval(sig1, sig2, c, sphi, cphi)
            d12 = lam * sphi * spsi + 0.5 * G * (1.0 + sphi) * (1.0 + spsi)
            fk1 = ((0.5 * (sig1 + sig3) * cphi + c * sphi) * dphi
                   - dc * cphi)
            fk2 = ((0.5 * (sig1 + sig2) * cphi + c * sphi) * dphi
                   - dc * cphi)
            dk1 = bp - bm * m1
            dk2 = bp - bm * (1.0 - m1)
        # Perzyna consistency per active surface: f_i = visc_H * x_i
        F1v = F1v - visc_H * x1
        F2v = F2v - visc_H * x2
        J11 = -d11 + fk1 * dk1 - visc_H
        J12 = -d12 + fk1 * dk2
        J21 = -d12 + fk2 * dk1
        J22 = -d11 + fk2 * dk2 - visc_H
        return sig1, sig2, sig3, dk, F1v, F2v, J11, J12, J21, J22

    s2_d = s2.detach()

    def edge_solve(right):
        x1 = torch.zeros_like(s1_d)
        x2 = torch.zeros_like(s1_d)
        for _ in range(_N_CWFS_EDGE):
            out = edge_sys(x1, x2, s1_d, s2_d, s3_d, kap_d, right)
            _, _, _, _, F1, F2, J11, J12, J21, J22 = [
                t.detach() for t in out]
            x1, x2 = newton_step(x1, x2, F1, F2, J11, J12, J21, J22)
            x1 = torch.clamp(x1, -2.0 * xcap, xcap).detach()
            x2 = torch.clamp(x2, -2.0 * xcap, xcap).detach()
        # differentiable polish with live tensors
        out = edge_sys(x1, x2, s1, s2, s3, kap, right)
        _, _, _, dk, F1, F2, J11, J12, J21, J22 = out
        x1, x2 = newton_step(x1, x2, F1, F2, J11, J12, J21, J22)
        sig1, sig2, sig3, dk, F1c, F2c, *_ = edge_sys(
            x1, x2, s1, s2, s3, kap, right)
        x1_d, x2_d = x1.detach(), x2.detach()
        ok = ((F1c.detach().abs() <= ftol.detach())
              & (F2c.detach().abs() <= ftol.detach())
              & (x1_d >= 0.0) & (x2_d >= 0.0)
              & ((sig2 >= sig3) if right else (sig1 >= sig2)).detach())
        return x1, x2, sig1, sig2, sig3, dk, ok

    x1r, x2r, s1r, s2r, s3r, dk_r, right_ok = edge_solve(True)
    x1l, x2l, s1l, s2l, s3l, dk_l, left_ok = edge_solve(False)

    # --- apex: CLOSED FORM.  d_kappa = e1 - e3 = (1+nu)*(s1-s3)/E is
    # independent of the apex position, so kappa_new and the softening-
    # dependent apex s_i = c(k)*cos(phi(k))/sin(phi(k)) are explicit.
    # Perzyna (visc_H > 0): Koiter-Perzyna RAY treatment mirroring
    # _mc_sorted, with the laws FROZEN at the exact-apex kappa update
    # kap + dk_a (frozen-strength linearization of the ray consistency,
    # module docstring); visc_H = 0 gives t_ap = 0 == the exact apex
    # projection, bit-for-bit.
    dk_a = (1.0 + nu) * (s1 - s3) / E
    c_a, sphi_a, cphi_a, _, _ = strength(kap + dk_a)
    s_apex = c_a * cphi_a / torch.clamp_min(sphi_a, 1e-12)
    ea1_a, ea2_a, ea3_a = _compliance(s1 - s_apex, s2 - s_apex,
                                      s3 - s_apex, E, nu)
    # NaN-safe sqrt (double-where): an APEX-RESIDENT lane -- committed
    # state placed exactly at the apex by an earlier return, deps = 0
    # in a later Newton iterate (routine at K0 != 1, where softening
    # opens hydrostatic-tension lobes) -- has L_sq = 0 exactly, and
    # d(sqrt)/dx -> inf there.  A plain sqrt(0) inside the sigma_new
    # graph poisons the consistent tangent of EVERY region through the
    # unselected-branch gradients of ``pick`` (0 * NaN = NaN), which is
    # how the rate-independent K0 = 0.5 system runs regressed while all
    # L > 0 goldens stayed green.  Forward values are bit-identical:
    # L_ap = 0 on the degenerate lane (== sqrt(0)), sqrt(L_sq)
    # elsewhere; the degenerate lane's sqrt-branch gradient is exactly
    # zero instead of NaN.  Regression-pinned by
    # tests/test_cwfs_perzyna.py::test_apex_resident_lane_tangent_*.
    L_sq = ea1_a * ea1_a + ea2_a * ea2_a + ea3_a * ea3_a
    pos_L = L_sq > 0.0
    L_ap = torch.where(pos_L,
                       torch.sqrt(torch.where(pos_L, L_sq,
                                              torch.ones_like(L_sq))),
                       torch.zeros_like(L_sq))
    f13_a = fval(s1, s3, c_a, sphi_a, cphi_a)     # trial f at frozen laws
    den_ap = f13_a + visc_H * L_ap
    den_ap = torch.where(den_ap.abs() > 1e-30, den_ap,
                         torch.ones_like(den_ap))
    t_ap = torch.clamp(visc_H * L_ap / den_ap, 0.0, 1.0)
    r_ap = 1.0 - t_ap
    a1 = s_apex + t_ap * (s1 - s_apex)
    a2 = s_apex + t_ap * (s2 - s_apex)
    a3 = s_apex + t_ap * (s3 - s_apex)
    ea1, ea2, ea3 = r_ap * ea1_a, r_ap * ea2_a, r_ap * ea3_a
    dk_apex = ea1 - ea3       # max - min of the compliance triple

    sel_face = plastic & face_ok
    need_edge = plastic & ~face_ok
    try_right = need_edge & (s1f < s2f)
    sel_right = try_right & right_ok
    try_left = need_edge & ~try_right
    sel_left = try_left & left_ok
    sel_apex = plastic & ~sel_face & ~sel_right & ~sel_left

    zero = torch.zeros_like(s1)

    def pick(vf, vr, vl, va, velse):
        return torch.where(
            sel_face, vf, torch.where(
                sel_right, vr, torch.where(
                    sel_left, vl, torch.where(sel_apex, va, velse))))

    t1 = pick(s1f, s1r, s1l, a1, s1)
    t2 = pick(s2f, s2r, s2l, a2, s2)
    t3 = pick(s3f, s3r, s3l, a3, s3)

    e1 = pick(xf * bp, x1r * bp, (x1l + x2l) * bp, ea1, zero)
    e2 = pick(zero, x2r * bp, x2l * bm, ea2, zero)
    e3 = pick(xf * bm, (x1r + x2r) * bm, x1l * bm, ea3, zero)

    dlam = pick(xf, x1r + x2r, x1l + x2l, r_ap * L_ap, zero)
    dkappa = pick(dk_f, dk_r, dk_l, dk_apex, zero)
    region = (sel_face.to(torch.int64) + 2 * sel_right.to(torch.int64)
              + 3 * sel_left.to(torch.int64) + 4 * sel_apex.to(torch.int64))
    return t1, t2, t3, e1, e2, e3, dlam, region, dkappa


# ------------------------------------------------------------ front end

def _integrate_perfect(deps3, sig4, E, nu, mat, kind):
    """Perfect-kind backward-Euler step (batch-1 body, bit-for-bit).

    Returns (sig_new4, deps_p4, dlam, region); ``kind`` in
    {"elastic", "mc", "hb"}.
    """
    E = torch.as_tensor(E, dtype=DTYPE)
    nu = torch.as_tensor(nu, dtype=DTYPE)
    mat = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in mat.items()}
    sig_tr = elastic_trial(deps3, sig4, E, nu)
    if kind == "elastic":
        z = torch.zeros_like(sig_tr[..., 0])
        return (sig_tr, torch.zeros_like(sig_tr), z,
                torch.zeros_like(z, dtype=torch.int64))
    sA, sB, sZ, c2, s2 = _principal4(sig_tr)
    s1, s2s, s3, m0, m2 = _sort3(sA, sB, sZ)
    if kind == "mc":
        t1, t2, t3, e1, e2, e3, dlam, region = _mc_sorted(
            s1, s2s, s3, E, nu, mat["c"], mat["phi"], mat["psi"])
    elif kind == "hb":
        t1, t2, t3, e1, e2, e3, dlam, region = _hb_sorted(
            s1, s2s, s3, E, nu, mat["sigma_ci"], mat["m_b"],
            mat["s"], mat["a_hb"], mat["psi_hb"])
    else:
        raise ValueError(f"unknown material kind {kind!r}")
    vA, vB, vZ = _unsort3(t1, t2, t3, m0, m2)
    eA, eB, eZ = _unsort3(e1, e2, e3, m0, m2)
    sig_pl = _recompose_stress(vA, vB, vZ, c2, s2)
    dep_pl = _recompose_strain(eA, eB, eZ, c2, s2)
    plast4 = (region > 0).unsqueeze(-1)
    sig_new = torch.where(plast4, sig_pl, sig_tr)
    deps_p = torch.where(plast4, dep_pl, torch.zeros_like(dep_pl))
    return sig_new, deps_p, dlam, region


def integrate_full(deps3, sig4, kappa, E, nu, mat, kind):
    """One backward-Euler step from the committed state (sig4, kappa).

    Canonical state-carrying entry (all tensors broadcast on leading
    dimensions; ``mat`` is a dict of scalar tensors/floats, see
    params.*.theta; ``kappa`` is the committed internal variable,
    ``dkappa`` its backward-Euler increment)::

        integrate_full(deps3, sig4, kappa, E, nu, mat, kind)
        -> (sig_new4, deps_p4, dlam, region, dkappa)

    Perfect kinds ("elastic"/"mc"/"hb") do not consume the committed
    ``kappa`` (their return map is state-free beyond sig) and keep
    their batch-1 semantics dkappa = dlam bit-for-bit; kind "cwfs"
    evolves kappa = equivalent plastic shear strain, dkappa =
    deps_p_max - deps_p_min of the returned principal plastic-strain
    increments (face returns: = dlam).

    Kind "cwfs" additionally honors an OPTIONAL mat key ``"visc_H"``
    (Perzyna regularization, module docstring CWFS-Perzyna item;
    params.CWFSMohrCoulomb.visc_H threaded by the FE system as a
    theta-independent constant).  An absent key or the value 0.0 is
    the exact rate-independent softening return, bit-for-bit.
    """
    if kind != "cwfs":
        out = _integrate_perfect(deps3, sig4, E, nu, mat, kind)
        return out + (out[2],)          # perfect kinds: dkappa = dlam
    E = torch.as_tensor(E, dtype=DTYPE)
    nu = torch.as_tensor(nu, dtype=DTYPE)
    kap = torch.as_tensor(kappa, dtype=DTYPE)
    mat = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in mat.items()}
    visc_H = mat.get("visc_H")
    if visc_H is None:
        visc_H = torch.zeros((), dtype=DTYPE)
    sig_tr = elastic_trial(deps3, sig4, E, nu)
    sA, sB, sZ, c2, s2 = _principal4(sig_tr)
    s1, s2s, s3, m0, m2 = _sort3(sA, sB, sZ)
    t1, t2, t3, e1, e2, e3, dlam, region, dkappa = _cwfs_sorted(
        s1, s2s, s3, kap, E, nu, mat["c_peak"], mat["c_res"],
        mat["gamma_c"], mat["phi_peak"], mat["phi_res"],
        mat["gamma_phi"], mat["psi"], visc_H=visc_H)
    vA, vB, vZ = _unsort3(t1, t2, t3, m0, m2)
    eA, eB, eZ = _unsort3(e1, e2, e3, m0, m2)
    sig_pl = _recompose_stress(vA, vB, vZ, c2, s2)
    dep_pl = _recompose_strain(eA, eB, eZ, c2, s2)
    plast4 = (region > 0).unsqueeze(-1)
    sig_new = torch.where(plast4, sig_pl, sig_tr)
    deps_p = torch.where(plast4, dep_pl, torch.zeros_like(dep_pl))
    return sig_new, deps_p, dlam, region, dkappa


def consistent_tangent(deps3_qp, sig4_qp, kappa_qp, E, nu, mat, kind):
    """Exact algorithmic tangent d(sig_inplane)/d(deps3), (nq, 3, 3).

    Differentiation is w.r.t. deps at FIXED committed (sig, kappa);
    ``kappa_qp`` carries the leading QP-batch shape.  Perfect kinds do
    not consume the committed kappa and skip its vmap axis.
    """
    idx = torch.tensor(IN_PLANE_IDX, dtype=torch.int64)

    if kind == "cwfs":
        def f_cwfs(d, s, k):
            return integrate_full(d, s, k, E, nu, mat, kind)[0][idx]

        return torch.func.vmap(torch.func.jacrev(f_cwfs, argnums=0))(
            deps3_qp, sig4_qp, torch.as_tensor(kappa_qp, dtype=DTYPE))

    zero = torch.zeros((), dtype=DTYPE)

    def f(d, s):
        return integrate_full(d, s, zero, E, nu, mat, kind)[0][idx]

    return torch.func.vmap(torch.func.jacrev(f, argnums=0))(
        deps3_qp, sig4_qp)


def yield_values(sig4, mat, kind, kappa=None):
    """Governing yield value(s) f_13 on the sorted principals (tests).

    Kind "cwfs" evaluates the laws at the committed ``kappa``.
    """
    mat = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in mat.items()}
    sA, sB, sZ, _, _ = _principal4(torch.as_tensor(sig4, dtype=DTYPE))
    s1, _, s3, _, _ = _sort3(sA, sB, sZ)
    if kind == "mc":
        sphi, cphi = torch.sin(mat["phi"]), torch.cos(mat["phi"])
        return (0.5 * (s1 - s3) + 0.5 * (s1 + s3) * sphi
                - mat["c"] * cphi)
    if kind == "hb":
        return (s1 - s3) - _hb_S(s1, mat["sigma_ci"], mat["m_b"],
                                 mat["s"], mat["a_hb"])
    if kind == "cwfs":
        if kappa is None:
            raise ValueError("yield_values kind 'cwfs' requires kappa")
        kap = torch.as_tensor(kappa, dtype=DTYPE)
        c = _cwfs_c(kap, mat["c_peak"], mat["c_res"], mat["gamma_c"])
        phi = _cwfs_phi(kap, mat["phi_peak"], mat["phi_res"],
                        mat["gamma_phi"])
        sphi, cphi = torch.sin(phi), torch.cos(phi)
        return 0.5 * (s1 - s3) + 0.5 * (s1 + s3) * sphi - c * cphi
    raise ValueError(kind)
