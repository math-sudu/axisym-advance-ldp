"""Build the refined construction model and restore accepted equilibria."""
from __future__ import annotations

import numpy as np

from wave2_src import axifem, scenario as scen
from runtime import (RefinementCoupled, load_npz, mc_sorted_roundoff, newton,
                     subdivide)


class AdvanceSolver(RefinementCoupled):
    newton = newton

    def solve_step(self, *args, **kwargs):
        self.spatial_trials = []
        return super().solve_step(*args, **kwargs)


def build(case, level=1):
    axifem._mc_sorted = mc_sorted_roundoff
    radial, axial = scen.radial_lines, scen.axial_lines
    scen.radial_lines = lambda *a, **k: subdivide(radial(*a, **k), level)
    scen.axial_lines = lambda *a, **k: subdivide(axial(*a, **k), level)
    try:
        model = scen.AdvanceScenario(scen.ARMS[case], liner_d=scen.A,
                                     n_ord=8, h_scale=1.0, tol_rel=1e-6)
    finally:
        scen.radial_lines, scen.axial_lines = radial, axial
    old = model.solver
    model.solver = AdvanceSolver(model.fe, model.ext, eps_ridge=old.eps_ridge)
    model.solver.set_basis(old.basis, keep_c=False)
    model.solver.d_lam_cap = 1 / 16
    return model


def configure(model, gap, round_length):
    model.dstep = round_length * scen.A
    model.n_steps = round(12 / round_length)
    model.liner_d = round(gap + round_length, 12) * scen.A
    for face in np.arange(1, model.n_steps + 1) * model.dstep:
        for position in (face, face - model.liner_d):
            if not np.isclose(model.z_lines, position, rtol=0, atol=1e-9).any():
                raise ValueError("Construction endpoint is not a mesh boundary")


def restore(model, path):
    meta, arrays = load_npz(path)
    for key, value in (("rho_lines_m", model.rho_lines), ("z_lines_m", model.z_lines),
                       ("wall_nodes", model.wall_nodes), ("F_back_2d", model.F_back_2d),
                       ("F_back_s0", model.F_back_s0)):
        np.testing.assert_array_equal(arrays[key], value, err_msg=key)
    for name, block in model.fe.blocks.items():
        np.testing.assert_array_equal(arrays[name + "_elems"], block["elems"])
        for field in ("sig", "active", "plast"):
            block[field] = arrays[name + "_" + field].copy()
    model.fe.u_n = arrays["displacement_m"].copy()
    model.fe.f_release = np.zeros(model.fe.ndof)
    model.fe.f_ext_fix = arrays["F_back_2d"].copy()
    model.solver.c = arrays["c_reduced"].copy()
    np.testing.assert_allclose(model.solver.c_phys, arrays["c_phys"], rtol=1e-11, atol=1e-12)
    model.history = [{"k": meta["k"], "x_f": meta["x_f"], "info": meta["info"],
                      "wall_s": meta["wall_s"], "wall": arrays["u_wall"]}]
    return meta
