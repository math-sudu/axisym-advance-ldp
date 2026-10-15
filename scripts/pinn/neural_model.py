"""Standalone geometry and common initial states for the neural histories."""
from __future__ import annotations

import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/review")]

import numpy as np

from runtime import corrected_kernel, geometry, jsonable, save_npz, state_arrays
from solver import build, configure, restore
from extract_lining_installation_events import round_count
from wave2_src import scenario as scen


def model_and_source(mesh):
    """Use the research geometry and relative paths, including its initial state."""
    if mesh in ("coarse", "production"):
        level = 0 if mesh == "coarse" else 1
        model = build("B", level=level)
        directory = (ROOT / "results/analysis/advancing_support/SR-01_spatial_resolution.cache"
                     / "newton-search-repair"
                     / f"supported_B_d1.00_production_L{level}_cap0p0625")
    elif mesh in ("design", "interior"):
        events = (np.round(np.arange(round_count(14.4, 0.2)) * 0.2 - 0.72, 12)
                  if mesh == "interior" else ())
        model = build("B", axial_track_spacing_over_a=0.05, final_face_over_a=14.4,
                      axial_event_positions_over_a=events)
        directory = ROOT / "results/analysis/lining_activation/construction_design"
        if mesh == "interior":
            directory /= "interior_position"
        directory /= "B_initial"
    else:
        raise ValueError(f"Unknown mesh: {mesh}")
    return model, directory / "step-000.npz"


def initialize(mesh):
    """Generate the conventional common equilibrium once; subsequent solves are neural."""
    model, source = model_and_source(mesh)
    if source.exists():
        restore(model, source)
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        geo = geometry(model, 0)
        record = model.step(0)
        arrays = state_arrays(model)
        arrays["c_reduced"] = model.solver.c.copy()
        meta = {"k": 0, "x_f": record["x_f"], "wall_s": record["wall_s"],
                "info": jsonable(record["info"]), "geometry": geo}
        save_npz(source, meta, arrays)
    return model, source


def initial_model(round_length=0.4, gap=0.6, mesh="coarse"):
    model, source = model_and_source(mesh)
    if not source.exists():
        raise FileNotFoundError(f"Run python -B scripts/pinn/initialize.py --mesh {mesh} first")
    restore(model, source)
    configure(model, gap, round_length, 12.0 if mesh in ("coarse", "production") else 14.4)
    check_installation(model, gap)
    return model, source


# check_installation is appended verbatim from the research construction module.


def check_installation(model, gap):
    block = model.fe.blocks["liner"]
    before = block["active"].copy()
    eligible = model.mesh.e_z1[block["elems"]] <= -gap * scen.A + 1e-9
    if np.any(before & ~eligible):
        raise ValueError("Initial lining extends beyond the requested installation front")
    newly = eligible & ~before
    np.testing.assert_array_equal(block["sig"][newly], 0.0)
    if model.fe.blocks["rock"]["active"][block["elems"][newly]].any():
        raise ValueError("New lining overlaps unexcavated rock")
    u, stress = model.fe.u_n.copy(), block["sig"].copy()
    model._liner_activation(model.dstep)
    np.testing.assert_array_equal(block["active"], eligible)
    np.testing.assert_array_equal(model.fe.u_n, u)
    np.testing.assert_array_equal(block["sig"], stress)
    block["active"] = before
