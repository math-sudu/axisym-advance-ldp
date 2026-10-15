"""Recover physical installation events from accepted advance checkpoints.

The fixed cohort is z/a in [3, 6]. An event joins the first inactive/active
mask transition to the preceding displacement and accepted face position.
The original terminal-only parameter grid supplies nominal inputs separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/lining_activation/installation_events.json"
A = 2.0
COHORT = (3.0, 6.0)


def read_state(path):
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key] for key in (
            "liner_active", "liner_e_z0", "liner_e_z1", "liner_sig",
            "liner_e_rho0", "liner_e_rho1",
            "displacement_m", "wall_nodes", "z_lines_m")}
        state["face_over_a"] = json.loads(str(data["meta"]))["x_f"] / A
    return state


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def round_count(final_face, round_length):
    if not np.isfinite([final_face, round_length]).all() or min(final_face, round_length) <= 0:
        raise ValueError("Advance length and round length must be finite and positive")
    count = round(final_face / round_length)
    if not np.isclose(count * round_length, final_face, rtol=0, atol=1e-10):
        raise ValueError("The target face must end a complete construction round")
    return count


def extract_run(source, case, nominal_distance, mesh_level, round_length,
                initial_state=None, final_face=12.0):
    initial_state = initial_state or source / "step-000.npz"
    previous = read_state(initial_state)
    np.testing.assert_allclose(previous["face_over_a"], 0.0)
    lo, hi = previous["liner_e_z0"] / A, previous["liner_e_z1"] / A
    in_cohort = (lo >= COHORT[0] - 1e-10) & (hi <= COHORT[1] + 1e-10)
    if previous["liner_active"][in_cohort].any():
        raise ValueError("The positive-z cohort is already lined at spin-up")
    initial_front = float(hi[previous["liner_active"]].max())
    events = []
    final_step = round_count(final_face, round_length)
    for step in range(1, final_step + 1):
        current = read_state(source / f"step-{step:03d}.npz")
        face, reference_face = current["face_over_a"], previous["face_over_a"]
        np.testing.assert_allclose(face - reference_face, round_length, atol=1e-12)
        np.testing.assert_array_equal(current["liner_e_z0"] / A, lo)
        np.testing.assert_array_equal(current["liner_e_z1"] / A, hi)
        now, before = current["liner_active"], previous["liner_active"]
        np.testing.assert_array_equal(now, hi <= face - nominal_distance + 1e-9 / A)
        if np.any(before & ~now):
            raise ValueError("Lining deactivated during excavation")
        new = now & ~before & in_cohort
        np.testing.assert_allclose(previous["liner_sig"][new], 0.0, atol=1e-14)
        wall_mm = -previous["displacement_m"][2 * previous["wall_nodes"]] * 1000
        for z0 in np.unique(lo[new]):
            selected = new & (lo == z0)
            z1 = float(np.unique(hi[selected]).item())
            thickness = current["liner_e_rho1"] - current["liner_e_rho0"]
            np.testing.assert_allclose(thickness[selected].sum(), 0.2, atol=1e-12)
            centre = (float(z0) + z1) / 2
            d_pre = reference_face - z1
            if not nominal_distance - round_length - 1e-10 <= d_pre < nominal_distance - 1e-10:
                raise ValueError("Observed installation distance contradicts the activation rule")
            reference_path = initial_state if step == 1 else source / f"step-{step-1:03d}.npz"
            events.append({
                "band_over_a": [float(z0), z1],
                "center_z_over_a": centre,
                "reference_step": step - 1,
                "reference_state": relative(reference_path),
                "installation_face_over_a": reference_face,
                "installation_edge_distance_over_a": d_pre,
                "installation_center_distance_over_a": reference_face - centre,
                "installation_convergence_mm": float(np.interp(
                    centre * A, previous["z_lines_m"], wall_mm)),
                "first_loaded_step": step,
                "first_loaded_face_over_a": face,
                "first_loaded_edge_distance_over_a": face - z1,
            })
        previous = current
    np.testing.assert_allclose(previous["face_over_a"], final_face)
    events.sort(key=lambda event: event["band_over_a"][0])
    edges = np.asarray([event["band_over_a"] for event in events])
    np.testing.assert_allclose(edges[[0, -1], [0, 1]], COHORT)
    np.testing.assert_allclose(edges[:-1, 1], edges[1:, 0])
    return {
        "case": case,
        "source_directory": relative(source),
        "mesh_level": mesh_level,
        "nominal_distance_over_a": nominal_distance,
        "round_length_over_a": round_length,
        "final_face_over_a": final_face,
        "installation_target_over_a": nominal_distance - round_length,
        "initial_state": relative(initial_state),
        "initial_lining_front_over_a": initial_front,
        "events": events,
    }
