"""Read neural construction histories at fixed material sections and batches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/review"))
from extract_lining_band_histories import finite_list, wall_band_means
from extract_lining_section_histories import section_quadrature, integrate_thrust
from extract_paired_section_strains import AxiMesh, elastic_stress

A = 2.0


def read_state(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "meta"}


def analyse(source, cohort=(3.0, 6.0), face=None):
    source = source.resolve()
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if run.get("method") == "installation_history_constrained_neural_field_v1":
        from history_response import analyse as analyse_history
        return analyse_history(source, cohort, face)
    controls = run["controls"]
    length = controls["round_length_over_a"]
    last = run["last_completed_round"]
    if face is not None:
        requested = round(face / length)
        if not np.isclose(requested * length, face) or requested > last:
            raise ValueError("The evaluation face must be a completed neural round")
        last = requested
    paths = [ROOT / controls["initial_state"],
             *[source / f"round-{step:03d}.npz" for step in range(1, last + 1)]]
    initial = read_state(paths[0])
    lo, hi = initial["liner_e_z0"] / A, initial["liner_e_z1"] / A
    selected = (lo >= cohort[0] - 1e-10) & (hi <= cohort[1] + 1e-10)
    if not selected.any() or initial["liner_active"][selected].any():
        raise ValueError("The cohort must contain lining installed during the neural history")
    native = np.unique(np.column_stack([lo[selected], hi[selected]]), axis=0)
    np.testing.assert_allclose(native[[0, -1], [0, 1]], cohort, atol=1e-10)
    np.testing.assert_allclose(native[:-1, 1], native[1:, 0], atol=1e-10)
    quadrature = section_quadrature(initial, native)
    z = np.array([row[0] for row in quadrature])
    axial_weights = np.repeat(np.diff(native).ravel() / 2, 2)
    bands = np.column_stack([np.arange(cohort[0], cohort[1] - 1e-10, 0.2),
                             np.arange(cohort[0] + 0.2, cohort[1] + 1e-10, 0.2)])
    if not len(bands) or not np.isclose(bands[-1, 1], cohort[1]):
        raise ValueError("The cohort length must be a multiple of the 0.2a readout width")
    elements = np.array([row[1] for row in quadrature])
    gauss = np.array([row[2] for row in quadrature])
    radial_weights = np.array([row[3] for row in quadrature])
    mesh = AxiMesh(initial["rho_lines_m"], initial["z_lines_m"]).build_B()
    actual_elements = initial["liner_elems"][elements]
    strain_operator, dofs = mesh.B_all[actual_elements, gauss], mesh.dof_all[actual_elements]
    reference_u = np.zeros(dofs.shape)
    reference_steps = np.full(len(z), -1)
    sections = [{"z_over_a": float(position), "axial_weight_over_a": float(weight)}
                for position, weight in zip(z, axial_weights)]
    previous, rows = initial, []
    max_stress_difference = 0.0
    final_strain = None
    for step, path in enumerate(paths):
        state = initial if step == 0 else read_state(path)
        newly = state["liner_active"] & ~previous["liner_active"]
        if np.any(previous["liner_active"] & ~state["liner_active"]):
            raise ValueError("Lining deactivated during the neural history")
        if newly.any():
            batch = [float(lo[newly].min()), float(hi[newly].max())]
            for i, (_, indices, _, _) in enumerate(quadrature):
                if newly[indices].all():
                    np.testing.assert_array_equal(previous["liner_sig"][indices], 0.0)
                    reference_steps[i] = step - 1
                    reference_u[i] = previous["displacement_m"][dofs[i]]
                    sections[i].update(
                        reference_step=step - 1,
                        reference_state=paths[step - 1].relative_to(ROOT).as_posix(),
                        first_loaded_step=step,
                        installation_face_over_a=(step - 1) * length,
                        installation_batch_over_a=batch,
                        installation_convergence_mm=rows[-1]["wall_convergence_mm"][i])
        thrust = integrate_thrust(state["liner_sig"][:, :, 2], quadrature)
        installed = state["liner_active"][elements].all(axis=1)
        wall = -1000 * state["displacement_m"][2 * state["wall_nodes"]]
        wall_z = state["z_lines_m"] / A
        values = thrust.copy()
        values[~installed] = np.nan
        rows.append({
            "step": step, "face_over_a": step * length,
            "source": path.relative_to(ROOT).as_posix(),
            "wall_convergence_mm": np.interp(z, wall_z, wall).tolist(),
            "band_convergence_mm": wall_band_means(wall_z, wall, bands).tolist(),
            "hoop_thrust_kN_per_m": finite_list(values),
            "mean_hoop_thrust_kN_per_m": float(np.average(thrust, weights=axial_weights))
            if installed.all() else None,
        })
        if installed.any():
            final_strain = np.einsum("sgij,sgj->sgi", strain_operator,
                                     state["displacement_m"][dofs] - reference_u)
            rebuilt = elastic_stress(final_strain)
            stored = state["liner_sig"][elements, gauss]
            np.testing.assert_allclose(rebuilt[installed], stored[installed], rtol=1e-10, atol=1e-9)
            max_stress_difference = max(max_stress_difference,
                                       float(np.abs(rebuilt[installed] - stored[installed]).max()))
        previous = state
    for i, reference in enumerate(reference_steps):
        if reference >= 0:
            rows[reference]["hoop_thrust_kN_per_m"][i] = 0.0
    history = np.array([row["hoop_thrust_kN_per_m"] for row in rows], dtype=float)
    for row, values in zip(rows, history):
        if np.isfinite(values).all():
            row["mean_hoop_thrust_kN_per_m"] = float(np.average(values, weights=axial_weights))
    convergence = np.array([row["band_convergence_mm"] for row in rows])
    groups = {}
    for i, section in enumerate(sections):
        if "installation_batch_over_a" in section:
            groups.setdefault(tuple(section["installation_batch_over_a"]), []).append(i)
    batches = []
    for (left, right), ids in groups.items():
        if left < cohort[0] - 1e-10 or right > cohort[1] + 1e-10:
            continue
        weights = axial_weights[ids]
        np.testing.assert_allclose(weights.sum(), right - left, atol=1e-10)
        values = history[-1, ids]
        hoop_strain = (radial_weights[ids] * final_strain[ids, :, 2]).sum(axis=1) / 0.2
        batches.append({
            "batch_over_a": [left, right], "section_indices": ids,
            "final_mean_thrust_kN_per_m": float(np.average(values, weights=weights)),
            "final_peak_thrust_kN_per_m": float(values.max()),
            "final_thrust_range_kN_per_m": float(np.ptp(values)),
            "post_installation_hoop_strain_range": float(np.ptp(hoop_strain)),
        })
    peak = None
    if np.isfinite(history).any():
        t, i = np.unravel_index(np.nanargmax(history), history.shape)
        peak = {"thrust_kN_per_m": float(history[t, i]), "z_over_a": float(z[i]),
                "face_over_a": rows[t]["face_over_a"], "section_index": int(i),
                "source": rows[t]["source"]}
    t, i = np.unravel_index(np.argmax(convergence), convergence.shape)
    return {
        "source_directory": source.relative_to(ROOT).as_posix(), "controls": controls,
        "cohort_over_a": list(cohort), "bands_over_a": bands.tolist(),
        "completed_rounds": last, "evaluation_face_over_a": last * length,
        "completed_requested_history": run["completed"],
        "summary": {
            "convergence_envelope_mm": float(convergence[t, i]),
            "convergence_control_face_over_a": rows[t]["face_over_a"],
            "convergence_control_band_over_a": bands[i].tolist(),
            "demand_envelope": peak,
            "final_mean_thrust_kN_per_m": rows[-1]["mean_hoop_thrust_kN_per_m"],
            "maximum_strain_stress_identity_difference_MPa": max_stress_difference,
        },
        "sections": sections, "history": rows, "complete_batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cohort", type=float, nargs=2, default=(3.0, 6.0))
    parser.add_argument("--face", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse(args.source, args.cohort, args.face)
    output = args.output or args.source / "construction_response.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
