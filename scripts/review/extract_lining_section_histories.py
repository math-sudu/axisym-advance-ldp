"""Recover thickness-integrated thrust at fixed axial Gauss sections.

The refined construction schemes reuse their saved round equilibria.
Each section starts with zero thrust at its actual installation reference;
the following checkpoint contains its first loaded state. Axial quadrature
must reproduce the existing native-element and fixed-band resultants.
"""
from __future__ import annotations

import json

import numpy as np

from compare_construction_schemes import SCHEDULES, SOURCE, select_runs
from extract_lining_band_histories import finite_list, overlap_weights
from extract_lining_installation_events import A, ROOT

EVENTS = SOURCE.with_name("installation_events.json")
OUT = SOURCE.with_name("section_histories.json")
GEOMETRY = ("liner_e_z0", "liner_e_z1", "liner_e_rho0", "liner_e_rho1", "liner_gp_z")


def read_state(path):
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key] for key in (*GEOMETRY, "liner_sig", "liner_active",
                                          "displacement_m", "wall_nodes", "z_lines_m")}
        state["face_over_a"] = json.loads(str(data["meta"]))["x_f"] / A
    return state


def section_quadrature(state, native_bands):
    """Return (z/a, element indices, GP indices, radial weights) per section."""
    quadrature = []
    for lo, hi in native_bands:
        elements = np.flatnonzero(
            np.isclose(state["liner_e_z0"] / A, lo, rtol=0, atol=1e-10)
            & np.isclose(state["liner_e_z1"] / A, hi, rtol=0, atol=1e-10))
        elements = elements[np.argsort(state["liner_e_rho0"][elements])]
        r0, r1 = (state[key][elements] for key in ("liner_e_rho0", "liner_e_rho1"))
        np.testing.assert_allclose(r1[:-1], r0[1:], rtol=0, atol=1e-10)
        np.testing.assert_allclose([r0[0], r1[-1]], [1.8, A], rtol=0, atol=1e-10)
        gp_z = state["liner_gp_z"][elements] / A
        for z in (lo + (hi - lo) * (1 - 1 / np.sqrt(3)) / 2,
                  lo + (hi - lo) * (1 + 1 / np.sqrt(3)) / 2):
            same_z = np.isclose(gp_z, z, rtol=0, atol=1e-10)
            np.testing.assert_array_equal(same_z.sum(axis=1), 2)
            radial, gp = np.nonzero(same_z)
            quadrature.append((float(z), elements[radial], gp, (r1 - r0)[radial] / 2))
    return quadrature


def integrate_thrust(hoop_stress, quadrature):
    """Integrate -sigma_tt dr in MPa m, converted to kN/m; no radius weight."""
    return np.array([-1000 * (hoop_stress[elements, gp] @ weights)
                     for _, elements, gp, weights in quadrature])


def peak_record(thrust, faces, sections):
    time, section = np.unravel_index(np.nanargmax(thrust), thrust.shape)
    return {"hoop_thrust_kN_per_m": float(thrust[time, section]),
            "step": int(time), "face_over_a": float(faces[time]),
            "section_index": int(section), "z_over_a": sections[section]["z_over_a"]}


def extract_run(run, band_run, bands):
    events = sorted(run["events"], key=lambda event: event["band_over_a"][0])
    native_bands = np.array([event["band_over_a"] for event in events])
    overlap = overlap_weights(native_bands, bands)
    initial = read_state(ROOT / run["initial_state"])
    quadrature = section_quadrature(initial, native_bands)
    sections = []
    for i, (z, elements, gp, weights) in enumerate(quadrature):
        event = events[i // 2]
        sections.append({
            "z_over_a": z, "native_band_over_a": event["band_over_a"],
            "native_event_index": i // 2,
            "axial_weight_over_a": float(np.diff(native_bands[i // 2])[0] / 2),
            "element_indices": elements.tolist(), "gauss_indices": gp.tolist(),
            "radial_weights_m": weights.tolist(),
            **{key: event[key] for key in ("reference_step", "reference_state",
                                         "installation_face_over_a", "first_loaded_step")},
            "installation_distance_over_a": event["installation_face_over_a"] - z,
        })
    reference_steps = np.array([section["reference_step"] for section in sections])
    first_loaded_steps = np.array([section["first_loaded_step"] for section in sections])
    np.testing.assert_array_equal(first_loaded_steps, reference_steps + 1)
    native_reference_steps = np.array([event["reference_step"] for event in events])
    band_installation_steps = np.array([native_reference_steps[w > 0].max() for w in overlap])
    axial_weights = np.array([section["axial_weight_over_a"] for section in sections])
    rows, values, faces = [], [], []
    previous_active = initial["liner_active"]
    for step, band_row in enumerate(band_run["history"]):
        path = ROOT / band_row["source"]
        state = initial if step == 0 else read_state(path)
        face = state["face_over_a"]
        np.testing.assert_allclose(face, band_row["face_over_a"], rtol=0, atol=1e-10)
        for key in GEOMETRY:
            np.testing.assert_array_equal(state[key], initial[key])
        thrust = integrate_thrust(state["liner_sig"][:, :, 2], quadrature)
        if not np.isfinite(thrust).all():
            raise ValueError(f"Nonfinite section thrust in {path}")
        wall_mm = -state["displacement_m"][2 * state["wall_nodes"]] * 1000
        convergence = np.interp([section["z_over_a"] * A for section in sections],
                                state["z_lines_m"], wall_mm)
        new = state["liner_active"] & ~previous_active
        for i, (_, elements, _, _) in enumerate(quadrature):
            np.testing.assert_array_equal(state["liner_active"][elements],
                                          step >= first_loaded_steps[i])
            if step == reference_steps[i]:
                sections[i]["installation_convergence_mm"] = float(convergence[i])
            if step == first_loaded_steps[i]:
                sections[i]["installation_batch_over_a"] = [
                    float(state["liner_e_z0"][new].min() / A),
                    float(state["liner_e_z1"][new].max() / A)]
        np.testing.assert_allclose(thrust[step < first_loaded_steps], 0, atol=1e-11)
        # Independent native-element integration checks the radial quadrature;
        # the saved fixed-band history also checks activation and axial weights.
        native = []
        for lo, hi in native_bands:
            selected = (np.isclose(state["liner_e_z0"] / A, lo, rtol=0, atol=1e-10)
                        & np.isclose(state["liner_e_z1"] / A, hi, rtol=0, atol=1e-10))
            thickness = state["liner_e_rho1"][selected] - state["liner_e_rho0"][selected]
            native.append(-1000 * (state["liner_sig"][selected, :, 2].mean(axis=1) @ thickness))
        np.testing.assert_allclose(thrust.reshape(-1, 2).mean(axis=1), native,
                                   rtol=1e-12, atol=1e-10)
        reconstructed = overlap @ thrust.reshape(-1, 2).mean(axis=1)
        saved = np.asarray(band_row["hoop_thrust_kN_per_m"], dtype=float)
        complete = step >= band_installation_steps
        np.testing.assert_array_equal(np.isfinite(saved), complete)
        np.testing.assert_allclose(reconstructed[complete], saved[complete],
                                   rtol=1e-12, atol=1e-10)
        thrust[step < reference_steps] = np.nan
        mean = float(np.average(thrust, weights=axial_weights)) if np.isfinite(thrust).all() else None
        if mean is not None:
            np.testing.assert_allclose(mean / (A * 1000), band_row["cohort_mean_pressure_MPa"],
                                       rtol=1e-12, atol=1e-12)
        rows.append({"step": step, "face_over_a": face, "source": band_row["source"],
                     "hoop_thrust_kN_per_m": finite_list(thrust),
                     "wall_convergence_mm": convergence.tolist(),
                     "cohort_mean_hoop_thrust_kN_per_m": mean,
                     "cohort_peak_to_mean": float(thrust.max() / mean) if mean else None})
        values.append(thrust)
        faces.append(face)
        previous_active = state["liner_active"]
    values = np.asarray(values)
    envelope = peak_record(values, faces, sections)
    envelope["source"] = rows[envelope["step"]]["source"]
    for i, section in enumerate(sections):
        peak_step = int(np.nanargmax(values[:, i]))
        section["peak_thrust_kN_per_m"] = float(values[peak_step, i])
        section["peak_face_over_a"] = faces[peak_step]
    return {
        **{key: run[key] for key in ("case", "mesh_level", "installation_target_over_a",
                                   "round_length_over_a", "source_directory", "initial_state")},
        "construction_rounds": len(rows) - 1, "sections": sections, "history": rows,
        "summary": {"demand_envelope": envelope,
                    "final_maximum_hoop_thrust_kN_per_m": float(values[-1].max()),
                    "final_mean_hoop_thrust_kN_per_m": rows[-1]["cohort_mean_hoop_thrust_kN_per_m"],
                    "final_peak_to_mean": rows[-1]["cohort_peak_to_mean"]},
    }


def build(events, band_histories):
    runs = []
    for case in SCHEDULES:
        for event_run, band_run in zip(select_runs(events, case), select_runs(band_histories, case), strict=True):
            for key in ("source_directory", "initial_state"):
                if event_run[key] != band_run[key]:
                    raise ValueError(f"Installation and band histories disagree on {key}")
            result = extract_run(event_run, band_run, np.array(band_histories["bands_over_a"]))
            runs.append(result)
            print(json.dumps({"case": case, "target": result["installation_target_over_a"],
                              "round": result["round_length_over_a"], **result["summary"]}), flush=True)
    return {
        **{key: band_histories[key] for key in ("radius_m", "cohort_over_a", "final_face_over_a",
                                               "evaluation_basis")},
        "installation_events": EVENTS.relative_to(ROOT).as_posix(),
        "band_history_source": SOURCE.relative_to(ROOT).as_posix(),
        "definitions": {
            "section_thrust": "Thickness integral of -sigma_tt at each stored axial Gauss position; "
                              "each radial Gauss point has weight h_r/2, without radius weighting.",
            "demand_envelope": "Maximum over fixed axial Gauss sections in z/a=[3,6] and saved "
                               "round equilibria from each section's stress-free installation to x_f/a=12.",
            "installation": "Zero thrust at the preceding equilibrium used as the activation reference. "
                            "Earlier thrust entries are null. Batch bounds come from the actual active-mask transition.",
            "cohort_mean": "Axial Gauss weights h_z/2, normalized by the cohort length; "
                           "mean and peak-to-mean ratio begin when every section is installed.",
            "wall_convergence": "Piecewise-linear wall displacement evaluated at the fixed section position.",
        },
        "runs": runs,
    }


if __name__ == "__main__":
    result = build(json.loads(EVENTS.read_text(encoding="utf-8")),
                   json.loads(SOURCE.read_text(encoding="utf-8")))
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
