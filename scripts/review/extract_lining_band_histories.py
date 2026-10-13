"""Read construction histories on fixed, nonoverlapping physical bands.

The installation-event file supplies the actual checkpoint directories and
reference states. Stresses at each accepted face are integrated through the
lining thickness and averaged by axial length. An incomplete band has no load
reading; a newly completed stress-free installation is a valid reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extract_lining_installation_events import A, COHORT, ROOT, read_state

EVENTS = ROOT / "results/analysis/lining_activation/installation_events.json"
OUT = ROOT / "results/analysis/lining_activation/fixed_band_histories.json"
WIDTH = 0.2
FINAL_FACE = 12.0


def overlap_weights(native_bands, bands):
    """Length weights for piecewise-constant native-band stress resultants."""
    native_bands, bands = np.asarray(native_bands), np.asarray(bands)
    np.testing.assert_allclose(native_bands[:-1, 1], native_bands[1:, 0], atol=1e-10)
    overlap = np.maximum(0.0, np.minimum(bands[:, None, 1], native_bands[None, :, 1])
                         - np.maximum(bands[:, None, 0], native_bands[None, :, 0]))
    overlap[overlap < 1e-10] = 0.0
    widths = bands[:, 1] - bands[:, 0]
    np.testing.assert_allclose(overlap.sum(axis=1), widths, atol=1e-10)
    return overlap / widths[:, None]


def wall_band_means(z_over_a, convergence_mm, bands):
    """Integrate the piecewise-linear wall displacement over each band."""
    z = np.asarray(z_over_a)
    if not np.isfinite(convergence_mm).all():
        raise ValueError("Wall convergence contains a nonfinite source value")
    if np.any(np.diff(z) <= 0) or bands[0][0] < z[0] or bands[-1][1] > z[-1]:
        raise ValueError("Wall nodes must cover the requested bands in axial order")
    result = []
    for lo, hi in bands:
        points = np.r_[lo, z[(z > lo) & (z < hi)], hi]
        values = np.interp(points, z, convergence_mm)
        result.append(np.diff(points) @ ((values[:-1] + values[1:]) / 2) / (hi - lo))
    return np.asarray(result)


def band_loads(native_thrust, weights, reference_steps, step):
    """Include a band only after its last constituent installation event."""
    if not np.isfinite(native_thrust).all():
        raise ValueError("Native hoop thrust contains a nonfinite source value")
    installed = np.array([np.all(reference_steps[row > 0] <= step) for row in weights])
    thrust = weights @ native_thrust
    thrust[~installed] = np.nan
    return thrust


def finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def peak_record(values, faces, bands, value_key):
    time, band = np.unravel_index(np.nanargmax(values), values.shape)
    return {value_key: float(values[time, band]), "face_over_a": float(faces[time]),
            "band_over_a": bands[band].tolist(), "band_index": int(band)}


def extract_run(run, bands):
    events = sorted(run["events"], key=lambda event: event["band_over_a"][0])
    native_bands = np.asarray([event["band_over_a"] for event in events])
    weights = overlap_weights(native_bands, bands)
    reference_steps = np.asarray([event["reference_step"] for event in events])
    first_loaded_steps = np.asarray([event["first_loaded_step"] for event in events])
    np.testing.assert_array_equal(first_loaded_steps, reference_steps + 1)
    full_installation_steps = np.array([reference_steps[row > 0].max() for row in weights])
    initial = read_state(ROOT / run["initial_state"])
    lo, hi = initial["liner_e_z0"] / A, initial["liner_e_z1"] / A
    thickness = initial["liner_e_rho1"] - initial["liner_e_rho0"]
    selectors = [np.isclose(lo, left, rtol=0, atol=1e-10)
                 & np.isclose(hi, right, rtol=0, atol=1e-10)
                 for left, right in native_bands]
    np.testing.assert_allclose([thickness[selected].sum() for selected in selectors], 0.2)
    reference = np.zeros(len(bands))
    rows, convergence, thrusts, faces = [], [], [], []
    final_step = round(FINAL_FACE / run["round_length_over_a"])
    for step in range(final_step + 1):
        path = (ROOT / run["initial_state"] if step == 0
                else ROOT / run["source_directory"] / f"step-{step:03d}.npz")
        state = initial if step == 0 else read_state(path)
        face = state["face_over_a"]
        np.testing.assert_allclose(face, step * run["round_length_over_a"], atol=1e-10)
        np.testing.assert_array_equal(state["liner_e_z0"] / A, lo)
        np.testing.assert_array_equal(state["liner_e_z1"] / A, hi)
        wall_mm = -state["displacement_m"][2 * state["wall_nodes"]] * 1000
        wall_z = state["z_lines_m"] / A
        u = wall_band_means(wall_z, wall_mm, bands)
        native_thrust = []
        for index, selected in enumerate(selectors):
            np.testing.assert_array_equal(state["liner_active"][selected],
                                          step >= first_loaded_steps[index])
            stress = state["liner_sig"][selected, :, 2].mean(axis=1)
            if step < first_loaded_steps[index]:
                np.testing.assert_allclose(stress, 0.0, atol=1e-14)
            native_thrust.append(-stress @ thickness[selected] * 1000)
            if step == reference_steps[index]:
                event = events[index]
                np.testing.assert_allclose(face, event["installation_face_over_a"])
                np.testing.assert_allclose(
                    np.interp(native_bands[index].mean(), wall_z, wall_mm),
                    event["installation_convergence_mm"], atol=1e-10)
                for band in np.flatnonzero(weights[:, index]):
                    overlap = [[max(bands[band, 0], native_bands[index, 0]),
                                min(bands[band, 1], native_bands[index, 1])]]
                    reference[band] += (weights[band, index]
                                        * wall_band_means(wall_z, wall_mm, overlap)[0])
        thrust = band_loads(native_thrust, weights, reference_steps, step)
        pressure = thrust / (A * 1000)
        mean_pressure = float(np.average(pressure, weights=np.diff(bands).ravel())) \
            if np.isfinite(pressure).all() else None
        rows.append({"step": step, "face_over_a": face,
                     "source": path.relative_to(ROOT).as_posix(),
                     "convergence_mm": u.tolist(), "hoop_thrust_kN_per_m": finite_list(thrust),
                     "pressure_MPa": finite_list(pressure),
                     "cohort_mean_pressure_MPa": mean_pressure,
                     "cohort_peak_to_mean": float(np.max(pressure) / mean_pressure)
                     if mean_pressure is not None and mean_pressure != 0 else None})
        convergence.append(u)
        thrusts.append(thrust)
        faces.append(face)
    convergence, thrusts, faces = map(np.asarray, (convergence, thrusts, faces))
    if not np.isfinite(thrusts[-1]).all():
        raise ValueError("The cohort has not completed installation by the evaluation face")
    widths = np.diff(bands).ravel()
    final_pressure = thrusts[-1] / (A * 1000)
    post = convergence[-1] - reference
    demand_peak = peak_record(thrusts, faces, bands, "hoop_thrust_kN_per_m")
    demand_peak["pressure_MPa"] = demand_peak["hoop_thrust_kN_per_m"] / (A * 1000)
    installation = []
    for band, step in enumerate(full_installation_steps):
        native_indices = np.flatnonzero(weights[band])
        installation.append({
            "band_over_a": bands[band].tolist(),
            "native_event_indices": native_indices.tolist(),
            "installation_faces_over_a": sorted({events[index]["installation_face_over_a"]
                                                 for index in native_indices}),
            "fully_installed_face_over_a": float(faces[step]),
            "reference_convergence_mm": float(reference[band]),
            "convergence_at_full_installation_mm": float(convergence[step, band]),
        })
    return {
        **{key: run[key] for key in ("case", "mesh_level", "nominal_distance_over_a",
                                   "round_length_over_a", "installation_target_over_a",
                                   "source_directory", "initial_state")},
        "construction_rounds": final_step,
        "cohort_fully_installed_face_over_a": float(faces[full_installation_steps.max()]),
        "band_installation": installation,
        "history": rows,
        "summary": {
            "installation_reference_mean_mm": float(np.average(reference, weights=widths)),
            "installation_reference_maximum_mm": float(reference.max()),
            "installation_reference_peak_band_over_a": bands[np.argmax(reference)].tolist(),
            "final_mean_convergence_mm": float(np.average(convergence[-1], weights=widths)),
            "final_maximum_convergence_mm": float(convergence[-1].max()),
            "final_mean_post_installation_convergence_mm": float(np.average(post, weights=widths)),
            "final_maximum_post_installation_convergence_mm": float(post.max()),
            "final_mean_hoop_thrust_kN_per_m": float(np.average(thrusts[-1], weights=widths)),
            "final_mean_pressure_MPa": float(np.average(final_pressure, weights=widths)),
            "final_maximum_pressure_MPa": float(final_pressure.max()),
            "final_peak_to_mean": rows[-1]["cohort_peak_to_mean"],
            "convergence_envelope": peak_record(convergence, faces, bands, "convergence_mm"),
            "demand_envelope": demand_peak,
        },
    }


def build(events_path):
    event_data = json.loads(events_path.read_text(encoding="utf-8"))
    np.testing.assert_allclose(event_data["radius_m"], A)
    np.testing.assert_allclose(event_data["cohort_over_a"], COHORT)
    edges = np.round(np.linspace(*COHORT, round((COHORT[1] - COHORT[0]) / WIDTH) + 1), 10)
    bands = np.column_stack((edges[:-1], edges[1:]))
    return {
        "radius_m": A, "cohort_over_a": list(COHORT), "band_width_over_a": WIDTH,
        "bands_over_a": bands.tolist(), "final_face_over_a": FINAL_FACE,
        "evaluation_basis": "common global face position",
        "installation_events": events_path.relative_to(ROOT).as_posix(),
        "definitions": {
            "spatial_readout": "Native Gauss-point hoop stress is integrated through thickness; "
                               "native resultants are averaged by axial overlap length. "
                               "Wall convergence is the band integral of piecewise-linear nodal displacement.",
            "sampling": "Accepted equilibria from x_f/a=0 through 12, including stress-free "
                        "installation at each accepted face before the next release. "
                        "Every band in a row uses the same actual face position.",
            "installation_reference": "Length average of each constituent section's convergence "
                                      "at its own actual installation event. This reference field is "
                                      "distinct from convergence at full-band installation.",
            "convergence_envelope": "Maximum band-averaged inward convergence relative to the "
                                    "initial zero displacement, over all saved equilibria including "
                                    "pre-installation states.",
            "demand_envelope": "Maximum over bands and saved equilibria from each band's complete "
                               "installation to x_f/a=12. Incomplete-band loads are null. "
                               "Cohort means and peak-to-mean ratios begin after all bands are installed.",
            "construction_rounds": "Advance rounds from x_f/a=0 to 12; excludes the initial 10a excavation.",
        },
        "runs": [extract_run(run, bands) for run in event_data["runs"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS)
    args = parser.parse_args()
    result = build(args.events.resolve())
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for run in result["runs"]:
        print(json.dumps({"case": run["case"], "mesh": run["mesh_level"],
                          "d_pre": run["installation_target_over_a"],
                          "round": run["round_length_over_a"], **run["summary"]}))


if __name__ == "__main__":
    main()
