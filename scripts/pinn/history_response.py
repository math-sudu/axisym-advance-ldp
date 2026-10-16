"""Read construction response from complete installation-reference biases."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_history_construction import METHOD, ROOT, base, initial_model, corrected_kernel, torch
from support_history_operator import InstallationReference, SupportHistory, SupportHistoryOperator
from extract_lining_band_histories import finite_list, wall_band_means
from extract_lining_section_histories import section_quadrature, integrate_thrust

A = base.scen.A


def read_state(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if key != "meta"}


def round_paths(source, last):
    """Follow an explicit checkpoint branch for its inherited completed rounds."""
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    inherited, paths = 0, []
    if "continuation_source" in run:
        start = torch.load(source / "continuation-start.pt", weights_only=True, map_location="cpu")
        inherited = start["run"]["last_completed_round"]
        parent = (ROOT / run["continuation_source"]).parent
        paths = round_paths(parent, min(inherited, last))
    return paths + [source / f"round-{i:03d}.npz" for i in range(inherited + 1, last + 1)]


def restore_field(model, state, initial=False):
    """Recover all active support stress using its saved complete history."""
    np.testing.assert_array_equal(state["rho_lines_m"], model.rho_lines)
    np.testing.assert_array_equal(state["z_lines_m"], model.z_lines)
    for name, block in model.fe.blocks.items():
        np.testing.assert_array_equal(state[name + "_elems"], block["elems"])
        for key in ("active", "sig", "plast"):
            block[key] = state[name + "_" + key].copy()
    model.fe.u_n = state["displacement_m"].copy()
    if initial:
        history = SupportHistory.from_initial(model.fe)
    else:
        events = state["reference_event"]
        history = SupportHistory(tuple(InstallationReference(
            state["reference_elements"][events == event],
            state["reference_bias"][events == event], int(event),
            "equivalent_initial" if event == 0 else "actual_event")
            for event in np.unique(events)))
    operator = SupportHistoryOperator(model.fe, history)
    stress = operator.recover_stress(model.fe.u_n)
    stored = model.fe.blocks["liner"]["sig"][model.fe.blocks["liner"]["active"]]
    np.testing.assert_allclose(stress, stored, rtol=1e-10, atol=2e-10)
    return history, operator, float(np.max(np.abs(stress - stored)))


def analyse(source: Path, cohort=(3.0, 6.0), face=None):
    source = source.resolve()
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if run["method"] != METHOD:
        raise ValueError("Complete-reference readout requires the history method")
    controls = run["controls"]
    length, last = controls["round_length_over_a"], run["last_completed_round"]
    if face is not None:
        requested = round(face / length)
        if not np.isclose(requested * length, face) or requested > last:
            raise ValueError("The evaluation face must be a completed round")
        last = requested
    with corrected_kernel():
        model, initial_path = initial_model(length, controls["installation_gap_over_a"], controls["mesh"])
    if initial_path != ROOT / controls["initial_state"]:
        raise ValueError("The readout initial state differs from the run")
    paths = [initial_path, *round_paths(source, last)]
    initial = read_state(initial_path)
    lo, hi = initial["liner_e_z0"] / A, initial["liner_e_z1"] / A
    selected = (lo >= cohort[0] - 1e-10) & (hi <= cohort[1] + 1e-10)
    if not selected.any() or initial["liner_active"][selected].any():
        raise ValueError("The cohort must contain lining born during this history")
    native = np.unique(np.column_stack([lo[selected], hi[selected]]), axis=0)
    np.testing.assert_allclose(native[[0, -1], [0, 1]], cohort, atol=1e-10)
    np.testing.assert_allclose(native[:-1, 1], native[1:, 0], atol=1e-10)
    quadrature = section_quadrature(initial, native)
    z = np.array([row[0] for row in quadrature])
    weights = np.repeat(np.diff(native).ravel() / 2, 2)
    count = round((cohort[1] - cohort[0]) / 0.2)
    np.testing.assert_allclose(count * 0.2, cohort[1] - cohort[0], atol=1e-10)
    edges = cohort[0] + 0.2 * np.arange(count + 1)
    bands = np.column_stack([edges[:-1], edges[1:]])
    local_elements = np.array([row[1] for row in quadrature])
    gauss = np.array([row[2] for row in quadrature])
    radial_weights = np.array([row[3] for row in quadrature])
    elements = initial["liner_elems"][local_elements]
    strain_operator, dofs = model.mesh.B_all[elements, gauss], model.mesh.dof_all[elements]
    sections = [{"z_over_a": float(position), "axial_weight_over_a": float(weight)}
                for position, weight in zip(z, weights)]
    events = {row["event_id"]: row for row in run["events"]}
    previous = initial
    known_references, batch_bounds = {}, {}
    rows, maximum_stress_difference = [], 0.0
    final_strain = np.full((*elements.shape, 4), np.nan)
    for step, path in enumerate(paths):
        state = initial if step == 0 else read_state(path)
        history, operator, error = restore_field(model, state, initial=step == 0)
        maximum_stress_difference = max(maximum_stress_difference, error)
        for record in history.records:
            if record.event_id in known_references:
                old = known_references[record.event_id]
                np.testing.assert_array_equal(record.elements, old.elements)
                np.testing.assert_array_equal(record.stress_bias, old.stress_bias)
            else:
                known_references[record.event_id] = record
                if record.event_id:
                    event = events[record.event_id]
                    np.testing.assert_array_equal(record.elements, event["born_elements"])
                    np.testing.assert_allclose(event["face_m"] / A, step * length, atol=1e-10)
                    left = float(model.mesh.e_z0[record.elements].min() / A)
                    right = float(model.mesh.e_z1[record.elements].max() / A)
                    batch_bounds[record.event_id] = [left, right]
                    for i, section_elements in enumerate(elements):
                        if np.isin(section_elements, record.elements).all():
                            sections[i].update(
                                event_id=record.event_id, reference_kind=record.kind,
                                reference_source=path.relative_to(ROOT).as_posix(),
                                reference_arrays=["reference_elements", "reference_bias", "reference_event"],
                                first_loaded_step=step, reference_step=step - 1,
                                installation_face_over_a=(event["face_m"] - length * A) / A,
                                installation_batch_over_a=[left, right],
                                installation_convergence_mm=rows[-1]["wall_convergence_mm"][i])
        if np.any(previous["liner_active"] & ~state["liner_active"]):
            raise ValueError("Lining deactivated during construction")
        installed = state["liner_active"][local_elements].all(axis=1)
        thrust = integrate_thrust(state["liner_sig"][:, :, 2], quadrature)
        values = np.where(installed, thrust, np.nan)
        wall = -1000 * state["displacement_m"][2 * state["wall_nodes"]]
        # C^{-1} bias is a complete strain reference, including the equivalent
        # initial record; no dormant-node displacement is inferred here.
        effective = np.full_like(final_strain, np.nan)
        components = np.full((len(z), 3), np.nan)
        mean_strain = np.full((len(z), 4), np.nan)
        if installed.any():
            lookup = np.searchsorted(operator.elements, elements[installed])
            np.testing.assert_array_equal(operator.elements[lookup], elements[installed])
            bias = operator.stress_bias[lookup, gauss[installed]]
            strain = np.einsum("sgij,sgj->sgi", strain_operator[installed], state["displacement_m"][dofs[installed]])
            effective[installed] = strain - np.linalg.solve(operator.C, bias.reshape(-1, 4).T).T.reshape(bias.shape)
            rebuilt = effective[installed] @ operator.C.T
            np.testing.assert_allclose(rebuilt, state["liner_sig"][local_elements[installed], gauss[installed]],
                                       rtol=1e-10, atol=2e-10)
            components[installed] = -1000 * np.einsum(
                "sg,sgc,c->sc", radial_weights[installed], effective[installed, :, :3], operator.C[2, :3])
            np.testing.assert_allclose(components[installed].sum(axis=1), values[installed], atol=2e-8, rtol=1e-10)
            mean_strain[installed] = np.einsum("sg,sgc->sc", radial_weights[installed], effective[installed]) / radial_weights[installed].sum(axis=1)[:, None]
        row = {"step": step, "face_over_a": step * length,
               "source": path.relative_to(ROOT).as_posix(),
               "wall_convergence_mm": np.interp(z, model.wall_z / A, wall).tolist(),
               "band_convergence_mm": wall_band_means(model.wall_z / A, wall, bands).tolist(),
               "hoop_thrust_kN_per_m": finite_list(values),
               "mean_hoop_thrust_kN_per_m": float(np.average(values, weights=weights)) if installed.all() else None,
               "mean_effective_strain": [finite_list(item) for item in mean_strain],
               "thrust_components_kN_per_m": [finite_list(item) for item in components]}
        rows.append(row)
        previous, final_strain = state, effective
    for i, section in enumerate(sections):
        if "event_id" in section:
            row = rows[section["reference_step"]]
            row["hoop_thrust_kN_per_m"][i] = 0.0
            row["mean_effective_strain"][i] = [0.0] * 4
            row["thrust_components_kN_per_m"][i] = [0.0] * 3
    values = np.array([row["hoop_thrust_kN_per_m"] for row in rows], dtype=float)
    for row, thrust in zip(rows, values):
        if np.isfinite(thrust).all():
            row["mean_hoop_thrust_kN_per_m"] = float(np.average(thrust, weights=weights))
    batches = []
    for event_id, (left, right) in batch_bounds.items():
        if left < cohort[0] - 1e-10 or right > cohort[1] + 1e-10:
            continue
        indices = [i for i, section in enumerate(sections) if section.get("event_id") == event_id]
        np.testing.assert_allclose(weights[indices].sum(), right - left, atol=1e-10)
        hoop = (radial_weights[indices] * final_strain[indices, :, 2]).sum(axis=1) / radial_weights[indices].sum(axis=1)
        batches.append({"event_id": event_id, "batch_over_a": [left, right], "section_indices": indices,
                        "final_mean_thrust_kN_per_m": float(np.average(values[-1, indices], weights=weights[indices])),
                        "final_peak_thrust_kN_per_m": float(np.max(values[-1, indices])),
                        "final_thrust_range_kN_per_m": float(np.ptp(values[-1, indices])),
                        "post_installation_hoop_strain_range": float(np.ptp(hoop))})
    peak = None
    if np.isfinite(values).any():
        t, i = np.unravel_index(np.nanargmax(values), values.shape)
        peak = {"thrust_kN_per_m": float(values[t, i]), "z_over_a": float(z[i]),
                "face_over_a": rows[t]["face_over_a"], "section_index": int(i), "source": rows[t]["source"]}
    convergence = np.array([row["band_convergence_mm"] for row in rows])
    t, i = np.unravel_index(np.argmax(convergence), convergence.shape)
    return {"method": METHOD, "source_directory": source.relative_to(ROOT).as_posix(), "controls": controls,
            "cohort_over_a": list(cohort), "bands_over_a": bands.tolist(), "completed_rounds": last,
            "evaluation_face_over_a": last * length, "completed_requested_history": run["completed"],
            "reference_definition": "Complete saved stress bias; effective strain = B u - C^{-1} bias. Actual events are distinct from the equivalent initial record.",
            "summary": {"convergence_envelope_mm": float(convergence[t, i]),
                        "convergence_control_face_over_a": rows[t]["face_over_a"],
                        "convergence_control_band_over_a": bands[i].tolist(), "demand_envelope": peak,
                        "final_mean_thrust_kN_per_m": rows[-1]["mean_hoop_thrust_kN_per_m"],
                        "maximum_strain_stress_identity_difference_MPa": maximum_stress_difference},
            "sections": sections, "history": rows, "complete_batches": batches}
