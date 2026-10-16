"""Compare physical readouts of history-method runs at identical event fronts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_history_construction import (
    METHOD, ROOT, base, batch_response, initial_model, np, restore_snapshot,
    corrected_kernel, SupportHistoryOperator, torch, write_json,
    network_from_controls, build_trial,
)
from extract_lining_band_histories import wall_band_means


def read_trial(checkpoint):
    """Diagnose a saved unaccepted trial without advancing its physical history."""
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    pending, controls = saved["pending"], saved["run"]["controls"]
    if saved["method"] != METHOD or pending is None or "last_relative_force" not in pending:
        raise ValueError("Trial diagnostics require a completed, unaccepted history-method trial")
    model, _ = initial_model(controls["round_length_over_a"], controls["installation_gap_over_a"], controls["mesh"])
    network = network_from_controls(controls)
    problem = build_trial(model, network, pending, pending["target"])
    with torch.no_grad():
        residual, u, _ = problem.residual()
    relative = float(residual.norm() / problem.force_scale)
    np.testing.assert_allclose(relative, pending["last_relative_force"], atol=2e-12, rtol=1e-10)
    displacement, force = u.numpy(), residual.numpy()
    correction = problem.P @ problem.factor.solve(force)
    correction[problem.support.internal_dofs] = problem.support.recover_direction(correction[problem.support.contact_dofs])
    wall_z, wall = model.wall_z / base.scen.A, -1000 * displacement[2 * model.wall_nodes]
    wall_correction = 1000 * correction[2 * model.wall_nodes]
    dofs = problem.interior
    z = model.mesh.node_z[dofs // 2] / base.scen.A
    face = problem.fronts[0] / base.scen.A
    support = problem.fronts[1] / base.scen.A
    local = np.abs(z - face) <= 1
    largest = np.argsort(np.abs(force[:len(dofs)]))[-8:][::-1]
    return {"source": checkpoint.resolve().relative_to(ROOT).as_posix(),
            "state_kind": "unaccepted_trial", "controls": controls,
            "accepted_round": saved["accepted"]["round"], "round": pending["state"]["round"],
            "release_fraction": pending["target"], "relative_force": relative,
            "face_over_a": face, "support_front_over_a": support,
            "face_convergence_mm": float(np.interp(face, wall_z, wall)),
            "support_front_convergence_mm": float(np.interp(support, wall_z, wall)),
            "wall_z_over_a": wall_z.tolist(), "wall_convergence_mm": wall.tolist(),
            "batches": batch_response(problem, displacement),
            "elastic_correction_face_mm": float(np.interp(face, wall_z, wall_correction)),
            "elastic_correction_wall_max_mm": float(np.abs(wall_correction).max()),
            "residual_squared_fraction_within_a_of_face": float(np.sum(force[:len(dofs)][local] ** 2) / np.sum(force ** 2)),
            "residual_squared_fraction_at_exterior": float(np.sum(force[len(dofs):] ** 2) / np.sum(force ** 2)),
            "largest_ground_residual_dofs": [
                {"r_over_a": float(model.mesh.node_rho[dofs[i] // 2] / base.scen.A),
                 "z_over_a": float(z[i]), "component": int(dofs[i] % 2), "residual": float(force[i])}
                for i in largest]}


def read_round(directory, index):
    saved = torch.load(directory / f"round-{index:03d}-checkpoint.pt", weights_only=True, map_location="cpu")
    if saved["method"] != METHOD or saved["pending"] is not None:
        raise ValueError("Readout requires an accepted history-method round")
    controls = saved["run"]["controls"]
    model, _ = initial_model(controls["round_length_over_a"], controls["installation_gap_over_a"], controls["mesh"])
    history = restore_snapshot(model, saved["accepted"])
    support = SupportHistoryOperator(model.fe, history)
    problem = SimpleNamespace(model=model, support=support, history=history)
    lining = model.fe.blocks["liner"]
    np.testing.assert_allclose(support.recover_stress(model.fe.u_n), lining["sig"][lining["active"]], atol=2e-11, rtol=1e-10)
    batches = batch_response(problem, model.fe.u_n)
    recorded = next(row for row in reversed(saved["run"]["increments"])
                    if row["round"] == index and row["material_committed"] and row["release_fraction"] == 1)
    np.testing.assert_array_equal([row["event_id"] for row in recorded["batches"]],
                                  [row["event_id"] for row in batches])
    for row, actual in zip(recorded["batches"], batches):
        np.testing.assert_allclose(row["section_thrust_kN_per_m"], actual["section_thrust_kN_per_m"], atol=2e-8, rtol=1e-10)
    event = saved["accepted"]["event"]
    wall_z, wall = model.wall_z / base.scen.A, -1000 * model.fe.u_n[2 * model.wall_nodes]
    for row, reference in zip(batches, history.records):
        row["material_interval_over_a"] = [float(model.mesh.e_z0[reference.elements].min() / base.scen.A),
                                            float(model.mesh.e_z1[reference.elements].max() / base.scen.A)]
        row["peak_z_over_a"] = row["section_z_over_a"][np.argmax(row["section_thrust_kN_per_m"])]
    left = min(row["material_interval_over_a"][0] for row in batches if row["reference_kind"] == "actual_event")
    right = event["face_m"] / base.scen.A
    count = round((right - left) / 0.2)
    np.testing.assert_allclose(count * 0.2, right - left, atol=1e-12, rtol=0)
    edges = left + 0.2 * np.arange(count + 1)
    bands = np.column_stack([edges[:-1], edges[1:]])
    convergence = wall_band_means(wall_z, wall, bands)
    return {"source": directory.resolve().relative_to(ROOT).as_posix(), "controls": controls,
            "round": index, "nodes": model.mesh.n_nodes, "elements": model.mesh.n_elems,
            "face_over_a": event["face_m"] / base.scen.A,
            "support_front_over_a": event["support_front_m"] / base.scen.A,
            "relative_force": recorded["relative_force"],
            "face_convergence_mm": float(np.interp(event["face_m"] / base.scen.A, wall_z, wall)),
            "support_front_convergence_mm": float(np.interp(event["support_front_m"] / base.scen.A, wall_z, wall)),
            "convergence_bands_over_a": bands.tolist(), "band_convergence_mm": convergence.tolist(),
            "maximum_band_convergence_mm": float(np.max(convergence)),
            "batches": batches, "wall_z_over_a": wall_z.tolist(), "wall_convergence_mm": wall.tolist()}


def difference(reference, candidate):
    for key in ("round", "face_over_a", "support_front_over_a"):
        np.testing.assert_allclose(reference[key], candidate[key], atol=1e-12, rtol=0)
    np.testing.assert_array_equal([row["event_id"] for row in reference["batches"]],
                                  [row["event_id"] for row in candidate["batches"]])
    rows = []
    for first, second in zip(reference["batches"], candidate["batches"]):
        if first["event_id"] != second["event_id"]:
            raise ValueError("Batch events differ")
        np.testing.assert_allclose(first["material_interval_over_a"], second["material_interval_over_a"], atol=1e-12, rtol=0)
        positions = np.asarray(first["section_z_over_a"])
        candidate_positions = np.asarray(second["section_z_over_a"])
        left = max(positions.min(), candidate_positions.min())
        right = min(positions.max(), candidate_positions.max())
        common = (positions >= left - 1e-10) & (positions <= right + 1e-10)
        matched = np.interp(positions[common], candidate_positions, second["section_thrust_kN_per_m"])
        reference_sections = np.asarray(first["section_thrust_kN_per_m"])[common]
        rows.append({"event_id": first["event_id"], "material_interval_over_a": first["material_interval_over_a"],
                     "mean_change_percent": 100 * (second["mean_thrust_kN_per_m"] / first["mean_thrust_kN_per_m"] - 1),
                     "peak_change_percent": 100 * (second["peak_thrust_kN_per_m"] / first["peak_thrust_kN_per_m"] - 1),
                     "reference_peak_z_over_a": first["peak_z_over_a"],
                     "candidate_peak_z_over_a": second["peak_z_over_a"],
                     "common_section_z_over_a": positions[common].tolist(),
                     "common_section_change_kN_per_m": (matched - reference_sections).tolist(),
                     "common_section_peak_change_percent": 100 * (matched.max() / reference_sections.max() - 1)})
    left = max(min(reference["wall_z_over_a"]), min(candidate["wall_z_over_a"]))
    right = min(max(reference["wall_z_over_a"]), max(candidate["wall_z_over_a"]))
    boundaries = [reference[key] + offset for key in ("face_over_a", "support_front_over_a")
                  for offset in (-1, 1)]
    positions = np.unique([*reference["wall_z_over_a"], *candidate["wall_z_over_a"], *boundaries])
    positions = positions[(positions >= left) & (positions <= right)]
    change = (np.interp(positions, candidate["wall_z_over_a"], candidate["wall_convergence_mm"])
              - np.interp(positions, reference["wall_z_over_a"], reference["wall_convergence_mm"]))
    zones, locations = {}, {}
    for key in ("face_over_a", "support_front_over_a"):
        selected = np.flatnonzero(np.abs(positions - reference[key]) <= 1 + 1e-12)
        controlling = selected[np.argmax(np.abs(change[selected]))]
        zones[key] = float(np.abs(change[controlling]))
        locations[key] = float(positions[controlling])
    np.testing.assert_allclose(reference["convergence_bands_over_a"], candidate["convergence_bands_over_a"], atol=1e-12, rtol=0)
    band_change = np.asarray(candidate["band_convergence_mm"]) - reference["band_convergence_mm"]
    controlling = int(np.argmax(np.abs(band_change)))
    return {"reference": reference["source"], "candidate": candidate["source"], "round": reference["round"],
            "face_convergence_change_mm": candidate["face_convergence_mm"] - reference["face_convergence_mm"],
            "support_front_convergence_change_mm": candidate["support_front_convergence_mm"] - reference["support_front_convergence_mm"],
            "maximum_band_convergence_change_mm": candidate["maximum_band_convergence_mm"] - reference["maximum_band_convergence_mm"],
            "largest_band_difference_mm": float(band_change[controlling]),
            "largest_difference_band_over_a": reference["convergence_bands_over_a"][controlling],
            "maximum_wall_change_within_a_of_front_mm": zones,
            "maximum_wall_change_position_over_a": locations, "batches": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--trial-checkpoints", type=Path, nargs="*", default=[],
                        help="Also diagnose named unaccepted trials; these are not spatial convergence results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fields, comparisons = [], []
    with corrected_kernel():
        for index in args.rounds:
            reference = read_round(args.reference, index)
            fields.append(reference)
            for directory in args.candidates:
                candidate = read_round(directory, index)
                fields.append(candidate)
                comparisons.append(difference(reference, candidate))
        trials = [read_trial(checkpoint) for checkpoint in args.trial_checkpoints]
    write_json(args.output, {"method": METHOD,
                            "peak_definition": "Maximum at each mesh's native axial Gauss sections; physical batch intervals match",
                            "common_sections": "Candidate section profile interpolated onto reference Gauss sections inside both profiles; no extrapolation",
                            "wall_comparison": "Piecewise-linear wall profiles compared at the union of both meshes' nodes and zone boundaries within one radius of each front; shared domain only",
                            "fields": fields, "comparisons": comparisons,
                            "unaccepted_trial_diagnostics": trials})
    summaries = [{**row, "batches": [{key: value for key, value in batch.items()
                                      if key not in ("common_section_z_over_a", "common_section_change_kN_per_m")}
                                     for batch in row["batches"]]} for row in comparisons]
    print(json.dumps(summaries), flush=True)
