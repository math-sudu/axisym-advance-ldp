"""Read common-mesh design histories and compare the 12a and 14.4a tasks."""
from __future__ import annotations

import argparse
import json
import numpy as np

import compare_construction_schemes as choices
import extract_lining_band_histories as bands
import extract_lining_installation_events as events
import extract_lining_section_histories as sections
import extract_paired_section_strains as strains

ROOT = events.ROOT
OUT = ROOT / "results/analysis/lining_activation/construction_design"
U_ALLOW = 10.846933637497699
T_ALLOW = 1900.0


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def response_at_face(band_run, section_run, face):
    step = events.round_count(face, band_run["round_length_over_a"])
    br, sr = band_run["history"][:step + 1], section_run["history"][:step + 1]
    np.testing.assert_allclose([br[-1]["face_over_a"], sr[-1]["face_over_a"]], face)
    convergence = np.array([row["convergence_mm"] for row in br])
    thrust = np.array([row["hoop_thrust_kN_per_m"] for row in sr], dtype=float)
    peak = sections.peak_record(thrust, np.array([row["face_over_a"] for row in sr]),
                                section_run["sections"])
    ui = np.unravel_index(np.argmax(convergence), convergence.shape)
    return {
        "installation_target_over_a": band_run["installation_target_over_a"],
        "round_length_over_a": band_run["round_length_over_a"],
        "final_face_over_a": face, "construction_rounds": step,
        "convergence_envelope_mm": float(convergence[ui]),
        "convergence_control_face_over_a": br[ui[0]]["face_over_a"],
        "demand_envelope_kN_per_m": peak["hoop_thrust_kN_per_m"],
        "demand_control_face_over_a": peak["face_over_a"],
        "demand_control_z_over_a": peak["z_over_a"],
        "final_mean_hoop_thrust_kN_per_m": sr[-1]["cohort_mean_hoop_thrust_kN_per_m"],
        "source_directory": band_run["source_directory"],
    }


def complete_batches(section_run, strain_run):
    """Retain complete physical batches for mechanism interpretation."""
    groups = {}
    for i, section in enumerate(section_run["sections"]):
        limits = tuple(section["installation_batch_over_a"])
        groups.setdefault(limits, []).append(i)
    result = []
    for (lo, hi), ids in sorted(groups.items()):
        if lo < events.COHORT[0] - 1e-9 or hi > events.COHORT[1] + 1e-9:
            continue
        weights = np.array([section_run["sections"][i]["axial_weight_over_a"] for i in ids])
        np.testing.assert_allclose(weights.sum(), hi - lo, atol=1e-10)
        records = [strain_run["sections"][i] for i in ids]
        thrust = np.array([r["final"]["hoop_thrust_kN_per_m"] for r in records])
        reference_u = np.array([r["installation_convergence_mm"] for r in records])
        hoop_strain = np.array([r["final"]["mean_effective_strain"][2] for r in records])
        low, high = int(thrust.argmin()), int(thrust.argmax())
        components = np.array([r["final"]["thrust_components_kN_per_m"] for r in records])
        result.append({
            "batch_over_a": [lo, hi], "section_indices": ids,
            "mean_hoop_thrust_kN_per_m": float(np.average(thrust, weights=weights)),
            "thrust_range_kN_per_m": float(np.ptp(thrust)),
            "installation_convergence_range_mm": float(np.ptp(reference_u)),
            "post_installation_hoop_strain_range": float(np.ptp(hoop_strain)),
            "high_minus_low_thrust_components_kN_per_m": (components[high] - components[low]).tolist(),
        })
    return {
        "installation_target_over_a": section_run["installation_target_over_a"],
        "round_length_over_a": section_run["round_length_over_a"],
        "final_face_over_a": section_run["history"][-1]["face_over_a"],
        "source_directory": section_run["source_directory"], "batches": result,
    }


def analyse(root=OUT):
    control_paths = sorted(root.glob("*_g*_round*/controls.json"))
    if not control_paths:
        raise FileNotFoundError(f"No design runs under {root}")
    controls = sorted([(p, json.loads(p.read_text(encoding="utf-8"))) for p in control_paths],
                      key=lambda item: (item[1]["case"], -item[1]["round_length_over_a"],
                                        item[1]["installation_target_over_a"]))
    control_paths, control_sets = zip(*controls, strict=True)
    geometry_keys = ("axial_track_spacing_over_a", "axial_track_end_over_a",
                     "exterior_anchor_over_a", "axial_event_positions_over_a")
    common_geometry = {key: control_sets[0].get(key) for key in geometry_keys}
    for control in control_sets[1:]:
        if {key: control.get(key) for key in geometry_keys} != common_geometry:
            raise ValueError("Design runs must share the same construction geometry")
    for case in {c["case"] for c in control_sets}:
        if len({c["initial_state"] for c in control_sets if c["case"] == case}) != 1:
            raise ValueError("Compared schedules must share their initial state")
    final_faces = {c["final_face_over_a"] for c in control_sets}
    if len(final_faces) != 1:
        raise ValueError("Design runs must share a final face")
    final_face = final_faces.pop()
    schedules, runs = {}, []
    for path, c in zip(control_paths, control_sets, strict=True):
        final = path.parent / f"step-{events.round_count(final_face, c['round_length_over_a']):03d}.npz"
        if not final.exists():
            raise FileNotFoundError(f"Complete the design run before analysis: {final}")
        schedules.setdefault(c["case"], []).append((c["installation_target_over_a"], c["round_length_over_a"]))
        run = events.extract_run(path.parent, c["case"], c["nominal_distance_over_a"],
                                 c["level"], c["round_length_over_a"],
                                 ROOT / c["initial_state"], final_face)
        runs.append(run)
    event_path, band_path = root / "installation_events.json", root / "fixed_band_histories.json"
    section_path = root / "section_histories.json"
    event_data = {"radius_m": events.A, "cohort_over_a": list(events.COHORT), "runs": runs}
    write_json(event_path, event_data)
    band_data = bands.build(event_path)
    write_json(band_path, band_data)
    section_data = sections.build(event_data, band_data, schedules, event_path, band_path)
    write_json(section_path, section_data)
    strain_data = strains.build(section_data, schedules, section_path)
    write_json(root / "paired_section_strains.json", strain_data)
    tasks = []
    for face in (12.0, final_face):
        for case in schedules:
            br = choices.select_runs(band_data, case, schedules)
            sr = choices.select_runs(section_data, case, schedules)
            responses = [response_at_face(b, s, face) for b, s in zip(br, sr, strict=True)]
            baseline = next(r for r in responses if np.isclose(r["installation_target_over_a"], 0.6)
                            and np.isclose(r["round_length_over_a"], 0.4))
            for r in responses:
                r["change_from_same_mesh_baseline_percent"] = {
                    key: 100 * (r[key] / baseline[key] - 1) for key in (
                        "convergence_envelope_mm", "demand_envelope_kN_per_m",
                        "final_mean_hoop_thrust_kN_per_m")}
            tasks.append({"case": case, "final_face_over_a": face, "schemes": responses,
                          "selection": choices.select_under_allowances(responses, U_ALLOW, T_ALLOW)})
    result = {
        **common_geometry,
        "convergence_allowance_mm": U_ALLOW, "thrust_allowance_kN_per_m": T_ALLOW,
        "tasks": tasks,
        "batch_distributions": [complete_batches(s, e) for s, e in
                                zip(section_data["runs"], strain_data["runs"], strict=True)],
    }
    write_json(root / "construction_choices.json", result)
    for task in tasks:
        print(json.dumps(task), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interior", action="store_true",
                        help="Read the common geometry with the 0.72a installation events")
    args = parser.parse_args()
    analyse(OUT / "interior_position" if args.interior else OUT)
