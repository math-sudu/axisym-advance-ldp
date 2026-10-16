"""Compare construction contrasts between two spatial discretizations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def schedule(response):
    controls = response["controls"]
    return (controls["installation_gap_over_a"], controls["round_length_over_a"])


def change(values, reference):
    return {key: float(100 * (value / reference[key] - 1)) for key, value in values.items()}


def stage_response(response, face):
    history = [row for row in response["history"] if row["face_over_a"] <= face + 1e-10]
    if not history or not np.isclose(history[-1]["face_over_a"], face, rtol=0, atol=1e-10):
        raise ValueError("Both grids must contain the requested physical stage")
    row = history[-1]
    if row["mean_hoop_thrust_kN_per_m"] is None:
        return None
    bands = np.asarray([item["band_convergence_mm"] for item in history])
    return {
        "convergence_envelope_mm": float(bands.max()),
        "mean_thrust_kN_per_m": row["mean_hoop_thrust_kN_per_m"],
        "final_peak_thrust_kN_per_m": max(row["hoop_thrust_kN_per_m"]),
        "thrust_envelope_kN_per_m": max(value for item in history
                                        for value in item["hoop_thrust_kN_per_m"]
                                        if value is not None),
    }


def matched_batches(reference, candidate):
    pairs = []
    for first in reference["complete_batches"]:
        centre = np.mean(first["batch_over_a"])
        for second in candidate["complete_batches"]:
            if not np.isclose(centre, np.mean(second["batch_over_a"]), rtol=0, atol=1e-10):
                continue
            first_section = reference["sections"][first["section_indices"][0]]
            second_section = candidate["sections"][second["section_indices"][0]]
            pairs.append({
                "centre_over_a": float(centre),
                "reference_batch_over_a": first["batch_over_a"],
                "candidate_batch_over_a": second["batch_over_a"],
                "reference_installation_face_over_a": first_section["installation_face_over_a"],
                "candidate_installation_face_over_a": second_section["installation_face_over_a"],
                "percent_change": change(
                    {key: second[key] for key in (
                        "final_mean_thrust_kN_per_m", "final_peak_thrust_kN_per_m",
                        "final_thrust_range_kN_per_m", "post_installation_hoop_strain_range")}, first),
            })
    return pairs


def spatial_sections(reference, candidate):
    """Compare profiles inside each batch without crossing installation jumps."""
    first, second = reference["sections"], candidate["sections"]
    first_values = reference["history"][-1]["hoop_thrust_kN_per_m"]
    second_values = candidate["history"][-1]["hoop_thrust_kN_per_m"]
    positions, values1, values2, events = [], [], [], []
    for event in dict.fromkeys(row["event_id"] for row in first):
        ids1 = [i for i, row in enumerate(first) if row["event_id"] == event]
        ids2 = [i for i, row in enumerate(second) if row["event_id"] == event]
        if not ids2:
            raise ValueError("Spatial comparison requires the same installation events")
        np.testing.assert_allclose(first[ids1[0]]["installation_batch_over_a"],
                                   second[ids2[0]]["installation_batch_over_a"], rtol=0, atol=1e-10)
        z1 = np.asarray([first[i]["z_over_a"] for i in ids1])
        z2 = np.asarray([second[i]["z_over_a"] for i in ids2])
        inside = (z1 >= z2.min() - 1e-10) & (z1 <= z2.max() + 1e-10)
        positions.extend(z1[inside].tolist())
        values1.extend(np.asarray([first_values[i] for i in ids1])[inside].tolist())
        values2.extend(np.interp(z1[inside], z2, [second_values[i] for i in ids2]).tolist())
        events.extend([event] * int(inside.sum()))
    return {"z_over_a": positions, "event_id": events,
            "reference_thrust_kN_per_m": values1, "candidate_thrust_kN_per_m": values2,
            "thrust_change_kN_per_m": (np.asarray(values2) - values1).tolist(),
            "peak_change_percent": float(100 * (max(values2) / max(values1) - 1))}


def convergence_partition(response):
    """Separate installation and subsequent convergence on aligned axial bands."""
    sections = response["sections"]
    z = np.asarray([row["z_over_a"] for row in sections])
    weight = np.asarray([row["axial_weight_over_a"] for row in sections])
    installation = np.asarray([row["installation_convergence_mm"] for row in sections])
    wall = np.asarray(response["history"][-1]["wall_convergence_mm"])
    initial, final = [], []
    for left, right in response["bands_over_a"]:
        selected = (z >= left) & (z <= right)
        initial.append(float(np.average(installation[selected], weights=weight[selected])))
        final.append(float(np.average(wall[selected], weights=weight[selected])))
    np.testing.assert_allclose(final, response["history"][-1]["band_convergence_mm"],
                               rtol=1e-10, atol=2e-10,
                               err_msg="Axial band edges must align with the section quadrature")
    return {"installation_mm": initial, "final_mm": final,
            "post_installation_mm": (np.asarray(final) - initial).tolist()}


def compare(reference, candidate):
    np.testing.assert_allclose(reference["cohort_over_a"], candidate["cohort_over_a"], atol=1e-10)
    np.testing.assert_allclose(reference["evaluation_face_over_a"], candidate["evaluation_face_over_a"], atol=1e-10)
    grids = [reference["responses"], candidate["responses"]]
    schedules = [schedule(row) for row in grids[0]]
    lookup = {schedule(row): row for row in grids[1]}
    if len(lookup) != len(schedules) or set(lookup) != set(schedules):
        raise ValueError("Paired grids must contain the same construction schedules")
    grids[1] = [lookup[key] for key in schedules]
    endpoint = reference["evaluation_face_over_a"]
    spatial = []
    for first, second in zip(*grids):
        np.testing.assert_allclose(first["bands_over_a"], second["bands_over_a"], atol=1e-10)
        first_values, second_values = (stage_response(row, endpoint) for row in (first, second))
        band_changes = []
        for row1, row2 in zip(first["history"], second["history"]):
            np.testing.assert_allclose(row1["face_over_a"], row2["face_over_a"], atol=1e-10)
            band_changes.append((np.asarray(row2["band_convergence_mm"])
                                 - row1["band_convergence_mm"]).tolist())
        partition1, partition2 = (convergence_partition(row) for row in (first, second))
        spatial.append({
            "schedule_gap_round_over_a": list(schedule(first)),
            "reference_source": first["source_directory"],
            "candidate_source": second["source_directory"],
            "reference_values": first_values, "candidate_values": second_values,
            "percent_change": change(second_values, first_values),
            "face_over_a": [row["face_over_a"] for row in first["history"]],
            "band_convergence_change_mm": band_changes,
            "complete_batches": matched_batches(first, second),
            "final_common_sections": spatial_sections(first, second),
            "convergence_partition_by_band": {
                "bands_over_a": first["bands_over_a"],
                "reference": partition1, "candidate": partition2,
                "difference_mm": {key: (np.asarray(partition2[key]) - partition1[key]).tolist()
                                  for key in partition1},
            },
        })
    contrasts = []
    for i in range(len(schedules) - 1):
        four = [grid[index] for grid in grids for index in (i, i + 1)]
        faces = sorted(set.intersection(*[
            {round(row["face_over_a"], 10) for row in response["history"]} for response in four]))
        stages = []
        for face in faces:
            values = [stage_response(response, face) for response in four]
            if any(value is None for value in values):
                continue
            reference_change = change(values[1], values[0])
            candidate_change = change(values[3], values[2])
            stages.append({
                "face_over_a": face,
                "reference_grid_percent_change": reference_change,
                "candidate_grid_percent_change": candidate_change,
                "difference_percentage_points": {
                    key: candidate_change[key] - reference_change[key] for key in reference_change},
            })
        contrasts.append({
            "reference_schedule_gap_round_over_a": list(schedules[i]),
            "candidate_schedule_gap_round_over_a": list(schedules[i + 1]),
            "stages": stages,
            "reference_grid_same_centre_batches": matched_batches(grids[0][i], grids[0][i + 1]),
            "candidate_grid_same_centre_batches": matched_batches(grids[1][i], grids[1][i + 1]),
        })
    return {
        "cohort_over_a": reference["cohort_over_a"], "evaluation_face_over_a": endpoint,
        "reference_mesh": grids[0][0]["controls"]["mesh"],
        "candidate_mesh": grids[1][0]["controls"]["mesh"],
        "common_section_sampling": "Candidate profile interpolated onto reference Gauss sections within each installation batch and both sampled extents; no extrapolation or interpolation across batch boundaries",
        "spatial_changes_by_schedule": spatial, "construction_contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(*(json.loads(path.read_text(encoding="utf-8"))
                       for path in (args.reference, args.candidate)))
    result["source_comparisons"] = [path.as_posix() for path in (args.reference, args.candidate)]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for contrast in result["construction_contrasts"]:
        print(json.dumps({"schedules": [contrast["reference_schedule_gap_round_over_a"],
                                       contrast["candidate_schedule_gap_round_over_a"]],
                          "final": contrast["stages"][-1]}))


if __name__ == "__main__":
    main()
