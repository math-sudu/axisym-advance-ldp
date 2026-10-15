"""Compare neural construction histories on a common material interval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from construction_response import analyse


def compare(sources, cohort=(3.0, 6.0), face=12.0):
    responses = [analyse(source, cohort, face) for source in sources]
    baseline = responses[0]
    if any(row["summary"]["final_mean_thrust_kN_per_m"] is None for row in responses):
        raise ValueError("Every scheme must have installed the complete observation interval")
    quantities = {
        "convergence_envelope_mm": lambda row: row["summary"]["convergence_envelope_mm"],
        "mean_thrust_kN_per_m": lambda row: row["summary"]["final_mean_thrust_kN_per_m"],
        "peak_thrust_kN_per_m": lambda row: row["summary"]["demand_envelope"]["thrust_kN_per_m"],
    }
    comparisons = []
    for source, response in zip(sources, responses):
        values = {name: get(response) for name, get in quantities.items()}
        changes = {name: 100 * (value / quantities[name](baseline) - 1)
                   for name, value in values.items()}
        run = json.loads((source / "run.json").read_text(encoding="utf-8"))
        numerical_history = []
        for step in range(1, response["completed_rounds"] + 1):
            records = [row for row in run["increments"] if row["round"] == step
                       and row.get("accepted", row["converged"])]
            numerical_history.append({
                "round": step,
                "release_fractions": [row["release_fraction"] for row in records],
                "force_tolerances": [row["force_tolerance"] for row in records],
                "relative_forces": [row["relative_force"] for row in records],
            })
        matched = []
        for batch in baseline["complete_batches"]:
            centre = np.mean(batch["batch_over_a"])
            for other in response["complete_batches"]:
                if not np.isclose(centre, np.mean(other["batch_over_a"]), rtol=0, atol=1e-10):
                    continue
                first_section = baseline["sections"][batch["section_indices"][0]]
                other_section = response["sections"][other["section_indices"][0]]
                matched.append({
                    "centre_over_a": float(centre),
                    "reference_batch_over_a": batch["batch_over_a"],
                    "batch_over_a": other["batch_over_a"],
                    "reference_installation_face_over_a": first_section["installation_face_over_a"],
                    "installation_face_over_a": other_section["installation_face_over_a"],
                    "reference_mean_thrust_kN_per_m": batch["final_mean_thrust_kN_per_m"],
                    "mean_thrust_kN_per_m": other["final_mean_thrust_kN_per_m"],
                    "reference_peak_thrust_kN_per_m": batch["final_peak_thrust_kN_per_m"],
                    "peak_thrust_kN_per_m": other["final_peak_thrust_kN_per_m"],
                    "reference_thrust_range_kN_per_m": batch["final_thrust_range_kN_per_m"],
                    "thrust_range_kN_per_m": other["final_thrust_range_kN_per_m"],
                    "reference_hoop_strain_range": batch["post_installation_hoop_strain_range"],
                    "hoop_strain_range": other["post_installation_hoop_strain_range"],
                })
        comparisons.append({
            "source_directory": response["source_directory"],
            "controls": response["controls"], "quantities": values,
            "numerical_history": numerical_history,
            "percent_change_from_first": changes,
            "same_centre_complete_batches": matched,
        })
    return {
        "cohort_over_a": list(cohort), "evaluation_face_over_a": face,
        "reference_source": baseline["source_directory"],
        "comparisons": comparisons, "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--cohort", type=float, nargs=2, default=(3.0, 6.0))
    parser.add_argument("--face", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.sources, args.cohort, args.face)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
