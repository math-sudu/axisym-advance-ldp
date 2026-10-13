"""Select compatible schemes under convergence and section-thrust allowances."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import fsum, isclose, isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/lining_activation/fixed_band_histories.json"
SECTION_SOURCE = SOURCE.with_name("section_histories.json")
OUT = SOURCE.with_name("construction_choices.json")
SCHEDULES = {
    "B": ((0.6, 0.4), (0.6, 0.2), (0.7, 0.2), (0.8, 0.2)),
    "C": ((0.6, 0.4), (0.6, 0.2)),
    "D": ((0.6, 0.4), (0.6, 0.2)),
}
RESPONSE_KEYS = ("convergence_envelope_mm", "demand_envelope_kN_per_m", "construction_rounds")


def select_under_allowances(schemes, convergence_mm, thrust_kN_per_m):
    """Return every feasible scheme and all ties for the fewest rounds."""
    if any(not isfinite(value) or value < 0 for value in (convergence_mm, thrust_kN_per_m)):
        raise ValueError("Allowances must be finite and nonnegative")
    feasible = [i for i, scheme in enumerate(schemes)
                if scheme["convergence_envelope_mm"] <= convergence_mm
                and scheme["demand_envelope_kN_per_m"] <= thrust_kN_per_m]
    rounds = min((schemes[i]["construction_rounds"] for i in feasible), default=None)
    return {
        "feasible_indices": feasible,
        "minimum_rounds": rounds,
        "selected_indices": [i for i in feasible if schemes[i]["construction_rounds"] == rounds],
    }


def allowance_map(schemes):
    """Partition nonnegative allowance space at the exact computed responses."""
    intervals = []
    for key in RESPONSE_KEYS[:2]:
        edges = sorted({0.0, *(scheme[key] for scheme in schemes)})
        intervals.append([[lo, hi] for lo, hi in zip(edges, edges[1:] + [None], strict=True)])
    convergence, thrust = intervals
    return {
        "convergence_intervals_mm": convergence,
        "thrust_intervals_kN_per_m": thrust,
        "cells": [[select_under_allowances(schemes, u_lower, t_lower)
                   for u_lower, _ in convergence] for t_lower, _ in thrust],
    }


def select_runs(data, case):
    selected = []
    for target, step in SCHEDULES[case]:
        matches = [run for run in data["runs"]
                   if run["case"] == case and run["mesh_level"] == 1
                   and isclose(run["installation_target_over_a"], target)
                   and isclose(run["round_length_over_a"], step)]
        if len(matches) != 1:
            raise ValueError(f"Expected one history for {case}, target={target}, round={step}")
        selected.append(matches[0])
    if len({run["initial_state"] for run in selected}) != 1:
        raise ValueError(f"Construction schemes for {case} have different initial states")
    return selected


def summarize(run, section_run):
    summary = run["summary"]
    convergence = summary["convergence_envelope"]
    band_demand = summary["demand_envelope"]
    demand = section_run["summary"]["demand_envelope"]
    control = section_run["sections"][demand["section_index"]]
    sections = section_run["sections"]
    length = fsum(section["axial_weight_over_a"] for section in sections)
    return {
        "installation_target_over_a": run["installation_target_over_a"],
        "round_length_over_a": run["round_length_over_a"],
        "construction_rounds": run["construction_rounds"],
        "source_directory": run["source_directory"],
        "complete_batch_center_distance_over_a": (run["installation_target_over_a"]
                                                   + run["round_length_over_a"] / 2),
        "mean_installation_distance_over_a": fsum(
            section["installation_distance_over_a"] * section["axial_weight_over_a"]
            for section in sections) / length,
        "installation_reference_mean_mm": summary["installation_reference_mean_mm"],
        "mean_post_installation_convergence_mm": summary["final_mean_post_installation_convergence_mm"],
        "convergence_envelope_mm": convergence["convergence_mm"],
        "convergence_control_band_over_a": convergence["band_over_a"],
        "convergence_control_face_over_a": convergence["face_over_a"],
        "demand_envelope_kN_per_m": demand["hoop_thrust_kN_per_m"],
        "demand_control_z_over_a": demand["z_over_a"],
        "demand_control_face_over_a": demand["face_over_a"],
        "demand_control_state": demand["source"],
        "demand_control_installation_face_over_a": control["installation_face_over_a"],
        "demand_control_installation_batch_over_a": control["installation_batch_over_a"],
        "demand_control_reference_state": control["reference_state"],
        "band_demand_envelope_kN_per_m": band_demand["hoop_thrust_kN_per_m"],
        "band_demand_control_band_over_a": band_demand["band_over_a"],
        "band_demand_control_face_over_a": band_demand["face_over_a"],
        "final_section_peak_to_mean": section_run["summary"]["final_peak_to_mean"],
        "final_mean_hoop_thrust_kN_per_m": summary["final_mean_hoop_thrust_kN_per_m"],
    }


def comparison(schemes, before, after):
    keys = (*RESPONSE_KEYS, "mean_installation_distance_over_a",
            "band_demand_envelope_kN_per_m", "installation_reference_mean_mm",
            "mean_post_installation_convergence_mm", "final_mean_hoop_thrust_kN_per_m")
    return {
        "before_index": before, "after_index": after,
        "change": {key: schemes[after][key] - schemes[before][key] for key in keys},
        "change_percent": {key: 100 * (schemes[after][key] / schemes[before][key] - 1)
                           for key in keys},
    }


def build(data, section_data):
    cases = {}
    for case in SCHEDULES:
        runs = select_runs(data, case)
        section_runs = select_runs(section_data, case)
        for run, section_run in zip(runs, section_runs, strict=True):
            for key in ("initial_state", "source_directory", "construction_rounds"):
                if run[key] != section_run[key]:
                    raise ValueError(f"Section and band histories disagree on {key}")
        schemes = [summarize(run, section_run) for run, section_run in zip(runs, section_runs, strict=True)]
        cases[case] = {
            "initial_state": runs[0]["initial_state"],
            "schemes": schemes,
            "allowance_map": allowance_map(schemes),
            "comparisons": [comparison(schemes, before, after)
                            for before, after in combinations(range(len(schemes)), 2)],
        }
    return {
        "history_source": SOURCE.relative_to(ROOT).as_posix(),
        "section_history_source": SECTION_SOURCE.relative_to(ROOT).as_posix(),
        **{key: data[key] for key in ("radius_m", "cohort_over_a", "band_width_over_a",
                                     "final_face_over_a", "evaluation_basis")},
        "demand_definition": section_data["definitions"]["demand_envelope"],
        "selection_rule": "Within each ground case, retain schemes with convergence and section "
                          "thrust no greater than the supplied allowances, then select all "
                          "feasible schemes having the fewest construction rounds. An empty "
                          "set means none of the evaluated candidates is feasible. Indices "
                          "refer to the case's ordered schemes.",
        "allowance_map_convention": "Intervals include their lower bound and exclude their "
                                    "upper bound; null upper bounds denote infinity. Cells "
                                    "are indexed by thrust interval, then convergence interval.",
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SCHEDULES)
    parser.add_argument("--convergence-allowance-mm", type=float)
    parser.add_argument("--thrust-allowance-kn-per-m", type=float)
    args = parser.parse_args()
    query = (args.case, args.convergence_allowance_mm, args.thrust_allowance_kn_per_m)
    if any(value is not None for value in query) and any(value is None for value in query):
        parser.error("Supply --case and both allowances together")
    result = build(json.loads(SOURCE.read_text(encoding="utf-8")),
                   json.loads(SECTION_SOURCE.read_text(encoding="utf-8")))
    if args.case is not None:
        schemes = result["cases"][args.case]["schemes"]
        try:
            selection = select_under_allowances(schemes, *query[1:])
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({
            "case": args.case,
            "convergence_allowance_mm": query[1],
            "thrust_allowance_kN_per_m": query[2],
            **selection,
            "feasible_schemes": [{"index": i, **schemes[i]} for i in selection["feasible_indices"]],
        }, indent=2, allow_nan=False))
    else:
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for case, group in result["cases"].items():
            print(f"{case}: {len(group['schemes'])} schemes; exact allowance map saved")
