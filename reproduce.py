"""Reproduce construction histories, section demand and allowance selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# CPU calculations use one numerical thread, as in the manuscript runs.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/review")]

import torch
from threadpoolctl import threadpool_limits

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
THREAD_LIMIT = threadpool_limits(limits=1)

from compare_construction_schemes import SCHEDULES
from runtime import geometry, jsonable, save_npz, state_arrays
from solver import build, configure, restore


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_directory(case, gap, length):
    name = f"{case}_g{gap:g}_round{length:g}".replace(".", "p")
    return ROOT / "results/runs" / name


def advance(model, step, directory):
    geo = geometry(model, step)
    record = model.step(step)
    arrays = state_arrays(model)
    arrays["c_reduced"] = model.solver.c.copy()
    meta = {"k": step, "x_f": record["x_f"], "wall_s": record["wall_s"],
            "info": jsonable(record["info"]), "geometry": geo}
    save_npz(directory / f"step-{step:03d}.npz", meta, arrays)
    print(json.dumps({"case": model.arm.name, "step": step,
                      "face_over_a": record["x_f"] / 2,
                      "seconds": record["wall_s"]}), flush=True)


def simulate(case, gap, length, stop_after=None):
    """Generate/reuse the shared spin-up, then resume this schedule."""
    if (gap, length) not in SCHEDULES[case]:
        raise ValueError(f"No manuscript schedule for {(case, gap, length)}")
    model = build(case)
    initial_dir = ROOT / "results/runs" / f"{case}_initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial = initial_dir / "step-000.npz"
    if not initial.exists():
        advance(model, 0, initial_dir)
    else:
        restore(model, initial)
    configure(model, gap, length)
    output = run_directory(case, gap, length)
    output.mkdir(parents=True, exist_ok=True)
    controls = {"case": case, "mesh_level": 1, "installation_target_over_a": gap,
                "round_length_over_a": length, "cap": 1 / 16, "tol_rel": 1e-6,
                "initial_state": initial.relative_to(ROOT).as_posix()}
    control_path = output / "controls.json"
    if control_path.exists():
        if json.loads(control_path.read_text(encoding="utf-8")) != controls:
            raise ValueError("Saved schedule controls differ")
    else:
        write_json(control_path, controls)
    saved = sorted(output.glob("step-*.npz"))
    first = restore(model, saved[-1])["k"] + 1 if saved else 1
    last = model.n_steps if stop_after is None else min(stop_after, model.n_steps)
    for step in range(first, last + 1):
        advance(model, step, output)
    write_json(output / "completion.json", {
        "completed": last == model.n_steps or first > model.n_steps,
        "last_completed_step": max(first - 1, last), "final_step": model.n_steps})


def analyse():
    import extract_lining_installation_events as events
    import extract_lining_band_histories as bands
    import extract_lining_section_histories as sections
    import extract_paired_section_strains as strains
    import compare_construction_schemes as choices

    runs = []
    for case, schedules in SCHEDULES.items():
        for gap, length in schedules:
            directory = run_directory(case, gap, length)
            final = directory / f"step-{round(12 / length):03d}.npz"
            if not final.is_file():
                raise FileNotFoundError(f"Complete the schedule before analysis: {final}")
            initial = ROOT / "results/runs" / f"{case}_initial/step-000.npz"
            runs.append(events.extract_run(directory, case, round(gap + length, 12),
                                           1, length, initial))
    event_data = {"radius_m": 2.0, "cohort_over_a": [3.0, 6.0], "runs": runs}
    write_json(events.OUT, event_data)
    band_data = bands.build(events.OUT)
    write_json(bands.OUT, band_data)
    section_data = sections.build(event_data, band_data)
    write_json(sections.OUT, section_data)
    write_json(strains.OUT, strains.build(section_data))
    result = choices.build(band_data, section_data)
    write_json(choices.OUT, result)
    for case, group in result["cases"].items():
        for scheme in group["schemes"]:
            print(json.dumps({"case": case, **{key: scheme[key] for key in (
                "installation_target_over_a", "round_length_over_a", "construction_rounds",
                "convergence_envelope_mm", "demand_envelope_kN_per_m")}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="Generate or resume accepted equilibria")
    run.add_argument("--all", action="store_true", help="Run all eight manuscript schedules")
    run.add_argument("--case", choices=SCHEDULES, default="B")
    run.add_argument("--gap", type=float, default=0.6, help="Installation front gap / radius")
    run.add_argument("--round-length", type=float, default=0.4, help="Round length / radius")
    run.add_argument("--stop-after", type=int, help="Stop after this completed round for a short run")
    commands.add_parser("analyse", help="Extract the eight-schedule response and strain histories")
    query = commands.add_parser("select", help="Query the generated allowance map")
    query.add_argument("--case", choices=SCHEDULES, required=True)
    query.add_argument("--convergence-mm", type=float, required=True)
    query.add_argument("--thrust-kn-per-m", type=float, required=True)
    args = parser.parse_args()
    if args.command == "simulate":
        if args.stop_after is not None and args.stop_after < 1:
            parser.error("--stop-after must be at least 1")
        schedules = [(case, gap, length) for case, pairs in SCHEDULES.items()
                     for gap, length in pairs] if args.all else [(args.case, args.gap, args.round_length)]
        for case, gap, length in schedules:
            simulate(case, gap, length, args.stop_after)
    elif args.command == "analyse":
        analyse()
    else:
        from compare_construction_schemes import OUT, select_under_allowances
        data = json.loads(OUT.read_text(encoding="utf-8"))
        schemes = data["cases"][args.case]["schemes"]
        selection = select_under_allowances(schemes, args.convergence_mm, args.thrust_kn_per_m)
        print(json.dumps({"case": args.case, **selection,
                          "feasible_schemes": [schemes[i] for i in selection["feasible_indices"]]},
                         indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
