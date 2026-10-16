"""Train a saved history-method release further without advancing its history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from threadpoolctl import threadpool_limits

from run_history_construction import (
    METHOD, ROOT, network_from_controls, base, batch_response, build_trial, initial_model,
    np, observe, corrected_kernel, torch, train_increment, weights, write_json, extend_network,
)
from neural_optimization import neural_linearization, refine_head, refine_network


def response(problem, event_id):
    with torch.no_grad():
        loss, residual, u, c = problem.objective()
    displacement = u.cpu().numpy()
    wall = -1000 * displacement[2 * problem.model.wall_nodes]
    front = problem.fronts[1]
    result = {"loss": float(loss),
              "relative_force": float(residual.norm() / problem.force_scale),
              **observe(problem, displacement, event_id),
              "support_front_convergence_mm": float(np.interp(front, problem.model.wall_z, wall)),
              "batches": batch_response(problem, displacement)}
    return displacement, c.cpu().numpy(), result


def response_change(problem, before_u, after_u, before, after):
    wall_z = problem.model.wall_z / base.scen.A
    wall_change = -1000 * (after_u - before_u)[2 * problem.model.wall_nodes]
    batches = []
    for old, new in zip(before["batches"], after["batches"]):
        section_change = np.asarray(new["section_thrust_kN_per_m"]) - old["section_thrust_kN_per_m"]
        row = {"event_id": old["event_id"],
               "maximum_section_change_kN_per_m": float(np.abs(section_change).max()),
               "before_peak_z_over_a": old["section_z_over_a"][np.argmax(old["section_thrust_kN_per_m"])],
               "after_peak_z_over_a": new["section_z_over_a"][np.argmax(new["section_thrust_kN_per_m"])]}
        for key in ("mean_thrust_kN_per_m", "peak_thrust_kN_per_m"):
            row[key + "_change"] = new[key] - old[key]
            row[key + "_change_percent"] = 100 * (new[key] / old[key] - 1)
        batches.append(row)
    zones = {}
    for name, front in zip(("face", "support_front"), problem.fronts):
        selected = np.abs(wall_z - front / base.scen.A) <= 1
        zones[name + "_plus_minus_a"] = float(np.abs(wall_change[selected]).max())
    return {"face_convergence_change_mm": after["face_convergence_mm"] - before["face_convergence_mm"],
            "support_front_convergence_change_mm": after["support_front_convergence_mm"] - before["support_front_convergence_mm"],
            "maximum_wall_change_mm": float(np.abs(wall_change).max()),
            "maximum_wall_change_by_zone_mm": zones, "batches": batches}


def refine(args):
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    accepted = torch.load(args.accepted_checkpoint, weights_only=True, map_location="cpu")
    if saved["method"] != METHOD or accepted["method"] != METHOD:
        raise ValueError("Both checkpoints must use the installation-history method")
    controls = saved["run"]["controls"]
    pending = saved["pending"]
    if pending is None or (pending["state"]["round"], pending["target"]) != (
            accepted["accepted"]["round"], accepted["accepted"]["release"]):
        raise ValueError("The accepted state must match the frozen trial")
    if args.support_local and not controls.get("support_local", False):
        extended = {**controls, "support_local": True}
        pending = {**pending, **{key: extend_network(pending[key], controls, extended)
                                 for key in ("anchor", "network")}}
        controls = extended
    with corrected_kernel():
        model, _ = initial_model(controls["round_length_over_a"],
                                 controls["installation_gap_over_a"], controls["mesh"])
    with corrected_kernel(), threadpool_limits(limits=args.cpu_threads, user_api="blas"):
        network = network_from_controls(controls).to(args.device)
        problem = build_trial(model, network, pending, pending["target"])
        event_id = pending["state"]["round"]
        before_u, _, before = response(problem, event_id)
        np.testing.assert_allclose(before_u, accepted["accepted"]["arrays"]["u"].numpy(),
                                   rtol=0, atol=2e-14)
        runs, elapsed = [], 0.0
        if args.resume:
            previous = torch.load(args.output.with_suffix(".pt"), weights_only=True, map_location="cpu")
            if previous["checkpoint"] != str(args.checkpoint.resolve()):
                raise ValueError("Refinement resume must use the original frozen checkpoint")
            network.load_state_dict(previous["network"])
            old_result = json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))
            runs, elapsed = old_result["training_runs"], old_result["elapsed_s"]
        observer = lambda field: observe(problem, field, event_id)
        start_u, _, start = response(problem, event_id)
        head = refine_head(problem, args.head_steps, args.tolerance, observe=observer,
                           solver=args.head_solver, rcond=args.head_rcond)
        refinement = refine_network(problem, args.gauss_newton_steps, args.tolerance,
                                    observe=observer, cg_iterations=args.cg_iterations)
        scale = neural_linearization(problem)[-1].rsqrt() if args.scaled_lbfgs and args.iterations else None
        _, _, training = train_increment(problem, args.iterations, args.tolerance, observe=observer,
                                         parameter_scale=scale)
        u, c, after = response(problem, event_id)
        # No trial may advance the physical anchor during this training-only check.
        frozen = pending["state"]["arrays"]
        np.testing.assert_array_equal(model.fe.u_n, frozen["u"].numpy())
        for name, block in model.fe.blocks.items():
            for key in ("sig", "plast", "active"):
                np.testing.assert_array_equal(block[key], frozen[name + "_" + key].numpy())
        runs.append({"iterations": args.iterations, "head_steps": args.head_steps,
                     "head_solver": args.head_solver, "head_rcond": args.head_rcond,
                     "head_scaling": "jacobian_column_norm",
                     "scaled_lbfgs": args.scaled_lbfgs,
                     "head": head, "gauss_newton_steps": args.gauss_newton_steps,
                     "cg_iterations": args.cg_iterations, "tolerance": args.tolerance,
                     "device": args.device, "cpu_threads": args.cpu_threads,
                     "blas_threads": args.cpu_threads,
                     "gauss_newton": refinement, "lbfgs": training,
                     "response_change": response_change(problem, start_u, u, start, after)})
        result = {"method": METHOD, "controls": controls, "round": event_id,
                  "release_fraction": pending["target"], "material_committed": False,
                  "frozen_checkpoint": args.checkpoint.resolve().relative_to(ROOT).as_posix(),
                  "accepted_checkpoint": args.accepted_checkpoint.resolve().relative_to(ROOT).as_posix(),
                  "before": before, "after": after,
                  "change": response_change(problem, before_u, u, before, after),
                  "training_runs": runs, "elapsed_s": elapsed + time.perf_counter() - started}
        write_json(args.output.with_suffix(".json"), result)
        np.savez_compressed(args.output.with_suffix(".npz"), displacement_m=u, c_reduced=c,
                            lining_stress_MPa=problem.support.recover_stress(u),
                            lining_elements=problem.support.elements,
                            wall_z_over_a=model.wall_z / base.scen.A,
                            wall_convergence_mm=-1000 * u[2 * model.wall_nodes])
        torch.save({"checkpoint": str(args.checkpoint.resolve()), "network": weights(network)},
                   args.output.with_suffix(".pt"))
        print(json.dumps({"round": event_id, "before_force": before["relative_force"],
                          "after_force": after["relative_force"], "change": result["change"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--head-steps", type=int, default=0)
    parser.add_argument("--head-solver", choices=("damped", "svd"), default="damped")
    parser.add_argument("--head-rcond", type=float, default=1e-9)
    parser.add_argument("--scaled-lbfgs", action="store_true")
    parser.add_argument("--support-local", action="store_true",
                        help="Add zero-output annular windows to the frozen trial before refinement")
    parser.add_argument("--gauss-newton-steps", type=int, default=0)
    parser.add_argument("--cg-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    refine(parser.parse_args())
