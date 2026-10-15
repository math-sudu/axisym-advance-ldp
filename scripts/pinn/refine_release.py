"""Check target-response sensitivity by further training one frozen increment."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from activation_pinn import IncrementProblem, LocalIncrementNetwork, physical_quantities
from run_construction import ROOT, initial_model, load_checkpoint, np, corrected_kernel, torch, write_json
from neural_optimization import refine_network
from extract_lining_band_histories import wall_band_means


def refine(args):
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, weights_only=True)
    controls = saved["run"]["controls"]
    if controls["architecture"] != "face-local":
        raise ValueError("This response check uses the face-local construction model")
    with corrected_kernel():
        model, _ = initial_model(controls["round_length_over_a"],
                                 controls["installation_gap_over_a"], controls.get("mesh", "coarse"))
        network = LocalIncrementNetwork(moving=True, face_patch=controls["round_length_over_a"])
        network.to(args.device)
        _, index, release, born, _, field_anchor = load_checkpoint(args.checkpoint, model, network)
        if not release < args.release <= 1:
            raise ValueError("The target release must follow the committed release")
        anchor = None
        if field_anchor is not None:
            anchor = copy.deepcopy(network)
            anchor.load_state_dict(field_anchor)
            anchor.requires_grad_(False)
        problem = IncrementProblem(model, args.release, network,
                                   coordinate_origin=(index - 1) * model.dstep,
                                   anchor_network=anchor)
        trained = torch.load(args.trained_checkpoint, weights_only=True)
        if (trained["round"], trained["release"]) != (index, args.release):
            raise ValueError("The trained checkpoint must end at the evaluated release")
        network.load_state_dict(trained["network"])

        selected = None
        if args.cohort is not None:
            left, right = args.cohort
            block = model.fe.blocks["liner"]
            elements = block["elems"]
            lo, hi = model.mesh.e_z0[elements] / 2, model.mesh.e_z1[elements] / 2
            selected = (lo >= left - 1e-10) & (hi <= right + 1e-10)
            if not selected.any() or not block["active"][selected].all():
                raise ValueError("The refinement cohort must be fully installed")
            np.testing.assert_allclose([lo[selected].min(), hi[selected].max()],
                                       [left, right], rtol=0, atol=1e-10)
            count = round((right - left) / 0.2)
            if count < 1 or not np.isclose(count * 0.2, right - left):
                raise ValueError("The cohort must span complete 0.2a readout bands")
            edges = np.linspace(left, right, count + 1)
            bands = np.column_stack([edges[:-1], edges[1:]])

        def observe(u):
            values = physical_quantities(model, u, index * model.dstep, born)
            if selected is not None:
                cohort = physical_quantities(model, u, index * model.dstep, selected)
                values["cohort_response"] = {
                    key.removeprefix("new_"): value for key, value in cohort.items()
                    if key.startswith("new_lining_")}
                wall = -1000 * u[2 * model.wall_nodes]
                convergence = wall_band_means(model.wall_z / 2, wall, bands)
                values["cohort_response"]["maximum_band_convergence_mm"] = float(max(convergence))
            return values

        def response():
            with torch.no_grad():
                loss, force, u, c = problem.objective()
            return u.cpu().numpy(), c.cpu().numpy(), {
                "loss": float(loss), "relative_force": float(force.norm() / problem.force_scale),
                **observe(u.cpu().numpy())}

        before_u, _, before = response()
        np.testing.assert_allclose(before_u, trained["arrays"]["displacement_m"].numpy(),
                                   rtol=1e-10, atol=1e-12)
        history = refine_network(problem, args.steps, args.tolerance,
                                 cg_iterations=args.cg_iterations, observe=observe)
        u, c, after = response()
        result = {
            "committed_checkpoint": args.checkpoint.resolve().relative_to(ROOT).as_posix(),
            "trained_checkpoint": args.trained_checkpoint.resolve().relative_to(ROOT).as_posix(),
            "round": index, "release_fraction": args.release, "controls": controls,
            "cohort_over_a": args.cohort,
            "training_controls": {"steps": args.steps, "cg_iterations": args.cg_iterations,
                                  "relative_force_tolerance": args.tolerance, "device": args.device,
                                  "cpu_threads": args.cpu_threads},
            "before": before, "after": after, "training_history": history,
            "maximum_displacement_change_mm": float(np.abs(u - before_u).max() * 1000),
            "elapsed_s": time.perf_counter() - started,
        }
        write_json(args.output.with_suffix(".json"), result)
        np.savez_compressed(args.output.with_suffix(".npz"), displacement_m=u, c_reduced=c)
        torch.save({key: value.cpu() for key, value in network.state_dict().items()},
                   args.output.with_suffix(".pt"))
        keys = ("relative_force", "face_convergence_mm", "new_lining_mean_thrust_kN_per_m",
                "new_lining_peak_thrust_kN_per_m")
        print(json.dumps({name: {key: result[name][key] for key in keys}
                          for name in ("before", "after")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--release", type=float, required=True)
    parser.add_argument("--cohort", type=float, nargs=2,
                        help="Installed material interval for final-load sensitivity")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cg-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=0.0002)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    refine(parser.parse_args())
