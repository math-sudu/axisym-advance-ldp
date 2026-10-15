"""Advance excavation and lining with an incremental FE-integrated PINN.

Network weights and committed material states are saved together after each
accepted release increment. The saved conventional spin-up supplies the
initial condition; subsequent equilibrium is obtained by neural training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from activation_pinn import (
    ROOT, IncrementNetwork, LocalIncrementNetwork, IncrementProblem,
    base, np, physical_quantities, restore, corrected_kernel, torch, train_increment,
)
from runtime import write_json
from neural_model import initial_model
from neural_optimization import refine_head, refine_network





def begin_round(model, index):
    face = index * model.dstep
    for position in (face - model.dstep, face, face - model.liner_d):
        if not np.isclose(model.z_lines, position, rtol=0, atol=1e-9).any():
            raise ValueError("Construction endpoint is not a mesh boundary")
    elements = model._rings(face - model.dstep, face)
    model.fe.f_release = model.fe.removal_force("rock", elements)
    model.fe.set_active("rock", elements, False)
    lining = model.fe.blocks["liner"]
    before = lining["active"].copy()
    model._liner_activation(face)
    born = lining["active"] & ~before
    if np.any(lining["sig"][born] != 0):
        raise ValueError("New lining must start stress-free")
    return born


def refresh_history(model, index, info, elapsed):
    model.history = [{
        "k": index, "x_f": index * model.dstep,
        "wall": -model.fe.u_n[2 * model.wall_nodes] / model.u_max,
        "info": info, "wall_s": elapsed,
    }]


def save_checkpoint(path, model, network, run, index, release, born, predictor=None, field_anchor=None):
    arrays = base.state_arrays(model)
    arrays.update(c_reduced=model.solver.c, f_release=model.fe.f_release)
    payload = {
        "network": {key: value.cpu() for key, value in network.state_dict().items()},
        "arrays": {key: torch.from_numpy(value.copy()) for key, value in arrays.items()},
        "run": run, "round": index, "release": release,
        "born": torch.from_numpy(born.copy()),
        "predictor": predictor,
        "field_anchor": field_anchor,
    }
    temporary = path.with_suffix(".pt.part")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, network):
    saved = torch.load(path, weights_only=True)
    arrays = {key: value.numpy() for key, value in saved["arrays"].items()}
    for name, expected in (("rho_lines_m", model.rho_lines), ("z_lines_m", model.z_lines)):
        np.testing.assert_array_equal(arrays[name], expected)
    for name, block in model.fe.blocks.items():
        np.testing.assert_array_equal(arrays[name + "_elems"], block["elems"])
        for field in ("sig", "active", "plast"):
            block[field] = arrays[name + "_" + field].copy()
    model.fe.u_n = arrays["displacement_m"].copy()
    model.fe.f_release = arrays["f_release"].copy()
    model.solver.c = arrays["c_reduced"].copy()
    network.load_state_dict(saved["network"])
    refresh_history(model, saved["round"], {}, saved["run"]["elapsed_s"])
    return (saved["run"], saved["round"], saved["release"], saved["born"].numpy(),
            saved.get("predictor"), saved.get("field_anchor"))


def predict_increment(problem, predictor):
    """Extrapolate the preceding neural parameter update if it lowers the loss."""
    network = problem.network
    original = {name: parameter.detach().clone() for name, parameter in network.named_parameters()}
    with torch.no_grad():
        before = float(problem.objective()[0])
        for name, parameter in network.named_parameters():
            parameter.add_(predictor[name].to(parameter.device))
        after = float(problem.objective()[0])
        used = np.isfinite(after) and after < before
        if not used:
            for name, parameter in network.named_parameters():
                parameter.copy_(original[name])
    return {"used": bool(used), "loss_before": before, "loss_predicted": after}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    with corrected_kernel():
        model, source = initial_model(args.round_length, args.gap, args.mesh)
        if args.rounds > model.n_steps:
            raise ValueError("Requested rounds extend beyond the construction track")
        network = (IncrementNetwork() if args.architecture == "global" else
                   LocalIncrementNetwork(moving=args.architecture in ("moving-local", "face-local"),
                                         face_patch=args.round_length
                                         if args.architecture == "face-local" else None))
        network.to(args.device)
        initial_weights = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        if args.initial_network is not None and not args.resume:
            weights = torch.load(args.initial_network, weights_only=True)
            with torch.no_grad():
                for name, parameter in network.named_parameters():
                    parameter.copy_(weights[name])
        controls = {
            "case": "B", "mesh_level": 0 if args.mesh == "coarse" else 1,
            "round_length_over_a": args.round_length,
            "installation_gap_over_a": args.gap, "architecture": args.architecture,
            "initial_state": source.relative_to(ROOT).as_posix(),
        }
        if args.mesh != "coarse":
            controls["mesh"] = args.mesh
        checkpoint = args.output / "checkpoint.pt"
        if args.resume:
            result, index, release, born, predictor, field_anchor = load_checkpoint(checkpoint, model, network)
            if result["controls"] != controls:
                raise ValueError("Resume geometry, initial state and architecture must match")
            previous_elapsed = result["elapsed_s"]
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            index, release = 1, 0.0
            predictor = None
            field_anchor = initial_weights if args.reuse_increment else None
            born = begin_round(model, index)
            previous_elapsed = 0.0
            result = {
                "method": "incremental_finite_element_integrated_PINN",
                "method_sources": ["10.1016/j.cma.2024.117474", "10.1016/j.ijmecsci.2025.110075"],
                "controls": controls, "field_supervision": False,
                "nodes": model.mesh.n_nodes, "elements": model.mesh.n_elems,
                "increments": [], "rounds": [], "last_completed_round": 0, "elapsed_s": 0.0,
            }
            if args.initial_network is not None:
                result["network_initialization"] = {
                    "weights": args.initial_network.resolve().relative_to(ROOT).as_posix(),
                    "transfer": "Trainable parameters only; coordinate buffers and physical initial state belong to the target model",
                }
        reuse_increment = args.reuse_increment or field_anchor is not None
        if reuse_increment and args.predict_increment:
            raise ValueError("A fixed neural reference cannot use parameter extrapolation")
        if reuse_increment and field_anchor is None:
            field_anchor = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            pending = result["increments"][-1] if result["increments"] else None
            pending_target = (pending is not None and not pending.get("accepted", pending["converged"])
                              and pending["round"] == index
                              and pending["release_fraction"] == min(1.0, round(release + args.release_step, 14)))
            if predictor is not None and not pending_target:
                # The preceding accepted increment was represented relative
                # to these older parameters. Retain that field for transfer.
                for name in predictor:
                    field_anchor[name] -= predictor[name]
        anchor_network = None
        if reuse_increment:
            anchor_network = copy.deepcopy(network)
            anchor_network.load_state_dict(field_anchor)
            anchor_network.requires_grad_(False)
        result["requested_rounds"] = args.rounds
        result["training_controls"] = {
            "iterations": args.iterations, "relative_force_tolerance": args.tolerance,
            "release_step": args.release_step, "device": args.device,
            "cpu_threads": args.cpu_threads,
            "predict_increment": args.predict_increment,
            "head_steps": args.head_steps,
            "gauss_newton_steps": args.gauss_newton_steps,
            "cg_iterations": args.cg_iterations,
            "reuse_increment": reuse_increment,
            "accept_training_limit": args.accept_training_limit,
        }
        result["completed"] = False
        save_checkpoint(checkpoint, model, network, result, index, release, born, predictor, field_anchor)
        while index <= args.rounds:
            if release >= 1:
                index += 1
                if index > args.rounds:
                    break
                born, release = begin_round(model, index), 0.0
            target = min(1.0, round(release + args.release_step, 14))
            if index in args.archive_rounds and target == 1.0:
                save_checkpoint(args.output / f"round-{index:03d}-before-final.pt",
                                model, network, result, index, release, born, predictor, field_anchor)
            origin = ((index - 1) * model.dstep
                      if args.architecture in ("moving-local", "face-local") else 0.0)
            problem = IncrementProblem(model, target, network, coordinate_origin=origin,
                                       anchor_network=anchor_network)
            accepted_weights = {key: value.clone() for key, value in network.state_dict().items()}
            pending = result["increments"][-1] if result["increments"] else None
            continue_trial = (
                args.resume and pending is not None and not pending.get("accepted", pending["converged"])
                and pending["round"] == index and pending["release_fraction"] == target)
            if continue_trial:
                # Freeze the original anchor/scales before loading the trial.
                # The committed material history remains at the prior release.
                network.load_state_dict(torch.load(args.output / "trial-network.pt", weights_only=True))
            prediction = None
            if not continue_trial and args.predict_increment and predictor is not None:
                prediction = predict_increment(problem, predictor)
            head_history = refine_head(
                problem, args.head_steps, args.tolerance,
                observe=lambda trial: physical_quantities(model, trial, index * model.dstep, born),
                face_only=args.architecture == "face-local",
            ) if args.head_steps else []
            neural_history = refine_network(
                problem, args.gauss_newton_steps, args.tolerance, cg_iterations=args.cg_iterations,
                observe=lambda trial: physical_quantities(model, trial, index * model.dstep, born)
            ) if args.gauss_newton_steps else []
            if args.device == "cuda" and (head_history or neural_history):
                torch.cuda.empty_cache()
            u, c, record = train_increment(
                problem, args.iterations, args.tolerance,
                observe=lambda trial: physical_quantities(model, trial, index * model.dstep, born))
            record.update(round=index, release_fraction=target,
                          iteration_limit=args.iterations, force_tolerance=args.tolerance,
                          accepted=record["converged"] or args.accept_training_limit,
                          increment_representation="fixed_neural_reference" if reuse_increment else "reset_neural_reference",
                          **physical_quantities(model, u, index * model.dstep, born))
            if continue_trial:
                record["continued_trial_from_evaluation"] = pending["evaluations"]
            if prediction is not None:
                record["parameter_prediction"] = prediction
            if head_history:
                record["head_refinement"] = head_history
            if neural_history:
                record["gauss_newton_refinement"] = neural_history
            result["increments"].append(record)
            result["elapsed_s"] = previous_elapsed + time.perf_counter() - started
            print(json.dumps({key: record[key] for key in (
                "round", "release_fraction", "relative_force", "converged", "accepted",
                "face_convergence_mm", "new_lining_peak_thrust_kN_per_m")}), flush=True)
            if not record["accepted"]:
                np.savez_compressed(args.output / "trial.npz", displacement_m=u, c_reduced=c)
                torch.save({key: value.cpu() for key, value in network.state_dict().items()},
                           args.output / "trial-network.pt")
                network.load_state_dict(accepted_weights)
                save_checkpoint(checkpoint, model, network, result, index, release, born, predictor, field_anchor)
                write_json(args.output / "run.json", result)
                return 2
            model.fe.commit(u)
            model.solver.c = c
            predictor = None if reuse_increment else {
                name: (parameter.detach() - accepted_weights[name]).cpu()
                for name, parameter in network.named_parameters()}
            release = target
            refresh_history(model, index, record, result["elapsed_s"])
            if release == 1:
                model.fe.f_release.fill(0)
                arrays = base.state_arrays(model)
                arrays["c_reduced"] = c
                meta = {"k": index, "x_f": index * model.dstep, "info": record,
                        "wall_s": result["elapsed_s"]}
                base.save_npz(args.output / f"round-{index:03d}.npz", meta, arrays)
                result["last_completed_round"] = index
                result["rounds"].append({key: record[key] for key in record
                                         if key not in ("history", "response_history", "head_refinement",
                                                        "gauss_newton_refinement")})
            save_checkpoint(checkpoint, model, network, result, index, release, born, predictor, field_anchor)
            if release == 1.0 and index in args.archive_rounds:
                save_checkpoint(args.output / f"round-{index:03d}-checkpoint.pt",
                                model, network, result, index, release, born, predictor, field_anchor)
            write_json(args.output / "run.json", result)
        result["completed"] = result["last_completed_round"] >= args.rounds
        result["elapsed_s"] = previous_elapsed + time.perf_counter() - started
        write_json(args.output / "run.json", result)
        save_checkpoint(checkpoint, model, network, result, model.history[-1]["k"], release, born, predictor, field_anchor)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round-length", type=float, default=0.4)
    parser.add_argument("--gap", type=float, default=0.6)
    parser.add_argument("--mesh", choices=["coarse", "production", "design", "interior"], default="coarse",
                        help="Use the original prototype or a shared manuscript event mesh")
    parser.add_argument("--release-step", type=float, default=0.0625)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--architecture", choices=["global", "local", "moving-local", "face-local"],
                        default="local")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    prediction = parser.add_mutually_exclusive_group()
    prediction.add_argument("--predict-increment", action="store_true",
                        help="Initialize each increment by extrapolating the last neural weight update")
    prediction.add_argument("--reuse-increment", action="store_true",
                            help="Transfer the normalized displacement increment with a fixed neural reference")
    parser.add_argument("--head-steps", type=int, default=0,
                        help="Damped output-layer least-squares steps before L-BFGS")
    parser.add_argument("--gauss-newton-steps", type=int, default=0,
                        help="Matrix-free least-squares steps for all neural parameters")
    parser.add_argument("--cg-iterations", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-training-limit", action="store_true",
                        help="Commit finite trained fields after the iteration limit; retain the actual force-convergence flag")
    parser.add_argument("--archive-rounds", type=int, nargs="*", default=[],
                        help="Preserve joint checkpoints before and after the final release of these rounds")
    parser.add_argument("--initial-network", type=Path,
                        help="Initialize trainable parameters from a saved neural model")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/pinn_activation/construction_B")
    args = parser.parse_args()
    if (args.rounds < 1 or min(args.iterations, args.head_steps, args.gauss_newton_steps) < 0
            or max(args.iterations, args.head_steps, args.gauss_newton_steps) == 0
            or args.cpu_threads < 1
            or args.cg_iterations < 1 or args.tolerance <= 0
            or not 0 < args.release_step <= 1 or args.round_length <= 0 or args.gap < 0):
        parser.error("Positive training and advance controls are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
