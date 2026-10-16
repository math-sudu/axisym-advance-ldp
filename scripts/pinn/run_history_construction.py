"""Run staged neural equilibrium with complete installation-history recovery.

checkpoint.pt keeps the last accepted physical state separately from a pending
event/trial. A failed release never advances that state. Restart reconstructs
the same anchor, installation references and active sets before loading trial
parameters. Optimizer state is restarted explicitly.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from threadpoolctl import threadpool_limits

from activation_pinn import ROOT, base, np, torch, train_increment
from run_construction import initial_model, refresh_history, corrected_kernel, write_json
from neural_optimization import neural_linearization, refine_head, refine_network
from support_history_operator import InstallationReference, SupportHistory, SupportHistoryOperator
from history_pinn import GroundNetwork, HistoryProblem, actual_fronts, prepare_event

METHOD = "installation_history_constrained_neural_field_v1"


def material_spec(model):
    return {name: {"kind": block["kind"], "E": float(block["E"]), "nu": float(block["nu"]),
                   "mat": {key: float(value) for key, value in block["mat"].items()}}
            for name, block in model.fe.blocks.items()}


def weights(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def network_from_controls(controls):
    return GroundNetwork(controls["width"], controls["seed"],
                         controls.get("face_patch_span_over_a"), controls.get("front_local", False),
                         controls.get("support_patch_span_over_a"), controls.get("support_local", False))


def extend_network(state, before, after):
    """Append zero-output branches while retaining the order of existing ones."""
    old_span, span = before.get("face_patch_span_over_a"), after.get("face_patch_span_over_a")
    old_local, local = before.get("front_local", False), after.get("front_local", False)
    old_support, support = before.get("support_patch_span_over_a"), after.get("support_patch_span_over_a")
    old_annulus, annulus = before.get("support_local", False), after.get("support_local", False)
    if old_annulus and (not annulus or old_span != span or old_local != local or old_support != support):
        raise ValueError("Existing annular networks cannot be removed or reordered")
    if old_span != span and (old_span is not None or span is None or old_local or old_support is not None):
        raise ValueError("Existing face windows cannot be changed or reordered")
    if old_local and not local:
        raise ValueError("Existing local front networks cannot be removed")
    if old_local != local and old_support is not None:
        raise ValueError("Existing support window cannot be reordered")
    if old_support != support and old_support is not None:
        raise ValueError("Existing support window cannot be changed or removed")
    original = network_from_controls(before)
    original.load_state_dict(state)
    expanded = network_from_controls(after)
    extended = weights(expanded)
    extended.update(weights(original))
    expanded.load_state_dict(extended)
    return weights(expanded)


def snapshot(model, history, index, release, born, event):
    arrays = {"u": model.fe.u_n, "c": model.solver.c, "f_release": model.fe.f_release,
              "rho": model.rho_lines, "z": model.z_lines}
    for name, block in model.fe.blocks.items():
        for key in ("elems", "active", "sig", "plast"):
            arrays[name + "_" + key] = block[key]
    return {"arrays": {key: torch.from_numpy(value.copy()) for key, value in arrays.items()},
            "references": [{"elements": torch.from_numpy(r.elements.copy()),
                            "stress_bias": torch.from_numpy(r.stress_bias.copy()),
                            "event_id": r.event_id, "kind": r.kind} for r in history.records],
            "round": index, "release": release, "born": torch.from_numpy(born.copy()),
            "event": event}


def restore_snapshot(model, saved):
    arrays = {key: value.cpu().numpy() for key, value in saved["arrays"].items()}
    np.testing.assert_array_equal(arrays["rho"], model.rho_lines)
    np.testing.assert_array_equal(arrays["z"], model.z_lines)
    for name, block in model.fe.blocks.items():
        np.testing.assert_array_equal(arrays[name + "_elems"], block["elems"])
        for key in ("active", "sig", "plast"):
            block[key] = arrays[name + "_" + key].copy()
    model.fe.u_n = arrays["u"].copy()
    model.fe.f_release = arrays["f_release"].copy()
    model.solver.c = arrays["c"].copy()
    history = SupportHistory(tuple(InstallationReference(
        row["elements"].numpy(), row["stress_bias"].numpy(), row["event_id"], row["kind"])
        for row in saved["references"]))
    fronts = actual_fronts(model)
    if fronts != (saved["event"]["face_m"], saved["event"]["support_front_m"]):
        raise ValueError("Checkpoint fronts disagree with its active elements")
    # This also verifies reference coverage before evaluating a trial field.
    SupportHistoryOperator(model.fe, history)
    return history


def save_checkpoint(path, accepted, accepted_weights, pending, result):
    payload = {"method": METHOD, "accepted": accepted, "network": accepted_weights,
               "pending": pending, "run": result}
    temporary = path.with_suffix(".pt.part")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_response(problem, displacement):
    """Read every batch from explicit recovered stress and physical quadrature."""
    mesh, support = problem.model.mesh, problem.support
    stress = support.recover_stress(displacement)
    rows = []
    for reference in problem.history.records:
        selected = np.isin(support.elements, reference.elements)
        elements = support.elements[selected]
        z, first, inverse = np.unique(np.round(mesh.gp_z[elements].ravel() / base.scen.A, 12),
                                       return_index=True, return_inverse=True)
        thickness = mesh.e_rho1[elements] - mesh.e_rho0[elements]
        thrust = np.bincount(inverse, weights=(-1000 * stress[selected, :, 2]
                                             * thickness[:, None] / 2).ravel())
        axial_weight = (mesh.e_z1[elements] - mesh.e_z0[elements])[first // 4] / 2
        rows.append({"event_id": reference.event_id, "reference_kind": reference.kind,
                     "mean_thrust_kN_per_m": float(np.average(thrust, weights=axial_weight)),
                     "peak_thrust_kN_per_m": float(thrust.max()),
                     "section_z_over_a": z.tolist(), "section_thrust_kN_per_m": thrust.tolist()})
    return rows


def observe(problem, displacement, event_id):
    batches = batch_response(problem, displacement)
    current = next((row for row in batches if row["event_id"] == event_id), None)
    wall = -1000 * displacement[2 * problem.model.wall_nodes]
    return {"face_convergence_mm": float(np.interp(problem.fronts[0], problem.model.wall_z, wall)),
            "new_lining_mean_thrust_kN_per_m": current["mean_thrust_kN_per_m"] if current else None,
            "new_lining_peak_thrust_kN_per_m": current["peak_thrust_kN_per_m"] if current else None}


def build_trial(model, network, pending, target):
    """Restore the frozen physical anchor, then the matching neural trial."""
    history = restore_snapshot(model, pending["state"])
    network.load_state_dict(pending["anchor"])
    problem = HistoryProblem(model, history, target, network)
    if target == pending["target"]:
        network.load_state_dict(pending["network"])
    return problem


def run(args):
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    with corrected_kernel():
        model, source = initial_model(args.round_length, args.gap, args.mesh)
    with corrected_kernel(), threadpool_limits(limits=args.cpu_threads, user_api="blas"):
        if args.rounds > model.n_steps:
            raise ValueError("Requested rounds extend beyond the event mesh")
        network = GroundNetwork(args.width, args.seed, args.face_patch_span,
                                args.front_local, args.support_patch_span, args.support_local).to(args.device)
        controls = {"mesh": args.mesh, "round_length_over_a": args.round_length,
                    "installation_gap_over_a": args.gap, "width": args.width, "seed": args.seed,
                    "initial_state": source.relative_to(ROOT).as_posix(),
                    "materials": material_spec(model)}
        if args.face_patch_span is not None:
            controls["face_patch_span_over_a"] = args.face_patch_span
        if args.front_local:
            controls["front_local"] = True
        if args.support_patch_span is not None:
            controls["support_patch_span_over_a"] = args.support_patch_span
        if args.support_local:
            controls["support_local"] = True
        if args.initial_network is not None:
            controls["initial_network"] = args.initial_network.resolve().relative_to(ROOT).as_posix()
            initial = torch.load(args.initial_network, weights_only=True, map_location="cpu")
            initial_weights = extend_network(initial["network"], initial["run"]["controls"], controls)
            network.load_state_dict(initial_weights)
        checkpoint = args.output / "checkpoint.pt"
        if args.resume or args.resume_from is not None:
            source_checkpoint = args.resume_from if args.resume_from is not None else checkpoint
            saved = torch.load(source_checkpoint, weights_only=True, map_location="cpu")
            source_payload = copy.deepcopy(saved) if args.resume_from is not None else None
            old_controls = saved["run"]["controls"]
            architecture_keys = ("face_patch_span_over_a", "front_local", "support_patch_span_over_a", "support_local")
            if (saved["method"] == METHOD and old_controls != controls
                    and {k: v for k, v in old_controls.items() if k not in architecture_keys}
                    == {k: v for k, v in controls.items() if k not in architecture_keys}):
                saved["network"] = extend_network(saved["network"], old_controls, controls)
                if saved["pending"] is not None:
                    for key in ("anchor", "network"):
                        saved["pending"][key] = extend_network(saved["pending"][key], old_controls, controls)
                saved["run"].setdefault("network_extensions", []).append({
                    "accepted_round": saved["accepted"]["round"],
                    "accepted_release": saved["accepted"]["release"],
                    "face_patch_span_over_a": args.face_patch_span,
                    "front_local": args.front_local,
                    "support_patch_span_over_a": args.support_patch_span,
                    "support_local": args.support_local,
                    "initial_output": "zero; existing fields and physical states preserved"})
                saved["run"]["controls"] = controls.copy()
            if saved["method"] != METHOD or saved["run"]["controls"] != controls:
                raise ValueError("Resume requires the same method, physical model and network")
            accepted, accepted_weights, pending, result = (
                saved[key] for key in ("accepted", "network", "pending", "run"))
            if args.resume_from is not None:
                args.output.mkdir(parents=True, exist_ok=False)
                torch.save(source_payload, args.output / "continuation-start.pt")
                result["continuation_source"] = source_checkpoint.resolve().relative_to(ROOT).as_posix()
            history = restore_snapshot(model, accepted)
            network.load_state_dict(accepted_weights)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            history = SupportHistory.from_initial(model.fe)
            fronts = actual_fronts(model)
            born = np.zeros_like(model.fe.blocks["liner"]["active"])
            accepted = snapshot(model, history, 0, 1.0, born,
                                {"event_id": 0, "face_m": fronts[0], "support_front_m": fronts[1]})
            accepted_weights, pending = weights(network), None
            result = {"method": METHOD, "controls": controls, "field_supervision": False,
                      "nodes": model.mesh.n_nodes, "elements": model.mesh.n_elems,
                      "events": [], "increments": [], "last_completed_round": 0,
                      "elapsed_s": 0.0, "completed": False}
        previous_elapsed = result["elapsed_s"]
        result["requested_rounds"] = args.rounds
        result["completed"] = False
        settings = {"relative_force_tolerance": args.tolerance, "release_step": args.release_step,
                    "accept_training_limit": args.accept_training_limit,
                    "face_patch_span_over_a": args.face_patch_span,
                    "front_local": args.front_local,
                    "support_patch_span_over_a": args.support_patch_span,
                    "support_local": args.support_local,
                    "head_steps": args.head_steps, "head_passes": "before_and_after_full_network",
                    "head_solver": args.head_solver, "head_rcond": args.head_rcond,
                    "head_scaling": "jacobian_column_norm",
                    "scaled_lbfgs": args.scaled_lbfgs,
                    "gauss_newton_steps": args.gauss_newton_steps, "cg_iterations": args.cg_iterations,
                    "iterations": args.iterations, "device": args.device, "cpu_threads": args.cpu_threads,
                    "blas_threads": args.cpu_threads,
                    "reference": "reset_in_current_event_coordinates", "optimizer": "restart_each_trial"}
        result["training_controls"] = settings
        while result["last_completed_round"] < args.rounds:
            if pending is None:
                index, release = accepted["round"], accepted["release"]
                if release >= 1:
                    index += 1
                    model, history, born, event = prepare_event(model, history, index)
                    release = 0.0
                    frozen = snapshot(model, history, index, release, born, event)
                    result["events"].append(event)
                else:
                    frozen = accepted
                pending = {"state": frozen, "target": min(1.0, round(release + args.release_step, 14)),
                           "anchor": accepted_weights, "network": accepted_weights}
            frozen = pending["state"]
            index, release = frozen["round"], frozen["release"]
            target = min(1.0, round(release + args.release_step, 14))
            problem = build_trial(model, network, pending, target)
            resumed_trial = target == pending["target"] and "last_relative_force" in pending
            if resumed_trial:
                with torch.no_grad():
                    restored = float(problem.residual()[0].norm() / problem.force_scale)
                result["increments"][-1]["restart_relative_force"] = restored
            pending["target"], pending["network"] = target, weights(network)
            save_checkpoint(checkpoint, accepted, accepted_weights, pending, result)
            response = lambda field: observe(problem, field, index)
            head = refine_head(problem, args.head_steps, args.tolerance, observe=response,
                               solver=args.head_solver, rcond=args.head_rcond)
            refinement = refine_network(problem, args.gauss_newton_steps, args.tolerance,
                                        observe=response, cg_iterations=args.cg_iterations)
            scale = neural_linearization(problem)[-1].rsqrt() if args.scaled_lbfgs and args.iterations else None
            u, c, record = train_increment(problem, args.iterations, args.tolerance, observe=response,
                                           parameter_scale=scale)
            finish = refine_head(problem, args.head_steps, args.tolerance, observe=response,
                                 solver=args.head_solver, rcond=args.head_rcond)
            with torch.no_grad():
                loss, force, field, exterior = problem.objective()
            u, c = field.cpu().numpy(), exterior.cpu().numpy()
            relative = float(force.norm() / problem.force_scale)
            record.update(loss=float(loss), relative_force=relative, converged=relative <= args.tolerance)
            record.update(round=index, release_fraction=target, settings=settings.copy(),
                          accepted=record["converged"] or args.accept_training_limit, material_committed=False,
                          output_layer_refinement=head,
                          output_layer_finish=finish,
                          gauss_newton_refinement=refinement, **response(u))
            record["acceptance_reason"] = ("force_tolerance" if record["converged"] else
                                           "training_limit" if record["accepted"] else "not_accepted")
            record["batches"] = batch_response(problem, u)
            internal = problem.support.K @ u[problem.support.dofs] - problem.support.h
            record["lining_internal_residual_norm"] = float(np.linalg.norm(
                internal[problem.support.i] - problem.internal_load))
            result["increments"].append(record)
            result["elapsed_s"] = previous_elapsed + time.perf_counter() - started
            print(json.dumps({key: record[key] for key in (
                "round", "release_fraction", "relative_force", "accepted",
                "face_convergence_mm", "new_lining_peak_thrust_kN_per_m")}), flush=True)
            if not record["accepted"]:
                pending["network"] = weights(network)
                pending["last_relative_force"] = record["relative_force"]
                save_checkpoint(checkpoint, accepted, accepted_weights, pending, result)
                write_json(args.output / "run.json", result)
                return 2
            if index in args.archive_rounds and target == 1.0:
                # Retain the true frozen material anchor for later training-only checks.
                pending["network"] = weights(network)
                save_checkpoint(args.output / f"round-{index:03d}-before-final.pt",
                                accepted, accepted_weights, pending, result)
            problem.commit(u, c)
            record["material_committed"] = True
            history = problem.history
            if target == 1.0:
                model.fe.f_release.fill(0)
            refresh_history(model, index, record, result["elapsed_s"])
            accepted = snapshot(model, history, index, target, frozen["born"].numpy(), frozen["event"])
            accepted_weights, pending = weights(network), None
            if target == 1.0:
                arrays = base.state_arrays(model)
                arrays["c_reduced"] = c
                arrays["reference_elements"] = np.concatenate([r.elements for r in history.records])
                arrays["reference_bias"] = np.concatenate([r.stress_bias for r in history.records])
                arrays["reference_event"] = np.concatenate([np.full(len(r.elements), r.event_id)
                                                            for r in history.records])
                base.save_npz(args.output / f"round-{index:03d}.npz",
                              {"method": METHOD, "k": index, "x_f": problem.fronts[0],
                               "info": record, "wall_s": result["elapsed_s"]}, arrays)
                result["last_completed_round"] = index
            save_checkpoint(checkpoint, accepted, accepted_weights, pending, result)
            if target == 1.0 and index in args.archive_rounds:
                save_checkpoint(args.output / f"round-{index:03d}-checkpoint.pt",
                                accepted, accepted_weights, pending, result)
            write_json(args.output / "run.json", result)
        result["completed"] = True
        result["elapsed_s"] = previous_elapsed + time.perf_counter() - started
        save_checkpoint(checkpoint, accepted, accepted_weights, pending, result)
        write_json(args.output / "run.json", result)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--round-length", type=float, default=0.4)
    parser.add_argument("--gap", type=float, default=0.6)
    parser.add_argument("--mesh", choices=("coarse", "production", "design", "interior"), default="coarse")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--face-patch-span", type=float,
                        help="Add a zero-output face window with this half-span in tunnel radii")
    parser.add_argument("--front-local", action="store_true",
                        help="Add overlapping networks with local radial and axial coordinates at the face")
    parser.add_argument("--support-patch-span", type=float,
                        help="Add a two-dimensional local window at the actual lining front")
    parser.add_argument("--support-local", action="store_true",
                        help="Resolve the annular ground around and behind the lining front")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--release-step", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--gauss-newton-steps", type=int, default=30)
    parser.add_argument("--head-steps", type=int, default=0)
    parser.add_argument("--head-solver", choices=("damped", "svd"), default="damped")
    parser.add_argument("--head-rcond", type=float, default=1e-9)
    parser.add_argument("--scaled-lbfgs", action="store_true")
    parser.add_argument("--initial-network", type=Path,
                        help="Reuse neural parameters only; all physical states start from the common initial state")
    parser.add_argument("--cg-iterations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--accept-training-limit", action="store_true",
                        help="Commit the trained field at its budget, retaining the actual force-stopping status")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true")
    restart.add_argument("--resume-from", type=Path,
                         help="Continue an explicit checkpoint in a new output directory")
    parser.add_argument("--archive-rounds", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/analysis/history_operator/short_B_g0p6_round0p4")
    args = parser.parse_args()
    if (min(args.rounds, args.width, args.cpu_threads, args.cg_iterations) < 1
            or min(args.gauss_newton_steps, args.head_steps, args.iterations) < 0
            or args.gauss_newton_steps + args.head_steps + args.iterations == 0
            or not 0 < args.release_step <= 1 or args.tolerance <= 0
            or args.round_length <= 0 or args.gap < 0
            or not 0 < args.head_rcond < 1
            or (args.support_patch_span is not None and args.support_patch_span <= 0)
            or (args.face_patch_span is not None and args.face_patch_span <= 0)):
        parser.error("Positive training and advance controls are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
