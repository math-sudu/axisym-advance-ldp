"""Recover activation-relative B-bar strains for the refined construction schemes.

Select the high and low sections inside each actual controlling batch. Also
follow the long-round pair at the same material coordinates in every scheme.
The saved stress is an independent check on the reference and strain operator.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from compare_construction_schemes import SECTION_SOURCE, SCHEDULES, select_runs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wave2_src.axifem import AxiMesh
from wave2_src.scenario import A, E_L, NU_L

OUT = SECTION_SOURCE.with_name("paired_section_strains.json")
LAM = E_L * NU_L / ((1 + NU_L) * (1 - 2 * NU_L))
MU = E_L / (2 * (1 + NU_L))
HOOP_COEFFICIENTS = np.array([LAM, LAM, LAM + 2 * MU])


def control_pair(run):
    """Find a low section in the peak's own batch at the peak's own state."""
    peak = run["summary"]["demand_envelope"]
    high = peak["section_index"]
    batch = run["sections"][high]["installation_batch_over_a"]
    indices = [i for i, section in enumerate(run["sections"])
               if section["installation_batch_over_a"] == batch]
    low = min(indices, key=lambda i: run["history"][peak["step"]]["hoop_thrust_kN_per_m"][i])
    return {"low_index": low, "high_index": high, "batch_indices": indices,
            "selection_face_over_a": peak["face_over_a"]}


def elastic_stress(strain):
    stress = np.empty_like(strain)
    stress[..., :3] = LAM * strain[..., :3].sum(axis=-1, keepdims=True) + 2 * MU * strain[..., :3]
    stress[..., 3] = MU * strain[..., 3]
    return stress


def integrate_components(strain, weights):
    """Signed rr, zz, tt contributions to compressive-positive hoop thrust."""
    return -1000 * np.einsum("sg,sgc,c->sc", weights, strain[..., :3], HOOP_COEFFICIENTS)


def extract_run(run, material_pair_z):
    sections = run["sections"]
    pair = control_pair(run)
    common_pair = [next(i for i, s in enumerate(sections)
                        if np.isclose(s["z_over_a"], z, rtol=0, atol=1e-10))
                   for z in material_pair_z]
    selected = sorted(set([pair["low_index"], pair["high_index"], *common_pair]))
    elements = np.array([s["element_indices"] for s in sections])
    gp = np.array([s["gauss_indices"] for s in sections])
    weights = np.array([s["radial_weights_m"] for s in sections])
    thickness = weights.sum(axis=1)
    references = np.array([s["reference_step"] for s in sections])
    installation_faces = np.array([s["installation_face_over_a"] for s in sections])
    installation_u = np.array([s["installation_convergence_mm"] for s in sections])
    with np.load(ROOT / run["initial_state"], allow_pickle=False) as state:
        mesh = AxiMesh(state["rho_lines_m"], state["z_lines_m"]).build_B()
        liner_elements = state["liner_elems"]
        np.testing.assert_allclose(mesh.gp_z[liner_elements], state["liner_gp_z"], rtol=0, atol=1e-12)
        for suffix in ("rho0", "rho1", "z0", "z1"):
            np.testing.assert_array_equal(getattr(mesh, "e_" + suffix)[liner_elements],
                                          state["liner_e_" + suffix])
    B = mesh.B_all[liner_elements[elements], gp]
    dofs = mesh.dof_all[liner_elements[elements]]
    reference_u = np.full(dofs.shape, np.nan)
    reference_strains = np.full((*dofs.shape[:2], 4), np.nan)
    histories = {i: [] for i in selected}
    max_stress_difference = 0.0
    max_thrust_difference = 0.0
    final = None
    for row in run["history"]:
        step = row["step"]
        if step < references.min():
            continue
        with np.load(ROOT / row["source"], allow_pickle=False) as state:
            u = state["displacement_m"][dofs]
            newly = references == step
            for i in np.flatnonzero(newly):
                if sections[i]["reference_state"] != row["source"]:
                    raise ValueError("Section reference is not the actual activation checkpoint")
            reference_u[newly] = u[newly]
            reference_strains[newly] = np.einsum("sgij,sgj->sgi", B[newly], u[newly])
            installed = step >= references
            np.testing.assert_array_equal(state["liner_active"][elements],
                                          np.broadcast_to((step > references)[:, None], elements.shape))
            strain = np.einsum("sgij,sgj->sgi", B, u - reference_u)
            stress = elastic_stress(strain)
            stored = state["liner_sig"][elements, gp]
            # This algebraic identity checks all four components, including shear.
            np.testing.assert_allclose(stress[installed], stored[installed], rtol=1e-10, atol=1e-10)
            max_stress_difference = max(max_stress_difference,
                                       float(np.abs(stress[installed] - stored[installed]).max()))
        components = integrate_components(strain, weights)
        thrust = components.sum(axis=1)
        saved_thrust = np.asarray(row["hoop_thrust_kN_per_m"], dtype=float)
        np.testing.assert_allclose(thrust[installed], saved_thrust[installed], rtol=1e-10, atol=1e-8)
        max_thrust_difference = max(max_thrust_difference,
                                   float(np.abs(thrust[installed] - saved_thrust[installed]).max()))
        mean_strain = np.einsum("sg,sgc->sc", weights, strain) / thickness[:, None]
        post_u = np.asarray(row["wall_convergence_mm"]) - installation_u
        records = [{"step": step, "face_over_a": row["face_over_a"],
                    "advance_since_installation_over_a": float(row["face_over_a"] - installation_faces[i]),
                    "wall_convergence_mm": row["wall_convergence_mm"][i],
                    "post_installation_convergence_mm": float(post_u[i]),
                    "mean_effective_strain": mean_strain[i].tolist(),
                    "thrust_components_kN_per_m": components[i].tolist(),
                    "hoop_thrust_kN_per_m": float(thrust[i])}
                   for i in range(len(sections))]
        for i in selected:
            if installed[i]:
                histories[i].append(records[i])
        final = records
    mean_reference = np.einsum("sg,sgc->sc", weights, reference_strains) / thickness[:, None]
    lo, hi = pair["low_index"], pair["high_index"]
    result_sections = []
    for i, section in enumerate(sections):
        result_sections.append({
            "section_index": i,
            **{key: section[key] for key in (
                "z_over_a", "native_band_over_a", "installation_batch_over_a", "reference_step",
                "reference_state", "installation_face_over_a", "installation_distance_over_a",
                "installation_convergence_mm")},
            "mean_reference_strain": mean_reference[i].tolist(),
            "final": final[i],
        })
    return {
        **{key: run[key] for key in ("case", "mesh_level", "installation_target_over_a",
                                   "round_length_over_a", "source_directory", "initial_state")},
        "control_pair": pair,
        "common_material_pair": {"low_index_in_long_scheme": common_pair[0],
                                 "high_index_in_long_scheme": common_pair[1]},
        "sections": result_sections,
        "histories": [{"section_index": i, "records": histories[i]} for i in selected],
        "summary": {
            "final_high_minus_low_thrust_kN_per_m": final[hi]["hoop_thrust_kN_per_m"] - final[lo]["hoop_thrust_kN_per_m"],
            "final_high_minus_low_components_kN_per_m":
                (np.array(final[hi]["thrust_components_kN_per_m"]) - final[lo]["thrust_components_kN_per_m"]).tolist(),
            "maximum_stress_reconstruction_difference_MPa": max_stress_difference,
            "maximum_thrust_reconstruction_difference_kN_per_m": max_thrust_difference,
        },
    }


def build(data, schedules=SCHEDULES, section_path=SECTION_SOURCE):
    runs = []
    for case in schedules:
        source_runs = select_runs(data, case, schedules)
        baseline = source_runs[0]
        pair = control_pair(baseline)
        pair_z = [baseline["sections"][pair[key]]["z_over_a"] for key in ("low_index", "high_index")]
        for run in source_runs:
            result = extract_run(run, pair_z)
            runs.append(result)
            print(json.dumps({"case": case, "target": run["installation_target_over_a"],
                              "round": run["round_length_over_a"], **result["summary"]}), flush=True)
    return {
        "section_history_source": section_path.relative_to(ROOT).as_posix(),
        "operator_source": "src/wave2_src/axifem.py",
        "radius_m": A, "lining_modulus_MPa": E_L, "lining_poisson_ratio": NU_L,
        "lambda_MPa": LAM, "mu_MPa": MU,
        "strain_components": ["rr", "zz", "tt", "engineering_rz"],
        "thrust_components": ["rr", "zz", "tt"],
        "definitions": {
            "effective_strain": "B_all @ (displacement - own activation-reference displacement), tension positive.",
            "reference_strain": "B_all @ own activation-reference displacement; a reference offset, not elastic strain in an active lining.",
            "mean_strain": "Thickness average at the fixed axial Gauss section, with the section-history radial weights.",
            "thrust_components": "-1000 integral [lambda*deps_rr, lambda*deps_zz, (lambda+2mu)*deps_tt] dr in kN/m; compressive thrust positive.",
            "control_pair": "Peak section and lowest-thrust section of the same actual installation batch at the peak's saved state.",
            "common_material_pair": "The long-round control pair's physical z coordinates followed in each scheme with its own activation reference.",
        },
        "runs": runs,
    }


if __name__ == "__main__":
    result = build(json.loads(SECTION_SOURCE.read_text(encoding="utf-8")))
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
