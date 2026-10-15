"""Resolve neural lining thrust into installation-relative strain contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from construction_response import ROOT, A, read_state, section_quadrature, AxiMesh
from extract_paired_section_strains import elastic_stress, integrate_components


SOURCE = ROOT / "results/analysis/pinn_activation/representative_history_comparison.json"
OUTPUT = SOURCE.with_name("representative_section_strains.json")


def extract(response, material_positions):
    sections = response["sections"]
    peak = response["summary"]["demand_envelope"]
    high = peak["section_index"]
    batch = sections[high]["installation_batch_over_a"]
    batch_ids = [i for i, section in enumerate(sections)
                 if np.allclose(section["installation_batch_over_a"], batch)]
    final_thrust = response["history"][-1]["hoop_thrust_kN_per_m"]
    low = min(batch_ids, key=lambda i: final_thrust[i])
    common = [next(i for i, section in enumerate(sections)
                   if np.isclose(section["z_over_a"], z, rtol=0, atol=1e-10))
              for z in material_positions]
    selected = sorted(set([low, high, *common]))

    initial = read_state(ROOT / response["controls"]["initial_state"])
    lo, hi = initial["liner_e_z0"] / A, initial["liner_e_z1"] / A
    left, right = response["cohort_over_a"]
    inside = (lo >= left - 1e-10) & (hi <= right + 1e-10)
    native = np.unique(np.column_stack([lo[inside], hi[inside]]), axis=0)
    quadrature = section_quadrature(initial, native)
    np.testing.assert_allclose([q[0] for q in quadrature],
                               [s["z_over_a"] for s in sections], atol=1e-10)
    elements = np.array([q[1] for q in quadrature])
    gauss = np.array([q[2] for q in quadrature])
    weights = np.array([q[3] for q in quadrature])
    thickness = weights.sum(axis=1)
    mesh = AxiMesh(initial["rho_lines_m"], initial["z_lines_m"]).build_B()
    physical_elements = initial["liner_elems"][elements]
    operator = mesh.B_all[physical_elements, gauss]
    dofs = mesh.dof_all[physical_elements]
    references = np.array([s["reference_step"] for s in sections])
    reference_u = np.zeros(dofs.shape)
    histories = {i: [] for i in selected}
    maximum_stress_difference = 0.0
    maximum_thrust_difference = 0.0

    for row in response["history"]:
        if row["step"] < references.min():
            continue
        state = read_state(ROOT / row["source"])
        current_u = state["displacement_m"][dofs]
        newly = references == row["step"]
        reference_u[newly] = current_u[newly]
        strain = np.einsum("sgij,sgj->sgi", operator, current_u - reference_u)
        components = integrate_components(strain, weights)
        loaded = row["step"] > references
        stored = state["liner_sig"][elements, gauss]
        if loaded.any():
            rebuilt = elastic_stress(strain)
            np.testing.assert_allclose(rebuilt[loaded], stored[loaded], rtol=1e-10, atol=1e-9)
            thrust = np.array(row["hoop_thrust_kN_per_m"], dtype=float)
            np.testing.assert_allclose(components[loaded].sum(axis=1), thrust[loaded],
                                       rtol=1e-10, atol=1e-8)
            maximum_stress_difference = max(maximum_stress_difference,
                                            float(np.abs(rebuilt[loaded] - stored[loaded]).max()))
            maximum_thrust_difference = max(maximum_thrust_difference,
                                            float(np.abs(components[loaded].sum(axis=1) - thrust[loaded]).max()))
        average_strain = np.einsum("sg,sgc->sc", weights, strain) / thickness[:, None]
        for i in selected:
            if row["step"] >= references[i]:
                histories[i].append({
                    "face_over_a": row["face_over_a"],
                    "post_installation_convergence_mm": row["wall_convergence_mm"][i]
                    - sections[i]["installation_convergence_mm"],
                    "average_strain_rr_zz_tt_rz": average_strain[i].tolist(),
                    "thrust_components_rr_zz_tt_kN_per_m": components[i].tolist(),
                    "thrust_kN_per_m": float(components[i].sum()),
                })

    return {
        "source_directory": response["source_directory"],
        "controls": response["controls"],
        "cohort_over_a": response["cohort_over_a"],
        "evaluation_face_over_a": response["evaluation_face_over_a"],
        "controlling_batch_over_a": batch,
        "low_index": low, "high_index": high, "fixed_material_indices": common,
        "selected_sections": [{"section_index": i, **sections[i], "history": histories[i]}
                              for i in selected],
        "final_sections": [{"section_index": i, "z_over_a": section["z_over_a"],
                            "installation_batch_over_a": section["installation_batch_over_a"],
                            "average_strain_rr_zz_tt_rz": average_strain[i].tolist(),
                            "thrust_components_rr_zz_tt_kN_per_m": components[i].tolist()}
                           for i, section in enumerate(sections)],
        "maximum_stress_identity_difference_MPa": maximum_stress_difference,
        "maximum_thrust_identity_difference_kN_per_m": maximum_thrust_difference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    comparison = json.loads(args.source.read_text(encoding="utf-8"))
    reference = comparison["responses"][0]
    high = reference["summary"]["demand_envelope"]["section_index"]
    batch = reference["sections"][high]["installation_batch_over_a"]
    batch_ids = [i for i, s in enumerate(reference["sections"])
                 if s["installation_batch_over_a"] == batch]
    low = min(batch_ids, key=lambda i: reference["history"][-1]["hoop_thrust_kN_per_m"][i])
    positions = [reference["sections"][i]["z_over_a"] for i in (low, high)]
    result = {"source": args.source.resolve().relative_to(ROOT).as_posix(),
              "fixed_material_positions_over_a": positions,
              "responses": [extract(r, positions) for r in comparison["responses"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in result["responses"]:
        print(json.dumps({k: r[k] for k in ("source_directory", "controlling_batch_over_a",
                         "maximum_stress_identity_difference_MPa",
                         "maximum_thrust_identity_difference_kN_per_m")}))


if __name__ == "__main__":
    main()
