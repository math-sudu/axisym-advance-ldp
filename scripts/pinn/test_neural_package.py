"""Exercise standalone construction, physical readout and checkpoint restart."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from run_construction import ROOT, run, torch
from compare_construction import compare
from construction_response import analyse
from section_strains import extract
from refine_release import refine


class NeuralPackageTests(unittest.TestCase):
    def test_training_limit_readout_and_restart(self):
        with tempfile.TemporaryDirectory(prefix="neural-test-", dir=ROOT / "results") as directory:
            for accept in (False, True):
                output = Path(directory) / str(accept)
                args = SimpleNamespace(
                    mesh="coarse", round_length=0.4, gap=0.6, rounds=1,
                    architecture="face-local", device="cpu", cpu_threads=1, initial_network=None,
                    resume=False, reuse_increment=True, predict_increment=False,
                    head_steps=0, gauss_newton_steps=0, cg_iterations=80,
                    release_step=0.5, iterations=1, tolerance=1e-12,
                    archive_rounds=[1], accept_training_limit=accept, output=output)
                self.assertEqual(run(args), 0 if accept else 2)
                saved = torch.load(output / "checkpoint.pt", weights_only=True)
                record = saved["run"]["increments"][-1]
                self.assertFalse(record["converged"])
                self.assertEqual(record["accepted"], accept)
                self.assertEqual(saved["release"], 1.0 if accept else 0.0)
                if not accept:
                    self.assertFalse(bool(torch.any(saved["arrays"]["liner_sig"][saved["born"]])))
                    continue
                response = analyse(output, (-1.0, -0.6), face=0.4)
                self.assertEqual(response["history"][0]["mean_hoop_thrust_kN_per_m"], 0)
                self.assertLess(response["summary"]["maximum_strain_stress_identity_difference_MPa"], 1e-10)
                positions = [s["z_over_a"] for s in response["sections"]]
                components = extract(response, [positions[0], positions[-1]])
                self.assertLess(components["maximum_thrust_identity_difference_kN_per_m"], 1e-9)
                comparison = compare([output], (-1.0, -0.6), face=0.4)
                json.dumps(comparison, allow_nan=False)
                self.assertTrue((output / "round-001-before-final.pt").exists())
                self.assertTrue((output / "round-001-checkpoint.pt").exists())
                refinement = output / "refinement"
                refine(SimpleNamespace(
                    checkpoint=output / "round-001-before-final.pt",
                    trained_checkpoint=output / "round-001-checkpoint.pt",
                    release=1.0, steps=0, tolerance=0.0002, cg_iterations=20,
                    device="cpu", cpu_threads=1, cohort=(-1.0, -0.6), output=refinement))
                refined = json.loads(refinement.with_suffix(".json").read_text())
                self.assertEqual(refined["maximum_displacement_change_mm"], 0)
                self.assertAlmostEqual(refined["before"]["cohort_response"]["lining_mean_thrust_kN_per_m"],
                                       response["summary"]["final_mean_thrust_kN_per_m"], places=9)
                args.resume = True
                self.assertEqual(run(args), 0)
                restored = torch.load(output / "checkpoint.pt", weights_only=True)
                torch.testing.assert_close(restored["arrays"]["displacement_m"],
                                           saved["arrays"]["displacement_m"], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
