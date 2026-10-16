# Tunnel convergence and lining thrust with neural fields

Code and data-generation scripts for **Tunnel convergence and lining thrust
under staged excavation using physics-informed neural fields**, by
Pengcheng Zhu and Tielin Chen.

The neural field represents ground displacement increments. Complete
installation references enter an affine support operator: the connected
elastic lining is condensed, and its displacement and stress are recovered
during each trial. Axisymmetric weak equilibrium, non-associated
Mohr--Coulomb integration and an elastic exterior govern training. Material
and installation histories change only when an increment is accepted.

## Install and generate the physical initial state

Tested with Python 3.13. Create and activate a virtual environment, then install:

~~~text
python -m venv .venv
python -m pip install -r requirements.txt
~~~

Between these commands, activate with `.venv\Scripts\Activate.ps1` in
PowerShell or `source .venv/bin/activate` on Linux/macOS. CPU is the default;
`--device cuda` requires a CUDA-capable PyTorch installation. Training uses
float64. `--cpu-threads` controls the CPU and BLAS thread budget.

Generate each required common equilibrium before starting a history:

~~~text
python -B scripts/pinn/initialize.py --mesh coarse
python -B scripts/pinn/initialize.py --mesh production
python -B scripts/pinn/initialize.py --mesh design
~~~

The conventional initializer supplies only the common physical state, with
the face at zero and the lining front at `z/a=-1`. Subsequent excavation and
installation use the neural solver. `production` has 7,995 nodes and 7,760
elements, with `0.1a` axial spacing. `design` uses `0.05a` axial spacing and
the same radial grid. `coarse` is useful for installation checks; its `0.2a`
spacing cannot represent the `0.7a` installation gap. Generated inputs,
fields and checkpoints stay under ignored `results/`.

## Check execution and physical readout

The first two initial states above are required by the tests:

~~~text
python -B -m unittest discover -s scripts/pinn -p "test_*.py"
python -B -m unittest discover -s scripts/review -p "test_*.py"
~~~

Tests cover same-field support assembly, complete references, zero-stress
installation, neural derivatives, frozen material history, acceptance and
restart, thrust integration and strain contributions. A small two-round
run also exercises the command-line interface:

~~~text
python -B scripts/pinn/run_history_construction.py --mesh coarse --width 6 --rounds 2 --face-patch-span 0.2 --front-local --support-patch-span 0.4 --support-local --release-step 0.5 --head-steps 1 --head-solver svd --head-rcond 1e-11 --gauss-newton-steps 0 --iterations 0 --tolerance 1e-12 --accept-training-limit --archive-rounds 2 --output results/history_smoke
python -B scripts/pinn/construction_response.py --source results/history_smoke --cohort -1 -0.2 --face 0.8
python -B scripts/pinn/compare_construction.py --sources results/history_smoke --cohort -1 -0.2 --face 0.8 --output results/history_smoke/comparison.json
~~~

This short run tests execution with deliberate training-budget acceptance.
`accepted` records physical commitment; `converged` records whether the force
stopping setting was met. Without `--accept-training-limit`, a failed trial
returns exit code 2 and preserves the last accepted material state. Repeat a
training command with `--resume` to restore its state, or increase `--rounds`
to continue it. Use a new output directory for a new run. `--resume-from`
branches from an explicit saved checkpoint and retains its parent-field link.

## Train the three construction schedules from scratch

These commands start with the same seeded, untrained network on each mesh.
They use the final representation and regular controls of the study:

~~~text
python -B scripts/pinn/run_history_construction.py --mesh production --rounds 15 --round-length 0.4 --gap 0.6 --face-patch-span 0.2 --front-local --support-patch-span 0.4 --support-local --release-step 0.5 --head-steps 6 --head-solver svd --head-rcond 1e-11 --gauss-newton-steps 0 --iterations 0 --tolerance 0.0006 --accept-training-limit --cpu-threads 8 --archive-rounds 10 12 15 --output results/history_production_g0p6_l0p4
python -B scripts/pinn/run_history_construction.py --mesh production --rounds 30 --round-length 0.2 --gap 0.7 --face-patch-span 0.2 --front-local --support-patch-span 0.4 --support-local --release-step 0.5 --head-steps 6 --head-solver svd --head-rcond 1e-11 --gauss-newton-steps 0 --iterations 0 --tolerance 0.0004 --accept-training-limit --cpu-threads 8 --archive-rounds 20 24 30 --output results/history_production_g0p7_l0p2
python -B scripts/pinn/run_history_construction.py --mesh production --rounds 30 --round-length 0.2 --gap 0.8 --face-patch-span 0.2 --front-local --support-patch-span 0.4 --support-local --release-step 0.5 --head-steps 6 --head-solver svd --head-rcond 1e-11 --gauss-newton-steps 0 --iterations 0 --tolerance 0.0004 --accept-training-limit --cpu-threads 8 --archive-rounds 20 24 30 --output results/history_production_g0p8_l0p2
~~~

Repeat with `--mesh design` and `history_design_...` output names for the
paired grid. These are substantial training jobs; `--rounds 1` exercises a
shorter run. `--head-steps` applies before and after the full-network stage.
Every attempt records its controls, force residual, acceptance and response
in `run.json`. Assess a new run's physical responses with training, release
and spatial checks appropriate to its conclusions.

### Initialization of the reported histories

The six reported histories used transferred neural parameters, separately
from their regenerated physical initial states. The transfer chain was:

1. A two-round coarse history trained a width-32, seed-17 network using
   Gauss--Newton and L-BFGS, with half releases and a `5e-4` force setting.
2. Its parameters initialized a production-grid startup trial. Face and
   support-front windows were added during that diagnostic, preserving the
   existing field. The shared parameter source was its **unaccepted** trial
   at round 1, release fraction 1, with a `0.2a` face window, local face
   coordinates and a `0.4a` support-front window.
3. All six final histories restored their own common physical initial state,
   loaded these same parameters and added zero-output annular branches.
   No material state or installation reference was transferred from the trial.

The research source is `control_B_g0p6_round0p4_production_local/checkpoint.pt`,
from which the parameter-only `spatial_initial_network.pt` was extracted.
These generated weights and the original fields are not included in this
source release. `--initial-network` accepts a locally generated history
checkpoint and transfers its parameters alone; `--resume` restores the full
physical and neural state. Seed 17 alone does not reconstruct the trained source.

The production long-round history initially used `3e-4`, accepted its second
round's final trial after a resume at `3.2e-4`, then used `4e-4` and `5e-4`
before the final `6e-4` setting. Early production steps used six SVD head
updates on each side of up to eight full-network updates, with `rcond=1e-9`.
The final `rcond=1e-11`, head-only controls began at round 6/release 1 for
the long schedule, round 11/release 1 for `(0.7,0.2)` and round 6/release 1
for `(0.8,0.2)`. The design histories used the final head-only controls
throughout, with CPU thread counts changing from 6 to 4 to 8. The SVD
implementation also changed during development. Original attempts and
controls remain in each research `run.json`.

Consequently, the fresh commands above reproduce the model and schedules,
not the historical optimization trajectories or the exact table values.
The table below reports the original six histories. The released readout
functions can analyse either newly generated fields or those original fields
when provided locally in their recorded directory structure.

## Read distributions, complete batches and paired spatial effects

After completing each set of three histories:

~~~text
python -B scripts/pinn/compare_construction.py --sources results/history_production_g0p6_l0p4 results/history_production_g0p7_l0p2 results/history_production_g0p8_l0p2 --cohort 1.4 3.4 --face 6 --output results/production_comparison.json
python -B scripts/pinn/compare_construction.py --sources results/history_design_g0p6_l0p4 results/history_design_g0p7_l0p2 results/history_design_g0p8_l0p2 --cohort 1.4 3.4 --face 6 --output results/design_comparison.json
python -B scripts/pinn/compare_paired_space.py --reference results/production_comparison.json --candidate results/design_comparison.json --output results/paired_comparison.json
~~~

The common material interval is `z/a=[1.4,3.4]`. Readout reconstructs lining
stress from the **complete** saved reference biases and checks against the
stored stress. Each response includes radial, axial and hoop thrust
contributions in `history`, fixed material sections, `0.2a` convergence
bands and complete batches with their actual bounds. An interval peak can
belong to a boundary-intersecting batch; complete-batch maxima use the full
batch. Paired profiles interpolate only inside each batch and both sampled
extents. The older `section_strains.py` serves the preceding FEINN method;
use the history response's embedded components for the current method.

Original finite-advance results at `6a`:

| Mesh | `(gap/a, round/a)` | Convergence envelope (mm) | Mean thrust (kN/m) | Peak thrust (kN/m) |
|---|---|---:|---:|---:|
| production | (0.6, 0.4) | 10.89048 | 1503.432 | 2095.533 |
| production | (0.7, 0.2) | 10.41434 | 1528.558 | 1759.680 |
| production | (0.8, 0.2) | 10.85777 | 1393.335 | 1615.281 |
| design | (0.6, 0.4) | 10.12410 | 1515.981 | 2161.808 |
| design | (0.7, 0.2) | 10.13305 | 1515.943 | 1826.891 |
| design | (0.8, 0.2) | 10.66540 | 1382.003 | 1661.607 |

The short-round peak reduction and rearward-installation trade-off persist
on both grids. Small shortening-induced convergence and mean-load changes
do not retain their signs. The interval still loads between `4.8a` and `6a`.

For a frozen final-release training check, using the archived checkpoints:

~~~text
python -B scripts/pinn/refine_history.py --checkpoint results/history_production_g0p7_l0p2/round-030-before-final.pt --accepted-checkpoint results/history_production_g0p7_l0p2/round-030-checkpoint.pt --head-steps 4 --head-solver svd --head-rcond 1e-12 --gauss-newton-steps 4 --cg-iterations 100 --iterations 0 --tolerance 0.00001 --cpu-threads 8 --output results/history_production_g0p7_l0p2/final_refinement
~~~

This writes separate refined fields and leaves the accepted history intact.
`compare_history_controls.py` compares accepted checkpoint fields at identical
fronts. A release check uses a separate history with a smaller `--release-step`.

## Model and source

Tunnel radius is 2 m; initial hydrostatic compression is 5 MPa. Rock modulus
is 1800 MPa, Poisson ratio 0.30, cohesion 0.50 MPa, friction 30 degrees and
dilation zero. Lining thickness is 0.2 m, modulus 18000 MPa and Poisson ratio
0.20. `--gap` is the minimum forward-edge distance at installation; the
batch-centre distance is `gap + round_length/2`. Internal displacements are
metres and tension-positive stresses are MPa; readouts use inward convergence
and compressive thrust as positive.

`scripts/pinn/` contains the neural methods, initializer and readouts. `src/`
contains material, finite-element and exterior kernels. Numerical function
bodies are exported from the research implementation; module wiring uses
only this package. Existing `run_construction.py` and `refine_release.py`
remain the earlier FEINN implementation. `reproduce.py` and [LEGACY.md](LEGACY.md)
serve the preceding conventional study.

Weak neural equilibrium follows Zhang et al.,
[CMA 433, 117474](https://doi.org/10.1016/j.cma.2024.117474) and
[IJMS 290, 110075](https://doi.org/10.1016/j.ijmecsci.2025.110075).
No displacement or stress labels enter training. The current implementation
retains each lining batch's complete installation reference while coupling
the connected support to the neural ground increments.
