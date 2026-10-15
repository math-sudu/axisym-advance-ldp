# Tunnel convergence and lining thrust with FEINNs

Code and data-generation scripts for **Tunnel convergence and lining thrust
under staged excavation using physics-informed neural networks**, by
Pengcheng Zhu and Tielin Chen.

The finite element-integrated neural network (FEINN) predicts displacement
increments. Axisymmetric quadrature, a non-associated Mohr--Coulomb return
map and an elastic exterior supply the weak equilibrium residual. Each
accepted increment commits the material history. Bonded lining is activated
stress-free and subsequently responds to strain relative to installation.
The common initial equilibrium is generated with the conventional solver;
excavation and installation after that state use neural equilibrium.

## Install

Tested with Python 3.13. From this checkout:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install:

```text
python -m pip install -r requirements.txt
```

CPU is the default. `--device cuda` selects a CUDA-capable PyTorch installation.
Calculations use float64. `--cpu-threads` sets PyTorch's CPU thread count.
Run the initial-state generator once before launching histories on that mesh.
Generated states, neural weights and readouts stay under ignored `results/`.

## Check the installation

Generate a small common equilibrium and run the neural and readout tests:

```text
python -B scripts/pinn/initialize.py --mesh coarse
python -B -m unittest discover -s scripts/pinn -p "test_*.py"
python -B -m unittest discover -s scripts/review -p "test_*.py"
```

These checks exercise neural gradients, zero-stress activation, frozen material
history, thrust quadrature, installation-relative strain, and joint checkpoint
restart. The integration test deliberately accepts a small training budget to
test state transport and preserves its actual non-convergence flag.

A short standalone run exercises the same command interface as full training:

```text
python -B scripts/pinn/run_construction.py --mesh coarse --architecture face-local --reuse-increment --rounds 1 --release-step 0.5 --gauss-newton-steps 2 --cg-iterations 20 --iterations 1 --tolerance 0.0003 --accept-training-limit --archive-rounds 1 --output results/neural_smoke
python -B scripts/pinn/construction_response.py --source results/neural_smoke --cohort -1 -0.6 --face 0.4
python -B scripts/pinn/compare_construction.py --sources results/neural_smoke --cohort -1 -0.6 --face 0.4 --output results/neural_smoke/comparison.json
python -B scripts/pinn/section_strains.py --source results/neural_smoke/comparison.json --output results/neural_smoke/strains.json
```

The short run verifies execution and readout, not the manuscript's numerical
response. Repeat its training command with `--resume` to restore the joint
state. Increase `--rounds` to continue excavation. A new run needs a new output
directory; restarting an existing run requires `--resume`.

## Generate neural construction histories

The manuscript mesh has 7,995 nodes and 7,760 elements. Generate its initial
equilibrium, then train the three schedules from fresh networks:

```text
python -B scripts/pinn/initialize.py --mesh production
python -B scripts/pinn/run_construction.py --mesh production --architecture face-local --reuse-increment --rounds 30 --round-length 0.4 --gap 0.6 --release-step 0.5 --gauss-newton-steps 20 --cg-iterations 160 --iterations 0 --tolerance 0.0006 --accept-training-limit --archive-rounds 20 30 --output results/neural_g0p6_l0p4
python -B scripts/pinn/run_construction.py --mesh production --architecture face-local --reuse-increment --rounds 60 --round-length 0.2 --gap 0.7 --release-step 0.5 --gauss-newton-steps 20 --cg-iterations 160 --iterations 0 --tolerance 0.0005 --accept-training-limit --archive-rounds 16 40 60 --output results/neural_g0p7_l0p2
python -B scripts/pinn/run_construction.py --mesh production --architecture face-local --reuse-increment --rounds 60 --round-length 0.2 --gap 0.8 --release-step 0.5 --gauss-newton-steps 20 --cg-iterations 160 --iterations 0 --tolerance 0.0005 --accept-training-limit --archive-rounds 40 60 --output results/neural_g0p8_l0p2
```

Each `run.json` records accepted release fractions, force residuals, optimization
traces and physical responses. `accepted` means the material state was committed;
`converged` means the chosen force tolerance was met. `--accept-training-limit`
permits commitment at the training limit and retains both flags. Assess the
physical responses with additional training and finer releases when needed.
Omitting that flag stops with exit code 2 at an unaccepted trial; `--resume`
continues it with the original frozen material state. Full training can take
many hours; `--rounds 1` provides a shorter production-mesh run.

These commands regenerate the model and construction schedules with the final
regular training controls. The reported histories used transferred initial
weights and earlier startup controls: the long-round history began with eight
releases in its first round, the same-centre short history used eight releases
in its first two rounds, and the rearward-installation history transferred
weights from the short-round startup refinement. Thus a fresh-network run is
not a replay of those optimization trajectories, and the rounded manuscript
values below are reference results rather than an exact-output test. The code
supports `--initial-network`, incremental `--resume`, and changing subsequent
training/release controls. Generated research checkpoints are not release inputs.

## Read the physical response

After completing the three histories:

```text
python -B scripts/pinn/compare_construction.py --sources results/neural_g0p6_l0p4 results/neural_g0p7_l0p2 results/neural_g0p8_l0p2 --cohort 3 6 --face 12 --output results/representative_history_comparison.json
python -B scripts/pinn/section_strains.py --source results/representative_history_comparison.json --output results/representative_section_strains.json
```

The comparison uses the same material interval `z/a=[3,6]` in every run.
It retains fixed sections, complete installation batches and matched-centre
batch pairs. Convergence is averaged over fixed `0.2a` bands; thrust integrates
hoop stress through thickness without a radius factor. Strain decomposition
reconstructs stress from each section's actual installation displacement and
checks radial, axial and hoop contributions against the saved thrust.

Reported reference responses for the three completed neural histories:

| Gap/radius | Round/radius | Convergence envelope (mm) | Mean thrust at 12a (kN/m) | Section-thrust envelope (kN/m) |
|---:|---:|---:|---:|---:|
| 0.6 | 0.4 | 10.933 | 1618.3 | 2163.6 |
| 0.7 | 0.2 | 10.619 | 1656.1 | 1864.5 |
| 0.8 | 0.2 | 11.044 | 1514.8 | 1695.0 |

For a frozen-increment training check, retain the final-release checkpoints
with `--archive-rounds`, then run, for example:

```text
python -B scripts/pinn/refine_release.py --checkpoint results/neural_g0p7_l0p2/round-060-before-final.pt --trained-checkpoint results/neural_g0p7_l0p2/round-060-checkpoint.pt --release 1 --steps 40 --cg-iterations 160 --tolerance 0.0002 --cohort 3 6 --output results/neural_g0p7_l0p2/final_refinement
```

This saves separate refined fields and leaves the accepted history intact.
A release sensitivity run uses a separate output directory and a smaller
`--release-step`; compare its completed fields at the same face and cohort.

## Model and source

The tunnel radius is 2 m and initial hydrostatic compression is 5 MPa.
Rock modulus is 1800 MPa, Poisson ratio 0.30, cohesion 0.50 MPa, friction
30 degrees and dilation zero. Lining thickness is 0.2 m, modulus 18000 MPa
and Poisson ratio 0.20. Every schedule starts with the face at zero and
the lining front at `z/a=-1`. `--gap` is the minimum forward-edge distance
at installation; batch-centre distance is `gap + round_length/2`.
Displacements are in metres and tension-positive stresses in MPa internally;
the readouts report inward convergence and compression as positive.

`scripts/pinn/` contains the neural solver, initialization and readouts.
`src/` contains the material, finite-element and exterior kernels and the
initial-equilibrium solver. The numerical neural bodies and readout functions
are exported from the research implementation; package wiring uses local
modules and generates its initial state without the research checkout.
The retained `reproduce.py` and [earlier study commands](LEGACY.md) reproduce
the preceding conventional construction study.

The FEINN formulation follows Zhang et al.,
[CMA 433, 117474](https://doi.org/10.1016/j.cma.2024.117474) and
[IJMS 290, 110075](https://doi.org/10.1016/j.ijmecsci.2025.110075).
This implementation uses axisymmetric weak equilibrium and the stated
non-associated rock law; no displacement or stress labels enter training.
