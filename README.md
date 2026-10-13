# Tunnel construction, convergence and lining thrust

Reproduction code for **Coordinating support installation and excavation rounds
to control tunnel convergence and lining thrust**, by Pengcheng Zhu and Tielin Chen.

The axisymmetric model couples a line-source and solid-harmonic elastic exterior
to a Mohr--Coulomb finite-element near zone. It advances excavation rings and
activates bonded lining stress-free at the preceding equilibrium. This release
generates the eight refined construction histories, thickness-integrated section
thrust, fixed-band convergence, paired-section B-bar strain contributions and
the allowance maps used in the construction analysis. All numerical inputs are
defined in the source; the initial equilibria and datasets are generated locally.

## Install

Tested with Python 3.13. Create a virtual environment and install the dependencies:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then run:

```text
python -m pip install -r requirements.txt
```

The solver uses CPU float64 calculations and one numerical thread. A GPU is not
required. Run one simulation process per checkout so shared initial states are
written once. The complete study can take several hours and stores several GB
of accepted displacement, stress and activation states under `results/`.

The readout tests use analytic stress and strain fields and exercise activation,
allowance boundaries and selection ties:

```text
python -B -m unittest discover -s scripts/review -p "test_*.py"
```

## Reproduce

Start with the initial equilibrium and one excavation round:

```text
python -B reproduce.py simulate --case B --gap 0.6 --round-length 0.4 --stop-after 1
```

Run all eight schedules, resuming any accepted states already present:

```text
python -B reproduce.py simulate --all
python -B reproduce.py analyse
```

Or generate one particular schedule:

```text
python -B reproduce.py simulate --case B --gap 0.7 --round-length 0.2
```

The full analysis expects all eight schedules. It writes the installation
events, fixed-band histories, section histories, paired-section strains and
construction choices to `results/analysis/lining_activation/`. Query feasible
schedules and all ties for the fewest rounds with:

```text
python -B reproduce.py select --case B --convergence-mm 10.847 --thrust-kn-per-m 1900
```

This query selects the `(gap/radius, round length/radius) = (0.7, 0.2)` schedule
with 60 rounds. With a thrust allowance of 2100 kN/m, `(0.6, 0.2)` and
`(0.7, 0.2)` tie; with 2200 kN/m, `(0.6, 0.4)` requires 30 rounds.
The original B reference convergence is approximately 10.84693364 mm.

## Model and readouts

The radius is 2 m and initial hydrostatic compression is 5 MPa. Rock elasticity
is 1800 MPa with Poisson ratio 0.30, friction 30 degrees and zero dilation.
Cohesions for B/C/D are 0.50/0.25/0.12 MPa. The 0.2 m thick lining has modulus
18000 MPa and Poisson ratio 0.20. The refined mesh halves every interval of
the production mesh, including the lining. Release increments are capped at
1/16 and the relative Newton force criterion is 1e-6.

Every schedule in a ground case starts from the same initial 10-radius
excavation with the lining front at `z/radius = -1`. The input `--gap` is the
minimum forward-edge distance **at installation**; the internal round-end
distance is `gap + round_length`. A complete batch has centre distance
`gap + round_length/2`. The compared advance ends at twelve radii, requiring
30 or 60 rounds.

Readouts cover `z/radius = [3, 6]`. Convergence is averaged over fixed 0.2-radius
bands. Section thrust integrates hoop stress through the lining thickness at
each axial Gauss position, with 60 sections on the refined mesh. Envelopes use
all saved round equilibria. Compression and inward convergence are positive
in the reported results; solver stresses are tension-positive in MPa and
displacements are in metres. Selection inputs are convergence in mm and
section hoop thrust in kN/m.

Expected rounded construction responses:

| Case | Gap/radius | Round/radius | Convergence (mm) | Peak section thrust (kN/m) |
|---|---:|---:|---:|---:|
| B | 0.6 | 0.4 | 10.847 | 2180 |
| B | 0.6 | 0.2 | 10.112 | 2033 |
| B | 0.7 | 0.2 | 10.586 | 1863 |
| B | 0.8 | 0.2 | 11.024 | 1703 |
| C | 0.6 | 0.4 | 14.471 | 2510 |
| C | 0.6 | 0.2 | 13.168 | 2336 |
| D | 0.6 | 0.4 | 19.490 | 2601 |
| D | 0.6 | 0.2 | 17.596 | 2366 |

`src/wave2_src/` contains the finite elements, exterior condensation and
excavation model; `src/axisym_src/` contains the elastic potentials;
`src/fedev_src/` contains constitutive and plane-strain reference functions.
`src/runtime.py` supplies the material return, release controller and globalized
Newton solve. `scripts/review/` contains the manuscript's response extraction
and construction-selection functions. The numerical kernels and readout
functions are exported from the research implementation; `reproduce.py`
provides a standalone run sequence without requiring prior result files.
