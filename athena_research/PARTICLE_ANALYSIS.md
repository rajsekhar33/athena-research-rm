# Particle Analysis in athena_research

This note describes the reusable particle-analysis workflow in `athena_research`.
It is intentionally kept at the package level, beside `operations_tests.ipynb`, rather
than inside `problem_plotting`. The `problem_plotting` directory should hold scripts that
are specific to one science problem; particle loading, deposition, and generic diagnostic
plots belong in reusable core utilities and package-level notebooks.

## Current Layout

```text
athena_research/
├── PARTICLE_ANALYSIS.md
├── particle_analysis_tests.ipynb
├── operations_tests.ipynb
└── core/
    ├── athena_data.py
    └── particle_io.py
```

The currently available particle entry points are:

- `athena_research.core.particle_io.read_particle_binary`
- `athena_research.core.particle_io.read_particle_binary_header`
- `athena_research.core.particle_io.read_particle_binary_positions`
- `athena_research.core.particle_io.read_particle_hdf5`
- `athena_research.core.particle_io.read_particle_vtk`
- `athena_research.core.particle_io.particle_column_xy`
- `athena_research.core.particle_io.particle_density_grid`
- `athena_research.core.particle_io.particle_profile_x`
- `athena_research.core.particle_io.wrap_particle_positions`
- `AthenaData.load_particles(...)`
- `AthenaData.get_particle_data(...)`

The package-level notebook `particle_analysis_tests.ipynb` shows how to use these helpers
for quick particle diagnostics and for gas/tracer comparison plots.

## AthenaK Outputs

For AthenaK Lagrangian tracer tests, write paired grid and particle outputs:

```ini
<output1>
file_type = bin
variable  = hydro_w
dt        = ...

<output2>
file_type = pbin
variable  = hydro_w
dt        = ...

<output3>
file_type = rst
dt        = ...

<output4>
file_type = prst
dt        = ...
```

Use `pbin`, not `bin_prtcl`, for the current AthenaK particle binary writer. The `.prtclbin`
file stores particle positions, particle integer fields, and the grid variables requested by
the `variable` line.

## Loading a Particle File Directly

```python
from pathlib import Path
from athena_research.core.particle_io import read_particle_binary

pfile = Path("/path/to/your/simulation/runs/square_ito/pbin")
pfile = sorted(pfile.glob("*.prtclbin"))[-1]

particles = read_particle_binary(pfile)
print(particles["time"], particles["ncycle"], particles["nparticles"])
print(particles.keys())
```

The direct reader returns a dictionary with standard fields:

- `time`, `dt`, `ncycle`, `nparticles`
- `nrdata`, `nidata`, `ngriddata`, `var_names`
- `x`, `y`, `z`
- `velx`, `vely`, `velz` when those real slots are written
- `gid`, `tag`, `lastmove`, `lastlevel`
- one array for each sampled grid variable in `var_names`

For very large particle files, avoid reading more arrays than the analysis needs:

```python
from athena_research.core.particle_io import read_particle_binary_positions

positions = read_particle_binary_positions(pfile)
```

This returns the same header metadata plus only `x`, `y`, and `z`.

## Loading Through AthenaData

```python
from athena_research.core.athena_data import AthenaData

ad = AthenaData(num=5)
ad.load_particles(
    filedir="/path/to/your/simulation/runs/square_ito/pbin",
    prefix="ItoSquare.hydro_w",
    suffix="prtclbin",
)

x = ad.get_particle_data("x")
y = ad.get_particle_data("y")
dens = ad.get_particle_data("dens")
```

`get_particle_data()` first checks values stored in the particle output. It also provides
simple derived coordinates such as `r`, `R`, `theta`, and `phi`. If a grid quantity was not
included in the `.prtclbin` file, regenerate the particle output with the needed `variable`
setting rather than relying on placeholder fallback values.

## Generic Particle Plots

Generic particle plots should use reusable functions outside `problem_plotting`. For example:

```python
import matplotlib.pyplot as plt
from athena_research.core.particle_io import particle_column_xy

column = particle_column_xy(particles, grid)
plt.imshow(
    column,
    origin="lower",
    extent=(grid["x1min"], grid["x1max"], grid["x2min"], grid["x2max"]),
)
plt.colorbar(label="normalized particle column")
```

Problem-specific scripts can import the same core readers, but they should live in a
problem-named script or package. The generic examples and tests stay here at the package
level.

## Ito Tracer Comparison Workflow

For the Ito tracer validation runs, the useful reusable steps are:

1. Locate the latest grid `.bin` and particle `.prtclbin` files for each method.
2. Read the gas grid with `athena_research.core.io_utils.read_binary`.
3. Read particle data with `athena_research.core.particle_io`; use
   `read_particle_binary_positions` for large outputs when only positions are needed.
4. Deposit particle positions onto the same grid used by the gas data.
5. Compare classical, Ito-2, and Lagrangian MC tracers with:
   - square-wave gas and tracer density profiles;
   - tracer/gas residuals;
   - turbulence gas and tracer columns;
   - joint gas/tracer histograms;
   - log10 tracer/gas ratio PDFs;
   - density power spectra.

The companion notebook gives a minimal, editable version of these steps. A more polished
paper-comparison plotting script should import the same reusable helpers rather than
duplicating particle readers or deposition code.

## Restart-Aware Analysis

For restart validation, compare matching `rst` and `prst` cycles. The particle restart file
contains the tracer positions and integer state needed to continue the stochastic pushers;
the analysis `.prtclbin` file is for diagnostics and plotting, not for restarting a run.

Use matching output numbers when comparing gas and particle diagnostics:

```text
bin/Problem.hydro_w.00005.bin
pbin/Problem.hydro_w.00005.prtclbin
rst/Problem.00005.rst
prst/Problem.00005.prtclrst
```
