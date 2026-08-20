# AthenaResearchRM

This Python package provides tools for analyzing data from Athenak astrophysical fluid dynamics simulations. It was originally built as a fork of Minghao Guo's Athenakit (https://github.com/mh-guo/AthenaKit), but has diverged significantly with new features and optimizations.

`AthenaData` is the central analysis container on this branch: it holds raw meshblock data, exposes derived quantities through `ad.data(...)`, and stores cached post-processing products such as `ad.dist`, `ad.dist2d`, `ad.vert`, `ad.rad`, `ad.spectra`, and `ad.sf`. Saving to `.h5data` writes these cached analysis products without duplicating the full raw Athena snapshot arrays.

## New: CPU/GPU and MPI Support

**Version 0.2.0** introduces comprehensive support for:
- **GPU acceleration** with CuPy (automatic CUDA kernel optimization)
- **CPU acceleration** with Numba JIT compilation
- **MPI distributed computing** for all major operations
- **Multi-GPU** computation on single nodes
- **Seamless backend switching** between CPU and GPU
- **Custom CUDA kernels** with CPU analogs
- **Particle I/O** for reading particle data from simulations
- **Backward compatible** - existing code works without changes

**New MPI support for all operations:**
- Basic Operations: `calc_sum(ad, varl=['dens'], use_mpi=True)`
- Histograms: `set_dist(ad, varl=['dens'], use_mpi=True)`
- Structure Functions: `set_sf(ad, varl=['velx','vely','velz'], use_mpi=True)`
- Power Spectra: `set_spectrum(ad, varl=['velx','vely','velz'], use_mpi=True)`
- Profiles: `set_vertical(ad, varl=['dens'], use_mpi=True)` and `set_radial(ad, varl=['dens'], use_mpi=True)`

**New particle I/O capabilities:**
- Read binary particle files: `read_particle_binary('particles.prtclbin')`
- Read HDF5 particle files: `read_particle_h5('particles.h5')`
- Read VTK particle files: `read_particle_vtk('particles.vtk')`

**Testing:**
- Interactive notebooks: [athena_research/operations_tests.ipynb](athena_research/operations_tests.ipynb), [athena_research/particle_analysis_tests.ipynb](athena_research/particle_analysis_tests.ipynb)

See [QUICKSTART.md](QUICKSTART.md) and [examples/README.md](examples/README.md) for the currently maintained usage guides in this checkout.

## Installation

### Minimal Base Install
```bash
git clone https://github.com/rajsekhar33/athena-research-rm.git
cd athena-research-rm
python -m pip install -e .
```

The base install now includes the packages required for the core `AthenaData` workflow and the standard operations layer: `numpy`, `matplotlib`, `h5py`, `scipy`, `psutil`, `astropy`, and `scikit-image`.

### Optional Extras

CPU acceleration with Numba:
```bash
python -m pip install -e .[cpu-acceleration]
```

MPI support:
```bash
python -m pip install -e .[mpi]
```

Plotting helper dependencies (cmasher, joblib, Pillow, packaging):
```bash
python -m pip install -e .[plotting]
```

Volume-rendering helpers:
```bash
python -m pip install -e .[volume-rendering]
```

GPU support requires installing the CuPy wheel that matches your CUDA stack explicitly. For example:
```bash
python -m pip install cupy-cuda12x
```

Full optional environment:
```bash
python -m pip install -e .[full]
```

If you use ionization or ion-column workflows, install the AstroPlasma submodule in the same environment as well.

## Quick Start

### Basic Usage
```python
from athena_research.core import AthenaData

# Load a simulation snapshot
ad = AthenaData()
ad.load('TRML.00010.athdf')

# Access physical quantities
density = ad.data('dens')
temperature = ad.data('temp')
```

### GPU-Accelerated Computing
```python
from athena_research.backends import get_backend

# Auto-detect and use GPU if available
backend = get_backend(backend_type='auto')
xp = backend.xp  # CuPy if GPU available, NumPy otherwise

# Perform GPU-accelerated operations
data = xp.random.random((500, 500, 500))
fft_result = xp.fft.fftn(data)

# Convert back to NumPy if needed
numpy_result = backend.asnumpy(fft_result)
```

### MPI Distributed Computing
```python
from athena_research.backends.mpi_utils import setup_mpi_environment

# Initialize MPI with GPU backend
backend, mpi = setup_mpi_environment(backend_type='gpu')

if mpi.rank == 0:
    data = load_large_dataset()
else:
    data = None

# Distribute work across nodes
local_data = mpi.distribute_array(data, axis=0)
local_result = process(local_data)
global_result = mpi.gather_array(local_result)
```

### Multi-GPU on Single Node

Multi-GPU usage is demonstrated in `examples/example_multi_gpu.py` using one MPI rank per GPU on a single node.

```bash
mpirun -np 4 python examples/example_multi_gpu.py basic
```

## Package structure

```
├── setup.py
├── README.md
├── QUICKSTART.md
├── pytest.ini
├── examples/
│   ├── README.md
│   ├── example_cpu_gpu.py
│   ├── example_mpi.py
│   └── example_multi_gpu.py
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   ├── test_operations_area_functions.py
│   ├── test_operations_basic.py
│   ├── test_operations_gradients.py
│   ├── test_operations_spectra.py
│   └── test_operations_structure_functions.py
athena_research/
├── __init__.py
├── PARTICLE_ANALYSIS.md
├── operations_tests.ipynb
├── particle_analysis_tests.ipynb
├── backends/
│   ├── __init__.py
│   ├── backend_manager.py
│   └── mpi_utils.py
├── core/
│   ├── __init__.py
│   ├── athena_data.py
│   ├── athena_dataset.py
│   ├── base.py
│   ├── data_functions.py
│   ├── io_utils.py
│   ├── particle_io.py
│   ├── units.py
│   └── utils.py
├── operations/
│   ├── __init__.py
│   ├── area_functions.py
│   ├── basic_operations.py
│   ├── grad_div_curl.py
│   ├── histograms.py
│   ├── marching_cubes_src/
│   ├── profiles.py
│   ├── spectra.py
│   ├── structure_functions.py
│   ├── verify_parseval_theorem.py
│   └── weighted_projection.py
└── utils/
    ├── __init__.py
    ├── batch_processing.py
    └── meshblock_utils.py
```

Note: `problem_physics/` (project-specific cooling/heating physics for thermal-instability
runs) is intentionally not part of this generic package. Problem-specific analysis code
will be added via separate branches as needed.

Notes

- Optional functionality (CuPy, mpi4py, numba) is summarized in `QUICKSTART.md` and `examples/README.md`. If an optional backend is not available, the package falls back to CPU code where implemented.
- Known limitation: The Helmholtz-decomposed structure functions and Helmholtz power spectra are not yet supported in MPI+GPU (multi-rank GPU) mode. When invoked in MPI+GPU runs with GPUs enabled, these functions will raise an error asking you to run the analysis on a single GPU (no MPI) or to use CPU mode. We are actively working to add robust MPI+GPU support for these routines.

## Running the test suite

```bash
python -m pip install -e .[dev]
pytest
```

Most tests are fast, self-contained unit/regression tests that need no simulation data.
A smaller set of correctness/cross-validation tests (marked `@pytest.mark.data`) load a
real snapshot and are skipped automatically unless `ATHENA_RESEARCH_TEST_DATA` points at a
directory containing grid-data snapshots (any basename/variable-set, `.athdf` or `.bin`).
Run before every commit.

## Example usage of the package - check athena_research/operations_tests.ipynb and athena_research/particle_analysis_tests.ipynb
