# Quick Start: CPU/GPU and MPI Computing

This quickstart demonstrates recommended usage patterns for the package's CPU/GPU and MPI
support. See `examples/` for runnable CPU/GPU/MPI example scripts.

## Installation

- Core (CPU):
  pip install -e .
- GPU support (optional):
  pip install -e .[gpu]
  pip install cupy-cuda11x  # or cupy-cuda12x depending on CUDA
- MPI support (optional):
  pip install -e .[mpi]

## Key APIs and examples

- Use the package `core` base for automatic backend selection:

```python
from athena_research.core import base
xp = base.xp           # CuPy if available, otherwise NumPy
asnumpy = base.asnumpy
```

- MPI utilities:

```python
from athena_research.backends.mpi_utils import MPIManager
mpi = MPIManager()
# Use mpi.broadcast, mpi.allreduce, mpi.gather, etc.
```

- Multi-GPU examples live in `examples/example_multi_gpu.py` (requires an optional backend implementation).

## Tips

- To force CPU-only mode before importing package:
  export ATHENA_RESEARCH_CPU_ONLY=1

- GPU assignment in multi-rank runs: map ranks to available devices (round-robin), e.g. in
  `examples/example_multi_gpu.py`.

- Synchronize CuPy before timing: `cp.cuda.Stream.null.synchronize()`

- Free CuPy memory when finished with large arrays: `cp.get_default_memory_pool().free_all_blocks()`

## Troubleshooting

- If CuPy is missing: `pip install cupy-cuda11x` (or cupy-cuda12x)
- If mpi4py is missing: install system MPI then `pip install mpi4py`
- If you see missing variables in MPI runs, inspect `ad.data('<name>', start, end)` on the owning rank (error messages from code print diagnostics)

## Examples and docs

- Examples: `examples/` (CPU/GPU/MPI examples)
- Additional repo guides: `README.md`, `examples/README.md`

