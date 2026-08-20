# Examples

This directory contains example scripts demonstrating the new backend system for CPU/GPU and MPI-based distributed computing.

## Prerequisites

### Basic (CPU only)
```bash
pip install -e ..
```

### GPU Support
```bash
pip install -e ..[gpu]
```

### MPI Support
```bash
pip install -e ..[mpi]
```

### Full Installation
```bash
pip install -e ..[full]
```

## Examples

### 1. CPU/GPU Backend (`example_cpu_gpu.py`)

Demonstrates basic usage of the backend system with automatic CPU/GPU selection.

**Run with default (auto) backend:**
```bash
python example_cpu_gpu.py
```

**Force CPU backend:**
```bash
python example_cpu_gpu.py --backend cpu
```

**Force GPU backend:**
```bash
python example_cpu_gpu.py --backend gpu
```

**Compare CPU vs GPU performance:**
```bash
python example_cpu_gpu.py --compare
```

### 2. MPI Distributed Computing (`example_mpi.py`)

Demonstrates MPI-based distributed computing across multiple nodes/processes.

**Basic MPI example:**
```bash
mpirun -np 4 python example_mpi.py basic
```

**Workload distribution:**
```bash
mpirun -np 4 python example_mpi.py workload
```

**Array distribution:**
```bash
mpirun -np 4 python example_mpi.py array
```

**MPI with GPU:**
```bash
mpirun -np 4 python example_mpi.py gpu
```

**Note:** The number after `-np` is the number of MPI processes. Each process will be assigned a GPU based on rank (rank 0 -> GPU 0, rank 1 -> GPU 1, etc.)

### 3. Multi-GPU on Single Node (`example_multi_gpu.py`)

Demonstrates using multiple GPUs on a single node with one MPI rank per GPU.

**Basic multi-GPU:**
```bash
mpirun -np 4 python example_multi_gpu.py basic
```

**FFT benchmark:**
```bash
mpirun -np 4 python example_multi_gpu.py fft
```

**Parallel dataset processing:**
```bash
mpirun -np 4 python example_multi_gpu.py parallel
```

**Memory information:**
```bash
mpirun -np 4 python example_multi_gpu.py memory
```

## When to Use What

### CPU Backend
- No GPU available
- Small datasets that fit in CPU memory
- Testing and development
- CPU cluster computing with MPI

### GPU Backend
- Large datasets
- Heavy numerical computations (FFT, matrix operations)
- Single GPU available
- Best performance for array operations

### MPI
- Multi-node cluster computing
- Datasets too large for single machine
- Distributed parallel processing
- Each node can use CPU or GPU backend

### Multi-GPU
- Single node with multiple GPUs
- Need to process multiple datasets in parallel
- Large computation that can be split across GPUs
- Better than MPI for single-node multi-GPU systems

## Performance Tips

1. **Use GPU for large arrays:** GPU shows significant speedup for arrays larger than ~10⁶ elements

2. **Memory management:** Free GPU memory between operations:
   ```python
   backend.free_memory()
   ```

3. **Synchronize before timing:** Always synchronize GPU operations before measuring time:
   ```python
   backend.synchronize()
   ```

4. **Batch processing:** Process multiple items together rather than one at a time

5. **Choose the right parallelism:**
   - Single machine, multiple GPUs → Multi-GPU manager
   - Multiple machines → MPI
   - Single machine, single GPU → Standard GPU backend

## Troubleshooting

### "CuPy not available"
Install CuPy for your CUDA version:
```bash
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### "mpi4py not available"
Install MPI library and mpi4py:
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev
pip install mpi4py

# macOS
brew install openmpi
pip install mpi4py
```

### "Out of memory" errors
1. Reduce batch size or array dimensions
2. Free memory explicitly: `backend.free_memory()`
3. Use MPI to distribute across multiple nodes

### Multi-GPU not using all GPUs
Check available GPUs:
```python
import cupy as cp
print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")
```

## Next Steps

After running these examples, check out:
- [README.md](../README.md)
- [QUICKSTART.md](../QUICKSTART.md)
