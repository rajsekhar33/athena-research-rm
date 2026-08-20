"""
Example: Multi-GPU computation on a single node using MPI.

This example demonstrates:
1. Assigning one GPU per MPI rank
2. Distributing arrays across GPUs
3. Running GPU work in parallel across ranks
4. Aggregating results back to rank 0

Run with:
    mpirun -np 4 python example_multi_gpu.py basic
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_multi_gpu():
    """Initialize one GPU per MPI rank."""
    from athena_research.backends import setup_mpi_environment

    backend, mpi = setup_mpi_environment(backend_type='gpu')
    if backend.backend_type != 'gpu':
        raise RuntimeError(
            "GPU backend was requested, but no CUDA-enabled backend is available."
        )
    return backend, mpi, backend.xp


def basic_multi_gpu():
    """Distribute a global array across ranks and gather it back."""
    backend, mpi, xp = _setup_multi_gpu()

    if mpi.rank == 0:
        data = np.random.random((256, 128, 64))
        print(f"Distributing array with shape {data.shape}")
    else:
        data = None

    local_chunk = mpi.distribute_array(data, axis=0, root=0)
    local_chunk_gpu = xp.asarray(local_chunk)

    local_summary = {
        'rank': mpi.rank,
        'device_id': backend.device_id,
        'chunk_shape': tuple(local_chunk_gpu.shape),
        'chunk_sum': float(xp.sum(local_chunk_gpu)),
        'chunk_max': float(xp.max(local_chunk_gpu)),
    }
    summaries = mpi.gather(local_summary, root=0)

    reconstructed = mpi.gather_array(local_chunk_gpu, axis=0, root=0)

    if mpi.rank == 0:
        reconstructed_np = backend.asnumpy(reconstructed)
        print("\nPer-rank chunk summaries:")
        for summary in summaries:
            print(
                f"  Rank {summary['rank']}: GPU {summary['device_id']}, "
                f"chunk={summary['chunk_shape']}, "
                f"sum={summary['chunk_sum']:.6e}, max={summary['chunk_max']:.6e}"
            )

        print("\nGathered result:")
        print(f"  Shape preserved: {reconstructed_np.shape == data.shape}")
        print(f"  Data preserved:  {np.allclose(reconstructed_np, data)}")

    backend.free_memory()


def multi_gpu_fft_benchmark():
    """Run the same FFT workload on each GPU and summarize timings."""
    backend, mpi, xp = _setup_multi_gpu()

    local_shape = (96, 96, 96)
    data = xp.random.random(local_shape)

    _ = xp.fft.fftn(data)
    backend.synchronize()

    start = time.time()
    result = xp.fft.fftn(data)
    backend.synchronize()
    elapsed = time.time() - start

    timing = {
        'rank': mpi.rank,
        'device_id': backend.device_id,
        'elapsed': elapsed,
        'mean_abs': float(xp.mean(xp.abs(result))),
    }
    timings = mpi.gather(timing, root=0)

    if mpi.rank == 0:
        print("\nPer-rank FFT timings:")
        for info in timings:
            print(
                f"  Rank {info['rank']}: GPU {info['device_id']}, "
                f"time={info['elapsed']:.4f}s, mean|F|={info['mean_abs']:.6e}"
            )

        elapsed_values = np.array([info['elapsed'] for info in timings])
        print(
            "\nSummary: "
            f"avg={elapsed_values.mean():.4f}s, "
            f"min={elapsed_values.min():.4f}s, "
            f"max={elapsed_values.max():.4f}s"
        )

    backend.free_memory()


def multi_gpu_parallel_processing():
    """Use MPIManager.parallel_compute to fan out independent GPU workloads."""
    backend, mpi, xp = _setup_multi_gpu()

    dataset_ids = list(range(mpi.size * 3)) if mpi.rank == 0 else None

    def process_dataset(dataset_id):
        data = xp.random.random((64, 64, 64))
        spectrum = xp.abs(xp.fft.fftn(data))
        backend.synchronize()
        return {
            'dataset_id': dataset_id,
            'rank': mpi.rank,
            'device_id': backend.device_id,
            'mean_abs': float(xp.mean(spectrum)),
            'max_abs': float(xp.max(spectrum)),
        }

    results = mpi.parallel_compute(process_dataset, dataset_ids, gather_results=True)

    if mpi.rank == 0:
        print("\nParallel dataset results:")
        for result in sorted(results, key=lambda item: item['dataset_id']):
            print(
                f"  Dataset {result['dataset_id']}: rank {result['rank']}, "
                f"GPU {result['device_id']}, mean={result['mean_abs']:.6e}, "
                f"max={result['max_abs']:.6e}"
            )

    backend.free_memory()


def memory_info_all_gpus():
    """Display memory information from each active GPU."""
    backend, mpi, _ = _setup_multi_gpu()

    mem_info = backend.get_memory_info()
    mem_info['rank'] = mpi.rank
    mem_info['device_id'] = backend.device_id
    all_info = mpi.gather(mem_info, root=0)

    if mpi.rank == 0:
        print("\nGPU memory information:")
        for info in sorted(all_info, key=lambda item: item['rank']):
            free_gb = info['free'] / 1e9
            total_gb = info['total'] / 1e9
            used_gb = info['used'] / 1e9
            print(
                f"  Rank {info['rank']} / GPU {info['device_id']}: "
                f"free={free_gb:.2f} GB, used={used_gb:.2f} GB, total={total_gb:.2f} GB"
            )

    backend.free_memory()


if __name__ == '__main__':
    print("=" * 80)
    print("Single-Node Multi-GPU Example")
    print("=" * 80)

    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        example = 'basic'

    if example == 'basic':
        basic_multi_gpu()
    elif example == 'fft':
        multi_gpu_fft_benchmark()
    elif example == 'parallel':
        multi_gpu_parallel_processing()
    elif example == 'memory':
        memory_info_all_gpus()
    else:
        print(f"Unknown example: {example}")
        print("Available examples: basic, fft, parallel, memory")
        print("\nUsage: mpirun -np 4 python example_multi_gpu.py [basic|fft|parallel|memory]")
        sys.exit(1)
