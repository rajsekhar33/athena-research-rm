"""
Example: MPI distributed computing

This example demonstrates:
1. Setting up MPI with backend
2. Distributing work across MPI ranks
3. Gathering results
4. GPU assignment with MPI

Run with: mpirun -np 4 python example_mpi.py
"""
import sys
from pathlib import Path

import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def mpi_basic_example():
    """Basic MPI operations."""
    from athena_research.backends.mpi_utils import setup_mpi_environment
    
    # Initialize MPI with auto backend
    backend, mpi = setup_mpi_environment(backend_type='auto')
    
    print(f"Rank {mpi.rank}/{mpi.size}: Using {backend.backend_type} backend")
    
    # Barrier to synchronize
    mpi.barrier()
    
    # Root broadcasts data
    if mpi.rank == 0:
        data = np.random.random((100, 100))
        print(f"Rank 0: Broadcasting data of shape {data.shape}")
    else:
        data = None
    
    data = mpi.broadcast(data, root=0)
    print(f"Rank {mpi.rank}: Received data of shape {data.shape}")
    
    # Each rank processes its data
    local_result = np.sum(data) * (mpi.rank + 1)
    print(f"Rank {mpi.rank}: Local result = {local_result:.6f}")
    
    # Reduce to get global sum
    global_sum = mpi.allreduce(local_result, op='sum')
    
    if mpi.rank == 0:
        print(f"\nGlobal sum: {global_sum:.6f}")


def mpi_workload_distribution():
    """Distribute workload across MPI ranks."""
    from athena_research.backends.mpi_utils import setup_mpi_environment
    
    backend, mpi = setup_mpi_environment(backend_type='auto')
    xp = backend.xp
    
    # Total number of tasks
    n_tasks = 100
    
    # Distribute workload
    start_idx, end_idx = mpi.distribute_workload(n_tasks)
    
    print(f"Rank {mpi.rank}: Processing tasks {start_idx} to {end_idx}")
    
    # Each rank processes its portion
    local_results = []
    for i in range(start_idx, end_idx):
        # Simulate some work
        result = xp.sum(xp.random.random((50, 50))) * i
        local_results.append(float(result))
    
    # Gather all results at root
    all_results = mpi.gather(local_results, root=0)
    
    if mpi.rank == 0:
        # Flatten results
        all_results = [item for sublist in all_results for item in sublist]
        print(f"\nProcessed {len(all_results)} tasks")
        print(f"Total sum: {sum(all_results):.2f}")


def mpi_array_distribution():
    """Distribute large array across MPI ranks."""
    from athena_research.backends.mpi_utils import setup_mpi_environment
    
    backend, mpi = setup_mpi_environment(backend_type='auto')
    xp = backend.xp
    
    # Create large array at root
    if mpi.rank == 0:
        print("Rank 0: Creating large array...")
        large_array = np.random.random((1000, 500, 500))
        print(f"Array shape: {large_array.shape}")
        print(f"Array size: {large_array.nbytes / 1e9:.2f} GB")
    else:
        large_array = None
    
    # Distribute array along axis 0
    print(f"Rank {mpi.rank}: Waiting for data distribution...")
    local_chunk = mpi.distribute_array(large_array, axis=0, root=0)
    
    print(f"Rank {mpi.rank}: Received chunk of shape {local_chunk.shape}")
    
    # Transfer to device
    local_chunk_device = xp.asarray(local_chunk)
    
    # Process locally (e.g., compute FFT)
    print(f"Rank {mpi.rank}: Computing FFT...")
    start = time.time()
    fft_result = xp.fft.fftn(local_chunk_device)
    backend.synchronize()
    elapsed = time.time() - start
    
    print(f"Rank {mpi.rank}: FFT took {elapsed:.3f} seconds")
    
    # Compute local statistics
    local_mean = float(xp.mean(xp.abs(fft_result)))
    local_max = float(xp.max(xp.abs(fft_result)))
    
    # Gather statistics
    all_means = mpi.gather(local_mean, root=0)
    all_maxes = mpi.gather(local_max, root=0)
    
    if mpi.rank == 0:
        print(f"\nStatistics from all ranks:")
        for rank, (mean, max_val) in enumerate(zip(all_means, all_maxes)):
            print(f"  Rank {rank}: mean={mean:.6f}, max={max_val:.6f}")
        
        global_mean = np.mean(all_means)
        global_max = np.max(all_maxes)
        print(f"\nGlobal: mean={global_mean:.6f}, max={global_max:.6f}")


def mpi_gpu_example():
    """MPI with GPU backend - each rank gets its own GPU."""
    from athena_research.backends.mpi_utils import setup_mpi_environment
    
    try:
        # Initialize with GPU backend
        backend, mpi = setup_mpi_environment(backend_type='gpu')
        
        print(f"Rank {mpi.rank}: Using GPU {backend.device_id}")
        
        # Get GPU memory info
        mem_info = backend.get_memory_info()
        print(f"Rank {mpi.rank}: GPU memory: {mem_info['free']/1e9:.2f} GB free")
        
        # Each rank does computation on its GPU
        xp = backend.xp
        data = xp.random.random((200, 200, 200))
        
        start = time.time()
        result = xp.fft.fftn(data)
        backend.synchronize()
        elapsed = time.time() - start
        
        print(f"Rank {mpi.rank}: FFT took {elapsed:.3f} seconds")
        
        # Reduce to get average time
        avg_time = mpi.allreduce(elapsed, op='sum') / mpi.size
        
        if mpi.rank == 0:
            print(f"\nAverage FFT time across all GPUs: {avg_time:.3f} seconds")
        
    except Exception as e:
        print(f"GPU example failed: {e}")
        print("Make sure you have GPUs available and CuPy installed")


if __name__ == '__main__':
    print("="*80)
    print("MPI Distributed Computing Example")
    print("="*80)
    print()
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        example = 'basic'
    
    if example == 'basic':
        print("Running basic MPI example...\n")
        mpi_basic_example()
    elif example == 'workload':
        print("Running workload distribution example...\n")
        mpi_workload_distribution()
    elif example == 'array':
        print("Running array distribution example...\n")
        mpi_array_distribution()
    elif example == 'gpu':
        print("Running GPU + MPI example...\n")
        mpi_gpu_example()
    else:
        print(f"Unknown example: {example}")
        print("Available examples: basic, workload, array, gpu")
        print("\nUsage: mpirun -np 4 python example_mpi.py [basic|workload|array|gpu]")
