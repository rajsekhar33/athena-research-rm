"""
Example: Basic CPU/GPU usage with the new backend system

This example demonstrates:
1. Auto backend selection
2. Forcing CPU or GPU backend
3. Basic array operations
4. Memory management
"""
import sys
from pathlib import Path

import numpy as np
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def benchmark_fft(backend_type='auto'):
    """Benchmark FFT performance on CPU vs GPU."""
    from athena_research.backends import get_backend
    
    # Initialize backend
    backend = get_backend(backend_type=backend_type)
    xp = backend.xp
    
    print(f"\n{'='*60}")
    print(f"Backend: {backend.backend_type.upper()}")
    if backend.backend_type == 'gpu':
        mem_info = backend.get_memory_info()
        print(f"GPU memory: {mem_info['free']/1e9:.2f} GB free / {mem_info['total']/1e9:.2f} GB total")
    print(f"{'='*60}\n")
    
    # Create test data
    sizes = [64, 128, 256, 512]
    
    for size in sizes:
        # Create random 3D array
        data_cpu = np.random.random((size, size, size))
        
        # Transfer to device
        data = xp.asarray(data_cpu)
        
        # Warm-up
        _ = xp.fft.fftn(data)
        backend.synchronize()
        
        # Benchmark
        n_iterations = 10
        start = time.time()
        for _ in range(n_iterations):
            result = xp.fft.fftn(data)
        backend.synchronize()
        elapsed = time.time() - start
        
        avg_time = elapsed / n_iterations
        print(f"Size {size}³: {avg_time*1000:.2f} ms per FFT")
        
        # Free memory
        del data, result
        backend.free_memory()
    
    return backend


def demonstrate_operations():
    """Demonstrate common operations with backend system."""
    from athena_research.backends import get_backend
    
    backend = get_backend(backend_type='auto')
    xp = backend.xp
    
    print(f"\nUsing {backend.backend_type.upper()} backend")
    print("-" * 60)
    
    # Create arrays
    print("\n1. Creating arrays...")
    a = xp.random.random((100, 100, 100))
    b = xp.random.random((100, 100, 100))
    
    # Basic operations
    print("2. Basic operations...")
    c = a + b
    d = a * b
    e = xp.sqrt(a)
    
    # Reductions
    print("3. Reductions...")
    mean_val = xp.mean(a)
    max_val = xp.max(a)
    sum_val = xp.sum(a)
    
    print(f"   Mean: {float(mean_val):.6f}")
    print(f"   Max:  {float(max_val):.6f}")
    print(f"   Sum:  {float(sum_val):.2f}")
    
    # Gradients
    print("4. Computing gradients...")
    grad = xp.gradient(a)
    print(f"   Gradient components: {len(grad)}")
    
    # FFT
    print("5. Computing FFT...")
    fft_result = xp.fft.fftn(a)
    print(f"   FFT shape: {fft_result.shape}")
    
    # Convert back to NumPy
    print("6. Converting back to NumPy...")
    result_numpy = backend.asnumpy(c)
    print(f"   Result type: {type(result_numpy)}")
    print(f"   Result shape: {result_numpy.shape}")
    
    # Memory info
    print("\n7. Memory information:")
    mem_info = backend.get_memory_info()
    if backend.backend_type == 'gpu':
        print(f"   Free:  {mem_info['free']/1e9:.2f} GB")
        print(f"   Total: {mem_info['total']/1e9:.2f} GB")
        print(f"   Used by CuPy: {mem_info['used_by_cupy']/1e9:.2f} GB")
    else:
        print(f"   Free:  {mem_info['free']/1e9:.2f} GB")
        print(f"   Total: {mem_info['total']/1e9:.2f} GB")
        print(f"   Used:  {mem_info['used']/1e9:.2f} GB")
    
    print("\nDone!")


def compare_cpu_gpu():
    """Compare CPU and GPU performance side-by-side."""
    print("\n" + "="*80)
    print("CPU vs GPU Performance Comparison")
    print("="*80)
    
    # Test CPU
    print("\nTesting CPU backend...")
    cpu_backend = benchmark_fft(backend_type='cpu')
    
    # Test GPU (if available)
    try:
        print("\nTesting GPU backend...")
        gpu_backend = benchmark_fft(backend_type='gpu')
    except Exception as e:
        print(f"\nGPU not available: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU/GPU Backend Example')
    parser.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], default='auto',
                       help='Backend type to use')
    parser.add_argument('--compare', action='store_true',
                       help='Compare CPU vs GPU performance')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_cpu_gpu()
    else:
        if args.backend != 'auto':
            benchmark_fft(backend_type=args.backend)
        demonstrate_operations()
