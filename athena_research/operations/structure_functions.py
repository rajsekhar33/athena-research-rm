"""
Structure function calculation for Athenak simulation data.

This module provides GPU-accelerated (CUDA) and CPU-based structure function
computation. For CPU mode, Numba JIT compilation provides significant speedup
(10-100x) over pure NumPy. Install numba for best CPU performance:
    pip install numba
"""
import numpy as np
from ..core.base import xp, asnumpy, cupy_enabled
from ..core.utils import clear_backend_memory
from ..utils.meshblock_utils import is_block_outside_xyz
from ..utils.batch_processing import determine_blocks_per_batch

# Try to import MPI utilities
try:
    from ..backends.mpi_utils import MPIManager
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

# Only define CUDA kernel if CuPy is actually being used
if cupy_enabled:
    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_kernel = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Simple linear congruential generator for random numbers
__device__ unsigned int simple_rand(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_kernel(const double* data1, const double* data2,
                            const double* weight1, const double* weight2,
                            const double* x1, const double* y1, const double* z1,
                            const double* x2, const double* y2, const double* z2,
                            const double* rbins, 
                            double* sf, double* num_bin_points, 
                            double* weight_sum_bin, const int n_pairs, 
                            const int n_points1, const int n_points2,
                            const double Lx, const double Ly, const double Lz,
                            const int nbins, const int max_order,
                            const bool periodic_x, const bool periodic_y, const bool periodic_z,
                            const unsigned int base_seed)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n_pairs) {
        // Initialize simple random number generator with deterministic seed
        // Combine base_seed with thread index for reproducible but varied sampling
        unsigned int seed = base_seed + idx * 1664525;
        
        // Randomly select points
        int i1 = simple_rand(&seed) % n_points1;
        int i2 = simple_rand(&seed) % n_points2;
        
        // Skip if same point in same block
        if ((data1 == data2 && i1 == i2) || weight1[i1] == 0 || weight2[i2] == 0) return;
        
        // Calculate distances with periodic boundaries
        double dist_x = fabs(x2[i2] - x1[i1]);
        double dist_y = fabs(y2[i2] - y1[i1]);
        double dist_z = fabs(z2[i2] - z1[i1]);
        
        if (periodic_x) dist_x = fmin(dist_x, Lx - dist_x);
        if (periodic_y) dist_y = fmin(dist_y, Ly - dist_y); 
        if (periodic_z) dist_z = fmin(dist_z, Lz - dist_z);
        
        double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
        
        // Structure function
        double diff = fabs(weight2[i2]*data2[i2] - weight1[i1]*data1[i1]);
        double sf_value[10];
        for(int p = 0; p < max_order; p++) {
            sf_value[p] = 0.0;
        }
        sf_value[0] = diff;
        for(int p = 1; p < max_order; p++) {
            sf_value[p] = diff * sf_value[p-1];
        }
        // Find bin and update atomically
        int bin_idx = binary_search_bin(rbins, r, nbins);
        if (bin_idx >= 0 && bin_idx<nbins-1) {
            atomicAdd(&num_bin_points[bin_idx], 1.0);
            atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
            for(int p = 0; p < max_order; p++) {
                atomicAdd(&sf[p * nbins + bin_idx], sf_value[p]);
            }
        }
    }
}
''', 'structure_function_kernel')

    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_kernel_exhaustive = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_kernel_exhaustive(const double* data1, const double* data2,
                            const double* weight1, const double* weight2,
                            const double* x1, const double* y1, const double* z1,
                            const double* x2, const double* y2, const double* z2,
                            const double* rbins, 
                            double* sf, double* num_bin_points, 
                            double* weight_sum_bin, 
                            const int n_points1, const int n_points2,
                            const double Lx, const double Ly, const double Lz,
                            const int nbins, const int max_order,
                            const bool periodic_x, const bool periodic_y, const bool periodic_z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Calculate the point indices from the thread index
    // If the two arrays are the same, we calculate the upper triangular portion to avoid redundant pairs
    int i1, i2;
    if (data1 == data2) {
        // Convert linear index to upper triangular matrix indices
        // We're computing pairs (i1,i2) where i1 < i2
        int k = idx;
        i1 = n_points1 - 2 - floor(sqrt((double)(-8.*k + 4.*n_points1*(n_points1-1.)-7.)/2.0 - 0.5));
        i2 = k + i1 + 1 - n_points1*(n_points1-1)/2 + (n_points1-i1)*((n_points1-i1)-1)/2;
        
        // Check if we're within bounds of valid pairs
        if (idx >= (n_points1 * (n_points1 - 1) / 2) || i1 < 0 || i2 >= n_points1) return;
    } else {
        // For different arrays, we compute all pairs
        i1 = idx / n_points2;
        i2 = idx % n_points2;
        
        // Check if we're within bounds
        if (i1 >= n_points1 || i2 >= n_points2) return;
    }
    
    // Skip if weights are zero
    if (weight1[i1] == 0 || weight2[i2] == 0) return;
    
    // Calculate distances with periodic boundaries
    double dist_x = fabs(x2[i2] - x1[i1]);
    double dist_y = fabs(y2[i2] - y1[i1]);
    double dist_z = fabs(z2[i2] - z1[i1]);
    
    if (periodic_x) dist_x = fmin(dist_x, Lx - dist_x);
    if (periodic_y) dist_y = fmin(dist_y, Ly - dist_y); 
    if (periodic_z) dist_z = fmin(dist_z, Lz - dist_z);
    
    double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
    
    // Structure function
    double diff = fabs(weight2[i2]*data2[i2] - weight1[i1]*data1[i1]);
    double sf_value[10];
    sf_value[0] = diff;
    for(int p = 1; p < max_order; p++) {
        sf_value[p] = diff * sf_value[p-1];
    }
    
    // Find bin and update atomically
    int bin_idx = binary_search_bin(rbins, r, nbins);
    if (bin_idx >= 0 && bin_idx < nbins-1) {
        atomicAdd(&num_bin_points[bin_idx], 1.0);
        atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
        for(int p = 0; p < max_order; p++) {
            atomicAdd(&sf[p * nbins + bin_idx], sf_value[p]);
        }
    }
}
''', 'structure_function_kernel_exhaustive')

    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_varlist_kernel = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Simple linear congruential generator for random numbers
__device__ unsigned int simple_rand(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_varlist_kernel(const double* data1, const double* data2, const int no_vars,
                            const double* weight1, const double* weight2,
                            const double* x1, const double* y1, const double* z1,
                            const double* x2, const double* y2, const double* z2,
                            const double* rbins, 
                            double* sf, double* num_bin_points, 
                            double* weight_sum_bin, const int n_pairs, 
                            const int n_points1, const int n_points2,
                            const double Lx, const double Ly, const double Lz,
                            const int nbins, const int max_order,
                            const bool periodic_x, const bool periodic_y, const bool periodic_z,
                            const unsigned int base_seed)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n_pairs) {
        // Initialize simple random number generator with deterministic seed
        // Combine base_seed with thread index for reproducible but varied sampling
        unsigned int seed = base_seed + idx * 1664525;
        
        // Randomly select points
        int i1 = simple_rand(&seed) % n_points1;
        int i2 = simple_rand(&seed) % n_points2;
        
        // Skip if same point in same block
        if ((data1 == data2 && i1 == i2) || weight1[i1] == 0 || weight2[i2] == 0) return;
        
        // Calculate distances with periodic boundaries
        double dist_x = fabs(x2[i2] - x1[i1]);
        double dist_y = fabs(y2[i2] - y1[i1]);
        double dist_z = fabs(z2[i2] - z1[i1]);
        
        if (periodic_x) dist_x = fmin(dist_x, Lx - dist_x);
        if (periodic_y) dist_y = fmin(dist_y, Ly - dist_y); 
        if (periodic_z) dist_z = fmin(dist_z, Lz - dist_z);
        
        double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
        
        // Find bin
        int bin_idx = binary_search_bin(rbins, r, nbins);
        // Update atomically
        if (bin_idx >= 0 && bin_idx<nbins-1) {
            atomicAdd(&num_bin_points[bin_idx], 1.0);
            atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
        }
        // calculate sf for each var in varlist
        for (int ivar = 0; ivar < no_vars; ivar++){
            int i_2 = ivar*n_points2+i2;
            int i_1 = ivar*n_points1+i1;
            double diff = fabs(weight2[i2]*data2[i_2] - weight1[i1]*data1[i_1]);
            double sf_value[10];
            for(int p = 0; p < max_order; p++) {
                sf_value[p] = 0.0;
            }
            sf_value[0] = diff;
            for(int p = 1; p < max_order; p++) {
                sf_value[p] = diff * sf_value[p-1];
            }
            // Update atomically
            int sf_bin_offset = ivar * nbins * max_order;
            if (bin_idx >= 0 && bin_idx<nbins-1) {
                for(int p = 0; p < max_order; p++) {
                    atomicAdd(&sf[sf_bin_offset + p * nbins + bin_idx], sf_value[p]);
                }
            }
        }
    }
}
''', 'structure_function_varlist_kernel')

    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_varlist_kernel_exhaustive = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_varlist_kernel_exhaustive(const double* data1, const double* data2, const int no_vars,
                            const double* weight1, const double* weight2,
                            const double* x1, const double* y1, const double* z1,
                            const double* x2, const double* y2, const double* z2,
                            const double* rbins, 
                            double* sf, double* num_bin_points, 
                            double* weight_sum_bin, const int n_pairs, 
                            const int n_points1, const int n_points2,
                            const double Lx, const double Ly, const double Lz,
                            const int nbins, const int max_order,
                            const bool periodic_x, const bool periodic_y, const bool periodic_z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Calculate the point indices from the thread index
    // If the two arrays are the same, we calculate the upper triangular portion to avoid redundant pairs
    int i1, i2;
    if (data1 == data2) {
        // Convert linear index to upper triangular matrix indices
        // We're computing pairs (i1,i2) where i1 < i2
        int k = idx;
        i1 = n_points1 - 2 - floor(sqrt((double)(-8.*k + 4.*n_points1*(n_points1-1.)-7.)/2.0 - 0.5));
        i2 = k + i1 + 1 - n_points1*(n_points1-1)/2 + (n_points1-i1)*((n_points1-i1)-1)/2;
        
        // Check if we're within bounds of valid pairs
        if (idx >= (n_points1 * (n_points1 - 1) / 2) || i1 < 0 || i2 >= n_points1) return;
    } else {
        // For different arrays, we compute all pairs
        i1 = idx / n_points2;
        i2 = idx % n_points2;
        
        // Check if we're within bounds
        if (i1 >= n_points1 || i2 >= n_points2) return;
    }
    
    // Skip if weights are zero
    if (weight1[i1] == 0 || weight2[i2] == 0) return;
    
    // Calculate distances with periodic boundaries
    double dist_x = fabs(x2[i2] - x1[i1]);
    double dist_y = fabs(y2[i2] - y1[i1]);
    double dist_z = fabs(z2[i2] - z1[i1]);
    
    if (periodic_x) dist_x = fmin(dist_x, Lx - dist_x);
    if (periodic_y) dist_y = fmin(dist_y, Ly - dist_y); 
    if (periodic_z) dist_z = fmin(dist_z, Lz - dist_z);
    
    double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
    
    // Find bin
    int bin_idx = binary_search_bin(rbins, r, nbins);
    
    // Update atomically
    if (bin_idx >= 0 && bin_idx < nbins-1) {
        atomicAdd(&num_bin_points[bin_idx], 1.0);
        atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
    
        // Calculate structure function for each variable
        for (int ivar = 0; ivar < no_vars; ivar++) {
            int i_2 = ivar*n_points2 + i2;
            int i_1 = ivar*n_points1 + i1;
            double diff = fabs(weight2[i2]*data2[i_2] - weight1[i1]*data1[i_1]);
            
            // Calculate powers for each order
            double sf_value[10];
            sf_value[0] = diff;
            for(int p = 1; p < max_order; p++) {
                sf_value[p] = diff * sf_value[p-1];
            }
            
            // Update atomically
            int sf_bin_offset = ivar * nbins * max_order;
            for(int p = 0; p < max_order; p++) {
                atomicAdd(&sf[sf_bin_offset + p * nbins + bin_idx], sf_value[p]);
            }
        }
    }
}
''', 'structure_function_varlist_kernel_exhaustive')


    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_helmholtz_kernel = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Simple linear congruential generator for random numbers
__device__ unsigned int simple_rand(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_helmholtz_kernel(
                             const double* data1_x, const double* data1_y, const double* data1_z,
                             const double* data2_x, const double* data2_y, const double* data2_z,
                             const double* weight1, const double* weight2,
                             const double* x1, const double* y1, const double* z1,
                             const double* x2, const double* y2, const double* z2,
                             const double* rbins, 
                             double* sf_comp, double* sf_sol,
                             double* num_bin_points, 
                             double* weight_sum_bin, const int n_pairs, 
                             const int n_points1, const int n_points2,
                             const double Lx, const double Ly, const double Lz,
                             const int nbins, const int max_order,
                             const bool periodic_x, const bool periodic_y, const bool periodic_z,
                             const unsigned int base_seed)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n_pairs) {
        // Initialize simple random number generator with deterministic seed
        // Combine base_seed with thread index for reproducible but varied sampling
        unsigned int seed = base_seed + idx * 1664525;
        
        // Randomly select points
        int i1 = simple_rand(&seed) % n_points1;
        int i2 = simple_rand(&seed) % n_points2;
        
        // Skip if same point in same block
        if ((data1_x == data2_x && i1 == i2) || weight1[i1] == 0 || weight2[i2] == 0) return;
        
        // Calculate distances with periodic boundaries
        // Note that here we need to conserve the sign 
        // of the distance, since we are calculating the
        // Helmholtz decomposition
        double dist_x = x2[i2] - x1[i1];
        if (periodic_x)
        {
            if(fabs(dist_x) > 0.5*Lx){
                if(dist_x>0) dist_x -= Lx;
                else dist_x += Lx;
            }
        } 
        double dist_y = y2[i2] - y1[i1];
        if (periodic_y)
        {
            if(fabs(dist_y) > 0.5*Ly){
                if(dist_y>0) dist_y -= Ly;
                else dist_y += Ly;
            }
        }
        double dist_z = z2[i2] - z1[i1];
        if (periodic_z)
        {
            if(fabs(dist_z) > 0.5*Lz){
                if(dist_z>0) dist_z -= Lz;
                else dist_z += Lz;
            }
        }
        
        double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
        
        // Structure function
        double diff_x = weight2[i2]*data2_x[i2] - weight1[i1]*data1_x[i1];
        double diff_y = weight2[i2]*data2_y[i2] - weight1[i1]*data1_y[i1];
        double diff_z = weight2[i2]*data2_z[i2] - weight1[i1]*data1_z[i1];
        double diff  = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
        double diff_dot_r = dist_x*diff_x + dist_y*diff_y + dist_z*diff_z;
        double diff_comp = diff_dot_r/(r+1e-10);
        double diff_sol = sqrt(diff*diff - diff_comp*diff_comp);
        double sf_value_comp[10];
        double sf_value_sol[10];
        for(int p = 0; p < max_order; p++) {
            sf_value_comp[p] = 0.0;
            sf_value_sol[p]  = 0.0;
        }
        sf_value_comp[0] = fabs(diff_comp);
        sf_value_sol[0]  = fabs(diff_sol);
        for(int p = 1; p < max_order; p++) {
            sf_value_comp[p] = fabs(diff_comp) * sf_value_comp[p-1];
            sf_value_sol[p]  = fabs(diff_sol)  * sf_value_sol[p-1];
        }
        // Find bin and update atomically
        int bin_idx = binary_search_bin(rbins, r, nbins);
        if (bin_idx >= 0 && bin_idx<nbins-1) {
            atomicAdd(&num_bin_points[bin_idx], 1.0);
            atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
            for(int p = 0; p < max_order; p++) {
                atomicAdd(&sf_comp[p * nbins + bin_idx], sf_value_comp[p]);
                atomicAdd(&sf_sol [p * nbins + bin_idx], sf_value_sol[p]);
            }
        }
    }
}
''', 'structure_function_helmholtz_kernel')

    # Requires CUDA compute capability >= 6.0 (e.g. Pascal or newer) for native double-precision atomicAdd
    structure_function_helmholtz_kernel_exhaustive = xp.RawKernel(r'''
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

// Binary search function for finding bin
__device__ int binary_search_bin(const double* rbins, const double r, const int nbins) {
    int left = 0;
    int right = nbins - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        if (r >= rbins[mid] && r < rbins[mid + 1]) {
            return mid;
        }
        if (r < rbins[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return -1;
}

extern "C" __global__
void structure_function_helmholtz_kernel_exhaustive(
                             const double* data1_x, const double* data1_y, const double* data1_z,
                             const double* data2_x, const double* data2_y, const double* data2_z,
                             const double* weight1, const double* weight2,
                             const double* x1, const double* y1, const double* z1,
                             const double* x2, const double* y2, const double* z2,
                             const double* rbins, 
                             double* sf_comp, double* sf_sol,
                             double* num_bin_points, 
                             double* weight_sum_bin, const int n_pairs, 
                             const int n_points1, const int n_points2,
                             const double Lx, const double Ly, const double Lz,
                             const int nbins, const int max_order,
                             const bool periodic_x, const bool periodic_y, const bool periodic_z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Calculate the point indices from the thread index
    // If the two arrays are the same, we calculate the upper triangular portion to avoid redundant pairs
    int i1, i2;
    if (data1_x == data2_x) {
        // Convert linear index to upper triangular matrix indices
        // We're computing pairs (i1,i2) where i1 < i2
        int k = idx;
        i1 = n_points1 - 2 - floor(sqrt(-8*k + 4*n_points1*(n_points1-1)-7)/2.0 - 0.5);
        i2 = k + i1 + 1 - n_points1*(n_points1-1)/2 + (n_points1-i1)*((n_points1-i1)-1)/2;
        
        // Check if we are within bounds of valid pairs
        if (idx >= (n_points1 * (n_points1 - 1) / 2) || i1 < 0 || i2 >= n_points1) return;
    } else {
        // For different arrays, we compute all pairs
        i1 = idx / n_points2;
        i2 = idx % n_points2;
        
        // Check if we're within bounds
        if (i1 >= n_points1 || i2 >= n_points2) return;
    }
    
    // Skip if weights are zero
    if (weight1[i1] == 0 || weight2[i2] == 0) return;
    
    // Calculate distances with periodic boundaries
    // Note that here we need to conserve the sign 
    // of the distance, since we are calculating the
    // Helmholtz decomposition
    double dist_x = x2[i2] - x1[i1];
    if (periodic_x)
    {
        if(fabs(dist_x) > 0.5*Lx){
            if(dist_x > 0) dist_x -= Lx;
            else dist_x += Lx;
        }
    } 
    
    double dist_y = y2[i2] - y1[i1];
    if (periodic_y)
    {
        if(fabs(dist_y) > 0.5*Ly){
            if(dist_y > 0) dist_y -= Ly;
            else dist_y += Ly;
        }
    }
    
    double dist_z = z2[i2] - z1[i1];
    if (periodic_z)
    {
        if(fabs(dist_z) > 0.5*Lz){
            if(dist_z > 0) dist_z -= Lz;
            else dist_z += Lz;
        }
    }
    
    double r = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
    
    // Structure function
    double diff_x = weight2[i2]*data2_x[i2] - weight1[i1]*data1_x[i1];
    double diff_y = weight2[i2]*data2_y[i2] - weight1[i1]*data1_y[i1];
    double diff_z = weight2[i2]*data2_z[i2] - weight1[i1]*data1_z[i1];
    double diff  = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
    double diff_dot_r = dist_x*diff_x + dist_y*diff_y + dist_z*diff_z;
    double diff_comp = diff_dot_r/(r+1e-10);
    double diff_sol = sqrt(diff*diff - diff_comp*diff_comp);
    
    double sf_value_comp[10];
    double sf_value_sol[10];
    
    sf_value_comp[0] = fabs(diff_comp);
    sf_value_sol[0]  = fabs(diff_sol);
    
    for(int p = 1; p < max_order; p++) {
        sf_value_comp[p] = fabs(diff_comp) * sf_value_comp[p-1];
        sf_value_sol[p]  = fabs(diff_sol)  * sf_value_sol[p-1];
    }
    
    // Find bin and update atomically
    int bin_idx = binary_search_bin(rbins, r, nbins);
    
    if (bin_idx >= 0 && bin_idx < nbins-1) {
        atomicAdd(&num_bin_points[bin_idx], 1.0);
        atomicAdd(&weight_sum_bin[bin_idx], weight1[i1] + weight2[i2]);
        
        for(int p = 0; p < max_order; p++) {
            atomicAdd(&sf_comp[p * nbins + bin_idx], sf_value_comp[p]);
            atomicAdd(&sf_sol [p * nbins + bin_idx], sf_value_sol[p]);
        }
    }
}
''', 'structure_function_helmholtz_kernel_exhaustive')

    # Anisotropic MHD SF kernels.  The local mean field B_bar = (B1+B2)/2 is used
    # both to classify pairs by geometry (l, l_prll, l_perp) and to form the
    # perpendicular increment magnitudes |delta v_perp| and |delta B_perp|.
    structure_function_aniso_mhd_kernel = xp.RawKernel(r"""
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

__device__ unsigned int simple_rand(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

__device__ int binary_search_bin(const double* bins, const double val, const int nbins) {
    int left = 0, right = nbins - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (val >= bins[mid] && val < bins[mid + 1]) return mid;
        if (val < bins[mid]) right = mid - 1;
        else left = mid + 1;
    }
    return -1;
}

extern "C" __global__
void structure_function_aniso_mhd_kernel(
        const double* vel1_x, const double* vel1_y, const double* vel1_z,
        const double* vel2_x, const double* vel2_y, const double* vel2_z,
        const double* bcc1_x, const double* bcc1_y, const double* bcc1_z,
        const double* bcc2_x, const double* bcc2_y, const double* bcc2_z,
        const double* weight1, const double* weight2,
        const double* x1, const double* y1, const double* z1,
        const double* x2, const double* y2, const double* z2,
        const double* lbins, const double* lprll_bins, const double* lperp_bins,
        double* sf_l_vel, double* sf_prll_vel, double* sf_perp_vel,
        double* sf_l_vel_perp, double* sf_prll_vel_perp, double* sf_perp_vel_perp,
        double* sf_l_bcc, double* sf_prll_bcc, double* sf_perp_bcc,
        double* sf_l_bcc_perp, double* sf_prll_bcc_perp, double* sf_perp_bcc_perp,
        double* num_l, double* num_prll, double* num_perp,
        double* wsum_l, double* wsum_prll, double* wsum_perp,
        const int n_pairs, const int n_points1, const int n_points2,
        const double Lx, const double Ly, const double Lz,
        const int nbins, const int nbins_prll, const int nbins_perp, const int max_order,
        const double cos_theta_prll_min, const double cos_theta_perp_max,
        const bool periodic_x, const bool periodic_y, const bool periodic_z,
        const unsigned int base_seed)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_pairs) return;

    unsigned int seed = base_seed + idx * 1664525u;
    int i1 = simple_rand(&seed) % n_points1;
    int i2 = simple_rand(&seed) % n_points2;

    if (weight1[i1] == 0 || weight2[i2] == 0) return;

    double lx = x2[i2] - x1[i1];
    double ly = y2[i2] - y1[i1];
    double lz = z2[i2] - z1[i1];
    if (periodic_x) { if (fabs(lx) > 0.5*Lx) lx = (lx > 0) ? lx - Lx : lx + Lx; }
    if (periodic_y) { if (fabs(ly) > 0.5*Ly) ly = (ly > 0) ? ly - Ly : ly + Ly; }
    if (periodic_z) { if (fabs(lz) > 0.5*Lz) lz = (lz > 0) ? lz - Lz : lz + Lz; }

    double bx = 0.5*(bcc1_x[i1] + bcc2_x[i2]);
    double by = 0.5*(bcc1_y[i1] + bcc2_y[i2]);
    double bz = 0.5*(bcc1_z[i1] + bcc2_z[i2]);
    double bmag = sqrt(bx*bx + by*by + bz*bz);
    if (bmag < 1e-30) return;
    bx /= bmag; by /= bmag; bz /= bmag;

    double l_dot_b = lx*bx + ly*by + lz*bz;
    double l_sq = lx*lx + ly*ly + lz*lz;
    double l_mag = sqrt(l_sq);
    if (l_mag < 1e-30) return;
    double l_prll = fabs(l_dot_b);
    double l_perp_sq = l_sq - l_dot_b*l_dot_b;
    if (l_perp_sq < 0.0) l_perp_sq = 0.0;
    double l_perp = sqrt(l_perp_sq);
    // l_prll uses |l.B_hat|, so the corresponding polar angle is symmetric
    // about pi/2 and should lie in [0, pi/2]. Clip for numerical safety so
    // acos(cos_theta) never wanders outside that range.
    double cos_theta = l_prll / l_mag;
    if (cos_theta < 0.0) cos_theta = 0.0;
    if (cos_theta > 1.0) cos_theta = 1.0;

    double w1 = weight1[i1], w2 = weight2[i2];
    double dvx = w2*vel2_x[i2] - w1*vel1_x[i1];
    double dvy = w2*vel2_y[i2] - w1*vel1_y[i1];
    double dvz = w2*vel2_z[i2] - w1*vel1_z[i1];
    double dv_sq = dvx*dvx + dvy*dvy + dvz*dvz;
    double dv = sqrt(dv_sq);
    double dv_dot_b = dvx*bx + dvy*by + dvz*bz;
    double dv_perp_sq = dv_sq - dv_dot_b*dv_dot_b;
    if (dv_perp_sq < 0.0) dv_perp_sq = 0.0;
    double dv_perp = sqrt(dv_perp_sq);

    double dbx = w2*bcc2_x[i2] - w1*bcc1_x[i1];
    double dby = w2*bcc2_y[i2] - w1*bcc1_y[i1];
    double dbz = w2*bcc2_z[i2] - w1*bcc1_z[i1];
    double db_sq = dbx*dbx + dby*dby + dbz*dbz;
    double db = sqrt(db_sq);
    double db_dot_b = dbx*bx + dby*by + dbz*bz;
    double db_perp_sq = db_sq - db_dot_b*db_dot_b;
    if (db_perp_sq < 0.0) db_perp_sq = 0.0;
    double db_perp = sqrt(db_perp_sq);

    int il = binary_search_bin(lbins, l_mag, nbins);
    if (il >= 0) {
        atomicAdd(&num_l[il], 1.0);
        atomicAdd(&wsum_l[il], w1 + w2);
        double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
        for (int p = 0; p < max_order; p++) {
            atomicAdd(&sf_l_vel[p * nbins + il], val_v);
            atomicAdd(&sf_l_vel_perp[p * nbins + il], val_vp);
            atomicAdd(&sf_l_bcc[p * nbins + il], val_b);
            atomicAdd(&sf_l_bcc_perp[p * nbins + il], val_bp);
            val_v *= dv;
            val_vp *= dv_perp;
            val_b *= db;
            val_bp *= db_perp;
        }
    }

    if (cos_theta > cos_theta_prll_min) {
        int iprll = binary_search_bin(lprll_bins, l_prll, nbins_prll);
        if (iprll >= 0) {
            atomicAdd(&num_prll[iprll], 1.0);
            atomicAdd(&wsum_prll[iprll], w1 + w2);
            double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
            for (int p = 0; p < max_order; p++) {
                atomicAdd(&sf_prll_vel[p * nbins_prll + iprll], val_v);
                atomicAdd(&sf_prll_vel_perp[p * nbins_prll + iprll], val_vp);
                atomicAdd(&sf_prll_bcc[p * nbins_prll + iprll], val_b);
                atomicAdd(&sf_prll_bcc_perp[p * nbins_prll + iprll], val_bp);
                val_v *= dv;
                val_vp *= dv_perp;
                val_b *= db;
                val_bp *= db_perp;
            }
        }
    }

    if (cos_theta <= cos_theta_perp_max) {
        int iperp = binary_search_bin(lperp_bins, l_perp, nbins_perp);
        if (iperp >= 0) {
            atomicAdd(&num_perp[iperp], 1.0);
            atomicAdd(&wsum_perp[iperp], w1 + w2);
            double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
            for (int p = 0; p < max_order; p++) {
                atomicAdd(&sf_perp_vel[p * nbins_perp + iperp], val_v);
                atomicAdd(&sf_perp_vel_perp[p * nbins_perp + iperp], val_vp);
                atomicAdd(&sf_perp_bcc[p * nbins_perp + iperp], val_b);
                atomicAdd(&sf_perp_bcc_perp[p * nbins_perp + iperp], val_bp);
                val_v *= dv;
                val_vp *= dv_perp;
                val_b *= db;
                val_bp *= db_perp;
            }
        }
    }
}
""", 'structure_function_aniso_mhd_kernel')

    structure_function_aniso_mhd_kernel_exhaustive = xp.RawKernel(r"""
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF2_OPERATORS__

__device__ int binary_search_bin(const double* bins, const double val, const int nbins) {
    int left = 0, right = nbins - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (val >= bins[mid] && val < bins[mid + 1]) return mid;
        if (val < bins[mid]) right = mid - 1;
        else left = mid + 1;
    }
    return -1;
}

extern "C" __global__
void structure_function_aniso_mhd_kernel_exhaustive(
        const double* vel1_x, const double* vel1_y, const double* vel1_z,
        const double* vel2_x, const double* vel2_y, const double* vel2_z,
        const double* bcc1_x, const double* bcc1_y, const double* bcc1_z,
        const double* bcc2_x, const double* bcc2_y, const double* bcc2_z,
        const double* weight1, const double* weight2,
        const double* x1, const double* y1, const double* z1,
        const double* x2, const double* y2, const double* z2,
        const double* lbins, const double* lprll_bins, const double* lperp_bins,
        double* sf_l_vel, double* sf_prll_vel, double* sf_perp_vel,
        double* sf_l_vel_perp, double* sf_prll_vel_perp, double* sf_perp_vel_perp,
        double* sf_l_bcc, double* sf_prll_bcc, double* sf_perp_bcc,
        double* sf_l_bcc_perp, double* sf_prll_bcc_perp, double* sf_perp_bcc_perp,
        double* num_l, double* num_prll, double* num_perp,
        double* wsum_l, double* wsum_prll, double* wsum_perp,
        const int n_points1, const int n_points2,
        const double Lx, const double Ly, const double Lz,
        const int nbins, const int nbins_prll, const int nbins_perp, const int max_order,
        const double cos_theta_prll_min, const double cos_theta_perp_max,
        const bool periodic_x, const bool periodic_y, const bool periodic_z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i1, i2;
    if (vel1_x == vel2_x) {
        int k = idx;
        i1 = n_points1 - 2 - (int)floor(sqrt((double)(-8*k + 4*n_points1*(n_points1-1)-7))/2.0 - 0.5);
        i2 = k + i1 + 1 - n_points1*(n_points1-1)/2 + (n_points1-i1)*((n_points1-i1)-1)/2;
        if (idx >= (n_points1*(n_points1-1)/2) || i1 < 0 || i2 >= n_points1) return;
    } else {
        i1 = idx / n_points2;
        i2 = idx % n_points2;
        if (i1 >= n_points1 || i2 >= n_points2) return;
    }

    if (weight1[i1] == 0 || weight2[i2] == 0) return;

    double lx = x2[i2] - x1[i1];
    double ly = y2[i2] - y1[i1];
    double lz = z2[i2] - z1[i1];
    if (periodic_x) { if (fabs(lx) > 0.5*Lx) lx = (lx > 0) ? lx - Lx : lx + Lx; }
    if (periodic_y) { if (fabs(ly) > 0.5*Ly) ly = (ly > 0) ? ly - Ly : ly + Ly; }
    if (periodic_z) { if (fabs(lz) > 0.5*Lz) lz = (lz > 0) ? lz - Lz : lz + Lz; }

    double bx = 0.5*(bcc1_x[i1] + bcc2_x[i2]);
    double by = 0.5*(bcc1_y[i1] + bcc2_y[i2]);
    double bz = 0.5*(bcc1_z[i1] + bcc2_z[i2]);
    double bmag = sqrt(bx*bx + by*by + bz*bz);
    if (bmag < 1e-30) return;
    bx /= bmag; by /= bmag; bz /= bmag;

    double l_dot_b = lx*bx + ly*by + lz*bz;
    double l_sq = lx*lx + ly*ly + lz*lz;
    double l_mag = sqrt(l_sq);
    if (l_mag < 1e-30) return;
    double l_prll = fabs(l_dot_b);
    double l_perp_sq = l_sq - l_dot_b*l_dot_b;
    if (l_perp_sq < 0.0) l_perp_sq = 0.0;
    double l_perp = sqrt(l_perp_sq);
    // l_prll uses |l.B_hat|, so the corresponding polar angle is symmetric
    // about pi/2 and should lie in [0, pi/2]. Clip for numerical safety so
    // acos(cos_theta) never wanders outside that range.
    double cos_theta = l_prll / l_mag;
    if (cos_theta < 0.0) cos_theta = 0.0;
    if (cos_theta > 1.0) cos_theta = 1.0;

    double w1 = weight1[i1], w2 = weight2[i2];
    double dvx = w2*vel2_x[i2] - w1*vel1_x[i1];
    double dvy = w2*vel2_y[i2] - w1*vel1_y[i1];
    double dvz = w2*vel2_z[i2] - w1*vel1_z[i1];
    double dv_sq = dvx*dvx + dvy*dvy + dvz*dvz;
    double dv = sqrt(dv_sq);
    double dv_dot_b = dvx*bx + dvy*by + dvz*bz;
    double dv_perp_sq = dv_sq - dv_dot_b*dv_dot_b;
    if (dv_perp_sq < 0.0) dv_perp_sq = 0.0;
    double dv_perp = sqrt(dv_perp_sq);

    double dbx = w2*bcc2_x[i2] - w1*bcc1_x[i1];
    double dby = w2*bcc2_y[i2] - w1*bcc1_y[i1];
    double dbz = w2*bcc2_z[i2] - w1*bcc1_z[i1];
    double db_sq = dbx*dbx + dby*dby + dbz*dbz;
    double db = sqrt(db_sq);
    double db_dot_b = dbx*bx + dby*by + dbz*bz;
    double db_perp_sq = db_sq - db_dot_b*db_dot_b;
    if (db_perp_sq < 0.0) db_perp_sq = 0.0;
    double db_perp = sqrt(db_perp_sq);

    int il = binary_search_bin(lbins, l_mag, nbins);
    if (il >= 0) {
        atomicAdd(&num_l[il], 1.0);
        atomicAdd(&wsum_l[il], w1 + w2);
        double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
        for (int p = 0; p < max_order; p++) {
            atomicAdd(&sf_l_vel[p * nbins + il], val_v);
            atomicAdd(&sf_l_vel_perp[p * nbins + il], val_vp);
            atomicAdd(&sf_l_bcc[p * nbins + il], val_b);
            atomicAdd(&sf_l_bcc_perp[p * nbins + il], val_bp);
            val_v *= dv;
            val_vp *= dv_perp;
            val_b *= db;
            val_bp *= db_perp;
        }
    }

    if (cos_theta > cos_theta_prll_min) {
        int iprll = binary_search_bin(lprll_bins, l_prll, nbins_prll);
        if (iprll >= 0) {
            atomicAdd(&num_prll[iprll], 1.0);
            atomicAdd(&wsum_prll[iprll], w1 + w2);
            double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
            for (int p = 0; p < max_order; p++) {
                atomicAdd(&sf_prll_vel[p * nbins_prll + iprll], val_v);
                atomicAdd(&sf_prll_vel_perp[p * nbins_prll + iprll], val_vp);
                atomicAdd(&sf_prll_bcc[p * nbins_prll + iprll], val_b);
                atomicAdd(&sf_prll_bcc_perp[p * nbins_prll + iprll], val_bp);
                val_v *= dv;
                val_vp *= dv_perp;
                val_b *= db;
                val_bp *= db_perp;
            }
        }
    }

    if (cos_theta <= cos_theta_perp_max) {
        int iperp = binary_search_bin(lperp_bins, l_perp, nbins_perp);
        if (iperp >= 0) {
            atomicAdd(&num_perp[iperp], 1.0);
            atomicAdd(&wsum_perp[iperp], w1 + w2);
            double val_v = dv, val_vp = dv_perp, val_b = db, val_bp = db_perp;
            for (int p = 0; p < max_order; p++) {
                atomicAdd(&sf_perp_vel[p * nbins_perp + iperp], val_v);
                atomicAdd(&sf_perp_vel_perp[p * nbins_perp + iperp], val_vp);
                atomicAdd(&sf_perp_bcc[p * nbins_perp + iperp], val_b);
                atomicAdd(&sf_perp_bcc_perp[p * nbins_perp + iperp], val_bp);
                val_v *= dv;
                val_vp *= dv_perp;
                val_b *= db;
                val_bp *= db_perp;
            }
        }
    }
}
""", 'structure_function_aniso_mhd_kernel_exhaustive')

else:
    # Dummy kernel objects for CPU mode - these will never be called
    structure_function_kernel = None
    structure_function_kernel_exhaustive = None
    structure_function_varlist_kernel = None
    structure_function_varlist_kernel_exhaustive = None
    structure_function_helmholtz_kernel = None
    structure_function_helmholtz_kernel_exhaustive = None
    structure_function_aniso_mhd_kernel = None
    structure_function_aniso_mhd_kernel_exhaustive = None

# Try to import numba for JIT compilation of CPU code
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        # If jit is called with arguments, return the decorator
        # If called without arguments (as a decorator), return the function itself
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

@jit(nopython=True)
def _compute_periodic_distance(x1, y1, z1, x2, y2, z2, Lx, Ly, Lz, periodic_x, periodic_y, periodic_z):
    """Compute distance between two points with periodic boundaries (Numba JIT)."""
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    if periodic_x and abs(dx) > 0.5 * Lx:
        dx = dx - np.sign(dx) * Lx
    if periodic_y and abs(dy) > 0.5 * Ly:
        dy = dy - np.sign(dy) * Ly
    if periodic_z and abs(dz) > 0.5 * Lz:
        dz = dz - np.sign(dz) * Lz
    
    return np.sqrt(dx*dx + dy*dy + dz*dz)

@jit(nopython=True)
def _compute_sf_batch_numba(data1, data2, weight1, weight2, 
                             x1, y1, z1, x2, y2, z2,
                             idx1, idx2, rbins, nbins, max_order,
                             Lx, Ly, Lz, periodic_x, periodic_y, periodic_z,
                             sf, num_bin_points, weight_sum_bin):
    """
    Compute structure function for a batch of point pairs using Numba JIT.
    
    This function is compiled with Numba for speed and processes batches of
    point pairs to accumulate structure function statistics.
    """
    n_pairs = len(idx1)
    
    for i in range(n_pairs):
        i1 = idx1[i]
        i2 = idx2[i]
        
        # Skip if weights are zero
        if weight1[i1] == 0 or weight2[i2] == 0:
            continue
        
        # Calculate distance with periodic boundaries
        dx = x2[i2] - x1[i1]
        dy = y2[i2] - y1[i1]
        dz = z2[i2] - z1[i1]
        
        if periodic_x and abs(dx) > 0.5 * Lx:
            dx = dx - np.sign(dx) * Lx
        if periodic_y and abs(dy) > 0.5 * Ly:
            dy = dy - np.sign(dy) * Ly
        if periodic_z and abs(dz) > 0.5 * Lz:
            dz = dz - np.sign(dz) * Lz
        
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Calculate structure function value
        diff = abs(weight2[i2] * data2[i2] - weight1[i1] * data1[i1])
        
        # Find bin using binary search
        bin_idx = np.searchsorted(rbins, r) - 1
        
        if 0 <= bin_idx < nbins:
            num_bin_points[bin_idx] += 1.0
            weight_sum_bin[bin_idx] += weight1[i1] + weight2[i2]
            
            # Compute powers of diff and accumulate
            diff_power = diff
            for order in range(max_order):
                sf[order, bin_idx] += diff_power
                diff_power *= diff

def _compute_sf_cpu_vectorized(data1, data2, weight1, weight2, 
                                x1, y1, z1, x2, y2, z2,
                                rbins, nbins, max_order, 
                                Lx, Ly, Lz, periodic_x, periodic_y, periodic_z,
                                n_samples, batch_size=10000, use_mpi=False, random_seed=None):
    """
    CPU fallback for structure function computation using vectorized NumPy operations.
    
    This implementation processes point pairs in batches for efficiency, using
    vectorized NumPy operations where possible and Numba JIT compilation for
    the inner loop if available (disabled in MPI mode to avoid deadlock).
    
    Parameters
    ----------
    data1, data2 : ndarray
        Data values at each point
    weight1, weight2 : ndarray
        Weights for each point
    x1, y1, z1, x2, y2, z2 : ndarray
        Coordinates of points
    rbins : ndarray
        Bin edges for separations
    nbins : int
        Number of bins
    max_order : int
        Maximum structure function order
    Lx, Ly, Lz : float
        Box dimensions for periodic wrapping
    periodic_x, periodic_y, periodic_z : bool
        Whether each direction has periodic boundaries
    n_samples : int
        Number of point pairs to sample
    batch_size : int
        Number of pairs to process in each batch
    
    Returns
    -------
    sf : ndarray
        Structure function values (max_order x nbins)
    num_bin_points : ndarray
        Number of pairs in each bin
    weight_sum_bin : ndarray
        Sum of weights in each bin
    """
    n_points1 = len(data1)
    n_points2 = len(data2)
    
    # Early return if either dataset is empty (e.g., idle MPI rank with no meshblocks)
    if n_points1 == 0 or n_points2 == 0:
        sf = np.zeros((max_order, nbins))
        num_bin_points = np.zeros(nbins)
        weight_sum_bin = np.zeros(nbins)
        return sf, num_bin_points, weight_sum_bin
    
    sf = np.zeros((max_order, nbins))
    num_bin_points = np.zeros(nbins)
    weight_sum_bin = np.zeros(nbins)
    
    # Process pairs in batches
    n_batches = int(np.ceil(n_samples / batch_size))
    # Use deterministic seed if provided for reproducible results
    rng = np.random.default_rng(random_seed)
    
    for batch_idx in range(n_batches):
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, int(n_samples - batch_idx * batch_size))
        
        # Generate random indices for this batch
        idx1 = rng.integers(0, n_points1, size=current_batch_size)
        idx2 = rng.integers(0, n_points2, size=current_batch_size)
        
        # Use Numba JIT version if available AND not in MPI mode
        # (Numba JIT compilation causes deadlock when all MPI ranks compile simultaneously)
        use_numba = NUMBA_AVAILABLE and not use_mpi
        
        if use_numba:
            _compute_sf_batch_numba(
                data1, data2, weight1, weight2,
                x1, y1, z1, x2, y2, z2,
                idx1, idx2, rbins, nbins, max_order,
                float(Lx), float(Ly), float(Lz),
                periodic_x, periodic_y, periodic_z,
                sf, num_bin_points, weight_sum_bin
            )
        else:
            # Pure NumPy fallback (vectorized where possible)
            # Extract coordinates and data for sampled points
            x1_batch = x1[idx1]
            y1_batch = y1[idx1]
            z1_batch = z1[idx1]
            x2_batch = x2[idx2]
            y2_batch = y2[idx2]
            z2_batch = z2[idx2]
            data1_batch = data1[idx1]
            data2_batch = data2[idx2]
            weight1_batch = weight1[idx1]
            weight2_batch = weight2[idx2]
            
            # Filter out zero-weight pairs
            valid_mask = (weight1_batch != 0) & (weight2_batch != 0)
            if not np.any(valid_mask):
                continue
            
            x1_batch = x1_batch[valid_mask]
            y1_batch = y1_batch[valid_mask]
            z1_batch = z1_batch[valid_mask]
            x2_batch = x2_batch[valid_mask]
            y2_batch = y2_batch[valid_mask]
            z2_batch = z2_batch[valid_mask]
            data1_batch = data1_batch[valid_mask]
            data2_batch = data2_batch[valid_mask]
            weight1_batch = weight1_batch[valid_mask]
            weight2_batch = weight2_batch[valid_mask]
            
            # Vectorized distance calculation
            dx = x2_batch - x1_batch
            dy = y2_batch - y1_batch
            dz = z2_batch - z1_batch
            
            # Apply periodic boundaries
            if periodic_x:
                dx = np.where(np.abs(dx) > 0.5 * Lx, dx - np.sign(dx) * Lx, dx)
            if periodic_y:
                dy = np.where(np.abs(dy) > 0.5 * Ly, dy - np.sign(dy) * Ly, dy)
            if periodic_z:
                dz = np.where(np.abs(dz) > 0.5 * Lz, dz - np.sign(dz) * Lz, dz)
            
            r_batch = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Vectorized structure function calculation
            diff = np.abs(weight2_batch * data2_batch - weight1_batch * data1_batch)
            
            # Find bins for all distances at once
            bin_indices = np.searchsorted(rbins, r_batch) - 1
            
            # Filter valid bins
            valid_bins = (bin_indices >= 0) & (bin_indices < nbins)
            bin_indices = bin_indices[valid_bins]
            diff = diff[valid_bins]
            weight1_batch = weight1_batch[valid_bins]
            weight2_batch = weight2_batch[valid_bins]
            
            # Accumulate using np.add.at for efficient in-place addition
            np.add.at(num_bin_points, bin_indices, 1.0)
            np.add.at(weight_sum_bin, bin_indices, weight1_batch + weight2_batch)
            
            # Accumulate structure function for each order
            diff_power = diff.copy()
            for order in range(max_order):
                np.add.at(sf[order], bin_indices, diff_power)
                diff_power *= diff
    
    return sf, num_bin_points, weight_sum_bin

def get_sf(ad, var, weights='ones', xyz=None, max_order=10, npairs=1e7, nbins=100, log_bin_flag=True, 
           nsamples_block_min=1000, debug=False):
    """
    Calculate structure functions for a given field using GPU acceleration.
    
    This function computes structure functions up to a specified order for a 3D field
    using CUDA kernels. It handles periodic boundaries and uses a block-based sampling
    strategy with 1/r^2 weighting to improve statistics at small separations.
    
    Parameters
    ----------
    ad : AthenaData
        Athena data object containing grid information and field data
    var : str or ndarray
        Field variable name or array to analyze
    weights : str or ndarray, optional
        Weights for the structure function calculation, default='ones'
    xyz : array or None, optional
        Coordinates for the region of interest, default=None
    max_order : int, optional
        Maximum order of structure functions to compute (<=10), default=10
    npairs : float, optional
        Target number of point pairs to sample, default=1e7
    nbins : int, optional
        Number of radial bins for binning separations, default=100
    log_bin_flag : bool, optional
        Whether to use logarithmic binning in radius, default=True
    nsamples_block_min : int, optional
        Minimum number of samples per block, default=1000
    debug : bool, optional
        If True, print detailed debug messages during execution. Default=False
        
    Returns
    -------
    r_ : cp.ndarray
        Array of bin centers for separations
    sf : cp.ndarray
        Structure functions for each order and separation
        Shape: (max_order, nbins)
    weight_sum_bin : cp.ndarray
        Sum of weights in each radial bin
    num_bin_points : cp.ndarray
        Number of point pairs in each radial bin
        
    Notes
    -----
    - Uses a CUDA kernel on GPU, with a CPU (optionally Numba-JIT) fallback
    - Implements block-based sampling to manage memory usage
    - Weights point pairs by 1/r^2 to improve small-scale statistics
    - Handles periodic boundary conditions in all directions
    - Maximum order is limited to 10 to manage memory usage
    
    Examples
    --------
    >>> ad = AthenaData(num=100)
    >>> r, sf, weights, counts = get_sf(ad, 'velx', max_order=3, npairs=1e6)
    >>> plt.loglog(r.get(), sf[1].get()/weights.get())  # Plot second-order SF
    """
    
    if debug:
        print(f"Debug: Starting get_sf with var={var}, weights={weights}")
    
    try:
        # Get the data
        if debug:
            print(f"Debug: Loading data for variable: {var}")
        data = ad.data(var) if type(var) is str else var
        if debug:
            print(f"Debug: Successfully loaded data, shape: {data[0].shape if hasattr(data, '__getitem__') else 'N/A'}")
        
        if debug:
            print(f"Debug: Loading weight data: {weights}")
        weight_data = ad.data(weights) if type(weights) is str else weights
        if debug:
            print(f"Debug: Successfully loaded weight data")
        
        if xyz is not None:
            if debug:
                print(f"Debug: Applying xyz filtering: {xyz}")
            weight_data = weight_data * ad.data('xyzbool', xyz=xyz)
            if debug:
                print(f"Debug: Successfully applied xyz filtering")
        
        # Load coordinate arrays
        if debug:
            print(f"Debug: Loading coordinate arrays")
        x = ad.data('x')
        y = ad.data('y')
        z = ad.data('z')
        if debug:
            print(f"Debug: Successfully loaded coordinates")
        
        # Get basic simulation parameters
        nmbs = ad.n_mbs
        if debug:
            print(f"Debug: Number of mesh blocks: {nmbs}")
        
        Lx, Ly, Lz = xp.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min)])
        
        # Compute cell sizes from first meshblock geometry (uniform grid)
        # mb_geometry format: [x1min, x1max, x2min, x2max, x3min, x3max]
        first_mb_geom = ad.mb_geometry[0, :]
        dx = (first_mb_geom[1] - first_mb_geom[0]) / ad.nx1
        dy = (first_mb_geom[3] - first_mb_geom[2]) / ad.nx2
        dz = (first_mb_geom[5] - first_mb_geom[4]) / ad.nx3
        
        rmax = xp.sqrt(0.25*(Lx**2+Ly**2+Lz**2))
        rmin = xp.min(xp.array([dx, dy, dz]))
        
        # Ensure rmin > 0 for logarithmic binning
        if rmin <= 0:
            rmin = xp.min(xp.array([dx, dy, dz])[xp.array([dx, dy, dz]) > 0])
            if debug:
                print(f"Debug: Adjusted rmin to first positive cell size: {rmin}")
        
        if debug:
            print(f"Debug: Box parameters - Lx:{Lx}, Ly:{Ly}, Lz:{Lz}, rmax:{rmax}, rmin:{rmin}")
            print(f"Debug: Cell sizes - dx:{dx}, dy:{dy}, dz:{dz}")
        
        # Check if the BCs are periodic
        periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
        periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
        periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']
        if debug:
            print(f"Debug: Periodic boundaries - x:{periodic_x}, y:{periodic_y}, z:{periodic_z}")
        
        # Make rbins
        if log_bin_flag:
            rbins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins+1)
            r_ = xp.sqrt(rbins[1:]*rbins[:-1])
        else:
            rbins = xp.linspace(0, rmax, nbins+1)
            r_ = 0.5*(rbins[1:]+rbins[:-1])
        if debug:
            print(f"Debug: Created {nbins} radial bins, log_bin_flag={log_bin_flag}")
        
        # We go over each possible pairs of blocks in a nested triangular for loop, 
        # choosing n_samples1 points in the first block and n_samples2 points in 
        # the second block. We then calculate the difference in the variable at 
        # these points and bin the results
        if max_order > 10:
            print('Warning: max_order is greater than 10, consider reducing the order to 10')
            max_order = 10
        
        n_samples1 = int(xp.sqrt(npairs)/nmbs)
        n_points_per_block = data[0].size
        n_samples1 = int(xp.min(xp.array((n_samples1, n_points_per_block))))
        if debug:
            print(f"Debug: Sampling parameters - n_samples1:{n_samples1}, n_points_per_block:{n_points_per_block}")
        
        # Send a warning if n_samples1 is too small
        if n_samples1 < nsamples_block_min:
            print(f'Warning: n_samples1 = {n_samples1} is less than {nsamples_block_min}, consider increasing the number of pairs')
        
        # Get geometry for block centers
        # In MPI mode with distributed data, use local slice; otherwise use full geometry
        if hasattr(ad, 'has_full_data') and not ad.has_full_data:
            # MPI distributed mode - use local slice
            mbl = ad.local_mb_start
            mbh = ad.local_mb_end
            local_mb_geometry = ad.mb_geometry[mbl:mbh, :]
        else:
            # Single process mode - nmbs equals total blocks, use all geometry
            local_mb_geometry = ad.mb_geometry[:nmbs, :]
        
        block_centers = xp.zeros((nmbs, 3))  # Store all block centers
        # Calculate block centers using array operations
        block_centers[:, 0] = 0.5 * xp.asarray(local_mb_geometry[:, 0] + local_mb_geometry[:, 1])
        block_centers[:, 1] = 0.5 * xp.asarray(local_mb_geometry[:, 2] + local_mb_geometry[:, 3])
        block_centers[:, 2] = 0.5 * xp.asarray(local_mb_geometry[:, 4] + local_mb_geometry[:, 5])
        if debug:
            print(f"Debug: Successfully created block centers, shape: {block_centers.shape}")
        
        # Create mesh of block indices
        b1_indices, b2_indices = xp.triu_indices(nmbs)
        if debug:
            print(f"Debug: Created block indices, total pairs: {len(b1_indices)}")
        
        # Calculate block distances vectorized
        b1b2_x = xp.abs(block_centers[b2_indices, 0] - block_centers[b1_indices, 0])
        b1b2_y = xp.abs(block_centers[b2_indices, 1] - block_centers[b1_indices, 1])
        b1b2_z = xp.abs(block_centers[b2_indices, 2] - block_centers[b1_indices, 2])
        
        # Handle periodic boundaries vectorized
        if periodic_x:
            b1b2_x = xp.minimum(b1b2_x, xp.abs(Lx - b1b2_x))
        if periodic_y:
            b1b2_y = xp.minimum(b1b2_y, xp.abs(Ly - b1b2_y))
        if periodic_z:
            b1b2_z = xp.minimum(b1b2_z, xp.abs(Lz - b1b2_z))
        
        block_distances = xp.sqrt(b1b2_x**2 + b1b2_y**2 + b1b2_z**2)
        if debug:
            print(f"Debug: Successfully calculated block distances")
        
        # Calculate n_samples2 for all block pairs vectorized
        n_samples2_norm = n_samples1/rmax/(rmax/rmin-1.0)
        # Suppress divide-by-zero warning (handled by xp.where)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            n_samples2_blocks = n_samples2_norm * xp.where(block_distances == 0, n_samples1, xp.power(block_distances/rmax, -2.0))
        n_samples2_blocks = xp.clip(n_samples2_blocks, nsamples_block_min, n_samples1).astype(int)
        if debug:
            print(f"Debug: Successfully calculated sampling weights")
        
        sf = xp.zeros(max_order * nbins, dtype=xp.float64)  # Flattened array
        num_bin_points = xp.zeros(nbins, dtype=xp.float64)
        weight_sum_bin = xp.zeros(nbins, dtype=xp.float64)
        if debug:
            print(f"Debug: Initialized result arrays")
        
        # Process each block pair
        if debug:
            print(f"Debug: Starting to process {len(b1_indices)} block pairs")
        
        for idx in range(len(b1_indices)):
            b1, b2 = b1_indices[idx], b2_indices[idx]
            
            if debug and idx % 100 == 0:  # Print every 100th iteration to avoid spam
                if debug:
                    print(f"Debug: Processing block pair {idx}/{len(b1_indices)}: ({b1}, {b2})")
            
            # Get flattened arrays and convert to contiguous memory
            data1 = xp.ascontiguousarray(data[b1].flatten(), dtype=xp.float64)
            data2 = xp.ascontiguousarray(data[b2].flatten(), dtype=xp.float64)
            weight1 = xp.ascontiguousarray(weight_data[b1].flatten(), dtype=xp.float64)
            weight2 = xp.ascontiguousarray(weight_data[b2].flatten(), dtype=xp.float64)
            x1 = xp.ascontiguousarray(x[b1].flatten(), dtype=xp.float64)
            y1 = xp.ascontiguousarray(y[b1].flatten(), dtype=xp.float64)
            z1 = xp.ascontiguousarray(z[b1].flatten(), dtype=xp.float64)
            x2 = xp.ascontiguousarray(x[b2].flatten(), dtype=xp.float64)
            y2 = xp.ascontiguousarray(y[b2].flatten(), dtype=xp.float64)
            z2 = xp.ascontiguousarray(z[b2].flatten(), dtype=xp.float64)
            sf = xp.ascontiguousarray(sf, dtype=xp.float64)
            num_bin_points = xp.ascontiguousarray(num_bin_points, dtype=xp.float64)
            weight_sum_bin = xp.ascontiguousarray(weight_sum_bin, dtype=xp.float64)

            # Configure CUDA grid
            threadsperblock = 256
            n_pairs = int(asnumpy(n_samples1 * n_samples2_blocks[idx]))  # Convert to host integer
            blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock

            if debug and idx < 5:  # Print kernel info for first few iterations
                if debug:
                    print(f"Debug: Kernel config for pair {idx} - n_pairs:{n_pairs}, blocks:{blockspergrid}, threads:{threadsperblock}")

            # Check if GPU kernel is available
            if not cupy_enabled or structure_function_kernel is None:
                # Use CPU fallback
                if debug:
                    accel_msg = "with Numba JIT" if NUMBA_AVAILABLE else "pure NumPy (slow - install numba for 10-100x speedup)"
                    if debug:
                        print(f"Debug: Using CPU fallback for structure function computation ({accel_msg})")
                
                # Convert to NumPy arrays
                data1_cpu = asnumpy(data1)
                data2_cpu = asnumpy(data2)
                weight1_cpu = asnumpy(weight1)
                weight2_cpu = asnumpy(weight2)
                x1_cpu = asnumpy(x1)
                y1_cpu = asnumpy(y1)
                z1_cpu = asnumpy(z1)
                x2_cpu = asnumpy(x2)
                y2_cpu = asnumpy(y2)
                z2_cpu = asnumpy(z2)
                rbins_cpu = asnumpy(rbins)
                
                # Compute structure function on CPU using vectorized implementation
                # Note: use_mpi=False because get_sf should never be called with MPI enabled
                # When MPI is needed, set_sf automatically calls get_sf_mb instead
                # Use deterministic seed matching GPU: idx * 1013904223, wrapped
                # to uint32 range (matches the GPU kernel's native uint32 wraparound).
                cpu_seed = int((idx * 1013904223) % (2**32))
                sf_cpu, num_cpu, weights_cpu = _compute_sf_cpu_vectorized(
                    data1_cpu, data2_cpu,
                    weight1_cpu, weight2_cpu,
                    x1_cpu, y1_cpu, z1_cpu,
                    x2_cpu, y2_cpu, z2_cpu,
                    rbins_cpu, nbins, max_order,
                    float(Lx), float(Ly), float(Lz),
                    periodic_x, periodic_y, periodic_z,
                    n_pairs, use_mpi=False, random_seed=cpu_seed
                )
                
                # Accumulate results - flatten sf_cpu to match sf shape
                sf += sf_cpu.flatten()
                num_bin_points += num_cpu
                weight_sum_bin += weights_cpu
                
                if debug:
                    print(f"Debug: CPU fallback computation completed for block pair {idx}")
            else:
                # Launch kernel with deterministic base seed
                # Use idx (block pair index) as base seed for reproducibility,
                # wrapped to uint32 range to match the kernel's native arithmetic.
                base_seed = np.uint32((idx * 1013904223) % (2**32))
                structure_function_kernel(
                    (blockspergrid,), (threadsperblock,),
                    (data1, data2,
                    weight1, weight2,
                    x1, y1, z1,
                    x2, y2, z2,
                    rbins, 
                    sf, num_bin_points,
                    weight_sum_bin, n_pairs, 
                    n_points_per_block, n_points_per_block,
                    float(Lx), float(Ly), float(Lz),
                    nbins, max_order,
                    periodic_x, periodic_y, periodic_z,
                    base_seed))
                    
                # Synchronize after kernel launch
                xp.cuda.Stream.null.synchronize()
        
        if debug:
            print(f"Debug: Completed processing all block pairs")
        
        # reshape sf into a 2D array
        sf = sf.reshape((max_order, nbins))
        
        if debug:
            print(f"Debug: Reshaped sf to final shape: {sf.shape}")
            if debug:
                print(f"Debug: Total points in bins: {num_bin_points.sum()}")
            if debug:
                print(f"Debug: Returning results")

        return r_, sf, weight_sum_bin, xp.array(num_bin_points, dtype=int)
    
    except Exception as e:
        print(f"Error: Top-level error in get_sf: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise


def get_sf_mb(ad, varlist, weights='ones', xyz=None, max_order=10, npairs=1e7, nbins=100, log_bin_flag=True, 
            nsamples_block_min=1000, sparse_data=False, debug=False, simultaneous_blocks=None, mpi_manager=None):
    """
    Calculate structure functions for multiple variables using GPU acceleration with optimized batch processing.
    
    This function computes structure functions up to a specified order for multiple fields
    using CUDA kernels. It handles periodic boundaries and uses a block-based sampling
    strategy with 1/r^2 weighting to improve statistics at small separations.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varlist : list of str 
        list of Field variable names 
    weights : str or list of str, optional
        Weights for the structure function calculation. Can be a single weight name or 
        a list of weight names. Default='ones'
    xyz : array or None, optional
        Coordinates for the region of interest, default=None
    max_order : int, optional
        Maximum order of structure functions to compute (<=10), default=10
    npairs : float, optional
        Target number of point pairs to sample, default=1e7
    nbins : int, optional
        Number of radial bins for binning separations, default=100
    log_bin_flag : bool, optional
        Whether to use logarithmic binning in radius, default=True
    nsamples_block_min : int, optional
        Minimum number of samples per block, default=1000
    sparse_data : bool, optional
        If True, filter data points based on valid weights. If False, use all data points
        in each block. Helps with datasets that have many zero-weight points. Default=False
    debug : bool, optional
        If True, print detailed debug messages during execution. Default=False
    simultaneous_blocks : int, optional
        Number of blocks to process per batch. If None, automatically determined based on 
        available memory. Default=None
    
    Returns
    -------
    r_ : xp.ndarray
        Array of bin centers for separations
    sf : xp.ndarray
        Structure functions for each weight, order, variable and separation
    weight_sum_bin : xp.ndarray
        Sum of weights in each radial bin for each weighting scheme
    num_bin_points : xp.ndarray
        Number of point pairs in each radial bin for each weighting scheme
    """
    
    # Establish rank early for debug output
    rank = mpi_manager.rank if mpi_manager else 0
    
    if debug and rank == 0:
        print(f"[DEBUG get_sf_mb] Starting with varlist={varlist}, weights={weights}")
        if simultaneous_blocks is not None:
            print(f"[DEBUG get_sf_mb] Using manual batch size={simultaneous_blocks}")
    
    # Get the domain information
    # For MPI distributed data, use local meshblock count
    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs = ad.local_mb_end - ad.local_mb_start
        mb_offset = ad.local_mb_start
    else:
        nmbs = ad.n_mbs
        mb_offset = 0
    
    no_vars = len(varlist)
    if debug and rank == 0:
        print(f"[DEBUG get_sf_mb] Number of mesh blocks: {nmbs} (local), Number of variables: {no_vars}")
        if mb_offset > 0:
            print(f"[DEBUG get_sf_mb] Meshblock offset: {mb_offset}")
    
    Lx, Ly, Lz = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min)])
    
    # Compute cell sizes from first meshblock geometry (uniform grid)
    # mb_geometry format: [x1min, x1max, x2min, x2max, x3min, x3max]
    first_mb_geom = ad.mb_geometry[0, :]
    dx = (first_mb_geom[1] - first_mb_geom[0]) / ad.nx1
    dy = (first_mb_geom[3] - first_mb_geom[2]) / ad.nx2
    dz = (first_mb_geom[5] - first_mb_geom[4]) / ad.nx3
    
    rmax = np.sqrt(0.25*(Lx**2+Ly**2+Lz**2))
    rmin_local = np.min(np.array([dx, dy, dz]))
    
    # Ensure rmin > 0 for logarithmic binning
    if rmin_local <= 0:
        rmin_local = np.min(np.array([dx, dy, dz])[np.array([dx, dy, dz]) > 0])
        if debug and rank == 0:
            print(f"[DEBUG get_sf_mb] Adjusted rmin to first positive cell size: {rmin_local}")
    
    # Use MPI global minimum to ensure consistency across ranks
    if mpi_manager is not None:
        rmin = mpi_manager.allreduce(rmin_local, op='min')
        if debug and rank == 0:
            print(f"[DEBUG get_sf_mb] Global rmin from MPI allreduce: {rmin} (local was {rmin_local})")
    else:
        rmin = rmin_local
    
    if debug and rank == 0:
        print(f"[DEBUG get_sf_mb] Box parameters - Lx:{Lx:.3f}, Ly:{Ly:.3f}, Lz:{Lz:.3f}, rmax:{rmax:.3f}, rmin:{rmin:.3f}")
        print(f"[DEBUG get_sf_mb] Cell sizes - dx:{dx:.6f}, dy:{dy:.6f}, dz:{dz:.6f}")
    
    # Check if boundaries are periodic
    periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
    periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
    periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']
    if debug and rank == 0:
        print(f"[DEBUG get_sf_mb] Periodic boundaries - x:{periodic_x}, y:{periodic_y}, z:{periodic_z}")
    
    # Handle weights parameter - convert to list if a single weight is provided
    weights_list = weights if isinstance(weights, list) else [weights]
    n_weights = len(weights_list)
    if debug and rank == 0:
        print(f"[DEBUG get_sf_mb] Processing {n_weights} weight schemes: {weights_list}")
    
    # Validate and adjust max_order
    if max_order > 10:
        print('Warning: max_order is greater than 10, setting to 10')
        max_order = 10
    
    # Calculate sampling parameters
    # IMPORTANT: For MPI mode, use GLOBAL block count
    # This ensures consistent sampling rates across all configurations
    n_samples1 = int(np.sqrt(npairs)/ad.n_mbs)  # Use global block count
    if n_samples1 < nsamples_block_min:
        print(f'Warning: n_samples1 = {n_samples1} is less than {nsamples_block_min}, consider increasing npairs')
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
    n_samples1 = min(n_samples1, n_points_per_block)
    if debug:
        print(f"Debug: Sampling parameters - n_samples1:{n_samples1}, n_points_per_block:{n_points_per_block}")
    
    # Use only local geometry for block centers
    block_centers = np.zeros((nmbs, 3))
    local_geometry = ad.mb_geometry[:nmbs, :]
    block_centers[:,0] = 0.5 * (local_geometry[:,0]+local_geometry[:,1])
    block_centers[:,1] = 0.5 * (local_geometry[:,2]+local_geometry[:,3])
    block_centers[:,2] = 0.5 * (local_geometry[:,4]+local_geometry[:,5])
    if debug:
        print(f"Debug: Successfully created block centers, shape: {block_centers.shape}")
    
    # Create radial bins
    if log_bin_flag:
        rbins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins+1)
        r_ = xp.sqrt(rbins[1:]*rbins[:-1])
    else:
        rbins = xp.linspace(0, rmax, nbins+1)
        r_ = 0.5*(rbins[1:]+rbins[:-1])
    if debug:
        print(f"Debug: Created {nbins} radial bins, log_bin_flag={log_bin_flag}")
    
    # Initialize output arrays
    sf = xp.zeros((n_weights, no_vars, max_order, nbins), dtype=xp.float64)
    num_bin_points = xp.zeros((n_weights, nbins), dtype=xp.float64)
    weight_sum_bin = xp.zeros((n_weights, nbins), dtype=xp.float64)
    if debug:
        print(f"Debug: Successfully initialized result arrays with shapes:")
        if debug:
            print(f"Debug:   sf: {sf.shape}")
        if debug:
            print(f"Debug:   num_bin_points: {num_bin_points.shape}")
        if debug:
            print(f"Debug:   weight_sum_bin: {weight_sum_bin.shape}")
    
    # Determine blocks per batch based on available memory or manual setting
    if simultaneous_blocks is not None:
        # Use manual batch size, but validate it
        blocks_per_batch = min(simultaneous_blocks, nmbs)
        if debug:
            print(f"Debug: Using manual batch_size={simultaneous_blocks}, blocks_per_batch={blocks_per_batch}")
        
        # Warn if batch size might be too large for memory
        if simultaneous_blocks > nmbs // 2:
            print(f"Warning: Large batch_size ({simultaneous_blocks}) may cause memory issues")
    else:
        # Use automatic determination with 3x safety factor for no_vars
        # to account for temporary arrays and CUDA kernel workspace
        blocks_per_batch = determine_blocks_per_batch(nmbs, no_vars * 3, n_weights, n_points_per_block)
        if debug:
            print(f"Debug: Using automatic batch sizing, blocks_per_batch={blocks_per_batch}")
    
    # Print an error if blocks_per_batch is zero
    if blocks_per_batch == 0:
        raise ValueError("Error: blocks_per_batch is zero. Please check the input parameters or reduce batch_size.") 
    
    # Keep track of which block pairs we've processed
    processed_pairs = set()
    if debug:
        print(f"Debug: Created empty processed_pairs set")
    
    # Block cache to store loaded data
    block_cache = {}
    
    # Determine if we're using MPI
    use_mpi = mpi_manager is not None
    
    # Different strategies for MPI vs non-MPI mode
    if use_mpi and mpi_manager is not None:
        # MPI MODE: Distribute GLOBAL block pairs across ranks
        if debug or mpi_manager.rank == 0:
            print(f"MPI Mode: Computing structure functions across ALL {ad.n_mbs} global blocks")
        
        # Synchronize all ranks before starting
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Synchronizing before batch processing")
        mpi_manager.comm.Barrier()
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: All ranks synchronized")
        
        total_global_blocks = ad.n_mbs
        if debug:
            total_pairs = total_global_blocks * (total_global_blocks + 1) // 2
            print(f"Debug: Rank {mpi_manager.rank}: Total global pairs: {total_pairs}")
        
        # Distribute pairs across ranks using round-robin without building the full global pair list.
        my_pairs = _build_rank_assigned_pairs(
            total_global_blocks, mpi_manager.rank, mpi_manager.size)
        
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Assigned {len(my_pairs)} pairs to process")
            if len(my_pairs) > 0:
                if debug:
                    print(f"Debug: Rank {mpi_manager.rank}: First pair: {my_pairs[0]}, Last pair: {my_pairs[-1]}")
        
        # Process assigned pairs
        _process_mpi_pairs(ad, my_pairs, block_cache, weights_list, varlist,
                          sf, num_bin_points, weight_sum_bin,
                          rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                          n_samples1, n_points_per_block, nsamples_block_min,
                          periodic_x, periodic_y, periodic_z, sparse_data, xyzlim=xyz,
                          mpi_manager=mpi_manager, mb_offset=mb_offset, debug=debug)
    else:
        # NON-MPI MODE: Process local blocks only
        # Process blocks in batches
        num_batches = (nmbs + blocks_per_batch - 1) // blocks_per_batch
        batches = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * blocks_per_batch
            batch_end = min(batch_start + blocks_per_batch, nmbs)
            batches.append(list(range(batch_start, batch_end)))
        if debug:
            print(f"Debug: Created {num_batches} batches with blocks_per_batch={blocks_per_batch}")
        
        # Process batch pairs
        if debug:
            print(f"Debug: Starting batch pair processing")
        _process_batch_pairs(ad, batches, block_cache, processed_pairs, weights_list, varlist, 
                                sf, num_bin_points, weight_sum_bin,
                                block_centers, rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                                n_samples1, n_points_per_block, nsamples_block_min, 
                                periodic_x, periodic_y, periodic_z, sparse_data, xyzlim=xyz, 
                                use_mpi=False, mpi_manager=None, mb_offset=0)
        if debug:
            print(f"Debug: Completed batch pair processing")
    
    # Handle any missed pairs (only in non-MPI mode)
    # In MPI mode, all pairs are handled via round-robin distribution
    if not use_mpi:
        total_pairs = (nmbs * (nmbs + 1)) // 2
        if len(processed_pairs) < total_pairs:
            if debug:
                print(f"Debug: Processing missed pairs - {len(processed_pairs)} of {total_pairs} completed")
            print(f"WARNING: Only processed {len(processed_pairs)} of {total_pairs} block pairs")
            print("Processing any missed pairs...")
            
            # Find any missing pairs
            missing_pairs = []
            for i in range(nmbs):
                for j in range(i, nmbs):
                    pair = (i, j)
                    if pair not in processed_pairs:
                        missing_pairs.append(pair)
            
            if missing_pairs:
                if debug:
                    print(f"Debug: Found {len(missing_pairs)} missing pairs")
                # Create batches from missing pairs - each pair becomes its own mini-batch
                missing_batches = [[p[0]] for p in missing_pairs] + [[p[1]] for p in missing_pairs]
                # Remove duplicates while preserving order
                unique_missing_batches = []
                seen = set()
                for batch in missing_batches:
                    block = batch[0]
                    if block not in seen:
                        seen.add(block)
                        unique_missing_batches.append(batch)
                        
                # Process the missing batches in the same way as regular batches
                _process_batch_pairs(ad, unique_missing_batches, block_cache, processed_pairs, 
                                        weights_list, varlist, sf, num_bin_points, weight_sum_bin,
                                        block_centers, rmax, rmin, Lx, Ly, Lz, rbins, 
                                        nbins, max_order, n_samples1, n_points_per_block, 
                                        nsamples_block_min, periodic_x, periodic_y, periodic_z, 
                                        sparse_data, xyzlim=xyz, debug=debug)
                if debug:
                    print(f"Debug: Completed processing missed pairs")
    
    # Final sync
    if cupy_enabled:
        xp.cuda.Stream.null.synchronize()
    
    # Reshape output to match input format
    if not isinstance(weights, list):
        # Return just the first weight's data (no weights dimension)
        sf = sf[0]
        num_bin_points = num_bin_points[0]
        weight_sum_bin = weight_sum_bin[0]
        if debug:
            print(f"Debug: Reshaped output for single weight, sf shape: {sf.shape}")
    
    if debug:
        print(f"Debug: Total points in bins: {num_bin_points.sum()}")
        if debug:
            print(f"Debug: Returning structure function results")
    
    # Gather results from all MPI ranks if using MPI
    if mpi_manager is not None:
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Aggregating results across {mpi_manager.size} ranks")
        
        # Convert to numpy for MPI communication
        sf_np = asnumpy(sf)
        num_bin_points_np = asnumpy(num_bin_points)
        weight_sum_bin_np = asnumpy(weight_sum_bin)
        
        # Sum results across all ranks
        sf_total = mpi_manager.reduce(sf_np, op='sum', root=0)
        num_bin_points_total = mpi_manager.reduce(num_bin_points_np, op='sum', root=0)
        weight_sum_bin_total = mpi_manager.reduce(weight_sum_bin_np, op='sum', root=0)
        
        # Broadcast aggregated results to all ranks
        sf_total = mpi_manager.broadcast(sf_total, root=0)
        num_bin_points_total = mpi_manager.broadcast(num_bin_points_total, root=0)
        weight_sum_bin_total = mpi_manager.broadcast(weight_sum_bin_total, root=0)
        
        # Convert back to GPU arrays if using CuPy
        sf = xp.asarray(sf_total)
        num_bin_points = xp.asarray(num_bin_points_total)
        weight_sum_bin = xp.asarray(weight_sum_bin_total)
        
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: After aggregation, total points: {num_bin_points.sum()}")

    if debug:
        print(f"Debug: Total points in bins: {num_bin_points.sum()}")
    return r_, sf, weight_sum_bin, xp.array(num_bin_points, dtype=int)

def _process_mpi_pairs(ad, my_pairs, block_cache, weights_list, varlist,
                      sf, num_bin_points, weight_sum_bin,
                      rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                      n_samples1, n_points_per_block, nsamples_block_min,
                      periodic_x, periodic_y, periodic_z, sparse_data=False, xyzlim=None,
                      mpi_manager=None, mb_offset=0, debug=False):
    """
    Process assigned block pairs in MPI mode with memory-efficient batching.
    
    PAIR COVERAGE GUARANTEE:
    - Caller generates ALL pairs (i,j) where 0 <= i <= j < n_mbs
    - Pairs are distributed round-robin: rank r processes pairs where idx % size == r
    - This ensures every pair is processed by exactly one rank
    - Results from all ranks are aggregated via MPI reduce after processing
    
    MEMORY MANAGEMENT:
    - Pairs processed in batches sized by determine_blocks_per_batch()
    - Batch size calculated from actual GPU/CPU memory and block dimensions
    - Blocks loaded/unloaded per batch to respect memory constraints
    
    OPTIMIZATION:
    - Pairs reordered to minimize remote block fetching:
      1. Both blocks local (no MPI needed)
      2. One block local (minimal MPI)
      3. Both blocks remote (maximum MPI)
    """
    rank = mpi_manager.rank if mpi_manager else 0
    size = mpi_manager.size if mpi_manager else 1
    
    if debug:
        print(f"Debug: Rank {rank}: Processing {len(my_pairs)} assigned pairs")
    
    # Get local block range for this rank
    local_start = mb_offset
    # Calculate number of local meshblocks
    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs_local = ad.local_mb_end - ad.local_mb_start
    else:
        nmbs_local = ad.n_mbs
    local_end = mb_offset + nmbs_local
    
    if debug:
        print(f"Debug: Rank {rank}: Local blocks [{local_start}, {local_end})")
    
    # DON'T filter pairs - process ALL assigned pairs
    # We'll load remote blocks via MPI as needed
    
    # Compute block centers for ALL GLOBAL blocks (needed for sampling weights)
    # Each rank computes this independently - no communication needed
    block_centers_global = np.zeros((ad.n_mbs, 3))
    full_geometry = ad.mb_geometry[:ad.n_mbs, :]
    block_centers_global[:, 0] = 0.5 * (full_geometry[:, 0] + full_geometry[:, 1])
    block_centers_global[:, 1] = 0.5 * (full_geometry[:, 2] + full_geometry[:, 3])
    block_centers_global[:, 2] = 0.5 * (full_geometry[:, 4] + full_geometry[:, 5])
    
    if debug:
        print(f"Debug: Rank {rank}: Created global block_centers with shape {block_centers_global.shape}")
        if debug:
            print(f"Debug: Rank {rank}: Processing {len(my_pairs)} assigned pairs (including remote blocks)")
    
    # Determine batch size based on actual memory and block size
    # Use same memory estimation as non-MPI version
    from athena_research.utils.batch_processing import determine_blocks_per_batch
    
    # Calculate blocks per batch based on memory
    blocks_per_batch = determine_blocks_per_batch(ad.n_mbs, len(varlist) * 3, len(weights_list), n_points_per_block)
    
    if debug:
        print(f"Debug: Rank {rank}: Calculated blocks_per_batch={blocks_per_batch} based on memory constraints")
    
    # Get all unique blocks needed by this rank
    all_blocks_needed = set()
    for global_b1, global_b2 in my_pairs:
        all_blocks_needed.add(global_b1)
        all_blocks_needed.add(global_b2)
    
    # Separate local vs remote blocks
    local_blocks = [b for b in all_blocks_needed if local_start <= b < local_end]
    remote_blocks = [b for b in all_blocks_needed if b < local_start or b >= local_end]
    
    if debug:
        print(f"Debug: Rank {rank}: Need {len(local_blocks)} local and {len(remote_blocks)} remote blocks")
    
    # Optimize pair ordering: prioritize pairs with both blocks local,
    # then pairs with one local block, then pairs with both remote
    # This minimizes remote block fetching per batch
    pairs_by_locality = {
        'both_local': [],
        'one_local': [],
        'both_remote': []
    }
    
    for global_b1, global_b2 in my_pairs:
        b1_local = (local_start <= global_b1 < local_end)
        b2_local = (local_start <= global_b2 < local_end)
        
        if b1_local and b2_local:
            pairs_by_locality['both_local'].append((global_b1, global_b2))
        elif b1_local or b2_local:
            pairs_by_locality['one_local'].append((global_b1, global_b2))
        else:
            pairs_by_locality['both_remote'].append((global_b1, global_b2))
    
    # Reorder pairs to process local-heavy batches first
    optimized_pairs = (pairs_by_locality['both_local'] + 
                      pairs_by_locality['one_local'] + 
                      pairs_by_locality['both_remote'])
    
    if debug:
        print(f"Debug: Rank {rank}: Pair locality - both_local:{len(pairs_by_locality['both_local'])}, "
              f"one_local:{len(pairs_by_locality['one_local'])}, both_remote:{len(pairs_by_locality['both_remote'])}")
    
    
    # Process pairs in batches
    pairs_processed = 0
    pair_idx = 0
    
    while pair_idx < len(optimized_pairs):
        # Determine blocks needed for this batch of pairs
        batch_blocks_needed = set()
        batch_end_idx = pair_idx
        
        # Greedily add pairs to batch until we hit memory limit
        while batch_end_idx < len(optimized_pairs) and len(batch_blocks_needed) < blocks_per_batch:
            global_b1, global_b2 = optimized_pairs[batch_end_idx]
            
            # Check if adding this pair would exceed batch size
            new_blocks = set()
            if global_b1 not in block_cache:
                new_blocks.add(global_b1)
            if global_b2 not in block_cache:
                new_blocks.add(global_b2)
            
            # If this would exceed limit and we already have some blocks, stop
            if len(batch_blocks_needed) + len(new_blocks) > blocks_per_batch and len(batch_blocks_needed) > 0:
                break
            
            batch_blocks_needed.update(new_blocks)
            batch_end_idx += 1
        
        # Extract batch pairs
        batch_pairs = optimized_pairs[pair_idx:batch_end_idx]
        
        if debug:
            print(f"Debug: Rank {rank}: Processing batch with {len(batch_blocks_needed)} blocks, {len(batch_pairs)} pairs")
        
        # Load blocks for this batch
        if batch_blocks_needed:
            _load_all_blocks_upfront(ad, block_cache, list(batch_blocks_needed),
                                    weights_list, varlist, 
                                    local_start, local_end, xyzlim, mpi_manager, debug=False)
        
        # Process all pairs in this batch
        for global_b1, global_b2 in batch_pairs:
            if debug and pairs_processed % 10 == 0:
                if debug:
                    print(f"Debug: Rank {rank}: Processing pair {pairs_processed}/{len(optimized_pairs)}: ({global_b1}, {global_b2})")
            
            # Check if blocks are outside xyz limits
            if xyzlim is not None:
                if is_block_outside_xyz(ad.mb_geometry, global_b1, xyzlim) or \
                   is_block_outside_xyz(ad.mb_geometry, global_b2, xyzlim):
                    continue
            
            # Verify both blocks are loaded
            if global_b1 not in block_cache or global_b2 not in block_cache:
                if debug:
                    print(f"Debug: Rank {rank}: Warning - blocks ({global_b1}, {global_b2}) not in cache, skipping")
                continue
        
            # Process the pair using GLOBAL indices
            _process_block_pair(ad, global_b1, global_b2, block_cache, block_centers_global,
                               weights_list, varlist, sf, num_bin_points, weight_sum_bin,
                               rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                               n_samples1, n_points_per_block, nsamples_block_min,
                               periodic_x, periodic_y, periodic_z, sparse_data, debug=False, use_mpi=True)
            
            pairs_processed += 1
        
        # Clear cache for this batch to free memory
        for block_id in batch_blocks_needed:
            if block_id in block_cache:
                del block_cache[block_id]
        
        if debug:
            print(f"Debug: Rank {rank}: Batch complete, cleared {len(batch_blocks_needed)} blocks, processed {pairs_processed}/{len(optimized_pairs)} pairs so far")
        
        # Move to next batch
        pair_idx = batch_end_idx
    
    if debug:
        print(f"Debug: Rank {rank}: Completed processing all {pairs_processed} assigned pairs")

def _load_all_blocks_upfront(ad, block_cache, blocks_needed, weights_list, varlist,
                             local_start, local_end, xyzlim, mpi_manager, debug=False):
    """
    Load ALL blocks needed upfront in ONE collective MPI operation.
    
    This avoids deadlock by ensuring all ranks do MPI communication together.
    """
    rank = mpi_manager.rank if mpi_manager else 0
    
    # Separate local and remote blocks
    local_blocks = [b for b in blocks_needed if local_start <= b < local_end and b not in block_cache]
    remote_blocks = [b for b in blocks_needed if (b < local_start or b >= local_end) and b not in block_cache]
    
    if debug:
        print(f"Debug: Rank {rank}: Loading {len(local_blocks)} local, {len(remote_blocks)} remote blocks")
    
    _load_local_blocks_into_cache(
        ad, block_cache, local_blocks, weights_list, varlist, xyzlim,
        cast_to_float=True)
    
    # Exchange remote blocks via MPI (all ranks participate together)
    _exchange_remote_blocks_mpi(ad, block_cache, remote_blocks, weights_list, varlist,
                                local_start, local_end, xyzlim, mpi_manager, debug)

def _load_blocks_for_mpi_pair(ad, block_cache, blocks_needed, weights_list, varlist,
                               local_start, local_end, xyzlim, mpi_manager, debug=False):
    """
    Load blocks for MPI pair computation using collective exchange.
    
    Strategy: All ranks announce their needs, then exchange data in one collective operation.
    """
    # Separate local and remote blocks
    local_blocks = [b for b in blocks_needed if local_start <= b < local_end and b not in block_cache]
    remote_blocks = [b for b in blocks_needed if (b < local_start or b >= local_end) and b not in block_cache]
    
    if debug and (local_blocks or remote_blocks):
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Loading {len(local_blocks)} local blocks, {len(remote_blocks)} remote blocks")
    
    _load_local_blocks_into_cache(
        ad, block_cache, local_blocks, weights_list, varlist, xyzlim,
        cast_to_float=False)
    
    # Load remote blocks via MPI if any
    if remote_blocks:
        _exchange_remote_blocks_mpi(ad, block_cache, remote_blocks, weights_list, varlist,
                                    local_start, local_end, xyzlim, mpi_manager, debug)

def _exchange_remote_blocks_mpi(ad, block_cache, remote_blocks, weights_list, varlist,
                                local_start, local_end, xyzlim, mpi_manager, debug=False):
    """
    Exchange remote blocks using MPI with proper synchronization to avoid deadlock.
    """
    rank = mpi_manager.rank
    size = mpi_manager.size
    
    # Synchronize before starting MPI communication
    mpi_manager.comm.Barrier()
    
    # Determine block ownership
    mbs_per_rank = ad.n_mbs // size
    remainder = ad.n_mbs % size
    
    def get_owner_rank(global_block_idx):
        if global_block_idx < remainder * (mbs_per_rank + 1):
            return global_block_idx // (mbs_per_rank + 1)
        else:
            offset = remainder * (mbs_per_rank + 1)
            return remainder + (global_block_idx - offset) // mbs_per_rank
    
    # Collect all ranks' requests using allgather
    my_requests = {r: [] for r in range(size)}
    for block_idx in remote_blocks:
        owner = get_owner_rank(block_idx)
        my_requests[owner].append(block_idx)
    
    # Remove empty requests
    my_requests = {k: v for k, v in my_requests.items() if v}
    
    if debug and my_requests:
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Requesting {sum(len(v) for v in my_requests.values())} blocks from {len(my_requests)} ranks")
    
    # All ranks gather everyone's requests
    all_requests = mpi_manager.comm.allgather(my_requests)
    
    # Synchronize after allgather
    mpi_manager.comm.Barrier()
    
    # Prepare data that other ranks need from me
    blocks_to_send = {}
    for requesting_rank, their_requests in enumerate(all_requests):
        if rank in their_requests:
            blocks_to_send[requesting_rank] = their_requests[rank]
    
    if debug and blocks_to_send:
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Sending {sum(len(v) for v in blocks_to_send.values())} blocks to {len(blocks_to_send)} ranks")
    
    # Prepare data for blocks I own
    owned_blocks_requested = sorted({
        block_idx
        for block_list in blocks_to_send.values()
        for block_idx in block_list
        if local_start <= block_idx < local_end
    })
    serialized_blocks = _build_serialized_block_payloads(
        ad, owned_blocks_requested, weights_list, varlist, xyzlim)
    data_to_send = {}
    for dest_rank, block_list in blocks_to_send.items():
        data_to_send[dest_rank] = {
            block_idx: serialized_blocks[block_idx]
            for block_idx in block_list
            if block_idx in serialized_blocks
        }
    
    # Synchronize before communication
    mpi_manager.comm.Barrier()
    
    # Use coordinated send/recv to avoid deadlock
    # Strategy: lower-ranked processes send first, higher-ranked receive first
    received_data_all = {}
    
    for other_rank in range(size):
        if other_rank == rank:
            continue
        
        # Determine send/recv order based on rank ordering
        if rank < other_rank:
            # Lower rank sends first, then receives
            if other_rank in data_to_send:
                mpi_manager.comm.send(data_to_send[other_rank], dest=other_rank, tag=1000 + rank)
            if other_rank in my_requests:
                received_data = mpi_manager.comm.recv(source=other_rank, tag=1000 + other_rank)
                received_data_all.update(received_data)
        else:
            # Higher rank receives first, then sends
            if other_rank in my_requests:
                received_data = mpi_manager.comm.recv(source=other_rank, tag=1000 + other_rank)
                received_data_all.update(received_data)
            if other_rank in data_to_send:
                mpi_manager.comm.send(data_to_send[other_rank], dest=other_rank, tag=1000 + rank)
    
    # Store received data in cache, converting to GPU arrays if needed
    for global_b, block_data in received_data_all.items():
        # Convert numpy arrays to appropriate backend (GPU/CPU)
        block_cache[global_b] = {
            'coords': {
                'x': xp.asarray(block_data['coords']['x'], dtype=xp.float64),
                'y': xp.asarray(block_data['coords']['y'], dtype=xp.float64),
                'z': xp.asarray(block_data['coords']['z'], dtype=xp.float64)
            },
            'xyz_region_status': block_data.get('xyz_region_status', 'inside'),
            'xyz_region_trimmed': block_data.get('xyz_region_trimmed', False),
            'weights': {k: xp.asarray(v, dtype=xp.float64) for k, v in block_data['weights'].items()},
            'vars': {k: xp.asarray(v, dtype=xp.float64) for k, v in block_data['vars'].items()}
        }
    
    # Final synchronization
    mpi_manager.comm.Barrier()
    
    if debug:
        print(f"Debug: Rank {mpi_manager.rank}: MPI block exchange completed")

def _process_batch_pairs(ad, batches, block_cache, processed_pairs, weights_list, varlist, 
                    sf, num_bin_points, weight_sum_bin,
                    block_centers, rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                    n_samples1, n_points_per_block, nsamples_block_min, 
                    periodic_x, periodic_y, periodic_z, sparse_data=False, xyzlim=None, debug=False, 
                    use_mpi=False, mpi_manager=None, mb_offset=0):
    """
    Helper function to process batch pairs for structure functions.
    
    In MPI mode, this function computes SF for ALL block pairs involving at least one
    local block. For remote blocks (on other ranks), it uses MPI communication to 
    exchange data as needed.
    """
    
    if debug:
        print(f"Debug: Starting _process_batch_pairs with {len(batches)} batches")
    # Create a set of blocks to skip if they're outside xyz limits
    blocks_to_skip = set()
    if xyzlim is not None:
        for batch in batches:
            for block_idx in batch:
                if is_block_outside_xyz(ad.mb_geometry, block_idx, xyzlim):
                    blocks_to_skip.add(block_idx)
        if debug:
            print(f"Debug: Found {len(blocks_to_skip)} blocks outside xyz limits")
        if blocks_to_skip:
            print(f"Skipping {len(blocks_to_skip)} blocks outside xyz limits")
            # Mark all pairs involving skipped blocks as processed
            for b1 in blocks_to_skip:
                for b2 in range(ad.n_mbs):
                    # Ensure that smaller index is first in the pair
                    pair = (min(b1, b2), max(b1, b2))
                    processed_pairs.add(pair)
    if debug:
        print(f"Debug: Processing {len(batches)} batches with {len(blocks_to_skip)} blocks to skip")
    # Iterate over batches
    for batch_idx1 in range(len(batches)):
        batch1 = batches[batch_idx1]
        
        # Filter batch1 to exclude blocks outside xyz limits and ensure integers
        batch1 = [b for b in batch1 if b not in blocks_to_skip]
        if not batch1:
            continue  # Skip empty batches
            
        # Load data for batch1
        _load_batch_data(ad, block_cache, batch1, weights_list, varlist, xyzlim=xyzlim,
                        mpi_manager=mpi_manager if use_mpi else None, mb_offset=mb_offset if use_mpi else 0)
        if debug:
            print(f"Debug: Loaded data for batch {batch_idx1}, blocks: {batch1}") 
        # Process intra-batch pairs
        for i in range(len(batch1)):
            for j in range(i, len(batch1)):
                b1, b2 = batch1[i], batch1[j]  # Ensure int conversion
                pair = (b1, b2)
                
                # In MPI mode, only process if this rank should handle this pair
                if use_mpi and mpi_manager is not None:
                    # Convert local indices to global indices
                    global_b1 = b1 + mb_offset
                    global_b2 = b2 + mb_offset
                    # Deterministic assignment: rank owning the first block processes the pair
                    owning_rank = global_b1 // ((ad.n_mbs + mpi_manager.size - 1) // mpi_manager.size)
                    if owning_rank != mpi_manager.rank:
                        if debug:
                            print(f"Debug: Rank {mpi_manager.rank}: Skipping pair ({global_b1}, {global_b2}) - assigned to rank {owning_rank}")
                        continue
                    if debug:
                        print(f"Debug: Rank {mpi_manager.rank}: Processing intra-batch pair ({global_b1}, {global_b2})")
                elif debug:
                    print(f"Debug: Processing intra-batch pair ({b1}, {b2})")
                
                if pair not in processed_pairs:
                    _process_block_pair(ad, b1, b2, block_cache, block_centers, weights_list, varlist,
                                        sf, num_bin_points, weight_sum_bin, 
                                        rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                                        n_samples1, n_points_per_block, nsamples_block_min,
                                        periodic_x, periodic_y, periodic_z, sparse_data, use_mpi=use_mpi)
                    processed_pairs.add(pair)
        
        # Process cross-batch pairs
        for batch_idx2 in range(batch_idx1 + 1, len(batches)):
            batch2 = batches[batch_idx2]
            
            # Filter batch2 to exclude blocks outside xyz limits and ensure integers
            batch2 = [b for b in batch2 if b not in blocks_to_skip]
            if not batch2:
                continue  # Skip empty batches
                
            # Load data for batch2
            _load_batch_data(ad, block_cache, batch2, weights_list, varlist, xyzlim=xyzlim,
                            mpi_manager=mpi_manager if use_mpi else None, mb_offset=mb_offset if use_mpi else 0)
            
            # Process all pairs between batch1 and batch2
            for b1 in batch1:
                for b2 in batch2:
                    pair = (b1, b2)
                    
                    # In MPI mode, only process if this rank should handle this pair
                    if use_mpi and mpi_manager is not None:
                        # Convert local indices to global indices
                        global_b1 = b1 + mb_offset
                        global_b2 = b2 + mb_offset
                        # Deterministic assignment: rank owning the first block processes the pair
                        owning_rank = global_b1 // ((ad.n_mbs + mpi_manager.size - 1) // mpi_manager.size)
                        if owning_rank != mpi_manager.rank:
                            continue
                    
                    if pair not in processed_pairs:
                        _process_block_pair(ad, b1, b2, block_cache, block_centers, weights_list, varlist,
                                            sf, num_bin_points, weight_sum_bin, 
                                            rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                                            n_samples1, n_points_per_block, nsamples_block_min,
                                            periodic_x, periodic_y, periodic_z, sparse_data, use_mpi=use_mpi)
                        processed_pairs.add(pair)
            
            # Clear batch2 data from cache
            for b in batch2:
                if b in block_cache:
                    del block_cache[b]
        
        # Clear batch1 data from cache
        for b in batch1:
            if b in block_cache:
                del block_cache[b]
        
        # Clean up GPU memory
        if cupy_enabled:
            xp.cuda.Stream.null.synchronize()
            xp.get_default_memory_pool().free_all_blocks()


def _load_batch_data(ad, block_cache, batch_blocks, weights_list, varlist, xyzlim=None, 
                     mpi_manager=None, mb_offset=0):
    """
    Load data for multiple blocks into the cache in a single efficient operation,
    skipping blocks that are already cached.
    
    In MPI mode with cross-rank pairs:
    - LOCAL blocks (within mb_offset range): loaded from local coordinate arrays
    - REMOTE blocks (outside mb_offset range): exchanged via MPI from owning rank
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    block_cache : dict
        Dictionary to store the loaded data
    batch_blocks : list of int
        List of GLOBAL block indices to load
    weights_list : list of str
        List of weight variables to load
    varlist : list of str
        List of variables to load
    xyzlim : array, optional
        Coordinates for the region of interest
    mpi_manager : MPIManager, optional
        MPI manager for cross-rank communication
    mb_offset : int, optional
        Global starting index of this rank's local meshblocks
    """
    if not batch_blocks:
        return
    
    # Filter out blocks that are already in the cache
    blocks_to_load = [b for b in batch_blocks if b not in block_cache]
    
    if not blocks_to_load:
        # All blocks are already in the cache
        return
    
    # In MPI mode, separate local and remote blocks
    if mpi_manager is not None:
        # Need to check local block range
        local_start = mb_offset
        if hasattr(ad, 'has_full_data') and not ad.has_full_data:
            local_end = mb_offset + (ad.local_mb_end - ad.local_mb_start)
        else:
            local_end = mb_offset + ad.n_mbs
        local_blocks = [b for b in blocks_to_load if local_start <= b < local_end]
        remote_blocks = [b for b in blocks_to_load if b < local_start or b >= local_end]
    else:
        local_blocks = blocks_to_load
        remote_blocks = []
    
    _load_local_blocks_into_cache(
        ad, block_cache, local_blocks, weights_list, varlist, xyzlim,
        cast_to_float=True)
    
    # Handle remote blocks via MPI
    if remote_blocks and mpi_manager is not None:
        _load_remote_blocks_mpi(ad, block_cache, remote_blocks, weights_list, varlist, 
                               mpi_manager, xyzlim)


def _classify_xyz_block_region(mb_geometry, block_idx, xyzlim):
    """Classify a meshblock as outside, partial, or inside using block bounds."""
    if xyzlim is None:
        return 'inside'
    x1min, x1max, x2min, x2max, x3min, x3max = mb_geometry[block_idx]
    if (x1max < xyzlim[0] or x1min > xyzlim[1] or
        x2max < xyzlim[2] or x2min > xyzlim[3] or
        x3max < xyzlim[4] or x3min > xyzlim[5]):
        return 'outside'
    if (x1min >= xyzlim[0] and x1max <= xyzlim[1] and
        x2min >= xyzlim[2] and x2max <= xyzlim[3] and
        x3min >= xyzlim[4] and x3max <= xyzlim[5]):
        return 'inside'
    return 'partial'


def _prepare_xyz_block_regions(ad, block_indices, xyzlim):
    """Classify blocks by bounds."""
    region_status_map = {}
    if xyzlim is None:
        for block_idx in block_indices:
            region_status_map[block_idx] = 'inside'
        return region_status_map

    for block_idx in block_indices:
        region_status = _classify_xyz_block_region(ad.mb_geometry, block_idx, xyzlim)
        region_status_map[block_idx] = region_status

    return region_status_map


def _load_local_blocks_into_cache(ad, block_cache, local_blocks, weights_list,
                                  varlist, xyzlim, cast_to_float=True):
    """Load only the requested local blocks, preserving sparse block selections."""
    if not local_blocks:
        return

    region_status_map = _prepare_xyz_block_regions(ad, local_blocks, xyzlim)

    for global_mbl, global_mbh in _iter_contiguous_block_ranges(local_blocks):
        weight_data_dict = {}
        for weight_name in weights_list:
            weight_data_dict[weight_name] = (
                ad.data(weight_name, global_mbl, global_mbh)
                if isinstance(weight_name, str) else weight_name)

        var_data_dict = {}
        for var_name in varlist:
            var_data_dict[var_name] = (
                ad.data(var_name, global_mbl, global_mbh)
                if isinstance(var_name, str) else var_name)

        for global_b in range(global_mbl, global_mbh):
            if global_b not in region_status_map:
                continue

            array_idx = global_b - global_mbl
            region_status = region_status_map[global_b]
            region_status, region_trimmed, coords_block, weights_block, vars_block = (
                _extract_xyz_region_block_data(
                    ad.mb_geometry[global_b],
                    weight_data_dict[weights_list[0]][array_idx].shape,
                    {weight_name: weight_data_dict[weight_name][array_idx]
                     for weight_name in weights_list},
                    {var_name: var_data_dict[var_name][array_idx]
                     for var_name in varlist},
                    region_status, xyzlim))

            block_cache[global_b] = {
                'coords': coords_block,
                'xyz_region_status': region_status,
                'xyz_region_trimmed': region_trimmed,
                'weights': {},
                'vars': {}
            }

            for weight_name in weights_list:
                weight_arr = weights_block[weight_name]
                block_cache[global_b]['weights'][weight_name] = (
                    xp.asarray(weight_arr, dtype=xp.float64)
                    if cast_to_float else weight_arr)

            for var_name in varlist:
                var_arr = vars_block[var_name]
                block_cache[global_b]['vars'][var_name] = (
                    xp.asarray(var_arr, dtype=xp.float64)
                    if cast_to_float else var_arr)

        del weight_data_dict
        del var_data_dict
        clear_backend_memory()


def _build_uniform_axis_values(axis_min, axis_max, n_axis, axis_slice,
                               array_module=xp):
    """Build 1D cell-center coordinates for a sliced Cartesian axis."""
    dx = (axis_max - axis_min) / n_axis
    idx = np.arange(axis_slice.start, axis_slice.stop, dtype=np.float64)
    return array_module.asarray(axis_min + (idx + 0.5) * dx, dtype=np.float64)


def _geometry_axis_slice(axis_min, axis_max, n_axis, lim_min, lim_max):
    """Infer the contiguous in-range cell-index slice from meshblock geometry."""
    dx = (axis_max - axis_min) / n_axis
    i_min = int(np.ceil((lim_min - axis_min) / dx - 0.5))
    i_max = int(np.floor((lim_max - axis_min) / dx - 0.5))
    i_min = max(i_min, 0)
    i_max = min(i_max, n_axis - 1)
    if i_min > i_max:
        return None
    return slice(i_min, i_max + 1)


def _build_xyz_active_slices_from_geometry(block_geometry, block_shape, xyzlim):
    """Build dense Cartesian slices using meshblock geometry only."""
    if xyzlim is None:
        return None
    nx3, nx2, nx1 = block_shape
    x1min, x1max, x2min, x2max, x3min, x3max = block_geometry
    x_slice = _geometry_axis_slice(x1min, x1max, nx1, xyzlim[0], xyzlim[1])
    y_slice = _geometry_axis_slice(x2min, x2max, nx2, xyzlim[2], xyzlim[3])
    z_slice = _geometry_axis_slice(x3min, x3max, nx3, xyzlim[4], xyzlim[5])
    if x_slice is None or y_slice is None or z_slice is None:
        return None
    return (z_slice, y_slice, x_slice)


def _build_block_coords_from_geometry(block_geometry, block_shape, active_slices,
                                      array_module=xp):
    """Construct flattened Cartesian coordinates directly from meshblock geometry."""
    nx3, nx2, nx1 = block_shape
    if active_slices is None:
        active_slices = (slice(0, nx3), slice(0, nx2), slice(0, nx1))
    z_slice, y_slice, x_slice = active_slices
    x1min, x1max, x2min, x2max, x3min, x3max = block_geometry

    x_vals = _build_uniform_axis_values(
        x1min, x1max, nx1, x_slice, array_module=array_module)
    y_vals = _build_uniform_axis_values(
        x2min, x2max, nx2, y_slice, array_module=array_module)
    z_vals = _build_uniform_axis_values(
        x3min, x3max, nx3, z_slice, array_module=array_module)

    nx = x_vals.size
    ny = y_vals.size
    nz = z_vals.size
    return {
        'x': array_module.tile(x_vals, ny * nz),
        'y': array_module.tile(array_module.repeat(y_vals, nx), nz),
        'z': array_module.repeat(z_vals, ny * nx),
    }


def _extract_xyz_region_block_data(block_geometry, block_shape,
                                   weight_blocks, var_blocks,
                                   region_status, xyzlim):
    """Trim a partial Cartesian block once at load time using index ranges."""
    if region_status == 'outside':
        empty = xp.asarray([], dtype=xp.float64)
        return (
            'outside',
            False,
            {'x': empty, 'y': empty, 'z': empty},
            {name: empty for name in weight_blocks},
            {name: empty for name in var_blocks},
        )

    active_slices = None
    region_trimmed = False
    if region_status == 'partial':
        active_slices = _build_xyz_active_slices_from_geometry(
            block_geometry, block_shape, xyzlim)
        if active_slices is None:
            empty = xp.asarray([], dtype=xp.float64)
            return (
                'outside',
                False,
                {'x': empty, 'y': empty, 'z': empty},
                {name: empty for name in weight_blocks},
                {name: empty for name in var_blocks},
            )
        region_trimmed = True

    def _flatten_or_slice(arr):
        if active_slices is None:
            return xp.asarray(arr.flatten(), dtype=xp.float64)
        return xp.asarray(arr[active_slices].flatten(), dtype=xp.float64)

    coords_block = _build_block_coords_from_geometry(
        block_geometry, block_shape, active_slices)
    weights_out = {
        name: _flatten_or_slice(block)
        for name, block in weight_blocks.items()
    }
    vars_out = {
        name: _flatten_or_slice(block)
        for name, block in var_blocks.items()
    }
    return region_status, region_trimmed, coords_block, weights_out, vars_out


def _iter_contiguous_block_ranges(block_indices):
    """Yield contiguous [start, stop) ranges from a block-index list."""
    if not block_indices:
        return

    sorted_blocks = sorted(set(block_indices))
    start = sorted_blocks[0]
    prev = start
    for block_idx in sorted_blocks[1:]:
        if block_idx == prev + 1:
            prev = block_idx
            continue
        yield start, prev + 1
        start = block_idx
        prev = block_idx
    yield start, prev + 1


def _build_rank_assigned_pairs(total_global_blocks, rank, size):
    """Build only this rank's upper-triangular block-pair list."""
    my_pairs = []
    pair_idx = 0
    for i in range(total_global_blocks):
        for j in range(i, total_global_blocks):
            if pair_idx % size == rank:
                my_pairs.append((i, j))
            pair_idx += 1
    return my_pairs


def _free_temporary_gpu_memory():
    """Release temporary CuPy allocations created while staging MPI payloads."""
    clear_backend_memory()


def _build_serialized_block_payloads(ad, block_indices, weights_list, varlist,
                                     xyzlim):
    """Build numpy-backed block payloads, loading each requested block once."""
    from ..core.base import asnumpy

    payloads = {}
    unique_blocks = sorted(set(block_indices))
    if not unique_blocks:
        return payloads

    region_status_map = _prepare_xyz_block_regions(ad, unique_blocks, xyzlim)

    for global_mbl, global_mbh in _iter_contiguous_block_ranges(unique_blocks):
        weight_data_dict = {}
        for weight_name in weights_list:
            weight_data_dict[weight_name] = (
                ad.data(weight_name, global_mbl, global_mbh)
                if isinstance(weight_name, str) else weight_name)

        var_data_dict = {}
        for var_name in varlist:
            var_data_dict[var_name] = (
                ad.data(var_name, global_mbl, global_mbh)
                if isinstance(var_name, str) else var_name)

        for global_b in range(global_mbl, global_mbh):
            array_idx = global_b - global_mbl
            region_status = region_status_map[global_b]
            block_geometry = ad.mb_geometry[global_b]
            block_shape = weight_data_dict[weights_list[0]][array_idx].shape

            if region_status == 'outside':
                empty = np.asarray([], dtype=np.float64)
                region_trimmed = False
                coords_block = {'x': empty, 'y': empty, 'z': empty}
                weights_block = {name: empty for name in weights_list}
                vars_block = {name: empty for name in varlist}
            else:
                active_slices = None
                region_trimmed = False
                if region_status == 'partial':
                    active_slices = _build_xyz_active_slices_from_geometry(
                        block_geometry, block_shape, xyzlim)
                    if active_slices is None:
                        empty = np.asarray([], dtype=np.float64)
                        region_status = 'outside'
                        coords_block = {'x': empty, 'y': empty, 'z': empty}
                        weights_block = {name: empty for name in weights_list}
                        vars_block = {name: empty for name in varlist}
                    else:
                        region_trimmed = True

                if region_status != 'outside':
                    coords_block = _build_block_coords_from_geometry(
                        block_geometry, block_shape, active_slices,
                        array_module=np)

                    def _flatten_to_numpy(arr):
                        arr_view = arr if active_slices is None else arr[active_slices]
                        return np.asarray(asnumpy(arr_view), dtype=np.float64).reshape(-1)

                    weights_block = {
                        name: _flatten_to_numpy(weight_data_dict[name][array_idx])
                        for name in weights_list
                    }
                    vars_block = {
                        name: _flatten_to_numpy(var_data_dict[name][array_idx])
                        for name in varlist
                    }

            payloads[global_b] = {
                'coords': coords_block,
                'xyz_region_status': region_status,
                'xyz_region_trimmed': region_trimmed,
                'weights': weights_block,
                'vars': vars_block
            }

        del weight_data_dict
        del var_data_dict
        _free_temporary_gpu_memory()

    return payloads

def _load_remote_blocks_mpi(ad, block_cache, remote_blocks, weights_list, varlist,
                            mpi_manager, xyzlim=None):
    """
    Load blocks from remote MPI ranks via collective communication.
    
    Uses MPI_Alltoall pattern: each rank announces which blocks it needs,
    then all ranks exchange the required data.
    
    Parameters
    ----------
    ad : AthenaData
        Athena data object
    block_cache : dict
        Cache to store loaded blocks
    remote_blocks : list of int
        Global indices of blocks to load from other ranks
    weights_list : list of str
        Weight variables to load
    varlist : list of str
        Variables to load
    mpi_manager : MPIManager
        MPI communication manager
    xyzlim : array, optional
        Region of interest limits
    """
    from ..core.base import xp
    
    if not remote_blocks:
        return
    
    rank = mpi_manager.rank
    size = mpi_manager.size
    
    # Synchronize before MPI communication to avoid deadlock
    mpi_manager.comm.Barrier()
    
    # Determine which rank owns each block
    mbs_per_rank = ad.n_mbs // size
    remainder = ad.n_mbs % size
    
    def get_owner_rank(global_block_idx):
        """Get the rank that owns a given global block index"""
        if global_block_idx < remainder * (mbs_per_rank + 1):
            return global_block_idx // (mbs_per_rank + 1)
        else:
            return remainder + (global_block_idx - remainder * (mbs_per_rank + 1)) // mbs_per_rank
    
    # Group remote blocks by owning rank
    requests_by_rank = {}
    for block_idx in remote_blocks:
        owner = get_owner_rank(block_idx)
        if owner not in requests_by_rank:
            requests_by_rank[owner] = []
        requests_by_rank[owner].append(block_idx)
    
    # Gather all ranks' requests at all ranks (allgather)
    all_requests = mpi_manager.comm.allgather(requests_by_rank)
    
    # Prepare data to send: what other ranks want from this rank
    data_to_send = {}
    for requesting_rank, their_requests in enumerate(all_requests):
        if rank in their_requests:
            data_to_send[requesting_rank] = {}

    owned_blocks_requested = sorted({
        block_idx
        for their_requests in all_requests
        for block_idx in their_requests.get(rank, [])
        if ad.local_mb_start <= block_idx < ad.local_mb_end
    })
    serialized_blocks = _build_serialized_block_payloads(
        ad, owned_blocks_requested, weights_list, varlist, xyzlim)
    for requesting_rank, their_requests in enumerate(all_requests):
        if rank not in their_requests:
            continue
        data_to_send[requesting_rank] = {
            block_idx: serialized_blocks[block_idx]
            for block_idx in their_requests[rank]
            if block_idx in serialized_blocks
        }
    
    # Exchange data using point-to-point communication
    # Send data to requesting ranks
    send_requests = []
    for dest_rank, blocks_data in data_to_send.items():
        if dest_rank != rank and blocks_data:
            send_requests.append(mpi_manager.comm.isend(blocks_data, dest=dest_rank, tag=100 + rank))
    
    # Receive data from owning ranks
    for source_rank in requests_by_rank.keys():
        if source_rank != rank:
            received_data = mpi_manager.comm.recv(source=source_rank, tag=100 + source_rank)
            
            # Store received blocks in cache
            for global_idx, block_data in received_data.items():
                block_cache[global_idx] = {
                    'coords': {},
                    'xyz_region_status': block_data.get('xyz_region_status', 'inside'),
                    'xyz_region_trimmed': block_data.get('xyz_region_trimmed', False),
                    'weights': {},
                    'vars': {}
                }
                
                # Convert back to xp arrays
                block_cache[global_idx]['coords']['x'] = xp.asarray(block_data['coords']['x'], dtype=xp.float64)
                block_cache[global_idx]['coords']['y'] = xp.asarray(block_data['coords']['y'], dtype=xp.float64)
                block_cache[global_idx]['coords']['z'] = xp.asarray(block_data['coords']['z'], dtype=xp.float64)
                
                for weight_name in block_data['weights']:
                    block_cache[global_idx]['weights'][weight_name] = xp.asarray(
                        block_data['weights'][weight_name], dtype=xp.float64)
                
                for var in block_data['vars']:
                    block_cache[global_idx]['vars'][var] = xp.asarray(
                        block_data['vars'][var], dtype=xp.float64)
    
    # Wait for all sends to complete
    for req in send_requests:
        req.wait()
            
def _process_block_pair(ad, b1, b2, block_cache, block_centers, weights_list, varlist,
                    sf, num_bin_points, weight_sum_bin, 
                    rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                    n_samples1, n_points_per_block, nsamples_block_min,
                    periodic_x, periodic_y, periodic_z, sparse_data=False, debug=False, use_mpi=False, random_seed=None):
    """Process a single pair of blocks for structure functions using varlist kernel"""
    
    if debug:
        print(f"Debug: Starting _process_block_pair for pair ({b1}, {b2})")
        if debug:
            print(f"Debug: n_samples1={n_samples1}, sparse_data={sparse_data}")
    
    # Check if blocks exist in cache (can be missing if rank has no data)
    if b1 not in block_cache or b2 not in block_cache:
        if debug:
            print(f"Debug: Skipping pair ({b1}, {b2}) - blocks not in cache")
        return
    
    try:
        # Calculate distance between blocks for sampling weight
        if debug:
            print(f"Debug: Calculating block distance")
            if debug:
                print(f"Debug: Block {b1} cache keys: coords={list(block_cache[b1]['coords'].keys())}, "
                  f"weights={list(block_cache[b1]['weights'].keys())}, vars={list(block_cache[b1]['vars'].keys())}")
            if debug:
                print(f"Debug: Block {b2} cache keys: coords={list(block_cache[b2]['coords'].keys())}, "
                  f"weights={list(block_cache[b2]['weights'].keys())}, vars={list(block_cache[b2]['vars'].keys())}")
            if debug:
                print(f"Debug: weights_list to process: {weights_list}")
        
        dist_x = abs(block_centers[b1, 0] - block_centers[b2, 0])
        dist_y = abs(block_centers[b1, 1] - block_centers[b2, 1])
        dist_z = abs(block_centers[b1, 2] - block_centers[b2, 2])
        
        # Apply periodic boundary conditions
        if periodic_x:
            dist_x = min(dist_x, abs(Lx - dist_x))
        if periodic_y:
            dist_y = min(dist_y, abs(Ly - dist_y))
        if periodic_z:
            dist_z = min(dist_z, abs(Lz - dist_z))
            
        block_distance = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
        
        if debug:
            print(f"Debug: Block distance = {block_distance}")
        
        # Calculate sample count based on distance
        n_samples2_norm = n_samples1/rmax/(rmax/rmin-1.0)
        n_samples2 = n_samples2_norm * (n_samples1 if block_distance == 0 else (block_distance/rmax)**-2.0)
        n_samples2 = int(np.clip(n_samples2, nsamples_block_min, n_samples1))
        
        if debug:
            print(f"Debug: n_samples2 = {n_samples2}")
        
        # Extract coordinate data from the cache
        if debug:
            print(f"Debug: Extracting coordinate data from cache")
        
        x1 = block_cache[b1]['coords']['x']
        y1 = block_cache[b1]['coords']['y']
        z1 = block_cache[b1]['coords']['z']
        x2 = block_cache[b2]['coords']['x']
        y2 = block_cache[b2]['coords']['y']
        z2 = block_cache[b2]['coords']['z']
        
        if debug:
            print(f"Debug: Coordinate arrays extracted, sizes: {x1.size}, {x2.size}")
        
        no_vars = len(varlist)
        
        # Loop over different weight schemes
        for iw, weight_name in enumerate(weights_list):
            if debug:
                print(f"Debug: Processing weight scheme {iw+1}/{len(weights_list)}: {weight_name}")
            
            # Get weights for this scheme
            weight1 = block_cache[b1]['weights'][weight_name]
            weight2 = block_cache[b2]['weights'][weight_name]
            
            if debug:
                print(f"Debug: Weight arrays extracted, sizes: {weight1.size}, {weight2.size}")
                if debug:
                    print(f"Debug: Weight ranges: w1[{weight1.min():.3f}, {weight1.max():.3f}], w2[{weight2.min():.3f}, {weight2.max():.3f}]")
            
            # Process data differently based on sparse_data flag
            if sparse_data:
                if debug:
                    print(f"Debug: Using sparse data processing")
                
                # Filter out points with zero or negative weights
                valid_points1 = weight1 > 0
                valid_points2 = weight2 > 0
                
                # Convert array sum to host scalar safely
                if cupy_enabled:
                    valid_count1 = int(xp.sum(valid_points1).get())
                    valid_count2 = int(xp.sum(valid_points2).get())
                else:
                    valid_count1 = int(xp.sum(valid_points1))
                    valid_count2 = int(xp.sum(valid_points2))
                
                if debug:
                    print(f"Debug: Valid points - block {b1}: {valid_count1}/{weight1.size}, block {b2}: {valid_count2}/{weight2.size}")
                
                # Skip if no valid points - break to skip entire block pair, not just this weight
                if valid_count1 == 0 or valid_count2 == 0:
                    if debug:
                        print(f"Debug: Skipping pair ({b1}, {b2}) for weight {weight_name} - no valid points")
                    continue  # Skip to next weight scheme
                
                if debug:
                    print(f"Debug: Extracting valid points")
                
                # Extract only valid points
                x1_valid = x1[valid_points1]
                y1_valid = y1[valid_points1]
                z1_valid = z1[valid_points1]
                weight1_valid = weight1[valid_points1]
                
                x2_valid = x2[valid_points2]
                y2_valid = y2[valid_points2]
                z2_valid = z2[valid_points2]
                weight2_valid = weight2[valid_points2]
                
                n_valid1 = x1_valid.size
                n_valid2 = x2_valid.size
                
                if debug:
                    print(f"Debug: Valid arrays created, sizes: {n_valid1}, {n_valid2}")
                
                # Prepare stacked data for varlist kernel
                if debug:
                    print(f"Debug: Preparing stacked data arrays")
                
                data1_valid = xp.zeros(no_vars * n_valid1, dtype=xp.float64)
                data2_valid = xp.zeros(no_vars * n_valid2, dtype=xp.float64)
                
                for ivar, var in enumerate(varlist):
                    if debug and ivar < 2:  # Debug first 2 variables
                        if debug:
                            print(f"Debug: Processing variable {ivar}: {var}")
                    
                    var_data1 = block_cache[b1]['vars'][var][valid_points1]
                    var_data2 = block_cache[b2]['vars'][var][valid_points2]
                    
                    # Store variables contiguously for each variable block
                    data1_valid[ivar * n_valid1:(ivar+1) * n_valid1] = var_data1
                    data2_valid[ivar * n_valid2:(ivar+1) * n_valid2] = var_data2
                
                n_samples1_adj = min(n_samples1, n_valid1)
                n_samples2_adj = min(n_samples2, n_valid2)
                
                if debug:
                    print(f"Debug: Adjusted samples: n_samples1_adj={n_samples1_adj}, n_samples2_adj={n_samples2_adj}")
                
            else:
                # Use all points, regardless of weight
                if debug:
                    print(f"Debug: Using all points (no sparse filtering)")
                
                x1_valid = x1
                y1_valid = y1
                z1_valid = z1
                weight1_valid = weight1
                
                x2_valid = x2
                y2_valid = y2
                z2_valid = z2
                weight2_valid = weight2
                
                n_valid1 = n_points_per_block
                n_valid2 = n_points_per_block
                n_samples1_adj = n_samples1
                n_samples2_adj = n_samples2
                
                # Prepare stacked data for varlist kernel
                data1_valid = xp.zeros(no_vars * n_valid1, dtype=xp.float64)
                data2_valid = xp.zeros(no_vars * n_valid2, dtype=xp.float64)
                
                for ivar, var in enumerate(varlist):
                    # Store variables contiguously for each variable block
                    data1_valid[ivar * n_valid1:(ivar+1) * n_valid1] = block_cache[b1]['vars'][var]
                    data2_valid[ivar * n_valid2:(ivar+1) * n_valid2] = block_cache[b2]['vars'][var]
            
            if debug:
                print(f"Debug: Making data arrays contiguous")
            
            # Make data contiguous
            data1_valid = xp.ascontiguousarray(data1_valid, dtype=xp.float64)
            data2_valid = xp.ascontiguousarray(data2_valid, dtype=xp.float64)
            
            if debug:
                print(f"Debug: Data arrays prepared - sizes: {data1_valid.size}, {data2_valid.size}")
                if debug:
                    print(f"Debug: Expected sizes: {no_vars * n_valid1}, {no_vars * n_valid2}")
            
            # Local arrays for accumulation - flat arrays for the kernel
            if debug:
                print(f"Debug: Creating local accumulation arrays")
            
            local_sf = xp.zeros(no_vars * max_order * nbins, dtype=xp.float64)
            local_num_bin_points = xp.zeros(nbins, dtype=xp.float64)
            local_weight_sum_bin = xp.zeros(nbins, dtype=xp.float64)
            
            # Configure CUDA grid
            threadsperblock = 256
            n_pairs = int(n_samples1_adj * n_samples2_adj)
            n_valid_pairs = n_valid1 * n_valid2
            
            if debug:
                print(f"Debug: Kernel configuration:")
                if debug:
                    print(f"Debug:   n_pairs = {n_pairs}")
                if debug:
                    print(f"Debug:   n_valid_pairs = {n_valid_pairs}")
                if debug:
                    print(f"Debug:   threadsperblock = {threadsperblock}")
            
            # **CRITICAL CHECK: Validate n_pairs is reasonable**
            if n_pairs > 1e10:  # More than 1 billion pairs
                if debug:
                    print(f"Debug: n_pairs ({n_pairs}) exceeds limit, reducing sampling")
                # Calculate scaling factor to bring n_pairs down to 1e9
                scale_factor = np.sqrt(1e10 / n_pairs)
                n_samples1_adj = max(1, int(n_samples1_adj * scale_factor))
                n_samples2_adj = max(1, int(n_samples2_adj * scale_factor))
                n_pairs = int(n_samples1_adj * n_samples2_adj)
                if debug:
                    print(f"Debug: Reduced sampling - n_samples1_adj={n_samples1_adj}, n_samples2_adj={n_samples2_adj}, n_pairs={n_pairs}")
            
            # Exhaustive-kernel path is currently disabled; always uses random sampling.
            use_exhaustive = False

            if use_exhaustive and n_valid_pairs < 5 * n_samples1_adj * n_samples2_adj:
                if debug:
                    print(f"Debug: Would use exhaustive kernel (DISABLED)")
            else:
                if debug:
                    print(f"Debug: Using random sampling kernel")
                
                # Calculate blockspergrid
                blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock
                
                if debug:
                    print(f"Debug: blockspergrid = {blockspergrid}")
                    if debug:
                        print(f"Debug: About to launch kernel...")
                
                if cupy_enabled:
                    mempool = xp.get_default_memory_pool()
                    used_bytes = mempool.used_bytes()
                    total_bytes = mempool.total_bytes()
                    if debug:
                        print(f"Debug: GPU memory before kernel - used: {used_bytes/1e9:.2f} GB, total: {total_bytes/1e9:.2f} GB")
                
                try:
                    # Check if GPU kernels are available
                    if not cupy_enabled or structure_function_varlist_kernel is None:
                        # Use CPU fallback
                        if debug:
                            accel_msg = "with Numba JIT" if NUMBA_AVAILABLE else "pure NumPy (slow - install numba for 10-100x speedup)"
                            if debug:
                                print(f"Debug: Using CPU fallback for structure function computation ({accel_msg})")
                        
                        # Convert to NumPy arrays
                        data1_cpu = asnumpy(data1_valid)
                        data2_cpu = asnumpy(data2_valid)
                        weight1_cpu = asnumpy(weight1_valid)
                        weight2_cpu = asnumpy(weight2_valid)
                        x1_cpu = asnumpy(x1_valid)
                        y1_cpu = asnumpy(y1_valid)
                        z1_cpu = asnumpy(z1_valid)
                        x2_cpu = asnumpy(x2_valid)
                        y2_cpu = asnumpy(y2_valid)
                        z2_cpu = asnumpy(z2_valid)
                        rbins_cpu = asnumpy(rbins)
                        
                        # Compute for each variable
                        for ivar, var in enumerate(varlist):
                            # Extract data for this variable
                            var_data1 = data1_cpu[ivar * n_valid1:(ivar+1) * n_valid1]
                            var_data2 = data2_cpu[ivar * n_valid2:(ivar+1) * n_valid2]
                            
                            # Compute structure function on CPU using vectorized implementation
                            # Use deterministic seed based on block pair (b1, b2) for reproducibility
                            # This ensures CPU and GPU sample the same point pairs
                            pair_seed = random_seed if random_seed is not None else (b1 * 10000 + b2)
                            var_sf_cpu, num_cpu, weights_cpu = _compute_sf_cpu_vectorized(
                                var_data1, var_data2,
                                weight1_cpu, weight2_cpu,
                                x1_cpu, y1_cpu, z1_cpu,
                                x2_cpu, y2_cpu, z2_cpu,
                                rbins_cpu, nbins, max_order,
                                float(Lx), float(Ly), float(Lz),
                                periodic_x, periodic_y, periodic_z,
                                n_pairs, use_mpi=use_mpi, random_seed=pair_seed
                            )
                            
                            # Accumulate results
                            sf[iw, ivar] += var_sf_cpu
                            
                            # Accumulate weights (only once per weight)
                            if ivar == 0:
                                num_bin_points[iw] += num_cpu
                                weight_sum_bin[iw] += weights_cpu
                        
                        if debug:
                            print(f"Debug: CPU fallback computation completed")
                    else:
                        # Use GPU kernel with deterministic base seed
                        # Use block indices to create reproducible seed
                        pair_seed = random_seed if random_seed is not None else (b1 * 10000 + b2)
                        base_seed = np.uint32((pair_seed * 1013904223) % (2**32))
                        structure_function_varlist_kernel(
                            (blockspergrid,), (threadsperblock,),
                            (data1_valid, data2_valid, no_vars,
                            weight1_valid, weight2_valid,
                            x1_valid, y1_valid, z1_valid,
                            x2_valid, y2_valid, z2_valid,
                            rbins, 
                            local_sf, local_num_bin_points,
                            local_weight_sum_bin, n_pairs, n_valid1, n_valid2,
                            float(Lx), float(Ly), float(Lz),
                            nbins, max_order,
                            periodic_x, periodic_y, periodic_z,
                            base_seed)
                        )
                        
                        if debug:
                            print(f"Debug: Kernel launched successfully")
                    
                except Exception as e:
                    print(f"ERROR: Kernel launch failed: {e}")
                    raise
            
            # Synchronize (only for GPU)
            if cupy_enabled and structure_function_varlist_kernel is not None:
                try:
                    xp.cuda.Stream.null.synchronize()
                    if debug:
                        print(f"Debug: CUDA synchronization completed")
                except Exception as e:
                    print(f"ERROR: CUDA synchronization failed: {e}")
                    raise
                
                # Process results for each variable (GPU only)
                if debug:
                    print(f"Debug: Processing kernel results")
                
                for ivar, var in enumerate(varlist):
                    # Calculate offsets for this variable in the flat array
                    sf_offset = ivar * max_order * nbins
                    
                    # Extract this variable's data and reshape
                    var_sf = local_sf[sf_offset:sf_offset + max_order * nbins].reshape(max_order, nbins)
                    
                    # Accumulate results
                    sf[iw, ivar] += var_sf
                
                # Add to weight statistics once per weight (shared across all variables)
                num_bin_points[iw] += local_num_bin_points
                weight_sum_bin[iw] += local_weight_sum_bin
            
            if debug:
                points_added = int(local_num_bin_points.sum())
                if debug:
                    print(f"Debug: Added {points_added} points for weight {weight_name} from blocks ({b1}, {b2})")
        
        if debug:
            print(f"Debug: Successfully completed processing for block pair ({b1}, {b2})")
            
    except Exception as e:
        print(f"ERROR: Failed in _process_block_pair for pair ({b1}, {b2}): {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise
    
def get_sf_helmholtz(ad, var, weights='ones', xyz=None, max_order=10, npairs=1e7, nbins=100, log_bin_flag=True, 
                     nsamples_block_min=1000, debug=False):
    """
    Calculate structure functions for a given field using GPU acceleration.
    
    This function computes structure functions with a Helmholtz decomposition up to 
    a specified order for a 3D field using CUDA kernels. It handles periodic 
    boundaries and uses a block-based sampling strategy with 1/r^2 weighting to 
    improve statistics at small separations.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    var : str 
        Field variable name such as 'vel', 'bcc', etc. 
    weights : str or ndarray, optional
        Weights for the structure function calculation, default='ones'
    xyz : array or None, optional
        Coordinates for the region of interest, default=None
    max_order : int, optional
        Maximum order of structure functions to compute (<=10), default=10
    npairs : float, optional
        Target number of point pairs to sample, default=1e7
    nbins : int, optional
        Number of radial bins for binning separations, default=100
    log_bin_flag : bool, optional
        Whether to use logarithmic binning in radius, default=True
    nsamples_block_min : int, optional
        Minimum number of samples per block, default=1000
    debug : bool, optional
        If True, print detailed debug messages during execution. Default=False
        
    Returns
    -------
    r_ : cp.ndarray
        Array of bin centers for separations
    sf_comp : cp.ndarray
        Compressive (longitudinal) component of structure functions for each order and separation
        Shape: (max_order, nbins)
    sf_sol : cp.ndarray
        Solenoidal component of Structure functions for each order and separation
        Shape: (max_order, nbins)
    weight_sum_bin : cp.ndarray
        Sum of weights in each radial bin
    num_bin_points : cp.ndarray
        Number of point pairs in each radial bin
        
    Notes
    -----
    - Uses CUDA kernels for parallel computation on GPU
    - Implements block-based sampling to manage memory usage
    - Weights point pairs by 1/r^2 to improve small-scale statistics
    - Handles periodic boundary conditions in all directions
    - Maximum order is limited to 10 to manage memory usage
    
    Examples
    --------
    >>> ad = AthenaData(num=100)
    >>> r, sf_comp, sf_sol, weights, counts = get_sf_helmholtz(ad, 'vel', max_order=3, npairs=1e9)
    >>> plt.loglog(r.get(), sf_sol[1].get()/weights.get())  # Plot second-order SF
    """
    
    if debug:
        print(f"Debug: Starting get_sf_helmholtz with var={var}, weights={weights}")
    
    try:
        # Handle weights parameter
        if isinstance(weights, str):
            weight_name = weights
        elif isinstance(weights, (int, float)):
            # If weights is a numeric value, treat it as 'ones'
            if debug:
                print(f"Warning: weights parameter is numeric ({weights}), treating as 'ones'")
            weight_name = 'ones'
        else:
            weight_name = str(weights)
        
        if debug:
            print(f"Debug: Using weight: {weight_name}")
        
        # Get the data
        if debug:
            print(f"Debug: Loading vector field data for {var}")
        try:
            data_x = ad.data(var+'x')
            data_y = ad.data(var+'y')
            data_z = ad.data(var+'z')
            if debug:
                print(f"Debug: Successfully loaded vector field data, shape: {data_x[0].shape}")
        except Exception as e:
            print(f"Error: Failed to load vector field data for {var}: {e}")
            raise
        
        # Get weight data
        if debug:
            print(f"Debug: Loading weight data for: {weight_name}")
        try:
            if weight_name == 'ones':
                weight_data = None  # Will create ones arrays later
                if debug:
                    print(f"Debug: Using ones for weight")
            else:
                weight_data = ad.data(weight_name)
                if debug:
                    print(f"Debug: Successfully loaded weight data for {weight_name}")
        except Exception as e:
            print(f"Error: Failed to load weight data for {weight_name}: {e}")
            # Fallback to ones if weight data can't be loaded
            print(f"Warning: Using 'ones' as fallback for weight {weight_name}")
            weight_data = None
        
        # Apply xyz filtering if specified
        if xyz is not None:
            if debug:
                print(f"Debug: Applying xyz filtering")
            try:
                xyz_weights = ad.data('xyzbool', xyz=xyz)
                if weight_data is not None:
                    weight_data *= xyz_weights
                else:
                    weight_data = xyz_weights
                if debug:
                    print(f"Debug: Successfully applied xyz filtering")
            except Exception as e:
                print(f"Error: Failed to apply xyz filtering: {e}")
                raise
        
        # Get basic simulation parameters
        nmbs = ad.n_mbs
        if debug:
            print(f"Debug: Number of mesh blocks: {nmbs}")
        
        # Load coordinate arrays
        if debug:
            print(f"Debug: Loading coordinate arrays")
        try:
            x = ad.data('x')
            y = ad.data('y')
            z = ad.data('z')
            if debug:
                print(f"Debug: Successfully loaded coordinates, x.shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"Error: Failed to load coordinate arrays: {e}")
            raise
        
        # Get simulation box parameters
        try:
            Lx, Ly, Lz = xp.array([(ad.x1max-ad.x1min),(ad.x2max-ad.x2min),(ad.x3max-ad.x3min)])
            
            # Compute cell sizes from first meshblock geometry (uniform grid)
            first_mb_geom = ad.mb_geometry[0, :]
            dx = (first_mb_geom[1] - first_mb_geom[0]) / ad.nx1
            dy = (first_mb_geom[3] - first_mb_geom[2]) / ad.nx2
            dz = (first_mb_geom[5] - first_mb_geom[4]) / ad.nx3
            
            rmax = xp.sqrt(0.25*(Lx**2+Ly**2+Lz**2))
            rmin = xp.min(xp.array([dx, dy, dz]))
            
            # Ensure rmin > 0 for logarithmic binning
            if rmin <= 0:
                rmin = xp.min(xp.array([dx, dy, dz])[xp.array([dx, dy, dz]) > 0])
                if debug:
                    print(f"Debug: Adjusted rmin to first positive cell size: {rmin}")
            
            if debug:
                print(f"Debug: Box parameters - Lx:{Lx}, Ly:{Ly}, Lz:{Lz}, rmax:{rmax}, rmin:{rmin}")
        except Exception as e:
            print(f"Error: Failed to get simulation box parameters: {e}")
            raise
        
        # Check boundary conditions
        try:
            periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
            periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
            periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']
            if debug:
                print(f"Debug: Periodic boundaries - x:{periodic_x}, y:{periodic_y}, z:{periodic_z}")
        except Exception as e:
            print(f"Error: Failed to check boundary conditions: {e}")
            raise
            
        # Make rbins
        try:
            if log_bin_flag:
                rbins = xp.logspace(xp.log10(rmin),xp.log10(rmax),nbins+1)
                r_ = xp.sqrt(rbins[1:]*rbins[:-1])
            else:
                rbins = xp.linspace(0,rmax,nbins+1)
                r_ = 0.5*(rbins[1:]+rbins[:-1])
            if debug:
                print(f"Debug: Created {nbins} radial bins, log_bin_flag={log_bin_flag}")
        except Exception as e:
            print(f"Error: Failed to create radial bins: {e}")
            raise
        
        # Validate max_order
        if max_order > 10:
            print('Warning: max_order is greater than 10, consider reducing the order to 10')
            max_order = 10
            
        # Calculate sampling parameters
        try:
            n_samples1 = int(xp.sqrt(npairs)/nmbs)
            if n_samples1 < nsamples_block_min:
                print(f'Warning: n_samples1 = {n_samples1} is less than {nsamples_block_min}, consider increasing npairs')
            n_points_per_block = data_x[0].size
            n_samples1 = int(xp.minimum(n_samples1, n_points_per_block))
            if debug:
                print(f"Debug: Sampling parameters - n_samples1:{n_samples1}, n_points_per_block:{n_points_per_block}")
        except Exception as e:
            print(f"Error: Failed to calculate sampling parameters: {e}")
            raise
        
        # Create block centers and distances
        try:
            if debug:
                print(f"Debug: Creating block centers array")
            block_centers = xp.zeros((nmbs, 3))
            block_centers[:,0] = 0.5 * xp.asarray(ad.mb_geometry[:,0]+ad.mb_geometry[:,1])
            block_centers[:,1] = 0.5 * xp.asarray(ad.mb_geometry[:,2]+ad.mb_geometry[:,3])
            block_centers[:,2] = 0.5 * xp.asarray(ad.mb_geometry[:,4]+ad.mb_geometry[:,5])
            if debug:
                print(f"Debug: Successfully created block centers, shape: {block_centers.shape}")
        except Exception as e:
            print(f"Error: Failed to create block centers: {e}")
            raise
        
        # Create mesh of block indices
        try:
            if debug:
                print(f"Debug: About to call xp.triu_indices({nmbs})")
            b1_indices, b2_indices = xp.triu_indices(nmbs)
            if debug:
                print(f"Debug: Raw indices types - b1: {type(b1_indices)}, b2: {type(b2_indices)}")
                if debug:
                    print(f"Debug: Raw indices shapes - b1: {b1_indices.shape}, b2: {b2_indices.shape}")
        except Exception as e:
            print(f"Error: Failed to create block indices: {e}")
            raise
        
        # Calculate block distances vectorized
        try:
            if debug:
                print(f"Debug: Calculating block distances")
            b1b2_x = xp.abs(block_centers[b2_indices,0] - block_centers[b1_indices,0])
            b1b2_y = xp.abs(block_centers[b2_indices,1] - block_centers[b1_indices,1])
            b1b2_z = xp.abs(block_centers[b2_indices,2] - block_centers[b1_indices,2])
            
            # Handle periodic boundaries vectorized
            if periodic_x:
                b1b2_x = xp.minimum(b1b2_x, xp.abs(Lx - b1b2_x))
            if periodic_y:
                b1b2_y = xp.minimum(b1b2_y, xp.abs(Ly - b1b2_y))
            if periodic_z:
                b1b2_z = xp.minimum(b1b2_z, xp.abs(Lz - b1b2_z))
            
            block_distances = xp.sqrt(b1b2_x**2 + b1b2_y**2 + b1b2_z**2)
            if debug:
                print(f"Debug: Successfully calculated block distances")
        except Exception as e:
            print(f"Error: Failed to calculate block distances: {e}")
            raise
        
        # Calculate n_samples2 for all block pairs vectorized
        try:
            if debug:
                print(f"Debug: Calculating sampling weights")
            n_samples2_norm = n_samples1/float(rmax)/(float(rmax)/float(rmin)-1.0)
            n_samples2_blocks = n_samples2_norm * xp.where(block_distances == 0, 
                                                         n_samples1, 
                                                         xp.power(block_distances/float(rmax), -2.0))
            n_samples2_blocks = xp.clip(n_samples2_blocks, nsamples_block_min, n_samples1).astype(xp.int32)
            if debug:
                print(f"Debug: Successfully calculated sampling weights")
        except Exception as e:
            print(f"Error: Failed to calculate sampling weights: {e}")
            raise
        
        # Initialize result arrays
        try:
            if debug:
                print(f"Debug: Initializing result arrays")
            sf_comp_flat = xp.zeros(max_order * nbins, dtype=xp.float64)
            sf_sol_flat = xp.zeros(max_order * nbins, dtype=xp.float64)
            num_bin_points = xp.zeros(nbins, dtype=xp.float64)
            weight_sum_bin = xp.zeros(nbins, dtype=xp.float64)
            if debug:
                print(f"Debug: Successfully initialized result arrays")
        except Exception as e:
            print(f"Error: Failed to initialize result arrays: {e}")
            raise
        
        # Process each block pair
        try:
            if debug:
                print(f"Debug: About to process {len(b1_indices)} block pairs")
            
            for idx in range(len(b1_indices)):
                try:
                    b1, b2 = b1_indices[idx], b2_indices[idx]
                    if debug:
                        print(f"Debug: Processing block pair {idx}: ({b1}, {b2})")
                    
                    # Get flattened arrays and convert to contiguous memory
                    data1_x = xp.ascontiguousarray(data_x[b1].flatten(), dtype=xp.float64)
                    data1_y = xp.ascontiguousarray(data_y[b1].flatten(), dtype=xp.float64)
                    data1_z = xp.ascontiguousarray(data_z[b1].flatten(), dtype=xp.float64)
                    data2_x = xp.ascontiguousarray(data_x[b2].flatten(), dtype=xp.float64)
                    data2_y = xp.ascontiguousarray(data_y[b2].flatten(), dtype=xp.float64)
                    data2_z = xp.ascontiguousarray(data_z[b2].flatten(), dtype=xp.float64)
                    
                    # Handle weight data
                    if weight_data is not None:
                        weight1 = xp.ascontiguousarray(weight_data[b1].flatten(), dtype=xp.float64)
                        weight2 = xp.ascontiguousarray(weight_data[b2].flatten(), dtype=xp.float64)
                    else:
                        # Create ones arrays for 'ones' weight
                        weight1 = xp.ones_like(data1_x, dtype=xp.float64)
                        weight2 = xp.ones_like(data2_x, dtype=xp.float64)
                    
                    x1 = xp.ascontiguousarray(x[b1].flatten(), dtype=xp.float64)
                    y1 = xp.ascontiguousarray(y[b1].flatten(), dtype=xp.float64)
                    z1 = xp.ascontiguousarray(z[b1].flatten(), dtype=xp.float64)
                    x2 = xp.ascontiguousarray(x[b2].flatten(), dtype=xp.float64)
                    y2 = xp.ascontiguousarray(y[b2].flatten(), dtype=xp.float64)
                    z2 = xp.ascontiguousarray(z[b2].flatten(), dtype=xp.float64)
                    
                    # Ensure arrays are contiguous
                    sf_comp_flat_contiguous = xp.ascontiguousarray(sf_comp_flat, dtype=xp.float64)
                    sf_sol_flat_contiguous = xp.ascontiguousarray(sf_sol_flat, dtype=xp.float64)
                    num_bin_points_contiguous = xp.ascontiguousarray(num_bin_points, dtype=xp.float64)
                    weight_sum_bin_contiguous = xp.ascontiguousarray(weight_sum_bin, dtype=xp.float64)

                    # Configure CUDA grid
                    threadsperblock = 256
                    n_pairs = int(n_samples1 * int(n_samples2_blocks[idx]))
                    blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock

                    # Launch kernel with deterministic seed based on block index
                    base_seed = idx * 12345 + 67890
                    structure_function_helmholtz_kernel(
                        (blockspergrid,), (threadsperblock,),
                        (data1_x, data1_y, data1_z, 
                        data2_x, data2_y, data2_z,
                        weight1, weight2,
                        x1, y1, z1,
                        x2, y2, z2,
                        rbins, 
                        sf_comp_flat_contiguous, sf_sol_flat_contiguous, 
                        num_bin_points_contiguous,
                        weight_sum_bin_contiguous, n_pairs, 
                        n_points_per_block, n_points_per_block,
                        float(Lx), float(Ly), float(Lz),
                        nbins, max_order,
                        periodic_x, periodic_y, periodic_z, base_seed))
                        
                    # Copy back updated results
                    sf_comp_flat = sf_comp_flat_contiguous
                    sf_sol_flat = sf_sol_flat_contiguous
                    num_bin_points = num_bin_points_contiguous
                    weight_sum_bin = weight_sum_bin_contiguous
                    
                    # Synchronize after kernel launch
                    if cupy_enabled:
                        xp.cuda.Stream.null.synchronize()
                
                    if debug:
                        print(f"Debug: Successfully processed block pair {idx}")
                    
                except Exception as e:
                    print(f"Error: Failed to process block pair {idx} ({b1}, {b2}): {e}")
                    raise
                    
            if debug:
                print(f"Debug: Successfully processed all block pairs")
            
        except Exception as e:
            print(f"Error: Failed during block pair processing: {e}")
            raise
        
        # Reshape to final dimensions
        try:
            sf_comp = sf_comp_flat.reshape((max_order, nbins))
            sf_sol = sf_sol_flat.reshape((max_order, nbins))
            
            if debug:
                print(f"Debug: Returning Helmholtz structure function results")
            
            return r_, sf_comp, sf_sol, weight_sum_bin, xp.array(num_bin_points, dtype=int)
                    
        except Exception as e:
            print(f"Error: Failed to return results: {e}")
            raise
    
    except Exception as e:
        print(f"Error: Top-level error in get_sf_helmholtz: {e}")
        import traceback
        traceback.print_exc()
        raise 

def get_sf_helmholtz_mb(ad, var, weights='ones', xyz=None, max_order=10, npairs=1e7, nbins=100, 
            log_bin_flag=True, nsamples_block_min=1000, sparse_data=False, debug=False, simultaneous_blocks=None, mpi_manager=None):
    """
    Calculate structure functions with Helmholtz decomposition using GPU acceleration and batch processing.

    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    var : str 
        Field variable prefix name such as 'vel' for velocity components
    weights : str or list of str, optional
        Weights for the structure function calculation. Default='ones'
    xyz : array or None, optional
        Coordinates for the region of interest, default=None
    max_order : int, optional
        Maximum order of structure functions to compute (<=10), default=10
    npairs : float, optional
        Target number of point pairs to sample, default=1e7
    nbins : int, optional
        Number of radial bins for binning separations, default=100
    log_bin_flag : bool, optional
        Whether to use logarithmic binning in radius, default=True
    nsamples_block_min : int, optional
        Minimum number of samples per block, default=1000
    sparse_data : bool, optional
        If True, filter data points based on valid weights. Default=False
    debug : bool, optional
        If True, print detailed debug messages. Default=False
    simultaneous_blocks : int, optional
        Number of blocks to process per batch. Default=None (auto)
    mpi_manager : MPIManager, optional
        MPI manager for distributed computation. Default=None
    
    Returns
    -------
    r_ : xp.ndarray
        Array of bin centers for separations
    sf_comp : xp.ndarray
        Compressible component of structure functions
    sf_sol : xp.ndarray
        Solenoidal component of structure functions
    weight_sum_bin : xp.ndarray
        Sum of weights in each radial bin
    num_bin_points : xp.ndarray
        Number of point pairs in each radial bin
    """
    
    # Establish rank early for debug output
    rank = mpi_manager.rank if mpi_manager else 0
    
    if debug and rank == 0:
        print(f"[DEBUG get_sf_helmholtz_mb] Starting with var={var}, weights={weights}")
        if simultaneous_blocks is not None:
            print(f"[DEBUG get_sf_helmholtz_mb] Using manual batch size={simultaneous_blocks}")
    
    # Get the domain information
    # For MPI distributed data, use local meshblock count
    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs = ad.local_mb_end - ad.local_mb_start
        mb_offset = ad.local_mb_start
    else:
        nmbs = ad.n_mbs
        mb_offset = 0
    
    # Create list of vector components for Helmholtz decomposition
    varlist = [f"{var}x", f"{var}y", f"{var}z"]
    no_vars = 3  # Always 3 for vector field
    if debug and rank == 0:
        print(f"[DEBUG get_sf_helmholtz_mb] Number of mesh blocks: {nmbs} (local), Number of variables: {no_vars}")
        if mb_offset > 0:
            print(f"[DEBUG get_sf_helmholtz_mb] Meshblock offset: {mb_offset}")
    
    Lx, Ly, Lz = np.array([(ad.x1max-ad.x1min), (ad.x2max-ad.x2min), (ad.x3max-ad.x3min)])
    
    # Compute cell sizes from first meshblock geometry (uniform grid)
    # mb_geometry format: [x1min, x1max, x2min, x2max, x3min, x3max]
    first_mb_geom = ad.mb_geometry[0, :]
    dx = (first_mb_geom[1] - first_mb_geom[0]) / ad.nx1
    dy = (first_mb_geom[3] - first_mb_geom[2]) / ad.nx2
    dz = (first_mb_geom[5] - first_mb_geom[4]) / ad.nx3
    
    rmax = np.sqrt(0.25*(Lx**2+Ly**2+Lz**2))
    rmin_local = np.min(np.array([dx, dy, dz]))
    
    # Ensure rmin > 0 for logarithmic binning
    if rmin_local <= 0:
        rmin_local = np.min(np.array([dx, dy, dz])[np.array([dx, dy, dz]) > 0])
        if debug and rank == 0:
            print(f"[DEBUG get_sf_helmholtz_mb] Adjusted rmin to first positive cell size: {rmin_local}")
    
    # Use MPI global minimum to ensure consistency across ranks
    if mpi_manager is not None:
        rmin = mpi_manager.allreduce(rmin_local, op='min')
        if debug and rank == 0:
            print(f"[DEBUG get_sf_helmholtz_mb] Global rmin from MPI allreduce: {rmin} (local was {rmin_local})")
    else:
        rmin = rmin_local
    
    if debug and rank == 0:
        print(f"[DEBUG get_sf_helmholtz_mb] Box parameters - Lx:{Lx:.3f}, Ly:{Ly:.3f}, Lz:{Lz:.3f}, rmax:{rmax:.3f}, rmin:{rmin:.3f}")
        print(f"[DEBUG get_sf_helmholtz_mb] Cell sizes - dx:{dx:.6f}, dy:{dy:.6f}, dz:{dz:.6f}")
    
    # Check if boundaries are periodic
    periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
    periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
    periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']
    if debug:
        print(f"Debug: Periodic boundaries - x:{periodic_x}, y:{periodic_y}, z:{periodic_z}")
    
    # Handle weights parameter - convert to list if a single weight is provided
    weights_list = weights if isinstance(weights, list) else [weights]
    n_weights = len(weights_list)
    if debug:
        print(f"Debug: Processing {n_weights} weight schemes: {weights_list}")
    
    # Validate and adjust max_order
    if max_order > 10:
        print('Warning: max_order is greater than 10, setting to 10')
        max_order = 10
    
    # Calculate sampling parameters
    # IMPORTANT: For MPI mode, use GLOBAL block count
    # This ensures consistent sampling rates across all configurations
    n_samples1 = int(np.sqrt(npairs)/ad.n_mbs)  # Use global block count
    if n_samples1 < nsamples_block_min:
        print(f'Warning: n_samples1 = {n_samples1} is less than {nsamples_block_min}, consider increasing npairs')
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
    n_samples1 = min(n_samples1, n_points_per_block)
    if debug:
        print(f"Debug: Sampling parameters - n_samples1:{n_samples1}, n_points_per_block:{n_points_per_block}")
    
    # Use only local geometry for block centers
    block_centers = np.zeros((nmbs, 3))
    local_geometry = ad.mb_geometry[:nmbs, :]
    block_centers[:,0] = 0.5 * (local_geometry[:,0]+local_geometry[:,1])
    block_centers[:,1] = 0.5 * (local_geometry[:,2]+local_geometry[:,3])
    block_centers[:,2] = 0.5 * (local_geometry[:,4]+local_geometry[:,5])
    if debug:
        print(f"Debug: Successfully created block centers, shape: {block_centers.shape}")
    
    # Create radial bins
    if log_bin_flag:
        rbins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins+1)
        r_ = xp.sqrt(rbins[1:]*rbins[:-1])
    else:
        rbins = xp.linspace(0, rmax, nbins+1)
        r_ = 0.5*(rbins[1:]+rbins[:-1])
    if debug:
        print(f"Debug: Created {nbins} radial bins, log_bin_flag={log_bin_flag}")
    
    # Initialize output arrays - HELMHOLTZ SPECIFIC: comp and sol components
    sf_comp = xp.zeros((n_weights, max_order, nbins), dtype=xp.float64)
    sf_sol = xp.zeros((n_weights, max_order, nbins), dtype=xp.float64)
    num_bin_points = xp.zeros((n_weights, nbins), dtype=xp.float64)
    weight_sum_bin = xp.zeros((n_weights, nbins), dtype=xp.float64)
    if debug:
        print(f"Debug: Successfully initialized result arrays with shapes:")
        if debug:
            print(f"Debug:   sf_comp: {sf_comp.shape}")
        if debug:
            print(f"Debug:   sf_sol: {sf_sol.shape}")
        if debug:
            print(f"Debug:   num_bin_points: {num_bin_points.shape}")
        if debug:
            print(f"Debug:   weight_sum_bin: {weight_sum_bin.shape}")
    
    # Determine blocks per batch based on available memory or manual setting
    if simultaneous_blocks is not None:
        # Use manual batch size, but validate it
        blocks_per_batch = min(simultaneous_blocks, nmbs)
        if debug:
            print(f"Debug: Using manual batch_size={simultaneous_blocks}, blocks_per_batch={blocks_per_batch}")
        
        # Warn if batch size might be too large for memory
        if simultaneous_blocks > nmbs // 2:
            print(f"Warning: Large batch_size ({simultaneous_blocks}) may cause memory issues")
    else:
        # Use automatic determination with 3x safety factor for no_vars (already 3 for vector)
        # to account for temporary arrays and CUDA kernel workspace  
        blocks_per_batch = determine_blocks_per_batch(nmbs, no_vars * 3, n_weights, n_points_per_block)
        if debug:
            print(f"Debug: Using automatic batch sizing, blocks_per_batch={blocks_per_batch}")
    
    # Print an error if blocks_per_batch is zero
    if blocks_per_batch == 0:
        raise ValueError("Error: blocks_per_batch is zero. Please check the input parameters or reduce batch_size.") 
    
    # Keep track of which block pairs we've processed
    processed_pairs = set()
    if debug:
        print(f"Debug: Created empty processed_pairs set")
    
    # Block cache to store loaded data
    block_cache = {}
    
    # Determine if we're using MPI
    use_mpi = mpi_manager is not None
    
    # Different strategies for MPI vs non-MPI mode
    if use_mpi and mpi_manager is not None:
        # MPI MODE: Distribute GLOBAL block pairs across ranks
        if debug or mpi_manager.rank == 0:
            print(f"MPI Mode: Computing Helmholtz structure functions across ALL {ad.n_mbs} global blocks")
        
        # Synchronize all ranks before starting
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Synchronizing before batch processing")
        mpi_manager.comm.Barrier()
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: All ranks synchronized")
        
        total_global_blocks = ad.n_mbs
        if debug:
            total_pairs = total_global_blocks * (total_global_blocks + 1) // 2
            print(f"Debug: Rank {mpi_manager.rank}: Total global pairs: {total_pairs}")
        
        # Distribute pairs across ranks using round-robin without materializing all pairs.
        my_pairs = _build_rank_assigned_pairs(
            total_global_blocks, mpi_manager.rank, mpi_manager.size)
        
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Assigned {len(my_pairs)} pairs to process")
            if len(my_pairs) > 0:
                if debug:
                    print(f"Debug: Rank {mpi_manager.rank}: First pair: {my_pairs[0]}, Last pair: {my_pairs[-1]}")
        
        # Process assigned pairs - HELMHOLTZ SPECIFIC
        _process_mpi_pairs_helmholtz(ad, my_pairs, block_cache, weights_list, varlist,
                          sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                          rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                          n_samples1, n_points_per_block, nsamples_block_min,
                          periodic_x, periodic_y, periodic_z, sparse_data, xyzlim=xyz,
                          mpi_manager=mpi_manager, mb_offset=mb_offset, debug=debug)
    else:
        # NON-MPI MODE: Process local blocks only
        # Process blocks in batches
        num_batches = (nmbs + blocks_per_batch - 1) // blocks_per_batch
        batches = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * blocks_per_batch
            batch_end = min(batch_start + blocks_per_batch, nmbs)
            batches.append(list(range(batch_start, batch_end)))
        if debug:
            print(f"Debug: Created {num_batches} batches with blocks_per_batch={blocks_per_batch}")
        
        # Process batch pairs - HELMHOLTZ SPECIFIC
        if debug:
            print(f"Debug: Starting batch pair processing")
        _process_batch_pairs_helmholtz(ad, batches, block_cache, processed_pairs, weights_list, varlist, 
                                sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                                block_centers, rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                                n_samples1, n_points_per_block, nsamples_block_min, 
                                periodic_x, periodic_y, periodic_z, sparse_data, xyzlim=xyz, 
                                use_mpi=False, mpi_manager=None, mb_offset=0, debug=debug)
        if debug:
            print(f"Debug: Completed batch pair processing")
    
    # Handle any missed pairs (only in non-MPI mode)
    # In MPI mode, all pairs are handled via round-robin distribution
    if not use_mpi:
        total_pairs = (nmbs * (nmbs + 1)) // 2
        if len(processed_pairs) < total_pairs:
            if debug:
                print(f"Debug: Processing missed pairs - {len(processed_pairs)} of {total_pairs} completed")
            print(f"WARNING: Only processed {len(processed_pairs)} of {total_pairs} block pairs")
            print("Processing any missed pairs...")
            
            # Find any missing pairs
            missing_pairs = []
            for i in range(nmbs):
                for j in range(i, nmbs):
                    pair = (i, j)
                    if pair not in processed_pairs:
                        missing_pairs.append(pair)
            
            if missing_pairs:
                if debug:
                    print(f"Debug: Found {len(missing_pairs)} missing pairs")
                # Create batches from missing pairs - each pair becomes its own mini-batch
                missing_batches = [[p[0]] for p in missing_pairs] + [[p[1]] for p in missing_pairs]
                # Remove duplicates while preserving order
                unique_missing_batches = []
                seen = set()
                for batch in missing_batches:
                    block = batch[0]
                    if block not in seen:
                        seen.add(block)
                        unique_missing_batches.append(batch)
                        
                # Process the missing batches in the same way as regular batches
                _process_batch_pairs_helmholtz(ad, unique_missing_batches, block_cache, processed_pairs, 
                                        weights_list, varlist, sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                                        block_centers, rmax, rmin, Lx, Ly, Lz, rbins, 
                                        nbins, max_order, n_samples1, n_points_per_block, 
                                        nsamples_block_min, periodic_x, periodic_y, periodic_z, 
                                        sparse_data, xyzlim=xyz, debug=debug,
                                        use_mpi=False, mpi_manager=None, mb_offset=0)
                if debug:
                    print(f"Debug: Completed processing missed pairs")
    
    # Final sync
    if cupy_enabled:
        xp.cuda.Stream.null.synchronize()
    
    # Reshape output to match input format
    if not isinstance(weights, list):
        # Return just the first weight's data (no weights dimension)
        sf_comp = sf_comp[0]
        sf_sol = sf_sol[0]
        num_bin_points = num_bin_points[0]
        weight_sum_bin = weight_sum_bin[0]
        if debug:
            print(f"Debug: Reshaped output for single weight, sf_comp shape: {sf_comp.shape}")
    
    if debug:
        print(f"Debug: Total points in bins: {num_bin_points.sum()}")
        if debug:
            print(f"Debug: Returning Helmholtz structure function results")
    
    # Gather results from all MPI ranks if using MPI
    if mpi_manager is not None:
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: Aggregating Helmholtz results across {mpi_manager.size} ranks")
        
        # Convert to numpy for MPI communication
        sf_comp_np = asnumpy(sf_comp)
        sf_sol_np = asnumpy(sf_sol)
        num_bin_points_np = asnumpy(num_bin_points)
        weight_sum_bin_np = asnumpy(weight_sum_bin)
        
        # Sum results across all ranks
        sf_comp_total = mpi_manager.reduce(sf_comp_np, op='sum', root=0)
        sf_sol_total = mpi_manager.reduce(sf_sol_np, op='sum', root=0)
        num_bin_points_total = mpi_manager.reduce(num_bin_points_np, op='sum', root=0)
        weight_sum_bin_total = mpi_manager.reduce(weight_sum_bin_np, op='sum', root=0)
        
        # Broadcast aggregated results to all ranks
        sf_comp_total = mpi_manager.broadcast(sf_comp_total, root=0)
        sf_sol_total = mpi_manager.broadcast(sf_sol_total, root=0)
        num_bin_points_total = mpi_manager.broadcast(num_bin_points_total, root=0)
        weight_sum_bin_total = mpi_manager.broadcast(weight_sum_bin_total, root=0)
        
        # Convert back to GPU arrays if using CuPy
        sf_comp = xp.asarray(sf_comp_total)
        sf_sol = xp.asarray(sf_sol_total)
        num_bin_points = xp.asarray(num_bin_points_total)
        weight_sum_bin = xp.asarray(weight_sum_bin_total)
        
        if debug:
            print(f"Debug: Rank {mpi_manager.rank}: MPI aggregation complete")
    
    return r_, sf_comp, sf_sol, weight_sum_bin, xp.array(num_bin_points, dtype=int)

def _process_mpi_pairs_helmholtz(ad, my_pairs, block_cache, weights_list, varlist,
                      sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                      rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                      n_samples1, n_points_per_block, nsamples_block_min,
                      periodic_x, periodic_y, periodic_z, sparse_data=False, xyzlim=None,
                      mpi_manager=None, mb_offset=0, debug=False):
    """
    Process assigned block pairs in MPI mode with Helmholtz decomposition.
    
    PAIR COVERAGE GUARANTEE:
    - Caller generates ALL pairs (i,j) where 0 <= i <= j < n_mbs
    - Pairs are distributed round-robin: rank r processes pairs where idx % size == r
    - This ensures every pair is processed by exactly one rank
    - Results from all ranks are aggregated via MPI reduce after processing
    
    MEMORY MANAGEMENT:
    - Pairs processed in batches sized by determine_blocks_per_batch()
    - Batch size calculated from actual GPU/CPU memory and block dimensions
    - Blocks loaded/unloaded per batch to respect memory constraints
    
    OPTIMIZATION:
    - Pairs reordered to minimize remote block fetching:
      1. Both blocks local (no MPI needed)
      2. One block local (minimal MPI)
      3. Both blocks remote (maximum MPI)
    """
    rank = mpi_manager.rank if mpi_manager else 0
    size = mpi_manager.size if mpi_manager else 1
    
    if debug:
        print(f"Debug: Rank {rank}: Processing {len(my_pairs)} assigned pairs (Helmholtz)")
    
    # Get local block range for this rank
    local_start = mb_offset
    # Calculate number of local meshblocks
    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs_local = ad.local_mb_end - ad.local_mb_start
    else:
        nmbs_local = ad.n_mbs
    local_end = mb_offset + nmbs_local
    
    if debug:
        print(f"Debug: Rank {rank}: Local blocks [{local_start}, {local_end})")
    
    # DON'T filter pairs - process ALL assigned pairs
    # We'll load remote blocks via MPI as needed
    
    # Compute block centers for ALL GLOBAL blocks (needed for sampling weights)
    # Each rank computes this independently - no communication needed
    block_centers_global = np.zeros((ad.n_mbs, 3))
    full_geometry = ad.mb_geometry[:ad.n_mbs, :]
    block_centers_global[:, 0] = 0.5 * (full_geometry[:, 0] + full_geometry[:, 1])
    block_centers_global[:, 1] = 0.5 * (full_geometry[:, 2] + full_geometry[:, 3])
    block_centers_global[:, 2] = 0.5 * (full_geometry[:, 4] + full_geometry[:, 5])
    
    if debug:
        print(f"Debug: Rank {rank}: Created global block_centers with shape {block_centers_global.shape}")
        if debug:
            print(f"Debug: Rank {rank}: Processing {len(my_pairs)} assigned pairs (including remote blocks)")
    
    # Determine batch size based on actual memory and block size
    # Use same memory estimation as non-MPI version
    from athena_research.utils.batch_processing import determine_blocks_per_batch
    
    # Calculate blocks per batch based on memory
    blocks_per_batch = determine_blocks_per_batch(ad.n_mbs, len(varlist) * 3, len(weights_list), n_points_per_block)
    
    if debug:
        print(f"Debug: Rank {rank}: Calculated blocks_per_batch={blocks_per_batch} based on memory constraints")
    
    # Get all unique blocks needed by this rank
    all_blocks_needed = set()
    for global_b1, global_b2 in my_pairs:
        all_blocks_needed.add(global_b1)
        all_blocks_needed.add(global_b2)
    
    # Separate local vs remote blocks
    local_blocks = [b for b in all_blocks_needed if local_start <= b < local_end]
    remote_blocks = [b for b in all_blocks_needed if b < local_start or b >= local_end]
    
    if debug:
        print(f"Debug: Rank {rank}: Need {len(local_blocks)} local and {len(remote_blocks)} remote blocks")
    
    # Optimize pair ordering: prioritize pairs with both blocks local,
    # then pairs with one local block, then pairs with both remote
    # This minimizes remote block fetching per batch
    pairs_by_locality = {
        'both_local': [],
        'one_local': [],
        'both_remote': []
    }
    
    for global_b1, global_b2 in my_pairs:
        b1_local = (local_start <= global_b1 < local_end)
        b2_local = (local_start <= global_b2 < local_end)
        
        if b1_local and b2_local:
            pairs_by_locality['both_local'].append((global_b1, global_b2))
        elif b1_local or b2_local:
            pairs_by_locality['one_local'].append((global_b1, global_b2))
        else:
            pairs_by_locality['both_remote'].append((global_b1, global_b2))
    
    # Reorder pairs to process local-heavy batches first
    optimized_pairs = (pairs_by_locality['both_local'] + 
                      pairs_by_locality['one_local'] + 
                      pairs_by_locality['both_remote'])
    
    if debug:
        print(f"Debug: Rank {rank}: Pair locality - both_local:{len(pairs_by_locality['both_local'])}, "
              f"one_local:{len(pairs_by_locality['one_local'])}, both_remote:{len(pairs_by_locality['both_remote'])}")
    
    
    # Process pairs in batches
    pairs_processed = 0
    pair_idx = 0
    
    while pair_idx < len(optimized_pairs):
        # Determine blocks needed for this batch of pairs
        batch_blocks_needed = set()
        batch_end_idx = pair_idx
        
        # Greedily add pairs to batch until we hit memory limit
        while batch_end_idx < len(optimized_pairs) and len(batch_blocks_needed) < blocks_per_batch:
            global_b1, global_b2 = optimized_pairs[batch_end_idx]
            
            # Check if adding this pair would exceed batch size
            new_blocks = set()
            if global_b1 not in block_cache:
                new_blocks.add(global_b1)
            if global_b2 not in block_cache:
                new_blocks.add(global_b2)
            
            # If this would exceed limit and we already have some blocks, stop
            if len(batch_blocks_needed) + len(new_blocks) > blocks_per_batch and len(batch_blocks_needed) > 0:
                break
            
            batch_blocks_needed.update(new_blocks)
            batch_end_idx += 1
        
        # Extract batch pairs
        batch_pairs = optimized_pairs[pair_idx:batch_end_idx]
        
        if debug:
            print(f"Debug: Rank {rank}: Processing batch with {len(batch_blocks_needed)} blocks, {len(batch_pairs)} pairs")
        
        # Load blocks for this batch
        if batch_blocks_needed:
            _load_all_blocks_upfront(ad, block_cache, list(batch_blocks_needed),
                                    weights_list, varlist,
                                    local_start, local_end, xyzlim, mpi_manager, debug=False)
        
        # Process all pairs in this batch
        for global_b1, global_b2 in batch_pairs:
            if debug and pairs_processed % 10 == 0:
                if debug:
                    print(f"Debug: Rank {rank}: Processing pair {pairs_processed}/{len(optimized_pairs)}: ({global_b1}, {global_b2})")
            
            # Check if blocks are outside xyz limits
            if xyzlim is not None:
                if is_block_outside_xyz(ad.mb_geometry, global_b1, xyzlim) or \
                   is_block_outside_xyz(ad.mb_geometry, global_b2, xyzlim):
                    continue
            
            # Verify both blocks are loaded
            if global_b1 not in block_cache or global_b2 not in block_cache:
                if debug:
                    print(f"Debug: Rank {rank}: Warning - blocks ({global_b1}, {global_b2}) not in cache, skipping")
                continue
        
            # Process the pair using GLOBAL indices (ONLY DIFFERENCE: Helmholtz kernel)
            _process_block_pair_helmholtz(ad, global_b1, global_b2, block_cache, block_centers_global,
                                         weights_list, varlist, sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                                         rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                                         n_samples1, n_points_per_block, nsamples_block_min,
                                         periodic_x, periodic_y, periodic_z, sparse_data, debug=False, use_mpi=True,
                                         random_seed=(global_b1 * 10000 + global_b2))
            
            pairs_processed += 1
        
        # Clear cache for this batch to free memory
        for block_id in batch_blocks_needed:
            if block_id in block_cache:
                del block_cache[block_id]
        
        if debug:
            print(f"Debug: Rank {rank}: Batch complete, cleared {len(batch_blocks_needed)} blocks, processed {pairs_processed}/{len(optimized_pairs)} pairs so far")
        
        # Move to next batch
        pair_idx = batch_end_idx
    
    if debug:
        print(f"Debug: Rank {rank}: Completed processing all {pairs_processed} assigned pairs")

def _process_batch_pairs_helmholtz(ad, batches, block_cache, processed_pairs, weights_list, varlist, 
                        sf_comp, sf_sol, num_bin_points, weight_sum_bin,
                        block_centers, rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                        n_samples1, n_points_per_block, nsamples_block_min, 
                        periodic_x, periodic_y, periodic_z, sparse_data=False, xyzlim=None, debug=False, 
                        use_mpi=False, mpi_manager=None, mb_offset=0):
    """
    Helper function to process batch pairs for Helmholtz-decomposed structure functions.
    
    In MPI mode, this function computes SF for ALL block pairs involving at least one
    local block. For remote blocks (on other ranks), it uses MPI communication to 
    exchange data as needed.
    """
    
    if debug:
        print(f"Debug: Starting Helmholtz batch pairs processing with {len(batches)} batches")
    
    # Create a set of blocks to skip if they're outside xyz limits
    blocks_to_skip = set()
    if xyzlim is not None:
        if debug:
            print(f"Debug: Checking blocks against xyz limits: {xyzlim}")
        for batch in batches:
            for block_idx in batch:
                if is_block_outside_xyz(ad.mb_geometry, block_idx, xyzlim):
                    blocks_to_skip.add(block_idx)
        
        if blocks_to_skip:
            if debug:
                print(f"Debug: Skipping {len(blocks_to_skip)} blocks outside xyz limits")
            print(f"Skipping {len(blocks_to_skip)} blocks outside xyz limits")
            # Mark all pairs involving skipped blocks as processed
            for b1 in blocks_to_skip:
                for b2 in range(ad.n_mbs):
                    # Ensure that smaller index is first in the pair
                    pair = (min(b1, b2), max(b1, b2))
                    processed_pairs.add(pair)
    
    for batch_idx1 in range(len(batches)):
        batch1 = batches[batch_idx1]
        if debug:
            print(f"Debug: Processing Helmholtz batch {batch_idx1+1}/{len(batches)}, size: {len(batch1)}")
        
        # Filter batch1 to exclude blocks outside xyz limits and ensure integers
        batch1 = [b for b in batch1 if b not in blocks_to_skip]
        if not batch1:
            if debug:
                print(f"Debug: Skipping empty batch {batch_idx1+1}")
            continue  # Skip empty batches
            
        # Load data for batch1
        if debug:
            print(f"Debug: Loading data for Helmholtz batch1: {len(batch1)} blocks")
        _load_batch_data(ad, block_cache, batch1, weights_list, varlist, xyzlim=xyzlim,
                        mpi_manager=mpi_manager if use_mpi else None, mb_offset=mb_offset if use_mpi else 0)
        
        # Process intra-batch pairs
        intra_pairs = 0
        for i in range(len(batch1)):
            for j in range(i, len(batch1)):
                b1, b2 = batch1[i], batch1[j]  # Ensure int conversion
                pair = (b1, b2)
                
                # In MPI mode, only process if this rank should handle this pair
                if use_mpi and mpi_manager is not None:
                    # Convert local indices to global indices
                    global_b1 = b1 + mb_offset
                    global_b2 = b2 + mb_offset
                    # Deterministic assignment: rank owning the first block processes the pair
                    owning_rank = global_b1 // ((ad.n_mbs + mpi_manager.size - 1) // mpi_manager.size)
                    if owning_rank != mpi_manager.rank:
                        if debug:
                            print(f"Debug: Rank {mpi_manager.rank}: Skipping pair ({global_b1}, {global_b2}) - assigned to rank {owning_rank}")
                        continue
                    if debug:
                        print(f"Debug: Rank {mpi_manager.rank}: Processing intra-batch pair ({global_b1}, {global_b2})")
                elif debug:
                    print(f"Debug: Processing intra-batch pair ({b1}, {b2})")
                
                if pair not in processed_pairs:
                    _process_block_pair_helmholtz(ad,
                        b1, b2, block_cache, block_centers, weights_list, varlist,
                        sf_comp, sf_sol, num_bin_points, weight_sum_bin, 
                        rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                        n_samples1, n_points_per_block, nsamples_block_min,
                        periodic_x, periodic_y, periodic_z, sparse_data, debug=debug, use_mpi=use_mpi
                    )
                    processed_pairs.add(pair)
                    intra_pairs += 1
        
        if debug:
            print(f"Debug: Processed {intra_pairs} intra-batch Helmholtz pairs for batch {batch_idx1+1}")
        
        # Process cross-batch pairs
        cross_pairs = 0
        for batch_idx2 in range(batch_idx1 + 1, len(batches)):
            batch2 = batches[batch_idx2]
            
            # Filter batch2 to exclude blocks outside xyz limits and ensure integers
            batch2 = [b for b in batch2 if b not in blocks_to_skip]
            if not batch2:
                continue  # Skip empty batches
                
            if debug:
                print(f"Debug: Loading data for Helmholtz batch2 ({batch_idx2+1}): {len(batch2)} blocks")
            # Load data for batch2
            _load_batch_data(ad, block_cache, batch2, weights_list, varlist, xyzlim=xyzlim,
                            mpi_manager=mpi_manager if use_mpi else None, mb_offset=mb_offset if use_mpi else 0)
            
            # Process all pairs between batch1 and batch2
            for b1 in batch1:
                for b2 in batch2:
                    pair = (b1, b2)
                    
                    # In MPI mode, only process if this rank should handle this pair
                    if use_mpi and mpi_manager is not None:
                        # Convert local indices to global indices
                        global_b1 = b1 + mb_offset
                        global_b2 = b2 + mb_offset
                        # Deterministic assignment: rank owning the first block processes the pair
                        owning_rank = global_b1 // ((ad.n_mbs + mpi_manager.size - 1) // mpi_manager.size)
                        if owning_rank != mpi_manager.rank:
                            continue
                    
                    if pair not in processed_pairs:
                        _process_block_pair_helmholtz(ad,
                            b1, b2, block_cache, block_centers, weights_list, varlist,
                            sf_comp, sf_sol, num_bin_points, weight_sum_bin, 
                            rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                            n_samples1, n_points_per_block, nsamples_block_min,
                            periodic_x, periodic_y, periodic_z, sparse_data, debug=debug, use_mpi=use_mpi
                        )
                        processed_pairs.add(pair)
                        cross_pairs += 1
            
            # Clear batch2 data from cache
            for b in batch2:
                if b in block_cache:
                    del block_cache[b]
            
            if debug:
                print(f"Debug: Processed {cross_pairs} cross-batch Helmholtz pairs between batches {batch_idx1+1} and {batch_idx2+1}")
        
        # Clear batch1 data from cache
        for b in batch1:
            if b in block_cache:
                del block_cache[b]
        
        # Clean up GPU memory
        if cupy_enabled:
            xp.cuda.Stream.null.synchronize()
            xp.get_default_memory_pool().free_all_blocks()
            
        if debug:
            print(f"Debug: Completed Helmholtz batch {batch_idx1+1}/{len(batches)}")
            
def _process_block_pair_helmholtz(ad, b1, b2, block_cache, block_centers, weights_list, varlist,
                            sf_comp, sf_sol, num_bin_points, weight_sum_bin, 
                            rmax, rmin, Lx, Ly, Lz, rbins, nbins, max_order,
                            n_samples1, n_points_per_block, nsamples_block_min,
                            periodic_x, periodic_y, periodic_z, sparse_data=False, debug=False, use_mpi=False, random_seed=None):
    """Process a single pair of blocks for Helmholtz-decomposed structure functions"""
    
    if debug:
        print(f"Debug: Processing Helmholtz block pair ({b1}, {b2})")
    
    # Calculate distance between blocks for sampling weight
    dist_x = abs(block_centers[b1, 0] - block_centers[b2, 0])
    dist_y = abs(block_centers[b1, 1] - block_centers[b2, 1])
    dist_z = abs(block_centers[b1, 2] - block_centers[b2, 2])
    
    # Apply periodic boundary conditions
    if periodic_x:
        dist_x = min(dist_x, abs(Lx - dist_x))
    if periodic_y:
        dist_y = min(dist_y, abs(Ly - dist_y))
    if periodic_z:
        dist_z = min(dist_z, abs(Lz - dist_z))
        
    block_distance = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
    
    # Calculate sample count based on distance
    n_samples2_norm = n_samples1/rmax/(rmax/rmin-1.0)
    n_samples2 = n_samples2_norm * (n_samples1 if block_distance == 0 else (block_distance/rmax)**-2.0)
    n_samples2 = int(np.clip(n_samples2, nsamples_block_min, n_samples1))
    
    if debug:
        print(f"Debug: Block distance: {block_distance:.3f}, n_samples2: {n_samples2}")
    
    # Extract coordinate data from the cache
    x1 = block_cache[b1]['coords']['x']
    y1 = block_cache[b1]['coords']['y']
    z1 = block_cache[b1]['coords']['z']
    x2 = block_cache[b2]['coords']['x']
    y2 = block_cache[b2]['coords']['y']
    z2 = block_cache[b2]['coords']['z']
    
    # Extract vector components
    vx1 = block_cache[b1]['vars'][varlist[0]]
    vy1 = block_cache[b1]['vars'][varlist[1]]
    vz1 = block_cache[b1]['vars'][varlist[2]]
    
    vx2 = block_cache[b2]['vars'][varlist[0]]
    vy2 = block_cache[b2]['vars'][varlist[1]]
    vz2 = block_cache[b2]['vars'][varlist[2]]
    
    if debug:
        print(f"Debug: Successfully extracted vector components for blocks ({b1}, {b2})")
    
    # Loop over different weight schemes
    for iw, weight_name in enumerate(weights_list):
        if debug:
            print(f"Debug: Processing weight scheme {iw+1}/{len(weights_list)}: {weight_name}")
        
        # Get weights for this scheme
        weight1 = block_cache[b1]['weights'][weight_name]
        weight2 = block_cache[b2]['weights'][weight_name]
        
        try:
            # Process data differently based on sparse_data flag
            if sparse_data:
                if debug:
                    print(f"Debug: Using sparse data filtering for weight {weight_name}")
                # Filter out points with zero or negative weights
                valid_points1 = weight1 > 0
                valid_points2 = weight2 > 0
                
                # Convert array sum to host scalar safely
                if cupy_enabled:
                    valid_count1 = int(xp.sum(valid_points1).get())
                    valid_count2 = int(xp.sum(valid_points2).get())
                else:
                    valid_count1 = int(xp.sum(valid_points1))
                    valid_count2 = int(xp.sum(valid_points2))
                
                if debug:
                    print(f"Debug: Valid points - block {b1}: {valid_count1}, block {b2}: {valid_count2}")
                
                # Skip if no valid points
                if valid_count1 == 0 or valid_count2 == 0:
                    if debug:
                        print(f"Debug: Skipping pair ({b1}, {b2}) - no valid points")
                    continue
                
                # Extract only valid points
                x1_valid = x1[valid_points1]
                y1_valid = y1[valid_points1]
                z1_valid = z1[valid_points1]
                weight1_valid = weight1[valid_points1]
                
                x2_valid = x2[valid_points2]
                y2_valid = y2[valid_points2]
                z2_valid = z2[valid_points2]
                weight2_valid = weight2[valid_points2]
                
                # Extract valid vector components
                vx1_valid = vx1[valid_points1]
                vy1_valid = vy1[valid_points1]
                vz1_valid = vz1[valid_points1]
                
                vx2_valid = vx2[valid_points2]
                vy2_valid = vy2[valid_points2]
                vz2_valid = vz2[valid_points2]
                
                # Get number of valid points
                n_valid1 = x1_valid.size
                n_valid2 = x2_valid.size
                
                # If valid points are fewer than what we need, adjust n_samples
                n_samples1_adj = min(n_samples1, n_valid1)
                n_samples2_adj = min(n_samples2, n_valid2)
            else:
                if debug:
                    print(f"Debug: Using all points (no sparse filtering) for weight {weight_name}")
                # Use all points, regardless of weight
                x1_valid = x1
                y1_valid = y1
                z1_valid = z1
                weight1_valid = weight1
                
                x2_valid = x2
                y2_valid = y2
                z2_valid = z2
                weight2_valid = weight2
                
                vx1_valid = vx1
                vy1_valid = vy1
                vz1_valid = vz1
                
                vx2_valid = vx2
                vy2_valid = vy2
                vz2_valid = vz2
                
                n_valid1 = n_points_per_block
                n_valid2 = n_points_per_block
                n_samples1_adj = n_samples1
                n_samples2_adj = n_samples2
            
            # Local arrays for accumulation
            local_sf_comp = xp.zeros((max_order, nbins), dtype=xp.float64)
            local_sf_sol = xp.zeros((max_order, nbins), dtype=xp.float64)
            local_sf_comp_flat = xp.ascontiguousarray(local_sf_comp.flatten(), dtype=xp.float64)
            local_sf_sol_flat = xp.ascontiguousarray(local_sf_sol.flatten(), dtype=xp.float64)
            local_num_bin_points = xp.zeros(nbins, dtype=xp.float64)
            local_weight_sum_bin = xp.zeros(nbins, dtype=xp.float64)
            
            # Configure CUDA grid
            threadsperblock = 256
            n_pairs = int(n_samples1_adj * n_samples2_adj)
            blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock
            n_valid_pairs = n_valid1 * n_valid2
            
            if debug:
                print(f"Debug: Kernel configuration - n_pairs: {n_pairs}, n_valid_pairs: {n_valid_pairs}")
            
            # Choose appropriate kernel based on number of valid pairs
            if n_valid_pairs < 5 * n_samples1_adj * n_samples2_adj:
                if debug:
                    print(f"Debug: Using exhaustive Helmholtz kernel for small dataset")
                # Use exhaustive kernel for small datasets
                blockspergrid = (n_valid_pairs + threadsperblock - 1) // threadsperblock
                
                structure_function_helmholtz_kernel_exhaustive(
                    (blockspergrid,), (threadsperblock,),
                    (vx1_valid, vy1_valid, vz1_valid,
                    vx2_valid, vy2_valid, vz2_valid,
                    weight1_valid, weight2_valid,
                    x1_valid, y1_valid, z1_valid,
                    x2_valid, y2_valid, z2_valid,
                    rbins, 
                    local_sf_comp_flat, local_sf_sol_flat, local_num_bin_points,
                    local_weight_sum_bin,
                    n_valid1, n_valid2,
                    float(Lx), float(Ly), float(Lz),
                    nbins, max_order,
                    periodic_x, periodic_y, periodic_z)
                )
            else:
                if debug:
                    print(f"Debug: Using random sampling Helmholtz kernel for large dataset")
                # Use random sampling kernel for larger datasets
                blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock
                
                # Use deterministic seed based on block pair (b1, b2) for reproducibility
                base_seed = random_seed if random_seed is not None else (b1 * 10000 + b2)
                structure_function_helmholtz_kernel(
                    (blockspergrid,), (threadsperblock,),
                    (vx1_valid, vy1_valid, vz1_valid,
                    vx2_valid, vy2_valid, vz2_valid,
                    weight1_valid, weight2_valid,
                    x1_valid, y1_valid, z1_valid,
                    x2_valid, y2_valid, z2_valid,
                    rbins, 
                    local_sf_comp_flat, local_sf_sol_flat, local_num_bin_points,
                    local_weight_sum_bin, n_pairs, n_valid1, n_valid2,
                    float(Lx), float(Ly), float(Lz),
                    nbins, max_order,
                    periodic_x, periodic_y, periodic_z, base_seed)
                )
            
            # Reshape and accumulate results
            local_sf_comp = local_sf_comp_flat.reshape((max_order, nbins))
            local_sf_sol = local_sf_sol_flat.reshape((max_order, nbins))
            
            sf_comp[iw] += local_sf_comp
            sf_sol[iw] += local_sf_sol
            
            # Add to weight statistics once per weight
            num_bin_points[iw] += local_num_bin_points
            weight_sum_bin[iw] += local_weight_sum_bin
            
            if debug:
                points_added = int(local_num_bin_points.sum())
                if debug:
                    print(f"Debug: Added {points_added} points for weight {weight_name} from blocks ({b1}, {b2})")
            
        except Exception as e:
            import warnings
            warnings.warn(
                f"Skipping Helmholtz block pair ({b1},{b2}) with weight {weight_name} "
                f"after processing error: {e}", stacklevel=2
            )
            if debug:
                import traceback
                traceback.print_exc()
            continue
    
    if debug:
        print(f"Debug: Successfully completed Helmholtz processing for block pair ({b1}, {b2})")

def set_sf(ad, varl=['dens','velx','vely','velz'], varsuf='', redo=False, auto_select=True, use_mpi=False, debug=False, **kwargs):
    """
    Calculate and store structure functions for multiple variables, automatically choosing between
    regular and memory-efficient methods based on available memory.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    varl : list of str, optional
        List of variable names to process. Default is ['dens','velx','vely','velz'].
    varsuf : str, optional
        Suffix to append to the variable names in the output dictionary.
    redo : bool, optional
        If True, recalculate results even if they exist. Default is False.
    auto_select : bool, optional
        Whether to automatically select method based on memory. Default is True.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    debug : bool, optional
        If True, print detailed debug information. Defaults to False.
    **kwargs : dict
        Additional arguments for structure function calculation.
        
    Returns
    -------
    dict
        Dictionary of calculated structure functions
    """
    # Initialize MPI if requested
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
        size = mpi_manager.size
        if rank == 0 and debug:
            print(f"[DEBUG set_sf] Using MPI with {size} ranks for structure function calculation")
    else:
        rank = 0
        size = 1
    
    # Initialize structure functions dictionary if needed
    if not hasattr(ad, 'sf'):
        ad.sf = {}
    
    # Check which variables need processing
    vars_to_process = []
    for var in varl:
        varname = var + varsuf
        if redo or varname not in ad.sf:
            vars_to_process.append(var)
    
    if not vars_to_process:
        if rank == 0 and debug:
            print(f"[DEBUG set_sf] No variables to process (all exist and redo=False)")
        return ad.sf  # Nothing to do
    
    if rank == 0 and debug:
        print(f"[DEBUG set_sf] Variables to process: {vars_to_process}")
    
    # Determine whether to use the memory-efficient batch method
    # Always use get_sf_mb when MPI is enabled (get_sf is only for single CPU/GPU)
    if use_mpi and mpi_manager is not None:
        use_mb_method = True
        if rank == 0 and debug:
            print(f"[DEBUG set_sf] Using batch method (get_sf_mb) for MPI distributed computation")
    else:
        use_mb_method = True  # Default
        
        if auto_select:
            n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
            n_weights = 1  # Default for single weight
            
            # Get weight info from kwargs
            weights = kwargs.get('weights', 'vol')
            if isinstance(weights, list):
                n_weights = len(weights)
            
            # Use determine_blocks_per_batch to assess memory with 3x safety factor
            # to account for temporary arrays and CUDA kernel workspace
            blocks_per_batch = determine_blocks_per_batch(
                ad.n_mbs, 
                len(vars_to_process) * 3, 
                n_weights,
                n_points_per_block
            )
            
            use_mb_method = blocks_per_batch < (ad.n_mbs * 3)
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf] Memory assessment: Can process {blocks_per_batch} of {ad.n_mbs} blocks at once")
                print(f"[DEBUG set_sf] Using {'batch (get_sf_mb)' if use_mb_method else 'standard (get_sf)'} method")
    
    try:
        # Choose the appropriate method based on memory assessment
        # CRITICAL: When MPI is enabled, ALWAYS use get_sf_mb (never get_sf)
        if use_mb_method or len(vars_to_process) > 1 or (use_mpi and mpi_manager is not None):
            # Memory-efficient batch method with multiple variables
            # Pass mpi_manager to get_sf_mb
            if rank == 0 and debug:
                print(f"[DEBUG set_sf] Calling get_sf_mb for {len(vars_to_process)} variables")
            
            r_, sf_result, weight_sum, num_bin_points = get_sf_mb(
                ad, 
                vars_to_process, 
                mpi_manager=mpi_manager,
                debug=debug,
                **kwargs
            )
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf] get_sf_mb returned:")
                print(f"[DEBUG set_sf]   r_.shape = {r_.shape}")
                print(f"[DEBUG set_sf]   sf_result type = {type(sf_result)}")
                if isinstance(sf_result, list):
                    print(f"[DEBUG set_sf]   sf_result length = {len(sf_result)}")
                    for i, sf_i in enumerate(sf_result):
                        print(f"[DEBUG set_sf]   sf_result[{i}].shape = {sf_i.shape}")
                else:
                    print(f"[DEBUG set_sf]   sf_result.shape = {sf_result.shape}")
            
            # Results are already aggregated across ranks if using MPI
            # Store results for each variable (only on rank 0 or all ranks)
            for i, var in enumerate(vars_to_process):
                varname = var + varsuf
                # Extract SF data for this variable
                if len(vars_to_process) > 1:
                    sf_var = sf_result[i]
                else:
                    sf_var = sf_result
                
                # Squeeze out singleton dimensions (e.g., n_weights=1)
                # This ensures consistent shape between single-node (4, 30) and MPI (1, 4, 30) -> (4, 30)
                sf_var_squeezed = asnumpy(sf_var).squeeze()
                
                if rank == 0 and debug:
                    print(f"[DEBUG set_sf] Storing {varname}:")
                    print(f"[DEBUG set_sf]   sf.shape = {sf_var_squeezed.shape}")
                    print(f"[DEBUG set_sf]   r.shape = {asnumpy(r_).shape}")
                    print(f"[DEBUG set_sf]   sum(sf) = {np.sum(sf_var_squeezed)}")
                    print(f"[DEBUG set_sf]   all_zeros = {np.all(sf_var_squeezed == 0)}")
                
                ad.sf[varname] = {
                    'r': asnumpy(r_),
                    'sf': sf_var_squeezed,
                    'weight_sum_bin': asnumpy(weight_sum).squeeze(),
                    'num_bin_points': asnumpy(num_bin_points).squeeze()
                }
        else:
            # Single variable case - use standard method (only when MPI is NOT enabled)
            # This branch should never be reached when use_mpi=True
            if rank == 0 and debug:
                print(f"[DEBUG set_sf] Using standard get_sf method (single variable, no MPI)")
            
            var = vars_to_process[0]
            varname = var + varsuf
            r_, sf_bin, weights_bin, counts_bin = get_sf(ad, var=var, debug=debug, **kwargs)
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf] get_sf returned:")
                print(f"[DEBUG set_sf]   r_.shape = {r_.shape}")
                print(f"[DEBUG set_sf]   sf_bin.shape = {sf_bin.shape}")
                print(f"[DEBUG set_sf]   sum(sf_bin) = {np.sum(asnumpy(sf_bin))}")
                print(f"[DEBUG set_sf]   all_zeros = {np.all(asnumpy(sf_bin) == 0)}")
            
            ad.sf[varname] = {
                'r': asnumpy(r_),
                'sf': asnumpy(sf_bin),
                'weight_sum_bin': asnumpy(weights_bin),
                'num_bin_points': asnumpy(counts_bin)
            }
    except Exception as e:
        if rank == 0:
            print(f"[ERROR set_sf] Error calculating structure functions: {e}")
            import traceback
            traceback.print_exc()
        raise

    return ad.sf

def set_sf_helmholtz(ad, var='vel', varsuf='', redo=False, auto_select=True, use_mpi=False, debug=False, **kwargs):
    """
    Calculate and store Helmholtz-decomposed structure functions, automatically choosing between
    regular and memory-efficient methods based on available memory.
    
    Parameters
    ----------
    ad : AthenaData
        The Athena data object containing simulation data
    var : str, optional
        Vector field variable prefix (e.g. 'vel'). Default is 'vel'.
    varsuf : str, optional
        Suffix to append to the variable name in the output dictionary.
    redo : bool, optional
        If True, recalculate results even if they exist. Default is False.
    auto_select : bool, optional
        Whether to automatically select method based on memory. Default is True.
    use_mpi : bool, optional
        If True, distributes computation across MPI ranks. Defaults to False.
    debug : bool, optional
        If True, print detailed debug information. Defaults to False.
    **kwargs : dict
        Additional arguments for Helmholtz structure function calculation.
        
    Returns
    -------
    dict
        Dictionary of calculated structure functions
    """
    # Initialize MPI if requested
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
        size = mpi_manager.size
        if rank == 0 and debug:
            print(f"[DEBUG set_sf_helmholtz] Using MPI with {size} ranks")
    else:
        rank = 0
        size = 1
    
    # Initialize structure functions dictionary if needed
    if not hasattr(ad, 'sf'):
        ad.sf = {}
    
    # Check if processing is needed
    varname = var + varsuf
    if not redo and varname in ad.sf:
        if rank == 0 and debug:
            print(f"[DEBUG set_sf_helmholtz] {varname} already exists and redo=False, skipping")
        return ad.sf  # Nothing to do
    
    if rank == 0 and debug:
        print(f"[DEBUG set_sf_helmholtz] Processing variable: {varname}")
    
    # Determine whether to use the memory-efficient batch method
    # Always use get_sf_helmholtz_mb when MPI is enabled (get_sf_helmholtz is only for single CPU/GPU)
    if use_mpi and mpi_manager is not None:
        use_mb_method = True
        if rank == 0 and debug:
            print(f"[DEBUG set_sf_helmholtz] Using batch method (get_sf_helmholtz_mb) for MPI distributed computation")
    else:
        use_mb_method = True  # Default
        
        if auto_select:
            n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
            n_weights = 1  # Default for single weight
            
            # Get weight info from kwargs
            weights = kwargs.get('weights', 'vol')
            if isinstance(weights, list):
                n_weights = len(weights)
            
            # Use determine_blocks_per_batch to assess memory with safety factor
            # Vector field has 3 components (x,y,z), apply 3x safety factor = 9 total
            blocks_per_batch = determine_blocks_per_batch(
                ad.n_mbs, 
                9,  # 3 vector components * 3 safety factor for temporaries
                n_weights,
                n_points_per_block
            )
            
            use_mb_method = blocks_per_batch < (ad.n_mbs * 3)
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf_helmholtz] Memory assessment: Can process {blocks_per_batch} of {ad.n_mbs} blocks at once")
                print(f"[DEBUG set_sf_helmholtz] Using {'batch (get_sf_helmholtz_mb)' if use_mb_method else 'standard (get_sf_helmholtz)'} method")
    
    try:
        # Choose the appropriate method based on memory assessment
        # CRITICAL: When MPI is enabled, ALWAYS use get_sf_helmholtz_mb (never get_sf_helmholtz)
        if use_mb_method or (use_mpi and mpi_manager is not None):
            # Memory-efficient batch method
            # Pass mpi_manager to get_sf_helmholtz_mb
            if rank == 0 and debug:
                print(f"[DEBUG set_sf_helmholtz] Calling get_sf_helmholtz_mb")
            
            r_, sf_comp, sf_sol, weight_sum, num_bin_points = get_sf_helmholtz_mb(
                ad, 
                var=var,
                mpi_manager=mpi_manager,
                debug=debug,
                **kwargs
            )
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf_helmholtz] get_sf_helmholtz_mb returned:")
                print(f"[DEBUG set_sf_helmholtz]   r_.shape = {r_.shape}")
                print(f"[DEBUG set_sf_helmholtz]   sf_comp.shape = {sf_comp.shape}")
                print(f"[DEBUG set_sf_helmholtz]   sf_sol.shape = {sf_sol.shape}")
        else:
            # Standard method (only when MPI is NOT enabled)
            # This branch should never be reached when use_mpi=True
            if rank == 0 and debug:
                print(f"[DEBUG set_sf_helmholtz] Using standard get_sf_helmholtz method")
            
            r_, sf_comp, sf_sol, weight_sum, num_bin_points = get_sf_helmholtz(
                ad,
                var=var,
                debug=debug,
                **kwargs
            )
            
            if rank == 0 and debug:
                print(f"[DEBUG set_sf_helmholtz] get_sf_helmholtz returned:")
                print(f"[DEBUG set_sf_helmholtz]   r_.shape = {r_.shape}")
                print(f"[DEBUG set_sf_helmholtz]   sf_comp.shape = {sf_comp.shape}")
                print(f"[DEBUG set_sf_helmholtz]   sf_sol.shape = {sf_sol.shape}")
        
        # Check for zero data
        sf_comp_numpy = asnumpy(sf_comp)
        sf_sol_numpy = asnumpy(sf_sol)
        
        if rank == 0 and debug:
            print(f"[DEBUG set_sf_helmholtz] Storing {varname}:")
            print(f"[DEBUG set_sf_helmholtz]   sum(sf_comp) = {np.sum(sf_comp_numpy)}")
            print(f"[DEBUG set_sf_helmholtz]   sum(sf_sol) = {np.sum(sf_sol_numpy)}")
            print(f"[DEBUG set_sf_helmholtz]   comp all_zeros = {np.all(sf_comp_numpy == 0)}")
            print(f"[DEBUG set_sf_helmholtz]   sol all_zeros = {np.all(sf_sol_numpy == 0)}")
        
        # Store results
        ad.sf[varname] = {
            'r': asnumpy(r_),
            'sf_comp': sf_comp_numpy,
            'sf_sol': sf_sol_numpy,
            'weight_sum_bin': asnumpy(weight_sum),
            'num_bin_points': asnumpy(num_bin_points)
        }
        
    except Exception as e:
        if rank == 0:
            print(f"[ERROR set_sf_helmholtz] Error calculating Helmholtz structure functions: {e}")
            import traceback
            traceback.print_exc()
        raise

    return ad.sf


# ---------------------------------------------------------------------------
# Anisotropic MHD Structure Functions
# Bins separations in 2D (l_prll, l_perp) relative to the local mean B_bar = (B1+B2)/2
# and stores the prll and perp components of deltavel and deltabcc separately.
# ---------------------------------------------------------------------------

def _allocate_aniso_mhd_accumulators(n_weights, max_order, nbins, nbins_prll, nbins_perp,
                                   array_module=xp):
    """Allocate nested accumulator arrays for anisotropic MHD SF products."""
    dtype = array_module.float64
    return {
        'vel': {
            'sf': array_module.zeros((n_weights, max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((n_weights, max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((n_weights, max_order, nbins_perp), dtype=dtype),
        },
        'vel_perp': {
            'sf': array_module.zeros((n_weights, max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((n_weights, max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((n_weights, max_order, nbins_perp), dtype=dtype),
        },
        'bcc': {
            'sf': array_module.zeros((n_weights, max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((n_weights, max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((n_weights, max_order, nbins_perp), dtype=dtype),
        },
        'bcc_perp': {
            'sf': array_module.zeros((n_weights, max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((n_weights, max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((n_weights, max_order, nbins_perp), dtype=dtype),
        },
        'counts': {
            'num': array_module.zeros((n_weights, nbins), dtype=dtype),
            'num_prll': array_module.zeros((n_weights, nbins_prll), dtype=dtype),
            'num_perp': array_module.zeros((n_weights, nbins_perp), dtype=dtype),
        },
        'weights': {
            'wsum': array_module.zeros((n_weights, nbins), dtype=dtype),
            'wsum_prll': array_module.zeros((n_weights, nbins_prll), dtype=dtype),
            'wsum_perp': array_module.zeros((n_weights, nbins_perp), dtype=dtype),
        },
    }


def _allocate_aniso_mhd_pair_workspace(max_order, nbins, nbins_prll, nbins_perp,
                                       array_module=xp):
    """Allocate reusable per-pair work buffers for anisotropic MHD SFs."""
    dtype = array_module.float64
    return {
        'vel': {
            'sf': array_module.zeros((max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((max_order, nbins_perp), dtype=dtype),
        },
        'vel_perp': {
            'sf': array_module.zeros((max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((max_order, nbins_perp), dtype=dtype),
        },
        'bcc': {
            'sf': array_module.zeros((max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((max_order, nbins_perp), dtype=dtype),
        },
        'bcc_perp': {
            'sf': array_module.zeros((max_order, nbins), dtype=dtype),
            'sf_prll': array_module.zeros((max_order, nbins_prll), dtype=dtype),
            'sf_perp': array_module.zeros((max_order, nbins_perp), dtype=dtype),
        },
        'counts': {
            'num': array_module.zeros(nbins, dtype=dtype),
            'num_prll': array_module.zeros(nbins_prll, dtype=dtype),
            'num_perp': array_module.zeros(nbins_perp, dtype=dtype),
        },
        'weights': {
            'wsum': array_module.zeros(nbins, dtype=dtype),
            'wsum_prll': array_module.zeros(nbins_prll, dtype=dtype),
            'wsum_perp': array_module.zeros(nbins_perp, dtype=dtype),
        },
    }


def _zero_aniso_mhd_pair_workspace(workspace):
    """Reset reusable anisotropic MHD pair work buffers in place."""
    for group in workspace.values():
        for arr in group.values():
            arr.fill(0)


def _squeeze_aniso_mhd_accumulators(accum):
    """Drop the leading weight dimension for single-weight outputs."""
    for group in accum.values():
        for key, arr in group.items():
            group[key] = arr[0]
    return accum


def _reduce_aniso_mhd_accumulators(accum, mpi_manager):
    """MPI-reduce and broadcast all anisotropic MHD accumulator arrays."""
    def _reduce(arr):
        arr_np = asnumpy(arr)
        total = mpi_manager.reduce(arr_np, op='sum', root=0)
        total = mpi_manager.broadcast(total, root=0)
        return xp.asarray(total)

    for group in accum.values():
        for key, arr in group.items():
            group[key] = _reduce(arr)
    return accum


DEFAULT_ANISO_MHD_SAME_BLOCK_PAIR_BOOST = 100.0
DEFAULT_ANISO_MHD_FACE_BLOCK_PAIR_BOOST = 16.0
DEFAULT_ANISO_MHD_EDGE_BLOCK_PAIR_BOOST = 4.0
DEFAULT_ANISO_MHD_CORNER_BLOCK_PAIR_BOOST = 2.0


def _classify_aniso_mhd_block_topology(ad, b1, b2,
                                       periodic_x, periodic_y, periodic_z):
    """Classify the meshblock-pair topology for anisotropic VSF sampling.

    The classification uses logical meshblock coordinates so same-block and
    same-level nearest-neighbour pairs can be preferentially sampled without
    affecting the isotropic structure-function path.
    """
    if b1 == b2:
        return 'same'

    if not hasattr(ad, 'mb_logical') or ad.mb_logical is None:
        return 'far'

    loc1 = np.asarray(ad.mb_logical[b1, :3], dtype=int)
    loc2 = np.asarray(ad.mb_logical[b2, :3], dtype=int)
    level1 = int(ad.mb_logical[b1, -1])
    level2 = int(ad.mb_logical[b2, -1])

    if level1 != level2:
        return 'far'

    root_counts = np.array([
        max(1, ad.Nx1 // ad.nx1),
        max(1, ad.Nx2 // ad.nx2),
        max(1, ad.Nx3 // ad.nx3),
    ], dtype=int)
    periodic_flags = (periodic_x, periodic_y, periodic_z)
    level_counts = root_counts * (2 ** level1)

    separated_axes = 0
    for axis in range(3):
        delta = abs(loc2[axis] - loc1[axis])
        if periodic_flags[axis]:
            delta = min(delta, max(0, level_counts[axis] - delta))
        if delta == 0:
            continue
        if delta == 1:
            separated_axes += 1
            continue
        return 'far'

    topology_names = ('same', 'face', 'edge', 'corner')
    return topology_names[separated_axes]


def _get_aniso_mhd_pair_multiplier(topology, use_nearfield_bias,
                                   same_block_pair_boost,
                                   face_block_pair_boost,
                                   edge_block_pair_boost,
                                   corner_block_pair_boost):
    """Return the topology-aware sampling boost for anisotropic VSFs."""
    if not use_nearfield_bias:
        return 1.0

    return {
        'same': same_block_pair_boost,
        'face': face_block_pair_boost,
        'edge': edge_block_pair_boost,
        'corner': corner_block_pair_boost,
    }.get(topology, 1.0)


# ---------------------------------------------------------------------------
# Anisotropic MHD Structure Functions
# Bins separations in 1D (l, l_prll, l_perp) relative to the local mean field
# B_bar = (B1+B2)/2 and stores both total and perpendicular-to-B increments.
# ---------------------------------------------------------------------------

def _process_block_pair_aniso_mhd(ad, b1, b2, block_cache, block_centers,
                                   weights_list, vel_varlist, bcc_varlist,
                                   sf_accum,
                                   rmax, rmin, Lx, Ly, Lz,
                                   lbins, lprll_bins, lperp_bins,
                                   nbins, nbins_prll, nbins_perp, max_order,
                                   cos_theta_prll_min, cos_theta_perp_max,
                                   n_samples1, n_points_per_block, nsamples_block_min,
                                   periodic_x, periodic_y, periodic_z,
                                   use_nearfield_bias=True,
                                   same_block_pair_boost=DEFAULT_ANISO_MHD_SAME_BLOCK_PAIR_BOOST,
                                   face_block_pair_boost=DEFAULT_ANISO_MHD_FACE_BLOCK_PAIR_BOOST,
                                   edge_block_pair_boost=DEFAULT_ANISO_MHD_EDGE_BLOCK_PAIR_BOOST,
                                   corner_block_pair_boost=DEFAULT_ANISO_MHD_CORNER_BLOCK_PAIR_BOOST,
                                   sparse_data=False, debug=False, use_mpi=False, random_seed=None,
                                   pair_workspace=None):
    """Process a single pair of blocks for anisotropic MHD structure functions."""

    if debug:
        print(f'Debug: Processing aniso_mhd block pair ({b1}, {b2})')

    region_status1 = block_cache[b1].get('xyz_region_status', 'inside')
    region_status2 = block_cache[b2].get('xyz_region_status', 'inside')
    region_trimmed1 = block_cache[b1].get('xyz_region_trimmed', False)
    region_trimmed2 = block_cache[b2].get('xyz_region_trimmed', False)
    if region_status1 == 'outside' or region_status2 == 'outside':
        if debug:
            print(f'Debug: Skipping aniso_mhd pair ({b1}, {b2}) due to outside block')
        return
    pair_sparse_data = sparse_data and (
        (region_status1 == 'partial' and not region_trimmed1) or
        (region_status2 == 'partial' and not region_trimmed2))

    dist_x = abs(block_centers[b1, 0] - block_centers[b2, 0])
    dist_y = abs(block_centers[b1, 1] - block_centers[b2, 1])
    dist_z = abs(block_centers[b1, 2] - block_centers[b2, 2])
    if periodic_x:
        dist_x = min(dist_x, abs(Lx - dist_x))
    if periodic_y:
        dist_y = min(dist_y, abs(Ly - dist_y))
    if periodic_z:
        dist_z = min(dist_z, abs(Lz - dist_z))
    block_distance = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)

    topology = _classify_aniso_mhd_block_topology(
        ad, b1, b2, periodic_x, periodic_y, periodic_z)
    pair_multiplier = _get_aniso_mhd_pair_multiplier(
        topology, use_nearfield_bias,
        same_block_pair_boost, face_block_pair_boost,
        edge_block_pair_boost, corner_block_pair_boost)
    same_block_pair = (b1 == b2)
    workspace = pair_workspace
    if workspace is None:
        workspace = _allocate_aniso_mhd_pair_workspace(
            max_order, nbins, nbins_prll, nbins_perp, array_module=xp)

    n_samples2_norm = n_samples1 / rmax / (rmax / rmin - 1.0)
    n_samples2 = n_samples2_norm * (n_samples1 if block_distance == 0 else (block_distance / rmax)**-2.0)
    n_samples2 = int(np.clip(n_samples2, nsamples_block_min, n_samples1))

    x1 = block_cache[b1]['coords']['x']
    y1 = block_cache[b1]['coords']['y']
    z1 = block_cache[b1]['coords']['z']
    x2 = block_cache[b2]['coords']['x']
    y2 = block_cache[b2]['coords']['y']
    z2 = block_cache[b2]['coords']['z']

    vx1 = block_cache[b1]['vars'][vel_varlist[0]]
    vy1 = block_cache[b1]['vars'][vel_varlist[1]]
    vz1 = block_cache[b1]['vars'][vel_varlist[2]]
    vx2 = block_cache[b2]['vars'][vel_varlist[0]]
    vy2 = block_cache[b2]['vars'][vel_varlist[1]]
    vz2 = block_cache[b2]['vars'][vel_varlist[2]]

    bx1 = block_cache[b1]['vars'][bcc_varlist[0]]
    by1 = block_cache[b1]['vars'][bcc_varlist[1]]
    bz1 = block_cache[b1]['vars'][bcc_varlist[2]]
    bx2 = block_cache[b2]['vars'][bcc_varlist[0]]
    by2 = block_cache[b2]['vars'][bcc_varlist[1]]
    bz2 = block_cache[b2]['vars'][bcc_varlist[2]]

    for iw, weight_name in enumerate(weights_list):
        weight1 = block_cache[b1]['weights'][weight_name]
        weight2 = block_cache[b2]['weights'][weight_name]
        _zero_aniso_mhd_pair_workspace(workspace)

        try:
            if pair_sparse_data:
                valid1 = weight1 > 0
                valid2 = weight2 > 0
                if cupy_enabled:
                    vc1 = int(xp.sum(valid1).get())
                    vc2 = int(xp.sum(valid2).get())
                else:
                    vc1 = int(xp.sum(valid1))
                    vc2 = int(xp.sum(valid2))
                if vc1 == 0 or vc2 == 0:
                    continue
                x1v, y1v, z1v = x1[valid1], y1[valid1], z1[valid1]
                n1 = vc1
                if same_block_pair:
                    # Preserve pointer identity so the exhaustive kernel can
                    # treat sparse same-block pairs as triangular, not n1*n2.
                    x2v, y2v, z2v = x1v, y1v, z1v
                    w1v = weight1[valid1]
                    w2v = w1v
                    vx1v, vy1v, vz1v = vx1[valid1], vy1[valid1], vz1[valid1]
                    vx2v, vy2v, vz2v = vx1v, vy1v, vz1v
                    bx1v, by1v, bz1v = bx1[valid1], by1[valid1], bz1[valid1]
                    bx2v, by2v, bz2v = bx1v, by1v, bz1v
                    n2 = n1
                else:
                    x2v, y2v, z2v = x2[valid2], y2[valid2], z2[valid2]
                    w1v, w2v = weight1[valid1], weight2[valid2]
                    vx1v, vy1v, vz1v = vx1[valid1], vy1[valid1], vz1[valid1]
                    vx2v, vy2v, vz2v = vx2[valid2], vy2[valid2], vz2[valid2]
                    bx1v, by1v, bz1v = bx1[valid1], by1[valid1], bz1[valid1]
                    bx2v, by2v, bz2v = bx2[valid2], by2[valid2], bz2[valid2]
                    n2 = vc2
            else:
                x1v, y1v, z1v = x1, y1, z1
                x2v, y2v, z2v = x2, y2, z2
                w1v, w2v = weight1, weight2
                vx1v, vy1v, vz1v = vx1, vy1, vz1
                vx2v, vy2v, vz2v = vx2, vy2, vz2
                bx1v, by1v, bz1v = bx1, by1, bz1
                bx2v, by2v, bz2v = bx2, by2, bz2
                n1 = x1v.size
                n2 = x2v.size

            n1_adj = min(n_samples1, n1)
            n2_adj = min(n_samples2, n2)

            loc_vel_sf = workspace['vel']['sf']
            loc_vel_sf_prll = workspace['vel']['sf_prll']
            loc_vel_sf_perp = workspace['vel']['sf_perp']
            loc_vel_perp_sf = workspace['vel_perp']['sf']
            loc_vel_perp_sf_prll = workspace['vel_perp']['sf_prll']
            loc_vel_perp_sf_perp = workspace['vel_perp']['sf_perp']
            loc_bcc_sf = workspace['bcc']['sf']
            loc_bcc_sf_prll = workspace['bcc']['sf_prll']
            loc_bcc_sf_perp = workspace['bcc']['sf_perp']
            loc_bcc_perp_sf = workspace['bcc_perp']['sf']
            loc_bcc_perp_sf_prll = workspace['bcc_perp']['sf_prll']
            loc_bcc_perp_sf_perp = workspace['bcc_perp']['sf_perp']
            loc_num = workspace['counts']['num']
            loc_num_prll = workspace['counts']['num_prll']
            loc_num_perp = workspace['counts']['num_perp']
            loc_wsum = workspace['weights']['wsum']
            loc_wsum_prll = workspace['weights']['wsum_prll']
            loc_wsum_perp = workspace['weights']['wsum_perp']

            threadsperblock = 256
            max_random_pairs = np.iinfo(np.int32).max // 2
            max_exhaustive_pairs = np.iinfo(np.int32).max // 4
            n_valid_pairs = n1 * n2
            n_exhaustive_pairs = (n1 * (n1 - 1) // 2) if same_block_pair else n_valid_pairs
            baseline_pairs = max(1, int(n1_adj * n2_adj))
            target_pairs = min(
                n_valid_pairs,
                max_random_pairs,
                max(1, int(np.ceil(baseline_pairs * pair_multiplier))))
            allow_exhaustive = not (region_trimmed1 or region_trimmed2)

            # The topology boost is meant to increase random sampling for
            # near-field pairs, not to force those pairs into the exhaustive
            # kernel. Keep the exhaustive crossover tied to the baseline
            # sampling budget so the boost does not explode runtime.
            if (allow_exhaustive and n_exhaustive_pairs > 0 and
                    n_exhaustive_pairs <= max_exhaustive_pairs and
                    n_exhaustive_pairs < 5 * baseline_pairs):
                blockspergrid = (n_exhaustive_pairs + threadsperblock - 1) // threadsperblock
                structure_function_aniso_mhd_kernel_exhaustive(
                    (blockspergrid,), (threadsperblock,),
                    (vx1v, vy1v, vz1v, vx2v, vy2v, vz2v,
                     bx1v, by1v, bz1v, bx2v, by2v, bz2v,
                     w1v, w2v,
                     x1v, y1v, z1v, x2v, y2v, z2v,
                     lbins, lprll_bins, lperp_bins,
                     loc_vel_sf.ravel(), loc_vel_sf_prll.ravel(), loc_vel_sf_perp.ravel(),
                     loc_vel_perp_sf.ravel(), loc_vel_perp_sf_prll.ravel(), loc_vel_perp_sf_perp.ravel(),
                     loc_bcc_sf.ravel(), loc_bcc_sf_prll.ravel(), loc_bcc_sf_perp.ravel(),
                     loc_bcc_perp_sf.ravel(), loc_bcc_perp_sf_prll.ravel(), loc_bcc_perp_sf_perp.ravel(),
                     loc_num, loc_num_prll, loc_num_perp,
                     loc_wsum, loc_wsum_prll, loc_wsum_perp,
                     n1, n2,
                     float(Lx), float(Ly), float(Lz),
                     nbins, nbins_prll, nbins_perp, max_order,
                     float(cos_theta_prll_min), float(cos_theta_perp_max),
                     periodic_x, periodic_y, periodic_z))
            else:
                n_pairs = int(min(target_pairs, max_random_pairs))
                if n_pairs <= 0:
                    continue
                blockspergrid = (n_pairs + threadsperblock - 1) // threadsperblock
                seed = random_seed if random_seed is not None else (b1 * 10000 + b2)
                structure_function_aniso_mhd_kernel(
                    (blockspergrid,), (threadsperblock,),
                    (vx1v, vy1v, vz1v, vx2v, vy2v, vz2v,
                     bx1v, by1v, bz1v, bx2v, by2v, bz2v,
                     w1v, w2v,
                     x1v, y1v, z1v, x2v, y2v, z2v,
                     lbins, lprll_bins, lperp_bins,
                     loc_vel_sf.ravel(), loc_vel_sf_prll.ravel(), loc_vel_sf_perp.ravel(),
                     loc_vel_perp_sf.ravel(), loc_vel_perp_sf_prll.ravel(), loc_vel_perp_sf_perp.ravel(),
                     loc_bcc_sf.ravel(), loc_bcc_sf_prll.ravel(), loc_bcc_sf_perp.ravel(),
                     loc_bcc_perp_sf.ravel(), loc_bcc_perp_sf_prll.ravel(), loc_bcc_perp_sf_perp.ravel(),
                     loc_num, loc_num_prll, loc_num_perp,
                     loc_wsum, loc_wsum_prll, loc_wsum_perp,
                     n_pairs, n1, n2,
                     float(Lx), float(Ly), float(Lz),
                     nbins, nbins_prll, nbins_perp, max_order,
                     float(cos_theta_prll_min), float(cos_theta_perp_max),
                     periodic_x, periodic_y, periodic_z, seed))

            sf_accum['vel']['sf'][iw] += loc_vel_sf
            sf_accum['vel']['sf_prll'][iw] += loc_vel_sf_prll
            sf_accum['vel']['sf_perp'][iw] += loc_vel_sf_perp
            sf_accum['vel_perp']['sf'][iw] += loc_vel_perp_sf
            sf_accum['vel_perp']['sf_prll'][iw] += loc_vel_perp_sf_prll
            sf_accum['vel_perp']['sf_perp'][iw] += loc_vel_perp_sf_perp
            sf_accum['bcc']['sf'][iw] += loc_bcc_sf
            sf_accum['bcc']['sf_prll'][iw] += loc_bcc_sf_prll
            sf_accum['bcc']['sf_perp'][iw] += loc_bcc_sf_perp
            sf_accum['bcc_perp']['sf'][iw] += loc_bcc_perp_sf
            sf_accum['bcc_perp']['sf_prll'][iw] += loc_bcc_perp_sf_prll
            sf_accum['bcc_perp']['sf_perp'][iw] += loc_bcc_perp_sf_perp
            sf_accum['counts']['num'][iw] += loc_num
            sf_accum['counts']['num_prll'][iw] += loc_num_prll
            sf_accum['counts']['num_perp'][iw] += loc_num_perp
            sf_accum['weights']['wsum'][iw] += loc_wsum
            sf_accum['weights']['wsum_prll'][iw] += loc_wsum_prll
            sf_accum['weights']['wsum_perp'][iw] += loc_wsum_perp

        except Exception as e:
            import warnings
            warnings.warn(
                f"Skipping aniso-MHD block pair ({b1},{b2}) with weight {weight_name} "
                f"after processing error: {e}", stacklevel=2
            )
            if debug:
                import traceback
                traceback.print_exc()
            continue

    if debug:
        print(f'Debug: Completed aniso_mhd block pair ({b1}, {b2})')


def _process_batch_pairs_aniso_mhd(ad, batches, block_cache, processed_pairs,
                                    weights_list, vel_varlist, bcc_varlist,
                                    sf_accum,
                                    block_centers, rmax, rmin, Lx, Ly, Lz,
                                    lbins, lprll_bins, lperp_bins,
                                    nbins, nbins_prll, nbins_perp, max_order,
                                    cos_theta_prll_min, cos_theta_perp_max,
                                    n_samples1, n_points_per_block, nsamples_block_min,
                                    periodic_x, periodic_y, periodic_z,
                                    use_nearfield_bias=True,
                                    same_block_pair_boost=DEFAULT_ANISO_MHD_SAME_BLOCK_PAIR_BOOST,
                                    face_block_pair_boost=DEFAULT_ANISO_MHD_FACE_BLOCK_PAIR_BOOST,
                                    edge_block_pair_boost=DEFAULT_ANISO_MHD_EDGE_BLOCK_PAIR_BOOST,
                                    corner_block_pair_boost=DEFAULT_ANISO_MHD_CORNER_BLOCK_PAIR_BOOST,
                                    sparse_data=False, xyzlim=None, debug=False,
                                    use_mpi=False, mpi_manager=None, mb_offset=0,
                                    pair_workspace=None):
    """Process batches of block pairs for anisotropic MHD structure functions."""

    if debug:
        print(f'Debug: Starting aniso_mhd batch processing with {len(batches)} batches')

    all_varlist = vel_varlist + bcc_varlist

    blocks_to_skip = set()
    if xyzlim is not None:
        for b in range(ad.n_mbs):
            if is_block_outside_xyz(ad.mb_geometry, b, xyzlim):
                blocks_to_skip.add(b)

    for batch_idx1, batch1 in enumerate(batches):
        blocks_to_load1 = [b for b in batch1 if b not in block_cache and b not in blocks_to_skip]
        if blocks_to_load1:
            _load_batch_data(ad, block_cache, blocks_to_load1, weights_list, all_varlist, xyzlim,
                             mpi_manager=mpi_manager, mb_offset=mb_offset)

        for batch_idx2 in range(batch_idx1, len(batches)):
            batch2 = batches[batch_idx2]
            blocks_to_load2 = [b for b in batch2 if b not in block_cache and b not in blocks_to_skip]
            if blocks_to_load2:
                _load_batch_data(ad, block_cache, blocks_to_load2, weights_list, all_varlist, xyzlim,
                                 mpi_manager=mpi_manager, mb_offset=mb_offset)

            for b1 in batch1:
                if b1 in blocks_to_skip or b1 not in block_cache:
                    continue
                b2_start = b1 if batch_idx1 == batch_idx2 else batch2[0]
                for b2 in batch2:
                    if b2 < b2_start:
                        continue
                    if b2 in blocks_to_skip or b2 not in block_cache:
                        continue
                    pair = (min(b1, b2), max(b1, b2))
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)

                    _process_block_pair_aniso_mhd(
                        ad, b1, b2, block_cache, block_centers,
                        weights_list, vel_varlist, bcc_varlist,
                        sf_accum,
                        rmax, rmin, Lx, Ly, Lz,
                        lbins, lprll_bins, lperp_bins,
                        nbins, nbins_prll, nbins_perp, max_order,
                        cos_theta_prll_min, cos_theta_perp_max,
                        n_samples1, n_points_per_block, nsamples_block_min,
                        periodic_x, periodic_y, periodic_z,
                        use_nearfield_bias=use_nearfield_bias,
                        same_block_pair_boost=same_block_pair_boost,
                        face_block_pair_boost=face_block_pair_boost,
                        edge_block_pair_boost=edge_block_pair_boost,
                        corner_block_pair_boost=corner_block_pair_boost,
                        sparse_data=sparse_data, debug=debug, use_mpi=use_mpi,
                        pair_workspace=pair_workspace)

            if batch_idx2 != batch_idx1:
                for b in batch2:
                    if b in block_cache and b not in batch1:
                        del block_cache[b]

        for b in batch1:
            if b in block_cache:
                del block_cache[b]

        if debug:
            print(f'Debug: Completed aniso_mhd batch {batch_idx1 + 1}/{len(batches)}')


def _process_mpi_pairs_aniso_mhd(ad, my_pairs, block_cache,
                                   weights_list, vel_varlist, bcc_varlist,
                                   sf_accum,
                                   rmax, rmin, Lx, Ly, Lz,
                                   lbins, lprll_bins, lperp_bins,
                                   nbins, nbins_prll, nbins_perp, max_order,
                                   cos_theta_prll_min, cos_theta_perp_max,
                                   n_samples1, n_points_per_block, nsamples_block_min,
                                   periodic_x, periodic_y, periodic_z,
                                   use_nearfield_bias=True,
                                   same_block_pair_boost=DEFAULT_ANISO_MHD_SAME_BLOCK_PAIR_BOOST,
                                   face_block_pair_boost=DEFAULT_ANISO_MHD_FACE_BLOCK_PAIR_BOOST,
                                   edge_block_pair_boost=DEFAULT_ANISO_MHD_EDGE_BLOCK_PAIR_BOOST,
                                   corner_block_pair_boost=DEFAULT_ANISO_MHD_CORNER_BLOCK_PAIR_BOOST,
                                   sparse_data=False, xyzlim=None,
                                   mpi_manager=None, mb_offset=0, debug=False,
                                   pair_workspace=None):
    """Distribute aniso_mhd block pairs across MPI ranks."""

    rank = mpi_manager.rank if mpi_manager else 0
    all_varlist = vel_varlist + bcc_varlist

    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs_local = ad.local_mb_end - ad.local_mb_start
    else:
        nmbs_local = ad.n_mbs
    local_start = mb_offset
    local_end = mb_offset + nmbs_local

    block_centers_global = np.zeros((ad.n_mbs, 3))
    full_geom = ad.mb_geometry[:ad.n_mbs, :]
    block_centers_global[:, 0] = 0.5 * (full_geom[:, 0] + full_geom[:, 1])
    block_centers_global[:, 1] = 0.5 * (full_geom[:, 2] + full_geom[:, 3])
    block_centers_global[:, 2] = 0.5 * (full_geom[:, 4] + full_geom[:, 5])

    from athena_research.utils.batch_processing import determine_blocks_per_batch
    blocks_per_batch = determine_blocks_per_batch(
        ad.n_mbs, len(all_varlist) * 3, len(weights_list), n_points_per_block)

    pairs_by_locality = {'both_local': [], 'one_local': [], 'both_remote': []}
    for gb1, gb2 in my_pairs:
        b1l = local_start <= gb1 < local_end
        b2l = local_start <= gb2 < local_end
        if b1l and b2l:
            pairs_by_locality['both_local'].append((gb1, gb2))
        elif b1l or b2l:
            pairs_by_locality['one_local'].append((gb1, gb2))
        else:
            pairs_by_locality['both_remote'].append((gb1, gb2))

    optimized_pairs = (pairs_by_locality['both_local'] +
                       pairs_by_locality['one_local'] +
                       pairs_by_locality['both_remote'])

    pairs_processed = 0
    pair_idx = 0

    while pair_idx < len(optimized_pairs):
        batch_blocks_needed = set()
        batch_end_idx = pair_idx
        while batch_end_idx < len(optimized_pairs) and len(batch_blocks_needed) < blocks_per_batch:
            gb1, gb2 = optimized_pairs[batch_end_idx]
            new_blocks = set()
            if gb1 not in block_cache:
                new_blocks.add(gb1)
            if gb2 not in block_cache:
                new_blocks.add(gb2)
            if len(batch_blocks_needed) + len(new_blocks) > blocks_per_batch and len(batch_blocks_needed) > 0:
                break
            batch_blocks_needed.update(new_blocks)
            batch_end_idx += 1

        batch_pairs = optimized_pairs[pair_idx:batch_end_idx]

        if batch_blocks_needed:
            _load_all_blocks_upfront(ad, block_cache, list(batch_blocks_needed),
                                     weights_list, all_varlist,
                                     local_start, local_end, xyzlim, mpi_manager, debug=False)

        for gb1, gb2 in batch_pairs:
            if xyzlim is not None:
                if is_block_outside_xyz(ad.mb_geometry, gb1, xyzlim) or                    is_block_outside_xyz(ad.mb_geometry, gb2, xyzlim):
                    continue
            if gb1 not in block_cache or gb2 not in block_cache:
                continue
            _process_block_pair_aniso_mhd(
                ad, gb1, gb2, block_cache, block_centers_global,
                weights_list, vel_varlist, bcc_varlist,
                sf_accum,
                rmax, rmin, Lx, Ly, Lz,
                lbins, lprll_bins, lperp_bins,
                nbins, nbins_prll, nbins_perp, max_order,
                cos_theta_prll_min, cos_theta_perp_max,
                n_samples1, n_points_per_block, nsamples_block_min,
                periodic_x, periodic_y, periodic_z,
                use_nearfield_bias=use_nearfield_bias,
                same_block_pair_boost=same_block_pair_boost,
                face_block_pair_boost=face_block_pair_boost,
                edge_block_pair_boost=edge_block_pair_boost,
                corner_block_pair_boost=corner_block_pair_boost,
                sparse_data=sparse_data, debug=False, use_mpi=True,
                random_seed=(gb1 * 10000 + gb2),
                pair_workspace=pair_workspace)
            pairs_processed += 1

        for block_id in batch_blocks_needed:
            if block_id in block_cache:
                del block_cache[block_id]

        pair_idx = batch_end_idx

    if debug:
        print(f'Debug: Rank {rank}: Completed {pairs_processed} aniso_mhd pairs')


def get_sf_aniso_mhd_mb(ad, vel_var='vel', bcc_var='bcc', weights='ones', xyz=None,
                         max_order=10, npairs=1e7, nbins=None, nbins_prll=50, nbins_perp=50,
                         theta_prll_max=None, theta_perp_min=None,
                         log_bin_flag=True, nsamples_block_min=1000,
                         use_nearfield_bias=True,
                         same_block_pair_boost=DEFAULT_ANISO_MHD_SAME_BLOCK_PAIR_BOOST,
                         face_block_pair_boost=DEFAULT_ANISO_MHD_FACE_BLOCK_PAIR_BOOST,
                         edge_block_pair_boost=DEFAULT_ANISO_MHD_EDGE_BLOCK_PAIR_BOOST,
                         corner_block_pair_boost=DEFAULT_ANISO_MHD_CORNER_BLOCK_PAIR_BOOST,
                         sparse_data=False, debug=False, simultaneous_blocks=None,
                         mpi_manager=None):
    """Calculate anisotropic SFs for total and local-B-perpendicular increments.

    For each sampled pair the local mean magnetic-field direction is defined by
    B_bar = (B1 + B2) / 2. We store the total increment magnitudes |delta v| and
    |delta B|, as well as the projected perpendicular magnitudes |delta v_perp|
    and |delta B_perp|, where the perpendicular direction is relative to B_bar.

    Every quantity is accumulated in three 1D bin families:
    - l: all valid pairs binned by |l|
    - l_prll: near-parallel pairs binned by |l . B_hat|
    - l_perp: near-perpendicular pairs binned by sqrt(|l|^2 - l_prll^2)
    """
    rank = mpi_manager.rank if mpi_manager else 0

    if debug and rank == 0:
        print(f'[DEBUG get_sf_aniso_mhd_mb] vel_var={vel_var}, bcc_var={bcc_var}, weights={weights}')

    if theta_prll_max is None:
        theta_prll_max = np.pi / 18.0
    if theta_perp_min is None:
        theta_perp_min = np.pi / 2.0 - np.pi / 18.0
    cos_theta_prll_min = np.cos(theta_prll_max)
    cos_theta_perp_max = np.cos(theta_perp_min)

    if nbins is None:
        nbins = max(nbins_prll, nbins_perp)

    if rank == 0 and debug:
        print(f'  theta_prll_max={np.degrees(theta_prll_max):.1f}deg  '
              f'theta_perp_min={np.degrees(theta_perp_min):.1f}deg')

    if hasattr(ad, 'has_full_data') and not ad.has_full_data:
        nmbs = ad.local_mb_end - ad.local_mb_start
        mb_offset = ad.local_mb_start
    else:
        nmbs = ad.n_mbs
        mb_offset = 0

    vel_varlist = [f'{vel_var}x', f'{vel_var}y', f'{vel_var}z']
    bcc_varlist = [f'{bcc_var}1', f'{bcc_var}2', f'{bcc_var}3']
    all_varlist = vel_varlist + bcc_varlist

    Lx = ad.x1max - ad.x1min
    Ly = ad.x2max - ad.x2min
    Lz = ad.x3max - ad.x3min

    first_mb = ad.mb_geometry[0, :]
    dx = (first_mb[1] - first_mb[0]) / ad.nx1
    dy = (first_mb[3] - first_mb[2]) / ad.nx2
    dz = (first_mb[5] - first_mb[4]) / ad.nx3
    rmax = np.sqrt(0.25 * (Lx**2 + Ly**2 + Lz**2))
    rmin_local = np.min(np.array([dx, dy, dz]))
    if rmin_local <= 0:
        rmin_local = np.min(np.array([dx, dy, dz])[np.array([dx, dy, dz]) > 0])
    rmin = mpi_manager.allreduce(rmin_local, op='min') if mpi_manager else rmin_local

    periodic_x = 'periodic' in ad._header['mesh']['ix1_bc']
    periodic_y = 'periodic' in ad._header['mesh']['ix2_bc']
    periodic_z = 'periodic' in ad._header['mesh']['ix3_bc']

    weights_list = weights if isinstance(weights, list) else [weights]
    n_weights = len(weights_list)

    if max_order > 10:
        print('Warning: max_order > 10, capping at 10')
        max_order = 10

    eligible_blocks = ad.n_mbs
    if xyz is not None:
        eligible_blocks = sum(
            not is_block_outside_xyz(ad.mb_geometry, b, xyz)
            for b in range(ad.n_mbs)
        )
    eligible_blocks = max(1, int(eligible_blocks))

    n_samples1 = max(1, int(np.sqrt(npairs) / eligible_blocks))
    if n_samples1 < nsamples_block_min:
        print(f'Warning: n_samples1 = {n_samples1} is less than {nsamples_block_min}, consider increasing npairs')
    n_points_per_block = ad.nx1 * ad.nx2 * ad.nx3
    n_samples1 = min(n_samples1, n_points_per_block)

    block_centers = np.zeros((nmbs, 3))
    local_geom = ad.mb_geometry[:nmbs, :]
    block_centers[:, 0] = 0.5 * (local_geom[:, 0] + local_geom[:, 1])
    block_centers[:, 1] = 0.5 * (local_geom[:, 2] + local_geom[:, 3])
    block_centers[:, 2] = 0.5 * (local_geom[:, 4] + local_geom[:, 5])

    if log_bin_flag:
        lbins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins + 1)
        lprll_bins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins_prll + 1)
        lperp_bins = xp.logspace(xp.log10(rmin), xp.log10(rmax), nbins_perp + 1)
        l_ = xp.sqrt(lbins[1:] * lbins[:-1])
        l_prll_ = xp.sqrt(lprll_bins[1:] * lprll_bins[:-1])
        l_perp_ = xp.sqrt(lperp_bins[1:] * lperp_bins[:-1])
    else:
        lbins = xp.linspace(0, rmax, nbins + 1)
        lprll_bins = xp.linspace(0, rmax, nbins_prll + 1)
        lperp_bins = xp.linspace(0, rmax, nbins_perp + 1)
        l_ = 0.5 * (lbins[1:] + lbins[:-1])
        l_prll_ = 0.5 * (lprll_bins[1:] + lprll_bins[:-1])
        l_perp_ = 0.5 * (lperp_bins[1:] + lperp_bins[:-1])

    sf_accum = _allocate_aniso_mhd_accumulators(
        n_weights, max_order, nbins, nbins_prll, nbins_perp, array_module=xp)
    pair_workspace = _allocate_aniso_mhd_pair_workspace(
        max_order, nbins, nbins_prll, nbins_perp, array_module=xp)

    if simultaneous_blocks is not None:
        blocks_per_batch = min(simultaneous_blocks, nmbs)
    else:
        blocks_per_batch = determine_blocks_per_batch(nmbs, len(all_varlist) * 3, n_weights, n_points_per_block)
    if blocks_per_batch == 0:
        raise ValueError('blocks_per_batch is zero -- reduce simultaneous_blocks or increase available memory')

    processed_pairs = set()
    block_cache = {}
    use_mpi = mpi_manager is not None

    if use_mpi:
        if rank == 0:
            print(f'MPI Mode: anisotropic MHD SF across {ad.n_mbs} global blocks')
        mpi_manager.comm.Barrier()

        total_global = ad.n_mbs
        my_pairs = _build_rank_assigned_pairs(total_global, rank, mpi_manager.size)

        _process_mpi_pairs_aniso_mhd(
            ad, my_pairs, block_cache,
            weights_list, vel_varlist, bcc_varlist,
            sf_accum,
            rmax, rmin, Lx, Ly, Lz,
            lbins, lprll_bins, lperp_bins,
            nbins, nbins_prll, nbins_perp, max_order,
            cos_theta_prll_min, cos_theta_perp_max,
            n_samples1, n_points_per_block, nsamples_block_min,
            periodic_x, periodic_y, periodic_z,
            use_nearfield_bias=use_nearfield_bias,
            same_block_pair_boost=same_block_pair_boost,
            face_block_pair_boost=face_block_pair_boost,
            edge_block_pair_boost=edge_block_pair_boost,
            corner_block_pair_boost=corner_block_pair_boost,
            sparse_data=sparse_data, xyzlim=xyz,
            mpi_manager=mpi_manager, mb_offset=mb_offset, debug=debug,
            pair_workspace=pair_workspace)
    else:
        num_batches = (nmbs + blocks_per_batch - 1) // blocks_per_batch
        batches = [list(range(i * blocks_per_batch, min((i + 1) * blocks_per_batch, nmbs)))
                   for i in range(num_batches)]

        _process_batch_pairs_aniso_mhd(
            ad, batches, block_cache, processed_pairs,
            weights_list, vel_varlist, bcc_varlist,
            sf_accum,
            block_centers, rmax, rmin, Lx, Ly, Lz,
            lbins, lprll_bins, lperp_bins,
            nbins, nbins_prll, nbins_perp, max_order,
            cos_theta_prll_min, cos_theta_perp_max,
            n_samples1, n_points_per_block, nsamples_block_min,
            periodic_x, periodic_y, periodic_z,
            use_nearfield_bias=use_nearfield_bias,
            same_block_pair_boost=same_block_pair_boost,
            face_block_pair_boost=face_block_pair_boost,
            edge_block_pair_boost=edge_block_pair_boost,
            corner_block_pair_boost=corner_block_pair_boost,
            sparse_data=sparse_data, xyzlim=xyz, debug=debug,
            use_mpi=False, mpi_manager=None, mb_offset=0,
            pair_workspace=pair_workspace)

        total_pairs = (eligible_blocks * (eligible_blocks + 1)) // 2
        if len(processed_pairs) < total_pairs:
            print(f'WARNING: Only {len(processed_pairs)} of {total_pairs} aniso_mhd block pairs processed')

    if cupy_enabled:
        xp.cuda.Stream.null.synchronize()

    if not isinstance(weights, list):
        sf_accum = _squeeze_aniso_mhd_accumulators(sf_accum)

    if mpi_manager is not None:
        sf_accum = _reduce_aniso_mhd_accumulators(sf_accum, mpi_manager)

    return {
        'l': l_,
        'l_prll': l_prll_,
        'l_perp': l_perp_,
        'vel': sf_accum['vel'],
        'vel_perp': sf_accum['vel_perp'],
        'bcc': sf_accum['bcc'],
        'bcc_perp': sf_accum['bcc_perp'],
        'counts': sf_accum['counts'],
        'weights': sf_accum['weights'],
    }


def set_sf_aniso_mhd(ad, vel_var='vel', bcc_var='bcc', varsuf='',
                      redo=False, use_mpi=False, debug=False, **kwargs):
    """Calculate and store anisotropic MHD structure functions.

    The stored products retain the original total-increment keys
    ``vel_var + varsuf`` and ``bcc_var + varsuf`` and add companion keys
    ``vel_var + '_perp' + varsuf`` and ``bcc_var + '_perp' + varsuf`` for the
    increment magnitudes perpendicular to the local mean magnetic field.

    Each stored entry contains:
    - ``l``, ``sf``, ``num``, ``wsum`` for all valid pairs binned by |l|
    - ``l_prll``, ``sf_prll``, ``num_prll``, ``wsum_prll`` for near-parallel pairs
    - ``l_perp``, ``sf_perp``, ``num_perp``, ``wsum_perp`` for near-perpendicular pairs
    """
    mpi_manager = None
    if use_mpi and MPI_AVAILABLE:
        mpi_manager = MPIManager()
        rank = mpi_manager.rank
    else:
        rank = 0

    if not hasattr(ad, 'sf'):
        ad.sf = {}

    vel_key = vel_var + varsuf
    vel_perp_key = vel_var + '_perp' + varsuf
    bcc_key = bcc_var + varsuf
    bcc_perp_key = bcc_var + '_perp' + varsuf

    done_keys = [vel_key, vel_perp_key, bcc_key, bcc_perp_key]
    if not redo and all(key in ad.sf for key in done_keys):
        if rank == 0 and debug:
            print(f'[DEBUG set_sf_aniso_mhd] {done_keys} already exist; skipping (redo=False)')
        return ad.sf

    if rank == 0 and debug:
        print(f'[DEBUG set_sf_aniso_mhd] Computing anisotropic MHD SFs: {done_keys}')

    try:
        result = get_sf_aniso_mhd_mb(
            ad,
            vel_var=vel_var,
            bcc_var=bcc_var,
            mpi_manager=mpi_manager,
            debug=debug,
            **kwargs)

        l_np = asnumpy(result['l'])
        l_prll_np = asnumpy(result['l_prll'])
        l_perp_np = asnumpy(result['l_perp'])
        num_np = asnumpy(result['counts']['num']).squeeze()
        num_prll_np = asnumpy(result['counts']['num_prll']).squeeze()
        num_perp_np = asnumpy(result['counts']['num_perp']).squeeze()
        wsum_np = asnumpy(result['weights']['wsum']).squeeze()
        wsum_prll_np = asnumpy(result['weights']['wsum_prll']).squeeze()
        wsum_perp_np = asnumpy(result['weights']['wsum_perp']).squeeze()

        def _build_entry(field_result):
            return {
                'l': l_np,
                'l_prll': l_prll_np,
                'l_perp': l_perp_np,
                'sf': asnumpy(field_result['sf']).squeeze(),
                'sf_prll': asnumpy(field_result['sf_prll']).squeeze(),
                'sf_perp': asnumpy(field_result['sf_perp']).squeeze(),
                'num': num_np,
                'num_prll': num_prll_np,
                'num_perp': num_perp_np,
                'wsum': wsum_np,
                'wsum_prll': wsum_prll_np,
                'wsum_perp': wsum_perp_np,
            }

        ad.sf[vel_key] = _build_entry(result['vel'])
        ad.sf[vel_perp_key] = _build_entry(result['vel_perp'])
        ad.sf[bcc_key] = _build_entry(result['bcc'])
        ad.sf[bcc_perp_key] = _build_entry(result['bcc_perp'])

        if rank == 0 and debug:
            print(f'[DEBUG set_sf_aniso_mhd] Stored {done_keys}; '
                  f'sf shape = {ad.sf[vel_key]["sf"].shape}, '
                  f'sf_prll shape = {ad.sf[vel_key]["sf_prll"].shape}, '
                  f'sf_perp shape = {ad.sf[vel_key]["sf_perp"].shape}')

    except Exception as e:
        if rank == 0:
            print(f'[ERROR set_sf_aniso_mhd] {e}')
            import traceback
            traceback.print_exc()
        raise

    return ad.sf
