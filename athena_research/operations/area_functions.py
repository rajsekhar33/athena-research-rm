"""
GPU-accelerated isosurface area calculation for scalar-field interfaces
(e.g. an iso-temperature surface).

Implements the Lewiner marching-cubes algorithm (identical to the C++ AthenaK
UserHistOutput `area1`), compiled once as a CuPy GPU kernel from the AthenaK
C++ source files, with a scikit-image CPU fallback. set_area() is the public
entry point (cached via ad.area_mc / ad.save(), MPI + multi-GPU aware) and
routes each requested step size to one of:
  - calc_area_mb_based: sharp marching cubes, one meshblock at a time,
    restricted to the z-levels bracketing the interface. Only handles step
    sizes that fit inside and evenly divide a single meshblock.
  - calc_area_global_coarse_sampled_offsets: for step sizes larger than a
    meshblock, samples a coarse grid over the full domain instead.
  - calc_area_smoothed_indicator_mc(_meshblocks): a Gaussian-smoothed phase
    indicator run through marching cubes at level 0.5, over the full domain.
  - calc_areas_all_steps: the original whole-domain slab method, used when
    set_area is called with use_mb_based=False.
"""
import os, re, warnings
import numpy as np

from ..core.base import xp, asnumpy, cupy_enabled

# ── MPI (optional) ───────────────────────────────────────────────────────────
try:
    from ..backends.mpi_utils import MPIManager
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False

try:
    import h5py
    _H5PY = True
except ImportError:
    _H5PY = False

try:
    import cupyx.scipy.ndimage as _cupyx_ndimage
    _CUPYX_NDIMAGE = True
except ImportError:
    _cupyx_ndimage = None
    _CUPYX_NDIMAGE = False

import skimage.measure
from ..utils.meshblock_utils import find_neighbour_blocks

try:
    from scipy import ndimage as _scipy_ndimage
    _SCIPY_NDIMAGE = True
except ImportError:
    _scipy_ndimage = None
    _SCIPY_NDIMAGE = False

# ── Paths to AthenaK C++ source ─────────────────────────────────────────────
# Prefer the local copy bundled with this package (marching_cubes_src/, see its
# README.md for provenance); fall back to a live athenak-RM checkout so this
# still works unchanged for anyone already relying on that path.
_LOCAL_UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'marching_cubes_src')
_ATHENAK_UTILS = os.path.expanduser('~/athenak-RM/src/utils')
_LOCAL_LUT_HPP = os.path.join(_LOCAL_UTILS, 'mc_luts.hpp')
_LOCAL_MC_HPP  = os.path.join(_LOCAL_UTILS, 'marching_cubes.hpp')
_LUT_HPP = _LOCAL_LUT_HPP if os.path.exists(_LOCAL_LUT_HPP) else os.path.join(_ATHENAK_UTILS, 'mc_luts.hpp')
_MC_HPP  = _LOCAL_MC_HPP if os.path.exists(_LOCAL_MC_HPP) else os.path.join(_ATHENAK_UTILS, 'marching_cubes.hpp')

# Cached compiled module (contains both kernels)
_MC_MODULE = None

# ── CUDA kernel additions ────────────────────────────────────────────────────
# Physically-scaled add_triangle: identical to the C++ version but the 6 lines
# that build edge vectors v1-v3 / w1-w3 are multiplied by (dx, dy, dz) so the
# resulting area is already in physical (code) units.
_ADD_TRIANGLE_PHYS = r"""
// Physically-scaled marching-cubes triangle area (dx/dy/dz are the physical
// extents of the current cube in x/y/z).
__device__ inline
double add_triangle_phys(const char *trig, char n, Cube cube,
                         double dx, double dy, double dz) {
  double xt[3],yt[3],zt[3];
  double area = 0.0;
  double xv, yv, zv;
  int nv;

  for( int t = 0 ; t < 3*n ; t++ ){
    switch( trig[t] ) {
      case  0:
        xt[ t % 3 ] = 1./(1-cube.v1/cube.v0);
        yt[ t % 3 ] = 0.0;
        zt[ t % 3 ] = 0.0;
        break;
      case  1:
        xt[ t % 3 ] = 1.0;
        yt[ t % 3 ] = 1./(1-cube.v2/cube.v1);
        zt[ t % 3 ] = 0.0;
        break;
      case  2:
        xt[ t % 3 ] = 1./(1-cube.v2/cube.v3);
        yt[ t % 3 ] = 1.0;
        zt[ t % 3 ] = 0.0;
        break;
      case  3:
        xt[ t % 3 ] = 0.0;
        yt[ t % 3 ] = 1./(1-cube.v3/cube.v0);
        zt[ t % 3 ] = 0.0;
        break;
      case  4:
        xt[ t % 3 ] = 1./(1-cube.v5/cube.v4);
        yt[ t % 3 ] = 0.0;
        zt[ t % 3 ] = 1.0;
        break;
      case  5:
        xt[ t % 3 ] = 1.0;
        yt[ t % 3 ] = 1./(1-cube.v6/cube.v5);
        zt[ t % 3 ] = 1.0;
        break;
      case  6:
        xt[ t % 3 ] = 1./(1-cube.v6/cube.v7);
        yt[ t % 3 ] = 1.0;
        zt[ t % 3 ] = 1.0;
        break;
      case  7:
        xt[ t % 3 ] = 0.0;
        yt[ t % 3 ] = 1./(1-cube.v7/cube.v4);
        zt[ t % 3 ] = 1.0;
        break;
      case  8:
        xt[ t % 3 ] = 0.0;
        yt[ t % 3 ] = 0.0;
        zt[ t % 3 ] = 1./(1-cube.v4/cube.v0);
        break;
      case  9:
        xt[ t % 3 ] = 1.0;
        yt[ t % 3 ] = 0.0;
        zt[ t % 3 ] = 1./(1-cube.v5/cube.v1);
        break;
      case 10:
        xt[ t % 3 ] = 1.0;
        yt[ t % 3 ] = 1.0;
        zt[ t % 3 ] = 1./(1-cube.v6/cube.v2);
        break;
      case 11:
        xt[ t % 3 ] = 0.0;
        yt[ t % 3 ] = 1.0;
        zt[ t % 3 ] = 1./(1-cube.v7/cube.v3);
        break;
      case 12:
        xv=0.0; yv=0.0; zv=0.0; nv=0;
        if (cube.v1*cube.v0<0){xv+=1./(1-cube.v1/cube.v0);yv+=0.0;zv+=0.0;nv++;}
        if (cube.v2*cube.v1<0){xv+=1.0;yv+=1./(1-cube.v2/cube.v1);zv+=0.0;nv++;}
        if (cube.v2*cube.v3<0){xv+=1./(1-cube.v2/cube.v3);yv+=1.0;zv+=0.0;nv++;}
        if (cube.v3*cube.v0<0){xv+=0.0;yv+=1./(1-cube.v3/cube.v0);zv+=0.0;nv++;}
        if (cube.v5*cube.v4<0){xv+=1./(1-cube.v5/cube.v4);yv+=0.0;zv+=1.0;nv++;}
        if (cube.v6*cube.v5<0){xv+=1.0;yv+=1./(1-cube.v6/cube.v5);zv+=1.0;nv++;}
        if (cube.v6*cube.v7<0){xv+=1./(1-cube.v6/cube.v7);yv+=1.0;zv+=1.0;nv++;}
        if (cube.v7*cube.v4<0){xv+=0.0;yv+=1./(1-cube.v7/cube.v4);zv+=1.0;nv++;}
        if (cube.v4*cube.v0<0){xv+=0.0;yv+=0.0;zv+=1./(1-cube.v4/cube.v0);nv++;}
        if (cube.v5*cube.v1<0){xv+=1.0;yv+=0.0;zv+=1./(1-cube.v5/cube.v1);nv++;}
        if (cube.v6*cube.v2<0){xv+=1.0;yv+=1.0;zv+=1./(1-cube.v6/cube.v2);nv++;}
        if (cube.v7*cube.v3<0){xv+=0.0;yv+=1.0;zv+=1./(1-cube.v7/cube.v3);nv++;}
        xt[ t % 3 ] = xv/nv;
        yt[ t % 3 ] = yv/nv;
        zt[ t % 3 ] = zv/nv;
        break;
      default:
        break;
    }

    if( t%3 == 2 ) {
      // Physical edge vectors: scale fractional [0,1] coords by cube physical size
      double v1 = (xt[0] - xt[1]) * dx;
      double v2 = (yt[0] - yt[1]) * dy;
      double v3 = (zt[0] - zt[1]) * dz;
      double w1 = (xt[0] - xt[2]) * dx;
      double w2 = (yt[0] - yt[2]) * dy;
      double w3 = (zt[0] - zt[2]) * dz;
      double u1 = v2*w3 - v3*w2;
      double u2 = v3*w1 - v1*w3;
      double u3 = v1*w2 - v2*w1;
      area += 0.5*sqrt(u1*u1 + u2*u2 + u3*u3);
    }
  }
  return area;
}
"""

# process_cube_phys: identical to process_cube() in marching_cubes.hpp except
# every add_triangle(tiling..., N, cube) call becomes
# add_triangle_phys(tiling..., N, cube, dx, dy, dz).
# Generated by a simple sed-like replacement below at build time.

_KERNEL_MAIN = r"""
extern "C" __global__ void mc_area_kernel(
    const double* __restrict__ data,  // (Nz, Ny, Nx) log10(T) + ghost cols
    double* __restrict__ area_out,    // accumulator (single double)
    int Nz, int Ny, int Nx,
    int s,                            // step size (cube side in cells)
    double iso,                       // log10(T_peak)
    double dz_cell,                   // physical z-spacing per cell
    double dy_cell,                   // physical y-spacing per cell
    double dx_cell                    // physical x-spacing per cell
) {
    // One thread per cube.
    // data has ghost columns at x=Nx-1 and y=Ny-1 (periodic wrapping).
    // z has no ghost (outflow BC); the last z-cube clamps K+s to Nz-1.
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int k = (int)(blockIdx.z * blockDim.z + threadIdx.z);

    int ni = (Nx - 1) / s;
    int nj = (Ny - 1) / s;
    int nk = (Nz - 1) / s;

    if (i >= ni || j >= nj || k >= nk) return;

    int I  = i * s,  J  = j * s,  K  = k * s;
    int Ks = K + s;
    if (Ks >= Nz) Ks = Nz - 1;

    #define D(kk,jj,ii) (data[(kk)*Ny*Nx + (jj)*Nx + (ii)] - iso)

    Cube cube(
        D(K,  J,  I  ), D(K,  J,  I+s),
        D(K,  J+s,I+s), D(K,  J+s,I  ),
        D(Ks, J,  I  ), D(Ks, J,  I+s),
        D(Ks, J+s,I+s), D(Ks, J+s,I  )
    );

    #undef D

    // Physical extent of this cube
    double cube_dx = (double)s * dx_cell;
    double cube_dy = (double)s * dy_cell;
    double cube_dz = (double)s * dz_cell;

    double area = process_cube_phys(cube, cube_dx, cube_dy, cube_dz);
    atomicAdd(area_out, area);
}
"""

# Meshblock-based kernel: processes one pre-padded meshblock.
# The padded array has shape (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost) where the
# ghost region provides neighbor data needed for cubes at meshblock edges.
# Only cubes with their lower corner inside [0,nx3) x [0,nx2) x [0,nx1) are
# processed, so no double-counting when looping over all meshblocks.
_KERNEL_MAIN_MB = r"""
extern "C" __global__ void mc_area_kernel_mb(
    const double* __restrict__ data,  // (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost)
    double* __restrict__ area_out,    // per-block accumulator (single double)
    int nx3, int nx2, int nx1,        // actual meshblock dims (no ghost)
    int s_ghost,                      // ghost thickness (padded dim = nx* + s_ghost)
    int s,                            // step size (cube side length in cells)
    double iso,
    double dz_cell, double dy_cell, double dx_cell
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int k = (int)(blockIdx.z * blockDim.z + threadIdx.z);

    int ni = nx1 / s;
    int nj = nx2 / s;
    int nk = nx3 / s;

    if (i >= ni || j >= nj || k >= nk) return;

    int I = i * s,  J = j * s,  K = k * s;

    int Nx_pad = nx1 + s_ghost;
    int Ny_pad = nx2 + s_ghost;

    #define D_MB(kk,jj,ii) (data[(kk)*Ny_pad*Nx_pad + (jj)*Nx_pad + (ii)] - iso)

    Cube cube(
        D_MB(K,   J,   I  ), D_MB(K,   J,   I+s),
        D_MB(K,   J+s, I+s), D_MB(K,   J+s, I  ),
        D_MB(K+s, J,   I  ), D_MB(K+s, J,   I+s),
        D_MB(K+s, J+s, I+s), D_MB(K+s, J+s, I  )
    );

    #undef D_MB

    double cube_dx = (double)s * dx_cell;
    double cube_dy = (double)s * dy_cell;
    double cube_dz = (double)s * dz_cell;

    double area = process_cube_phys(cube, cube_dx, cube_dy, cube_dz);
    atomicAdd(area_out, area);
}
"""

# ── Source preprocessing ─────────────────────────────────────────────────────

def _preprocess_cpp_for_cuda(src: str) -> str:
    """Minimal C++ → CUDA device-code preprocessing."""
    # Replace Kokkos macro with CUDA device marker
    src = src.replace('KOKKOS_INLINE_FUNCTION\n', '__device__ inline\n')
    src = src.replace('KOKKOS_INLINE_FUNCTION ', '__device__ inline ')
    # Replace Athena 'Real' with 'double'
    src = re.sub(r'\bReal\b', 'double', src)
    # Remove ALL #include lines (both "local" and <system>).
    # CUDA device code does not need standard C++ headers and NVRTC/nvcc
    # cannot always find them via the CuPy compilation path.
    src = re.sub(r'#include\s*(?:"[^"]+"|<[^>]+>)\s*\n', '', src)
    # Remove header guards
    src = re.sub(r'#ifndef\s+\w+\s*\n#define\s+\w+\s*\n', '', src)
    src = re.sub(r'#endif\s*//[^\n]*', '', src)
    # Mark Cube default constructor as __device__ too
    src = src.replace('\nCube::Cube():', '\n__device__ Cube::Cube():')
    src = src.replace('  Cube();\n', '  __device__ Cube();\n')
    return src


def _build_process_cube_phys(mc_src_preprocessed: str) -> str:
    """
    Extract process_cube() from the preprocessed marching_cubes source and
    produce process_cube_phys() that calls add_triangle_phys() with physical
    cube dimensions.
    """
    # Locate process_cube
    marker = 'double process_cube('
    idx = mc_src_preprocessed.find(marker)
    if idx == -1:
        raise RuntimeError("Could not locate process_cube in MC source.")

    # Extract the full function body by matching braces
    depth, inside, end = 0, False, idx
    for pos, ch in enumerate(mc_src_preprocessed[idx:]):
        if ch == '{':
            depth += 1
            inside = True
        elif ch == '}' and inside:
            depth -= 1
            if depth == 0:
                end = idx + pos + 1
                break

    pc_body = mc_src_preprocessed[idx:end]

    # Rename and add dx/dy/dz parameters
    pc_phys = pc_body.replace(
        'double process_cube(Cube cube)',
        'double process_cube_phys(Cube cube, double dx, double dy, double dz)'
    )

    # Redirect all add_triangle(..., cube) → add_triangle_phys(..., cube, dx, dy, dz)
    # IMPORTANT: only patch lines that contain add_triangle_phys (i.e. the renamed
    # add_triangle calls), NOT test_face() or test_interior() which also end with
    # ', cube)' but must NOT receive the extra dx/dy/dz arguments.
    pc_phys = pc_phys.replace('add_triangle(', 'add_triangle_phys(')
    patched_lines = []
    for line in pc_phys.split('\n'):
        if 'add_triangle_phys(' in line:
            line = re.sub(r',\s*cube\)', ', cube, dx, dy, dz)', line)
        patched_lines.append(line)
    pc_phys = '\n'.join(patched_lines)

    return '\n// ── process_cube_phys (physical area) ──\n' + pc_phys + '\n'


def _build_cuda_source() -> str:
    """
    Build the complete CUDA source for the marching-cubes area kernel by
    reading and preprocessing the AthenaK C++ source files.
    """
    if not (os.path.exists(_LUT_HPP) and os.path.exists(_MC_HPP)):
        raise FileNotFoundError(
            f"AthenaK source not found at {_ATHENAK_UTILS}. "
            "Set the _ATHENAK_UTILS path in area_functions.py if the source "
            "is elsewhere."
        )

    with open(_LUT_HPP) as f:
        lut_src = _preprocess_cpp_for_cuda(f.read())
    with open(_MC_HPP) as f:
        mc_src = _preprocess_cpp_for_cuda(f.read())

    # Derive a physically-scaled process_cube_phys from mc_src; the original
    # process_cube it's derived from stays in mc_src too (unused, harmless).
    proc_cube_phys = _build_process_cube_phys(mc_src)

    cuda_src = (
        # Define float.h constants stripped along with the include.
        # FLT_EPSILON is used by test_face / test_interior in the MC source.
        '#ifndef FLT_EPSILON\n'
        '#define FLT_EPSILON 1.192093e-07f\n'
        '#endif\n'
        + lut_src
        + mc_src           # Cube struct, test_face, test_interior, add_triangle, process_cube
        + _ADD_TRIANGLE_PHYS
        + proc_cube_phys
        + _KERNEL_MAIN     # slab-based kernel
        + _KERNEL_MAIN_MB  # meshblock-based kernel
    )
    return cuda_src


# ── Kernel compilation (lazy, cached) ────────────────────────────────────────

def _get_module():
    """Compile and cache the CuPy RawModule containing both MC kernels."""
    global _MC_MODULE
    if _MC_MODULE is not None:
        return _MC_MODULE
    if not cupy_enabled:
        return None
    try:
        cuda_src = _build_cuda_source()
        # CuPy 13+ compiles lazily: actual NVRTC/nvcc compilation is deferred
        # until the first get_function() call, so we probe one function here
        # to force compilation and catch errors in this try/except block.
        mod = xp.RawModule(code=cuda_src, options=('--std=c++14', '--maxrregcount=64'))
        mod.get_function('mc_area_kernel')      # force compile
        mod.get_function('mc_area_kernel_mb')   # force compile
        _MC_MODULE = mod
        print('GPU marching-cubes kernels compiled successfully.')
        return _MC_MODULE
    except Exception as exc:
        warnings.warn(
            f'GPU kernel compilation failed ({_cupy_compile_hint(exc)}). '
            'Falling back to scikit-image on CPU.',
            stacklevel=3
        )
        return None


def _get_kernel():
    """Return the slab kernel (mc_area_kernel)."""
    mod = _get_module()
    return mod.get_function('mc_area_kernel') if mod is not None else None


def _get_kernel_mb():
    """Return the meshblock kernel (mc_area_kernel_mb)."""
    mod = _get_module()
    return mod.get_function('mc_area_kernel_mb') if mod is not None else None


# ── Core computation helpers ──────────────────────────────────────────────────

def _aug_array(log_slab: np.ndarray) -> np.ndarray:
    """
    Add one periodic ghost column in x1 (axis=-1) and x2 (axis=-2).
    Output shape: (Nz, Ny+1, Nx+1).
    """
    Nz, Ny, Nx = log_slab.shape
    aug = np.empty((Nz, Ny + 1, Nx + 1), dtype=np.float64)
    aug[:, :-1, :-1] = log_slab
    aug[:, :-1, -1]  = log_slab[:, :,  0]   # x1 periodic
    aug[:, -1,  :-1] = log_slab[:,  0, :]   # x2 periodic
    aug[:, -1,  -1]  = log_slab[:,  0,  0]  # corner
    return aug


def _calc_area_skimage(aug: np.ndarray, iso: float, ad, step_size: int) -> float:
    """Scikit-image (CPU) marching-cubes on the augmented slab."""
    Nz, Ny, Nx = aug.shape
    # aug has ghost: Ny = Ny_real+1, Nx = Nx_real+1
    Ny_real = Ny - 1
    Nx_real = Nx - 1
    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / Ny_real
    dx = (ad.x1max - ad.x1min) / Nx_real
    verts, faces, _, _ = skimage.measure.marching_cubes(
        aug, level=iso, step_size=step_size,
        spacing=(dz, dy, dx), allow_degenerate=False
    )
    return float(skimage.measure.mesh_surface_area(verts, faces))


def _calc_area_gpu(aug: np.ndarray, iso: float, ad, step_size: int) -> float:
    """GPU marching-cubes on the augmented slab (requires compiled kernel)."""
    kernel = _get_kernel()
    if kernel is None:
        raise RuntimeError('GPU kernel not available.')

    Nz, Ny, Nx = aug.shape  # aug has ghost cols
    Ny_real = Ny - 1
    Nx_real = Nx - 1
    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / Ny_real
    dx = (ad.x1max - ad.x1min) / Nx_real

    data_gpu   = xp.asarray(aug.astype(np.float64))
    area_gpu   = xp.zeros(1, dtype=np.float64)

    s  = step_size
    ni = (Nx - 1) // s
    nj = (Ny - 1) // s
    nk = (Nz - 1) // s

    block = (4, 4, 4)
    grid  = (
        (ni + 3) // 4,
        (nj + 3) // 4,
        (nk + 3) // 4,
    )
    kernel(
        grid, block,
        (data_gpu, area_gpu,
         np.int32(Nz), np.int32(Ny), np.int32(Nx),
         np.int32(s),
         np.float64(iso),
         np.float64(dz), np.float64(dy), np.float64(dx))
    )
    return float(asnumpy(area_gpu)[0])


# ── Interface-slab helpers (shared with notebook) ────────────────────────────

def find_interface_z_levels(ad, T_peak, n_buffer=1):
    """
    Return the x3min values of meshblock z-levels that bracket the T=T_peak
    isosurface, plus n_buffer extra levels on each side.
    """
    mb_geo   = ad.mb_geometry
    gamma    = ad.gamma
    eint_raw = ad.data_raw['eint']
    dens_raw = ad.data_raw['dens']

    z3_unique  = np.sort(np.unique(mb_geo[:, 4]))
    log_T_by_z = np.empty(len(z3_unique))
    for iz, z3min in enumerate(z3_unique):
        idx_list = np.where(mb_geo[:, 4] == z3min)[0]
        log_T_by_z[iz] = np.mean([
            np.log10(float((gamma - 1.0) * eint_raw[m].mean() / dens_raw[m].mean()))
            for m in idx_list
        ])

    iz_cross = int(np.argmin(np.abs(log_T_by_z - np.log10(T_peak))))
    iz_low   = max(0, iz_cross - n_buffer)
    iz_high  = min(len(z3_unique) - 1, iz_cross + n_buffer)
    return z3_unique[iz_low : iz_high + 1], iz_cross


def _build_slab(ad, T_peak, z_selected):
    """
    Reconstruct the temperature slab for the selected meshblock z-levels.
    Returns (slab, log_slab, iso) where slab has shape (Nz_slab, Nx2, Nx1).
    """
    mb_geo   = ad.mb_geometry
    gamma    = ad.gamma
    eint_raw = ad.data_raw['eint']
    dens_raw = ad.data_raw['dens']
    _, nx3_mb, nx2_mb, nx1_mb = eint_raw.shape

    Nz_slab = len(z_selected) * nx3_mb
    Nx1, Nx2 = ad.Nx1, ad.Nx2

    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x2_vals = np.sort(np.unique(mb_geo[:, 2]))
    slab    = np.empty((Nz_slab, Nx2, Nx1), dtype=np.float32)

    for m in np.where(np.isin(mb_geo[:, 4], z_selected))[0]:
        i1 = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
        i2 = int(np.searchsorted(x2_vals, mb_geo[m, 2]))
        iz = int(np.searchsorted(z_selected, mb_geo[m, 4]))
        slab[iz*nx3_mb:(iz+1)*nx3_mb,
             i2*nx2_mb:(i2+1)*nx2_mb,
             i1*nx1_mb:(i1+1)*nx1_mb] = (gamma - 1.0) * eint_raw[m] / dens_raw[m]

    iso      = float(np.log10(T_peak))
    log_slab = np.log10(slab.astype(np.float64))

    if iso < log_slab.min() or iso > log_slab.max():
        warnings.warn(
            f'T_peak ({T_peak:.4g}) outside slab temperature range '
            f'[{slab.min():.4g}, {slab.max():.4g}]. '
            'Try increasing n_mb_buffer.',
            stacklevel=4
        )
    return slab, log_slab, iso


# ── Block-by-block helpers ────────────────────────────────────────────────────

def _find_mc_neighbors(ad):
    """
    Build +x, +y, +z neighbor lookup for every meshblock.

    Uses find_neighbour_blocks which correctly handles periodic BCs (x1/x2)
    and outflow BCs (x3) via the mb_logical array.

    Returns
    -------
    list of (ix_plus, iy_plus, iz_plus) per meshblock.
        ix_plus / iy_plus : int  (always valid; x1/x2 are periodic)
        iz_plus           : int or None  (None when at the +z domain boundary)
    """
    neighbors = []
    for m in range(ad.n_mbs):
        x_plus = find_neighbour_blocks(ad, m, 'x+')
        y_plus = find_neighbour_blocks(ad, m, 'y+')
        z_plus = find_neighbour_blocks(ad, m, 'z+')

        ix = x_plus[0] if x_plus else m   # x1 is periodic → always has neighbor
        iy = y_plus[0] if y_plus else m   # x2 is periodic → always has neighbor
        iz = z_plus[0] if z_plus else None # None at +z domain boundary (outflow)

        neighbors.append((ix, iy, iz))
    return neighbors


def _build_all_log_temp(ad, T_peak):
    """
    Compute log10(T) for every meshblock.

    Returns
    -------
    np.ndarray, shape (n_mb, nx3, nx2, nx1), dtype float64
    """
    gamma = ad.gamma
    eint = np.asarray(ad.data_raw['eint'], dtype=np.float64)
    dens = np.asarray(ad.data_raw['dens'], dtype=np.float64)
    return np.log10((gamma - 1.0) * eint / dens)


def _build_padded_block(log_temp_all, m, neighbors, s_ghost, nx3, nx2, nx1):
    """
    Assemble a padded array for meshblock m.

    The padded array has shape (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost).
    The extra s_ghost layers in each + direction are filled from neighbour
    meshblock data so the MC kernel can access all 8 corners of every cube
    whose lower-left-back corner lies within [0, nx3) × [0, nx2) × [0, nx1).

    Boundary conditions
    -------------------
    x1 (+x) : periodic — neighbour always exists.
    x2 (+y) : periodic — neighbour always exists.
    x3 (+z) : outflow  — if no neighbour (at domain top), the last z-layer is
                         replicated into the ghost region (constant extrapolation).
    """
    ix_plus, iy_plus, iz_plus = neighbors[m]

    pad = np.empty((nx3 + s_ghost, nx2 + s_ghost, nx1 + s_ghost), dtype=np.float64)

    # ── core block ────────────────────────────────────────────────────────────
    pad[:nx3, :nx2, :nx1] = log_temp_all[m]

    # ── +x ghost: first s_ghost x-columns of x+ neighbour ───────────────────
    pad[:nx3, :nx2, nx1:] = log_temp_all[ix_plus][:nx3, :nx2, :s_ghost]

    # ── +y ghost and +xy corner ───────────────────────────────────────────────
    pad[:nx3, nx2:, :nx1] = log_temp_all[iy_plus][:nx3, :s_ghost, :nx1]
    ixy = neighbors[iy_plus][0]              # +x of the +y neighbour
    pad[:nx3, nx2:, nx1:] = log_temp_all[ixy][:nx3, :s_ghost, :s_ghost]

    # ── +z ghost and its three corners ───────────────────────────────────────
    if iz_plus is not None:
        pad[nx3:, :nx2, :nx1] = log_temp_all[iz_plus][:s_ghost, :nx2, :nx1]
        iz_x  = neighbors[iz_plus][0]        # +x of the +z neighbour
        iz_y  = neighbors[iz_plus][1]        # +y of the +z neighbour
        iz_xy = neighbors[iz_y][0]           # +x of (+y of +z) = (+x,+y,+z) neighbour
        pad[nx3:, :nx2, nx1:] = log_temp_all[iz_x ][:s_ghost, :nx2, :s_ghost]
        pad[nx3:, nx2:, :nx1] = log_temp_all[iz_y ][:s_ghost, :s_ghost, :nx1]
        pad[nx3:, nx2:, nx1:] = log_temp_all[iz_xy][:s_ghost, :s_ghost, :s_ghost]
    else:
        # Outflow BC at +z domain boundary: replicate last real z-layer
        pad[nx3:, :nx2, :nx1] = pad[nx3 - 1:nx3, :nx2, :nx1]
        pad[nx3:, :nx2, nx1:] = pad[nx3 - 1:nx3, :nx2, nx1:]
        pad[nx3:, nx2:, :]    = pad[nx3 - 1:nx3, nx2:, :]

    return pad


def _calc_area_gpu_mb(padded_gpu, iso, nx3, nx2, nx1, s_ghost, s, dz, dy, dx):
    """
    Run mc_area_kernel_mb on one GPU-resident padded block for step_size=s.

    Parameters
    ----------
    padded_gpu : xp.ndarray  shape (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost)
    iso        : float       log10(T_peak)
    nx3, nx2, nx1 : int      actual (unpadded) meshblock dimensions
    s_ghost    : int         ghost layer thickness used when building padded array
    s          : int         step size for this kernel call
    dz, dy, dx : float       physical cell spacings

    Returns
    -------
    float  Area contribution from this meshblock.
    """
    kernel = _get_kernel_mb()
    if kernel is None:
        raise RuntimeError('GPU kernel not available.')

    area_gpu = xp.zeros(1, dtype=np.float64)

    ni = nx1 // s
    nj = nx2 // s
    nk = nx3 // s

    block = (8, 8, 8)
    grid  = ((ni + 7) // 8, (nj + 7) // 8, (nk + 7) // 8)

    kernel(
        grid, block,
        (padded_gpu, area_gpu,
         np.int32(nx3), np.int32(nx2), np.int32(nx1),
         np.int32(s_ghost), np.int32(s),
         np.float64(iso),
         np.float64(dz), np.float64(dy), np.float64(dx))
    )
    return float(asnumpy(area_gpu)[0])


# ── Public API ───────────────────────────────────────────────────────────────

STEP_SIZES = [1, 2, 4, 8, 16, 32, 64]
AREA_METHOD_MARCHING_CUBES = 'marching_cubes'
AREA_METHOD_SMOOTHED_INDICATOR_MC = 'smoothed_indicator_mc'
AREA_METHOD_VERSION = {
    AREA_METHOD_MARCHING_CUBES: 1,
    AREA_METHOD_SMOOTHED_INDICATOR_MC: 1,
}

def calc_area_one_step(ad, T_peak, step_size=1, n_mb_buffer=1, use_gpu=True):
    """
    Compute the T=T_peak isosurface area for a single step size.

    Uses the interface-slab optimisation to avoid reconstructing the full
    Nx3 × Nx2 × Nx1 array.  Matches the C++ `area1` (step_size=1) exactly.

    Parameters
    ----------
    ad : AthenaData
    T_peak : float
    step_size : int
    n_mb_buffer : int
        Extra meshblock z-levels on each side of the interface slab.
    use_gpu : bool
        Use GPU (CuPy) kernel if available; fall back to scikit-image.

    Returns
    -------
    float  Physical surface area.
    """
    z_sel, iz_cross = find_interface_z_levels(ad, T_peak, n_buffer=n_mb_buffer)
    _, log_slab, iso = _build_slab(ad, T_peak, z_sel)
    Nz_slab = log_slab.shape[0]

    aug = _aug_array(log_slab)

    if use_gpu and cupy_enabled and _get_kernel() is not None:
        return _calc_area_gpu(aug, iso, ad, step_size)
    else:
        return _calc_area_skimage(aug, iso, ad, step_size)


def calc_areas_all_steps(ad, T_peak, step_sizes=None, n_mb_buffer=1, use_gpu=True,
                         verbose=True):
    """
    Compute isosurface area for all requested step sizes on a single slab
    (slab is built only once).

    Parameters
    ----------
    ad : AthenaData
    T_peak : float
    step_sizes : list of int, optional
        Defaults to [1, 2, 4, 8, 16, 32, 64].
    n_mb_buffer : int
    use_gpu : bool
    verbose : bool

    Returns
    -------
    step_sizes : np.ndarray
    areas : np.ndarray
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    z_sel, iz_cross = find_interface_z_levels(ad, T_peak, n_buffer=n_mb_buffer)
    _, log_slab, iso = _build_slab(ad, T_peak, z_sel)
    Nz_slab = log_slab.shape[0]

    if verbose:
        print(f'  Slab: {Nz_slab}/{ad.Nx3} z-cells '
              f'({100.*Nz_slab/ad.Nx3:.1f}%,  interface idx={iz_cross})')

    aug = _aug_array(log_slab)

    use_kernel = use_gpu and cupy_enabled and _get_kernel() is not None
    if verbose:
        print(f'  Backend: {"GPU (CuPy kernel)" if use_kernel else "CPU (scikit-image)"}')

    areas = np.empty(len(step_sizes))
    for k, s in enumerate(step_sizes):
        if use_kernel:
            areas[k] = _calc_area_gpu(aug, iso, ad, s)
        else:
            areas[k] = _calc_area_skimage(aug, iso, ad, s)
        if verbose:
            print(f'    step={s:3d}  area={areas[k]:.6g}')

    return np.asarray(step_sizes, dtype=int), areas


def calc_area_mb_based(ad, T_peak, step_sizes=None, n_mb_buffer=1, use_gpu=True,
                       verbose=True, mpi_manager=None):
    """
    Compute isosurface area block by block, keeping GPU memory bounded to one
    padded meshblock at a time.

    This meshblock-based method can only process step sizes that:
      1. fit inside one meshblock, and
      2. evenly divide the meshblock dimensions.

    For example, if the meshblock size is 128^3, then step_size=256 is not
    valid for this method, even if the full domain is 512^3. Such large step
    sizes should be computed with calc_area_global_coarse_sampled_offsets().

    Parameters
    ----------
    ad          : AthenaData  data_raw must contain 'eint' and 'dens'
    T_peak      : float
    step_sizes  : list of int, optional
    n_mb_buffer : int
    use_gpu     : bool
    verbose     : bool
    mpi_manager : MPIManager or None

    Returns
    -------
    step_sizes : np.ndarray
    areas      : np.ndarray
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    step_sizes = [int(s) for s in step_sizes]

    rank = mpi_manager.rank if mpi_manager is not None else 0

    n_mb, nx3, nx2, nx1 = ad.data_raw['eint'].shape

    # Filter to step sizes that fit inside one meshblock and evenly divide it.
    valid_steps = []

    for s in step_sizes:
        s = int(s)

        fits_in_block = (s <= nx1 and s <= nx2 and s <= nx3)
        divides_block = (nx1 % s == 0 and nx2 % s == 0 and nx3 % s == 0)

        if fits_in_block and divides_block:
            valid_steps.append(s)
        else:
            if rank == 0:
                warnings.warn(
                    f'step_size={s} is not valid for meshblock dims '
                    f'({nx3}×{nx2}×{nx1}); skipping in meshblock method. '
                    'This step size should be computed with '
                    'calc_area_global_coarse_sampled_offsets().',
                    stacklevel=2,
                )


    if not valid_steps:
        raise ValueError(
            'No valid step sizes for the current meshblock dimensions in '
            'calc_area_mb_based().'
        )

    s_ghost = max(valid_steps)

    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1
    iso = float(np.log10(T_peak))

    use_kernel = use_gpu and cupy_enabled and _get_module() is not None

    if not use_kernel:
        # CPU fallback: use the slab-based approach.
        if rank == 0 and verbose:
            print('  Block-by-block: GPU unavailable; falling back to slab method.')

        return calc_areas_all_steps(
            ad,
            T_peak,
            step_sizes=valid_steps,
            n_mb_buffer=n_mb_buffer,
            use_gpu=False,
            verbose=(verbose and rank == 0),
        )

    # Identify the selected z-levels.
    z_selected, iz_cross = find_interface_z_levels(
        ad,
        T_peak,
        n_buffer=n_mb_buffer,
    )
    slab_indices = np.where(np.isin(ad.mb_geometry[:, 4], z_selected))[0]
    n_slab = len(slab_indices)

    # MPI: each rank processes every mpi_size-th block.
    if mpi_manager is not None and mpi_manager.size > 1:
        my_slab = slab_indices[rank::mpi_manager.size]
    else:
        my_slab = slab_indices

    if rank == 0 and verbose:
        print(
            f'  Block-by-block GPU: {n_slab}/{n_mb} slab meshblocks  '
            f'(z-level {iz_cross}, buffer={n_mb_buffer}, '
            f'{s_ghost}-cell ghost layer)'
        )
        print(f'  Meshblock-valid step sizes: {valid_steps}')

    # Precompute log10(T) for all meshblocks.
    log_temp_all = _build_all_log_temp(ad, T_peak)

    # Build neighbor lookup.
    neighbors = _find_mc_neighbors(ad)

    # Running totals per valid step size.
    totals_gpu = [xp.zeros(1, dtype=np.float64) for _ in valid_steps]

    kernel = _get_kernel_mb()

    n_active = 0

    for m in my_slab:
        block_log = log_temp_all[m]

        # Quick skip: isosurface cannot pass through this block.
        if block_log.max() <= iso or block_log.min() >= iso:
            continue

        n_active += 1

        # Build padded CPU array, transfer once to GPU for all valid steps.
        padded = _build_padded_block(
            log_temp_all,
            m,
            neighbors,
            s_ghost,
            nx3,
            nx2,
            nx1,
        )
        padded_gpu = xp.asarray(padded)

        for k, s in enumerate(valid_steps):
            ni = nx1 // s
            nj = nx2 // s
            nk = nx3 // s

            blk = (4, 4, 4)
            grd = (
                (ni + 3) // 4,
                (nj + 3) // 4,
                (nk + 3) // 4,
            )

            kernel(
                grd,
                blk,
                (
                    padded_gpu,
                    totals_gpu[k],
                    np.int32(nx3),
                    np.int32(nx2),
                    np.int32(nx1),
                    np.int32(s_ghost),
                    np.int32(s),
                    np.float64(iso),
                    np.float64(dz),
                    np.float64(dy),
                    np.float64(dx),
                ),
            )

        if rank == 0 and verbose and (n_active % max(1, len(my_slab) // 10) == 0):
            print(
                f'    block {n_active:4d}/{len(my_slab)} active ...',
                end='\r',
            )

    areas = np.array([float(asnumpy(t)[0]) for t in totals_gpu], dtype=float)

    # MPI allreduce.
    if mpi_manager is not None and mpi_manager.size > 1:
        areas = mpi_manager.comm.allreduce(areas, op=mpi_manager.MPI.SUM)

    if rank == 0 and verbose:
        print(
            f'    Processed {n_active}/{len(my_slab)} active slab blocks '
            f'(isosurface present)'
        )
        for s, a in zip(valid_steps, areas):
            print(f'    step={s:3d}  area={a:.6g}')

    return np.asarray(valid_steps, dtype=int), areas



# ── h5data caching ────────────────────────────────────────────────────────────

def _build_meshblock_lookup(ad):
    """
    Build a lookup from integer meshblock coordinates to meshblock index.

    Assumes ad.data_raw['eint'] has shape:
        (n_mbs, nx3, nx2, nx1)

    Returns
    -------
    lookup : dict
        Maps (i1_block, i2_block, i3_block) to meshblock id.
    """
    mb_geo = ad.mb_geometry

    x1_starts = np.sort(np.unique(mb_geo[:, 0]))
    x2_starts = np.sort(np.unique(mb_geo[:, 2]))
    x3_starts = np.sort(np.unique(mb_geo[:, 4]))

    lookup = {}

    for m in range(ad.n_mbs):
        i1b = int(np.searchsorted(x1_starts, mb_geo[m, 0]))
        i2b = int(np.searchsorted(x2_starts, mb_geo[m, 2]))
        i3b = int(np.searchsorted(x3_starts, mb_geo[m, 4]))
        lookup[(i1b, i2b, i3b)] = m

    return lookup


def _sample_log_temp_global_index(ad, lookup, gk, gj, gi):
    """
    Sample log10(T) at one global cell index using BCs from ad._header.

    x1, x2, x3 periodicity is determined from:
        ad._header['mesh']['ix1_bc']
        ad._header['mesh']['ix2_bc']
        ad._header['mesh']['ix3_bc']
    """
    gamma = ad.gamma
    eint = ad.data_raw['eint']
    dens = ad.data_raw['dens']

    _, nx3, nx2, nx1 = eint.shape

    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)

    gi = _apply_bc_index(gi, ad.Nx1, periodic_x)
    gj = _apply_bc_index(gj, ad.Nx2, periodic_y)
    gk = _apply_bc_index(gk, ad.Nx3, periodic_z)

    i1b = gi // nx1
    i2b = gj // nx2
    i3b = gk // nx3

    li = gi % nx1
    lj = gj % nx2
    lk = gk % nx3

    m = lookup[(i1b, i2b, i3b)]

    T = (gamma - 1.0) * float(eint[m, lk, lj, li]) / float(dens[m, lk, lj, li])
    return float(np.log10(T))


def _get_periodic_flags(ad):
    """Read periodic boundary conditions from the Athena header."""
    mesh_header = getattr(ad, '_header', {}).get('mesh', {})

    def is_periodic_pair(axis):
        inner = str(mesh_header.get(f'ix{axis}_bc', '')).strip().lower()
        outer = str(mesh_header.get(f'ox{axis}_bc', '')).strip().lower()
        inner_periodic = 'periodic' in inner
        outer_periodic = 'periodic' in outer
        if inner_periodic != outer_periodic:
            warnings.warn(
                f'Mismatched x{axis} boundary conditions: '
                f'ix{axis}_bc={inner!r}, ox{axis}_bc={outer!r}. '
                'Treating the axis as periodic because one side is periodic.',
                stacklevel=3,
            )
        return inner_periodic or outer_periodic

    periodic_x = is_periodic_pair(1)
    periodic_y = is_periodic_pair(2)
    periodic_z = is_periodic_pair(3)

    return periodic_x, periodic_y, periodic_z


def _apply_bc_index(idx, n, periodic):
    """Apply Athena boundary condition to one integer index."""
    idx = int(idx)
    n = int(n)

    if periodic:
        return idx % n

    return max(0, min(idx, n - 1))


def _scipy_mode_tuple_for_mesh(ad):
    """Return scipy.ndimage modes in array order (x3, x2, x1)."""
    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)
    return (
        'wrap' if periodic_z else 'nearest',
        'wrap' if periodic_y else 'nearest',
        'wrap' if periodic_x else 'nearest',
    )


def _build_global_phase_indicator(ad, T_peak, phase='cold', dtype=np.float32):
    """
    Reconstruct a full-domain phase indicator on the native grid.

    The returned array has shape (Nx3, Nx2, Nx1). Values are 1 inside the
    selected phase and 0 outside. For area of the interface, cold and hot
    indicators are complements and should give the same 0.5 isosurface after
    smoothing.
    """
    phase = str(phase).lower()
    if phase not in ('cold', 'hot'):
        raise ValueError(f'Unknown phase={phase!r}. Use "cold" or "hot".')

    mb_geo = ad.mb_geometry
    gamma = ad.gamma
    eint_raw = ad.data_raw['eint']
    dens_raw = ad.data_raw['dens']
    _, nx3_mb, nx2_mb, nx1_mb = eint_raw.shape

    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x2_vals = np.sort(np.unique(mb_geo[:, 2]))
    x3_vals = np.sort(np.unique(mb_geo[:, 4]))

    indicator = np.empty((ad.Nx3, ad.Nx2, ad.Nx1), dtype=dtype)

    for m in range(ad.n_mbs):
        i1 = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
        i2 = int(np.searchsorted(x2_vals, mb_geo[m, 2]))
        i3 = int(np.searchsorted(x3_vals, mb_geo[m, 4]))

        temp = (gamma - 1.0) * eint_raw[m] / dens_raw[m]
        if phase == 'cold':
            block = temp <= T_peak
        else:
            block = temp >= T_peak

        indicator[
            i3*nx3_mb:(i3+1)*nx3_mb,
            i2*nx2_mb:(i2+1)*nx2_mb,
            i1*nx1_mb:(i1+1)*nx1_mb,
        ] = block.astype(dtype, copy=False)

    return indicator


def _build_all_phase_indicator(ad, T_peak, phase='cold'):
    """Build a bool phase mask in the native meshblock ordering."""
    phase = str(phase).lower()
    if phase not in ('cold', 'hot'):
        raise ValueError(f'Unknown phase={phase!r}. Use "cold" or "hot".')

    temp = (ad.gamma - 1.0) * ad.data_raw['eint'] / ad.data_raw['dens']
    if phase == 'cold':
        return np.asarray(temp <= T_peak, dtype=bool)
    return np.asarray(temp >= T_peak, dtype=bool)


def _axis_indices_with_bc(start, stop, n, periodic):
    """Return integer cell indices in [start, stop), after applying BCs."""
    idx = np.arange(int(start), int(stop), dtype=np.int64)
    if periodic:
        return idx % int(n)
    return np.clip(idx, 0, int(n) - 1)


def _copy_phase_region_from_blocks(indicator_all, ad, lookup,
                                   k_start, k_stop,
                                   j_start, j_stop,
                                   i_start, i_stop):
    """Copy an arbitrary global-index region from meshblock-ordered masks."""
    _, nx3, nx2, nx1 = indicator_all.shape
    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)

    gk = _axis_indices_with_bc(k_start, k_stop, ad.Nx3, periodic_z)
    gj = _axis_indices_with_bc(j_start, j_stop, ad.Nx2, periodic_y)
    gi = _axis_indices_with_bc(i_start, i_stop, ad.Nx1, periodic_x)

    bk = gk // nx3
    bj = gj // nx2
    bi = gi // nx1
    lk = gk % nx3
    lj = gj % nx2
    li = gi % nx1

    out = np.empty((len(gk), len(gj), len(gi)), dtype=np.float32)

    for ubk in np.unique(bk):
        kpos = np.where(bk == ubk)[0]
        for ubj in np.unique(bj):
            jpos = np.where(bj == ubj)[0]
            for ubi in np.unique(bi):
                ipos = np.where(bi == ubi)[0]
                m = lookup[(int(ubi), int(ubj), int(ubk))]
                out[np.ix_(kpos, jpos, ipos)] = indicator_all[m][
                    np.ix_(lk[kpos], lj[jpos], li[ipos])
                ]

    return out


def _smooth_phase_indicator(indicator, sigma_cells, mode):
    """Smooth the phase indicator using scipy.ndimage with fixed dtype output."""
    if not _SCIPY_NDIMAGE:
        raise ImportError(
            'scipy.ndimage is required for smoothed-indicator area estimates.'
        )

    return _scipy_ndimage.gaussian_filter(
        indicator,
        sigma=float(sigma_cells),
        mode=mode,
        output=np.float32,
    )


def _cupy_compile_hint(exc):
    """Return an actionable CUDA/CuPy environment hint for compile failures."""
    msg = str(exc)
    if 'invalid value for --gpu-architecture' not in msg:
        return msg

    return (
        f'{msg}\n'
        'CuPy/NVRTC rejected the selected GPU architecture. This usually means '
        'the active CUDA toolkit is older than the GPU architecture requested '
        'by CuPy, or CUDA_PATH/CUDA_HOME/LD_LIBRARY_PATH point to inconsistent '
        'CUDA installations. Use a CUDA toolkit compatible with the allocated '
        'GPU and make sure CUDA_PATH, CUDA_HOME, and LD_LIBRARY_PATH refer to '
        'the same CUDA installation.'
    )


def calc_area_smoothed_indicator_mc(ad, T_peak, step_sizes=None, phase='cold',
                                    sigma_factor=1.0, verbose=True,
                                    compare_scipy=False, scan_axis='auto'):
    """
    Compute scale-dependent interface area with smoothed-indicator marching cubes.

    For each requested step size s, this method builds the phase indicator
    chi = 1(T <= T_peak) by default, smooths chi with Gaussian sigma
    sigma_factor * s cells, and runs marching cubes on chi_smooth = 0.5.

    scan_axis is recorded for future streaming implementations. The current
    full-grid implementation is scan-order independent.
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    step_sizes = [int(s) for s in step_sizes]
    if any(s <= 0 for s in step_sizes):
        raise ValueError(f'All step sizes must be positive: {step_sizes}')

    if sigma_factor <= 0.0:
        raise ValueError('sigma_factor must be positive for smoothed MC.')

    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1
    spacing = (dz, dy, dx)
    mode = _scipy_mode_tuple_for_mesh(ad)

    if verbose:
        print(
            '  Smoothed-indicator MC: '
            f'phase={phase}, sigma_factor={sigma_factor:g}, '
            f'scan_axis={scan_axis}, mode={mode}'
        )

    indicator = _build_global_phase_indicator(ad, T_peak, phase=phase)

    if compare_scipy:
        indicator_ref = _build_global_phase_indicator(ad, T_peak, phase=phase)
        if not np.array_equal(indicator, indicator_ref):
            raise RuntimeError('Internal phase-indicator reconstruction check failed.')

    out_areas = []

    for s in step_sizes:
        sigma_cells = float(sigma_factor) * float(s)
        smoothed = _smooth_phase_indicator(indicator, sigma_cells, mode)
        vmin = float(np.nanmin(smoothed))
        vmax = float(np.nanmax(smoothed))

        if not (vmin <= 0.5 <= vmax):
            raise RuntimeError(
                f'Smoothed indicator does not bracket level=0.5 for step={s}: '
                f'range=[{vmin:.6g}, {vmax:.6g}]'
            )

        aug = _aug_array(smoothed)
        area = _calc_area_skimage(aug, 0.5, ad, 1)
        out_areas.append(area)

        if verbose:
            print(
                f'    smoothed MC step={s:3d}, '
                f'sigma={sigma_cells:.4g} cells, area={area:.6g}'
            )

    return np.asarray(step_sizes, dtype=int), np.asarray(out_areas, dtype=float)


def calc_area_smoothed_indicator_mc_meshblocks(
    ad,
    T_peak,
    step_sizes=None,
    phase='cold',
    sigma_factor=1.0,
    truncate=4.0,
    use_gpu=True,
    verbose=True,
    mpi_manager=None,
    scan_axis='auto',
    max_pad_gb=16.0,
):
    """
    Smoothed-indicator marching cubes with bounded memory.

    Each meshblock is padded by a Gaussian halo, smoothed locally, cropped to
    the meshblock core plus one positive ghost cell, and passed to marching
    cubes. Only cubes whose lower-left corner belongs to the core meshblock are
    counted, so summing blocks tiles the domain without overlap.
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    step_sizes = [int(s) for s in step_sizes]
    if any(s <= 0 for s in step_sizes):
        raise ValueError(f'All step sizes must be positive: {step_sizes}')
    if sigma_factor <= 0.0:
        raise ValueError('sigma_factor must be positive for smoothed MC.')

    rank = mpi_manager.rank if mpi_manager is not None else 0

    n_mb, nx3, nx2, nx1 = ad.data_raw['eint'].shape
    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1
    spacing = (dz, dy, dx)

    if mpi_manager is not None and mpi_manager.size > 1:
        my_blocks = np.arange(rank, n_mb, mpi_manager.size, dtype=int)
    else:
        my_blocks = np.arange(n_mb, dtype=int)

    use_kernel = (
        use_gpu and cupy_enabled and _CUPYX_NDIMAGE and _get_kernel_mb() is not None
    )
    if not use_kernel and not _SCIPY_NDIMAGE:
        raise ImportError(
            'scipy.ndimage is required for CPU fallback when GPU smoothing is unavailable.'
        )

    if rank == 0 and verbose:
        print(
            '  Smoothed-indicator meshblock MC: '
            f'phase={phase}, sigma_factor={sigma_factor:g}, '
            f'truncate={truncate:g}, scan_axis={scan_axis}, '
            f'n_blocks={n_mb}, '
            f'backend={"GPU cupyx+CuPy MC" if use_kernel else "CPU scipy+skimage"}'
        )

    mb_geo = ad.mb_geometry
    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x2_vals = np.sort(np.unique(mb_geo[:, 2]))
    x3_vals = np.sort(np.unique(mb_geo[:, 4]))

    totals = np.zeros(len(step_sizes), dtype=np.float64)
    step_work = []

    for s in step_sizes:
        sigma_cells = float(sigma_factor) * float(s)
        halo = int(np.ceil(float(truncate) * sigma_cells)) + 1
        pad_shape = (nx3 + 1 + 2*halo, nx2 + 1 + 2*halo, nx1 + 1 + 2*halo)
        core_shape = (nx3 + 1, nx2 + 1, nx1 + 1)
        pad_gb = float(np.prod(pad_shape)) * np.dtype(np.float32).itemsize / 1024.0**3
        core64_gb = float(np.prod(core_shape)) * np.dtype(np.float64).itemsize / 1024.0**3
        work_gb = pad_gb if not use_kernel else 2.0 * pad_gb + core64_gb
        if work_gb > float(max_pad_gb):
            raise MemoryError(
                f'Smoothed-indicator MC step={s} would require a padded '
                f'meshblock of shape {pad_shape} ({work_gb:.2f} GiB working memory). '
                f'This exceeds max_pad_gb={max_pad_gb:.2f}; use smaller '
                'steps, reduce smoothing_sigma_factor, or increase max_pad_gb '
                'deliberately.'
            )
        step_work.append((s, sigma_cells, halo, pad_shape, work_gb))

    lookup = _build_meshblock_lookup(ad)
    indicator_all = _build_all_phase_indicator(ad, T_peak, phase=phase)

    for step_index, (s, sigma_cells, halo, pad_shape, work_gb) in enumerate(step_work):
        if rank == 0 and verbose:
            print(
                f'    step={s:3d}: sigma={sigma_cells:.4g} cells, '
                f'halo={halo}, padded block={pad_shape} '
                f'({work_gb:.2f} GiB working memory)'
            )
        local_total = 0.0
        active_blocks = 0

        for n_done, m in enumerate(my_blocks, start=1):
            i1b = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
            i2b = int(np.searchsorted(x2_vals, mb_geo[m, 2]))
            i3b = int(np.searchsorted(x3_vals, mb_geo[m, 4]))

            i0 = i1b * nx1
            j0 = i2b * nx2
            k0 = i3b * nx3

            pad = _copy_phase_region_from_blocks(
                indicator_all,
                ad,
                lookup,
                k0 - halo,
                k0 + nx3 + 1 + halo,
                j0 - halo,
                j0 + nx2 + 1 + halo,
                i0 - halo,
                i0 + nx1 + 1 + halo,
            )

            if float(pad.min()) > 0.5 or float(pad.max()) < 0.5:
                continue

            if use_kernel:
                pad_gpu = xp.asarray(pad, dtype=xp.float32)
                try:
                    smoothed_gpu = _cupyx_ndimage.gaussian_filter(
                        pad_gpu,
                        sigma=sigma_cells,
                        mode='nearest',
                        truncate=float(truncate),
                    )
                except Exception as exc:
                    del pad_gpu
                    raise RuntimeError(_cupy_compile_hint(exc)) from exc
                core_gpu = smoothed_gpu[
                    halo:halo + nx3 + 1,
                    halo:halo + nx2 + 1,
                    halo:halo + nx1 + 1,
                ]

                if float(asnumpy(core_gpu.min())) > 0.5 or float(asnumpy(core_gpu.max())) < 0.5:
                    del pad_gpu, smoothed_gpu, core_gpu
                    continue

                active_blocks += 1
                core_gpu64 = xp.ascontiguousarray(core_gpu, dtype=xp.float64)
                area = _calc_area_gpu_mb(
                    core_gpu64,
                    0.5,
                    nx3,
                    nx2,
                    nx1,
                    1,
                    1,
                    dz,
                    dy,
                    dx,
                )
                del pad_gpu, smoothed_gpu, core_gpu, core_gpu64
            else:
                smoothed = _scipy_ndimage.gaussian_filter(
                    pad,
                    sigma=sigma_cells,
                    mode='nearest',
                    truncate=float(truncate),
                    output=np.float32,
                )
                core = smoothed[
                    halo:halo + nx3 + 1,
                    halo:halo + nx2 + 1,
                    halo:halo + nx1 + 1,
                ]
                core = np.ascontiguousarray(core)

                if float(core.min()) > 0.5 or float(core.max()) < 0.5:
                    continue

                active_blocks += 1
                verts, faces, _, _ = skimage.measure.marching_cubes(
                    core,
                    level=0.5,
                    spacing=spacing,
                    allow_degenerate=False,
                )
                area = float(skimage.measure.mesh_surface_area(verts, faces))

            local_total += area

            if rank == 0 and verbose and n_done % max(1, len(my_blocks) // 10) == 0:
                print(
                    f'    step={s:3d} block {n_done}/{len(my_blocks)} '
                    f'active={active_blocks}',
                    end='\r',
                )

        if mpi_manager is not None and mpi_manager.size > 1:
            local_total = mpi_manager.comm.allreduce(
                local_total,
                op=mpi_manager.MPI.SUM,
            )
            active_blocks = mpi_manager.comm.allreduce(
                active_blocks,
                op=mpi_manager.MPI.SUM,
            )

        totals[step_index] = float(local_total)

        if rank == 0 and verbose:
            print(
                f'    smoothed MC step={s:3d}, '
                f'sigma={sigma_cells:.4g} cells, halo={halo}, '
                f'active_blocks={active_blocks}, area={local_total:.6g}'
            )

    return np.asarray(step_sizes, dtype=int), totals


def _build_coarse_sampled_grid_offset(ad, T_peak, step_size,
                                      offset_k=0, offset_j=0, offset_i=0,
                                      verbose=True):
    """
    Build a coarse log10(T) grid using BCs from ad._header.

    Periodic directions are closed with offset + N.
    Non-periodic directions are closed at N - 1.

    For your usual setup:
        x1 periodic
        x2 periodic
        x3 non-periodic
    this gives periodic closure in x/y and clamped closure in z.
    """
    s = int(step_size)
    ok = int(offset_k)
    oj = int(offset_j)
    oi = int(offset_i)

    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)

    lookup = _build_meshblock_lookup(ad)

    def make_indices(n, offset, periodic, axis_name):
        n = int(n)
        offset = int(offset)

        if periodic:
            idx = list(range(offset, offset + n, s))
            if idx[-1] != offset + n:
                idx.append(offset + n)
        else:
            # For non-periodic directions, avoid phase-shifting unless you
            # deliberately want that. The endpoint is N - 1.
            offset = max(0, min(offset, n - 1))
            idx = list(range(offset, n, s))
            if not idx:
                idx = [0]
            if idx[-1] != n - 1:
                idx.append(n - 1)

            # Warn if the last cell is not the same physical size as the others.
            if verbose and ((n - 1 - offset) % s != 0):
                print(
                    f'  WARNING: nonuniform final coarse interval along {axis_name}: '
                    f'N={n}, offset={offset}, step={s}. '
                    'skimage spacing assumes uniform intervals.'
                )

        return idx

    z_idx = make_indices(ad.Nx3, ok, periodic_z, 'x3')
    y_idx = make_indices(ad.Nx2, oj, periodic_y, 'x2')
    x_idx = make_indices(ad.Nx1, oi, periodic_x, 'x1')

    coarse = np.empty((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float64)

    for kk, gk in enumerate(z_idx):
        for jj, gj in enumerate(y_idx):
            for ii, gi in enumerate(x_idx):
                coarse[kk, jj, ii] = _sample_log_temp_global_index(
                    ad, lookup, gk, gj, gi
                )

    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1

    spacing = (s * dz, s * dy, s * dx)

    if verbose:
        iso = float(np.log10(T_peak))
        print(
            f'  Global coarse grid step={s}, '
            f'offset=({ok},{oj},{oi}), '
            f'periodic=(x3={periodic_z}, x2={periodic_y}, x1={periodic_x}), '
            f'shape={coarse.shape}, '
            f'logT range=[{coarse.min():.6g}, {coarse.max():.6g}], '
            f'iso={iso:.6g}'
        )

    return coarse, spacing


def calc_area_global_coarse_sampled_offsets(ad, T_peak, step_sizes=None,
                                            offsets='half', verbose=True):
    """
    Compute large-step isosurface areas using offset-averaged globally sampled
    coarse grids.

    This function is intended for step sizes that are too large for the
    meshblock-based method, for example:

        step_size=256 with 128^3 meshblocks

    Boundary conditions are read from the Athena header through
    _get_periodic_flags(ad). Offset averaging is applied only along directions
    that are periodic.

    Parameters
    ----------
    ad : AthenaData
        Loaded Athena snapshot.
    T_peak : float
        Temperature level defining the isosurface.
    step_sizes : list[int] or None
        Step sizes to compute. If None, uses STEP_SIZES.
    offsets : {'none', 'half'}
        'none' : use only zero offset.
        'half' : use offsets [0, s//2] along periodic directions.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    step_sizes : np.ndarray
        Computed step sizes.
    areas : np.ndarray
        Offset-averaged areas.
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    step_sizes = [int(s) for s in step_sizes]
    iso = float(np.log10(T_peak))

    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)

    out_steps = []
    out_areas = []

    for s in step_sizes:
        s = int(s)

        if offsets == 'none':
            base_offsets = [0]
        elif offsets == 'half':
            base_offsets = sorted(set([0, s // 2]))
        else:
            raise ValueError(
                f'Unknown offsets={offsets}. Use "none" or "half".'
            )

        # Only offset directions that are periodic according to the header.
        i_offsets = base_offsets if periodic_x else [0]  # x1
        j_offsets = base_offsets if periodic_y else [0]  # x2
        k_offsets = base_offsets if periodic_z else [0]  # x3

        if verbose:
            print(
                f'  Global coarse offset method for step={s}: '
                f'periodic=(x1={periodic_x}, x2={periodic_y}, x3={periodic_z}), '
                f'i_offsets={i_offsets}, '
                f'j_offsets={j_offsets}, '
                f'k_offsets={k_offsets}'
            )

        phase_areas = []
        skipped_phases = 0

        for ok in k_offsets:
            for oj in j_offsets:
                for oi in i_offsets:
                    coarse, spacing = _build_coarse_sampled_grid_offset(
                        ad,
                        T_peak,
                        s,
                        offset_k=ok,
                        offset_j=oj,
                        offset_i=oi,
                        verbose=verbose,
                    )

                    cmin = float(np.nanmin(coarse))
                    cmax = float(np.nanmax(coarse))

                    # If this offset does not bracket the isosurface, skip it.
                    if iso < cmin or iso > cmax:
                        skipped_phases += 1
                        if verbose:
                            print(
                                f'    skipping step={s}, '
                                f'offset=({ok},{oj},{oi}): '
                                f'iso={iso:.6g} not in '
                                f'[{cmin:.6g}, {cmax:.6g}]'
                            )
                        continue

                    try:
                        verts, faces, _, _ = skimage.measure.marching_cubes(
                            coarse,
                            level=iso,
                            spacing=spacing,
                            allow_degenerate=False,
                        )

                        area = float(skimage.measure.mesh_surface_area(verts, faces))

                    except Exception as exc:
                        skipped_phases += 1
                        if verbose:
                            print(
                                f'    skipping step={s}, '
                                f'offset=({ok},{oj},{oi}): '
                                f'marching_cubes failed: {exc}'
                            )
                        continue

                    if np.isfinite(area) and area > 0.0:
                        phase_areas.append(area)

                        if verbose:
                            print(
                                f'    global coarse step={s:3d}, '
                                f'offset=({ok},{oj},{oi}), '
                                f'area={area:.6g}'
                            )
                    else:
                        skipped_phases += 1
                        if verbose:
                            print(
                                f'    skipping step={s}, '
                                f'offset=({ok},{oj},{oi}): '
                                f'bad area={area}'
                            )

        if not phase_areas:
            raise RuntimeError(
                f'All global coarse offset phases failed for step={s}. '
                f'iso={iso:.6g}. '
                'The coarse sampling is probably too sparse or the chosen '
                'step size does not robustly bracket the isosurface.'
            )

        area_mean = float(np.mean(phase_areas))
        area_std = float(np.std(phase_areas))

        if verbose:
            print(
                f'    global coarse offset-averaged step={s:3d}: '
                f'area={area_mean:.6g}, '
                f'std={area_std:.6g}, '
                f'n_used={len(phase_areas)}, '
                f'n_skipped={skipped_phases}'
            )

        out_steps.append(s)
        out_areas.append(area_mean)

    return np.asarray(out_steps, dtype=int), np.asarray(out_areas, dtype=float)


def _area_cache_metadata_matches(area_mc, requested_metadata):
    """Return True when an area_mc cache was made with matching parameters."""
    if not area_mc:
        return False

    method = requested_metadata.get('method')
    cached_method = area_mc.get('method', AREA_METHOD_MARCHING_CUBES)
    cached_method = str(cached_method).lower().replace('-', '_')
    if cached_method != method:
        return False

    if int(area_mc.get('method_version', 1)) != int(requested_metadata['method_version']):
        return False

    for key in ('T_peak', 'smoothing_sigma_factor'):
        if key in requested_metadata:
            try:
                if not np.isclose(float(area_mc.get(key)), float(requested_metadata[key])):
                    return False
            except Exception:
                return False

    for key in ('phase', 'scan_axis'):
        if key in requested_metadata:
            if str(area_mc.get(key, '')).lower() != str(requested_metadata[key]).lower():
                return False

    return True

def set_area(ad, T_peak, h5_path: str, step_sizes=None, redo=False,
             n_mb_buffer=1, use_gpu=True, verbose=True,
             use_mb_based=True, use_mpi=False,
             area_method=AREA_METHOD_MARCHING_CUBES,
             smoothing_sigma_factor=1.0,
             phase='cold', scan_axis='auto',
             compare_scipy=False,
             max_pad_gb=16.0):
    """
    Compute and cache isosurface areas.

    Routing (per requested step size):
      - area_method=smoothed_indicator_mc -> calc_area_smoothed_indicator_mc_meshblocks()
      - use_mb_based=True and the step fits inside one meshblock -> calc_area_mb_based()
      - use_mb_based=True and the step doesn't fit -> calc_area_global_coarse_sampled_offsets()
      - use_mb_based=False -> calc_areas_all_steps()

    Cache behavior:
      - redo=False computes only missing step sizes
      - redo=True recomputes all requested step sizes
    """
    if step_sizes is None:
        step_sizes = STEP_SIZES

    step_sizes = sorted({int(s) for s in step_sizes})
    area_method = str(area_method).lower().replace('-', '_')
    if area_method not in AREA_METHOD_VERSION:
        raise ValueError(
            f'Unknown area_method={area_method!r}. '
            f'Use {sorted(AREA_METHOD_VERSION)}.'
        )

    requested_metadata = {
        'method': area_method,
        'method_version': AREA_METHOD_VERSION[area_method],
        'T_peak': float(T_peak),
    }
    if area_method == AREA_METHOD_SMOOTHED_INDICATOR_MC:
        requested_metadata.update({
            'phase': str(phase).lower(),
            'smoothing_sigma_factor': float(smoothing_sigma_factor),
            'scan_axis': str(scan_axis).lower(),
        })

    # MPI setup
    mpi_manager = None
    if use_mpi and _MPI_AVAILABLE:
        mpi_manager = MPIManager()

    rank = mpi_manager.rank if mpi_manager is not None else 0

    # Existing cache
    existing_area_mc = None if redo else getattr(ad, 'area_mc', None)
    if existing_area_mc and not _area_cache_metadata_matches(
        existing_area_mc,
        requested_metadata,
    ):
        if rank == 0 and verbose:
            print('  Ignoring cached areas because area method metadata differs.')
        existing_area_mc = None

    cached_step_area = {}

    if existing_area_mc:
        cached_steps = [
            int(s)
            for s in np.asarray(existing_area_mc.get('step_sizes', [])).flat
        ]
        cached_areas = np.asarray(
            existing_area_mc.get('areas', []),
            dtype=float,
        ).flat

        for s, a in zip(cached_steps, cached_areas):
            s = int(s)
            a = float(a)

            # Do not trust failed old values.
            if np.isfinite(a) and a > 0.0:
                cached_step_area[s] = a
            else:
                if rank == 0 and verbose:
                    print(f'  Ignoring bad cached area: step={s}, area={a}')

    # Full cache hit
    if not redo and set(step_sizes).issubset(set(cached_step_area.keys())):
        if rank == 0 and verbose:
            print('  Using cached areas from ad.area_mc')
        return ad.area_mc

    # Determine missing/recomputed steps
    if redo:
        missing_steps = step_sizes.copy()
    else:
        missing_steps = [
            s for s in step_sizes
            if s not in cached_step_area
        ]

    if len(missing_steps) == 0:
        if rank == 0 and verbose:
            print('  No missing step sizes to compute.')
        return ad.area_mc

    n_mb, nx3, nx2, nx1 = ad.data_raw['eint'].shape

    if rank == 0 and verbose:
        print(
            f'  Global mesh: ({ad.Nx3}, {ad.Nx2}, {ad.Nx1}); '
            f'meshblocks: n_mb={n_mb}, block=({nx3}, {nx2}, {nx1})'
        )
        print(f'  Requested step sizes: {step_sizes}')
        print(f'  Step sizes to compute: {missing_steps}')

    # Split steps by method
    smooth_steps = []
    mb_steps = []
    global_steps = []
    slab_steps = []

    for s in missing_steps:
        fits_in_block = (s <= nx1 and s <= nx2 and s <= nx3)
        divides_block = (nx1 % s == 0 and nx2 % s == 0 and nx3 % s == 0)

        if area_method == AREA_METHOD_SMOOTHED_INDICATOR_MC:
            smooth_steps.append(s)
        elif use_mb_based and fits_in_block and divides_block:
            mb_steps.append(s)
        elif use_mb_based:
            global_steps.append(s)
        else:
            slab_steps.append(s)

    if rank == 0 and verbose:
        if smooth_steps:
            print(f'  Smoothed-indicator MC step size(s): {smooth_steps}')
        if mb_steps:
            print(f'  Meshblock method step size(s): {mb_steps}')
        if global_steps:
            print(f'  Global coarse method step size(s): {global_steps}')
        if slab_steps:
            print(f'  Slab method step size(s): {slab_steps}')

    computed = {}

    # 0. Smoothed-indicator marching cubes.
    if smooth_steps:
        s_sm, a_sm = calc_area_smoothed_indicator_mc_meshblocks(
            ad,
            T_peak,
            step_sizes=smooth_steps,
            phase=phase,
            sigma_factor=smoothing_sigma_factor,
            use_gpu=use_gpu,
            verbose=(verbose and rank == 0),
            mpi_manager=mpi_manager,
            scan_axis=scan_axis,
            max_pad_gb=max_pad_gb,
        )

        for s, a in zip(s_sm, a_sm):
            computed[int(s)] = float(a)

    # 1. Meshblock method
    if mb_steps:
        if use_gpu and cupy_enabled:
            s_mb, a_mb = calc_area_mb_based(
                ad,
                T_peak,
                step_sizes=mb_steps,
                n_mb_buffer=n_mb_buffer,
                use_gpu=True,
                verbose=(verbose and rank == 0),
                mpi_manager=mpi_manager,
            )
        else:
            s_mb, a_mb = calc_areas_all_steps(
                ad,
                T_peak,
                step_sizes=mb_steps,
                n_mb_buffer=n_mb_buffer,
                use_gpu=use_gpu,
                verbose=(verbose and rank == 0),
            )

        for s, a in zip(s_mb, a_mb):
            computed[int(s)] = float(a)

    # 2. Global coarse method for large steps
    if global_steps:
        if mpi_manager is not None and mpi_manager.size > 1:
            raise RuntimeError(
                'Global coarse sampled large-step method is not MPI-distributed. '
                f'Run without use_mpi for step size(s): {global_steps}'
            )

        s_gl, a_gl = calc_area_global_coarse_sampled_offsets(
            ad,
            T_peak,
            step_sizes=global_steps,
            offsets='half',
            verbose=(verbose and rank == 0),
        )

        for s, a in zip(s_gl, a_gl):
            computed[int(s)] = float(a)

    # 3. Original slab method, only when use_mb_based=False
    if slab_steps:
        s_sl, a_sl = calc_areas_all_steps(
            ad,
            T_peak,
            step_sizes=slab_steps,
            n_mb_buffer=n_mb_buffer,
            use_gpu=use_gpu,
            verbose=(verbose and rank == 0),
        )

        for s, a in zip(s_sl, a_sl):
            computed[int(s)] = float(a)

    # Validate computation
    missing_after_compute = [
        s for s in missing_steps
        if s not in computed
    ]

    if missing_after_compute:
        raise RuntimeError(
            f'Area computation did not return step size(s): '
            f'{missing_after_compute}'
        )

    bad_new = {
        s: a for s, a in computed.items()
        if (not np.isfinite(a)) or a <= 0.0
    }

    if bad_new:
        raise RuntimeError(
            'Newly computed area contains nonpositive or nonfinite values: '
            f'{bad_new}'
        )

    # Merge cache on rank 0
    if rank == 0:
        dx = float((ad.x1max - ad.x1min) / ad.Nx1)

        merged = {}

        # Preserve existing valid cached values
        if _area_cache_metadata_matches(getattr(ad, 'area_mc', None), requested_metadata):
            old_steps = [
                int(s)
                for s in np.asarray(ad.area_mc.get('step_sizes', [])).flat
            ]
            old_areas = np.asarray(
                ad.area_mc.get('areas', []),
                dtype=float,
            ).flat

            for s, a in zip(old_steps, old_areas):
                s = int(s)
                a = float(a)
                if np.isfinite(a) and a > 0.0:
                    merged[s] = a

        # Add or replace newly computed values
        for s, a in computed.items():
            merged[int(s)] = float(a)

        merged_steps = sorted(merged.keys())
        merged_areas = np.array(
            [merged[s] for s in merged_steps],
            dtype=float,
        )

        ad.area_mc = {
            'step_sizes': merged_steps,
            'areas': merged_areas,
            'time': float(ad.time),
            'T_peak': float(T_peak),
            'dx': dx,
            **requested_metadata,
        }

    # Broadcast cache from rank 0
    if mpi_manager is not None and mpi_manager.size > 1:
        ad.area_mc = mpi_manager.broadcast(
            ad.area_mc if rank == 0 else None,
            root=0,
        )

    return ad.area_mc
