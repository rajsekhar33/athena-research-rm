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
// Interior vertex of tiling entry 12 (the mean of the iso crossings on the
// cube edges); same rule as interior_vertex() in marching_cubes.hpp, kept
// here so this string compiles against any copy of that header.  Pass 0 is
// the strict sign change; when every crossing edge touches the iso level
// exactly (three or more corners on the level) it finds no edge and the mean
// would be 0/0, so pass 1 uses process_cube's own classification (v < 0 is
// inside), whose interpolation stays finite on those edges.
__device__ inline bool mc_interior_edge_crosses(double a, double b, int pass) {
  return (pass == 0) ? (a*b < 0) : ((a < 0) != (b < 0));
}

__device__ inline
void mc_interior_vertex(Cube cube, double &x, double &y, double &z) {
  int nv = 0;
  for (int pass = 0; pass < 2 && nv == 0; pass++) {
    x = 0.0; y = 0.0; z = 0.0;
    if (mc_interior_edge_crosses(cube.v1, cube.v0, pass)) {x += 1./(1-cube.v1/cube.v0); nv++;}
    if (mc_interior_edge_crosses(cube.v2, cube.v1, pass)) {x += 1.0; y += 1./(1-cube.v2/cube.v1); nv++;}
    if (mc_interior_edge_crosses(cube.v2, cube.v3, pass)) {x += 1./(1-cube.v2/cube.v3); y += 1.0; nv++;}
    if (mc_interior_edge_crosses(cube.v3, cube.v0, pass)) {y += 1./(1-cube.v3/cube.v0); nv++;}
    if (mc_interior_edge_crosses(cube.v5, cube.v4, pass)) {x += 1./(1-cube.v5/cube.v4); z += 1.0; nv++;}
    if (mc_interior_edge_crosses(cube.v6, cube.v5, pass)) {x += 1.0; y += 1./(1-cube.v6/cube.v5); z += 1.0; nv++;}
    if (mc_interior_edge_crosses(cube.v6, cube.v7, pass)) {x += 1./(1-cube.v6/cube.v7); y += 1.0; z += 1.0; nv++;}
    if (mc_interior_edge_crosses(cube.v7, cube.v4, pass)) {y += 1./(1-cube.v7/cube.v4); z += 1.0; nv++;}
    if (mc_interior_edge_crosses(cube.v4, cube.v0, pass)) {z += 1./(1-cube.v4/cube.v0); nv++;}
    if (mc_interior_edge_crosses(cube.v5, cube.v1, pass)) {x += 1.0; z += 1./(1-cube.v5/cube.v1); nv++;}
    if (mc_interior_edge_crosses(cube.v6, cube.v2, pass)) {x += 1.0; y += 1.0; z += 1./(1-cube.v6/cube.v2); nv++;}
    if (mc_interior_edge_crosses(cube.v7, cube.v3, pass)) {y += 1.0; z += 1./(1-cube.v7/cube.v3); nv++;}
  }
  if (nv == 0) { x = 0.5; y = 0.5; z = 0.5; return; }
  x /= nv; y /= nv; z /= nv;
}

// Physically-scaled marching-cubes triangle area (dx/dy/dz are the physical
// extents of the current cube in x/y/z).
__device__ inline
double add_triangle_phys(const char *trig, char n, Cube cube,
                         double dx, double dy, double dz) {
  double xt[3],yt[3],zt[3];
  double area = 0.0;

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
        mc_interior_vertex(cube, xt[ t % 3 ], yt[ t % 3 ], zt[ t % 3 ]);
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
    double iso,                       // log10(T_ref)
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

    // 64-bit offsets: a full-width slab at N=2048 exceeds 2^31 cells.
    #define D(kk,jj,ii) (data[((long long)(kk)*Ny + (jj))*(long long)Nx + (ii)] - iso)

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

    #define D_MB(kk,jj,ii) (data[((long long)(kk)*Ny_pad + (jj))*(long long)Nx_pad + (ii)] - iso)

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
    proc_cube_align = _build_process_cube_align(proc_cube_phys)

    cuda_src = (
        # Define float.h constants stripped along with the include.
        # FLT_EPSILON is used by test_face / test_interior in the MC source.
        '#ifndef FLT_EPSILON\n'
        '#define FLT_EPSILON 1.192093e-07f\n'
        '#endif\n'
        + lut_src
        + mc_src              # Cube struct, test_face, test_interior, add_triangle, process_cube
        + _ADD_TRIANGLE_PHYS
        + proc_cube_phys
        + _KERNEL_MAIN        # slab-based area kernel
        + _KERNEL_MAIN_MB     # meshblock area kernel
        + _BOX_AVG_B_DEVICE   # box-average B over s×s×s block for alignment kernel
        + _ADD_TRIANGLE_ALIGN # per-triangle alignment accumulator
        + proc_cube_align     # process_cube_align (generated)
        + _KERNEL_ALIGN_MB    # meshblock alignment kernel
        + _KERNEL_ALIGN_MB_BATCH  # one flattened launch for an alignment batch
        + _KERNEL_ALIGN_COARSE  # grouped coarse-cube alignment kernel
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
        mod.get_function('mc_area_kernel')       # force compile
        mod.get_function('mc_area_kernel_mb')    # force compile
        mod.get_function('mc_align_kernel_mb')   # force compile
        mod.get_function('mc_align_kernel_mb_batch')  # force compile
        mod.get_function('mc_align_kernel_coarse')  # force compile
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

def _aug_array(log_slab: np.ndarray, periodic_xy=(True, True)) -> np.ndarray:
    """
    Add one ghost column in x1 (axis=-1) and x2 (axis=-2): the wrapped first
    column on a periodic axis, the repeated last column otherwise.
    Output shape: (Nz, Ny+1, Nx+1).
    """
    Nz, Ny, Nx = log_slab.shape
    aug = np.empty((Nz, Ny + 1, Nx + 1), dtype=np.float64)
    aug[:, :-1, :-1] = log_slab
    aug[:, :-1, -1]  = log_slab[:, :, 0 if periodic_xy[0] else -1]
    aug[:, -1, :]    = aug[:, 0 if periodic_xy[1] else -2, :]
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

def _fetch_temp_mb(ad, idx: int) -> np.ndarray:
    """Fetch ONE meshblock's temperature via the proper `ad.data(var, mbl,
    mbh)` accessor (always an explicit range) -- never `ad.data_raw[var]`
    and never for the whole domain at once. See MESHBLOCK_DATA_ACCESS.md at
    the repo root: a no-range call bypasses the xp GPU/CPU backend
    abstraction, skips MPI-safe local-index translation, and is
    memory-unsafe for large runs. `ad.data("temp", ...)` is the
    already-registered, EOS-aware temperature data_func (see
    core/data_functions.py), so this also replaces the manual
    `(gamma-1)*eint/dens` recomputation these callers used to do.
    """
    return asnumpy(ad.data("temp", idx, idx + 1))[0]


def _fetch_log_temp_mb(ad, idx: int) -> np.ndarray:
    """Fetch ONE meshblock's log10(temperature) -- see `_fetch_temp_mb`."""
    return np.log10(_fetch_temp_mb(ad, idx))


def _select_interface_levels(profile, level_min, level_max, target, n_buffer,
                             periodic_z, label='T_ref'):
    """
    Choose the contiguous run of meshblock z-levels [lo, hi] for a slab.

    The level whose mean ``profile`` is closest to ``target`` plus n_buffer
    levels on each side is always kept. A level that owns cubes crossing the
    isosurface is added as well: cubes owned by level iz take their upper
    corners from level iz+1, so iz is needed whenever the value range of
    levels iz and iz+1 together brackets ``target``. A mean profile alone
    misses a corrugated or multiply-connected interface whose meshblocks lie
    outside that window, and those cubes would be silently dropped.

    Returns (lo, hi, crossing, needed) with ``needed`` the per-level flag.
    """
    n_levels = len(profile)
    crossing = int(np.argmin(np.abs(np.asarray(profile) - target)))
    lo = max(0, crossing - n_buffer)
    hi = min(n_levels - 1, crossing + n_buffer)

    needed = np.zeros(n_levels, dtype=bool)
    for iz in range(n_levels):
        above = iz + 1
        if above >= n_levels:
            above = 0 if periodic_z else iz
        vmin = min(level_min[iz], level_min[above])
        vmax = max(level_max[iz], level_max[above])
        needed[iz] = bool(vmin <= target <= vmax and vmin < vmax)

    if np.any(needed):
        idx = np.flatnonzero(needed)
        new_lo = min(lo, int(idx[0]))
        new_hi = max(hi, int(idx[-1]))
        if (new_lo, new_hi) != (lo, hi):
            warnings.warn(
                f'The {label} isosurface crosses meshblock z-levels outside '
                f'the interface window {lo}..{hi} (crossing level {crossing} '
                f'+- n_buffer={n_buffer}); selecting levels {new_lo}..{new_hi} '
                f'of {n_levels} so that no crossing meshblock is skipped.',
                stacklevel=3,
            )
        lo, hi = new_lo, new_hi
    return lo, hi, crossing, needed


def _interface_z_level_selection(ad, T_ref, n_buffer=1):
    """Return (z3_unique, lo, hi, crossing, needed) for T=T_ref."""
    mb_geo = ad.mb_geometry

    z3_unique  = np.sort(np.unique(mb_geo[:, 4]))
    log_T_by_z = np.empty(len(z3_unique))
    log_T_min  = np.empty(len(z3_unique))
    log_T_max  = np.empty(len(z3_unique))
    for iz, z3min in enumerate(z3_unique):
        idx_list = np.where(mb_geo[:, 4] == z3min)[0]
        means, mins, maxs = [], [], []
        for m in idx_list:
            temp = _fetch_temp_mb(ad, int(m))
            means.append(np.log10(float(temp.mean())))
            # log10 of the extreme in the data dtype equals the extreme of
            # the per-cell log10 that the marching-cubes kernels compare.
            mins.append(float(np.log10(np.nanmin(temp))))
            maxs.append(float(np.log10(np.nanmax(temp))))
        log_T_by_z[iz] = np.mean(means)
        log_T_min[iz] = min(mins)
        log_T_max[iz] = max(maxs)

    periodic_z = _get_periodic_flags(ad)[2]
    lo, hi, iz_cross, needed = _select_interface_levels(
        log_T_by_z, log_T_min, log_T_max, float(np.log10(T_ref)),
        n_buffer, periodic_z, label=f'T={T_ref:.4g}')
    return z3_unique, lo, hi, iz_cross, needed


def find_interface_z_levels(ad, T_ref, n_buffer=1):
    """
    Return the x3min values of the contiguous meshblock z-levels that contain
    every cube crossing the T=T_ref isosurface, and the index of the level
    whose mean log10(T) is closest to log10(T_ref).

    The selection always includes that crossing level +- n_buffer levels and
    is widened (with a warning) to every level whose meshblocks, or the
    blocks directly above them, bracket T_ref.
    """
    z3_unique, lo, hi, iz_cross, _ = _interface_z_level_selection(
        ad, T_ref, n_buffer=n_buffer)
    return z3_unique[lo : hi + 1], iz_cross


def _slab_z_levels(ad, T_ref, n_buffer=1):
    """
    Interface z-levels for the stacked-slab method.

    The slab has no ghost layer above its top level, so when that level owns
    crossing cubes the next level up is appended to supply their upper
    corners (the meshblock method gets them from its +z padding instead).
    """
    z3_unique, lo, hi, iz_cross, needed = _interface_z_level_selection(
        ad, T_ref, n_buffer=n_buffer)
    if needed[hi] and hi + 1 < len(z3_unique):
        hi += 1
    return z3_unique[lo : hi + 1], iz_cross


def _build_slab(ad, T_ref, z_selected):
    """
    Reconstruct the temperature slab for the selected meshblock z-levels.
    Returns (slab, log_slab, iso) where slab has shape (Nz_slab, Nx2, Nx1).
    """
    mb_geo = ad.mb_geometry
    nx3_mb, nx2_mb, nx1_mb = ad.nx3, ad.nx2, ad.nx1

    Nz_slab = len(z_selected) * nx3_mb
    Nx1, Nx2 = ad.Nx1, ad.Nx2

    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x2_vals = np.sort(np.unique(mb_geo[:, 2]))
    slab    = np.empty((Nz_slab, Nx2, Nx1), dtype=np.float32)

    for m in np.where(np.isin(mb_geo[:, 4], z_selected))[0]:
        m = int(m)
        i1 = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
        i2 = int(np.searchsorted(x2_vals, mb_geo[m, 2]))
        iz = int(np.searchsorted(z_selected, mb_geo[m, 4]))
        slab[iz*nx3_mb:(iz+1)*nx3_mb,
             i2*nx2_mb:(i2+1)*nx2_mb,
             i1*nx1_mb:(i1+1)*nx1_mb] = _fetch_temp_mb(ad, m)

    iso      = float(np.log10(T_ref))
    log_slab = np.log10(slab.astype(np.float64))

    if iso < log_slab.min() or iso > log_slab.max():
        warnings.warn(
            f'T_ref ({T_ref:.4g}) outside slab temperature range '
            f'[{slab.min():.4g}, {slab.max():.4g}]. '
            'Try increasing n_mb_buffer.',
            stacklevel=4
        )
    return slab, log_slab, iso


# ── Block-by-block helpers ────────────────────────────────────────────────────

def _find_mc_neighbors(ad):
    """
    Build +x, +y, +z neighbor lookup for every meshblock.

    Uses find_neighbour_blocks, which reads the periodicity of every axis
    from the header via the mb_logical array.

    Returns
    -------
    list of (ix_plus, iy_plus, iz_plus) per meshblock.
        Each entry is a meshblock index, or None when the block sits at the
        + boundary of a non-periodic axis.
    """
    neighbors = []
    for m in range(ad.n_mbs):
        x_plus = find_neighbour_blocks(ad, m, 'x+')
        y_plus = find_neighbour_blocks(ad, m, 'y+')
        z_plus = find_neighbour_blocks(ad, m, 'z+')

        ix = x_plus[0] if x_plus else None
        iy = y_plus[0] if y_plus else None
        iz = z_plus[0] if z_plus else None

        neighbors.append((ix, iy, iz))
    return neighbors


# Ghost octants of a padded block as (dz, dy, dx) offsets, core excluded.
_GHOST_OCTANTS = tuple(o for o in np.ndindex(2, 2, 2) if o != (0, 0, 0))


def _octant_source(neighbors, m, octant):
    """
    Meshblock supplying ghost octant (dz, dy, dx) of block m, walking the
    neighbor chain in z, y, x order; None if the chain leaves the domain.
    """
    dz, dy, dx = octant
    b = m
    for axis, step in ((2, dz), (1, dy), (0, dx)):
        if step:
            b = neighbors[b][axis]
            if b is None:
                return None
    return b


def _octant_slices(octant, s_ghost, nx3, nx2, nx1):
    """(destination, source) slices of a ghost octant in a padded block."""
    dst, src = [], []
    for step, n in zip(octant, (nx3, nx2, nx1)):
        dst.append(slice(n, None) if step else slice(0, n))
        src.append(slice(0, s_ghost) if step else slice(0, n))
    return tuple(dst), tuple(src)


def _replicate_missing_ghosts(pad, missing, nx3, nx2, nx1):
    """
    Fill the + ghost layer of every non-periodic boundary axis by repeating
    the last real layer (constant extrapolation, as outflow ghost cells do).
    ``missing`` is (x, y, z) booleans; axes are processed x, y, z so every
    octant whose neighbor chain left the domain ends up overwritten from a
    layer that is already valid.
    """
    if missing[0]:
        pad[..., nx1:] = pad[..., nx1 - 1:nx1]
    if missing[1]:
        pad[..., nx2:, :] = pad[..., nx2 - 1:nx2, :]
    if missing[2]:
        pad[..., nx3:, :, :] = pad[..., nx3 - 1:nx3, :, :]
    return pad


def _build_padded_block(ad, m, neighbors, s_ghost, nx3, nx2, nx1):
    """
    Assemble a padded array for meshblock m.

    The padded array has shape (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost).
    The extra s_ghost layers in each + direction are filled from neighbour
    meshblock data so the MC kernel can access all 8 corners of every cube
    whose lower-left-back corner lies within [0, nx3) × [0, nx2) × [0, nx1).

    Fetches only the (up to 8) individual meshblocks actually needed --
    self, +x, +y, +xy, +z, +xz, +yz, +xyz -- via `_fetch_log_temp_mb`, never
    a precomputed whole-domain array (see MESHBLOCK_DATA_ACCESS.md). A small
    per-call cache avoids re-fetching the same meshblock twice within one
    padded block.

    Boundary conditions
    -------------------
    Periodic axes always have a neighbour. At the + boundary of a
    non-periodic axis the last real layer is replicated into the ghost
    region (constant extrapolation, like outflow ghost cells).
    """
    cache = {}

    def _get(idx):
        if idx not in cache:
            cache[idx] = _fetch_log_temp_mb(ad, idx)
        return cache[idx]

    pad = np.empty((nx3 + s_ghost, nx2 + s_ghost, nx1 + s_ghost), dtype=np.float64)
    pad[:nx3, :nx2, :nx1] = _get(m)

    for octant in _GHOST_OCTANTS:
        b = _octant_source(neighbors, m, octant)
        if b is None:
            continue
        dst, src = _octant_slices(octant, s_ghost, nx3, nx2, nx1)
        pad[dst] = _get(b)[src]

    missing = tuple(n is None for n in neighbors[m])
    return _replicate_missing_ghosts(pad, missing, nx3, nx2, nx1)


def _calc_area_gpu_mb(padded_gpu, iso, nx3, nx2, nx1, s_ghost, s, dz, dy, dx):
    """
    Run mc_area_kernel_mb on one GPU-resident padded block for step_size=s.

    Parameters
    ----------
    padded_gpu : xp.ndarray  shape (nx3+s_ghost, nx2+s_ghost, nx1+s_ghost)
    iso        : float       log10(T_ref)
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

def calc_area_one_step(ad, T_ref, step_size=1, n_mb_buffer=1, use_gpu=True):
    """
    Compute the T=T_ref isosurface area for a single step size.

    Uses the interface-slab optimisation to avoid reconstructing the full
    Nx3 × Nx2 × Nx1 array.  Matches the C++ `area1` (step_size=1) exactly.

    Parameters
    ----------
    ad : AthenaData
    T_ref : float
    step_size : int
    n_mb_buffer : int
        Extra meshblock z-levels on each side of the interface slab.
    use_gpu : bool
        Use GPU (CuPy) kernel if available; fall back to scikit-image.

    Returns
    -------
    float  Physical surface area.
    """
    z_sel, iz_cross = _slab_z_levels(ad, T_ref, n_buffer=n_mb_buffer)
    _, log_slab, iso = _build_slab(ad, T_ref, z_sel)
    Nz_slab = log_slab.shape[0]

    aug = _aug_array(log_slab, _get_periodic_flags(ad)[:2])

    if use_gpu and cupy_enabled and _get_kernel() is not None:
        return _calc_area_gpu(aug, iso, ad, step_size)
    else:
        return _calc_area_skimage(aug, iso, ad, step_size)


def calc_areas_all_steps(ad, T_ref, step_sizes=None, n_mb_buffer=1, use_gpu=True,
                         verbose=True):
    """
    Compute isosurface area for all requested step sizes on a single slab
    (slab is built only once).

    Parameters
    ----------
    ad : AthenaData
    T_ref : float
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

    z_sel, iz_cross = _slab_z_levels(ad, T_ref, n_buffer=n_mb_buffer)
    _, log_slab, iso = _build_slab(ad, T_ref, z_sel)
    Nz_slab = log_slab.shape[0]

    if verbose:
        print(f'  Slab: {Nz_slab}/{ad.Nx3} z-cells '
              f'({100.*Nz_slab/ad.Nx3:.1f}%,  interface idx={iz_cross})')

    aug = _aug_array(log_slab, _get_periodic_flags(ad)[:2])

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


def calc_area_mb_based(ad, T_ref, step_sizes=None, n_mb_buffer=1, use_gpu=True,
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
    ad          : AthenaData  (must support ad.data("temp", mbl, mbh))
    T_ref      : float
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

    n_mb, nx3, nx2, nx1 = ad.n_mbs, ad.nx3, ad.nx2, ad.nx1

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
    iso = float(np.log10(T_ref))

    use_kernel = use_gpu and cupy_enabled and _get_module() is not None

    if not use_kernel:
        # CPU fallback: use the slab-based approach.
        if rank == 0 and verbose:
            print('  Block-by-block: GPU unavailable; falling back to slab method.')

        return calc_areas_all_steps(
            ad,
            T_ref,
            step_sizes=valid_steps,
            n_mb_buffer=n_mb_buffer,
            use_gpu=False,
            verbose=(verbose and rank == 0),
        )

    # Identify the selected z-levels.
    z_selected, iz_cross = find_interface_z_levels(
        ad,
        T_ref,
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

    # Build neighbor lookup.
    neighbors = _find_mc_neighbors(ad)

    # Running totals per valid step size.
    totals_gpu = [xp.zeros(1, dtype=np.float64) for _ in valid_steps]

    kernel = _get_kernel_mb()

    n_active = 0

    for m in my_slab:
        m = int(m)
        # Include the positive-side face/edge/corner vertices before culling.
        # A uniform block can still own cubes crossing into a neighboring
        # block. The padded array stays bounded to one meshblock and its
        # ghosts; _build_padded_block fetches explicit meshblock ranges.
        padded = _build_padded_block(
            ad,
            m,
            neighbors,
            s_ghost,
            nx3,
            nx2,
            nx1,
        )
        # The cube kernel classifies strictly negative vertices as cold;
        # an exact-level vertex belongs to the warm side.
        if not (padded.min() < iso <= padded.max()):
            continue
        n_active += 1
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

    Assumes every meshblock has shape (ad.nx3, ad.nx2, ad.nx1).

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


def _sample_log_temp_global_index(ad, lookup, gk, gj, gi, mb_cache=None):
    """
    Sample log10(T) at one global cell index using BCs from ad._header.

    x1, x2, x3 periodicity is determined from:
        ad._header['mesh']['ix1_bc']
        ad._header['mesh']['ix2_bc']
        ad._header['mesh']['ix3_bc']

    `mb_cache` (optional dict) lets a caller sampling many indices reuse a
    meshblock's data across calls instead of re-fetching it (via
    `_fetch_temp_mb`, never `ad.data_raw`) for every single cell.
    """
    nx3, nx2, nx1 = ad.nx3, ad.nx2, ad.nx1

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

    if mb_cache is not None:
        if m not in mb_cache:
            mb_cache[m] = _fetch_temp_mb(ad, m)
        temp_mb = mb_cache[m]
    else:
        temp_mb = _fetch_temp_mb(ad, m)

    return float(np.log10(float(temp_mb[lk, lj, li])))


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


def _build_global_phase_indicator(ad, T_ref, phase='cold', dtype=np.float32):
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
    nx3_mb, nx2_mb, nx1_mb = ad.nx3, ad.nx2, ad.nx1

    x1_vals = np.sort(np.unique(mb_geo[:, 0]))
    x2_vals = np.sort(np.unique(mb_geo[:, 2]))
    x3_vals = np.sort(np.unique(mb_geo[:, 4]))

    indicator = np.empty((ad.Nx3, ad.Nx2, ad.Nx1), dtype=dtype)

    for m in range(ad.n_mbs):
        i1 = int(np.searchsorted(x1_vals, mb_geo[m, 0]))
        i2 = int(np.searchsorted(x2_vals, mb_geo[m, 2]))
        i3 = int(np.searchsorted(x3_vals, mb_geo[m, 4]))

        temp = _fetch_temp_mb(ad, m)
        if phase == 'cold':
            block = temp <= T_ref
        else:
            block = temp >= T_ref

        indicator[
            i3*nx3_mb:(i3+1)*nx3_mb,
            i2*nx2_mb:(i2+1)*nx2_mb,
            i1*nx1_mb:(i1+1)*nx1_mb,
        ] = block.astype(dtype, copy=False)

    return indicator


def _build_all_phase_indicator(ad, T_ref, phase='cold'):
    """Build a bool phase mask in the native meshblock ordering."""
    phase = str(phase).lower()
    if phase not in ('cold', 'hot'):
        raise ValueError(f'Unknown phase={phase!r}. Use "cold" or "hot".')

    n_mb = ad.n_mbs
    out = np.empty((n_mb, ad.nx3, ad.nx2, ad.nx1), dtype=bool)
    for m in range(n_mb):
        temp_mb = _fetch_temp_mb(ad, m)
        out[m] = (temp_mb <= T_ref) if phase == 'cold' else (temp_mb >= T_ref)
    return out


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


def calc_area_smoothed_indicator_mc(ad, T_ref, step_sizes=None, phase='cold',
                                    sigma_factor=1.0, verbose=True,
                                    compare_scipy=False, scan_axis='auto'):
    """
    Compute scale-dependent interface area with smoothed-indicator marching cubes.

    For each requested step size s, this method builds the phase indicator
    chi = 1(T <= T_ref) by default, smooths chi with Gaussian sigma
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

    indicator = _build_global_phase_indicator(ad, T_ref, phase=phase)

    if compare_scipy:
        indicator_ref = _build_global_phase_indicator(ad, T_ref, phase=phase)
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

        aug = _aug_array(smoothed, _get_periodic_flags(ad)[:2])
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
    T_ref,
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

    n_mb, nx3, nx2, nx1 = ad.n_mbs, ad.nx3, ad.nx2, ad.nx1
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
    indicator_all = _build_all_phase_indicator(ad, T_ref, phase=phase)

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


def _build_coarse_sampled_grid_offset(ad, T_ref, step_size,
                                      offset_k=0, offset_j=0, offset_i=0,
                                      verbose=True, return_indices=False):
    """
    Build a coarse log10(T) grid using BCs from ad._header.

    Periodic directions are closed with offset + N.
    Non-periodic directions are closed at N - 1.

    The final interval along an axis is shorter than step_size whenever the
    step does not tile it (N not a multiple of the step on a periodic axis,
    a step larger than N, or the N - 1 closure of a non-periodic axis), so
    the returned uniform ``spacing`` is only exact for the other intervals.
    With ``return_indices=True`` the global cell index of every sample along
    (x3, x2, x1) is returned as well; marching-cubes vertices must be mapped
    through those indices to get physical positions.

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

        return idx

    z_idx = make_indices(ad.Nx3, ok, periodic_z, 'x3')
    y_idx = make_indices(ad.Nx2, oj, periodic_y, 'x2')
    x_idx = make_indices(ad.Nx1, oi, periodic_x, 'x1')

    coarse = np.empty((len(z_idx), len(y_idx), len(x_idx)), dtype=np.float64)

    mb_cache = {}
    for kk, gk in enumerate(z_idx):
        for jj, gj in enumerate(y_idx):
            for ii, gi in enumerate(x_idx):
                coarse[kk, jj, ii] = _sample_log_temp_global_index(
                    ad, lookup, gk, gj, gi, mb_cache=mb_cache
                )

    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1

    spacing = (s * dz, s * dy, s * dx)

    if verbose:
        iso = float(np.log10(T_ref))
        print(
            f'  Global coarse grid step={s}, '
            f'offset=({ok},{oj},{oi}), '
            f'periodic=(x3={periodic_z}, x2={periodic_y}, x1={periodic_x}), '
            f'shape={coarse.shape}, '
            f'logT range=[{coarse.min():.6g}, {coarse.max():.6g}], '
            f'iso={iso:.6g}'
        )

    if return_indices:
        return coarse, spacing, (z_idx, y_idx, x_idx)
    return coarse, spacing


def _coarse_vertices_to_physical(verts, indices, cell_spacing):
    """
    Map marching-cubes vertices in coarse-sample index space to physical
    (x3, x2, x1) positions. Vertices lie on cube edges, so a piecewise-linear
    map through the sampled global cell indices is exact even where the
    final interval is shorter than the others.
    """
    out = np.empty_like(np.asarray(verts, dtype=np.float64))
    for axis, (idx, d) in enumerate(zip(indices, cell_spacing)):
        idx = np.asarray(idx, dtype=np.float64)
        out[:, axis] = np.interp(verts[:, axis], np.arange(len(idx)), idx) * d
    return out


def calc_area_global_coarse_sampled_offsets(ad, T_ref, step_sizes=None,
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
    T_ref : float
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
    iso = float(np.log10(T_ref))

    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)
    cell_spacing = (
        (ad.x3max - ad.x3min) / ad.Nx3,
        (ad.x2max - ad.x2min) / ad.Nx2,
        (ad.x1max - ad.x1min) / ad.Nx1,
    )

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
                    coarse, spacing, indices = _build_coarse_sampled_grid_offset(
                        ad,
                        T_ref,
                        s,
                        offset_k=ok,
                        offset_j=oj,
                        offset_i=oi,
                        verbose=verbose,
                        return_indices=True,
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
                            allow_degenerate=False,
                        )
                        verts = _coarse_vertices_to_physical(
                            verts, indices, cell_spacing)

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

    for key in ('T_ref', 'smoothing_sigma_factor'):
        if key in requested_metadata:
            cached_value = area_mc.get(key)
            if cached_value is None and key == 'T_ref':
                # Caches written before the T_peak -> T_ref rename.
                cached_value = area_mc.get('T_peak')
            try:
                if not np.isclose(float(cached_value), float(requested_metadata[key])):
                    return False
            except Exception:
                return False

    for key in ('phase', 'scan_axis'):
        if key in requested_metadata:
            if str(area_mc.get(key, '')).lower() != str(requested_metadata[key]).lower():
                return False

    # Caches written without the slab buffer stay readable; a recorded
    # buffer must match the requested one.
    if 'n_mb_buffer' in requested_metadata:
        cached_buffer = area_mc.get('n_mb_buffer')
        if cached_buffer is not None:
            try:
                if int(np.asarray(cached_buffer).ravel()[0]) != int(requested_metadata['n_mb_buffer']):
                    return False
            except Exception:
                return False

    return True

def set_area(ad, T_ref, h5_path: str, step_sizes=None, redo=False,
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
        'T_ref': float(T_ref),
    }
    if area_method == AREA_METHOD_MARCHING_CUBES:
        requested_metadata['n_mb_buffer'] = int(n_mb_buffer)
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

    n_mb, nx3, nx2, nx1 = ad.n_mbs, ad.nx3, ad.nx2, ad.nx1

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
            T_ref,
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
                T_ref,
                step_sizes=mb_steps,
                n_mb_buffer=n_mb_buffer,
                use_gpu=True,
                verbose=(verbose and rank == 0),
                mpi_manager=mpi_manager,
            )
        else:
            s_mb, a_mb = calc_areas_all_steps(
                ad,
                T_ref,
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
            T_ref,
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
            T_ref,
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
            'T_ref': float(T_ref),
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


_BOX_AVG_B_DEVICE = r"""
// Arithmetic mean of the s×s×s cell block starting at (K,J,I) in the padded array.
// For s=1 this reduces to the single cell value at (K,J,I).
__device__ inline double box_avg_B(
    const double* __restrict__ f, int Ny_pad, int Nx_pad,
    int K, int J, int I, int s
) {
    double sum = 0.0;
    for (int dk = 0; dk < s; dk++)
        for (int dj = 0; dj < s; dj++)
            for (int di = 0; di < s; di++)
                sum += f[(K+dk)*Ny_pad*Nx_pad + (J+dj)*Nx_pad + (I+di)];
    return sum * (1.0 / (double)(s*s*s));
}
"""


_ADD_TRIANGLE_ALIGN = r"""
__device__ inline
void add_triangle_align(const char *trig, char n, Cube cube,
                        double dx, double dy, double dz,
                        const double* __restrict__ bx,
                        const double* __restrict__ by,
                        const double* __restrict__ bz,
                        int Ny_pad, int Nx_pad,
                        int I, int J, int K, int s,
                        double* __restrict__ area_out,
                        double* __restrict__ align_out) {
  double xt[3],yt[3],zt[3];

  for( int t = 0 ; t < 3*n ; t++ ){
    switch( trig[t] ) {
      case  0: xt[t%3]=1./(1-cube.v1/cube.v0); yt[t%3]=0.; zt[t%3]=0.; break;
      case  1: xt[t%3]=1.; yt[t%3]=1./(1-cube.v2/cube.v1); zt[t%3]=0.; break;
      case  2: xt[t%3]=1./(1-cube.v2/cube.v3); yt[t%3]=1.; zt[t%3]=0.; break;
      case  3: xt[t%3]=0.; yt[t%3]=1./(1-cube.v3/cube.v0); zt[t%3]=0.; break;
      case  4: xt[t%3]=1./(1-cube.v5/cube.v4); yt[t%3]=0.; zt[t%3]=1.; break;
      case  5: xt[t%3]=1.; yt[t%3]=1./(1-cube.v6/cube.v5); zt[t%3]=1.; break;
      case  6: xt[t%3]=1./(1-cube.v6/cube.v7); yt[t%3]=1.; zt[t%3]=1.; break;
      case  7: xt[t%3]=0.; yt[t%3]=1./(1-cube.v7/cube.v4); zt[t%3]=1.; break;
      case  8: xt[t%3]=0.; yt[t%3]=0.; zt[t%3]=1./(1-cube.v4/cube.v0); break;
      case  9: xt[t%3]=1.; yt[t%3]=0.; zt[t%3]=1./(1-cube.v5/cube.v1); break;
      case 10: xt[t%3]=1.; yt[t%3]=1.; zt[t%3]=1./(1-cube.v6/cube.v2); break;
      case 11: xt[t%3]=0.; yt[t%3]=1.; zt[t%3]=1./(1-cube.v7/cube.v3); break;
      case 12:
        mc_interior_vertex(cube, xt[t%3], yt[t%3], zt[t%3]);
        break;
      default: break;
    }

    if( t%3 == 2 ) {
      double v1=(xt[0]-xt[1])*dx, v2=(yt[0]-yt[1])*dy, v3=(zt[0]-zt[1])*dz;
      double w1=(xt[0]-xt[2])*dx, w2=(yt[0]-yt[2])*dy, w3=(zt[0]-zt[2])*dz;
      double u1=v2*w3-v3*w2, u2=v3*w1-v1*w3, u3=v1*w2-v2*w1;
      double unorm = sqrt(u1*u1+u2*u2+u3*u3);
      double area  = 0.5*unorm;
      atomicAdd(area_out, area);
      if( unorm < 1.0e-30 ) continue;

      // B averaged over the s×s×s cell block that defines this MC cube.
      double bx_c = box_avg_B(bx, Ny_pad, Nx_pad, K, J, I, s);
      double by_c = box_avg_B(by, Ny_pad, Nx_pad, K, J, I, s);
      double bz_c = box_avg_B(bz, Ny_pad, Nx_pad, K, J, I, s);
      double bnorm = sqrt(bx_c*bx_c+by_c*by_c+bz_c*bz_c);
      if( bnorm < 1.0e-30 ) continue;

      // b̂·n̂ = (bcc1*u1 + bcc2*u2 + bcc3*u3) / (|B|*|u|)
      // bx=bcc1=x1-comp, u1=x1-comp of n̂ — axes consistent
      double dot = (bx_c*u1 + by_c*u2 + bz_c*u3) / (bnorm * unorm);
      atomicAdd(align_out, area * dot * dot);
    }
  }
}
"""


_KERNEL_ALIGN_MB = r"""
extern "C" __global__ void mc_align_kernel_mb(
    const double* __restrict__ data,   // padded log10(T)
    const double* __restrict__ bx,     // padded bcc1  (x1-component of B)
    const double* __restrict__ by,     // padded bcc2  (x2-component)
    const double* __restrict__ bz,     // padded bcc3  (x3-component)
    double* __restrict__ area_out,
    double* __restrict__ align_out,    // sum of area*(b̂·n̂)²
    int nx3, int nx2, int nx1,
    int s_ghost,
    int s,
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

    int I = i*s, J = j*s, K = k*s;

    int Nx_pad = nx1 + s_ghost;
    int Ny_pad = nx2 + s_ghost;

    #define _D(kk,jj,ii) (data[(kk)*Ny_pad*Nx_pad+(jj)*Nx_pad+(ii)]-iso)

    Cube cube(
        _D(K,   J,   I  ), _D(K,   J,   I+s),
        _D(K,   J+s, I+s), _D(K,   J+s, I  ),
        _D(K+s, J,   I  ), _D(K+s, J,   I+s),
        _D(K+s, J+s, I+s), _D(K+s, J+s, I  )
    );

    #undef _D

    double cube_dx = (double)s * dx_cell;
    double cube_dy = (double)s * dy_cell;
    double cube_dz = (double)s * dz_cell;

    process_cube_align(cube, cube_dx, cube_dy, cube_dz,
                       bx, by, bz, Ny_pad, Nx_pad, I, J, K, s,
                       area_out, align_out);
}
"""


_KERNEL_ALIGN_MB_BATCH = r"""
extern "C" __global__ void mc_align_kernel_mb_batch(
    const double* __restrict__ data,
    const double* __restrict__ bx,
    const double* __restrict__ by,
    const double* __restrict__ bz,
    double* __restrict__ area_out,
    double* __restrict__ align_out,
    int n_blocks, int nx3, int nx2, int nx1,
    int s_ghost, int s, double iso,
    double dz_cell, double dy_cell, double dx_cell
) {
    // 64-bit thread and block offsets: n_blocks * cubes_per_block and
    // mb * block_size can exceed 2^31 for large batches of large meshblocks.
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nx1 / s, nj = nx2 / s, nk = nx3 / s;
    long long cubes_per_block = (long long)ni * nj * nk;
    if (tid >= (long long)n_blocks * cubes_per_block) return;

    long long mb = tid / cubes_per_block;
    int local = (int)(tid - mb * cubes_per_block);
    int i = local % ni;
    int j = (local / ni) % nj;
    int k = local / (ni * nj);
    int I = i*s, J = j*s, K = k*s;
    int Nx_pad = nx1 + s_ghost, Ny_pad = nx2 + s_ghost;
    long long block_size = (long long)(nx3 + s_ghost) * Ny_pad * Nx_pad;
    const double* d = data + mb * block_size;
    const double* bxp = bx + mb * block_size;
    const double* byp = by + mb * block_size;
    const double* bzp = bz + mb * block_size;

    #define _D(kk,jj,ii) (d[(kk)*Ny_pad*Nx_pad+(jj)*Nx_pad+(ii)]-iso)
    Cube cube(
        _D(K,   J,   I  ), _D(K,   J,   I+s),
        _D(K,   J+s, I+s), _D(K,   J+s, I  ),
        _D(K+s, J,   I  ), _D(K+s, J,   I+s),
        _D(K+s, J+s, I+s), _D(K+s, J+s, I  )
    );
    #undef _D

    process_cube_align(cube, (double)s*dx_cell, (double)s*dy_cell,
                       (double)s*dz_cell, bxp, byp, bzp, Ny_pad, Nx_pad,
                       I, J, K, s, area_out, align_out);
}
"""


_KERNEL_ALIGN_COARSE = r"""
extern "C" __global__ void mc_align_kernel_coarse(
    const double* __restrict__ cubes,
    const double* __restrict__ bx_avg,
    const double* __restrict__ by_avg,
    const double* __restrict__ bz_avg,
    double* __restrict__ area_out,
    double* __restrict__ align_out,
    int n_cubes,
    double cube_dz, double cube_dy, double cube_dx
) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= n_cubes) return;

    const double* v = cubes + 8*c;
    Cube cube(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    // box_avg_B with s=1 reads element zero.  Each pointer addresses one
    // pre-averaged coarse-cube field value, so no large padded array is needed.
    process_cube_align(cube, cube_dx, cube_dy, cube_dz,
                       bx_avg + c, by_avg + c, bz_avg + c,
                       2, 2, 0, 0, 0, 1, area_out, align_out);
}
"""


def _build_process_cube_align(proc_cube_phys_src: str) -> str:
    """
    Generate process_cube_align from the already-built process_cube_phys source.

    process_cube_phys uses:
      return add_triangle_phys(TILING, N, cube, dx, dy, dz);
      return 0.0;   (no-triangle cases)

    Transformations applied:
      1. Change function name/signature: double→void, add B arrays + position + outputs
      2. return add_triangle_phys(  →  add_triangle_align(   (drop return, rename)
      3. , cube, dx, dy, dz);  →  , cube, dx, dy, dz, bx, by, bz, Ny_pad, Nx_pad,
                                    I, J, K, s, area_out, align_out);
      4. return 0.0;  →  return;   (no-triangle case, void function)
    """
    src = proc_cube_phys_src

    # 1. Rename and extend signature.
    src = src.replace(
        'double process_cube_phys(Cube cube, double dx, double dy, double dz)',
        '__device__ void process_cube_align(\n'
        '    Cube cube, double dx, double dy, double dz,\n'
        '    const double* __restrict__ bx,\n'
        '    const double* __restrict__ by,\n'
        '    const double* __restrict__ bz,\n'
        '    int Ny_pad, int Nx_pad,\n'
        '    int I, int J, int K, int s,\n'
        '    double* __restrict__ area_out,\n'
        '    double* __restrict__ align_out)',
    )

    # 2. Replace "return add_triangle_phys(" with a compound open-brace.
    #    The original "return" prevented switch-case fall-through; we preserve that
    #    by wrapping each call as "{ add_triangle_align(...); return; }".
    src = src.replace('return add_triangle_phys(', '{ add_triangle_align(')

    # 3. Close the compound statement: extend args and add return + closing brace.
    src = src.replace(
        ', cube, dx, dy, dz);',
        ', cube, dx, dy, dz, bx, by, bz, Ny_pad, Nx_pad, I, J, K, s, area_out, align_out); return; }',
    )

    # 4. No-triangle case: return 0.0; → return;
    src = src.replace('return 0.0;', 'return;')

    return '\n// ── process_cube_align (field-normal alignment) ──\n' + src + '\n'


def _get_kernel_align_mb():
    """Return the meshblock alignment kernel (mc_align_kernel_mb)."""
    mod = _get_module()
    return mod.get_function('mc_align_kernel_mb') if mod is not None else None


def _get_kernel_align_mb_batch():
    """Return the flattened, multi-meshblock alignment kernel."""
    mod = _get_module()
    return mod.get_function('mc_align_kernel_mb_batch') if mod is not None else None


def _get_kernel_align_coarse():
    """Return the GPU kernel for meshblock-grouped coarse alignment cubes."""
    mod = _get_module()
    return mod.get_function('mc_align_kernel_coarse') if mod is not None else None








def _iter_contiguous_block_runs(block_indices):
    """Yield half-open contiguous global meshblock-index runs."""
    if not block_indices:
        return
    ordered = sorted({int(m) for m in block_indices})
    first = previous = ordered[0]
    for block in ordered[1:]:
        if block == previous + 1:
            previous = block
            continue
        yield first, previous + 1
        first = previous = block
    yield first, previous + 1


def _alignment_batch_fields(ad, block_indices, iso_field, vector_fields,
                            log_iso_field=False):
    """Load a scalar and vector field using contiguous ``ad.data`` calls.

    The returned arrays stay on the GPU.  ``ad.data`` is deliberately used for
    both derived and native fields: it is the common data-function interface
    and avoids a Python/HDF5 load for every block.
    """
    fields = {}
    for mbl, mbh in _iter_contiguous_block_runs(block_indices):
        scalar = xp.asarray(ad.data(iso_field, mbl=mbl, mbh=mbh), dtype=xp.float64)
        fields[(mbl, mbh)] = {
            'scalar': xp.log10(scalar) if log_iso_field else scalar,
            'vector0': xp.asarray(ad.data(vector_fields[0], mbl=mbl, mbh=mbh), dtype=xp.float64),
            'vector1': xp.asarray(ad.data(vector_fields[1], mbl=mbl, mbh=mbh), dtype=xp.float64),
            'vector2': xp.asarray(ad.data(vector_fields[2], mbl=mbl, mbh=mbh), dtype=xp.float64),
        }
    return fields


def _alignment_gather_blocks(field_runs, indices, field):
    """Gather selected meshblocks from contiguous-range GPU loads."""
    pieces = []
    for index in indices:
        for (mbl, mbh), values in field_runs.items():
            if mbl <= index < mbh:
                pieces.append(values[field][index - mbl])
                break
        else:
            raise RuntimeError(f'alignment batch omitted required meshblock {index}')
    return xp.stack(pieces, axis=0)


def _build_alignment_padded_batch(field_runs, centers, neighbors, s_ghost, field):
    """Build GPU-resident padded arrays for a batch of center meshblocks."""
    nx3, nx2, nx1 = next(iter(field_runs.values()))[field].shape[1:]
    ncenters = len(centers)
    out = xp.empty((ncenters, nx3 + s_ghost, nx2 + s_ghost, nx1 + s_ghost),
                   dtype=xp.float64)

    out[:, :nx3, :nx2, :nx1] = _alignment_gather_blocks(field_runs, centers, field)
    for octant in _GHOST_OCTANTS:
        sources = [_octant_source(neighbors, m, octant) for m in centers]
        rows = [i for i, b in enumerate(sources) if b is not None]
        if not rows:
            continue
        values = _alignment_gather_blocks(
            field_runs, [sources[i] for i in rows], field)
        dst, src = _octant_slices(octant, s_ghost, nx3, nx2, nx1)
        if len(rows) == ncenters:
            out[(slice(None),) + dst] = values[(slice(None),) + src]
        else:
            out[(rows,) + dst] = values[(slice(None),) + src]

    for i, m in enumerate(centers):
        missing = tuple(n is None for n in neighbors[m])
        if any(missing):
            _replicate_missing_ghosts(out[i], missing, nx3, nx2, nx1)
    return out


def _alignment_required_blocks(centers, neighbors):
    """Return center blocks and the +direction blocks needed for their padding."""
    required = set()
    for m in centers:
        required.add(m)
        for octant in _GHOST_OCTANTS:
            b = _octant_source(neighbors, m, octant)
            if b is not None:
                required.add(b)
    return sorted(required)


def _release_alignment_gpu_memory():
    """Return unused batch allocations before the next (possibly larger) phase."""
    if cupy_enabled:
        xp.cuda.Stream.null.synchronize()
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


def _alignment_gpu_budget_bytes():
    """Return a conservative usable allocation budget for alignment batches.

    The value deliberately depends on the *currently free* device memory, not
    on an assumed GPU model.  This lets the calculation coexist with another
    process and makes the batching safe for meshblocks of any shape.
    """
    if not cupy_enabled:
        return 0
    free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    # Reserve half of the currently free memory for CUDA/CuPy workspaces and
    # allocations owned by ad.data().  A fractional reserve scales with the
    # device capacity and its current level of sharing.
    del total_bytes
    return max(0, int(0.50 * free_bytes))


def _alignment_blocks_per_batch(nx3, nx2, nx1, max_ghost, max_blocks=None):
    """Choose a local alignment batch from actual block and GPU memory sizes.

    Four double-precision input fields are resident, together with four
    padded fields and their gather temporaries.  This is an intentionally
    conservative upper bound so a smaller launch is selected before an
    allocation failure can abort a snapshot.
    """
    cells = int(nx3) * int(nx2) * int(nx1)
    padded_cells = (int(nx3) + int(max_ghost)) * (int(nx2) + int(max_ghost)) \
        * (int(nx1) + int(max_ghost))
    # The largest padding stencil uses eight neighbour blocks per centre.  The
    # factor accounts for stack/gather copies, the CuPy allocator's size
    # classes, and work arrays associated with all requested smoothing scales.
    # It is deliberately measured in bytes per *actual* block, not calibrated
    # to a particular mesh layout.
    bytes_per_block = int(2.5 * 4 * 8 * (8 * cells + padded_cells))
    budget = _alignment_gpu_budget_bytes()
    if budget <= 0:
        return 1
    result = max(1, budget // max(1, bytes_per_block))
    return min(int(max_blocks), result) if max_blocks is not None else result


def _alignment_coarse_groups_per_batch(nx3, nx2, nx1, blocks_per_group,
                                        max_groups=None):
    """Choose a meshblock-spanning-scale batch from actual array sizes."""
    cells = int(nx3) * int(nx2) * int(nx1)
    # Four resident double fields plus one full-group gather used by the
    # sequential vector reductions.  Extra headroom covers non-contiguous
    # ad.data() runs and CuPy's reduction workspace.
    # A cube also needs its eight vertices; at small grouped scales those can
    # lie outside the blocks used for the vector average, so include them in
    # the bound as well.
    resident_blocks = int(blocks_per_group) + 8
    bytes_per_group = int(1.5 * 5 * resident_blocks * cells * 8)
    budget = _alignment_gpu_budget_bytes()
    if budget <= 0:
        return 1
    result = max(1, budget // max(1, bytes_per_group))
    return min(int(max_groups), result) if max_groups is not None else result


def _alignment_cpu_triangles(verts, faces, bx, by, bz):
    """Return area and area-weighted alignment from CPU MC triangles.

    ``verts`` use scikit-image's (x3, x2, x1) ordering, whereas Athena's
    bcc1/2/3 are (x1, x2, x3), hence the deliberately reversed normal indices.
    """
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    area = 0.5 * norm
    valid = norm > 1.0e-30
    if not np.any(valid):
        return 0.0, 0.0
    normal = normal[valid]
    norm = norm[valid]
    area = area[valid]
    bx, by, bz = bx[valid], by[valid], bz[valid]
    bnorm = np.sqrt(bx*bx + by*by + bz*bz)
    valid_b = bnorm > 1.0e-30
    dot = (bx[valid_b] * normal[valid_b, 2]
           + by[valid_b] * normal[valid_b, 1]
           + bz[valid_b] * normal[valid_b, 0]) / (bnorm[valid_b] * norm[valid_b])
    return float(area.sum()), float(np.sum(area[valid_b] * dot * dot))


def _alignment_cpu_block(log_pad, bx_pad, by_pad, bz_pad, iso, s, dz, dy, dx,
                         nx3, nx2, nx1):
    """CPU fallback for one padded meshblock at one local smoothing scale."""
    coarse = np.asarray(log_pad[::s, ::s, ::s], dtype=np.float64)
    if coarse.max() <= iso or coarse.min() >= iso:
        return 0.0, 0.0
    verts, faces, _, _ = skimage.measure.marching_cubes(
        coarse, level=iso, spacing=(s * dz, s * dy, s * dx), allow_degenerate=False)
    centroid = np.mean(verts[faces], axis=1)
    kk = np.clip((centroid[:, 0] / (s * dz)).astype(int), 0, nx3 // s - 1)
    jj = np.clip((centroid[:, 1] / (s * dy)).astype(int), 0, nx2 // s - 1)
    ii = np.clip((centroid[:, 2] / (s * dx)).astype(int), 0, nx1 // s - 1)

    def coarse_b(field):
        core = np.asarray(field[:nx3, :nx2, :nx1], dtype=np.float64)
        return core.reshape(nx3 // s, s, nx2 // s, s, nx1 // s, s).mean(axis=(1, 3, 5))

    bxc, byc, bzc = coarse_b(bx_pad), coarse_b(by_pad), coarse_b(bz_pad)
    return _alignment_cpu_triangles(verts, faces, bxc[kk, jj, ii], byc[kk, jj, ii], bzc[kk, jj, ii])


def _calc_vector_nhat_alignment_cpu_local(ad, iso, centers, neighbors, step_sizes,
                                          dz, dy, dx, iso_field, vector_fields,
                                          log_iso_field):
    """CPU fallback using the same batched ``ad.data`` and padding path."""
    nx3, nx2, nx1 = ad.nx3, ad.nx2, ad.nx1
    area = {s: 0.0 for s in step_sizes}
    align = {s: 0.0 for s in step_sizes}
    blocks_per_batch = 4
    for first in range(0, len(centers), blocks_per_batch):
        batch = [int(m) for m in centers[first:first + blocks_per_batch]]
        fields = _alignment_batch_fields(ad, _alignment_required_blocks(batch, neighbors),
                                         iso_field, vector_fields, log_iso_field)
        for s in step_sizes:
            log = asnumpy(_build_alignment_padded_batch(fields, batch, neighbors, s, 'scalar'))
            bx = asnumpy(_build_alignment_padded_batch(fields, batch, neighbors, s, 'vector0'))
            by = asnumpy(_build_alignment_padded_batch(fields, batch, neighbors, s, 'vector1'))
            bz = asnumpy(_build_alignment_padded_batch(fields, batch, neighbors, s, 'vector2'))
            for index in range(len(batch)):
                da, dalign = _alignment_cpu_block(log[index], bx[index], by[index], bz[index],
                                                   iso, s, dz, dy, dx, nx3, nx2, nx1)
                area[s] += da
                align[s] += dalign
    return area, align


def _find_interface_z_levels_for_field(ad, iso_field, iso_value, log_iso_field,
                                       n_buffer=1):
    """Find the interface slab for an arbitrary scalar through ``ad.data``.

    Same selection rule as :func:`find_interface_z_levels`.
    """
    zstarts = np.sort(np.unique(ad.mb_geometry[:, 4]))
    profile = np.empty(len(zstarts), dtype=float)
    level_min = np.empty(len(zstarts), dtype=float)
    level_max = np.empty(len(zstarts), dtype=float)
    for iz, zstart in enumerate(zstarts):
        blocks = np.where(ad.mb_geometry[:, 4] == zstart)[0].tolist()
        total = 0.0
        count = 0
        vmin, vmax = np.inf, -np.inf
        for mbl, mbh in _iter_contiguous_block_runs(blocks):
            values = xp.asarray(ad.data(iso_field, mbl=mbl, mbh=mbh), dtype=xp.float64)
            if log_iso_field:
                values = xp.log10(values)
            total += float(asnumpy(xp.sum(values)))
            count += int(values.size)
            vmin = min(vmin, float(asnumpy(xp.nanmin(values))))
            vmax = max(vmax, float(asnumpy(xp.nanmax(values))))
        profile[iz] = total / count
        level_min[iz] = vmin
        level_max[iz] = vmax
    target = float(np.log10(iso_value)) if log_iso_field else float(iso_value)
    lo, hi, crossing, _ = _select_interface_levels(
        profile, level_min, level_max, target, n_buffer,
        _get_periodic_flags(ad)[2], label=f'{iso_field}={iso_value:.4g}')
    return zstarts[lo:hi + 1], crossing


def _calc_vector_nhat_alignment_grouped_mb(ad, iso, z_selected, step_sizes,
                                            iso_field, vector_fields,
                                            log_iso_field):
    """Evaluate meshblock-spanning scales with batched GPU data/reductions."""
    nx3, nx2, nx1 = ad.nx3, ad.nx2, ad.nx1
    if any(s % nx1 or s % nx2 or s % nx3 for s in step_sizes):
        raise ValueError(
            'grouped alignment scales must be integer multiples of each '
            f'meshblock dimension {(nx1, nx2, nx3)}: {step_sizes}'
        )

    kernel = _get_kernel_align_coarse()
    if kernel is None:
        raise RuntimeError('mc_align_kernel_coarse GPU kernel not available.')

    nblk1, nblk2, nblk3 = ad.Nx1 // nx1, ad.Nx2 // nx2, ad.Nx3 // nx3
    lookup = _build_meshblock_lookup(ad)
    periodic_x, periodic_y, periodic_z = _get_periodic_flags(ad)
    zstarts = np.sort(np.unique(ad.mb_geometry[:, 4]))
    slab_bz = set(np.where(np.isin(zstarts, z_selected))[0].tolist())
    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1
    out = {}

    for s in step_sizes:
        q1, q2, q3 = s // nx1, s // nx2, s // nx3
        cube_starts = []
        for bk in range(0, nblk3, q3):
            if not slab_bz.intersection(range(bk, bk + q3)):
                continue
            for bj in range(0, nblk2, q2):
                for bi in range(0, nblk1, q1):
                    gi, gj, gk = bi * nx1, bj * nx2, bk * nx3
                    cube_starts.append((gi, gj, gk))

        if not cube_starts:
            out[s] = (0.0, 0.0)
            continue

        # This scale covers all blocks in each selected coarse cube.  A single
        # contiguous range load is much faster than block-at-a-time H2D copies;
        # the box averages are reductions on the GPU, not host ``.sum`` calls.
        logical_to_mb = lookup
        vertex_mb, vertex_k, vertex_j, vertex_i, group_mb = [], [], [], [], []
        for gi, gj, gk in cube_starts:
            group = []
            for bk in range(gk // nx3, gk // nx3 + q3):
                for bj in range(gj // nx2, gj // nx2 + q2):
                    for bi in range(gi // nx1, gi // nx1 + q1):
                        group.append(logical_to_mb[(bi, bj, bk)])
            group_mb.append(group)
            for dk, dj, di in ((0, 0, 0), (0, 0, s), (0, s, s), (0, s, 0),
                               (s, 0, 0), (s, 0, s), (s, s, s), (s, s, 0)):
                ii = _apply_bc_index(gi + di, ad.Nx1, periodic_x)
                jj = _apply_bc_index(gj + dj, ad.Nx2, periodic_y)
                kk = _apply_bc_index(gk + dk, ad.Nx3, periodic_z)
                bi, li = divmod(ii, nx1)
                bj, lj = divmod(jj, nx2)
                bk, lk = divmod(kk, nx3)
                vertex_mb.append(logical_to_mb[(bi, bj, bk)])
                vertex_k.append(lk)
                vertex_j.append(lj)
                vertex_i.append(li)

        # Loading every selected coarse cube at once can make the resident
        # field arrays and indexed temporaries exceed device memory.  Chunk
        # coarse cubes according to their true block shape and free each chunk
        # before the next one.
        # Each chunk still uses contiguous ad.data() ranges and one flattened
        # kernel launch, rather than a meshblock-at-a-time data path.
        groups_per_batch = _alignment_coarse_groups_per_batch(
            nx3, nx2, nx1, max(len(group) for group in group_mb))
        total_area = 0.0
        total_align = 0.0
        for first in range(0, len(cube_starts), groups_per_batch):
            last = min(first + groups_per_batch, len(cube_starts))
            batch_groups = group_mb[first:last]
            batch_vertex_mb = vertex_mb[8 * first:8 * last]
            batch_vertex_k = vertex_k[8 * first:8 * last]
            batch_vertex_j = vertex_j[8 * first:8 * last]
            batch_vertex_i = vertex_i[8 * first:8 * last]
            blocks = sorted({m for group in batch_groups for m in group}
                            | set(batch_vertex_mb))
            field_runs = _alignment_batch_fields(
                ad, blocks, iso_field, vector_fields, log_iso_field)
            temp = _alignment_gather_blocks(field_runs, batch_vertex_mb, 'scalar')
            values = temp[xp.arange(len(batch_vertex_mb)), xp.asarray(batch_vertex_k),
                          xp.asarray(batch_vertex_j), xp.asarray(batch_vertex_i)]
            cubes_gpu = values.reshape(last - first, 8) - iso

            # Materialize only one component's group gather at a time.  Its
            # lifetime ends before the next component is reduced.
            def reduce_groups(field):
                means = []
                for group in batch_groups:
                    group_values = _alignment_gather_blocks(field_runs, group, field)
                    means.append(xp.mean(group_values, axis=(0, 1, 2, 3)))
                    del group_values
                    _release_alignment_gpu_memory()
                return xp.stack(means)

            bx_gpu = reduce_groups('vector0')
            by_gpu = reduce_groups('vector1')
            bz_gpu = reduce_groups('vector2')
            area_gpu = xp.zeros(1, dtype=xp.float64)
            align_gpu = xp.zeros(1, dtype=xp.float64)
            ncubes = last - first
            kernel(
                ((ncubes + 127) // 128,), (128,),
                (cubes_gpu, bx_gpu, by_gpu, bz_gpu, area_gpu, align_gpu,
                 np.int32(ncubes), np.float64(s * dz), np.float64(s * dy), np.float64(s * dx)),
            )
            total_area += float(asnumpy(area_gpu)[0])
            total_align += float(asnumpy(align_gpu)[0])
            del field_runs, temp, values, cubes_gpu, bx_gpu, by_gpu, bz_gpu, area_gpu, align_gpu
            _release_alignment_gpu_memory()
        out[s] = (total_area, total_align)

    return out


def _alignment_results_finite(area_list, cos2_list):
    """
    True when every area is finite and non-negative and every mean cos^2 is
    finite, except the documented NaN for a scale with zero area.
    """
    area = np.asarray(area_list, dtype=float).ravel()
    cos2 = np.asarray(cos2_list, dtype=float).ravel()
    if area.shape != cos2.shape:
        return False
    if not np.all(np.isfinite(area)) or np.any(area < 0.0):
        return False
    return bool(np.all(np.isfinite(cos2[area > 0.0])))


def _check_alignment_results_finite(step_sizes, area_list, cos2_list):
    """Refuse to cache non-finite alignment results."""
    if not _alignment_results_finite(area_list, cos2_list):
        raise RuntimeError(
            'Alignment produced non-finite values; refusing to cache them: '
            f'step_sizes={list(step_sizes)}, total_area={list(area_list)}, '
            f'mean_cos2={list(cos2_list)}'
        )


def calc_vector_nhat_alignment_mb_based(ad, iso_value, iso_field, vector_fields,
                                        n_mb_buffer=1, step_sizes=None,
                                        log_iso_field=False, use_gpu=True,
                                        verbose=True, mpi_manager=None,
                                        redo=False, cache_attr='vector_nhat_alignment'):
    """
    Compute area-weighted <(v̂·n̂)²> for a scalar isosurface and vector field.

    Requires a uniform mesh; the periodicity of each axis is read from the
    header (non-periodic + boundaries replicate the last layer).
    MPI work sharing requires a full-domain input on every rank.

    Processes batches of slab meshblocks.  For each batch and step size s:
      1. Load ``iso_field`` and ``vector_fields`` via contiguous-range ``ad.data`` calls.
      2. Build GPU-resident padded blocks (ghost cells from neighbours).
      3. Run one flattened CUDA kernel which, for every MC triangle at scale s:
           - n̂: from the cross product of the coarsened triangle edges
           - v̂: from the box average of the vector over the s×s×s cell block
         and accumulates both area and area*(v̂·n̂)² atomically.

    v̂ = the supplied three-component vector, box-averaged at scale s.
    n̂ = unit normal to the triangle, at spatial scale s*dx.
    Isotropic (random) expectation: <(v̂·n̂)²> = 1/3.

    Results are cached in ``getattr(ad, cache_attr)`` after computation.  The caller is
    responsible for calling ad.save() to persist to the h5data file.

    Parameters
    ----------
    ad          : AthenaData
    iso_value   : float
    iso_field   : str
                 scalar field name accepted by ``ad.data``.
    vector_fields : sequence of three str
                 x1/x2/x3 component field names accepted by ``ad.data``.
    n_mb_buffer : int         extra meshblock z-levels around the interface slab
    step_sizes  : list of int  smoothing scales (default [1] = native resolution)
    log_iso_field : bool
                 take log10 of the scalar and iso_value before marching cubes.
    use_gpu     : bool
    verbose     : bool
    mpi_manager : MPIManager or None
    redo        : bool        ignore cached ad.align_mc and recompute

    Returns
    -------
    mean_cos2  : float   area-weighted <(v̂·n̂)²> at step_sizes[0]  (nan if no isosurface)
    total_area : float   total isosurface area at step_sizes[0]

    Full multi-scale results are stored under ``cache_attr``.
    """
    if not isinstance(iso_field, str):
        raise TypeError('iso_field must be an ad.data field name')
    if len(vector_fields) != 3 or not all(isinstance(v, str) for v in vector_fields):
        raise ValueError('vector_fields must contain exactly three ad.data field names')
    vector_fields = tuple(vector_fields)
    if step_sizes is None:
        step_sizes = [1]
    step_sizes = sorted(set(int(s) for s in step_sizes))
    if not step_sizes or any(s <= 0 for s in step_sizes):
        raise ValueError('step_sizes must contain positive integers')
    if not getattr(ad, 'has_full_data', True):
        raise ValueError('alignment requires all meshblocks on each rank, including neighbor halos')

    rank = mpi_manager.rank if mpi_manager is not None else 0

    # Return cached result if available and compatible.
    if not redo:
        cached = getattr(ad, cache_attr, None)
        if (cached is not None
                and np.isclose(float(cached.get('iso_value', cached.get('T_peak', np.nan))), float(iso_value))
                and int(cached.get('n_mb_buffer', -1)) == int(n_mb_buffer)
                and list(cached.get('step_sizes', [])) == step_sizes
                and cached.get('iso_field', iso_field) == iso_field
                and tuple(cached.get('vector_fields', vector_fields)) == vector_fields
                and bool(cached.get('log_iso_field', log_iso_field)) == bool(log_iso_field)
                and _alignment_results_finite(cached.get('total_area', []),
                                              cached.get('mean_cos2', []))):
            if rank == 0 and verbose:
                print('  Using cached alignment from ad.align_mc')
            cos2_list = cached['mean_cos2']
            area_list = cached['total_area']
            return float(cos2_list[0]), float(area_list[0])

    n_mb = ad.n_mbs
    nx3, nx2, nx1 = ad.nx3, ad.nx2, ad.nx1

    dz = (ad.x3max - ad.x3min) / ad.Nx3
    dy = (ad.x2max - ad.x2min) / ad.Nx2
    dx = (ad.x1max - ad.x1min) / ad.Nx1
    iso = float(np.log10(iso_value)) if log_iso_field else float(iso_value)

    use_kernel = use_gpu and cupy_enabled and _get_module() is not None

    kernel = _get_kernel_align_mb_batch() if use_kernel else None
    if use_kernel and kernel is None:
        raise RuntimeError('mc_align_kernel_mb_batch GPU kernel not available.')

    # Retain the established temperature slab selector bit-for-bit for the
    # legacy B/T wrapper.  Other fields use the same z-profile criterion via
    # contiguous-range ``ad.data`` loads.
    if iso_field == 'temp' and log_iso_field:
        z_selected, iz_cross = find_interface_z_levels(ad, iso_value, n_buffer=n_mb_buffer)
    else:
        z_selected, iz_cross = _find_interface_z_levels_for_field(
            ad, iso_field, iso_value, log_iso_field, n_buffer=n_mb_buffer)
    slab_indices = np.where(np.isin(ad.mb_geometry[:, 4], z_selected))[0]
    n_slab = len(slab_indices)

    if mpi_manager is not None and mpi_manager.size > 1:
        my_slab = slab_indices[rank::mpi_manager.size]
    else:
        my_slab = slab_indices

    if rank == 0 and verbose:
        print(
            f'  Vector-normal alignment: {n_slab}/{n_mb} slab meshblocks '
            f'(z-level {iz_cross}, buffer={n_mb_buffer}), '
            f'step_sizes={step_sizes}'
        )

    neighbors = _find_mc_neighbors(ad)
    # Accumulators for each step size.
    results_area  = {s: 0.0 for s in step_sizes}
    results_align = {s: 0.0 for s in step_sizes}
    # A meshblock-local scale must tile every local dimension.  Otherwise the
    # integer cube counts in the CUDA kernel would drop remainder cells.  A
    # non-local scale is handled below only when it is an integral multiple of
    # the full meshblock shape; all other requests are rejected there.
    local_steps = [
        s for s in step_sizes
        if s <= min(nx1, nx2, nx3)
        and nx1 % s == 0 and nx2 % s == 0 and nx3 % s == 0
    ]
    grouped_steps = [s for s in step_sizes if s not in local_steps]
    if grouped_steps and mpi_manager is not None and mpi_manager.size > 1:
        raise RuntimeError(
            'meshblock-grouped alignment scales require one rank; distribute '
            'snapshots, not the same grouped coarse cubes, across ranks.'
        )
    if not use_kernel:
        if grouped_steps:
            raise RuntimeError(
                'The portable CPU alignment backend currently supports step sizes '
                f'up to one meshblock ({min(nx1, nx2, nx3)} cells).'
            )
        if rank == 0 and verbose:
            print('  GPU unavailable; using the portable scikit-image CPU alignment backend.')
        results_area, results_align = _calc_vector_nhat_alignment_cpu_local(
            ad, iso, my_slab, neighbors, local_steps, dz, dy, dx,
            iso_field, vector_fields, log_iso_field)
        if mpi_manager is not None and mpi_manager.size > 1:
            for s in step_sizes:
                results_area[s] = mpi_manager.comm.allreduce(results_area[s], op=mpi_manager.MPI.SUM)
                results_align[s] = mpi_manager.comm.allreduce(results_align[s], op=mpi_manager.MPI.SUM)
        cos2_list = [float(results_align[s] / results_area[s]) if results_area[s] else np.nan
                     for s in step_sizes]
        area_list = [float(results_area[s]) for s in step_sizes]
        _check_alignment_results_finite(step_sizes, area_list, cos2_list)
        setattr(ad, cache_attr, {
            'step_sizes': step_sizes, 'mean_cos2': cos2_list, 'total_area': area_list,
            'iso_value': float(iso_value), 'iso_field': iso_field,
            'vector_fields': list(vector_fields), 'log_iso_field': bool(log_iso_field),
            'n_mb_buffer': int(n_mb_buffer),
        })
        return cos2_list[0], area_list[0]
    # Keep one device accumulator per scale for the entire slab.  Blocks are
    # loaded in contiguous ``ad.data`` ranges and a single flat kernel launch
    # covers every cube in each GPU-resident batch.
    local_area_gpu = {s: xp.zeros(1, dtype=xp.float64) for s in local_steps}
    local_align_gpu = {s: xp.zeros(1, dtype=xp.float64) for s in local_steps}
    # The launch size follows both the actual meshblock dimensions and the
    # currently available device memory.  This supports arbitrary block shapes
    # and device capacities without a machine-specific tuning constant.
    blocks_per_batch = _alignment_blocks_per_batch(
        nx3, nx2, nx1, max(local_steps, default=1))
    if rank == 0 and verbose:
        print(f'  Alignment batch size: {blocks_per_batch} meshblocks '
              f'(block shape {nx3}x{nx2}x{nx1})')
    n_active = 0
    for batch_start in range(0, len(my_slab), blocks_per_batch):
        centers = [int(m) for m in my_slab[batch_start:batch_start + blocks_per_batch]]
        field_runs = _alignment_batch_fields(
            ad, _alignment_required_blocks(centers, neighbors),
            iso_field, vector_fields, log_iso_field)
        center_log = _alignment_gather_blocks(field_runs, centers, 'scalar')
        active = xp.any(center_log < iso, axis=(1, 2, 3)) & xp.any(
            center_log > iso, axis=(1, 2, 3))
        n_active += int(asnumpy(xp.count_nonzero(active)))

        for s in local_steps:
            log_gpu = _build_alignment_padded_batch(
                field_runs, centers, neighbors, s, 'scalar')
            bx_gpu = _build_alignment_padded_batch(
                field_runs, centers, neighbors, s, 'vector0')
            by_gpu = _build_alignment_padded_batch(
                field_runs, centers, neighbors, s, 'vector1')
            bz_gpu = _build_alignment_padded_batch(
                field_runs, centers, neighbors, s, 'vector2')
            cubes_per_block = (nx1 // s) * (nx2 // s) * (nx3 // s)
            nthreads = len(centers) * cubes_per_block
            kernel(
                ((nthreads + 255) // 256,), (256,),
                (
                    log_gpu, bx_gpu, by_gpu, bz_gpu,
                    local_area_gpu[s], local_align_gpu[s],
                    np.int32(len(centers)), np.int32(nx3), np.int32(nx2), np.int32(nx1),
                    np.int32(s), np.int32(s), np.float64(iso),
                    np.float64(dz), np.float64(dy), np.float64(dx),
                ),
            )
            # Every smoothing scale has a distinct padded shape.  Synchronize
            # and release it now, instead of allowing the CuPy pool to retain
            # all of those size classes until the end of the batch.  The two
            # scalar accumulators remain resident, so this changes neither the
            # arithmetic nor the one-flat-launch-per-scale design.
            del log_gpu, bx_gpu, by_gpu, bz_gpu
            _release_alignment_gpu_memory()

        if rank == 0 and verbose:
            print(f'    queued {min(batch_start + len(centers), len(my_slab)):4d}/{len(my_slab)} slab blocks')
        del field_runs, center_log, active
        _release_alignment_gpu_memory()

    # Exactly one synchronization per smoothing scale after all slab kernels
    # have been enqueued.
    for s in local_steps:
        results_area[s] += float(asnumpy(local_area_gpu[s])[0])
        results_align[s] += float(asnumpy(local_align_gpu[s])[0])

    # Coarse scales spanning several meshblocks use globally aligned cubes,
    # avoiding both zero-cube launches and duplicate inter-meshblock faces.
    if grouped_steps:
        grouped = _calc_vector_nhat_alignment_grouped_mb(
            ad, iso, z_selected, grouped_steps, iso_field, vector_fields, log_iso_field,
        )
        for s, (area, alignment) in grouped.items():
            results_area[s] += area
            results_align[s] += alignment

    # MPI reduction across ranks.
    if mpi_manager is not None and mpi_manager.size > 1:
        for s in step_sizes:
            results_area[s]  = mpi_manager.comm.allreduce(results_area[s],  op=mpi_manager.MPI.SUM)
            results_align[s] = mpi_manager.comm.allreduce(results_align[s], op=mpi_manager.MPI.SUM)

    cos2_list = []
    area_list = []
    for s in step_sizes:
        tot_area  = results_area[s]
        tot_align = results_align[s]
        mc2 = (tot_align / tot_area) if tot_area > 0.0 else np.nan
        cos2_list.append(float(mc2))
        area_list.append(float(tot_area))

    if rank == 0 and verbose:
        for s, mc2, tot_area in zip(step_sizes, cos2_list, area_list):
            print(
                f'    s={s}: {n_active}/{len(my_slab)} active; '
                f'area={tot_area:.6g}  <(v̂·n̂)²>={mc2:.4f}'
            )

    _check_alignment_results_finite(step_sizes, area_list, cos2_list)
    setattr(ad, cache_attr, {
        'step_sizes':  step_sizes,
        'mean_cos2':   cos2_list,
        'total_area':  area_list,
        'iso_value':   float(iso_value),
        'iso_field':   iso_field,
        'vector_fields': list(vector_fields),
        'log_iso_field': bool(log_iso_field),
        'n_mb_buffer': int(n_mb_buffer),
    })

    return cos2_list[0], area_list[0]


def calc_bhat_nhat_alignment_mb_based(ad, T_peak, n_mb_buffer=1,
                                      step_sizes=None, use_gpu=True,
                                      verbose=True, mpi_manager=None,
                                      redo=False):
    """Backward-compatible T=T_peak magnetic-field alignment wrapper.

    New callers needing another scalar isosurface or vector should use
    :func:`calc_vector_nhat_alignment_mb_based` directly.
    """
    result = calc_vector_nhat_alignment_mb_based(
        ad, T_peak, iso_field='temp', vector_fields=('bcc1', 'bcc2', 'bcc3'),
        n_mb_buffer=n_mb_buffer, step_sizes=step_sizes, log_iso_field=True,
        use_gpu=use_gpu, verbose=verbose, mpi_manager=mpi_manager, redo=redo,
        cache_attr='align_mc')
    # Preserve the pre-generalization cache schema consumed by the plotting
    # script, while retaining the explicit generic-field metadata.
    ad.align_mc['T_peak'] = float(T_peak)
    return result
