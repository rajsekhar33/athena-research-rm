"""
Particle I/O functions for AthenaData.

This module provides functions to read particle data from various formats:
- Binary particle files (.prtclbin)
- HDF5 particle files (.h5)
- VTK particle files (.vtk)
"""

import re
import numpy as np
import h5py
from pathlib import Path


_PARTICLE_MAGIC_NUMBER = 43


def read_particle_binary_header(filename):
    """
    Read only the header metadata from an AthenaK ``.prtclbin`` file.

    Parameters
    ----------
    filename : str or Path
        Path to the binary particle file.

    Returns
    -------
    dict
        Header dictionary with simulation time, cycle, particle counts, and
        particle-grid variable names.
    """
    filename = Path(filename)
    with filename.open('rb') as f:
        header_int = np.fromfile(f, dtype=np.int64, count=5)
        if len(header_int) != 5:
            raise ValueError(f"{filename} is too short to be a particle binary")

        magic_number, nparticles, nrdata, nidata, ngriddata = header_int
        if magic_number != _PARTICLE_MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number {magic_number}, "
                f"expected {_PARTICLE_MAGIC_NUMBER}"
            )

        header_real = np.fromfile(f, dtype=np.float64, count=3)
        if len(header_real) != 3:
            raise ValueError(f"{filename} is missing particle time metadata")
        time, dt, ncycle = header_real

        var_names = []
        for _ in range(int(ngriddata)):
            varname_bytes = f.read(16)
            var_names.append(varname_bytes.decode('utf-8').rstrip('\x00'))

    return {
        'time': float(time),
        'dt': float(dt),
        'ncycle': int(ncycle),
        'nparticles': int(nparticles),
        'nrdata': int(nrdata),
        'nidata': int(nidata),
        'ngriddata': int(ngriddata),
        'var_names': var_names,
    }


def read_particle_binary_positions(filename):
    """
    Read only x/y/z particle positions from an AthenaK ``.prtclbin`` file.

    This is useful for large outputs where deposition or plotting only needs
    particle positions and loading velocities, integer fields, and sampled grid
    quantities would waste memory.

    Parameters
    ----------
    filename : str or Path
        Path to the binary particle file.

    Returns
    -------
    dict
        Header fields plus ``x``, ``y``, and ``z`` position arrays.
    """
    filename = Path(filename)
    with filename.open('rb') as f:
        header_int = np.fromfile(f, dtype=np.int64, count=5)
        if len(header_int) != 5:
            raise ValueError(f"{filename} is too short to be a particle binary")

        magic_number, nparticles, nrdata, nidata, ngriddata = header_int
        nparticles = int(nparticles)
        nrdata = int(nrdata)
        nidata = int(nidata)
        ngriddata = int(ngriddata)
        if magic_number != _PARTICLE_MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number {magic_number}, "
                f"expected {_PARTICLE_MAGIC_NUMBER}"
            )
        if nrdata < 3:
            raise ValueError(f"{filename} has nrdata={nrdata}; need x/y/z")

        header_real = np.fromfile(f, dtype=np.float64, count=3)
        if len(header_real) != 3:
            raise ValueError(f"{filename} is missing particle time metadata")
        time, dt, ncycle = header_real

        var_names = []
        for _ in range(ngriddata):
            varname_bytes = f.read(16)
            var_names.append(varname_bytes.decode('utf-8').rstrip('\x00'))

        x = np.fromfile(f, dtype=np.float64, count=nparticles)
        y = np.fromfile(f, dtype=np.float64, count=nparticles)
        z = np.fromfile(f, dtype=np.float64, count=nparticles)
        if len(x) != nparticles or len(y) != nparticles or len(z) != nparticles:
            raise ValueError(f"{filename} ended before all particle positions")

    return {
        'time': float(time),
        'dt': float(dt),
        'ncycle': int(ncycle),
        'nparticles': nparticles,
        'nrdata': nrdata,
        'nidata': nidata,
        'ngriddata': ngriddata,
        'var_names': var_names,
        'x': x,
        'y': y,
        'z': z,
    }


def wrap_particle_positions(values, xmin, xmax):
    """
    Wrap particle coordinates into the periodic interval [xmin, xmax).
    """
    span = xmax - xmin
    return ((np.asarray(values) - xmin) % span) + xmin


def _normalize_mean(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    if not np.isfinite(mean) or mean == 0.0:
        return np.zeros_like(values, dtype=float)
    return values / mean


def particle_density_grid(particles, grid, normalize=True):
    """
    Deposit particles into the cell grid described by an AthenaK binary dump.

    This helper uses nearest-cell histogram deposition. It is intended for fast
    particle-count diagnostics and tracer/gas comparison plots.

    Parameters
    ----------
    particles : dict
        Particle data containing ``x``, ``y``, and ``z`` arrays.
    grid : dict
        AthenaK grid dictionary returned by ``athena_research.core.io_utils.read_binary``.
    normalize : bool, optional
        If True, divide the deposited counts by their mean.

    Returns
    -------
    numpy.ndarray
        Particle count grid with shape ``(Nx3, Nx2, Nx1)``.
    """
    xedges = np.linspace(grid['x1min'], grid['x1max'], grid['Nx1'] + 1)
    yedges = np.linspace(grid['x2min'], grid['x2max'], grid['Nx2'] + 1)
    zedges = np.linspace(grid['x3min'], grid['x3max'], grid['Nx3'] + 1)
    x = wrap_particle_positions(particles['x'], grid['x1min'], grid['x1max'])
    y = wrap_particle_positions(particles['y'], grid['x2min'], grid['x2max'])
    z = wrap_particle_positions(particles['z'], grid['x3min'], grid['x3max'])
    counts, _ = np.histogramdd((z, y, x), bins=(zedges, yedges, xedges))
    if normalize:
        counts = _normalize_mean(counts)
    return counts


def particle_column_xy(particles, grid, normalize=True):
    """
    Deposit particles onto an x-y column-density map.

    Parameters
    ----------
    particles : dict
        Particle data containing ``x`` and ``y`` arrays.
    grid : dict
        AthenaK grid dictionary returned by ``athena_research.core.io_utils.read_binary``.
    normalize : bool, optional
        If True, divide the deposited column by its mean.

    Returns
    -------
    numpy.ndarray
        Particle column map with shape ``(Nx2, Nx1)``.
    """
    xedges = np.linspace(grid['x1min'], grid['x1max'], grid['Nx1'] + 1)
    yedges = np.linspace(grid['x2min'], grid['x2max'], grid['Nx2'] + 1)
    x = wrap_particle_positions(particles['x'], grid['x1min'], grid['x1max'])
    y = wrap_particle_positions(particles['y'], grid['x2min'], grid['x2max'])
    counts, _, _ = np.histogram2d(y, x, bins=(yedges, xedges))
    if normalize:
        counts = _normalize_mean(counts)
    return counts


def particle_profile_x(particles, grid, normalize=True):
    """
    Deposit particles into a one-dimensional x profile.

    Parameters
    ----------
    particles : dict
        Particle data containing an ``x`` array.
    grid : dict
        AthenaK grid dictionary returned by ``athena_research.core.io_utils.read_binary``.
    normalize : bool, optional
        If True, divide the deposited profile by its mean.

    Returns
    -------
    numpy.ndarray
        Particle count profile with shape ``(Nx1,)``.
    """
    xedges = np.linspace(grid['x1min'], grid['x1max'], grid['Nx1'] + 1)
    x = wrap_particle_positions(particles['x'], grid['x1min'], grid['x1max'])
    counts, _ = np.histogram(x, bins=xedges)
    if normalize:
        counts = _normalize_mean(counts)
    return counts


def read_particle_binary(filename):
    """
    Read AthenaK binary particle output file (.prtclbin)
    
    Parameters
    ----------
    filename : str or Path
        Path to the binary particle file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'time': simulation time
        - 'dt': timestep
        - 'ncycle': cycle number
        - 'nparticles': total number of particles
        - 'nrdata': number of real data fields
        - 'nidata': number of integer data fields
        - 'ngriddata': number of grid data fields
        - 'var_names': list of grid variable names
        - 'x', 'y', 'z': particle positions
        - 'velx', 'vely', 'velz': particle velocities
        - 'gid': MeshBlock global IDs
        - 'tag': particle tags
        - <var_name>: grid data at particle locations
    """
    data = {}
    
    with open(filename, 'rb') as f:
        # Read header (5 int64_t + 3 double)
        header_int = np.fromfile(f, dtype=np.int64, count=5)
        magic_number = header_int[0]
        nparticles = header_int[1]
        nrdata = header_int[2]
        nidata = header_int[3]
        ngriddata = header_int[4]
        
        header_real = np.fromfile(f, dtype=np.float64, count=3)
        time = header_real[0]
        dt = header_real[1]
        ncycle = int(header_real[2])
        
        # Verify magic number
        if magic_number != 43:
            raise ValueError(f"Invalid magic number {magic_number}, expected 43")
        
        # Read variable names (16 bytes each)
        var_names = []
        for i in range(ngriddata):
            varname_bytes = f.read(16)
            varname = varname_bytes.decode('utf-8').rstrip('\x00')
            var_names.append(varname)
        
        # Store header info
        data['time'] = time
        data['dt'] = dt
        data['ncycle'] = ncycle
        data['nparticles'] = nparticles
        data['nrdata'] = nrdata
        data['nidata'] = nidata
        data['ngriddata'] = ngriddata
        data['var_names'] = var_names
        
        # Read particle real data
        rdata_names = ['x', 'y', 'z', 'velx', 'vely', 'velz']
        # Also check for alternative velocity names
        alt_rdata_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        
        for i in range(nrdata):
            rdata = np.fromfile(f, dtype=np.float64, count=nparticles)
            if i < len(rdata_names):
                data[rdata_names[i]] = rdata
                # Also store alternative names if they exist
                if i >= 3 and i < len(alt_rdata_names):
                    data[alt_rdata_names[i]] = rdata
        
        # Read particle integer data
        idata_names = ['gid', 'tag', 'lastmove', 'lastlevel']
        for i in range(nidata):
            idata = np.fromfile(f, dtype=np.int32, count=nparticles)
            if i < len(idata_names):
                data[idata_names[i]] = idata
            else:
                data[f'idata_{i}'] = idata
        
        # Read grid data at particle locations
        for i in range(ngriddata):
            griddata = np.fromfile(f, dtype=np.float64, count=nparticles)
            data[var_names[i]] = griddata
    
    return data


def read_particle_hdf5(filename):
    """
    Read AthenaK HDF5 particle output file (.h5)
    
    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 particle file
        
    Returns
    -------
    dict
        Dictionary with same structure as read_particle_binary
    """
    data = {}
    
    with h5py.File(filename, 'r') as f:
        # Read metadata
        metadata = f['Metadata']
        data['time'] = metadata.attrs['time']
        data['dt'] = metadata.attrs['dt']
        data['ncycle'] = metadata.attrs['ncycle']
        data['nparticles'] = metadata.attrs['nparticles']
        data['nrdata'] = metadata.attrs['nrdata']
        data['nidata'] = metadata.attrs['nidata']
        data['ngriddata'] = metadata.attrs['ngriddata']
        
        # Read variable names
        var_names_bytes = metadata['var_names'][:]
        data['var_names'] = [vn.decode('utf-8') for vn in var_names_bytes]
        
        # Read particle real data
        if 'ParticleRealData' in f:
            rdata_grp = f['ParticleRealData']
            for key in rdata_grp.keys():
                data[key] = rdata_grp[key][:]
        
        # Read particle integer data
        if 'ParticleIntData' in f:
            idata_grp = f['ParticleIntData']
            for key in idata_grp.keys():
                data[key] = idata_grp[key][:]
        
        # Read grid data
        if 'GridData' in f:
            griddata_grp = f['GridData']
            for key in griddata_grp.keys():
                data[key] = griddata_grp[key][:]
    
    return data


def read_particle_vtk(filename):
    """
    Read an AthenaK legacy VTK particle file.

    The layout (``outputs/vtk_prtcl.cpp``) is: a ``# vtk DataFile`` line, a
    comment line carrying ``time= <t>`` and ``cycle=<n>``, ``BINARY``,
    ``DATASET UNSTRUCTURED_GRID``, then ``POINTS <n> float`` followed by
    3n big-endian positions, ``POINT_DATA <n>``, and one
    ``SCALARS <name> <type>`` / ``LOOKUP_TABLE default`` record per
    integer particle field (``gid``, ``ptag``). AthenaK writes those
    integer fields as float32, so they are converted back to int64 here
    (exact below 2**24).
    
    Parameters
    ----------
    filename : str or Path
        Path to the VTK particle file
        
    Returns
    -------
    dict
        Dictionary containing particle data
    """
    dtypes = {'float': '>f4', 'double': '>f8', 'int': '>i4'}
    with open(filename, 'rb') as f:
        if not f.readline().startswith(b'# vtk DataFile'):
            raise ValueError(f"{filename}: not a legacy VTK file")
        comment = f.readline().decode('ascii', errors='replace')
        time_match = re.search(r'time=\s*([-+0-9.eE]+)', comment)
        cycle_match = re.search(r'cycle=\s*(-?\d+)', comment)
        if time_match is None or cycle_match is None:
            raise ValueError(f"{filename}: no 'time=' / 'cycle=' in header line {comment!r}")
        time = float(time_match.group(1))
        cycle = int(cycle_match.group(1))

        nparticles = None
        positions = None
        scalars = {}
        while True:
            line = f.readline()
            if not line:
                break
            sp = line.split()
            if not sp:
                continue
            if sp[0] == b'POINTS':
                nparticles = int(sp[1])
                dt = np.dtype(dtypes[sp[2].decode()])
                positions = np.frombuffer(f.read(3 * nparticles * dt.itemsize),
                                          dtype=dt).reshape(nparticles, 3)
            elif sp[0] == b'SCALARS':
                if nparticles is None:
                    raise ValueError(f"{filename}: SCALARS before POINTS")
                name = sp[1].decode()
                dt = np.dtype(dtypes[sp[2].decode()])
                if f.readline().split()[:1] != [b'LOOKUP_TABLE']:
                    raise ValueError(f"{filename}: SCALARS {name} without LOOKUP_TABLE")
                scalars[name] = np.frombuffer(f.read(nparticles * dt.itemsize), dtype=dt)
        if positions is None:
            raise ValueError(f"{filename}: no POINTS record")

    out = {
        'time': time,
        'ncycle': cycle,
        'nparticles': nparticles,
        'x': positions[:, 0].astype(np.float64),
        'y': positions[:, 1].astype(np.float64),
        'z': positions[:, 2].astype(np.float64),
        'var_names': []
    }
    for name, values in scalars.items():
        key = 'tag' if name == 'ptag' else name
        if name in ('gid', 'ptag'):
            out[key] = np.rint(values).astype(np.int64)
        else:
            out[key] = values.astype(np.float64)
    return out
