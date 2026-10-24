"""
I/O utilities for reading and writing Athenak data files.
"""
import glob
import os
import os.path as osp
import re
import struct

import numpy as np
import h5py
from .base import xp, cupy_enabled, asnumpy


def read_binary(filename, mb_range=None):
    """
    Reads a bin file from filename to dictionary.

    Originally written by Lev Arzamasskiy (leva@ias.edu) on 11/15/2021
    Updated to support mesh refinement by George Wong (gnwong@ias.edu) on 01/27/2022

    args:
      filename - string
          filename of bin file to read
      mb_range - None, (start, end) tuple, or callable
          Meshblocks whose field data is read. Metadata (index, logical
          location, geometry) is always returned for every block, but field
          data only for blocks ``start <= mb < end``; the other blocks are
          skipped with a seek. A callable is called with the total block
          count and must return the ``(start, end)`` tuple. None reads all.

    returns:
      filedata - dict
          dictionary of fluid file data. ``filedata['mb_data'][var][i]`` is
          global meshblock ``filedata['mb_data_start'] + i``.
    """

    filedata = {}

    # load file and get size
    fp = open(filename, 'rb')
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    # load header information and validate file format
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(f"bad file format \"{code_header[0].decode('utf-8')}\" " +
                        "(should be \"Athena\")")
    version = code_header[-1].split(b'=')[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b'=')[-1])
    pheader = {}
    for _ in range(pheader_count-1):
        key, val = [x.strip() for x in fp.readline().decode('utf-8').split('=')]
        pheader[key] = val
    time = float(pheader['time'])
    cycle = int(pheader['cycle'])
    locsizebytes = int(pheader['size of location'])
    varsizebytes = int(pheader['size of variable'])

    nvars = int(fp.readline().split(b'=')[-1])
    var_list = [v.decode('utf-8') for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b'=')[-1])
    header = [line.decode('utf-8').split('#')[0].strip()
              for line in fp.read(header_size).split(b'\n')]
    header = [line for line in header if len(line) > 0]

    if locsizebytes not in [4, 8]:
        raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
    if varsizebytes not in [4, 8]:
        raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

    locfmt = 'd' if locsizebytes == 8 else 'f'
    vardtype = np.dtype('=f8' if varsizebytes == 8 else '=f4')

    # load grid information from header and validate
    def get_from_header(header, blockname, keyname):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith('<'):
            blockname = '<' + blockname
        if blockname[-1] != '>':
            blockname += '>'
        block = '<none>'
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line
                continue
            key, value = line.split('=')
            if block == blockname and key.strip() == keyname:
                return value.strip()
        raise KeyError(f'no parameter called {blockname}/{keyname}')

    Nx1 = int(get_from_header(header, '<mesh>', 'nx1'))
    Nx2 = int(get_from_header(header, '<mesh>', 'nx2'))
    Nx3 = int(get_from_header(header, '<mesh>', 'nx3'))
    nx1 = int(get_from_header(header, '<meshblock>', 'nx1'))
    nx2 = int(get_from_header(header, '<meshblock>', 'nx2'))
    nx3 = int(get_from_header(header, '<meshblock>', 'nx3'))
    nghost = int(get_from_header(header, '<mesh>', 'nghost'))

    x1min = float(get_from_header(header, '<mesh>', 'x1min'))
    x1max = float(get_from_header(header, '<mesh>', 'x1max'))
    x2min = float(get_from_header(header, '<mesh>', 'x2min'))
    x2max = float(get_from_header(header, '<mesh>', 'x2max'))
    x3min = float(get_from_header(header, '<mesh>', 'x3min'))
    x3max = float(get_from_header(header, '<mesh>', 'x3max'))

    # load data from each meshblock
    n_vars = len(var_list)
    mb_count = 0

    mb_index = []
    mb_logical = []
    mb_geometry = []
    mb_data_offset = []

    mb_data = {}
    for var in var_list:
        mb_data[var] = []

    # Pass 1: per-block metadata only; field data is skipped with a seek so
    # the block count (needed to resolve a callable mb_range) is known
    # before any field data is read.
    while fp.tell() < filesize:
        mb_index.append(np.array(struct.unpack('@6i', fp.read(24))) - nghost)
        nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0])+1
        nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2])+1
        nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4])+1

        mb_logical.append(np.array(struct.unpack('@4i', fp.read(16))))
        mb_geometry.append(np.array(struct.unpack('=6'+locfmt,
                                    fp.read(6*locsizebytes))))
        mb_data_offset.append(fp.tell())
        fp.seek(varsizebytes * nx1_out*nx2_out*nx3_out*n_vars, 1)
        mb_count += 1

    if mb_range is None:
        mb_start, mb_end = 0, mb_count
    elif callable(mb_range):
        mb_start, mb_end = mb_range(mb_count)
    else:
        mb_start, mb_end = mb_range
    mb_start, mb_end = int(mb_start), int(mb_end)
    if not 0 <= mb_start <= mb_end <= mb_count:
        fp.close()
        raise ValueError(f"mb_range ({mb_start}, {mb_end}) outside the "
                         f"{mb_count} meshblocks in {filename}")

    # Pass 2: field data for the selected blocks. Values are returned as
    # float64 (exact for float32 files), the dtype callers expect.
    for mb in range(mb_start, mb_end):
        idx = mb_index[mb]
        nx1_out = idx[1] - idx[0] + 1
        nx2_out = idx[3] - idx[2] + 1
        nx3_out = idx[5] - idx[4] + 1
        count = nx1_out*nx2_out*nx3_out*n_vars
        fp.seek(mb_data_offset[mb])
        data = np.frombuffer(fp.read(varsizebytes * count), dtype=vardtype,
                             count=count).astype(np.float64)
        data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
        for vari, var in enumerate(var_list):
            mb_data[var].append(data[vari])

    fp.close()

    filedata['header'] = header
    filedata['time'] = time
    filedata['cycle'] = cycle
    filedata['var_names'] = var_list

    filedata['Nx1'] = Nx1
    filedata['Nx2'] = Nx2
    filedata['Nx3'] = Nx3
    filedata['nvars'] = nvars

    filedata['x1min'] = x1min
    filedata['x1max'] = x1max
    filedata['x2min'] = x2min
    filedata['x2max'] = x2max
    filedata['x3min'] = x3min
    filedata['x3max'] = x3max

    filedata['n_mbs'] = mb_count
    filedata['nx1_mb'] = nx1
    filedata['nx2_mb'] = nx2
    filedata['nx3_mb'] = nx3
    filedata['nx1_out_mb'] = (mb_index[0][1] - mb_index[0][0])+1
    filedata['nx2_out_mb'] = (mb_index[0][3] - mb_index[0][2])+1
    filedata['nx3_out_mb'] = (mb_index[0][5] - mb_index[0][4])+1

    filedata['mb_index'] = np.array(mb_index)
    filedata['mb_logical'] = np.array(mb_logical)
    filedata['mb_geometry'] = np.array(mb_geometry)
    filedata['mb_data'] = mb_data
    filedata['mb_data_start'] = mb_start

    return filedata

def atomic_write(filename, writer):
    """Write ``filename`` via a temporary file in the same directory.

    ``writer(path)`` writes the complete file to ``path``; only after it
    returns is the result moved over ``filename`` with ``os.replace``, so an
    interrupted or failed write never leaves a truncated file behind. The
    permission bits of an existing ``filename`` are kept (otherwise the
    process umask applies, as for a plain ``open``).
    """
    import tempfile
    filename = os.fspath(filename)
    directory = osp.dirname(osp.abspath(filename))
    fd, temporary = tempfile.mkstemp(prefix='.' + osp.basename(filename) + '.',
                                     suffix='.tmp', dir=directory)
    os.close(fd)
    try:
        if osp.exists(filename):
            mode = os.stat(filename).st_mode & 0o7777
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        writer(temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, filename)
    finally:
        if osp.exists(temporary):
            os.unlink(temporary)


def save_dict_to_hdf5(dic, filename):
    """
    Save dictionary to HDF5 file.

    The file is written to a temporary name in the same directory and moved
    into place only once complete (see :func:`atomic_write`), so a crash
    mid-write never truncates an existing cache.

    Parameters
    ----------
    dic:
        python dictionary to be converted to hdf5 format
    filename:
        desired name of hdf5 file
    """
    exclude_keys = {'data_raw', 'mb_data'} if (str(filename).endswith('.h5data') or '.h5data.' in str(filename)) else set()

    def _write(path):
        with h5py.File(path, 'w') as h5file:
            _recursively_save_dict_contents_to_group(h5file, '/', dic, exclude_keys=exclude_keys)

    atomic_write(filename, _write)


def hdf5_value_supported(item):
    """True if :func:`_recursively_save_dict_contents_to_group` can store
    ``item`` (as a dataset, an empty marker for None, or a group)."""
    if item is None or isinstance(item, dict):
        return True
    if isinstance(item, np.ndarray):
        return item.dtype.kind not in ('O', 'V', 'M', 'm')
    if cupy_enabled and isinstance(item, xp.ndarray):
        return True
    if isinstance(item, np.generic):
        return item.dtype.kind not in ('O', 'V', 'M', 'm')
    return isinstance(item, (str, bytes, bool, int, float, complex,
                             tuple, list, os.PathLike))


def _to_hdf5_value(item):
    """Convert a leaf value to something h5py stores without loss."""
    if cupy_enabled and isinstance(item, xp.ndarray):
        item = asnumpy(item)
    if isinstance(item, np.ndarray) and item.dtype.kind == 'U':
        item = item.astype('S')
    elif isinstance(item, os.PathLike):
        item = os.fspath(item)
    elif isinstance(item, np.str_):
        item = str(item)
    elif isinstance(item, np.bytes_):
        item = bytes(item)
    return item


def _recursively_save_dict_contents_to_group(h5file, path, dic, exclude_keys=None):
    """
    Helper function to save dictionary contents to HDF5 group.

    Keys are stored as ``str(key)``; a key containing ``'/'`` is rejected
    because HDF5 would silently turn it into nested groups. ``None`` is
    stored as an empty (shape-less) dataset and loads back as ``None``; an
    empty dict is stored as an empty group. Values of any other unsupported
    type raise ``TypeError`` instead of being dropped.

    Parameters
    ----------
    h5file:
        h5py file to be written to
    path:
        path within h5py file to saved dictionary
    dic:
        python dictionary to be converted to hdf5 format
    """
    if exclude_keys is None:
        exclude_keys = set()

    for key, item in dic.items():
        key = str(key)
        if key in exclude_keys:
            continue
        if '/' in key:
            raise ValueError(f"cannot save key {key!r} under {path!r}: HDF5 "
                             f"names may not contain '/'")
        name = path + key
        if isinstance(item, dict):
            h5file.require_group(name)
            _recursively_save_dict_contents_to_group(
                h5file, name + '/', item, exclude_keys=exclude_keys
            )
            continue
        if item is None:
            h5file.create_dataset(name, data=h5py.Empty('f'))
            continue
        if not hdf5_value_supported(item):
            raise TypeError(f"cannot save {name!r}: unsupported type "
                            f"{type(item).__name__}")
        try:
            h5file[name] = _to_hdf5_value(item)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"cannot save {name!r} ({type(item).__name__}): "
                            f"{exc}") from exc


def load_dict_from_hdf5(filename, mode='r', exclude_keys=None):
    """
    Load dictionary from HDF5 file.
    
    Parameters
    ----------
    filename:
        name of hdf5 file
    mode:
        file access mode
        
    Returns
    -------
    dict:
        dictionary containing the HDF5 file contents
    """
    if mode not in ['r', 'r+', 'a+']:
        raise Exception('>>> read mode error')
    if exclude_keys is None and (str(filename).endswith('.h5data') or '.h5data.' in str(filename)):
        exclude_keys = {'data_raw', 'mb_data'}
    elif exclude_keys is None:
        exclude_keys = set()
    else:
        exclude_keys = set(exclude_keys)
    with h5py.File(filename, mode) as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/', exclude_keys=exclude_keys)

def _recursively_load_dict_contents_from_group(h5file, path, exclude_keys=None):
    """
    Helper function to load dictionary contents from HDF5 group.
    
    Parameters
    ----------
    h5file:
        h5py file to be read from
    path:
        path within h5py file to load from
        
    Returns
    -------
    dict:
        dictionary containing the HDF5 group contents
    """
    if exclude_keys is None:
        exclude_keys = set()

    dic = {}
    for key, item in h5file[path].items():
        if str(key) in exclude_keys:
            continue
        if isinstance(item, h5py.Dataset):
            if item.shape is None:
                # Empty dataset: how None is stored.
                dic[key] = None
                continue
            v = item[()]
            # h5py returns byte strings for string datasets saved without
            # special_dtype; decode them to str so callers never see bytes.
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            elif isinstance(v, np.ndarray) and v.dtype.kind in ('S', 'O'):
                # arrays of byte-strings → arrays of str
                v = v.astype(str)
            dic[key] = v
        elif isinstance(item, h5py.Group):
            dic[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + '/', exclude_keys=exclude_keys
            )
    return dic


# ---------------------------------------------------------------------------
# Athena 4.2 (classic Athena++) VTK reader
#
# The VTK file carries per-grid geometry and field arrays only. Mesh layout,
# meshblock decomposition, gamma, and EOS live in a separate athinput file.
# The reader parses both, maps Athena 4.2 conventions to AthenaK conventions,
# and returns the same `filedata` dict shape that `read_binary` produces, so
# `AthenaData._load_from_binary` can consume it unchanged.
#
# Low-level VTK parsing is adapted from pyathena's reader
# (pyathena/io/read_vtk.py; no runtime dependency).
# ---------------------------------------------------------------------------


# Variable mapping table: Athena 4.2 VTK field name -> AthenaK raw key(s).
# nvar == 1 means a scalar copied directly; nvar == 3 means a 3-component
# VECTORS field split into three AthenaK scalars listed in `keys`.
_VTK_VAR_MAP = {
    'density':         {'nvar': 1, 'keys': ['dens']},
    'velocity':        {'nvar': 3, 'keys': ['velx', 'vely', 'velz']},
    'cell_centered_B': {'nvar': 3, 'keys': ['bcc1', 'bcc2', 'bcc3']},
}


# TIGRESS-classic / Athena 4.2 default unit system.
#   length_unit = 1 pc
#   time_unit   = 1 s * pc/km = pc / (km/s)
#   mass_unit   = (mu_H * m_H) * pc^3, so rho_code = n_H in cm^-3 by construction
# mu is temperature-dependent (T < 10^4.2 K: mu_low = 1.295; else mu_high = 0.618).
# The single mu stored in the synthesised <units> block is the high-T value;
# attach_tigress_temperature() injects a temp_K derived variable that picks
# the right branch per cell.
TIGRESS_PC_CGS = 3.0856775814913673e18         # 1 pc in cm
TIGRESS_KMS_CGS = 1.0e5                         # 1 km/s in cm/s
TIGRESS_TIME_CGS = TIGRESS_PC_CGS / TIGRESS_KMS_CGS   # = pc / (km/s) in s
TIGRESS_DENSITY_CGS = 2.38858753789e-24         # mu_H * m_H in g/cm^3
TIGRESS_MASS_CGS = TIGRESS_DENSITY_CGS * TIGRESS_PC_CGS ** 3  # mass per pc^3
TIGRESS_MU_LOW = 1.295
TIGRESS_MU_HIGH = 0.618
TIGRESS_MU_H = 1.4271
TIGRESS_T_MU_THRESHOLD_K = 10 ** 4.2            # = 15848.93 K

TIGRESS_DEFAULT_UNITS = {
    'length_cgs': TIGRESS_PC_CGS,
    'mass_cgs':   TIGRESS_MASS_CGS,
    'time_cgs':   TIGRESS_TIME_CGS,
    'mu':         TIGRESS_MU_HIGH,
}


def attach_tigress_temperature(ad, *, mu_low=TIGRESS_MU_LOW,
                                mu_high=TIGRESS_MU_HIGH,
                                mu_H=TIGRESS_MU_H,
                                T_threshold_K=TIGRESS_T_MU_THRESHOLD_K):
    """Attach a TIGRESS-style temperature derived variable to an AthenaData.

    The mean molecular weight is piecewise: ``mu_low`` below ``T_threshold_K``,
    ``mu_high`` above. Self-consistency is enforced by picking the high-T
    branch when its result is at or above the threshold, else the low-T
    branch — this matches the convention used in TIGRESS analysis pipelines.

    The function reads ``T_code = (gamma - 1) * eint / dens`` (already
    available as ``ad.data_func['temp']``) and converts to Kelvin using
    ``T_K = T_code * vunit^2 * mu * m_p / k_B`` where ``vunit`` is the unit
    system's velocity scale.

    Side effects on ``ad``:
      - sets attributes ``mu_low``, ``mu_high``, ``mu_H``, ``T_mu_threshold_K``
      - registers ``ad.data_func['temp_K']`` returning T in Kelvin
    """
    from .base import xp

    # Proton mass in cgs. We use m_p (not amu) to match the convention in
    # TIGRESS papers and pyathena's K_factor = 121.148 K per (km/s)^2.
    M_P_CGS = 1.67262192369e-24

    ad.mu_low = float(mu_low)
    ad.mu_high = float(mu_high)
    ad.mu_H = float(mu_H)
    ad.T_mu_threshold_K = float(T_threshold_K)

    # Capture the params now so the closure isn't sensitive to later mutation.
    _mu_low = float(mu_low)
    _mu_high = float(mu_high)
    _thresh = float(T_threshold_K)

    def temp_K(self, mbl=None, mbh=None):
        T_code = self.data_func['temp'](self, mbl, mbh)
        K_per_codemu = (self.units.vunit ** 2) * M_P_CGS / self.units.k_boltzmann_cgs
        T_low = T_code * (_mu_low * K_per_codemu)
        T_high = T_code * (_mu_high * K_per_codemu)
        # Pick T_high when it's self-consistent (>= threshold), else T_low.
        return xp.where(T_high >= _thresh, T_high, T_low)

    ad.data_func['temp_K'] = temp_K

# Fields that may appear in Athena 4.2 VTK output but the operations layer
# does not consume; the reader skips them silently.
_VTK_SKIP_FIELDS = {'face_centered_B1', 'face_centered_B2', 'face_centered_B3'}


def _vtk_parse_line(line, grid):
    """Parse a single VTK header line, mutating ``grid``.

    Mirrors pyathena's _vtk_parse_line (read_vtk.py:672). Sets keys such as
    ``time``, ``level``, ``Nx`` (from DIMENSIONS), ``le`` (from ORIGIN),
    ``dx`` (from SPACING), ``ncells`` (from CELL_DATA). Sets ``read_field``
    when a SCALARS or VECTORS marker is encountered so the caller knows to
    stop scanning the header.
    """
    sp = line.strip().split()
    if not sp:
        return
    if b'vtk' in sp:
        grid['vtk_version'] = sp[-1]
    elif b'time=' in sp:
        time_index = sp.index(b'time=')
        grid['time'] = float(sp[time_index + 1].rstrip(b','))
        if b'level' in sp:
            grid['level'] = int(sp[time_index + 3].rstrip(b','))
        if b'domain' in sp:
            grid['domain'] = int(sp[time_index + 5].rstrip(b','))
        if sp[0] == b'PRIMITIVE':
            grid['prim_var_type'] = True
    elif b'DIMENSIONS' in sp:
        grid['Nx'] = np.array(sp[-3:]).astype('int')
    elif b'ORIGIN' in sp:
        grid['le'] = np.array(sp[-3:]).astype('float64')
    elif b'SPACING' in sp:
        grid['dx'] = np.array(sp[-3:]).astype('float64')
    elif b'CELL_DATA' in sp:
        grid['ncells'] = int(sp[-1])
    elif b'SCALARS' in sp:
        grid['read_field'] = sp[1]
        grid['read_type'] = 'scalar'
    elif b'VECTORS' in sp:
        grid['read_field'] = sp[1]
        grid['read_type'] = 'vector'
    elif b'POINTS' in sp:
        grid['npoint'] = int(sp[1])


def _vtk_set_grid(source):
    """Walk a VTK file's header and return a grid metadata dict.

    ``source`` may be a filesystem path (str / Path) or an ``open_fn``
    callable returning a fresh seekable file-like object. The dict carries
    an ``open_fn`` field so downstream readers don't have to remember which
    form was used.

    Stops at the first SCALARS/VECTORS marker (the start of the data
    section). The ``DIMENSIONS`` value is node-counted; we convert it to
    cell counts here (Nx -= 1) to match the AthenaK convention used by
    callers. Mirrors pyathena's _set_grid (read_vtk.py:517).
    """
    if callable(source):
        open_fn = source
        label = '<callable>'
    else:
        open_fn = lambda path=str(source): open(path, 'rb')
        label = str(source)
    g = {
        'filename': label,
        'open_fn': open_fn,
        'data': {},
        'read_field': None,
        'read_type': None,
    }
    with open_fn() as fp:
        while g['read_field'] is None:
            g['data_offset'] = fp.tell()
            line = fp.readline()
            if not line:
                raise IOError(f'{label}: reached EOF before any field marker')
            _vtk_parse_line(line, g)
    g['Nx'] = np.asarray(g['Nx']) - 1
    # A node count of 1 marks a collapsed (2D/1D) axis whose SPACING carries
    # no geometry; a one-cell axis of a 3D grid keeps its real spacing.
    collapsed = g['Nx'] == 0
    g['Nx'][collapsed] = 1
    g['dx'][collapsed] = 1.0
    g['re'] = g['le'] + g['Nx'] * g['dx']
    return g


def _vtk_set_field_map(grid):
    """Scan a VTK stream from data_offset to EOF, mapping field -> byte layout.

    Direct port of pyathena's _set_field_map (read_vtk.py:615). Reads via
    ``grid['open_fn']`` so it works on tarfile members too. Returns
    ``{field_name: {offset, nvar, dtype, dsize, ndata, read_table}}``.
    """
    field_map = {}
    Nx = grid['Nx']
    with grid['open_fn']() as fp:
        fp.seek(0, 2)
        eof = fp.tell()
        offset = grid['data_offset']
        fp.seek(offset)

        while offset < eof:
            line = fp.readline()
            sp = line.strip().split()
            if len(sp) < 3:
                break
            field = sp[1].decode('utf-8')
            entry = {
                'read_table': False,
                'title': line,
            }
            if b'SCALARS' in line:
                fp.readline()
                entry['read_table'] = True
                entry['nvar'] = 1
            elif b'VECTORS' in line:
                entry['nvar'] = 3
            else:
                raise TypeError(
                    f'{grid["filename"]}: unknown record type {sp[0]!r} '
                    f'at offset {offset}'
                )

            entry['offset'] = offset
            entry['ndata'] = entry['nvar'] * grid['ncells']
            if field == 'face_centered_B1':
                entry['ndata'] = (Nx[0] + 1) * Nx[1] * Nx[2]
            elif field == 'face_centered_B2':
                entry['ndata'] = Nx[0] * (Nx[1] + 1) * Nx[2]
            elif field == 'face_centered_B3':
                entry['ndata'] = Nx[0] * Nx[1] * (Nx[2] + 1)

            if sp[2] == b'int':
                entry['dtype'] = 'i'
            elif sp[2] == b'float':
                entry['dtype'] = 'f'
            elif sp[2] == b'double':
                entry['dtype'] = 'd'
            else:
                raise TypeError(
                    f'{grid["filename"]}: unsupported dtype {sp[2]!r} for {field}'
                )

            entry['dsize'] = entry['ndata'] * struct.calcsize(entry['dtype'])
            fp.seek(entry['dsize'], 1)
            offset = fp.tell()
            tmp = fp.readline()
            if len(tmp) > 1:
                fp.seek(offset)
            else:
                offset = fp.tell()
            field_map[field] = entry
    return field_map


def _vtk_read_array(grid, field, field_map):
    """Read one field, returning shape (nx3, nx2, nx1[, nvar]).

    VTK stores binary blocks big-endian; NumPy 2.0 changed how byte order
    is swapped on ndarrays, so we branch on the runtime numpy version
    (mirrors read_vtk.py:436-440).
    """
    fm = field_map[field]
    with grid['open_fn']() as fp:
        fp.seek(fm['offset'])
        fp.readline()  # SCALARS/VECTORS title
        if fm['read_table']:
            fp.readline()  # LOOKUP_TABLE default
        raw = np.frombuffer(buffer=fp.read(fm['dsize']), dtype=fm['dtype'])
    np_version = tuple(int(x) for x in np.__version__.split('.')[:2])
    if np_version >= (2, 0):
        arr = raw.view(raw.dtype.newbyteorder())
    else:
        arr = raw.newbyteorder()
    Nx = grid['Nx']
    if fm['nvar'] == 1:
        arr = arr.reshape(np.flipud(Nx))
    else:
        arr = arr.reshape((*np.flipud(Nx), fm['nvar']))
    return arr


def _vtk_parse_filename(filename):
    """Decompose a VTK path into ``(dirname, problem_id, num, suffix, ext,
    mpi_mode, nonzero_id)``.

    Adapted from pyathena's _parse_filename (read_vtk.py:546). Detects
    multi-process output by checking whether the parent directory is named
    ``idN``.
    """
    sep = osp.sep
    dirname = osp.dirname(filename)
    nonzero_id = False
    dirname_last = dirname.split(sep)[-1] if dirname else ''
    if dirname_last.startswith('id') and dirname_last[2:].isdigit():
        dirname = sep.join(dirname.split(sep)[:-1])
        mpi_mode = True
    else:
        mpi_mode = False

    base = osp.basename(filename)
    base_split = base.split('.')
    if len(base_split) == 3:
        problem_id = '.'.join(base_split[:-2])
        num = base_split[-2]
        suffix = None
        ext = base_split[-1]
    else:
        try:
            inum = -3
            int(base_split[inum])
            suffix = base_split[-2]
        except (ValueError, IndexError):
            inum = -2
            suffix = None

        if mpi_mode and int(dirname_last[2:]) != 0:
            problem_id = '.'.join(base_split[:inum])
            problem_id = problem_id.replace('-' + dirname_last, '')
            nonzero_id = True
        else:
            problem_id = '.'.join(base_split[:inum])

        num = base_split[inum]
        ext = base_split[-1]

    return dirname, problem_id, num, suffix, ext, mpi_mode, nonzero_id


def _vtk_discover_files(filename, id0_only=False):
    """Return the list of per-grid VTK files belonging to one snapshot.

    In MPI mode an Athena 4.2 run writes ``<dirname>/id0/<problem>.<num>.vtk``,
    ``<dirname>/id1/<problem>-id1.<num>.vtk``, ... . We glob the siblings of
    the supplied file. In non-MPI mode (or with ``id0_only=True``) we return
    just the given path.
    """
    dirname, problem_id, num, suffix, ext, mpi_mode, _ = _vtk_parse_filename(filename)
    if not mpi_mode or id0_only:
        return [filename]
    if suffix is None:
        pattern = osp.join(dirname, f'id*/{problem_id}-id*.{num}.{ext}')
    else:
        pattern = osp.join(dirname, f'id*/{problem_id}-id*.{num}.{suffix}.{ext}')
    fnames = [filename] + glob.glob(pattern)
    # Preserve order, drop duplicates (id0 file may match both lists).
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(fnames))


# Athinput parsing -----------------------------------------------------------

_ATHINPUT_BLOCK_RE = re.compile(r'^<\s*([A-Za-z0-9_]+)\s*>\s*$')
_ATHINPUT_KV_RE = re.compile(r'^([A-Za-z0-9_\[\]]+)\s*=\s*(.+?)\s*$')


def _read_athena_par(filename):
    """Parse an Athena 4.2 athinput-style file into ``{block: {key: str}}``.

    Lightweight INI-style parser: ``<block>`` opens a block, ``key = value``
    populates it, ``#`` and ``!`` start comments. Values are returned as
    stripped strings; the caller is responsible for ``int()`` / ``float()``.
    Tolerates inline ``# comment`` after a value.
    """
    par = {}
    block = None
    with open(filename, 'r') as fp:
        for raw in fp:
            # Drop comments.
            line = raw.split('#', 1)[0].split('!', 1)[0].strip()
            if not line:
                continue
            m = _ATHINPUT_BLOCK_RE.match(line)
            if m:
                block = m.group(1)
                par.setdefault(block, {})
                continue
            if block is None:
                continue
            kv = _ATHINPUT_KV_RE.match(line)
            if kv:
                par[block][kv.group(1)] = kv.group(2).strip()
    return par


_ATHINPUT_MAX_PARENTS = 2


def _discover_athinput(vtk_path, problem_id):
    """Search for an Athena 4.2 parameter file alongside the VTK output.

    Real datasets place the parameter file in surprising spots — next to the
    rundir, inside an ``input/<config>/`` directory, or as a runtime ``.par``
    dump. The search covers the VTK file's directory and at most
    ``_ATHINPUT_MAX_PARENTS`` parents (plus their ``input/``-style
    subdirectories) and looks for, in priority order:

    1. ``athinput.<problem_id>`` exactly,
    2. ``<problem_id>.par`` exactly (runtime dump),
    3. the single ``athinput.*`` or ``*.par`` file in the run directory
       itself (the VTK directory, or the parent of an ``idN/`` directory).

    Wildcard matches are never taken from parent directories and never
    when several candidates exist, so an unrelated parameter file further
    up the tree cannot be picked silently.

    Returns the absolute path or ``None`` if nothing matches.
    """
    candidate_dirs = []
    cur = osp.dirname(osp.abspath(vtk_path))
    run_dirs = [cur]
    last = osp.basename(cur)
    if last.startswith('id') and last[2:].isdigit():
        run_dirs.append(osp.dirname(cur))
    for _ in range(_ATHINPUT_MAX_PARENTS + 1):
        candidate_dirs.append(cur)
        parent = osp.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Also peek into common sibling directories that hold inputs.
    for root in list(candidate_dirs):
        for sub in ('input', 'inputs', 'pars', 'par'):
            extra = osp.join(root, sub)
            if osp.isdir(extra):
                # Search up to one level inside (e.g. input/MHD/R8_4pc.par).
                candidate_dirs.append(extra)
                try:
                    for entry in sorted(os.listdir(extra)):
                        full = osp.join(extra, entry)
                        if osp.isdir(full):
                            candidate_dirs.append(full)
                except OSError:
                    pass

    # Pass 1: exact filename match in each candidate dir.
    for d in candidate_dirs:
        for name in (f'athinput.{problem_id}', f'{problem_id}.par'):
            path = osp.join(d, name)
            if osp.exists(path):
                return path
    # Pass 2: an unambiguous wildcard match in the run directory only.
    for d in run_dirs:
        matches = sorted(set(glob.glob(osp.join(d, 'athinput.*')))
                         | set(glob.glob(osp.join(d, '*.par'))))
        matches = [m for m in matches if osp.isfile(m)]
        if len(matches) == 1:
            return matches[0]
    return None


def _athinput_value(par, block, key, cast=str, default=None, required=False):
    """Lookup helper with type casting and clear error messages."""
    if block in par and key in par[block]:
        return cast(par[block][key])
    if required:
        raise KeyError(f"athinput is missing required key <{block}>/{key}")
    return default


VALID_LAYOUTS = ('auto', 'merged', 'idN', 'tar', 'per_var')


def _detect_vtk_layout(filename):
    """Auto-detect which physical layout a VTK snapshot uses.

    See :func:`read_athena_vtk` for the layout taxonomy. Returns one of
    ``'merged'``, ``'idN'``, ``'tar'``, ``'per_var'``.

    Priority: tar archive > inside an idN/ subdir > input file carries a
    per-variable suffix > sibling tar > sibling idN tree > sibling suffix
    files > merged. The suffix check beats the idN tree because real
    datasets host both side-by-side (a top-level per-variable file plus
    the original idN/ directories), and users who pass a suffix file are
    asking for the merged-variable view.
    """
    if filename.endswith('.tar'):
        return 'tar'

    dirname, problem_id, num, suffix, ext, mpi_mode, _ = _vtk_parse_filename(filename)
    if mpi_mode:
        return 'idN'

    # Input itself has a per-variable suffix → per_var.
    if suffix is not None:
        return 'per_var'

    # Sibling tar bundle? (e.g. <dir>/<problem>.<num>.tar)
    if osp.exists(osp.join(dirname, f'{problem_id}.{num}.tar')):
        return 'tar'

    # Top-level idN/ tree adjacent to this file?
    id_tree = glob.glob(osp.join(dirname, f'id*/{problem_id}*.{num}.{ext}'))
    if id_tree:
        return 'idN'

    # Sibling suffix files (input is un-suffixed but suffix variants exist).
    pattern = osp.join(dirname, f'{problem_id}.{num}.*.{ext}')
    siblings = [s for s in glob.glob(pattern)
                if osp.abspath(s) != osp.abspath(filename)]
    if siblings:
        return 'per_var'

    return 'merged'


def _athinput_mesh(par):
    """Pull mesh + meshblock parameters out of the parsed athinput."""
    if 'domain2' in par:
        raise NotImplementedError(
            "Athena 4.2 multi-domain SMR output is not supported by "
            "read_athena_vtk (only <domain1> is handled). Use pyathena's "
            "AthenaDataSet for the heterogeneous case."
        )
    Nx1 = _athinput_value(par, 'domain1', 'Nx1', int, required=True)
    Nx2 = _athinput_value(par, 'domain1', 'Nx2', int, required=True)
    Nx3 = _athinput_value(par, 'domain1', 'Nx3', int, required=True)
    NGrid_x1 = _athinput_value(par, 'domain1', 'NGrid_x1', int, default=1)
    NGrid_x2 = _athinput_value(par, 'domain1', 'NGrid_x2', int, default=1)
    NGrid_x3 = _athinput_value(par, 'domain1', 'NGrid_x3', int, default=1)
    x1min = _athinput_value(par, 'domain1', 'x1min', float, required=True)
    x1max = _athinput_value(par, 'domain1', 'x1max', float, required=True)
    x2min = _athinput_value(par, 'domain1', 'x2min', float, required=True)
    x2max = _athinput_value(par, 'domain1', 'x2max', float, required=True)
    x3min = _athinput_value(par, 'domain1', 'x3min', float, required=True)
    x3max = _athinput_value(par, 'domain1', 'x3max', float, required=True)
    if Nx1 % NGrid_x1 or Nx2 % NGrid_x2 or Nx3 % NGrid_x3:
        raise ValueError(
            f"Nx ({Nx1},{Nx2},{Nx3}) not divisible by NGrid "
            f"({NGrid_x1},{NGrid_x2},{NGrid_x3}); cannot derive meshblock dims"
        )
    return {
        'Nx': (Nx1, Nx2, Nx3),
        'NGrid': (NGrid_x1, NGrid_x2, NGrid_x3),
        'xmin': (x1min, x2min, x3min),
        'xmax': (x1max, x2max, x3max),
        'nx_mb': (Nx1 // NGrid_x1, Nx2 // NGrid_x2, Nx3 // NGrid_x3),
        'n_mbs': NGrid_x1 * NGrid_x2 * NGrid_x3,
    }


def _athinput_eos(par):
    """Pull gamma and (optional) iso_csound out of the parsed athinput.

    Kept as a thin shim around _athinput_physics for callers that only
    need the EOS pair.
    """
    physics = _athinput_physics(par)
    return physics['gamma'], physics['iso_csound']


def _athinput_physics(par):
    """Extract physics + EOS + shearing-box parameters from an athinput.

    The Athena 4.2 ``<configure>`` block (a runtime dump of compile-time
    flags) is the authoritative source for MHD vs hydro (``gas``) and EOS
    (``eq_state``). ``ShearingBox`` toggles whether ``Omega`` and ``qshear``
    in ``<problem>`` describe a rotating frame.

    Returns a dict with the following keys:

    - ``gamma`` (float): adiabatic index from ``<problem>/gamma``.
    - ``iso_csound`` (float or None): isothermal sound speed from
      ``<problem>/iso_csound`` / ``cs0`` / ``cs`` / ``iso_sound_speed``.
    - ``is_mhd_configured`` (bool or None): True iff ``<configure>/gas``
      starts with ``mhd``; None when ``<configure>`` is absent.
    - ``eos_configured`` (str or None): lowercase value of
      ``<configure>/eq_state`` when present.
    - ``is_shearing_box`` (bool): True iff ``<configure>/ShearingBox`` is
      ``yes``.
    - ``Omega`` (float or None): ``<problem>/Omega`` (only when shearing
      box is on). Stored in athinput's units (km/s/pc in TIGRESS code
      conventions); ad.units.vunit / ad.units.length_cgs gives the
      conversion to 1/s.
    - ``qshear`` (float or None): ``<problem>/qshear`` (only when shearing
      box is on). Dimensionless shear parameter ``-d ln Omega / d ln R``.
    - ``configure`` (dict): the raw parsed ``<configure>`` sub-dict (string
      values) for callers that want to inspect other compile-time flags.
    """
    out = {
        'gamma': _athinput_value(par, 'problem', 'gamma', float,
                                 default=5.0 / 3.0),
        'iso_csound': None,
        'is_mhd_configured': None,
        'eos_configured': None,
        'is_shearing_box': False,
        'Omega': None,
        'qshear': None,
        'configure': dict(par.get('configure', {})),
    }
    for iso_key in ('iso_csound', 'cs0', 'cs', 'iso_sound_speed'):
        if 'problem' in par and iso_key in par['problem']:
            out['iso_csound'] = float(par['problem'][iso_key])
            break
    cfg = par.get('configure', {})
    if 'gas' in cfg:
        out['is_mhd_configured'] = cfg['gas'].strip().lower().startswith('mhd')
    if 'eq_state' in cfg:
        out['eos_configured'] = cfg['eq_state'].strip().lower()
    # ShearingBox is authoritative when present in <configure>; otherwise
    # infer it from Omega being defined in <problem> (user-edited athinputs
    # without a runtime <configure> dump still set Omega/qshear).
    if 'ShearingBox' in cfg:
        out['is_shearing_box'] = cfg['ShearingBox'].strip().lower() == 'yes'
    elif 'problem' in par and 'Omega' in par['problem']:
        out['is_shearing_box'] = True
    if out['is_shearing_box'] and 'problem' in par:
        if 'Omega' in par['problem']:
            out['Omega'] = float(par['problem']['Omega'])
        if 'qshear' in par['problem']:
            out['qshear'] = float(par['problem']['qshear'])
    return out


def _resolve_athinput(athinput, filename, problem_id):
    """Resolve an athinput arg (path, dict, or None) to a parsed dict."""
    if isinstance(athinput, dict):
        return athinput
    path = athinput or _discover_athinput(filename, problem_id)
    if path is None:
        raise FileNotFoundError(
            f"could not locate athinput for VTK {filename!r}; "
            f"pass `athinput=...` explicitly or place an athinput / .par "
            f"file alongside the simulation output"
        )
    print(f"read_athena_vtk: using athinput {osp.abspath(path)}")
    return _read_athena_par(path)


# Maps an Athena 4.2 VTK field to the list of AthenaK keys it populates.
# nvar==1 → scalar copy; nvar==3 → split components into the listed keys.
def _vtk_field_to_athenak_keys(vtk_field):
    if vtk_field in _VTK_VAR_MAP:
        return _VTK_VAR_MAP[vtk_field]['keys']
    if vtk_field == 'pressure':
        return ['eint']
    if vtk_field == 'momentum':
        return ['velx', 'vely', 'velz']
    return None  # ignored


def _populate_block_from_array(target, vtk_field, arr, *, gamma, density=None):
    """Apply variable-name + value mapping for one VTK field on one block.

    ``target`` is a mutable dict ``{athenak_key: ndarray}`` for ONE meshblock.
    ``arr`` is the raw field array for the same block, shape (nz, ny, nx)
    for scalars or (nz, ny, nx, nvar) for vectors. Returns the density array
    so callers can chain it through to a later momentum->velocity step.
    """
    if vtk_field == 'density':
        target['dens'] = np.ascontiguousarray(arr)
        return target['dens']
    if vtk_field == 'velocity':
        target['velx'] = np.ascontiguousarray(arr[..., 0])
        target['vely'] = np.ascontiguousarray(arr[..., 1])
        target['velz'] = np.ascontiguousarray(arr[..., 2])
        return density
    if vtk_field == 'momentum':
        if density is None:
            raise ValueError("momentum present without density on this block")
        target['velx'] = np.ascontiguousarray(arr[..., 0] / density)
        target['vely'] = np.ascontiguousarray(arr[..., 1] / density)
        target['velz'] = np.ascontiguousarray(arr[..., 2] / density)
        return density
    if vtk_field == 'cell_centered_B':
        target['bcc1'] = np.ascontiguousarray(arr[..., 0])
        target['bcc2'] = np.ascontiguousarray(arr[..., 1])
        target['bcc3'] = np.ascontiguousarray(arr[..., 2])
        return density
    if vtk_field == 'pressure':
        target['eint'] = np.ascontiguousarray(arr / (gamma - 1.0))
        return density
    return density  # unknown / skipped field


def _slice_full_grid_into_blocks(arr, mesh):
    """Slice a full-domain array into NGrid_x1*NGrid_x2*NGrid_x3 sub-blocks.

    ``arr`` has shape ``(Nx3, Nx2, Nx1[, nvar])`` (z, y, x, optional vector
    component). Returns a list of length ``n_mbs`` in lexicographic order
    ``(k, j, i)`` — same order downstream geometry code builds.
    """
    NGrid_x1, NGrid_x2, NGrid_x3 = mesh['NGrid']
    nx1_mb, nx2_mb, nx3_mb = mesh['nx_mb']
    blocks = []
    for k in range(NGrid_x3):
        for j in range(NGrid_x2):
            for i in range(NGrid_x1):
                sl = (
                    slice(k * nx3_mb, (k + 1) * nx3_mb),
                    slice(j * nx2_mb, (j + 1) * nx2_mb),
                    slice(i * nx1_mb, (i + 1) * nx1_mb),
                )
                blocks.append(arr[sl])
    return blocks


def _build_mb_geometry(mesh):
    """Construct (mb_geometry, mb_logical, mb_index) for a uniform NGrid mesh."""
    NGrid_x1, NGrid_x2, NGrid_x3 = mesh['NGrid']
    nx1_mb, nx2_mb, nx3_mb = mesh['nx_mb']
    xmin = mesh['xmin']
    xmax = mesh['xmax']
    Lmb = [(xmax[a] - xmin[a]) / mesh['NGrid'][a] for a in range(3)]
    n_mbs = mesh['n_mbs']
    mb_geometry = np.zeros((n_mbs, 6), dtype='float64')
    mb_logical = np.zeros((n_mbs, 4), dtype='int64')
    mb_index = np.zeros((n_mbs, 6), dtype='int64')
    idx = 0
    for k in range(NGrid_x3):
        for j in range(NGrid_x2):
            for i in range(NGrid_x1):
                le = (xmin[0] + i * Lmb[0],
                      xmin[1] + j * Lmb[1],
                      xmin[2] + k * Lmb[2])
                re = (le[0] + Lmb[0], le[1] + Lmb[1], le[2] + Lmb[2])
                mb_geometry[idx] = [le[0], re[0], le[1], re[1], le[2], re[2]]
                mb_logical[idx] = [i, j, k, 0]
                mb_index[idx] = [i * nx1_mb, (i + 1) * nx1_mb - 1,
                                 j * nx2_mb, (j + 1) * nx2_mb - 1,
                                 k * nx3_mb, (k + 1) * nx3_mb - 1]
                idx += 1
    return mb_geometry, mb_logical, mb_index


def _synthesize_header(mesh, *, time, gamma, eos, iso_csound,
                       has_mhd, physics=None, header_overrides=None):
    """Build the AthenaK-style header list of strings.

    ``physics`` (optional) carries the parsed athinput data from
    :func:`_athinput_physics`. When supplied, the shearing-box block is
    appended with ``Omega`` and ``qshear``.
    """
    Nx1, Nx2, Nx3 = mesh['Nx']
    nx1_mb, nx2_mb, nx3_mb = mesh['nx_mb']
    x1min, x2min, x3min = mesh['xmin']
    x1max, x2max, x3max = mesh['xmax']
    fluid_block = 'mhd' if has_mhd else 'hydro'
    lines = [
        '<mesh>',
        f'nx1={Nx1}', f'nx2={Nx2}', f'nx3={Nx3}', 'nghost=0',
        f'x1min={x1min}', f'x1max={x1max}',
        f'x2min={x2min}', f'x2max={x2max}',
        f'x3min={x3min}', f'x3max={x3max}',
        '<meshblock>',
        f'nx1={nx1_mb}', f'nx2={nx2_mb}', f'nx3={nx3_mb}',
        f'<{fluid_block}>',
        f'gamma={gamma}',
        f'eos={eos}',
    ]
    if eos == 'isothermal':
        if iso_csound is None:
            raise ValueError(
                "isothermal EOS requires `iso_csound`/`cs0`/`cs` in <problem>"
            )
        lines.append(f'iso_sound_speed={iso_csound}')
    if physics and physics.get('is_shearing_box'):
        lines.append('<shearing_box>')
        if physics.get('Omega') is not None:
            lines.append(f'Omega={physics["Omega"]}')
        if physics.get('qshear') is not None:
            lines.append(f'qshear={physics["qshear"]}')
    lines.append('<time>')
    lines.append(f'time={time}')

    if header_overrides:
        cleaned = []
        current_block = None
        override_keys_by_block = {b: set(k.keys()) for b, k in header_overrides.items()}
        for line in lines:
            if line.startswith('<'):
                current_block = line.strip('<>')
                cleaned.append(line)
                continue
            if current_block in override_keys_by_block:
                key = line.split('=', 1)[0]
                if key in override_keys_by_block[current_block]:
                    continue
            cleaned.append(line)
        lines = cleaned
        for block, kvs in header_overrides.items():
            lines.append(f'<{block}>')
            for k, v in kvs.items():
                lines.append(f'{k}={v}')
    return lines


def _assemble_filedata(block_data, mesh, *, time, gamma, iso_csound,
                       has_mhd, has_pressure, physics=None,
                       header_overrides=None):
    """Package per-meshblock arrays into the AthenaK ``filedata`` dict.

    ``block_data`` is a list of ``{athenak_key: ndarray}`` dicts, one per
    meshblock in lexicographic ``(k, j, i)`` order. Variables present in
    the first block (any of them non-None) define ``var_names``.

    When the ``physics`` dict is supplied (see :func:`_athinput_physics`),
    ``<configure>/gas`` and ``<configure>/eq_state`` override the
    field-detection booleans for MHD and adiabatic-vs-isothermal, and the
    shearing-box parameters land as top-level keys
    (``is_shearing_box``, ``Omega``, ``qshear``, ``configure``).
    """
    n_mbs = mesh['n_mbs']
    if len(block_data) != n_mbs:
        raise ValueError(
            f"block_data has {len(block_data)} entries; mesh expects "
            f"n_mbs={n_mbs}"
        )
    # var_names: union across blocks, in a stable order matching _VTK_VAR_MAP
    # plus eint.
    seen = set()
    var_names = []
    canonical_order = ['dens', 'velx', 'vely', 'velz',
                       'bcc1', 'bcc2', 'bcc3', 'eint']
    for v in canonical_order:
        for blk in block_data:
            if v in blk and blk[v] is not None:
                if v not in seen:
                    var_names.append(v)
                    seen.add(v)
                break
    # Also catch any unconventional keys callers added.
    for blk in block_data:
        for v in blk:
            if v not in seen and blk[v] is not None:
                var_names.append(v)
                seen.add(v)

    mb_data = {var: [] for var in var_names}
    for blk in block_data:
        for var in var_names:
            mb_data[var].append(blk.get(var))

    # Validate: no None entries in mb_data — every block must contribute
    # every variable present.
    for var, arrs in mb_data.items():
        for i, a in enumerate(arrs):
            if a is None:
                raise ValueError(
                    f"meshblock {i} is missing variable {var!r}; cannot "
                    f"assemble filedata"
                )

    mb_geometry, mb_logical, mb_index = _build_mb_geometry(mesh)

    # <configure> wins over field-detection when present.
    if physics is not None and physics.get('is_mhd_configured') is not None:
        has_mhd = bool(physics['is_mhd_configured'])
    if physics is not None and physics.get('eos_configured'):
        eos = physics['eos_configured']
    else:
        eos = 'adiabatic' if has_pressure else 'isothermal'

    header_lines = _synthesize_header(
        mesh, time=time, gamma=gamma, eos=eos, iso_csound=iso_csound,
        has_mhd=has_mhd, physics=physics, header_overrides=header_overrides,
    )

    Nx1, Nx2, Nx3 = mesh['Nx']
    nx1_mb, nx2_mb, nx3_mb = mesh['nx_mb']
    x1min, x2min, x3min = mesh['xmin']
    x1max, x2max, x3max = mesh['xmax']
    filedata = {
        'header': header_lines,
        'time': time,
        'cycle': 0,
        'var_names': var_names,
        'Nx1': Nx1, 'Nx2': Nx2, 'Nx3': Nx3,
        'nvars': len(var_names),
        'x1min': x1min, 'x1max': x1max,
        'x2min': x2min, 'x2max': x2max,
        'x3min': x3min, 'x3max': x3max,
        'n_mbs': n_mbs,
        'nx1_mb': nx1_mb, 'nx2_mb': nx2_mb, 'nx3_mb': nx3_mb,
        'nx1_out_mb': nx1_mb, 'nx2_out_mb': nx2_mb, 'nx3_out_mb': nx3_mb,
        'mb_index': np.asarray(mb_index),
        'mb_logical': np.asarray(mb_logical),
        'mb_geometry': np.asarray(mb_geometry),
        'mb_data': mb_data,
    }
    # Thread shearing-box / configure metadata through as top-level keys so
    # AthenaData._load_from_dic attaches them as attributes automatically.
    if physics is not None:
        filedata['is_shearing_box'] = bool(physics.get('is_shearing_box'))
        if physics.get('Omega') is not None:
            filedata['Omega'] = float(physics['Omega'])
        if physics.get('qshear') is not None:
            filedata['qshear'] = float(physics['qshear'])
        if physics.get('configure'):
            filedata['configure'] = dict(physics['configure'])
    return filedata


def _local_mesh_from_grid(grid, par_nx_mb):
    """Derive a mesh dict from a single VTK grid + athinput's meshblock dims.

    Real-world merged VTKs sometimes store a sub-volume of the original
    simulation (e.g. the disk midplane cube cropped out of a tall column),
    so the file's ``DIMENSIONS`` / ``ORIGIN`` / ``SPACING`` is the source of
    truth for actual geometry. The athinput's per-rank meshblock size
    (``Nx_global / NGrid``) is reused to keep blocks the same shape as the
    original run — when that doesn't divide the file's Nx evenly on an
    axis, that axis falls back to a single block.
    """
    file_Nx = tuple(int(x) for x in grid['Nx'])
    file_le = tuple(float(x) for x in grid['le'])
    file_re = tuple(float(x) for x in grid['re'])
    nx_mb = list(par_nx_mb)
    NGrid = [1, 1, 1]
    for a in range(3):
        if nx_mb[a] > 0 and file_Nx[a] % nx_mb[a] == 0:
            NGrid[a] = file_Nx[a] // nx_mb[a]
        else:
            # Athinput meshblock doesn't divide this axis cleanly; treat the
            # whole file as a single block along this dimension.
            NGrid[a] = 1
            nx_mb[a] = file_Nx[a]
    return {
        'Nx': file_Nx,
        'NGrid': tuple(NGrid),
        'nx_mb': tuple(nx_mb),
        'xmin': file_le,
        'xmax': file_re,
        'n_mbs': NGrid[0] * NGrid[1] * NGrid[2],
    }


def _read_layout_merged(filename, par, mesh, header_overrides):
    """Layout A: one VTK file holding a single big uniform grid.

    The VTK file is authoritative for geometry (Nx, extents, dx); the
    athinput contributes gamma/EOS and the per-rank meshblock size, which
    we use to decompose the file's grid into logical meshblocks.
    """
    physics = _athinput_physics(par)
    gamma, iso_csound = physics['gamma'], physics['iso_csound']
    grid = _vtk_set_grid(filename)
    fm = _vtk_set_field_map(grid)

    local_mesh = _local_mesh_from_grid(grid, mesh['nx_mb'])

    has_pressure = 'pressure' in fm
    has_mhd = 'cell_centered_B' in fm
    has_momentum = 'momentum' in fm and 'velocity' not in fm

    block_data = [{} for _ in range(local_mesh['n_mbs'])]

    # Read density first (momentum needs it for the per-block division).
    if 'density' in fm:
        dens_full = _vtk_read_array(grid, 'density', fm)
        for i, sub in enumerate(_slice_full_grid_into_blocks(dens_full, local_mesh)):
            _populate_block_from_array(block_data[i], 'density',
                                       np.ascontiguousarray(sub), gamma=gamma)

    # Iterate the remaining fields in a deterministic order.
    for vtk_field in fm:
        if vtk_field in ('density',):
            continue
        if vtk_field not in ('velocity', 'momentum', 'cell_centered_B', 'pressure'):
            continue
        arr_full = _vtk_read_array(grid, vtk_field, fm)
        for i, sub in enumerate(_slice_full_grid_into_blocks(arr_full, local_mesh)):
            density = block_data[i].get('dens')
            _populate_block_from_array(block_data[i], vtk_field,
                                       np.ascontiguousarray(sub),
                                       gamma=gamma, density=density)

    return _assemble_filedata(
        block_data, local_mesh, time=float(grid['time']),
        gamma=gamma, iso_csound=iso_csound,
        has_mhd=has_mhd, has_pressure=has_pressure,
        physics=physics, header_overrides=header_overrides,
    )


def _effective_mesh_from_files(grids, mesh, *, source_label):
    """Derive an effective mesh from a set of per-file grids + global Nx.

    Each grid in ``grids`` is assumed to cover one (logical) meshblock of
    uniform size; the per-meshblock dims come from the first grid's Nx.
    The effective ``NGrid`` is computed as ``global_Nx / per_file_Nx`` so
    partially-merged trees (where some axes have been merged across
    ranks) are handled transparently — the file count just needs to match
    the product. ``mesh`` (from athinput) supplies the global Nx and
    extents.
    """
    file_Nx = tuple(int(x) for x in grids[0]['Nx'])
    global_Nx = tuple(mesh['Nx'])
    NGrid = []
    for a in range(3):
        if file_Nx[a] == 0 or global_Nx[a] % file_Nx[a] != 0:
            raise ValueError(
                f"{source_label}: athinput global Nx={global_Nx} not evenly "
                f"divisible by per-file Nx={file_Nx} on axis {a}; cannot "
                f"derive meshblock decomposition"
            )
        NGrid.append(global_Nx[a] // file_Nx[a])
    expected_n_mbs = NGrid[0] * NGrid[1] * NGrid[2]
    if len(grids) != expected_n_mbs:
        raise ValueError(
            f"{source_label}: file count {len(grids)} doesn't match "
            f"NGrid_eff={tuple(NGrid)} (product {expected_n_mbs}); "
            f"the partial-merge layout must tile the global mesh exactly"
        )
    # Cross-check every grid has the same per-file Nx.
    for g in grids[1:]:
        gNx = tuple(int(x) for x in g['Nx'])
        if gNx != file_Nx:
            raise ValueError(
                f"{g['filename']!r}: per-file Nx={gNx} doesn't match "
                f"reference {file_Nx}; mixed-size grids are not supported"
            )
    return {
        'Nx': global_Nx,
        'NGrid': tuple(NGrid),
        'nx_mb': file_Nx,
        'xmin': mesh['xmin'],
        'xmax': mesh['xmax'],
        'n_mbs': expected_n_mbs,
    }


def _read_layout_idN(filename, par, mesh, header_overrides, id0_only=False):
    """Layout B: per-rank VTK files in an idN/ tree.

    Handles partially-merged trees too: the effective meshblock dims come
    from the per-file ``Nx``, so a tree that has been merged along some
    axes (e.g. x-y merged, z-decomposition kept) loads with the merged
    block size. The athinput's ``NGrid_*`` keys are not required to match
    the file count.
    """
    physics = _athinput_physics(par)
    gamma, iso_csound = physics['gamma'], physics['iso_csound']
    files = _vtk_discover_files(filename, id0_only=id0_only)
    if not files:
        raise ValueError(f"no VTK files found under {osp.dirname(filename)!r}")

    grids = [_vtk_set_grid(f) for f in files]
    field_maps = [_vtk_set_field_map(g) for g in grids]

    if id0_only:
        # Single-block partial load. NGrid_eff must be (1,1,1) for the
        # downstream geometry to be coherent; tell the user otherwise.
        if tuple(int(x) for x in grids[0]['Nx']) != tuple(mesh['Nx']):
            raise NotImplementedError(
                "id0_only=True only supports single-meshblock domains "
                "(per-file Nx must equal athinput global Nx). Pass the "
                "whole idN/ tree or use layout='merged'."
            )
        effective_mesh = _effective_mesh_from_files(
            grids, mesh, source_label=files[0])
    else:
        effective_mesh = _effective_mesh_from_files(
            grids, mesh, source_label=osp.dirname(filename))

    has_pressure = any('pressure' in fm for fm in field_maps)
    has_mhd = any('cell_centered_B' in fm for fm in field_maps)
    n_mbs = effective_mesh['n_mbs']
    block_data = [{} for _ in range(n_mbs)]
    ref_time = float(grids[0]['time'])

    # Order grids by logical (i, j, k) so we agree with _build_mb_geometry.
    NGrid = effective_mesh['NGrid']
    Lmb = [(effective_mesh['xmax'][a] - effective_mesh['xmin'][a]) / NGrid[a]
           for a in range(3)]
    order = []
    for g in grids:
        le = g['le']
        i = int(round((le[0] - effective_mesh['xmin'][0]) / Lmb[0])) if Lmb[0] > 0 else 0
        j = int(round((le[1] - effective_mesh['xmin'][1]) / Lmb[1])) if Lmb[1] > 0 else 0
        k = int(round((le[2] - effective_mesh['xmin'][2]) / Lmb[2])) if Lmb[2] > 0 else 0
        flat = (k * NGrid[1] + j) * NGrid[0] + i
        order.append(flat)
    perm = sorted(range(len(grids)), key=lambda idx: order[idx])

    for slot, file_idx in enumerate(perm):
        g = grids[file_idx]
        fm = field_maps[file_idx]
        density = None
        if 'density' in fm:
            arr = _vtk_read_array(g, 'density', fm)
            density = _populate_block_from_array(
                block_data[slot], 'density', arr, gamma=gamma)
        for vtk_field in fm:
            if vtk_field == 'density':
                continue
            if vtk_field not in ('velocity', 'momentum', 'cell_centered_B', 'pressure'):
                continue
            arr = _vtk_read_array(g, vtk_field, fm)
            _populate_block_from_array(
                block_data[slot], vtk_field, arr,
                gamma=gamma, density=density)

    return _assemble_filedata(
        block_data, effective_mesh, time=ref_time,
        gamma=gamma, iso_csound=iso_csound,
        has_mhd=has_mhd, has_pressure=has_pressure,
        physics=physics, header_overrides=header_overrides,
    )


def _read_layout_tar(filename, par, mesh, header_overrides):
    """Layout C: a single tar bundle holding the pyathena idN/ structure."""
    import tarfile

    if not filename.endswith('.tar'):
        # The user passed a VTK file but layout was forced to 'tar'; assume
        # the tar lives next to it.
        d, problem_id, num, _suffix, _ext, _mpi, _ = _vtk_parse_filename(filename)
        tar_path = osp.join(d, f'{problem_id}.{num}.tar')
        if not osp.exists(tar_path):
            raise FileNotFoundError(
                f"no tar bundle found at {tar_path!r}"
            )
    else:
        tar_path = filename

    tf = tarfile.open(tar_path, 'r')
    # Eagerly collect VTK members; the tarfile object stays open for the
    # duration of this read because we'll re-extract per member.
    members = [m for m in tf.getmembers()
               if m.isfile() and m.name.endswith('.vtk')]
    if not members:
        tf.close()
        raise ValueError(f"no .vtk members found inside {tar_path!r}")

    def make_open_fn(member):
        # tarfile.extractfile returns a non-seekable ExFileObject; copy into
        # a BytesIO so _vtk_set_field_map / _vtk_read_array can seek.
        import io
        def _open(_member=member):
            data = tf.extractfile(_member).read()
            return io.BytesIO(data)
        return _open

    physics = _athinput_physics(par)
    gamma, iso_csound = physics['gamma'], physics['iso_csound']
    grids = []
    field_maps = []
    for m in members:
        g = _vtk_set_grid(make_open_fn(m))
        # Replace the generic label with the tar member name for diagnostics.
        g['filename'] = f'{tar_path}::{m.name}'
        grids.append(g)
        field_maps.append(_vtk_set_field_map(g))

    try:
        effective_mesh = _effective_mesh_from_files(
            grids, mesh, source_label=tar_path)
    except Exception:
        tf.close()
        raise

    has_pressure = any('pressure' in fm for fm in field_maps)
    has_mhd = any('cell_centered_B' in fm for fm in field_maps)
    n_mbs = effective_mesh['n_mbs']
    block_data = [{} for _ in range(n_mbs)]
    ref_time = float(grids[0]['time'])

    NGrid = effective_mesh['NGrid']
    Lmb = [(effective_mesh['xmax'][a] - effective_mesh['xmin'][a]) / NGrid[a]
           for a in range(3)]
    order = []
    for g in grids:
        le = g['le']
        i = int(round((le[0] - effective_mesh['xmin'][0]) / Lmb[0])) if Lmb[0] > 0 else 0
        j = int(round((le[1] - effective_mesh['xmin'][1]) / Lmb[1])) if Lmb[1] > 0 else 0
        k = int(round((le[2] - effective_mesh['xmin'][2]) / Lmb[2])) if Lmb[2] > 0 else 0
        flat = (k * NGrid[1] + j) * NGrid[0] + i
        order.append(flat)
    perm = sorted(range(len(grids)), key=lambda idx: order[idx])

    for slot, file_idx in enumerate(perm):
        g = grids[file_idx]
        fm = field_maps[file_idx]
        density = None
        if 'density' in fm:
            arr = _vtk_read_array(g, 'density', fm)
            density = _populate_block_from_array(
                block_data[slot], 'density', arr, gamma=gamma)
        for vtk_field in fm:
            if vtk_field == 'density':
                continue
            if vtk_field not in ('velocity', 'momentum', 'cell_centered_B', 'pressure'):
                continue
            arr = _vtk_read_array(g, vtk_field, fm)
            _populate_block_from_array(
                block_data[slot], vtk_field, arr,
                gamma=gamma, density=density)
    tf.close()

    return _assemble_filedata(
        block_data, effective_mesh, time=ref_time,
        gamma=gamma, iso_csound=iso_csound,
        has_mhd=has_mhd, has_pressure=has_pressure,
        physics=physics, header_overrides=header_overrides,
    )


def _read_layout_per_var(filename, par, mesh, header_overrides, extra_files=None):
    """Layout D: multiple merged-grid VTK files, one variable group each.

    Each sibling ``<problem>.<num>.<suffix>.vtk`` is treated as a Layout A
    file; we read its arrays, slice into NGrid meshblocks, and merge with
    the others into one ``block_data`` list.
    """
    physics = _athinput_physics(par)
    gamma, iso_csound = physics['gamma'], physics['iso_csound']
    dirname, problem_id, num, _suffix, ext, _mpi, _ = _vtk_parse_filename(filename)
    if extra_files is None:
        pattern = osp.join(dirname, f'{problem_id}.{num}.*.{ext}')
        siblings = sorted(set(glob.glob(pattern)) | {filename})
    else:
        siblings = sorted(set(extra_files) | {filename})

    if not siblings:
        raise ValueError(f"no readable VTK files for {filename!r}")

    # Derive the local mesh from the first file (per-var siblings must agree
    # on geometry; we cross-check below).
    ref_grid = _vtk_set_grid(siblings[0])
    local_mesh = _local_mesh_from_grid(ref_grid, mesh['nx_mb'])
    ref_time = float(ref_grid['time'])
    has_pressure = False
    has_mhd = False
    block_data = [{} for _ in range(local_mesh['n_mbs'])]

    # First pass: read density (whichever file holds it) so subsequent
    # momentum->velocity conversions have it on hand.
    for fpath in siblings:
        g = _vtk_set_grid(fpath)
        fm = _vtk_set_field_map(g)
        if not np.isclose(float(g['time']), ref_time):
            raise ValueError(
                f"per-var files disagree on time: {fpath!r} reports "
                f"{g['time']} vs reference {ref_time}"
            )
        if tuple(int(x) for x in g['Nx']) != local_mesh['Nx']:
            raise ValueError(
                f"{fpath!r}: file Nx={tuple(g['Nx'])} doesn't match "
                f"reference Nx={local_mesh['Nx']}; per-var files must agree "
                f"on geometry"
            )
        if 'density' in fm:
            arr_full = _vtk_read_array(g, 'density', fm)
            for i, sub in enumerate(_slice_full_grid_into_blocks(arr_full, local_mesh)):
                _populate_block_from_array(
                    block_data[i], 'density',
                    np.ascontiguousarray(sub), gamma=gamma)

    # Second pass: read remaining fields, now that density is in place.
    for fpath in siblings:
        g = _vtk_set_grid(fpath)
        fm = _vtk_set_field_map(g)
        if 'cell_centered_B' in fm:
            has_mhd = True
        if 'pressure' in fm:
            has_pressure = True
        for vtk_field in fm:
            if vtk_field == 'density':
                continue
            if vtk_field not in ('velocity', 'momentum', 'cell_centered_B', 'pressure'):
                continue
            arr_full = _vtk_read_array(g, vtk_field, fm)
            for i, sub in enumerate(_slice_full_grid_into_blocks(arr_full, local_mesh)):
                density = block_data[i].get('dens')
                _populate_block_from_array(
                    block_data[i], vtk_field,
                    np.ascontiguousarray(sub),
                    gamma=gamma, density=density)

    return _assemble_filedata(
        block_data, local_mesh, time=ref_time,
        gamma=gamma, iso_csound=iso_csound,
        has_mhd=has_mhd, has_pressure=has_pressure,
        physics=physics, header_overrides=header_overrides,
    )


def read_athena_vtk(filename, *, athinput=None, header_overrides=None,
                    layout='auto', id0_only=False, problem_id=None,
                    extra_vtk_files=None):
    """Read an Athena 4.2 VTK snapshot into the AthenaK ``filedata`` schema.

    Four physical layouts are supported. The reader either auto-detects or
    accepts an explicit ``layout=`` override:

    - ``'merged'`` — single VTK file holding one big uniform grid covering
      the whole domain. The reader slices the arrays into
      ``NGrid_x1*NGrid_x2*NGrid_x3`` logical meshblocks per athinput.
    - ``'idN'`` — per-rank VTK files in an ``idN/`` tree
      (``id0/<problem>.<num>.vtk``, ``id1/<problem>-id1.<num>.vtk``, ...).
    - ``'tar'`` — a single ``.tar`` bundle holding the same ``idN/`` layout
      inside. Members are streamed directly via ``tarfile``; no temp-dir
      extraction. Pass either the ``.tar`` path or any VTK with a sibling
      tar.
    - ``'per_var'`` — multiple sibling files
      ``<problem>.<num>.<suffix>.vtk`` each holding a subset of variables
      but covering the full domain. The reader discovers siblings and
      merges variable groups across them (override the discovery with
      ``extra_vtk_files=[...]``).

    Parameters
    ----------
    filename : str
        Path to one VTK file (or a ``.tar`` bundle).
    athinput : str or dict, optional
        Path to an athinput / ``.par`` file, or an already-parsed dict.
        Auto-discovered upward from ``filename`` when omitted.
    header_overrides : dict, optional
        ``{block: {key: value}}`` merged into the synthesised AthenaK
        header. Useful for attaching a ``<units>`` block.
    layout : str, optional
        One of ``'auto'`` (default), ``'merged'``, ``'idN'``, ``'tar'``,
        ``'per_var'``. Auto-detection inspects the path and its siblings.
    id0_only : bool, optional
        Layout ``'idN'`` only: read just the supplied file, skipping the
        sibling glob. Requires a single-meshblock domain.
    problem_id : str, optional
        Override the problem id inferred from the filename. Used for
        athinput discovery and MPI globbing.
    extra_vtk_files : list of str, optional
        Layout ``'per_var'`` only: replace sibling auto-discovery with an
        explicit list of paths.

    Returns
    -------
    dict
        ``filedata`` dict with the same keys as :func:`read_binary`.
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(
            f"layout={layout!r} is not one of {VALID_LAYOUTS}"
        )
    parsed_pid = _vtk_parse_filename(filename)[1]
    if problem_id is None:
        problem_id = parsed_pid

    par = _resolve_athinput(athinput, filename, problem_id)
    mesh = _athinput_mesh(par)

    if layout == 'auto':
        layout = _detect_vtk_layout(filename)

    if layout == 'merged':
        return _read_layout_merged(filename, par, mesh, header_overrides)
    if layout == 'idN':
        return _read_layout_idN(filename, par, mesh, header_overrides,
                                id0_only=id0_only)
    if layout == 'tar':
        return _read_layout_tar(filename, par, mesh, header_overrides)
    if layout == 'per_var':
        return _read_layout_per_var(filename, par, mesh, header_overrides,
                                    extra_files=extra_vtk_files)
    # Unreachable — VALID_LAYOUTS check above.
    raise ValueError(f"layout={layout!r} not handled")
