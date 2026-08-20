"""
Core AthenaData class for handling Athenak simulation data.
"""
from pathlib import Path
import numpy as np
import warnings
import json
import h5py
import pickle

from .io_utils import read_binary, load_dict_from_hdf5, save_dict_to_hdf5
from .units import Units
from . import utils
from . import particle_io

class AthenaData:
    """
    Class for handling and analyzing Athena++ simulation data.
    """
    def __init__(self, num=0, version='1.0', cache=False):
        """
        Initialize an AthenaData object.
        
        Parameters
        ----------
        num : int, optional
            File number
        version : str, optional
            Data format version
        """
        self.num = num
        self.version = version
        self._header = {}
        self.binary = {}
        self.coord = {}
        self.data_raw = {}
        self.data_func = {}
        self.dist = {}
        self.dist2d = {}
        self.vert = {}
        self.rad = {}
        self.ener_flux_profs = {}
        self.slice = {}
        self.sum = {}
        self.max = {}
        self.min = {}
        self.avg = {}
        self.spectra = {}
        self.sf = {}
        self.cf = {}  # Correlation functions
        self.area_mc = {}
        self.units = None
        self.is_mhd = False
        self.vertical_profile_func = None
        self.radial_profile_func = None
        self.gradient_func = None
        self.divergence_func = None
        self.curl_func = None
        # Initialize rendering dictionary for 3D visualization caching
        self.rendering = {}
        # Initialize particle data dictionary
        self.particles = {}
        # MPI-related attributes for distributed data
        self.mpi_rank = 0
        self.mpi_size = 1
        self.local_mb_start = 0  # Starting meshblock index for this rank
        self.local_mb_end = 0    # Ending meshblock index for this rank (exclusive)
        self.has_full_data = True  # True if this rank has all meshblocks
        # Control caching of derived data functions (full-domain results).
        # Default: disabled. Set to True to cache results across calls, at
        # the cost of extra memory for high-resolution runs.
        self.cache_data_funcs = bool(cache)
        
    def load(self, filename, config=True, **kwargs):
        """
        Load data from a file.
        
        Parameters
        ----------
        filename : str
            Path to the data file
        config : bool, optional
            Whether to configure the object after loading
        **kwargs : dict, optional
            Additional arguments for specific file types
            
        Returns
        -------
        self : AthenaData
            The loaded AthenaData object
        """
        self.filename = filename
        if (filename.endswith('.bin')):
            self.binary_name = filename
            self.load_binary(filename, **kwargs)
        elif (filename.endswith('.athdf')):
            self.athdf_name = filename
            self.load_athdf(filename, **kwargs)
        elif (filename.endswith(('.h5','.hdf5', '.h5data'))):
            self.hdf5_name = filename
            self.load_hdf5(filename, **kwargs)
        elif (filename.endswith('.pkl')):
            self.pickle_name = filename
            self.load_pickle(filename, **kwargs)
        else:
            raise ValueError(f"Unknown file type: {filename.split('.')[-1]}")
            
        if (config):
            self.config()
        
        return self

    def save(self, filename, except_keys=None,
             default_except_keys=None, **kwargs):
        """
        Save the AthenaData object to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the file
        except_keys : list, optional
            Additional keys to exclude from saving (added to defaults)
        default_except_keys : list, optional
            Override the default keys to exclude. By default excludes:
            ['binary', 'h5file', 'h5dic', 'coord', 'data_raw', 'data_func',
            'meshbdata_func', 'rendering', 'h5_supplement']
            This prevents raw simulation data from being saved, keeping only
            post-processed results (histograms, profiles, spectra, etc.)
        **kwargs : dict, optional
            Additional arguments for specific file types
            
        Returns
        -------
        None
        
        Notes
        -----
        When saving h5data files, raw data (data_raw) is automatically excluded
        to keep file sizes small. Only post-processed results like distributions,
        profiles, spectra, and structure functions are saved.
        """
        if default_except_keys is None:
            # Default exclusion list - prevents raw data from being saved
            default_except_keys = ['binary', 'h5file', 'h5dic', 'coord',
                                   'data_raw', 'data_func', 'meshbdata_func',
                                   'rendering', 'h5_supplement']
        
        if except_keys is None:
            except_keys = []
        
        # For h5data files, ensure data_raw is always excluded
        if filename.endswith('.h5data'):
            if 'data_raw' not in default_except_keys and 'data_raw' not in except_keys:
                except_keys = list(except_keys) + ['data_raw']
            print(f"Saving post-processed data only (raw data excluded)")
            
        dic = {}
        for k, v in self.__dict__.items():
            if (k not in except_keys + default_except_keys and not callable(v)):
                dic[k] = v
        
        # Verify data_raw is not in the dictionary for h5data files
        if filename.endswith('.h5data') and 'data_raw' in dic:
            print("Warning: Removing data_raw from h5data save")
            del dic['data_raw']
        
        if (filename.endswith(('.h5','.hdf5', '.h5data'))):
            self.save_hdf5(dic, filename, **kwargs)
        elif (filename.endswith(('.p','.pkl'))):
            self.save_pickle(dic, filename, **kwargs)
        else:
            raise ValueError(f"Unknown file type: {filename.split('.')[-1]}")

    def load_binary(self, filename):
        """
        Load data from an Athena++ binary file.
        
        Parameters
        ----------
        filename : str
            Path to the binary file
            
        Returns
        -------
        None
        """
        self._load_from_binary(read_binary(filename))
        
    def load_athdf(self, filename):
        """
        Load data from an Athena++ HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to the HDF5 file
            
        Returns
        -------
        None
        """
        self._load_from_athdf(filename)
        
    def load_pickle(self, filename, **kwargs):
        """
        Load data from a pickle file.
        
        Parameters
        ----------
        filename : str
            Path to the pickle file
        **kwargs : dict, optional
            Additional arguments
            
        Returns
        -------
        None
        """
        self._load_from_dic(pickle.load(open(filename, 'rb')), **kwargs)
        
    def load_hdf5(self, filename, **kwargs):
        """
        Load data from an HDF5 file.
        
        Parameters
        ----------
        filename : str
            Path to the HDF5 file
        **kwargs : dict, optional
            Additional arguments
            
        Returns
        -------
        None
        """
        self._load_from_dic(load_dict_from_hdf5(filename), **kwargs)
        
    def save_hdf5(self, dic, filename):
        """
        Save dictionary to an HDF5 file.
        
        Parameters
        ----------
        dic : dict
            Dictionary to save
        filename : str
            Path to save the HDF5 file
            
        Returns
        -------
        None
        """
        save_dict_to_hdf5(dic, filename)
        
    def save_pickle(self, dic, filename):
        """
        Save dictionary to a pickle file.
        
        Parameters
        ----------
        dic : dict
            Dictionary to save
        filename : str
            Path to save the pickle file
            
        Returns
        -------
        None
        """
        pickle.dump(dic, open(filename, 'wb'))
        
    def _load_from_dic(self, dic, except_keys=None):
        """
        Load data from a dictionary.
        
        Parameters
        ----------
        dic : dict
            Dictionary containing the data
        except_keys : list, optional
            Keys to exclude from loading
            
        Returns
        -------
        None
        """
        if except_keys is None:
            except_keys = ['header', 'data', 'binary', 'coord', 'data_raw']
            
        for k, v in dic.items():
            if (k not in except_keys):
                self.__dict__[k] = v
                
    def _load_from_binary(self, binary):
        """
        Load data from a binary dictionary.
        
        For MPI runs, only loads the meshblocks assigned to this rank to save memory.
        Call set_mpi_distribution() before loading to enable distributed loading.
        
        Parameters
        ----------
        binary : dict
            Dictionary containing the binary data
            
        Returns
        -------
        None
        """
        self.binary = binary
        self._load_from_dic(self.binary)
        
        # Set mesh metadata
        self.n_mbs = binary['n_mbs']
        self.mb_logical = np.asarray(binary['mb_logical'])
        self.mb_geometry = np.asarray(binary['mb_geometry'])
        
        # Finalize MPI distribution now that we know n_mbs
        self._finalize_mpi_distribution()
        
        # Determine which meshblocks to load
        if self.has_full_data:
            # Load all meshblocks (single process or non-MPI mode)
            mb_start = 0
            mb_end = self.n_mbs
        else:
            # Load only meshblocks for this rank (MPI mode)
            mb_start = self.local_mb_start
            mb_end = self.local_mb_end
        
        # Load data for assigned meshblocks only
        for var in self.binary['var_names']:
            self.data_raw[var] = np.asarray(binary['mb_data'][var][mb_start:mb_end])
            
        self._config_header(self.binary['header'])
        self._config_attrs_from_header()
        
    def _load_from_athdf(self, filename):
        """
        Load data from an Athena++ HDF5 file.
        
        For MPI runs, only loads the meshblocks assigned to this rank to save memory.
        Call set_mpi_distribution() before loading to enable distributed loading.
        
        Parameters
        ----------
        filename : str
            Path to the HDF5 file
            
        Returns
        -------
        None
        """
        h5file = h5py.File(filename, mode='r')
        self.h5file = h5file
        self._config_header(self.h5file.attrs['Header'])
        self._config_attrs_from_header()
        self.time = self.h5file.attrs['Time']
        self.cycle = self.h5file.attrs['NumCycles']
        self.n_mbs = self.h5file.attrs['NumMeshBlocks']
        
        # Finalize MPI distribution now that we know n_mbs
        self._finalize_mpi_distribution()
        
        # Determine which meshblocks to load
        if self.has_full_data:
            # Load all meshblocks (single process or non-MPI mode)
            mb_start = 0
            mb_end = self.n_mbs
        else:
            # Load only meshblocks for this rank (MPI mode)
            mb_start = self.local_mb_start
            mb_end = self.local_mb_end
        
        # Load metadata arrays (small, so keep full arrays on all ranks)
        # These are needed for coordinate calculations and global mesh understanding
        x1f_full = h5file['x1f'][()]
        x2f_full = h5file['x2f'][()]
        x3f_full = h5file['x3f'][()]
        
        self.mb_logical = np.append(
            h5file['LogicalLocations'][()], 
            h5file['Levels'][()].reshape(-1, 1),
            axis=1
        )
        self.mb_geometry = np.asarray([
            x1f_full[:, 0], x1f_full[:, -1],
            x2f_full[:, 0], x2f_full[:, -1],
            x3f_full[:, 0], x3f_full[:, -1],
        ]).T
        
        # Load only the data for the meshblocks assigned to this rank
        # Read directly from HDF5 file with slicing to avoid loading full arrays
        n_var_read = 0
        for ds_n, num in enumerate(self.h5file.attrs['NumVariables']):
            dataset_name = self.h5file.attrs['DatasetNames'][ds_n].decode("utf-8")
            for i in range(num):
                var = self.h5file.attrs['VariableNames'][n_var_read+i].decode("utf-8")
                # Read only the needed meshblock range directly from file
                self.data_raw[var] = np.asarray(h5file[dataset_name][i, mb_start:mb_end, :, :, :])
            n_var_read += num
    
    def set_mpi_distribution(self, rank, size):
        """
        Configure MPI-aware data distribution.
        
        Call this BEFORE loading data to enable distributed loading where each
        rank only loads its assigned meshblocks.
        
        Parameters
        ----------
        rank : int
            MPI rank of this process
        size : int
            Total number of MPI processes
        """
        self.mpi_rank = rank
        self.mpi_size = size
        self.has_full_data = (size == 1)
        
        if not self.has_full_data:
            # Will be set properly after we know n_mbs from file
            # For now just mark that we're in distributed mode
            pass
    
    def _finalize_mpi_distribution(self):
        """
        Finalize MPI distribution after n_mbs is known from file.
        
        Called internally after loading metadata but before loading data arrays.
        """
        if self.n_mbs > 0:
            if self.has_full_data:
                # Single process mode - process all meshblocks
                self.local_mb_start = 0
                self.local_mb_end = self.n_mbs
            else:
                # Distribute meshblocks across ranks
                mbs_per_rank = self.n_mbs // self.mpi_size
                remainder = self.n_mbs % self.mpi_size
                
                if self.mpi_rank < remainder:
                    self.local_mb_start = self.mpi_rank * (mbs_per_rank + 1)
                    self.local_mb_end = self.local_mb_start + mbs_per_rank + 1
                else:
                    self.local_mb_start = self.mpi_rank * mbs_per_rank + remainder
                    self.local_mb_end = self.local_mb_start + mbs_per_rank

    def global_to_local_mb(self, global_mb_start, global_mb_end=None):
        """
        Convert global meshblock indices to local indices for this rank.
        
        Returns None if the requested range is not on this rank.
        
        Parameters
        ----------
        global_mb_start : int
            Global starting meshblock index
        global_mb_end : int, optional
            Global ending meshblock index (exclusive). If None, uses global_mb_start+1
            
        Returns
        -------
        tuple or None
            (local_start, local_end) if this rank owns any of the requested meshblocks,
            None otherwise
        """
        if self.has_full_data:
            # No distribution - global and local are the same
            if global_mb_end is None:
                global_mb_end = global_mb_start + 1 if global_mb_start is not None else self.n_mbs
            return (global_mb_start if global_mb_start is not None else 0, 
                    global_mb_end if global_mb_end is not None else self.n_mbs)
        
        # Handle None values
        if global_mb_start is None:
            global_mb_start = 0
        if global_mb_end is None:
            global_mb_end = self.n_mbs
            
        # Check if this rank owns any part of the requested range
        if global_mb_end <= self.local_mb_start or global_mb_start >= self.local_mb_end:
            return None  # This rank doesn't own any of these meshblocks
        
        # Calculate local indices
        local_start = max(0, global_mb_start - self.local_mb_start)
        local_end = min(self.local_mb_end - self.local_mb_start, 
                       global_mb_end - self.local_mb_start)
        
        return (local_start, local_end)
    
    def owns_meshblock(self, global_mb_index):
        """
        Check if this rank owns the specified global meshblock.
        
        Parameters
        ----------
        global_mb_index : int
            Global meshblock index
            
        Returns
        -------
        bool
            True if this rank owns the meshblock
        """
        if self.has_full_data:
            return True
        return self.local_mb_start <= global_mb_index < self.local_mb_end
    
    def get_local_mb_count(self):
        """Get the number of meshblocks stored locally on this rank."""
        if self.has_full_data:
            return self.n_mbs
        return self.local_mb_end - self.local_mb_start
    
    def iter_local_meshblocks(self):
        """
        Iterate over locally owned meshblock indices.
        
        Yields both global and local meshblock indices.
        
        Yields
        ------
        tuple
            (global_index, local_index) for each locally owned meshblock
        """
        local_count = self.get_local_mb_count()
        for local_idx in range(local_count):
            if self.has_full_data:
                global_idx = local_idx
            else:
                global_idx = self.local_mb_start + local_idx
            yield (global_idx, local_idx)


            
    def config(self):
        """
        Configure the AthenaData object.
        
        This sets up coordinates, data functions, and other properties.
        
        Returns
        -------
        None
        """
        if (self.data_raw and not self.coord): 
            self.set_coord()
        from .data_functions import config_data_functions
        config_data_functions(self)
        
        # Ensure filename is a str (some h5data may store a bytes 'filename' field)
        if isinstance(self.filename, (bytes, bytearray)):
            try:
                self.filename = self.filename.decode('utf-8')
            except Exception:
                self.filename = str(self.filename)

        self.path = str(Path(self.filename).parent)
        self.num = int(self.filename.split('.')[-2])
        self.is_mhd = 'mhd' in self._header.keys()
        # set up vertical/radial profile functions
        try:
            from ..operations.profiles import set_vertical
            self.register_vertical_profile_func(set_vertical)
        except ImportError:
            self.register_vertical_profile_func(None)
        try:
            from ..operations.profiles import set_radial
            self.register_radial_profile_func(set_radial)
        except ImportError:
            self.register_radial_profile_func(None)
        
    def _config_header(self, header):
        """
        Configure header information.
        
        Parameters
        ----------
        header : list
            List of header lines
            
        Returns
        -------
        None
        """
        block = None
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line.strip('<').strip('>')
                self._header[block] = {}
                continue
            key, value = line.split('=')
            self._header[block][key.strip()] = value

    def header(self, blockname, keyname, astype=str, default=None):
        """
        Get a value from the header.
        
        Parameters
        ----------
        blockname : str
            Name of the header block
        keyname : str
            Name of the key
        astype : type, optional
            Type to convert the value to
        default : any, optional
            Default value if the key is not found
            
        Returns
        -------
        any
            The header value or the default
        """
        blockname = blockname.strip()
        keyname = keyname.strip()
        
        if blockname in self._header.keys():
            if keyname in self._header[blockname].keys():
                return astype(self._header[blockname][keyname])
                
        warnings.warn(f'Warning: no parameter called {blockname}/{keyname}, return default value: {default}')
        return default
    
    def _config_attrs_from_header(self):
        """
        Configure attributes from the header information.
        
        Returns
        -------
        None
        """
        self.Nx1 = json.loads(self.header('mesh', 'nx1'))
        self.Nx2 = json.loads(self.header('mesh', 'nx2'))
        self.Nx3 = json.loads(self.header('mesh', 'nx3'))
        self.nx1 = json.loads(self.header('meshblock', 'nx1'))
        self.nx2 = json.loads(self.header('meshblock', 'nx2'))
        self.nx3 = json.loads(self.header('meshblock', 'nx3'))
        self.nghost = json.loads(self.header('mesh', 'nghost'))
        self.x1min = json.loads(self.header('mesh', 'x1min'))
        self.x1max = json.loads(self.header('mesh', 'x1max'))
        self.x2min = json.loads(self.header('mesh', 'x2min'))
        self.x2max = json.loads(self.header('mesh', 'x2max'))
        self.x3min = json.loads(self.header('mesh', 'x3min'))
        self.x3max = json.loads(self.header('mesh', 'x3max'))
        
        # Initialize units from header information
        try:
            length_cgs = json.loads(self.header('units', 'length_cgs'))
            mass_cgs = json.loads(self.header('units', 'mass_cgs'))
            time_cgs = json.loads(self.header('units', 'time_cgs'))
            mu = json.loads(self.header('units', 'mu'))
            self.units = Units(lunit=length_cgs, munit=mass_cgs, tunit=time_cgs, mu=mu)
        except:
            # Use default unit values if not found in header
            length_cgs = 1.0
            mass_cgs = 1.0
            time_cgs = 1.0
            mu = 0.618  # Default mean molecular weight
            self.units = Units(lunit=length_cgs, munit=mass_cgs, tunit=time_cgs, mu=mu)
            print("Warning: Using default unit values as they were not found in the header")
            
        # Get gamma with error handling
        try:
            self.gamma = json.loads(self.header('hydro','gamma')) if 'hydro' in self._header.keys() else json.loads(self.header('mhd','gamma'))
        except (json.JSONDecodeError, KeyError, TypeError):
            print("Warning: Could not parse gamma from header, using default value 5/3")
            self.gamma = 5.0/3.0

        # Get EOS with proper error handling
        try:
            if 'hydro' in self._header.keys():
                eos_raw = self.header('hydro','eos')
            else:
                eos_raw = self.header('mhd','eos')
            self.eos = eos_raw.strip().lower()

        except (KeyError, AttributeError):
            print("Warning: EOS not found in header, assuming adiabatic")
            self.eos = 'adiabatic'
        # if eos is isothermal, also set the iso_sound_speed and set gamma = 1.0
        if self.eos == 'isothermal':
            self.iso_sound_speed = json.loads(self.header('hydro', 'iso_sound_speed')) if 'hydro' in self._header.keys() else json.loads(self.header('mhd', 'iso_sound_speed'))
            self.gamma = 1.0
        print(f"EOS: {self.eos}, gamma: {self.gamma}")
        # Update units gamma value to match simulation
        self.units.gamma = self.gamma
        
    def set_coord(self):
        """
        Set up coordinate information for each meshblock.

        Coordinates are computed on-demand from mesh geometry rather than
        stored, to keep memory usage low for large grids. This method only
        stores the metadata needed for that on-demand computation. For MPI
        runs, metadata is available for all meshblocks (needed for global
        indices).
        """
        # Initialize coordinate dictionary if not already done
        if not hasattr(self, 'coord'):
            self.coord = {}
        
        # Store metadata (no actual coordinate arrays)
        # The mesh geometry (mb_geometry) contains all information needed
        # to compute coordinates on-demand
        self.coord['_on_demand'] = True
        
        # Note: Actual coordinate computation happens in data_functions.py
        # via get_data('x'), get_data('y'), get_data('z')
        
        return
        
    def load_both_formats(self, filedir, file_num, prefix, config=True, **kwargs):
        """
        Load both athdf and h5data files for 3D visualization.
        
        This method attempts to load both file formats when available, 
        starting with athdf (preferred for 3D) and supplementing with 
        h5data (cheap to load). This provides the most complete dataset
        for visualization.
        
        Parameters
        ----------
        filedir : str
            Directory containing the files
        file_num : int
            File number to load
        prefix : str
            File prefix
        config : bool, optional
            Whether to configure after loading
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        self : AthenaData
            The loaded object
        """
        import os
        from pathlib import Path
        
        # Create filenames
        h5_suffix = '.h5data'
        athdf_suffix = '.athdf'
        h5filename = prefix + str(file_num).rjust(5, '0') + h5_suffix
        athdf_filename = prefix + str(file_num).rjust(5, '0') + athdf_suffix
        
        # Check both main directory and bin/ subdirectory
        search_paths = [filedir, os.path.join(filedir, 'bin/')]
        
        h5filepath = None
        athdf_filepath = None
        
        # Find available files
        for search_path in search_paths:
            h5_candidate = os.path.join(search_path, h5filename)
            athdf_candidate = os.path.join(search_path, athdf_filename)
            
            if os.path.exists(h5_candidate) and h5filepath is None:
                h5filepath = h5_candidate
                
            if os.path.exists(athdf_candidate) and athdf_filepath is None:
                athdf_filepath = athdf_candidate
        
        # Load athdf first (preferred for 3D coordinates and refined data)
        if athdf_filepath:
            print(f"  Primary load: athdf file from {Path(athdf_filepath).parent.name}/")
            self.load(athdf_filepath, config=config, **kwargs)
            
        # Load h5data as supplement (cheap and provides additional data)
        if h5filepath and athdf_filepath:
            try:
                print(f"  Supplemental load: h5data file from {Path(h5filepath).parent.name}/")
                # Load h5data without overwriting main attributes
                h5_data = load_dict_from_hdf5(h5filepath)
                # Store supplemental data for potential use
                if not hasattr(self, 'h5_supplement'):
                    self.h5_supplement = {}
                self.h5_supplement = h5_data
            except Exception as e:
                print(f"  Warning: Could not load supplemental h5data: {e}")
                
        elif h5filepath and not athdf_filepath:
            # Fallback to h5data only
            print(f"  Fallback load: h5data file from {Path(h5filepath).parent.name}/")
            self.load(h5filepath, config=config, **kwargs)
            
        else:
            # No files found
            searched_locations = []
            for path in search_paths:
                searched_locations.append(f"    {os.path.join(path, h5filename)}")
                searched_locations.append(f"    {os.path.join(path, athdf_filename)}")
            
            error_msg = (f"No data file found for file number {file_num}\n"
                        f"  Searched locations:\n" + "\n".join(searched_locations))
            raise FileNotFoundError(error_msg)
            
        return self

    def get_cached_or_compute(self, var, cache_key=None, compute_func=None, 
                             redo=False, cache_location='rendering', **kwargs):
        """
        Get data from cache or compute if not available.
        
        This method first checks the specified cache location for the requested
        data. If found and redo=False, returns cached data. Otherwise, computes
        the data using the provided function and caches the result.
        
        Parameters
        ----------
        var : str
            Variable name
        cache_key : str, optional
            Key to use for caching. If None, uses var name
        compute_func : callable, optional
            Function to compute the data if not cached
        redo : bool, optional
            If True, force recomputation even if cached data exists
        cache_location : str, optional
            Which cache dictionary to use ('rendering', 'data_raw', etc.)
        **kwargs : dict
            Additional arguments for compute_func
            
        Returns
        -------
        ndarray or None
            Cached or computed data
        """
        if cache_key is None:
            cache_key = var
            
        # Get cache dictionary
        if cache_location == 'rendering':
            cache_dict = self.rendering
        elif cache_location == 'data_raw':
            cache_dict = self.data_raw
        else:
            raise ValueError(f"Unknown cache location: {cache_location}")
        
        # Check cache first (unless redo is True)
        if not redo and cache_key in cache_dict:
            print(f"  ✓ Using cached {var} from {cache_location}")
            return cache_dict[cache_key]
        
        # Compute data if not cached or redo requested
        if compute_func is not None:
            print(f"  Computing new {var} data...")
            try:
                data = compute_func(var, **kwargs)
                # Cache the result
                cache_dict[cache_key] = data
                print(f"  ✓ Computed and cached {var} in {cache_location}")
                return data
            except Exception as e:
                print(f"  ⚠ Could not compute {var}: {e}")
                return None
        else:
            print(f"  ⚠ No compute function provided for {var}")
            return None

    def get(self, var, redo=False, **kwargs):
        """
        Get a rendering-cache value, computing and caching it on first access.
        
        Parameters
        ----------
        var : str
            Variable name
        redo : bool, optional
            If True, force recomputation
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        ndarray
            Variable data
        """
        # Check rendering cache first for commonly used 3D visualization variables
        viz_vars = ['temp', 'dens', 'pgas', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3']
        
        if var in viz_vars:
            cache_key = f"data_{var}"
            cached_data = self.get_cached_or_compute(
                var, 
                cache_key=cache_key,
                compute_func=self._compute_standard_var,
                redo=redo,
                **kwargs
            )
            if cached_data is not None:
                return cached_data
        
        # Fall back to standard data retrieval
        return self.data(var, **kwargs)
        
    def _compute_standard_var(self, var, **kwargs):
        """
        Compute standard variable using existing data method.
        
        Parameters
        ----------
        var : str
            Variable name
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        ndarray
            Computed variable data
        """
        return self.data(var, **kwargs)

    def get_coord(self):
        """
        Retrieve coordinate data, with caching.
        
        Returns
        -------
        tuple
            (x_mesh, y_mesh, z_mesh) coordinate arrays
        """
        cache_key = "coord_meshes"
        
        # Check cache first
        if cache_key in self.rendering:
            print("  ✓ Using cached coordinate meshes")
            return self.rendering[cache_key]
            
        # Compute coordinate meshes if not cached
        print("  Computing coordinate meshes...")
        try:
            # Use existing coordinate data to create meshes
            if not self.coord:
                self.set_coord()
                
            # Create full domain meshes by concatenating meshblock data
            x_coords = []
            y_coords = []
            z_coords = []
            
            for mb in range(self.n_mbs):
                x_coords.append(self.coord['x'][mb])
                y_coords.append(self.coord['y'][mb])
                z_coords.append(self.coord['z'][mb])
            
            # Only the first meshblock's coordinates are used here; this
            # does not assemble a true multi-meshblock domain mesh.
            x_mesh = x_coords[0]
            y_mesh = y_coords[0] 
            z_mesh = z_coords[0]
            
            coord_result = (x_mesh, y_mesh, z_mesh)
            
            # Cache the result
            self.rendering[cache_key] = coord_result
            print("  ✓ Computed and cached coordinate meshes")
            
            return coord_result
            
        except Exception as e:
            print(f"  ⚠ Could not compute coordinate meshes: {e}")
            # Return basic meshes as fallback
            return np.mgrid[self.x1min:self.x1max:complex(self.Nx1), 
                          self.x2min:self.x2max:complex(self.Nx2),
                          self.x3min:self.x3max:complex(self.Nx3)]
    
    def data(self, var, mbl=None, mbh=None, **kwargs):
        """
        Get data for the specified variable, optionally restricted to a
        meshblock subrange.

        Parameters
        ----------
        var : str or array-like
            Variable name or data array
        mbl : int, optional
            Starting meshblock index. If None, uses the full domain.
        mbh : int, optional
            Ending meshblock index. If None, uses the full domain.
        **kwargs : dict
            Additional arguments for derived variables
            
        Returns
        -------
        ndarray
            Requested data array
        """
        from .data_functions import get_data
        return get_data(self, var, mbl, mbh, **kwargs)

    def register_vertical_profile_func(self, func=None):
        """Register a function to calculate vertical profiles.
        
        Parameters
        ----------
        func : callable, optional
            Function that calculates vertical profiles. If None, will attempt to import.
        """
        if func is None:
            # Don't use relative import here - use absolute import
            try:
                import athena_research.operations.profiles as profiles_module
                self.vertical_profile_func = profiles_module.set_vertical
            except (ImportError, AttributeError):
                raise ImportError("Could not import vertical profile function")
        else:
            self.vertical_profile_func = func
        return self.vertical_profile_func

    def register_radial_profile_func(self, func=None):
        """Register a function to calculate radial profiles."""
        if func is None:
            try:
                import athena_research.operations.profiles as profiles_module
                self.radial_profile_func = profiles_module.set_radial
            except (ImportError, AttributeError):
                raise ImportError("Could not import radial profile function")
        else:
            self.radial_profile_func = func
        return self.radial_profile_func
    
    def load_particles(self, filename=None, filedir=None, file_num=None, 
                      prefix='prtcl', suffix='prtclbin', **kwargs):
        """
        Load particle data from binary, HDF5, or VTK files.
        
        This method automatically detects the file format and loads particle data
        along with any grid quantities interpolated to particle locations.
        
        Parameters
        ----------
        filename : str, optional
            Full path to particle file. If provided, filedir/file_num/prefix are ignored.
        filedir : str, optional
            Directory containing particle files
        file_num : int, optional
            File number to load. If None, uses self.num
        prefix : str, optional
            Filename prefix (default: 'prtcl')
        suffix : str, optional
            File suffix/extension (default: 'prtclbin')
            Supported: 'prtclbin' (binary), 'h5' (HDF5), 'vtk' (VTK)
        **kwargs : dict
            Additional arguments (for future extensions)
            
        Returns
        -------
        self : AthenaData
            Returns self for method chaining
            
        Examples
        --------
        >>> ad = AthenaData(num=10)
        >>> ad.load('simulation.00010.athdf')
        >>> ad.load_particles(filedir='pbin', prefix='TRML')
        >>> print(ad.particles['nparticles'])
        >>> print(ad.particles['x'], ad.particles['y'], ad.particles['z'])
        >>> print(ad.particles['dens'])  # density interpolated to particles
        """
        # Determine filename
        if filename is None:
            if filedir is None:
                raise ValueError("Either filename or filedir must be provided")
            if file_num is None:
                file_num = self.num
            
            # Construct filename
            filename = f"{filedir}/{prefix}.{file_num:05d}.{suffix}"
        
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Particle file not found: {filename}")
        
        # Load based on file extension
        if filename.suffix == '.prtclbin' or suffix == 'prtclbin':
            self.particles = particle_io.read_particle_binary(filename)
        elif filename.suffix in ['.h5', '.hdf5'] or suffix in ['h5', 'hdf5']:
            self.particles = particle_io.read_particle_hdf5(filename)
        elif filename.suffix == '.vtk' or suffix == 'vtk':
            self.particles = particle_io.read_particle_vtk(filename)
        else:
            raise ValueError(f"Unsupported particle file format: {filename.suffix}")
        
        print(f"Loaded {self.particles['nparticles']} particles from {filename}")
        print(f"Time: {self.particles['time']:.6e}, Cycle: {self.particles['ncycle']}")
        if len(self.particles['var_names']) > 0:
            print(f"Grid variables at particle locations: {self.particles['var_names']}")
        
        return self
    
    def get_particle_data(self, var):
        """
        Get particle data for a given variable.
        
        Parameters
        ----------
        var : str
            Variable name. Can be:
            - Particle positions: 'x', 'y', 'z'
            - Particle velocities: 'velx', 'vely', 'velz'
            - Grid data interpolated to particles: any var from var_names
            - Derived quantities: 'r', 'R', 'vtot', etc.
            
        Returns
        -------
        array
            Particle data array
            
        Examples
        --------
        >>> ad.load_particles('pbin/TRML.00001.prtclbin')
        >>> x = ad.get_particle_data('x')
        >>> dens = ad.get_particle_data('dens')
        >>> r = ad.get_particle_data('r')  # radial distance
        """
        if not self.particles:
            raise ValueError("No particle data loaded. Call load_particles() first.")
        
        # Direct particle data
        if var in self.particles:
            return self.particles[var]
        
        # Try to interpolate from grid if variable not found in particles
        # This is a fallback for when grid variables are not included in particle output
        if hasattr(self, 'data') and callable(getattr(self, 'data', None)):
            try:
                # Check if it's a velocity component that might be available in grid
                if var in ['velx', 'vely', 'velz', 'vx', 'vy', 'vz']:
                    grid_var = var
                    if var in ['vx', 'vy', 'vz']:
                        # Map vx,vy,vz to velx,vely,velz
                        grid_var = 'vel' + var[1]
                    if grid_var in ['velx', 'vely', 'velz']:
                        # Proper grid interpolation onto particle positions is not
                        # implemented; the zeros below are NOT real data.
                        if 'nparticles' in self.particles:
                            warnings.warn(
                                f"'{var}' not found in particle data and grid "
                                "interpolation is not implemented; returning all-zero "
                                "placeholder data, not physically meaningful values."
                            )
                            return np.zeros(self.particles['nparticles'])
                elif var in ['dens', 'temp', 'pres', 'eint'] and hasattr(self, 'particles') and 'x' in self.particles:
                    # Same caveat as above: no real interpolation implemented.
                    if 'nparticles' in self.particles:
                        warnings.warn(
                            f"'{var}' not found in particle data and grid "
                            "interpolation is not implemented; returning all-zero "
                            "placeholder data, not physically meaningful values."
                        )
                        return np.zeros(self.particles['nparticles'])
            except:
                pass  # Fall through to error
        
        # Derived quantities
        if var == 'r':
            return np.sqrt(self.particles['x']**2 + 
                          self.particles['y']**2 + 
                          self.particles['z']**2)
        elif var == 'R':
            return np.sqrt(self.particles['x']**2 + self.particles['y']**2)
        elif var == 'vtot':
            if 'velx' in self.particles:
                return np.sqrt(self.particles['velx']**2 + 
                              self.particles['vely']**2 + 
                              self.particles['velz']**2)
        elif var == 'theta':
            r = self.get_particle_data('r')
            return np.arccos(self.particles['z'] / r)
        elif var == 'phi':
            return np.arctan2(self.particles['y'], self.particles['x'])
        
        raise ValueError(f"Variable '{var}' not found in particle data")