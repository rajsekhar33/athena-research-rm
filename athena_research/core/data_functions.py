"""
Data functions for Athenak data analysis.

This module provides common data functions used across the athena_analysis package.
"""
import numpy as np
from .utils import xyz_bool

from .base import xp

# Placeholder generic defaults for the Thotbool/Tcoldbool/Trangebool derived
# variables below -- this package has no built-in notion of a physically
# meaningful hot/cold threshold, so callers should pass real T_hot/T_cold
# values via kwargs for any physically meaningful result.
Thot = 1.0
Tcold = 1.0

# List of derived variables that can be computed
derived_var_list = [
    'xyzbool', 'xyzbool_vol', 'xyzbool_mass',
    'tcool', 'Tcoldbool', 'Tcoldbool_vol', 'Tcoldbool_mass',
    'Thotbool', 'Thotbool_vol', 'Thotbool_mass',
    'Trangebool', 'Trangebool_vol', 'Trangebool_mass',
    'nrho', 'nprs', 'nrho_hot', 'nprs_hot', 'nrho_range', 'nprs_range',
    'nvelx', 'nvely', 'nvelz', 
    'eflxwave', 'eflxwave_hot', 'eflxwave_range',
    'delrhodelT', 'delrhodelT_hot', 'delrhodelT_range',
    'delvrdelT', 'delvrdelT_hot', 'delvrdelT_range',
    'curx', 'cury', 'curz', 'cur',
    'sqrt_rho',  # Square root of density for weighting
]

def config_data_functions(self):
    """
    Configure derived data functions that work with both domain and meshblock data.
    
    This unified approach allows the same functions to work across the full domain
    or for specific meshblock ranges, making the code more maintainable.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData object to configure
        
    Returns
    -------
    None
        Modifies self.data_func in place
    """
    
    # Configure the data_func and meshbdata_func dictionaries with lambda functions
    # that can accept optional meshblock range parameters
    
    
    print(f"Using temperature thresholds: T_cold={Tcold}, T_hot={Thot}")
    
    # Coordinate computation functions (computed on-demand from mesh geometry)
    def _compute_coords(self, coord_name, mbl=None, mbh=None):
        """Compute coordinates on-demand from mesh geometry to save memory."""
        # Determine global meshblock range
        global_mbl = mbl if mbl is not None else 0
        global_mbh = mbh if mbh is not None else len(self.mb_geometry)
        
        # Handle MPI distribution: only compute for locally-owned blocks
        if (mbl is not None or mbh is not None) and not self.has_full_data:
            # Compute intersection of requested range with locally-owned range
            actual_mbl = max(global_mbl, self.local_mb_start)
            actual_mbh = min(global_mbh, self.local_mb_end)
            if actual_mbl >= actual_mbh:
                # This rank doesn't own any of the requested meshblocks
                return xp.zeros((0, self.nx3, self.nx2, self.nx1))
        else:
            # Not using MPI or requesting full domain
            actual_mbl = global_mbl
            actual_mbh = global_mbh
        
        # Get meshblock geometry for the blocks this rank will compute
        mb_geo = xp.asarray(self.mb_geometry[actual_mbl:actual_mbh], dtype=xp.float64)
        
        # The number of meshblocks is determined by mb_geo size
        n_mbs = len(mb_geo)
        nx1, nx2, nx3 = self.nx1, self.nx2, self.nx3
        
        # Handle empty range
        if n_mbs == 0:
            return xp.zeros((0, nx3, nx2, nx1))
        
        # Create coordinate centers for each meshblock using vectorized broadcasting.
        if coord_name == 'x':
            frac = (xp.arange(nx1, dtype=xp.float64) + 0.5) / nx1
            coord_centers = mb_geo[:, 0:1] + (mb_geo[:, 1:2] - mb_geo[:, 0:1]) * frac[xp.newaxis, :]
            coord_3d = xp.broadcast_to(coord_centers[:, xp.newaxis, xp.newaxis, :],
                                       (n_mbs, nx3, nx2, nx1))
        elif coord_name == 'y':
            frac = (xp.arange(nx2, dtype=xp.float64) + 0.5) / nx2
            coord_centers = mb_geo[:, 2:3] + (mb_geo[:, 3:4] - mb_geo[:, 2:3]) * frac[xp.newaxis, :]
            coord_3d = xp.broadcast_to(coord_centers[:, xp.newaxis, :, xp.newaxis],
                                       (n_mbs, nx3, nx2, nx1))
        elif coord_name == 'z':
            frac = (xp.arange(nx3, dtype=xp.float64) + 0.5) / nx3
            coord_centers = mb_geo[:, 4:5] + (mb_geo[:, 5:6] - mb_geo[:, 4:5]) * frac[xp.newaxis, :]
            coord_3d = xp.broadcast_to(coord_centers[:, :, xp.newaxis, xp.newaxis],
                                       (n_mbs, nx3, nx2, nx1))
        else:
            raise ValueError(f"Unknown coordinate: {coord_name}")
        
        return xp.asarray(coord_3d)
    
    def _compute_cell_width(self, dim_name, mbl=None, mbh=None):
        """Compute cell widths on-demand from mesh geometry."""
        # Determine global meshblock range
        global_mbl = mbl if mbl is not None else 0
        global_mbh = mbh if mbh is not None else len(self.mb_geometry)
        
        # Handle MPI distribution: only compute for locally-owned blocks
        if (mbl is not None or mbh is not None) and not self.has_full_data:
            # Compute intersection of requested range with locally-owned range
            actual_mbl = max(global_mbl, self.local_mb_start)
            actual_mbh = min(global_mbh, self.local_mb_end)
            if actual_mbl >= actual_mbh:
                # This rank doesn't own any of the requested meshblocks
                if dim_name in ['dx_mb', 'dy_mb', 'dz_mb']:
                    return xp.zeros(0)
                else:
                    return xp.zeros((0, self.nx3, self.nx2, self.nx1))
        else:
            # Not using MPI or requesting full domain
            actual_mbl = global_mbl
            actual_mbh = global_mbh
        
        # Get meshblock geometry for the blocks this rank will compute
        mb_geo = xp.asarray(self.mb_geometry[actual_mbl:actual_mbh], dtype=xp.float64)
        n_mbs = len(mb_geo)
        
        nx1, nx2, nx3 = self.nx1, self.nx2, self.nx3
        
        # Handle empty range
        if n_mbs == 0:
            if dim_name in ['dx_mb', 'dy_mb', 'dz_mb']:
                return xp.zeros(0)
            else:
                return xp.zeros((0, nx3, nx2, nx1))
        
        # Compute cell widths with broadcasting instead of Python loops over meshblocks.
        if dim_name == 'dx':
            widths_1d = xp.asarray((mb_geo[:, 1] - mb_geo[:, 0]) / nx1, dtype=xp.float64)
            widths = xp.broadcast_to(widths_1d[:, xp.newaxis, xp.newaxis, xp.newaxis],
                                     (n_mbs, nx3, nx2, nx1))
        elif dim_name == 'dy':
            widths_1d = xp.asarray((mb_geo[:, 3] - mb_geo[:, 2]) / nx2, dtype=xp.float64)
            widths = xp.broadcast_to(widths_1d[:, xp.newaxis, xp.newaxis, xp.newaxis],
                                     (n_mbs, nx3, nx2, nx1))
        elif dim_name == 'dz':
            widths_1d = xp.asarray((mb_geo[:, 5] - mb_geo[:, 4]) / nx3, dtype=xp.float64)
            widths = xp.broadcast_to(widths_1d[:, xp.newaxis, xp.newaxis, xp.newaxis],
                                     (n_mbs, nx3, nx2, nx1))
        elif dim_name == 'dx_mb':
            widths = xp.asarray((mb_geo[:, 1] - mb_geo[:, 0]) / nx1, dtype=xp.float64)
        elif dim_name == 'dy_mb':
            widths = xp.asarray((mb_geo[:, 3] - mb_geo[:, 2]) / nx2, dtype=xp.float64)
        elif dim_name == 'dz_mb':
            widths = xp.asarray((mb_geo[:, 5] - mb_geo[:, 4]) / nx3, dtype=xp.float64)
        else:
            raise ValueError(f"Unknown cell width: {dim_name}")
        
        return xp.asarray(widths)
    
    # Register coordinate functions
    self.data_func['x'] = lambda self, mbl=None, mbh=None: _compute_coords(self, 'x', mbl, mbh)
    self.data_func['y'] = lambda self, mbl=None, mbh=None: _compute_coords(self, 'y', mbl, mbh)
    self.data_func['z'] = lambda self, mbl=None, mbh=None: _compute_coords(self, 'z', mbl, mbh)
    
    # Register cell width functions
    self.data_func['dx'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dx', mbl, mbh)
    self.data_func['dy'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dy', mbl, mbh)
    self.data_func['dz'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dz', mbl, mbh)
    self.data_func['dx_mb'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dx_mb', mbl, mbh)
    self.data_func['dy_mb'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dy_mb', mbl, mbh)
    self.data_func['dz_mb'] = lambda self, mbl=None, mbh=None: _compute_cell_width(self, 'dz_mb', mbl, mbh)
    
    # Basic fields
    self.data_func['zeros'] = lambda self, mbl=None, mbh=None: \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['ones'] = lambda self, mbl=None, mbh=None: \
        xp.ones(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['rand'] = lambda self, mbl=None, mbh=None: \
        xp.random.random(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['vol'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dx', mbl, mbh) * get_data(self, 'dy', mbl, mbh) * get_data(self, 'dz', mbl, mbh)
    
    self.data_func['vol_mb'] = lambda self, mbl=None, mbh=None: \
        (get_data(self, 'dx_mb', mbl, mbh) * get_data(self, 'dy_mb', mbl, mbh) * \
         get_data(self, 'dz_mb', mbl, mbh))[:, xp.newaxis, xp.newaxis, xp.newaxis] * \
         self.data_func['ones'](self, mbl, mbh)
    
    self.data_func['neg_vol'] = lambda self, mbl=None, mbh=None: \
        -self.data_func['vol'](self, mbl, mbh)
    
    # Coordinate transformations
    self.data_func['r'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(get_data(self, 'x', mbl, mbh)**2 + \
                get_data(self, 'y', mbl, mbh)**2 + \
                get_data(self, 'z', mbl, mbh)**2)
    
    self.data_func['R'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(get_data(self, 'x', mbl, mbh)**2 + get_data(self, 'y', mbl, mbh)**2)
    
    self.data_func['theta'] = lambda self, mbl=None, mbh=None: \
        xp.arccos(get_data(self, 'z', mbl, mbh) / self.data_func['r'](self, mbl, mbh))
    
    self.data_func['phi'] = lambda self, mbl=None, mbh=None: \
        xp.arctan2(get_data(self, 'y', mbl, mbh), get_data(self, 'x', mbl, mbh))
    
    self.data_func['b'] = lambda self, mbl=None, mbh=None: \
        xp.rad2deg(xp.arcsin(get_data(self, 'z', mbl, mbh) / self.data_func['r'](self, mbl, mbh)))
    
    self.data_func['l'] = lambda self, mbl=None, mbh=None: \
        xp.rad2deg(xp.arctan2(get_data(self, 'y', mbl, mbh), get_data(self, 'x', mbl, mbh)))
    
    # Solar-centric coordinates (for galactic simulations)
    self.data_func['x_sol'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'x', mbl, mbh) - 8.0 * self.units.kpc_cgs / self.units.lunit
    
    self.data_func['r_sol'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['x_sol'](self, mbl, mbh)**2 + \
                get_data(self, 'y', mbl, mbh)**2 + \
                get_data(self, 'z', mbl, mbh)**2)
    
    self.data_func['b_sol'] = lambda self, mbl=None, mbh=None: \
        xp.rad2deg(xp.arcsin(get_data(self, 'z', mbl, mbh) / self.data_func['r_sol'](self, mbl, mbh)))
    
    self.data_func['l_sol'] = lambda self, mbl=None, mbh=None: \
        xp.rad2deg(xp.arctan2(get_data(self, 'y', mbl, mbh), self.data_func['x_sol'](self, mbl, mbh)))
    
    # Mass and thermodynamics
    self.data_func['mass'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * get_data(self, 'dens', mbl, mbh)
    
    self.data_func['neg_mass'] = lambda self, mbl=None, mbh=None: \
        -self.data_func['mass'](self, mbl, mbh)
    
    self.data_func['pres'] = lambda self, mbl=None, mbh=None: \
        (self.gamma * get_data(self, 'dens', mbl, mbh) * self.iso_sound_speed**2) if self.eos == 'isothermal' else \
        ((self.gamma - 1.0) * get_data(self, 'eint', mbl, mbh))
    
    self.data_func['temp'] = lambda self, mbl=None, mbh=None: \
        (self.iso_sound_speed**2 * xp.ones(get_data(self, 'dens', mbl, mbh).shape)) if self.eos == 'isothermal' else \
        ((self.gamma - 1.0) * get_data(self, 'eint', mbl, mbh) / get_data(self, 'dens', mbl, mbh))
    
    self.data_func['temp_cgs'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'temp', mbl, mbh) * self.units.temperature_cgs
    
    self.data_func['eint'] = lambda self, mbl=None, mbh=None: \
        (get_data(self, 'dens', mbl, mbh) * self.data_func['temp'](self, mbl, mbh)) if self.eos == 'isothermal' else \
        (xp.asarray(self.data_raw['eint'][mbl:mbh]) if (mbl is not None or mbh is not None) else xp.asarray(self.data_raw['eint']))
    
    self.data_func['sqrt_rho'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(get_data(self, 'dens', mbl, mbh))
    
    self.data_func['entropy'] = lambda self, mbl=None, mbh=None: \
        self.data_func['pres'](self, mbl, mbh) / get_data(self, 'dens', mbl, mbh)**self.gamma
    
    self.data_func['cs2'] = lambda self, mbl=None, mbh=None: \
        (self.iso_sound_speed**2 * xp.ones(get_data(self, 'dens', mbl, mbh).shape)) if self.eos == 'isothermal' else \
        (self.gamma * self.data_func['pres'](self, mbl, mbh) / get_data(self, 'dens', mbl, mbh))
    
    self.data_func['cs'] = lambda self, mbl=None, mbh=None: \
        (self.iso_sound_speed * xp.ones(get_data(self, 'dens', mbl, mbh).shape)) if self.eos == 'isothermal' else \
        xp.sqrt(self.data_func['cs2'](self, mbl, mbh))
    
    # Velocity and momentum fields
    self.data_func['vtot2'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'velx', mbl, mbh)**2 + get_data(self, 'vely', mbl, mbh)**2 + get_data(self, 'velz', mbl, mbh)**2
    
    self.data_func['vtot'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['vtot2'](self, mbl, mbh))
    
    self.data_func['mach'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['vtot2'](self, mbl, mbh) / self.data_func['cs2'](self, mbl, mbh))
    
    self.data_func['mach^2'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vtot2'](self, mbl, mbh) / self.data_func['cs2'](self, mbl, mbh)
    
    self.data_func['velr'] = lambda self, mbl=None, mbh=None: \
        (get_data(self, 'velx', mbl, mbh) * get_data(self, 'x', mbl, mbh) + \
         get_data(self, 'vely', mbl, mbh) * get_data(self, 'y', mbl, mbh) + \
         get_data(self, 'velz', mbl, mbh) * get_data(self, 'z', mbl, mbh)) / \
        self.data_func['r'](self, mbl, mbh)
    
    self.data_func['vtheta'] = lambda self, mbl=None, mbh=None: \
        (get_data(self, 'z', mbl, mbh) * self.data_func['velr'](self, mbl, mbh) - \
         self.data_func['r'](self, mbl, mbh) * get_data(self, 'velz', mbl, mbh)) / \
         self.data_func['R'](self, mbl, mbh)
    
    self.data_func['vphi'] = lambda self, mbl=None, mbh=None: \
        (get_data(self, 'x', mbl, mbh) * get_data(self, 'vely', mbl, mbh) - \
         get_data(self, 'y', mbl, mbh) * get_data(self, 'velx', mbl, mbh)) / \
         self.data_func['r'](self, mbl, mbh)
    
    self.data_func['vrot'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['vtot2'](self, mbl, mbh) - \
                self.data_func['velr'](self, mbl, mbh)**2)
    
    # Momentum components
    self.data_func['momx'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'velx', mbl, mbh) * get_data(self, 'dens', mbl, mbh)
    
    self.data_func['momy'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'vely', mbl, mbh) * get_data(self, 'dens', mbl, mbh)
    
    self.data_func['momz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'velz', mbl, mbh) * get_data(self, 'dens', mbl, mbh)
    
    self.data_func['momr'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velr'](self, mbl, mbh) * get_data(self, 'dens', mbl, mbh)
    
    # Inflow/outflow velocities
    self.data_func['velin'] = lambda self, mbl=None, mbh=None: \
        xp.minimum(self.data_func['velr'](self, mbl, mbh), 0.0)
    
    self.data_func['velout'] = lambda self, mbl=None, mbh=None: \
        xp.maximum(self.data_func['velr'](self, mbl, mbh), 0.0)
    
    self.data_func['velrin'] = lambda self, mbl=None, mbh=None: \
        xp.minimum(self.data_func['velr'](self, mbl, mbh), 0.0)
    
    self.data_func['velrout'] = lambda self, mbl=None, mbh=None: \
        xp.maximum(self.data_func['velr'](self, mbl, mbh), 0.0)
    
    # Angular momentum components
    self.data_func['jx'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'y', mbl, mbh) * get_data(self, 'velz', mbl, mbh) - \
        get_data(self, 'z', mbl, mbh) * get_data(self, 'vely', mbl, mbh)
    
    self.data_func['jy'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'z', mbl, mbh) * get_data(self, 'velx', mbl, mbh) - \
        get_data(self, 'x', mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['jz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'x', mbl, mbh) * get_data(self, 'vely', mbl, mbh) - \
        get_data(self, 'y', mbl, mbh) * get_data(self, 'velx', mbl, mbh)
    
    self.data_func['momtot'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['vtot'](self, mbl, mbh)
    
    # Energy variables
    self.data_func['ekin'] = lambda self, mbl=None, mbh=None: \
        0.5 * get_data(self, 'dens', mbl, mbh) * self.data_func['vtot2'](self, mbl, mbh)
    
    self.data_func['emag'] = lambda self, mbl=None, mbh=None: \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape) if not self.is_mhd else \
        0.5 * (get_data(self, 'bcc1', mbl, mbh)**2 + \
               get_data(self, 'bcc2', mbl, mbh)**2 + \
               get_data(self, 'bcc3', mbl, mbh)**2)
    
    self.data_func['etot'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) + \
        get_data(self, 'eint', mbl, mbh) + \
        self.data_func['emag'](self, mbl, mbh)
    
    # Angular momentum totals
    self.data_func['amx'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'y', mbl, mbh) * get_data(self, 'velz', mbl, mbh) - \
        get_data(self, 'z', mbl, mbh) * get_data(self, 'vely', mbl, mbh)
    
    self.data_func['amy'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'z', mbl, mbh) * get_data(self, 'velx', mbl, mbh) - \
        get_data(self, 'x', mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['amz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'x', mbl, mbh) * get_data(self, 'vely', mbl, mbh) - \
        get_data(self, 'y', mbl, mbh) * get_data(self, 'velx', mbl, mbh)
    
    self.data_func['amtot'] = lambda self, mbl=None, mbh=None: \
        self.data_func['r'](self, mbl, mbh) * self.data_func['vrot'](self, mbl, mbh)
    
    # Radial fluxes
    self.data_func['mflxr'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh)
    
    self.data_func['mflxrin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velrin'](self, mbl, mbh)
    
    self.data_func['mflxrout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velrout'](self, mbl, mbh)
    
    self.data_func['mdot'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh)
    
    self.data_func['mdotin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velin'](self, mbl, mbh)
    
    self.data_func['mdotout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velout'](self, mbl, mbh)
    
    # Momentum fluxes
    self.data_func['momflxr'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh)**2
    
    self.data_func['momflxrin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh) * \
        self.data_func['velrin'](self, mbl, mbh)
    
    self.data_func['momflxrout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh) * \
        self.data_func['velrout'](self, mbl, mbh)
    
    self.data_func['momflx'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh)**2
    
    self.data_func['momflxin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh) * \
        self.data_func['velin'](self, mbl, mbh)
    
    self.data_func['momflxout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velr'](self, mbl, mbh) * \
        self.data_func['velout'](self, mbl, mbh)
    
    # Energy fluxes
    self.data_func['ekflxr'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velr'](self, mbl, mbh)
    
    self.data_func['ekflxrin'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velrin'](self, mbl, mbh)
    
    self.data_func['ekflxrout'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velrout'](self, mbl, mbh)
    
    self.data_func['eflxrtot'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velr'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    self.data_func['ekflx'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velr'](self, mbl, mbh)
    
    self.data_func['ekflxin'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velin'](self, mbl, mbh)
    
    self.data_func['ekflxout'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velout'](self, mbl, mbh)
    
    self.data_func['eflxtot'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velr'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    self.data_func['eflxtotin'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velin'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    self.data_func['eflxtotout'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velout'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    # Z-direction fluxes
    self.data_func['velzin'] = lambda self, mbl=None, mbh=None: \
        xp.minimum(get_data(self, 'velz', mbl, mbh), 0.0)
    
    self.data_func['velzout'] = lambda self, mbl=None, mbh=None: \
        xp.maximum(get_data(self, 'velz', mbl, mbh), 0.0)
    
    self.data_func['mflxz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['mflxzin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velzin'](self, mbl, mbh)
    
    self.data_func['mflxzout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * self.data_func['velzout'](self, mbl, mbh)
    
    self.data_func['momflxz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * get_data(self, 'velz', mbl, mbh)**2
    
    self.data_func['momflxzin'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * get_data(self, 'velz', mbl, mbh) * \
        self.data_func['velzin'](self, mbl, mbh)
    
    self.data_func['momflxzout'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'dens', mbl, mbh) * get_data(self, 'velz', mbl, mbh) * \
        self.data_func['velzout'](self, mbl, mbh)
    
    self.data_func['ekflxz'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['ekflxzin'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velzin'](self, mbl, mbh)
    
    self.data_func['ekflxzout'] = lambda self, mbl=None, mbh=None: \
        self.data_func['ekin'](self, mbl, mbh) * self.data_func['velzout'](self, mbl, mbh)
    
    self.data_func['eflxztot'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'velz', mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    self.data_func['eflxztotin'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velzin'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    self.data_func['eflxztotout'] = lambda self, mbl=None, mbh=None: \
        self.data_func['velzout'](self, mbl, mbh) * (self.data_func['ekin'](self, mbl, mbh) + \
        (self.gamma) * get_data(self, 'eint', mbl, mbh) + self.data_func['emag'](self, mbl, mbh))
    
    # Magnetic field variables
    self.data_func['bccx'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'bcc1', mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
        
    self.data_func['bccy'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'bcc2', mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
        
    self.data_func['bccz'] = lambda self, mbl=None, mbh=None: \
        get_data(self, 'bcc3', mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccr'] = lambda self, mbl=None, mbh=None: \
        (self.data_func['bccx'](self, mbl, mbh) * get_data(self, 'x', mbl, mbh) + \
         self.data_func['bccy'](self, mbl, mbh) * get_data(self, 'y', mbl, mbh) + \
         self.data_func['bccz'](self, mbl, mbh) * get_data(self, 'z', mbl, mbh)) / \
        self.data_func['r'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['btot2'] = lambda self, mbl=None, mbh=None: \
        self.data_func['bccx'](self, mbl, mbh)**2 + \
        self.data_func['bccy'](self, mbl, mbh)**2 + \
        self.data_func['bccz'](self, mbl, mbh)**2 if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['btot'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['btot2'](self, mbl, mbh))
    
    self.data_func['brot'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['btot2'](self, mbl, mbh) - \
                self.data_func['bccr'](self, mbl, mbh)**2) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['v_A^2'] = lambda self, mbl=None, mbh=None: \
        self.data_func['btot2'](self, mbl, mbh) / get_data(self, 'dens', mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
        
    self.data_func['v_A'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['v_A^2'](self, mbl, mbh))
    
    self.data_func['mach_A'] = lambda self, mbl=None, mbh=None: \
        xp.sqrt(self.data_func['vtot2'](self, mbl, mbh) / \
                self.data_func['v_A^2'](self, mbl, mbh)) if self.is_mhd else \
        xp.full(get_data(self, 'dens', mbl, mbh).shape, xp.inf)
    
    self.data_func['mach_A^2'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vtot2'](self, mbl, mbh) / \
        self.data_func['v_A^2'](self, mbl, mbh) if self.is_mhd else \
        xp.full(get_data(self, 'dens', mbl, mbh).shape, xp.inf)
    
    self.data_func['beta'] = lambda self, mbl=None, mbh=None: \
        self.data_func['pres'](self, mbl, mbh) / \
        (0.5 * self.data_func['btot2'](self, mbl, mbh)) if self.is_mhd else \
        xp.full(get_data(self, 'dens', mbl, mbh).shape, xp.inf)
    
    self.data_func['pth_plus_mag'] = lambda self, mbl=None, mbh=None: \
        self.data_func['pres'](self, mbl, mbh) + \
        (0.5 * self.data_func['btot2'](self, mbl, mbh)) if self.is_mhd else \
        self.data_func['pres'](self, mbl, mbh)
    
    # Weighted quantities for projections
    self.data_func['temp_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * self.data_func['temp'](self, mbl, mbh)
    
    self.data_func['temp_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * self.data_func['temp'](self, mbl, mbh)
    
    self.data_func['velx_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * get_data(self, 'velx', mbl, mbh)
    
    self.data_func['velx_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * get_data(self, 'velx', mbl, mbh)
    
    self.data_func['vely_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * get_data(self, 'vely', mbl, mbh)
    
    self.data_func['vely_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * get_data(self, 'vely', mbl, mbh)
    
    self.data_func['velz_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['velz_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * get_data(self, 'velz', mbl, mbh)
    
    self.data_func['bccx_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * self.data_func['bccx'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccx_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * self.data_func['bccx'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccy_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * self.data_func['bccy'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccy_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * self.data_func['bccy'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccz_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * self.data_func['bccz'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)
    
    self.data_func['bccz_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * self.data_func['bccz'](self, mbl, mbh) if self.is_mhd else \
        xp.zeros(get_data(self, 'dens', mbl, mbh).shape)

    # Current density components (code units)
    # Note: `bcc` stores B/sqrt(4*pi), so curl(bcc) == curl(B)/sqrt(4*pi)
    # j = c * curl(B)/sqrt(4*pi) => j = c * curl(bcc)
    # Helper to compute (and cache for full-domain) current components
    def _compute_and_cache_cur(self, mbl=None, mbh=None):
        """Return (curx, cury, curz) for given meshblock range.
        If mbl,mbh are None (full-domain) the result will be cached in
        self.data_raw['curx'/'cury'/'curz'/'cur'] to avoid repeated recomputation.
        """
        # Lazy import to avoid circular imports
        try:
            from ..operations.grad_div_curl import _curl_original as _curl_local
        except Exception:
            _curl_local = None

        # If not MHD or curl unavailable, return zeros of appropriate shape
        if (not getattr(self, 'is_mhd', False)) or (_curl_local is None):
            shape = get_data(self, 'dens', mbl, mbh).shape
            return xp.zeros(shape), xp.zeros(shape), xp.zeros(shape)

        # If full-domain and already cached, return cached arrays
        if mbl is None and mbh is None and 'curx' in self.data_raw:
            return xp.asarray(self.data_raw['curx']), xp.asarray(self.data_raw['cury']), xp.asarray(self.data_raw['curz'])

        # Compute curl via grad_div_curl implementation
        try:
            curl_comp = _curl_local(self, 'bcc1', 'bcc2', 'bcc3', mbl=mbl, mbh=mbh)
            curx = self.units.speed_of_light * curl_comp[0]
            cury = self.units.speed_of_light * curl_comp[1]
            curz = self.units.speed_of_light * curl_comp[2]
        except Exception:
            shape = get_data(self, 'dens', mbl, mbh).shape
            curx = xp.zeros(shape)
            cury = xp.zeros(shape)
            curz = xp.zeros(shape)

        # Cache full-domain results to avoid recomputing
        if mbl is None and mbh is None:
            try:
                self.data_raw['curx'] = xp.asarray(curx)
                self.data_raw['cury'] = xp.asarray(cury)
                self.data_raw['curz'] = xp.asarray(curz)
                self.data_raw['cur'] = xp.sqrt(self.data_raw['curx']**2 + self.data_raw['cury']**2 + self.data_raw['curz']**2)
            except Exception:
                # If caching fails, ignore and continue returning values
                pass

        return curx, cury, curz

    # Register data functions that use the helper above
    self.data_func['curx'] = lambda self, mbl=None, mbh=None: xp.asarray(_compute_and_cache_cur(self, mbl, mbh)[0])
    self.data_func['cury'] = lambda self, mbl=None, mbh=None: xp.asarray(_compute_and_cache_cur(self, mbl, mbh)[1])
    self.data_func['curz'] = lambda self, mbl=None, mbh=None: xp.asarray(_compute_and_cache_cur(self, mbl, mbh)[2])
    self.data_func['cur']  = lambda self, mbl=None, mbh=None: xp.sqrt(self.data_func['curx'](self, mbl, mbh)**2 + self.data_func['cury'](self, mbl, mbh)**2 + self.data_func['curz'](self, mbl, mbh)**2)
    
    self.data_func['beta_mw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['mass'](self, mbl, mbh) * self.data_func['beta'](self, mbl, mbh) if self.is_mhd else \
        xp.full_like(self.data_func['mass'](self, mbl, mbh), xp.inf)
    
    self.data_func['beta_vw'] = lambda self, mbl=None, mbh=None: \
        self.data_func['vol'](self, mbl, mbh) * self.data_func['beta'](self, mbl, mbh) if self.is_mhd else \
        xp.full_like(self.data_func['vol'](self, mbl, mbh), xp.inf)
    
    # Special derived fields for cooling and thermal properties
    self.data_func['tcool'] = lambda self, mbl=None, mbh=None, cooling_rate_field='coolr': \
        get_data(self, 'eint', mbl, mbh) / \
        xp.where(get_data(self, cooling_rate_field, mbl, mbh) > 0, 
                 get_data(self, cooling_rate_field, mbl, mbh), 
                 xp.inf)
    
    # Boolean mask fields for various regions
    self.data_func['Tcoldbool'] = lambda self, mbl=None, mbh=None, T_cold=Tcold: \
    xp.where(self.data_func['temp_cgs'](self, mbl, mbh) < xp.asarray(T_cold if T_cold is not None else Tcold), 
            xp.ones_like(get_data(self, 'dens', mbl, mbh)), 
            xp.zeros_like(get_data(self, 'dens', mbl, mbh)))

    self.data_func['Thotbool'] = lambda self, mbl=None, mbh=None, T_hot=Thot: \
        xp.where(self.data_func['temp_cgs'](self, mbl, mbh) >= xp.asarray(T_hot if T_hot is not None else Thot), 
                xp.ones_like(get_data(self, 'dens', mbl, mbh)), 
                xp.zeros_like(get_data(self, 'dens', mbl, mbh)))

    self.data_func['Trangebool'] = lambda self, mbl=None, mbh=None, T_min=None, T_max=None: \
        xp.where(xp.logical_and(
                    self.data_func['temp_cgs'](self, mbl, mbh) >= xp.asarray(T_min if T_min is not None else Tcold), 
                    self.data_func['temp_cgs'](self, mbl, mbh) <= xp.asarray(T_max if T_max is not None else Thot)), 
                xp.ones_like(get_data(self, 'dens', mbl, mbh)), 
                xp.zeros_like(get_data(self, 'dens', mbl, mbh)))
    
    self.data_func['xyzbool'] = lambda self, mbl=None, mbh=None, xyz=None: \
        xyz_bool(get_data(self, 'x', mbl, mbh), 
                 get_data(self, 'y', mbl, mbh),
                 get_data(self, 'z', mbl, mbh), xyz)
    
    # Volume-weighted boolean masks
    self.data_func['Tcoldbool_vol'] = lambda self, mbl=None, mbh=None, T_cold=None: \
        self.data_func['Tcoldbool'](self, mbl, mbh, T_cold) * self.data_func['vol'](self, mbl, mbh)
    
    self.data_func['Thotbool_vol'] = lambda self, mbl=None, mbh=None, T_hot=None: \
        self.data_func['Thotbool'](self, mbl, mbh, T_hot) * self.data_func['vol'](self, mbl, mbh)
    
    self.data_func['Trangebool_vol'] = lambda self, mbl=None, mbh=None, T_min=None, T_max=None: \
        self.data_func['Trangebool'](self, mbl, mbh, T_min, T_max) * self.data_func['vol'](self, mbl, mbh)
    
    self.data_func['xyzbool_vol'] = lambda self, mbl=None, mbh=None, xyz=None: \
        self.data_func['xyzbool'](self, mbl, mbh, xyz) * self.data_func['vol'](self, mbl, mbh)
    
    # Mass-weighted boolean masks
    self.data_func['Tcoldbool_mass'] = lambda self, mbl=None, mbh=None, T_cold=None: \
        self.data_func['Tcoldbool'](self, mbl, mbh, T_cold) * self.data_func['mass'](self, mbl, mbh)
    
    self.data_func['Thotbool_mass'] = lambda self, mbl=None, mbh=None, T_hot=None: \
        self.data_func['Thotbool'](self, mbl, mbh, T_hot) * self.data_func['mass'](self, mbl, mbh)
    
    self.data_func['Trangebool_mass'] = lambda self, mbl=None, mbh=None, T_min=None, T_max=None: \
        self.data_func['Trangebool'](self, mbl, mbh, T_min, T_max) * self.data_func['mass'](self, mbl, mbh)
    
    self.data_func['xyzbool_mass'] = lambda self, mbl=None, mbh=None, xyz=None: \
        self.data_func['xyzbool'](self, mbl, mbh, xyz) * self.data_func['mass'](self, mbl, mbh)
    
    self.meshbdata_func = self.data_func

    if not hasattr(self, 'data_dict'):
        self.data_dict = {}
    self.data_list = []
    
    # Add method for users to register custom derived variables
    def add_data_func(self, name, func):
        """Add a custom data function."""
        self.data_func[name] = func
        self.meshbdata_func[name] = func
        return
    
    # Attach the method to the class instance
    self.add_data_func = add_data_func.__get__(self)

def get_data(self, var, mbl=None, mbh=None, **kwargs):
    """
    Unified data access function for AthenaData objects.
    
    This function retrieves data for any variable, handling raw data, coordinates,
    and derived quantities consistently across both full domain and meshblock access patterns.
    
    Parameters
    ----------
    self : AthenaData
        The AthenaData instance to fetch data from
    var : str or array-like
        The variable to retrieve or a data array to pass through
    mbl : int, optional
        Starting meshblock index for partial domain access
    mbh : int, optional
        Ending meshblock index for partial domain access
    **kwargs : dict
        Additional arguments for derived variables that require parameters
        
    Returns
    -------
    ndarray
        The requested data array
    """
    # If var is not a string, return it directly
    if not isinstance(var, str):
        return var
        
    # Choose domain vs meshblock based on mbl/mbh parameters
    is_meshblock_access = (mbl is not None or mbh is not None)
    
    # Handle MPI distribution: convert global indices to local
    if is_meshblock_access and not self.has_full_data:
        local_range = self.global_to_local_mb(mbl, mbh)
        if local_range is None:
            # This rank doesn't own any of the requested meshblocks
            # Return empty array with correct shape
            shape = (0,) + (self.nx3, self.nx2, self.nx1)
            return xp.zeros(shape)
        local_mbl, local_mbh = local_range
    else:
        local_mbl, local_mbh = mbl, mbh
    
    global_mbl = 0 if mbl is None else mbl
    global_mbh = self.n_mbs if mbh is None else mbh

    # First check raw data (but skip eint if it doesn't exist)
    # Note: Coordinates are now computed on-demand via data_func, not stored in coord
    # Support caching of full-domain raw arrays as well as per-rank local-range caches
    local_cache_key = f"{var}_local"
    if (is_meshblock_access and (local_cache_key in self.data_raw)
            and (global_mbl == self.local_mb_start and global_mbh == self.local_mb_end)):
        # Use cached per-rank local-range array
        return xp.asarray(self.data_raw[local_cache_key])

    if var in self.data_raw.keys():
        if is_meshblock_access:
            # Meshblock-specific data - use local indices
            arr = self.data_raw[var][local_mbl:local_mbh]
        else:
            # Full domain data
            arr = self.data_raw[var]

        return xp.asarray(arr)
    
    # Check derived data functions (including on-demand computed coordinates)
    elif var in self.data_func.keys():
        if is_meshblock_access:
            # Compute meshblock-specific derived data
            result = xp.asarray(self.data_func[var](self, mbl, mbh, **kwargs))

            # If the requested meshblock range equals the locally-stored range
            # we can cache the result under a per-rank key to avoid recomputing
            try:
                if (getattr(self, 'cache_data_funcs', False)
                        and (global_mbl == self.local_mb_start)
                        and (global_mbh == self.local_mb_end)):
                    self.data_raw[f"{var}_local"] = xp.asarray(result)
            except Exception:
                pass

            return result
        else:
            # Full-domain derived data: compute and optionally cache the result
            try:
                result = xp.asarray(self.data_func[var](self, **kwargs))
            except Exception as e:
                # Try once more before failing to be resilient to transient errors
                try:
                    result = xp.asarray(self.data_func[var](self, **kwargs))
                except Exception as e2:
                    raise

            # Cache full-domain derived variables in data_raw when enabled
            if getattr(self, 'cache_data_funcs', False):
                try:
                    self.data_raw[var] = xp.asarray(result)
                except Exception:
                    # If caching fails, ignore and continue returning the result
                    pass

            return result
    
    # Special derived variables
    elif var in derived_var_list:
        # Handle correlation functions
        if var in ['eflxwave', 'eflxwave_hot', 'eflxwave_range',
                   'delrhodelT', 'delrhodelT_hot', 'delrhodelT_range',
                   'delvrdelT', 'delvrdelT_hot', 'delvrdelT_range']:
            
            if var in ['eflxwave', 'eflxwave_hot', 'eflxwave_range']:
                data1_name = 'pres'
                data2_name = 'velr'
            elif var in ['delrhodelT', 'delrhodelT_hot', 'delrhodelT_range']:
                data1_name = 'dens'
                data2_name = 'temp'
            else:
                data1_name = 'velr'
                data2_name = 'temp'
            
            # Determine weight and suffix based on variable name
            if var in ['delrhodelT_hot', 'eflxwave_hot', 'delvrdelT_hot']:
                weight_data = get_data(self, 'Thotbool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
                varsuf = '_vw_hot'
            elif var in ['delrhodelT_range', 'eflxwave_range', 'delvrdelT_range']:
                weight_data = get_data(self, 'Trangebool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
                varsuf = '_vw_range'
            else:
                weight_data = get_data(self, 'vol', mbl, mbh)
                varsuf = ''
            
            # Calculate vertical profiles with set_vertical_mb
            bins = self.Nx3

            from ..operations.profiles import set_vertical_mb
            set_vertical_mb(self, varl=[data1_name, data2_name], bins=bins,
                           weights=weight_data, varsuf=varsuf, redo=False,
                           mbl=mbl, mbh=mbh)


            # Extract the profiles and calculate the correlation
            z_prof = self.vert[data1_name + varsuf]['z']
            data1_z_prof = xp.asarray(self.vert[data1_name + varsuf]['profile'])
            data2_z_prof = xp.asarray(self.vert[data2_name + varsuf]['profile'])
            var_z_loc = xp.digitize(get_data(self, 'z', mbl, mbh), (z_prof[1:] + z_prof[:-1]) * 0.5)
            delta_data1 = get_data(self, data1_name, mbl, mbh) - data1_z_prof[var_z_loc]
            delta_data2 = get_data(self, data2_name, mbl, mbh) - data2_z_prof[var_z_loc]
            
            return delta_data1 * delta_data2
            
        # Boolean function for spatial region filtering
        elif var == 'xyzbool':
            return xyz_bool(get_data(self, 'x', mbl, mbh), 
                           get_data(self, 'y', mbl, mbh),
                           get_data(self, 'z', mbl, mbh), 
                           kwargs.get('xyz', None))
        elif var == 'xyzbool_vol':
            return get_data(self, 'xyzbool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
        elif var == 'xyzbool_mass':
            return get_data(self, 'xyzbool', mbl, mbh, **kwargs) * get_data(self, 'mass', mbl, mbh)
            
        # Temperature-based boolean filters
        elif var == 'tcool':
            if hasattr(self, 'rho_eint_t_cool'):
                return self.rho_eint_t_cool(get_data(self, 'dens', mbl, mbh), get_data(self, 'eint', mbl, mbh))
            else:
                # Use data_func implementation if method not available
                return self.data_func['tcool'](self, mbl, mbh, **kwargs)
        elif var == 'Tcoldbool':
            if hasattr(self, 'rho_eint_Tcoldbool'):
                if 'temp_max' in kwargs:
                    return self.rho_eint_Tcoldbool(get_data(self, 'dens', mbl, mbh),
                                               get_data(self, 'eint', mbl, mbh),
                                               T_cold=kwargs['temp_max'])
                else:
                    return self.rho_eint_Tcoldbool(get_data(self, 'dens', mbl, mbh),
                                               get_data(self, 'eint', mbl, mbh))
            else:
                # Use data_func implementation
                return self.data_func['Tcoldbool'](self, mbl, mbh, **kwargs)
        elif var == 'Tcoldbool_vol':
            return get_data(self, 'Tcoldbool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
        elif var == 'Tcoldbool_mass':
            return get_data(self, 'Tcoldbool', mbl, mbh, **kwargs) * get_data(self, 'mass', mbl, mbh)
        elif var == 'Thotbool':
            if hasattr(self, 'rho_eint_Thotbool'):
                if 'temp_min' in kwargs:
                    return self.rho_eint_Thotbool(get_data(self, 'dens', mbl, mbh),
                                              get_data(self, 'eint', mbl, mbh),
                                              T_hot=kwargs['temp_min'])
                else:
                    return self.rho_eint_Thotbool(get_data(self, 'dens', mbl, mbh),
                                              get_data(self, 'eint', mbl, mbh))
            else:
                # Use data_func implementation
                return self.data_func['Thotbool'](self, mbl, mbh, **kwargs)
        elif var == 'Thotbool_vol':
            return get_data(self, 'Thotbool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
        elif var == 'Thotbool_mass':
            return get_data(self, 'Thotbool', mbl, mbh, **kwargs) * get_data(self, 'mass', mbl, mbh)
        elif var == 'Trangebool':
            if hasattr(self, 'rho_eint_Trangebool'):
                if 'temp_min' in kwargs and 'temp_max' in kwargs:
                    return self.rho_eint_Trangebool(get_data(self, 'dens', mbl, mbh),
                                                get_data(self, 'eint', mbl, mbh),
                                                temp_min=kwargs['temp_min'],
                                                temp_max=kwargs['temp_max'])
                else:
                    # Fall back to global values if defined
                    if hasattr(self, 'T_min') and hasattr(self, 'T_max'):
                        return self.rho_eint_Trangebool(get_data(self, 'dens', mbl, mbh),
                                                    get_data(self, 'eint', mbl, mbh),
                                                    temp_min=self.T_min,
                                                    temp_max=self.T_max)
            else:
                # Use data_func implementation
                return self.data_func['Trangebool'](self, mbl, mbh, **kwargs)
        elif var == 'Trangebool_vol':
            return get_data(self, 'Trangebool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
        elif var == 'Trangebool_mass':
            return get_data(self, 'Trangebool', mbl, mbh, **kwargs) * get_data(self, 'mass', mbl, mbh)
            
        # Normalized density and pressure profiles
        elif var in ['nrho', 'nrho_hot', 'nrho_range', 'nprs', 'nprs_hot', 'nprs_range']:
            if var in ['nrho', 'nrho_hot', 'nrho_range']:
                mb_var = 'dens'
            elif var in ['nprs', 'nprs_hot', 'nprs_range']:
                mb_var = 'pres'
                
            if var in ['nrho_hot', 'nprs_hot']:
                weight_data = get_data(self, 'Thotbool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
                varsuf = '_vw_hot'
            elif var in ['nrho_range', 'nprs_range']:
                weight_data = get_data(self, 'Trangebool', mbl, mbh, **kwargs) * get_data(self, 'vol', mbl, mbh)
                varsuf = '_vw_range'
            else:
                weight_data = get_data(self, 'vol', mbl, mbh)
                varsuf = ''
                
            bins = self.Nx3
            vert=self.vertical_profile_func(self, varl=[mb_var], bins=bins,
                                   weights=weight_data, varsuf=varsuf, redo=False)
            z_prof = xp.asarray(vert[mb_var+varsuf]['z'])
            var_z_prof = xp.asarray(vert[mb_var+varsuf]['profile'])
            var_z_loc = xp.digitize(get_data(self, 'z', mbl, mbh), (z_prof[1:]+z_prof[:-1])*0.5)
            return get_data(self, mb_var, mbl, mbh)/var_z_prof[var_z_loc]
        elif var in ['nvelx', 'nvely', 'nvelz']:
            mb_var = var[1:]
            weight_data = get_data(self, 'vol', mbl, mbh)
            varsuf = ''
            bins = self.Nx3
            vert=self.vertical_profile_func(self, varl=[mb_var], bins=bins,
                                   weights=weight_data, varsuf=varsuf, redo=False)
            z_prof = xp.asarray(vert[mb_var+varsuf]['z'])
            var_z_prof = xp.asarray(vert[mb_var+varsuf]['profile'])
            var_z_loc = xp.digitize(get_data(self, 'z', mbl, mbh), (z_prof[1:]+z_prof[:-1])*0.5)
            return get_data(self, mb_var, mbl, mbh)-var_z_prof[var_z_loc]
        
    # If we reach here, the requested variable wasn't found
    raise ValueError(f"No variable called '{var}' found")
