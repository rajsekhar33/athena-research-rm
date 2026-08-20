"""
Unit system handling for Athenak simulations.
"""

import numpy as np
from astropy import constants as const

class Units:
    """
    Class for handling unit conversions in Athena++ simulations.
    
    This class provides a standardized way to convert between code units
    and physical units, with support for common astrophysical quantities.
    """
    
    def __init__(self, lunit=None, munit=None, tunit=None, mu=0.618, x_h=0.715):
        """
        Initialize a Units object with specified base units.
        
        Parameters
        ----------
        lunit : float, optional
            Length unit in cm
        munit : float, optional
            Mass unit in g
        tunit : float, optional
            Time unit in s
        mu : float, optional
            Mean molecular weight
        """
        # Base units in CGS
        self.cm_cgs = 1.0
        self.g_cgs = 1.0
        self.s_cgs = 1.0
        
        # Astronomical constants in CGS
        self.pc_cgs = const.pc.cgs.value
        self.kpc_cgs = const.kpc.cgs.value
        self.msun_cgs = const.M_sun.cgs.value
        self.atomic_mass_unit_cgs = const.u.cgs.value
        
        # Time units
        self.yr_cgs = 3.15576e+7
        self.myr_cgs = 3.15576e+13
        
        # Velocity units
        self.cm_s_cgs = 1.0
        self.km_s_cgs = 1.0e5
        
        # Energy and pressure units
        self.g_cm3_cgs = 1.0
        self.erg_cgs = 1.0
        self.dyne_cm2_cgs = 1.0
        
        # Temperature units
        self.kelvin_cgs = 1.0
        self.k_boltzmann_cgs = const.k_B.cgs.value
        self.kev_to_kelvin = 11604525.0061598
        
        # Other constants
        self.grav_constant_cgs = const.G.cgs.value
        self.speed_of_light_cgs = const.c.cgs.value
        
        # Mean molecular weight
        self.mu = mu
        
        # Hydrogen mass fraction
        self.x_h = x_h
        
        # Gamma (adiabatic index)
        self.gamma = 5.0/3.0
        
        # Set custom units if provided
        self.length_cgs = lunit if lunit is not None else self.kpc_cgs
        self.mass_cgs = munit if munit is not None else (self.mu * self.atomic_mass_unit_cgs * self.length_cgs**3)
        
        # Derive time unit if not provided
        if tunit is None:
            # 100 Myrs
            self.time_cgs = 100.*self.myr_cgs
        else:
            self.time_cgs = tunit
            
        # Derived units
        self.vunit = self.length_cgs / self.time_cgs
        self.dunit = self.mass_cgs / self.length_cgs**3
        self.punit = self.mass_cgs / (self.length_cgs * self.time_cgs**2)
        self.eunit = self.mass_cgs * self.length_cgs**2 / self.time_cgs**2
        
    @property
    def velocity_cgs(self):
        return self.length_cgs/self.time_cgs
    @property
    def density_cgs(self):
        return self.mass_cgs/self.length_cgs**3
    @property
    def energy_cgs(self):
        return self.mass_cgs*self.velocity_cgs**2
    @property
    def pressure_cgs(self):
        return self.energy_cgs/self.length_cgs**3
    @property
    def temperature_cgs(self):
        return self.velocity_cgs**2*self.mu*self.atomic_mass_unit_cgs/self.k_boltzmann_cgs
    @property
    def grav_constant(self):
        return self.grav_constant_cgs*self.density_cgs*self.time_cgs**2
    @property
    def speed_of_light(self):
        return self.speed_of_light_cgs/self.velocity_cgs
    @property
    def number_density_cgs(self):
        return self.density_cgs/self.mu/self.atomic_mass_unit_cgs
    @property
    def hydrogen_number_density_cgs(self):
        return self.density_cgs*self.x_h/self.atomic_mass_unit_cgs
    @property
    def cooling_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs**2
    @property
    def heating_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs
    @property
    def conductivity_cgs(self):
        return self.pressure_cgs*self.velocity_cgs*self.length_cgs/self.temperature_cgs
    @property
    def entropy_kevcm2(self):
        kev_erg=1.60218e-09
        gamma=5./3.
        return self.pressure_cgs/kev_erg/self.number_density_cgs**gamma