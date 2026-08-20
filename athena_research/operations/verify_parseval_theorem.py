#!/usr/bin/env python
"""
Verification script for Parseval's theorem and Helmholtz decomposition consistency.

This script verifies that:
1. Parseval's theorem holds: sum of |data|^2 in real space = sum of power spectrum in Fourier space
2. Standard and memory-efficient spectral methods give consistent results
3. Helmholtz decomposition is consistent between direct calculation and spectral decomposition
4. All spectral analysis tools are working correctly

Usage:
python verify_parseval_theorem.py --filedir /path/to/data/ --fstart 0 --fend 10 [--mhd]
python verify_parseval_theorem.py --filedir ./simulation_data/ --fstart 25 --fend 51 --verbose
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Athena_research imports -- always import the array backend from core.base
# rather than hand-rolling a second copy of the CuPy/NumPy selection logic.
from athena_research import AthenaData
from athena_research.core.base import xp, asnumpy, cupy_enabled
from athena_research.operations.spectra import (
    set_spectrum, set_spectrum_helmholtz,
    get_spectrum, get_spectrum_mb,
    get_spectrum_helmholtz, get_spectrum_helmholtz_mb
)
from athena_research.operations.histograms import set_dist
from athena_research.operations.basic_operations import calc_sum, calc_avg

if cupy_enabled:
    print("CUDA available. Using GPU for calculations.")
    print(f"GPU: {xp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Memory: {xp.cuda.runtime.memGetInfo()[1]/1e9:.2f} GB total")

    # Select the GPU with the most free memory (core.base defaults to device 0)
    def select_gpu_with_max_memory():
        num_gpus = xp.cuda.runtime.getDeviceCount()
        max_free_memory = 0
        selected_gpu = 0

        for gpu_id in range(num_gpus):
            # Set the device first
            xp.cuda.Device(gpu_id).use()
            free_memory, total_memory = xp.cuda.runtime.memGetInfo()
            print(f"GPU {gpu_id}: {free_memory/1e9:.2f} GB free / {total_memory/1e9:.2f} GB total")

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                selected_gpu = gpu_id

        print(f"Selected GPU {selected_gpu} with {max_free_memory/1e9:.2f} GB free memory")
        return selected_gpu

    gpu_id = select_gpu_with_max_memory()
    xp.cuda.Device(gpu_id).use()
else:
    print("CUDA not available. Using CPU for calculations.")

class ParsevalVerification:
    """Class to handle all Parseval's theorem verification tests."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {}
        self.tolerance = 1e-10  # Relative tolerance for comparisons
        
    def clear_gpu_memory(self):
        """Clear GPU memory if CUDA is available."""
        if cupy_enabled:
            try:
                xp.cuda.Stream.null.synchronize()
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not clear GPU memory: {e}")

    def monitor_gpu_memory(self, prefix=""):
        """Monitor GPU memory usage."""
        if cupy_enabled:
            try:
                free_memory, total_memory = xp.cuda.runtime.memGetInfo()
                usage_percent = ((total_memory - free_memory) / total_memory) * 100
                if self.verbose:
                    print(f"{prefix}GPU memory: {usage_percent:.1f}% used "
                          f"({(total_memory - free_memory)/1e9:.2f} GB / {total_memory/1e9:.2f} GB)")
                return usage_percent > 90
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not check GPU memory: {e}")
        return False

    def verify_parseval_scalar(self, ad, var, method='standard'):
        """
        Verify Parseval's theorem for a scalar field.
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var : str
            Variable name
        method : str
            'standard' or 'memory_efficient'
            
        Returns
        -------
        dict
            Results of verification
        """
        print(f"  Testing Parseval's theorem for {var} using {method} method...")
        
        # Calculate mean using calc_avg function
        calc_avg(ad, varl=[var], weights='vol', redo=True)
        mean_val = ad.avg[var]
        
        # Get refined data to calculate variance directly
        data = ad.get_refined_data(var)
        vol_weights = ad.get_refined_data('vol')
        
        # Ensure arrays are the correct type for operations
        data = xp.asarray(data)
        vol_weights = xp.asarray(vol_weights)
        mean_val = xp.asarray(mean_val)
        
        # Subtract mean from data before calculating variance
        data_centered = data - mean_val
        
        # Calculate variance in real space
        variance_real = xp.sum(vol_weights * data_centered**2) / xp.sum(vol_weights)
        
        # Calculate power spectrum
        if method == 'standard':
            k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum(
                ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1
            )
        else:  # memory_efficient
            k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb(
                ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1, ndiv=4
            )
        
        # Ensure E_spectrum is the correct array type
        E_spectrum = xp.asarray(E_spectrum)
        
        # Normalize spectrum by total number of grid points
        E_spectrum = E_spectrum / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        # E_spectrum already holds the total power in each bin (not a density),
        # so Parseval's theorem is sum(E_spectrum) == <f^2>, with no dk weighting.
        # Bin 0 is the DC/k=0 mode (mean^2); exclude it since variance_real above
        # already has the mean subtracted.
        spectrum_sum = xp.sum(E_spectrum[1:])
        
        # Calculate relative error
        rel_error = abs(asnumpy(spectrum_sum - variance_real) / asnumpy(variance_real))
        
        result = {
            'variable': var,
            'method': method,
            'variance_real': asnumpy(variance_real),
            'spectrum_sum': asnumpy(spectrum_sum),
            'relative_error': rel_error,
            'passed': rel_error < self.tolerance
        }
        
        if self.verbose:
            print(f"    Real space variance: {result['variance_real']:.6e}")
            print(f"    Spectrum sum: {result['spectrum_sum']:.6e}")
            print(f"    Relative error: {result['relative_error']:.6e}")
            print(f"    Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result

    def verify_parseval_vector(self, ad, var_base, method='standard'):
        """
        Verify Parseval's theorem for a vector field (e.g., velocity).
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var_base : str
            Base variable name (e.g., 'vel' for velx, vely, velz)
        method : str
            'standard' or 'memory_efficient'
            
        Returns
        -------
        dict
            Results of verification
        """
        print(f"  Testing Parseval's theorem for {var_base} vector using {method} method...")
        
        # Calculate means for all velocity components using calc_avg
        vel_components = [var_base + 'x', var_base + 'y', var_base + 'z']
        calc_avg(ad, varl=vel_components, weights='vol', redo=True)
        
        # Get velocity data and means
        velx = ad.get_refined_data(var_base + 'x')
        vely = ad.get_refined_data(var_base + 'y')
        velz = ad.get_refined_data(var_base + 'z')
        vol_weights = ad.get_refined_data('vol')
        
        # Get the means
        mean_velx = ad.avg[var_base + 'x']
        mean_vely = ad.avg[var_base + 'y']
        mean_velz = ad.avg[var_base + 'z']
        
        # Ensure arrays are the correct type for operations
        velx = xp.asarray(velx)
        vely = xp.asarray(vely)
        velz = xp.asarray(velz)
        vol_weights = xp.asarray(vol_weights)
        mean_velx = xp.asarray(mean_velx)
        mean_vely = xp.asarray(mean_vely)
        mean_velz = xp.asarray(mean_velz)
        
        # Subtract means from velocity components
        velx_centered = velx - mean_velx
        vely_centered = vely - mean_vely
        velz_centered = velz - mean_velz
        
        # Total kinetic energy density (per unit volume) using centered data
        ke_density = 0.5 * (velx_centered**2 + vely_centered**2 + velz_centered**2)
        total_ke_real = xp.sum(vol_weights * ke_density) / xp.sum(vol_weights)
        
        # Calculate power spectra for each component
        ke_spectrum_total = 0
        for comp in ['x', 'y', 'z']:
            var = var_base + comp
            if method == 'standard':
                k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum(
                    ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1
                )
            else:  # memory_efficient
                k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb(
                    ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1, ndiv=4
                )
            
            # Ensure E_spectrum is the correct array type
            E_spectrum = xp.asarray(E_spectrum)
            
            # Normalize spectrum by total number of grid points
            E_spectrum = E_spectrum / (ad.Nx1 * ad.Nx2 * ad.Nx3)
            
            # E_spectrum is already total power per bin; sum bins directly
            # (excluding the DC/k=0 bin, since the real-space side is mean-subtracted).
            ke_spectrum_total += xp.sum(E_spectrum[1:])
        
        # The total should be 2 * total_ke_real because E_spectrum gives variance, not energy
        # and kinetic energy = 0.5 * (velx^2 + vely^2 + velz^2)
        ke_spectrum_total *= 0.5
        
        # Calculate relative error
        rel_error = abs(asnumpy(ke_spectrum_total - total_ke_real) / asnumpy(total_ke_real))
        
        result = {
            'variable': var_base,
            'method': method,
            'ke_real': asnumpy(total_ke_real),
            'ke_spectrum': asnumpy(ke_spectrum_total),
            'relative_error': rel_error,
            'passed': rel_error < self.tolerance
        }
        
        if self.verbose:
            print(f"    Real space KE: {result['ke_real']:.6e}")
            print(f"    Spectrum KE: {result['ke_spectrum']:.6e}")
            print(f"    Relative error: {result['relative_error']:.6e}")
            print(f"    Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result

    def compare_spectral_methods(self, ad, var):
        """
        Compare standard and memory-efficient spectral methods.
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var : str
            Variable name
            
        Returns
        -------
        dict
            Comparison results
        """
        print(f"  Comparing spectral methods for {var}...")
        
        # Standard method
        k1_, kbins1, nk1, E_spectrum1, E_spectral_density1, E_spectrum_norm1 = get_spectrum(
            ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1
        )
        
        # Ensure E_spectrum1 is the correct array type
        E_spectrum1 = xp.asarray(E_spectrum1)
        
        # Normalize spectrum by total number of grid points
        E_spectrum1 = E_spectrum1 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        self.clear_gpu_memory()
        
        # Memory-efficient method
        k2_, kbins2, nk2, E_spectrum2, E_spectral_density2, E_spectrum_norm2 = get_spectrum_mb(
            ad, var, strat_flag=False, skip=0.0, nbins=ad.Nx1, ndiv=4
        )
        
        # Ensure E_spectrum2 is the correct array type
        E_spectrum2 = xp.asarray(E_spectrum2)
        
        # Normalize spectrum by total number of grid points
        E_spectrum2 = E_spectrum2 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        # Compare results
        k_diff = xp.max(xp.abs(k1_ - k2_))
        spectrum_diff = xp.max(xp.abs(E_spectrum1 - E_spectrum2))
        spectrum_rel_diff = spectrum_diff / xp.max(E_spectrum1)
        
        result = {
            'variable': var,
            'k_max_diff': asnumpy(k_diff),
            'spectrum_max_diff': asnumpy(spectrum_diff),
            'spectrum_rel_diff': asnumpy(spectrum_rel_diff),
            'passed': asnumpy(spectrum_rel_diff) < self.tolerance
        }
        
        if self.verbose:
            print(f"    Max k difference: {result['k_max_diff']:.6e}")
            print(f"    Max spectrum difference: {result['spectrum_max_diff']:.6e}")
            print(f"    Relative spectrum difference: {result['spectrum_rel_diff']:.6e}")
            print(f"    Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result

    def verify_helmholtz_consistency(self, ad, var_base='vel', method='standard'):
        """
        Verify that Helmholtz decomposition is consistent between methods.
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var_base : str
            Base variable name (default 'vel')
        method : str
            'standard' or 'memory_efficient'
            
        Returns
        -------
        dict
            Verification results
        """
        print(f"  Testing Helmholtz decomposition consistency for {var_base} using {method} method...")
        
        # Calculate means for all velocity components using calc_avg
        vel_components = [var_base + 'x', var_base + 'y', var_base + 'z']
        calc_avg(ad, varl=vel_components, weights='vol', redo=True)
        
        # Method 1: Calculate real-space velocity variance (sigma_v)
        velx = ad.get_refined_data(var_base + 'x')
        vely = ad.get_refined_data(var_base + 'y')
        velz = ad.get_refined_data(var_base + 'z')
        vol_weights = ad.get_refined_data('vol')
        
        # Get the means
        mean_velx = ad.avg[var_base + 'x']
        mean_vely = ad.avg[var_base + 'y']
        mean_velz = ad.avg[var_base + 'z']
        
        # Ensure arrays are the correct type for operations
        velx = xp.asarray(velx)
        vely = xp.asarray(vely)
        velz = xp.asarray(velz)
        vol_weights = xp.asarray(vol_weights)
        mean_velx = xp.asarray(mean_velx)
        mean_vely = xp.asarray(mean_vely)
        mean_velz = xp.asarray(mean_velz)
        
        # Subtract means from velocity components
        velx_centered = velx - mean_velx
        vely_centered = vely - mean_vely
        velz_centered = velz - mean_velz
        
        # Calculate velocity dispersion (sigma_v) - weighted variance of total velocity using centered data
        vel_mag_squared = velx_centered**2 + vely_centered**2 + velz_centered**2
        sigma_v_squared = xp.sum(vol_weights * vel_mag_squared) / xp.sum(vol_weights)
        
        # Also calculate individual component variances for comparison using centered data
        var_velx = xp.sum(vol_weights * velx_centered**2) / xp.sum(vol_weights) 
        var_vely = xp.sum(vol_weights * vely_centered**2) / xp.sum(vol_weights)
        var_velz = xp.sum(vol_weights * velz_centered**2) / xp.sum(vol_weights)
        total_component_variance = var_velx + var_vely + var_velz
        
        # Method 2: Helmholtz decomposition via spectra
        if method == 'standard':
            k_, kbins, nk, E_spectrum_comp, E_spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz(
                ad, var=var_base, strat_flag=False, skip=0.0, nbins=200
            )
        else:  # memory_efficient
            k_, kbins, nk, E_spectrum_comp, E_spectrum_sol, E_spectral_density_comp, E_spectral_density_sol = get_spectrum_helmholtz_mb(
                ad, var=var_base, strat_flag=False, skip=0.0, nbins=200, ndiv=4
            )
        
        # Ensure spectra are the correct array type
        E_spectrum_comp = xp.asarray(E_spectrum_comp)
        E_spectrum_sol = xp.asarray(E_spectrum_sol)
        
        # Normalize spectra by total number of grid points
        E_spectrum_comp = E_spectrum_comp / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        E_spectrum_sol = E_spectrum_sol / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        # E_spectrum_comp/sol are already total power per bin; sum bins directly
        # (excluding the DC/k=0 bin, which has no meaningful comp/sol split).
        total_comp = xp.sum(E_spectrum_comp[1:])
        total_sol = xp.sum(E_spectrum_sol[1:])
        total_helmholtz = total_comp + total_sol
        
        # Compare with individual component spectra from get_spectrum
        total_components = 0
        for comp in ['x', 'y', 'z']:
            var = var_base + comp
            if method == 'standard':
                k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum(
                    ad, var, strat_flag=False, skip=0.0, nbins=200
                )
            else:
                k_, kbins, nk, E_spectrum, E_spectral_density, E_spectrum_norm = get_spectrum_mb(
                    ad, var, strat_flag=False, skip=0.0, nbins=200, ndiv=4
                )
            
            # Ensure E_spectrum is the correct array type
            E_spectrum = xp.asarray(E_spectrum)
            
            # Normalize spectrum by total number of grid points
            E_spectrum = E_spectrum / (ad.Nx1 * ad.Nx2 * ad.Nx3)
            
            # E_spectrum is already total power per bin; sum bins directly
            # (excluding the DC/k=0 bin, since the real-space side is mean-subtracted).
            total_components += xp.sum(E_spectrum[1:])
        
        # Calculate relative errors for different consistency checks
        # 1. Helmholtz decomposition: sol + comp should equal sum of components
        rel_error_helmholtz = abs(asnumpy(total_helmholtz - total_components) / asnumpy(total_components))
        
        # 2. Parseval check: sigma_v should equal Helmholtz total
        rel_error_sigma_v = abs(asnumpy(sigma_v_squared - total_helmholtz) / asnumpy(sigma_v_squared))
        
        # 3. Component consistency: sigma_v should equal sum of component variances  
        rel_error_components = abs(asnumpy(sigma_v_squared - total_component_variance) / asnumpy(sigma_v_squared))
        
        result = {
            'variable': var_base,
            'method': method,
            'sigma_v_squared': asnumpy(sigma_v_squared),
            'total_component_variance': asnumpy(total_component_variance),
            'total_components_spectral': asnumpy(total_components),
            'total_compressive': asnumpy(total_comp),
            'total_solenoidal': asnumpy(total_sol),
            'total_helmholtz': asnumpy(total_helmholtz),
            'rel_error_helmholtz': rel_error_helmholtz,
            'rel_error_sigma_v': rel_error_sigma_v,
            'rel_error_components': rel_error_components,
            'comp_fraction': asnumpy(total_comp / total_helmholtz),
            'sol_fraction': asnumpy(total_sol / total_helmholtz),
            'passed': (rel_error_helmholtz < self.tolerance and 
                      rel_error_sigma_v < self.tolerance and
                      rel_error_components < self.tolerance)
        }
        
        if self.verbose:
            print(f"    σ_v² (real space): {result['sigma_v_squared']:.6e}")
            print(f"    Sum component variances: {result['total_component_variance']:.6e}")
            print(f"    Sum component spectra: {result['total_components_spectral']:.6e}")
            print(f"    Total Helmholtz (comp + sol): {result['total_helmholtz']:.6e}")
            print(f"    Compressive fraction: {result['comp_fraction']:.3f}")
            print(f"    Solenoidal fraction: {result['sol_fraction']:.3f}")
            print(f"    Helmholtz consistency error: {result['rel_error_helmholtz']:.6e}")
            print(f"    σ_v vs Helmholtz error: {result['rel_error_sigma_v']:.6e}")
            print(f"    σ_v vs components error: {result['rel_error_components']:.6e}")
            print(f"    Test {'PASSED' if result['passed'] else 'FAILED'}")
            
            # Show which specific tests failed
            if not result['passed']:
                if result['rel_error_helmholtz'] >= self.tolerance:
                    print(f"      ❌ Helmholtz decomposition consistency failed")
                if result['rel_error_sigma_v'] >= self.tolerance:
                    print(f"      ❌ σ_v vs Helmholtz total failed") 
                if result['rel_error_components'] >= self.tolerance:
                    print(f"      ❌ σ_v vs component sum failed")
        
        return result

    def compare_helmholtz_methods(self, ad, var_base='vel'):
        """
        Compare standard and memory-efficient Helmholtz decomposition methods.
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var_base : str
            Base variable name
            
        Returns
        -------
        dict
            Comparison results
        """
        print(f"  Comparing Helmholtz methods for {var_base}...")
        
        # Standard method
        k1_, kbins1, nk1, E_spectrum_comp1, E_spectrum_sol1, E_spectral_density_comp1, E_spectral_density_sol1 = get_spectrum_helmholtz(
            ad, var=var_base, strat_flag=False, skip=0.0, nbins=200
        )
        
        # Ensure spectra are the correct array type
        E_spectrum_comp1 = xp.asarray(E_spectrum_comp1)
        E_spectrum_sol1 = xp.asarray(E_spectrum_sol1)
        
        # Normalize spectra by total number of grid points
        E_spectrum_comp1 = E_spectrum_comp1 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        E_spectrum_sol1 = E_spectrum_sol1 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        self.clear_gpu_memory()
        
        # Memory-efficient method
        k2_, kbins2, nk2, E_spectrum_comp2, E_spectrum_sol2, E_spectral_density_comp2, E_spectral_density_sol2 = get_spectrum_helmholtz_mb(
            ad, var=var_base, strat_flag=False, skip=0.0, nbins=200, ndiv=4
        )
        
        # Ensure spectra are the correct array type
        E_spectrum_comp2 = xp.asarray(E_spectrum_comp2)
        E_spectrum_sol2 = xp.asarray(E_spectrum_sol2)
        
        # Normalize spectra by total number of grid points
        E_spectrum_comp2 = E_spectrum_comp2 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        E_spectrum_sol2 = E_spectrum_sol2 / (ad.Nx1 * ad.Nx2 * ad.Nx3)
        
        # Compare results
        comp_diff = xp.max(xp.abs(E_spectrum_comp1 - E_spectrum_comp2))
        sol_diff = xp.max(xp.abs(E_spectrum_sol1 - E_spectrum_sol2))
        comp_rel_diff = comp_diff / xp.max(E_spectrum_comp1)
        sol_rel_diff = sol_diff / xp.max(E_spectrum_sol1)
        
        result = {
            'variable': var_base,
            'comp_max_diff': asnumpy(comp_diff),
            'sol_max_diff': asnumpy(sol_diff),
            'comp_rel_diff': asnumpy(comp_rel_diff),
            'sol_rel_diff': asnumpy(sol_rel_diff),
            'passed': (asnumpy(comp_rel_diff) < self.tolerance and 
                      asnumpy(sol_rel_diff) < self.tolerance)
        }
        
        if self.verbose:
            print(f"    Max compressive difference: {result['comp_max_diff']:.6e}")
            print(f"    Max solenoidal difference: {result['sol_max_diff']:.6e}")
            print(f"    Relative comp. difference: {result['comp_rel_diff']:.6e}")
            print(f"    Relative sol. difference: {result['sol_rel_diff']:.6e}")
            print(f"    Test {'PASSED' if result['passed'] else 'FAILED'}")
        
        return result

    def run_full_verification(self, ad, var_list):
        """
        Run complete verification suite.
        
        Parameters
        ----------
        ad : AthenaData
            Athena data object
        var_list : list
            List of variables to test
            
        Returns
        -------
        dict
            Complete results
        """
        print(f"\n{'='*60}")
        print("RUNNING COMPLETE PARSEVAL VERIFICATION SUITE")
        print(f"{'='*60}")
        
        results = {
            'parseval_scalar': [],
            'parseval_vector': [],
            'method_comparison': [],
            'helmholtz_consistency': [],
            'helmholtz_comparison': [],
            'summary': {}
        }
        
        # Test 1: Parseval's theorem for scalar fields
        print(f"\n{'-'*40}")
        print("TEST 1: PARSEVAL'S THEOREM - SCALAR FIELDS")
        print(f"{'-'*40}")
        
        scalar_vars = [v for v in var_list if v not in ['velx', 'vely', 'velz']]
        for var in scalar_vars:
            try:
                # Test both methods
                for method in ['standard', 'memory_efficient']:
                    result = self.verify_parseval_scalar(ad, var, method)
                    results['parseval_scalar'].append(result)
                    self.clear_gpu_memory()
            except Exception as e:
                print(f"    ERROR testing {var}: {e}")
        
        # Test 2: Parseval's theorem for vector fields
        print(f"\n{'-'*40}")
        print("TEST 2: PARSEVAL'S THEOREM - VECTOR FIELDS")
        print(f"{'-'*40}")
        
        if all(v in var_list for v in ['velx', 'vely', 'velz']):
            try:
                for method in ['standard', 'memory_efficient']:
                    result = self.verify_parseval_vector(ad, 'vel', method)
                    results['parseval_vector'].append(result)
                    self.clear_gpu_memory()
            except Exception as e:
                print(f"    ERROR testing velocity: {e}")
        
        # Test 3: Method comparison for spectral calculations
        print(f"\n{'-'*40}")
        print("TEST 3: SPECTRAL METHOD COMPARISON")
        print(f"{'-'*40}")
        
        for var in var_list:
            try:
                result = self.compare_spectral_methods(ad, var)
                results['method_comparison'].append(result)
                self.clear_gpu_memory()
            except Exception as e:
                print(f"    ERROR comparing methods for {var}: {e}")
        
        # Test 4: Helmholtz decomposition consistency
        print(f"\n{'-'*40}")
        print("TEST 4: HELMHOLTZ DECOMPOSITION CONSISTENCY")
        print(f"{'-'*40}")
        
        if all(v in var_list for v in ['velx', 'vely', 'velz']):
            try:
                for method in ['standard', 'memory_efficient']:
                    result = self.verify_helmholtz_consistency(ad, 'vel', method)
                    results['helmholtz_consistency'].append(result)
                    self.clear_gpu_memory()
            except Exception as e:
                print(f"    ERROR testing Helmholtz consistency: {e}")
        
        # Test 5: Helmholtz method comparison
        print(f"\n{'-'*40}")
        print("TEST 5: HELMHOLTZ METHOD COMPARISON")
        print(f"{'-'*40}")
        
        if all(v in var_list for v in ['velx', 'vely', 'velz']):
            try:
                result = self.compare_helmholtz_methods(ad, 'vel')
                results['helmholtz_comparison'].append(result)
                self.clear_gpu_memory()
            except Exception as e:
                print(f"    ERROR comparing Helmholtz methods: {e}")
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def generate_summary(self, results):
        """Generate summary of all tests."""
        print(f"\n{'='*60}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*60}")
        
        categories = [
            ('Parseval Scalar', results['parseval_scalar']),
            ('Parseval Vector', results['parseval_vector']),
            ('Method Comparison', results['method_comparison']),
            ('Helmholtz Consistency', results['helmholtz_consistency']),
            ('Helmholtz Comparison', results['helmholtz_comparison'])
        ]
        
        total_tests = 0
        total_passed = 0
        
        for cat_name, cat_results in categories:
            if cat_results:
                passed = sum(1 for r in cat_results if r['passed'])
                total = len(cat_results)
                total_tests += total
                total_passed += passed
                print(f"{cat_name:25}: {passed:2d}/{total:2d} passed ({100*passed/total:5.1f}%)")
        
        print(f"{'-'*40}")
        print(f"{'OVERALL':25}: {total_passed:2d}/{total_tests:2d} passed ({100*total_passed/total_tests:5.1f}%)")
        
        if total_passed == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Your spectral analysis tools are working correctly.")
        else:
            print(f"\n⚠️  SOME TESTS FAILED. Please investigate the issues above.")
        
        # Store summary
        results['summary'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0
        }

def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(
        description='Verify Parseval\'s theorem and spectral analysis consistency',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Custom directory mode:
  python verify_parseval_theorem.py --filedir /path/to/data/ --fstart 0 --fend 1 --mhd
  python verify_parseval_theorem.py --filedir sol_512_T1e6_n0_003_L1_dens_perturb/bin/ --fstart 10 --fend 11
  
  # Custom base prefix:
  python verify_parseval_theorem.py --filedir data/ --fstart 0 --fend 1 --prefix "Sim"
  python verify_parseval_theorem.py --filedir data/ --fstart 0 --fend 1 --prefix "Custom" --mhd
        """
    )
    
    # Directory and file options
    parser.add_argument('--filedir', type=str, required=True,
                       help='Directory path containing simulation files')
    parser.add_argument('--fstart', type=int, default=0,
                       help='Starting file number (default: 0)')
    parser.add_argument('--fend', type=int, default=1,
                       help='Ending file number (default: 1)')
    parser.add_argument('--fstep', type=int, default=1,
                       help='File step (default: 1)')
    parser.add_argument('--mhd', action='store_true',
                       help='Use MHD file prefix')
    
    # File prefix options
    parser.add_argument('--prefix', type=str, default='Turb',
                       help='Base file prefix (default: Turb). Full prefix will be constructed as: '
                            'prefix.hydro_w. for hydro or prefix.mhd_w_bcc. for MHD')
    
    # Output options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output with detailed information')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Relative tolerance for verification tests (default: 1e-10)')
    
    args = parser.parse_args()
    
    # Process the directory path
    filedir = args.filedir
    if not filedir.endswith('/'):
        filedir += '/'
    
    # Check if directory exists
    if not os.path.exists(filedir):
        parser.error(f'Directory not found: {filedir}')
    
    print(f"Parseval Verification Plan:")
    print(f"  Directory: {filedir}")
    print(f"  Files: {args.fstart} to {args.fend-1} (step: {args.fstep})")
    print(f"  MHD: {'Yes' if args.mhd else 'No'}")
    print(f"  Base Prefix: {args.prefix}")
    if args.mhd:
        print(f"  Full Prefix: {args.prefix}.mhd_w_bcc.")
    else:
        print(f"  Full Prefix: {args.prefix}.hydro_w.")
    print(f"  Tolerance: {args.tolerance:.0e}")
    print(f"  Verbose: {'Yes' if args.verbose else 'No'}")
    print()
    
    # Initialize verification object
    verifier = ParsevalVerification(verbose=args.verbose)
    verifier.tolerance = args.tolerance
    
    total_start_time = time.time()
    all_results = []
    
    # File suffixes
    hdf_suffix = '.athdf'
    h5_suffix = '.h5data'
    
    print(f"\n{'='*60}")
    print(f"Processing directory: {filedir}")
    print(f"{'='*60}")
    
    for filenum in range(args.fstart, args.fend, args.fstep):
        try:
            # Construct filenames using base prefix
            if args.mhd:
                prefix = f"{args.prefix}.mhd_w_bcc."
            else:
                prefix = f"{args.prefix}.hydro_w."
            
            filenum_str = f"{filenum:05d}"
            hdf_filename = prefix + filenum_str + hdf_suffix
            h5filename = prefix + filenum_str + h5_suffix
            
            print(f"\nProcessing file {filenum}: {hdf_filename}")
            
            # Load data
            hdf_filepath = filedir + hdf_filename
            if os.path.exists(hdf_filepath):
                ad = AthenaData()
                ad.load(hdf_filepath, config=True)
            else:
                print(f"  File not found: {hdf_filepath}")
                continue
            
            # Determine variable list
            if args.mhd:
                var_list = ['velx', 'vely', 'velz', 'dens', 'bcc1', 'bcc2', 'bcc3']
            else:
                var_list = ['velx', 'vely', 'velz', 'dens']
            
            # Run verification
            results = verifier.run_full_verification(ad, var_list)
            results['file_info'] = {
                'directory': filedir,
                'filename': hdf_filename,
                'filenum': filenum,
                'is_mhd': args.mhd
            }
            all_results.append(results)
            
            # Clean up
            verifier.clear_gpu_memory()
            del ad
            
        except Exception as e:
            print(f"Error processing file {filenum}: {e}")
            continue
    
    # Final summary
    total_end_time = time.time()
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processing completed in {total_end_time - total_start_time:.2f} seconds")
    print(f"Files processed: {len(all_results)}")
    
    if all_results:
        # Aggregate results
        total_tests = sum(r['summary']['total_tests'] for r in all_results)
        total_passed = sum(r['summary']['total_passed'] for r in all_results)
        
        print(f"Total tests run: {total_tests}")
        print(f"Total tests passed: {total_passed}")
        print(f"Overall success rate: {100*total_passed/total_tests:.1f}%")
        
        if total_passed == total_tests:
            print(f"\n🎉 ALL TESTS PASSED ACROSS ALL FILES!")
            print("Your spectral analysis tools are working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check the detailed output above.")
            
            # Show which files had failures
            failed_files = []
            for result in all_results:
                if result['summary']['total_passed'] < result['summary']['total_tests']:
                    failed_files.append(result['file_info']['filename'])
            
            if failed_files:
                print("Files with test failures:")
                for filename in failed_files:
                    print(f"  - {filename}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
