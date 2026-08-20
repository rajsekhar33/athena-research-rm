from setuptools import setup, find_packages

setup(
    name="athena_research",
    version="0.2.0",
    description="Tools for analyzing Athenak simulation data with CPU/GPU and MPI support",
    author="Rajsekhar Mohapatra",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.3.0",
        "h5py>=3.0.0",
        "scipy>=1.5.0",  # Used for signal processing (windows, hamming) and ndimage (gaussian_filter)
        "psutil>=5.7.0",  # For CPU/GPU memory monitoring
        "astropy>=5.0.0",  # Required by core.units
        "scikit-image>=0.19.0",  # Required by area_functions CPU fallback
    ],
    extras_require={
        "gpu": [
            # Note: Users should install appropriate version for their CUDA
            # pip install cupy-cuda11x or cupy-cuda12x
            "cupy-cuda11x>=10.0.0",  # Default to CUDA 11.x
        ],
        "mpi": [
            "mpi4py>=3.0.0",  # Requires MPI installation (OpenMPI, MPICH, etc.)
        ],
        "cpu-acceleration": [
            "numba>=0.53.0",  # For CPU JIT compilation
        ],
        "plotting": [
            "cmasher>=1.6.0",
            "joblib>=1.1.0",
            "Pillow>=9.0.0",
            "packaging>=21.0",
        ],
        "volume-rendering": [
            "pyvista>=0.38.0",
        ],
        "full": [
            "cupy-cuda11x>=10.0.0",
            "mpi4py>=3.0.0",
            "numba>=0.53.0",
            "cmasher>=1.6.0",
            "joblib>=1.1.0",
            "Pillow>=9.0.0",
            "packaging>=21.0",
            "pyvista>=0.38.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)