"""
Setup script for bates_kernel_cpp CUDA extension module.

This module provides GPU-accelerated Monte Carlo simulation for the
Bates stochastic volatility + jump diffusion model.

Build options:
    pip install .                           # Standard install
    pip install -e .                        # Development install
    python setup.py build_ext --inplace     # Build in place
    python setup.py build_ext --cuda-arch=86,90  # Specify GPU architectures

Supported CUDA architectures:
    - sm_75: Turing (RTX 20 series, T4)
    - sm_80: Ampere (A100)
    - sm_86: Ampere (RTX 30 series, A10, A40)
    - sm_89: Ada Lovelace (RTX 40 series, L40)
    - sm_90: Hopper (H100)
"""

import os
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def find_cuda_home():
    """
    Find the CUDA installation path.

    Search order:
    1. CUDA_HOME environment variable
    2. nvcc location from PATH
    3. Common installation paths
    4. Default to /usr/local/cuda
    """
    # Check environment variable first
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home

    # Try to find nvcc in PATH
    try:
        nvcc = subprocess.check_output(['which', 'nvcc'], stderr=subprocess.DEVNULL).decode().strip()
        if nvcc:
            cuda_home = str(Path(nvcc).parent.parent)
            if os.path.isdir(cuda_home):
                return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try common paths
    common_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-12',
        '/usr/local/cuda-11',
        '/opt/cuda',
    ]
    for path in common_paths:
        if os.path.isdir(path):
            return path

    # Default fallback
    print("Warning: CUDA_HOME not set and nvcc not in PATH. Using /usr/local/cuda")
    return '/usr/local/cuda'


# CUDA configuration
CUDA_HOME = find_cuda_home()
NVCC = os.path.join(CUDA_HOME, 'bin', 'nvcc')

# Verify CUDA installation
if not os.path.isfile(NVCC):
    raise RuntimeError(
        f"nvcc not found at {NVCC}. "
        f"Please set CUDA_HOME environment variable or ensure CUDA is installed."
    )


class CudaBuildExt(build_ext):
    """
    Custom build_ext command to compile CUDA files with nvcc.

    Features:
    - Automatic CUDA architecture detection
    - Custom architecture specification via --cuda-arch
    - Proper compilation flags for CUDA/C++ interoperability
    """

    user_options = build_ext.user_options + [
        ('cuda-arch=', None, 'Comma-separated CUDA architectures (e.g., 75,80,86,90)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cuda_arch = None

    def finalize_options(self):
        super().finalize_options()
        if self.cuda_arch is None:
            # Default architectures: Turing, Ampere, Ada Lovelace, Hopper
            self.cuda_arch = ['75', '80', '86', '89', '90']
        elif isinstance(self.cuda_arch, str):
            self.cuda_arch = [a.strip() for a in self.cuda_arch.split(',')]

    def build_extension(self, ext):
        """Build a single extension with CUDA support."""
        # Separate CUDA and non-CUDA source files
        cuda_sources = [s for s in ext.sources if s.endswith('.cu')]
        cpp_sources = [s for s in ext.sources if not s.endswith('.cu')]

        # Get output file path
        ext_fullpath = self.get_ext_fullpath(ext.name)
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        # Compile CUDA sources
        cuda_objects = []
        for cuda_src in cuda_sources:
            cuda_obj = os.path.join(build_temp, os.path.splitext(os.path.basename(cuda_src))[0] + '.o')
            self._compile_cuda(cuda_src, cuda_obj, ext)
            cuda_objects.append(cuda_obj)

        # Compile C++ sources
        cpp_objects = []
        for cpp_src in cpp_sources:
            cpp_obj = os.path.join(build_temp, os.path.splitext(os.path.basename(cpp_src))[0] + '.o')
            self._compile_cpp(cpp_src, cpp_obj, ext)
            cpp_objects.append(cpp_obj)

        # Link everything together
        all_objects = cuda_objects + cpp_objects
        self._link(all_objects, ext_fullpath, ext)

    def _compile_cuda(self, source, output, ext):
        """Compile a CUDA source file using nvcc."""
        # Generate architecture flags
        arch_flags = []
        for arch in self.cuda_arch:
            arch_flags.extend([
                f'--generate-code=arch=compute_{arch},code=sm_{arch}'
            ])

        # Include directories
        include_dirs = [
            pybind11.get_include(),
            os.path.join(CUDA_HOME, 'include'),
        ]
        if hasattr(sys, 'prefix'):
            include_dirs.append(os.path.join(sys.prefix, 'include'))

        # Build command
        cmd = [
            NVCC,
            '-c', source,
            '-o', output,
            '-O3',
            '-std=c++17',
            '-Xcompiler', '-fPIC',
            '-Xcompiler', '-fvisibility=hidden',
        ]

        # Add include directories
        for inc in include_dirs:
            cmd.extend(['-I', inc])

        # Add architecture flags
        cmd.extend(arch_flags)

        print(f"Compiling CUDA: {source}")
        subprocess.check_call(cmd)

    def _compile_cpp(self, source, output, ext):
        """Compile a C++ source file using the system compiler."""
        # Include directories
        include_dirs = [
            pybind11.get_include(),
            os.path.join(CUDA_HOME, 'include'),
        ]
        if hasattr(sys, 'prefix'):
            include_dirs.append(os.path.join(sys.prefix, 'include'))

        # Get Python include path
        try:
            import sysconfig
            python_include = sysconfig.get_path('include')
            if python_include:
                include_dirs.append(python_include)
        except Exception:
            pass

        # Build command
        cmd = [
            'g++',
            '-c', source,
            '-o', output,
            '-O3',
            '-std=c++17',
            '-fPIC',
            '-fvisibility=hidden',
        ]

        # Add include directories
        for inc in include_dirs:
            cmd.extend(['-I', inc])

        print(f"Compiling C++: {source}")
        subprocess.check_call(cmd)

    def _link(self, objects, output, ext):
        """Link object files into a shared library."""
        # Library directories
        library_dirs = [
            os.path.join(CUDA_HOME, 'lib64'),
            os.path.join(CUDA_HOME, 'lib'),
        ]

        # Build command using nvcc as linker
        cmd = [
            NVCC,
            '-shared',
            '-o', output,
        ]

        # Add object files
        cmd.extend(objects)

        # Add library directories
        for lib_dir in library_dirs:
            if os.path.isdir(lib_dir):
                cmd.extend(['-L', lib_dir])

        # Add libraries
        cmd.extend(['-lcudart', '-lcurand'])

        # Compiler flags for linking
        cmd.extend(['-Xcompiler', '-fPIC'])

        print(f"Linking: {output}")
        subprocess.check_call(cmd)


# Define the extension module
cuda_module = Extension(
    'bates_kernel_cpp',
    sources=[
        'bates_kernel.cu',
        'bates_wrapper.cu',
    ],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(CUDA_HOME, 'include'),
    ],
    library_dirs=[
        os.path.join(CUDA_HOME, 'lib64'),
        os.path.join(CUDA_HOME, 'lib'),
    ],
    libraries=['cudart', 'curand'],
    language='c++',
)


# Package metadata
setup(
    name='bates_kernel_cpp',
    version='1.1.0',
    author='Monte Carlo Sim CUDA Contributors',
    description='GPU-accelerated Bates model Monte Carlo option pricing',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=[cuda_module],
    cmdclass={'build_ext': CudaBuildExt},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pybind11>=2.10.0',
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'mypy>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    keywords='monte-carlo cuda gpu finance options bates heston stochastic-volatility',
    zip_safe=False,
)
