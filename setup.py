import os
import subprocess
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --- Helper function to find CUDA path ---
def find_cuda_home():
    """Find the CUDA installation path."""
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        try:
            nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: CUDA_HOME environment variable not set and nvcc not in PATH. Assuming /usr/local/cuda")
            cuda_home = '/usr/local/cuda'
    return cuda_home

CUDA_HOME = find_cuda_home()
NVCC = os.path.join(CUDA_HOME, 'bin', 'nvcc')

# --- Custom build_ext command ---
# This inherits from the standard build_ext class
class CudaBuildExt(build_ext):
    """
    Custom build_ext command to compile CUDA files with nvcc.
    """
    # This is a mapping from source file extension to compiler
    # We are telling setuptools that .cu and .cuh files are compiled by nvcc
    user_options = build_ext.user_options + [
        ('cuda-arch=', None, 'CUDA architecture flags'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cuda_arch = None

    def finalize_options(self):
        super().finalize_options()
        if self.cuda_arch is None:
            # You can add more architectures here if needed
            self.cuda_arch = ['75', '80', '86']

    def build_extensions(self):
        # Add CUDA include path
        self.compiler.add_include_dir(os.path.join(CUDA_HOME, 'include'))
        
        # We need to manually tell the linker where to find CUDA libraries
        for ext in self.extensions:
            ext.library_dirs.append(os.path.join(CUDA_HOME, 'lib64'))
        
        super().build_extensions()

# This is the original compiler class that setuptools uses
from distutils.unixccompiler import UnixCCompiler
# We are overriding its _compile method
original_compile = UnixCCompiler._compile

def custom_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    """
    A custom compile function that intercepts .cu files and uses nvcc.
    """
    if ext == '.cu':
        # This is a CUDA file, use nvcc
        # We must modify the compiler executable and arguments
        # Get the original compiler (e.g., 'gcc')
        original_compiler = self.compiler_so[0]
        # Replace it with nvcc
        self.compiler_so[0] = NVCC
        # Add specific nvcc flags
        # The arch flags are now handled properly
        arch_flags = [f'--generate-code=arch=compute_{arch},code=sm_{arch}' for arch in self.compiler.cuda_arch]
        extra_postargs = ['-O3', '-std=c++17', '-Xcompiler', '-fPIC', *arch_flags]
    
    # Call the original compile function with potentially modified arguments
    original_compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)
    
    if ext == '.cu':
        # Restore the original compiler
        self.compiler_so[0] = original_compiler

# Monkey-patch the compiler
UnixCCompiler._compile = custom_compile
# Also monkey-patch our custom build class to carry the arch flags
build_ext.cuda_arch = CudaBuildExt.user_options[0][2]
build_ext.cuda_arch = None


# --- Define the Extension Module ---
cuda_module = Extension(
    'bates_kernel_cpp',
    # List all sources, both .cu and .cpp (if any)
    sources=['bates_kernel.cu', 'bates_wrapper.cu'],
    # Add pybind11's headers
    include_dirs=[pybind11.get_include()],
    # Libraries to link against
    libraries=['cudart', 'curand'],
    language='c++'
)

# --- The main setup call ---
setup(
    name='bates_kernel_cpp',
    version='1.0',
    description='Custom CUDA kernel for Bates model',
    ext_modules=[cuda_module],
    # Use our custom build class
    cmdclass={'build_ext': CudaBuildExt},
    zip_safe=False
)
