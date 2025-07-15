#!/bin/bash
set -e

# --- Find Paths ---
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC="$CUDA_HOME/bin/nvcc"
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

# --- Step 1: Compile the C++ wrapper into an object file ---
# We compile the .cpp file first using g++ as it's pure C++
echo ">>> Compiling C++ wrapper (bates_wrapper.cpp) with g++..."
g++ -O3 -c bates_wrapper.cpp -o bates_wrapper.o -fPIC -std=c++17 -I "$PYTHON_INCLUDE" -I "$PYBIND_INCLUDE" -I "$CUDA_HOME/include"

# --- Step 2: Compile the CUDA code and LINK everything with NVCC ---
echo ">>> Compiling CUDA code and linking with nvcc..."
# THIS IS THE KEY CHANGE. nvcc is now the final linker.
# It will compile bates_kernel.cu and link it with the pre-compiled bates_wrapper.o
$NVCC -O3 -shared -o bates_kernel_cpp.so bates_kernel.cu bates_wrapper.o \
    -Xcompiler '-fPIC' \
    -std=c++17 \
    -L "$CUDA_HOME/lib64" -lcudart -lcurand \
    --generate-code=arch=compute_75,code=sm_75 \
    --generate-code=arch=compute_86,code=sm_86

# --- Cleanup ---
rm bates_kernel.o
rm bates_wrapper.o

echo ">>> Build successful! Created bates_kernel_cpp.so"
