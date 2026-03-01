#!/bin/bash
# Build script for bates_kernel_cpp CUDA extension
#
# Usage:
#   ./build.sh              # Build with default architectures
#   ./build.sh 86           # Build for specific architecture (sm_86)
#   ./build.sh 75,86,90     # Build for multiple architectures
#
# Supported architectures:
#   75 - Turing (RTX 20 series, T4)
#   80 - Ampere (A100)
#   86 - Ampere (RTX 30 series, A10, A40)
#   89 - Ada Lovelace (RTX 40 series, L40)
#   90 - Hopper (H100)

set -e

# --- Configuration ---
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NVCC="$CUDA_HOME/bin/nvcc"

# Check NVCC exists
if [ ! -f "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    echo "Please set CUDA_HOME environment variable or ensure CUDA is installed."
    exit 1
fi

# Get Python includes
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")

# Parse architecture argument
if [ -n "$1" ]; then
    IFS=',' read -ra ARCH_LIST <<< "$1"
else
    # Default architectures
    ARCH_LIST=(75 80 86 89 90)
fi

# Build architecture flags
ARCH_FLAGS=""
for arch in "${ARCH_LIST[@]}"; do
    ARCH_FLAGS="$ARCH_FLAGS --generate-code=arch=compute_${arch},code=sm_${arch}"
done

echo "=== Bates Model CUDA Build ==="
echo "CUDA Home: $CUDA_HOME"
echo "Architectures: ${ARCH_LIST[*]}"
echo ""

# --- Step 1: Compile CUDA kernels ---
echo ">>> Compiling bates_kernel.cu..."
$NVCC -c bates_kernel.cu -o bates_kernel.o \
    -O3 -std=c++17 \
    -Xcompiler '-fPIC -fvisibility=hidden' \
    -I "$PYBIND_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    -I "$CUDA_HOME/include" \
    $ARCH_FLAGS

# --- Step 2: Compile wrapper ---
echo ">>> Compiling bates_wrapper.cu..."
$NVCC -c bates_wrapper.cu -o bates_wrapper.o \
    -O3 -std=c++17 \
    -Xcompiler '-fPIC -fvisibility=hidden' \
    -I "$PYBIND_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    -I "$CUDA_HOME/include" \
    $ARCH_FLAGS

# --- Step 3: Link ---
echo ">>> Linking..."
$NVCC -shared -o bates_kernel_cpp.so \
    bates_kernel.o bates_wrapper.o \
    -Xcompiler '-fPIC' \
    -L "$CUDA_HOME/lib64" -L "$CUDA_HOME/lib" \
    -lcudart -lcurand

# --- Cleanup ---
rm -f bates_kernel.o bates_wrapper.o

# --- Verify ---
echo ""
echo "=== Build Complete ==="
echo "Created: bates_kernel_cpp.so"
echo ""
echo "To use:"
echo "  python -c \"import bates_kernel_cpp; print(bates_kernel_cpp.__doc__)\""
echo ""
echo "To run tests:"
echo "  pytest test_bates.py -v"
