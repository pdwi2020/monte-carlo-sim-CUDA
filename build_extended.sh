#!/bin/bash
# Build script for extended bates_kernel with QE scheme, barriers, and Greeks
#
# Usage:
#   ./build_extended.sh              # Build with default architectures
#   ./build_extended.sh 86           # Build for specific architecture (sm_86)
#   ./build_extended.sh 75,86,90     # Build for multiple architectures
#
# Supported architectures:
#   75 - Turing (RTX 20 series, T4)
#   80 - Ampere (A100)
#   86 - Ampere (RTX 30 series, A10, A40, A6000)
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

echo "=== Extended Bates Model CUDA Build ==="
echo "CUDA Home: $CUDA_HOME"
echo "Architectures: ${ARCH_LIST[*]}"
echo ""
echo "Features:"
echo "  - QE (Quadratic-Exponential) variance scheme"
echo "  - Barrier options (knock-in/knock-out)"
echo "  - Lookback options"
echo "  - Control variates"
echo "  - Greeks (Delta, Gamma, Vega, Theta, Rho)"
echo ""

# --- Step 1: Compile extended CUDA kernels ---
echo ">>> Compiling bates_kernel_extended.cu..."
$NVCC -c bates_kernel_extended.cu -o bates_kernel_extended.o \
    -O3 -std=c++17 \
    -Xcompiler '-fPIC -fvisibility=hidden' \
    -I "$PYBIND_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    -I "$CUDA_HOME/include" \
    $ARCH_FLAGS

# --- Step 2: Compile extended wrapper ---
echo ">>> Compiling bates_wrapper_extended.cpp..."
$NVCC -c bates_wrapper_extended.cpp -o bates_wrapper_extended.o \
    -O3 -std=c++17 \
    -Xcompiler '-fPIC -fvisibility=hidden' \
    -I "$PYBIND_INCLUDE" \
    -I "$PYTHON_INCLUDE" \
    -I "$CUDA_HOME/include" \
    $ARCH_FLAGS

# --- Step 3: Link ---
echo ">>> Linking..."
$NVCC -shared -o bates_extended.so \
    bates_kernel_extended.o bates_wrapper_extended.o \
    -Xcompiler '-fPIC' \
    -L "$CUDA_HOME/lib64" -L "$CUDA_HOME/lib" \
    -lcudart -lcurand

# --- Cleanup ---
rm -f bates_kernel_extended.o bates_wrapper_extended.o

# --- Verify ---
echo ""
echo "=== Build Complete ==="
echo "Created: bates_extended.so"
echo ""
echo "To use:"
echo "  python -c \"import bates_extended; print(bates_extended.__doc__)\""
echo ""
echo "Example:"
echo "  import bates_extended"
echo "  result = bates_extended.price("
echo "      num_paths=100000, num_steps=252, T=1.0, K=100.0,"
echo "      S0=100.0, r=0.05, v0=0.04,"
echo "      kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,"
echo "      lambda_j=0.1, mu_j=-0.05, sigma_j=0.1,"
echo "      payoff_type='asian_put'"
echo "  )"
echo "  print(f'Price: \${result.price:.4f}')"
echo ""
echo "To run tests:"
echo "  pytest test_bates_extended.py -v"
echo ""
