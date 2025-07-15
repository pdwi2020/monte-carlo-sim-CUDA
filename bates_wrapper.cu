#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono> // Required for the seed
#include "cuda_runtime.h"

namespace py = pybind11;

// Forward declaration of our CUDA kernel from the other .cu file.
void bates_path_generator_kernel(
    double* S_path, 
    const int num_paths, const int num_steps, const double T,
    const double S0, const double r, double v0,
    const double kappa, const double theta, const double xi, const double rho,
    const double lambda_j, const double mu_j, const double sigma_j,
    const double k_drift,
    const double* Z1_in, const double* Z2_in,
    const unsigned long long seed);

// This is the function that will be exposed to Python
py::array_t<double> generate_bates_paths_cpp(
    int num_paths, int num_steps, double T,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    double k_drift,
    // THE FIX: Use numpy arrays as inputs directly
    py::array_t<double, py::array::c_style | py::array::forcecast> Z1_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z2_py
) {
    // --- 1. Allocate GPU Memory ---
    double *d_S_path, *d_Z1, *d_Z2;
    size_t s_path_size = (size_t)(num_steps + 1) * num_paths * sizeof(double);
    size_t z_size = (size_t)num_steps * num_paths * sizeof(double);

    cudaMalloc(&d_S_path, s_path_size);
    cudaMalloc(&d_Z1, z_size);
    cudaMalloc(&d_Z2, z_size);

    // --- 2. Copy Input Data from Python (NumPy array) to GPU ---
    // Use .data() for contiguous numpy arrays
    cudaMemcpy(d_Z1, Z1_py.data(), z_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z2, Z2_py.data(), z_size, cudaMemcpyHostToDevice);

    // --- 3. Launch Kernel ---
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;
    
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    bates_path_generator_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path, num_paths, num_steps, T,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift,
        d_Z1, d_Z2, seed
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // --- 4. Copy Results from GPU back to Python ---
    auto S_path_py = py::array_t<double>({num_steps + 1, num_paths});
    cudaMemcpy(S_path_py.mutable_data(), d_S_path, s_path_size, cudaMemcpyDeviceToHost);

    // --- 5. Free GPU Memory ---
    cudaFree(d_S_path);
    cudaFree(d_Z1);
    cudaFree(d_Z2);

    return S_path_py;
}

// --- 6. Create the Python Module ---
// I also fixed the py.arg error here
PYBIND11_MODULE(bates_kernel_cpp, m) {
    m.doc() = "Custom CUDA C++ kernel for Bates model path generation";
    m.def("generate_paths", &generate_bates_paths_cpp, "Generates Bates paths on the GPU",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"), py::arg("k_drift"),
          py::arg("Z1"), py::arg("Z2") // These were py.arg, corrected to py::arg
    );
}
