/**
 * @file bates_wrapper.cu
 * @brief PyBind11 wrapper for Bates model path generation on GPU
 *
 * Exposes the generate_paths() function to Python for generating
 * stock price paths using the Bates stochastic volatility + jump model.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include "cuda_runtime.h"

namespace py = pybind11;

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA kernel error at ") + __FILE__ + ":" +        \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

// ============================================================================
// RAII Wrapper for GPU Memory
// ============================================================================

template <typename T>
class CudaMemory {
public:
    CudaMemory() : ptr_(nullptr), size_(0) {}

    explicit CudaMemory(size_t count) : ptr_(nullptr), size_(count * sizeof(T)) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size_));
        }
    }

    ~CudaMemory() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    CudaMemory(CudaMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_;
    size_t size_;
};

// ============================================================================
// Input Validation
// ============================================================================

inline void validate_path_generation_parameters(
    int num_paths, int num_steps, double T,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double sigma_j,
    size_t z1_size, size_t z2_size
) {
    if (num_paths <= 0) {
        throw std::invalid_argument("num_paths must be positive, got " + std::to_string(num_paths));
    }
    if (num_paths > 100000000) {
        throw std::invalid_argument("num_paths exceeds maximum (100M), got " + std::to_string(num_paths));
    }
    if (num_steps <= 0) {
        throw std::invalid_argument("num_steps must be positive, got " + std::to_string(num_steps));
    }
    if (num_steps > 10000) {
        throw std::invalid_argument("num_steps exceeds maximum (10000), got " + std::to_string(num_steps));
    }
    if (T <= 0.0) {
        throw std::invalid_argument("Time to maturity T must be positive, got " + std::to_string(T));
    }
    if (S0 <= 0.0) {
        throw std::invalid_argument("Initial stock price S0 must be positive, got " + std::to_string(S0));
    }
    if (v0 < 0.0) {
        throw std::invalid_argument("Initial variance v0 cannot be negative, got " + std::to_string(v0));
    }
    if (kappa <= 0.0) {
        throw std::invalid_argument("Mean reversion speed kappa must be positive, got " + std::to_string(kappa));
    }
    if (theta < 0.0) {
        throw std::invalid_argument("Long-term variance theta cannot be negative, got " + std::to_string(theta));
    }
    if (xi < 0.0) {
        throw std::invalid_argument("Vol-of-vol xi cannot be negative, got " + std::to_string(xi));
    }
    if (rho < -1.0 || rho > 1.0) {
        throw std::invalid_argument("Correlation rho must be in [-1, 1], got " + std::to_string(rho));
    }
    if (lambda_j < 0.0) {
        throw std::invalid_argument("Jump intensity lambda_j cannot be negative, got " + std::to_string(lambda_j));
    }
    if (sigma_j < 0.0) {
        throw std::invalid_argument("Jump volatility sigma_j cannot be negative, got " + std::to_string(sigma_j));
    }

    // Validate array dimensions
    size_t expected_z_size = static_cast<size_t>(num_steps) * num_paths;
    if (z1_size != expected_z_size) {
        throw std::invalid_argument(
            "Z1 array size mismatch: expected " + std::to_string(expected_z_size) +
            ", got " + std::to_string(z1_size));
    }
    if (z2_size != expected_z_size) {
        throw std::invalid_argument(
            "Z2 array size mismatch: expected " + std::to_string(expected_z_size) +
            ", got " + std::to_string(z2_size));
    }
}

// Forward declaration of CUDA kernel
__global__ void bates_path_generator_kernel(
    double* S_path,
    const int num_paths, const int num_steps, const double T,
    const double S0, const double r, double v0,
    const double kappa, const double theta, const double xi, const double rho,
    const double lambda_j, const double mu_j, const double sigma_j,
    const double k_drift,
    const double* Z1_in, const double* Z2_in,
    const unsigned long long seed);

// Forward declaration of full pricing pipeline (defined in bates_kernel.cu)
extern "C" double price_bates_full_sequence(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
);

// Forward declaration of block-size-tunable pricing pipeline (defined in bates_kernel.cu)
extern "C" double price_bates_full_sequence_with_block_size(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block
);

// Forward declaration of configurable pricing pipeline (defined in bates_kernel.cu)
extern "C" double price_bates_full_sequence_with_config(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block, int num_streams
);

// Forward declaration of mixed-precision pricing pipeline (defined in bates_kernel.cu)
extern "C" double price_bates_full_sequence_mixed_precision(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
);

// Forward declaration of mixed-precision configurable pipeline (defined in bates_kernel.cu)
extern "C" double price_bates_full_sequence_mixed_precision_with_config(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block, int num_streams
);

/**
 * @brief Generate Bates model stock price paths on GPU
 *
 * @param num_paths Number of Monte Carlo paths
 * @param num_steps Number of time steps
 * @param T Time to maturity
 * @param S0 Initial stock price
 * @param r Risk-free rate
 * @param v0 Initial variance
 * @param kappa Mean reversion speed
 * @param theta Long-term variance
 * @param xi Volatility of variance
 * @param rho Correlation
 * @param lambda_j Jump intensity
 * @param mu_j Mean of log jump size
 * @param sigma_j Std dev of log jump size
 * @param k_drift Jump-compensated drift
 * @param Z1_py Pre-generated random numbers for variance
 * @param Z2_py Pre-generated random numbers for stock
 * @return NumPy array of shape [num_steps+1, num_paths]
 */
py::array_t<double> generate_bates_paths_cpp(
    int num_paths, int num_steps, double T,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    double k_drift,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z1_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> Z2_py
) {
    // Validate input parameters
    validate_path_generation_parameters(
        num_paths, num_steps, T, S0, r, v0,
        kappa, theta, xi, rho, lambda_j, sigma_j,
        Z1_py.size(), Z2_py.size());

    // Calculate sizes
    size_t s_path_size = static_cast<size_t>(num_steps + 1) * num_paths * sizeof(double);
    size_t z_size = static_cast<size_t>(num_steps) * num_paths * sizeof(double);

    // Allocate GPU memory with RAII
    CudaMemory<double> d_S_path(static_cast<size_t>(num_steps + 1) * num_paths);
    CudaMemory<double> d_Z1(static_cast<size_t>(num_steps) * num_paths);
    CudaMemory<double> d_Z2(static_cast<size_t>(num_steps) * num_paths);

    // Copy input data from Python (NumPy array) to GPU
    CUDA_CHECK(cudaMemcpy(d_Z1.get(), Z1_py.data(), z_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Z2.get(), Z2_py.data(), z_size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Launch kernel
    bates_path_generator_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path.get(), num_paths, num_steps, T,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift,
        d_Z1.get(), d_Z2.get(), seed
    );
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from GPU back to Python
    auto S_path_py = py::array_t<double>({num_steps + 1, num_paths});
    CUDA_CHECK(cudaMemcpy(S_path_py.mutable_data(), d_S_path.get(), s_path_size, cudaMemcpyDeviceToHost));

    // Memory automatically freed by RAII destructors
    return S_path_py;
}

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(bates_kernel_cpp, m) {
    m.doc() = R"doc(
        Custom CUDA C++ kernel for Bates model path generation.

        The Bates model combines Heston stochastic volatility with Merton
        jump-diffusion for realistic option pricing that captures:
        - Mean-reverting stochastic variance
        - Jump risk (crash/rally)
        - Correlation between price and volatility

        Functions:
            generate_paths: Generate Monte Carlo paths on GPU
    )doc";

    m.def("generate_paths", &generate_bates_paths_cpp,
          R"doc(
              Generate stock price paths using the Bates model on GPU.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity (jumps per year)
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size
                  k_drift (float): Jump-compensated drift
                  Z1 (ndarray): Random numbers for variance [num_steps x num_paths]
                  Z2 (ndarray): Random numbers for stock [num_steps x num_paths]

              Returns:
                  ndarray: Stock price paths [num_steps+1 x num_paths]
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"), py::arg("k_drift"),
          py::arg("Z1"), py::arg("Z2")
    );

    m.def("price_bates_full_sequence", &price_bates_full_sequence,
          R"doc(
              Price an Asian put option using the full Bates CUDA pipeline.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size

              Returns:
                  float: Discounted option price
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j")
    );

    m.def("price_bates_full_sequence_with_block_size", &price_bates_full_sequence_with_block_size,
          R"doc(
              Price an Asian put option using the full Bates CUDA pipeline with a
              configurable thread block size.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size
                  threads_per_block (int): CUDA threads per block (power of two)

              Returns:
                  float: Discounted option price
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("threads_per_block")
    );

    m.def("price_bates_full_sequence_with_config", &price_bates_full_sequence_with_config,
          R"doc(
              Price an Asian put option using the full Bates CUDA pipeline with
              configurable block size and number of streams.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size
                  threads_per_block (int): CUDA threads per block (power of two)
                  num_streams (int): CUDA streams for chunk execution

              Returns:
                  float: Discounted option price
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("threads_per_block"), py::arg("num_streams")
    );

    m.def("price_bates_full_sequence_mixed_precision", &price_bates_full_sequence_mixed_precision,
          R"doc(
              Price an Asian put option using the mixed-precision Bates CUDA pipeline.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size

              Returns:
                  float: Discounted option price
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j")
    );

    m.def("price_bates_full_sequence_mixed_precision_with_config",
          &price_bates_full_sequence_mixed_precision_with_config,
          R"doc(
              Price an Asian put option using the mixed-precision Bates CUDA pipeline
              with configurable block size and number of streams.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size
                  threads_per_block (int): CUDA threads per block (power of two)
                  num_streams (int): CUDA streams for chunk execution

              Returns:
                  float: Discounted option price
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("threads_per_block"), py::arg("num_streams")
    );
}
