/**
 * @file bates_kernel_extended.cu
 * @brief Extended CUDA kernels with QE scheme, barrier options, and Greeks
 *
 * Extensions over the base bates_kernel.cu:
 * 1. Quadratic-Exponential (QE) scheme for variance (Andersen, 2008)
 * 2. Barrier option payoffs (knock-in/knock-out)
 * 3. Lookback option payoffs
 * 4. Greeks calculation (Delta, Gamma, Vega)
 * 5. Control variate support
 *
 * References:
 * - Andersen, L. (2008). "Simple and efficient simulation of the Heston model."
 * - Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering."
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <string>
#include <limits>
#include <memory>
#include <vector>

// ============================================================================
// CUDA Error Checking (reused from bates_kernel.cu)
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

#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t status = (call);                                        \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuRAND error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + ": error code " +                   \
                std::to_string(static_cast<int>(status)));                     \
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
// Constants
// ============================================================================

// QE scheme switching threshold
__constant__ double QE_PSI_CRIT = 1.5;

// Payoff type enum
enum PayoffType {
    ASIAN_PUT = 0,
    ASIAN_CALL = 1,
    ASIAN_GEOM_PUT = 2,
    ASIAN_GEOM_CALL = 3,
    EUROPEAN_PUT = 4,
    EUROPEAN_CALL = 5,
    BARRIER_UP_OUT_PUT = 6,
    BARRIER_UP_IN_PUT = 7,
    BARRIER_DOWN_OUT_PUT = 8,
    BARRIER_DOWN_IN_PUT = 9,
    BARRIER_UP_OUT_CALL = 10,
    BARRIER_UP_IN_CALL = 11,
    BARRIER_DOWN_OUT_CALL = 12,
    BARRIER_DOWN_IN_CALL = 13,
    LOOKBACK_FIXED_PUT = 14,
    LOOKBACK_FIXED_CALL = 15
};

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Inverse CDF of standard normal distribution (Acklam's approximation)
 */
__device__ double inv_norm_cdf(double p) {
    // Coefficients for rational approximation
    const double a1 = -3.969683028665376e+01;
    const double a2 =  2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 =  1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 =  2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 =  1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 =  6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 =  4.374664141464968e+00;
    const double c6 =  2.938163982698783e+00;

    const double d1 =  7.784695709041462e-03;
    const double d2 =  3.224671290700398e-01;
    const double d3 =  2.445134137142996e+00;
    const double d4 =  3.754408661907416e+00;

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r;

    if (p < p_low) {
        q = sqrt(-2.0 * log(p));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }
}

/**
 * @brief QE scheme for variance step
 *
 * Andersen's moment-matching scheme that:
 * - Exactly matches first two moments of variance distribution
 * - Uses quadratic scheme for psi <= psi_crit
 * - Uses exponential scheme for psi > psi_crit
 */
__device__ double qe_variance_step(
    double v_current,
    double U,  // Uniform random in (0, 1)
    double dt,
    double kappa,
    double theta,
    double xi
) {
    // Compute conditional mean and variance
    double exp_kdt = exp(-kappa * dt);
    double m = theta + (v_current - theta) * exp_kdt;

    double s2 = v_current * xi * xi * exp_kdt / kappa * (1.0 - exp_kdt) +
                theta * xi * xi / (2.0 * kappa) * (1.0 - exp_kdt) * (1.0 - exp_kdt);

    // Coefficient of variation squared
    double psi = s2 / (m * m + 1e-10);

    double v_new;

    if (psi <= QE_PSI_CRIT) {
        // Quadratic scheme
        double inv_psi = 1.0 / (psi + 1e-10);
        double b2 = 2.0 * inv_psi - 1.0 + sqrt(2.0 * inv_psi) * sqrt(fmax(2.0 * inv_psi - 1.0, 0.0));
        double a = m / (1.0 + b2);

        double Z = inv_norm_cdf(fmax(fmin(U, 0.9999999), 0.0000001));
        v_new = a * (sqrt(b2) + Z) * (sqrt(b2) + Z);
    } else {
        // Exponential scheme
        double p = (psi - 1.0) / (psi + 1.0);
        double beta = (1.0 - p) / (m + 1e-10);

        if (U <= p) {
            v_new = 0.0;
        } else {
            v_new = log((1.0 - p) / (1.0 - U + 1e-10)) / (beta + 1e-10);
        }
    }

    return fmax(v_new, 0.0);
}

// ============================================================================
// QE Scheme Bates Path Generator
// ============================================================================

/**
 * @brief KERNEL: Bates model path simulation with QE variance scheme
 *
 * Uses Andersen's Quadratic-Exponential scheme for variance,
 * providing better accuracy and no negative variance.
 */
__global__ void bates_path_qe_kernel(
    double* S_path,
    double* v_path,
    const int num_paths,
    const int num_steps,
    const double T,
    const double S0,
    const double r,
    const double v0,
    const double kappa,
    const double theta,
    const double xi,
    const double rho,
    const double lambda_j,
    const double mu_j,
    const double sigma_j,
    const double k_drift,
    const double* Z_S_in,  // Standard normal for stock
    const double* U_v_in,  // Uniform for QE variance
    const unsigned long long seed
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const double dt = T / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);

    // Initialize per-thread RNG for jumps
    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;

    // Store initial values
    S_path[static_cast<size_t>(path_id)] = current_S;
    if (v_path != nullptr) {
        v_path[static_cast<size_t>(path_id)] = current_v;
    }

    // Precompute correlation complement
    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        // Generate jumps
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);
        double jump_component = 0.0;
        for (unsigned int j = 0; j < num_jumps; ++j) {
            jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
        }

        // Load random numbers
        const size_t idx = static_cast<size_t>(step) * num_paths + path_id;
        const double Z_S = Z_S_in[idx];
        const double U_v = U_v_in[idx];

        // QE step for variance
        double v_next = qe_variance_step(current_v, U_v, dt, kappa, theta, xi);

        // Correlated Brownian motion for stock using current variance
        double v_pos = fmax(current_v, 0.0);
        double sqrt_v = sqrt(v_pos);

        // Generate Z_v from U_v for correlation
        double Z_v = inv_norm_cdf(fmax(fmin(U_v, 0.9999999), 0.0000001));
        double W_S = rho * Z_v + rho_compl * Z_S;

        // Update stock price
        current_S *= exp((r - k_drift - 0.5 * v_pos) * dt +
                         sqrt_v * W_S * sqrt_dt + jump_component);
        current_v = v_next;

        // Store results using 64-bit indexing to avoid overflow.
        const size_t s_idx = static_cast<size_t>(step + 1) * static_cast<size_t>(num_paths) +
                             static_cast<size_t>(path_id);
        S_path[s_idx] = current_S;
        if (v_path != nullptr) {
            v_path[s_idx] = current_v;
        }
    }
}

// ============================================================================
// Barrier Option Payoff Kernel
// ============================================================================

/**
 * @brief KERNEL: Compute barrier option payoffs from paths
 */
__global__ void barrier_payoff_kernel(
    const double* S_path,
    double* payoffs,
    const int num_paths,
    const int num_steps,
    const double K,
    const double barrier,
    const double rebate,
    const int payoff_type  // PayoffType enum
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    // Find max and min along path
    double max_S = S_path[path_id];
    double min_S = S_path[path_id];
    double sum_S = S_path[path_id];

    for (int step = 1; step <= num_steps; ++step) {
        const size_t s_idx = static_cast<size_t>(step) * static_cast<size_t>(num_paths) +
                             static_cast<size_t>(path_id);
        double S = S_path[s_idx];
        max_S = fmax(max_S, S);
        min_S = fmin(min_S, S);
        sum_S += S;
    }

    const size_t s_idx = static_cast<size_t>(num_steps) * static_cast<size_t>(num_paths) +
                         static_cast<size_t>(path_id);
    double S_T = S_path[s_idx];
    double avg_S = sum_S / static_cast<double>(num_steps + 1);

    double payoff = 0.0;

    switch (payoff_type) {
        case ASIAN_PUT:
            payoff = fmax(K - avg_S, 0.0);
            break;

        case ASIAN_CALL:
            payoff = fmax(avg_S - K, 0.0);
            break;

        case EUROPEAN_PUT:
            payoff = fmax(K - S_T, 0.0);
            break;

        case EUROPEAN_CALL:
            payoff = fmax(S_T - K, 0.0);
            break;

        case BARRIER_UP_OUT_PUT:
            payoff = (max_S >= barrier) ? rebate : fmax(K - S_T, 0.0);
            break;

        case BARRIER_UP_IN_PUT:
            payoff = (max_S >= barrier) ? fmax(K - S_T, 0.0) : rebate;
            break;

        case BARRIER_DOWN_OUT_PUT:
            payoff = (min_S <= barrier) ? rebate : fmax(K - S_T, 0.0);
            break;

        case BARRIER_DOWN_IN_PUT:
            payoff = (min_S <= barrier) ? fmax(K - S_T, 0.0) : rebate;
            break;

        case BARRIER_UP_OUT_CALL:
            payoff = (max_S >= barrier) ? rebate : fmax(S_T - K, 0.0);
            break;

        case BARRIER_UP_IN_CALL:
            payoff = (max_S >= barrier) ? fmax(S_T - K, 0.0) : rebate;
            break;

        case BARRIER_DOWN_OUT_CALL:
            payoff = (min_S <= barrier) ? rebate : fmax(S_T - K, 0.0);
            break;

        case BARRIER_DOWN_IN_CALL:
            payoff = (min_S <= barrier) ? fmax(S_T - K, 0.0) : rebate;
            break;

        case LOOKBACK_FIXED_PUT:
            payoff = fmax(K - min_S, 0.0);
            break;

        case LOOKBACK_FIXED_CALL:
            payoff = fmax(max_S - K, 0.0);
            break;

        default:
            payoff = fmax(K - avg_S, 0.0);  // Default to Asian put
    }

    payoffs[path_id] = payoff;
}

// ============================================================================
// Combined QE + Payoff Kernel (Memory Efficient)
// ============================================================================

/**
 * @brief KERNEL: Bates QE simulation with inline payoff computation
 *
 * Computes path and payoff in one kernel without storing full paths,
 * significantly reducing memory usage.
 */
__global__ void bates_qe_payoff_kernel(
    double* payoffs,
    const int num_paths,
    const int num_steps,
    const double T,
    const double K,
    const double S0,
    const double r,
    const double v0,
    const double kappa,
    const double theta,
    const double xi,
    const double rho,
    const double lambda_j,
    const double mu_j,
    const double sigma_j,
    const double k_drift,
    const double* Z_S_in,
    const double* U_v_in,
    const unsigned long long seed,
    const int payoff_type,
    const double barrier,
    const double rebate
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const double dt = T / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);

    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;

    // Path statistics
    double sum_S = current_S;
    double sum_log_S = log(current_S);
    double max_S = current_S;
    double min_S = current_S;

    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);
        double jump_component = 0.0;
        for (unsigned int j = 0; j < num_jumps; ++j) {
            jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
        }

        const size_t idx = static_cast<size_t>(step) * num_paths + path_id;
        const double Z_S = Z_S_in[idx];
        const double U_v = U_v_in[idx];

        // QE variance step
        double v_next = qe_variance_step(current_v, U_v, dt, kappa, theta, xi);

        double v_pos = fmax(current_v, 0.0);
        double sqrt_v = sqrt(v_pos);

        double Z_v = inv_norm_cdf(fmax(fmin(U_v, 0.9999999), 0.0000001));
        double W_S = rho * Z_v + rho_compl * Z_S;

        current_S *= exp((r - k_drift - 0.5 * v_pos) * dt +
                         sqrt_v * W_S * sqrt_dt + jump_component);
        current_v = v_next;

        // Update statistics
        sum_S += current_S;
        sum_log_S += log(current_S);
        max_S = fmax(max_S, current_S);
        min_S = fmin(min_S, current_S);
    }

    double S_T = current_S;
    double avg_S = sum_S / static_cast<double>(num_steps + 1);
    double geom_avg_S = exp(sum_log_S / static_cast<double>(num_steps + 1));

    double payoff = 0.0;

    switch (payoff_type) {
        case ASIAN_PUT:
            payoff = fmax(K - avg_S, 0.0);
            break;
        case ASIAN_CALL:
            payoff = fmax(avg_S - K, 0.0);
            break;
        case ASIAN_GEOM_PUT:
            payoff = fmax(K - geom_avg_S, 0.0);
            break;
        case ASIAN_GEOM_CALL:
            payoff = fmax(geom_avg_S - K, 0.0);
            break;
        case EUROPEAN_PUT:
            payoff = fmax(K - S_T, 0.0);
            break;
        case EUROPEAN_CALL:
            payoff = fmax(S_T - K, 0.0);
            break;
        case BARRIER_UP_OUT_PUT:
            payoff = (max_S >= barrier) ? rebate : fmax(K - S_T, 0.0);
            break;
        case BARRIER_UP_IN_PUT:
            payoff = (max_S >= barrier) ? fmax(K - S_T, 0.0) : rebate;
            break;
        case BARRIER_DOWN_OUT_PUT:
            payoff = (min_S <= barrier) ? rebate : fmax(K - S_T, 0.0);
            break;
        case BARRIER_DOWN_IN_PUT:
            payoff = (min_S <= barrier) ? fmax(K - S_T, 0.0) : rebate;
            break;
        case BARRIER_UP_OUT_CALL:
            payoff = (max_S >= barrier) ? rebate : fmax(S_T - K, 0.0);
            break;
        case BARRIER_UP_IN_CALL:
            payoff = (max_S >= barrier) ? fmax(S_T - K, 0.0) : rebate;
            break;
        case BARRIER_DOWN_OUT_CALL:
            payoff = (min_S <= barrier) ? rebate : fmax(S_T - K, 0.0);
            break;
        case BARRIER_DOWN_IN_CALL:
            payoff = (min_S <= barrier) ? fmax(S_T - K, 0.0) : rebate;
            break;
        case LOOKBACK_FIXED_PUT:
            payoff = fmax(K - min_S, 0.0);
            break;
        case LOOKBACK_FIXED_CALL:
            payoff = fmax(max_S - K, 0.0);
            break;
        default:
            payoff = fmax(K - avg_S, 0.0);
    }

    payoffs[path_id] = payoff;
}

// ============================================================================
// Control Variate Payoff Kernel
// ============================================================================

/**
 * @brief KERNEL: Compute arithmetic and geometric Asian payoffs for control variate
 */
__global__ void asian_cv_payoff_kernel(
    double* arith_payoffs,
    double* geom_payoffs,
    const int num_paths,
    const int num_steps,
    const double T,
    const double K,
    const double S0,
    const double r,
    const double v0,
    const double kappa,
    const double theta,
    const double xi,
    const double rho,
    const double lambda_j,
    const double mu_j,
    const double sigma_j,
    const double k_drift,
    const double* Z_S_in,
    const double* U_v_in,
    const unsigned long long seed,
    const bool is_call  // true for call, false for put
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const double dt = T / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);

    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;

    double sum_S = current_S;
    double sum_log_S = log(current_S);

    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);
        double jump_component = 0.0;
        for (unsigned int j = 0; j < num_jumps; ++j) {
            jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
        }

        const size_t idx = static_cast<size_t>(step) * num_paths + path_id;
        const double Z_S = Z_S_in[idx];
        const double U_v = U_v_in[idx];

        double v_next = qe_variance_step(current_v, U_v, dt, kappa, theta, xi);

        double v_pos = fmax(current_v, 0.0);
        double sqrt_v = sqrt(v_pos);

        double Z_v = inv_norm_cdf(fmax(fmin(U_v, 0.9999999), 0.0000001));
        double W_S = rho * Z_v + rho_compl * Z_S;

        current_S *= exp((r - k_drift - 0.5 * v_pos) * dt +
                         sqrt_v * W_S * sqrt_dt + jump_component);
        current_v = v_next;

        sum_S += current_S;
        sum_log_S += log(current_S);
    }

    double avg_S = sum_S / static_cast<double>(num_steps + 1);
    double geom_avg_S = exp(sum_log_S / static_cast<double>(num_steps + 1));

    if (is_call) {
        arith_payoffs[path_id] = fmax(avg_S - K, 0.0);
        geom_payoffs[path_id] = fmax(geom_avg_S - K, 0.0);
    } else {
        arith_payoffs[path_id] = fmax(K - avg_S, 0.0);
        geom_payoffs[path_id] = fmax(K - geom_avg_S, 0.0);
    }
}

// ============================================================================
// Reduction Kernel
// ============================================================================

__global__ void reduce_sum_kernel(
    const double* payoffs,
    double* partial_sums,
    const int num_paths
) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double thread_sum = 0.0;
    for (unsigned int idx = i; idx < static_cast<unsigned int>(num_paths); idx += stride) {
        thread_sum += payoffs[idx];
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_sq_kernel(
    const double* payoffs,
    double* partial_sums,
    double* partial_sum_sqs,
    const int num_paths
) {
    extern __shared__ double sdata[];
    double* sdata_sq = &sdata[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    double thread_sum = 0.0;
    double thread_sum_sq = 0.0;
    for (unsigned int idx = i; idx < static_cast<unsigned int>(num_paths); idx += stride) {
        double val = payoffs[idx];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    sdata[tid] = thread_sum;
    sdata_sq[tid] = thread_sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata_sq[tid] += sdata_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
        partial_sum_sqs[blockIdx.x] = sdata_sq[0];
    }
}

// ============================================================================
// RAII Helpers (same as bates_kernel.cu)
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

    T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void memset(int value) {
        if (ptr_) {
            CUDA_CHECK(cudaMemset(ptr_, value, size_));
        }
    }

private:
    T* ptr_;
    size_t size_;
};

class CurandGenerator {
public:
    CurandGenerator() : gen_(nullptr) {
        CURAND_CHECK(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    }

    ~CurandGenerator() {
        if (gen_) {
            curandDestroyGenerator(gen_);
        }
    }

    void setSeed(unsigned long long seed) {
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen_, seed));
    }

    void generateNormalDouble(double* output, size_t n, double mean, double stddev) {
        CURAND_CHECK(curandGenerateNormalDouble(gen_, output, n, mean, stddev));
    }

    void generateUniformDouble(double* output, size_t n) {
        CURAND_CHECK(curandGenerateUniformDouble(gen_, output, n));
    }

private:
    curandGenerator_t gen_;
};

// ============================================================================
// External C Interface
// ============================================================================

extern "C" {

/**
 * @brief Price Bates model option with QE scheme
 *
 * @param num_paths Number of Monte Carlo paths
 * @param num_steps Number of time steps
 * @param T Time to maturity
 * @param K Strike price
 * @param S0 Initial stock price
 * @param r Risk-free rate
 * @param v0 Initial variance
 * @param kappa Mean reversion speed
 * @param theta Long-term variance
 * @param xi Vol of vol
 * @param rho Correlation
 * @param lambda_j Jump intensity
 * @param mu_j Mean log jump size
 * @param sigma_j Std log jump size
 * @param payoff_type PayoffType enum value
 * @param barrier Barrier level (for barrier options)
 * @param rebate Rebate (for barrier options)
 * @param price Output price
 * @param std_error Output standard error
 */
void price_bates_qe(
    int num_paths,
    int num_steps,
    double T,
    double K,
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    int payoff_type,
    double barrier,
    double rebate,
    double* price,
    double* std_error
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

    // Jump-compensated drift
    double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);

    // Seed
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Allocate memory
    size_t num_randoms = static_cast<size_t>(num_paths) * num_steps;

    CudaMemory<double> d_Z_S(num_randoms);
    CudaMemory<double> d_U_v(num_randoms);
    CudaMemory<double> d_payoffs(num_paths);

    // Generate random numbers
    CurandGenerator gen;
    gen.setSeed(seed);
    gen.generateNormalDouble(d_Z_S.get(), num_randoms, 0.0, 1.0);
    gen.generateUniformDouble(d_U_v.get(), num_randoms);

    // Launch kernel
    bates_qe_payoff_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_payoffs.get(),
        num_paths, num_steps, T, K, S0, r, v0,
        kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift,
        d_Z_S.get(), d_U_v.get(), seed,
        payoff_type, barrier, rebate
    );
    CUDA_CHECK_KERNEL();

    // Reduce
    int reduction_blocks = std::min(blocks_per_grid, 256);
    CudaMemory<double> d_partial_sums(reduction_blocks);
    CudaMemory<double> d_partial_sum_sqs(reduction_blocks);

    reduce_sum_sq_kernel<<<reduction_blocks, threads_per_block,
                          2 * threads_per_block * sizeof(double)>>>(
        d_payoffs.get(), d_partial_sums.get(), d_partial_sum_sqs.get(), num_paths
    );
    CUDA_CHECK_KERNEL();

    // Copy back
    std::vector<double> h_partial_sums(reduction_blocks);
    std::vector<double> h_partial_sum_sqs(reduction_blocks);

    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums.get(),
                          reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_sum_sqs.data(), d_partial_sum_sqs.get(),
                          reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    // Final reduction on CPU
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < reduction_blocks; ++i) {
        sum += h_partial_sums[i];
        sum_sq += h_partial_sum_sqs[i];
    }

    double mean = sum / num_paths;
    double variance = sum_sq / num_paths - mean * mean;
    double discount = exp(-r * T);

    *price = mean * discount;
    *std_error = sqrt(variance / num_paths) * discount;
}

/**
 * @brief Price Asian put with control variates using QE scheme
 */
void price_asian_cv(
    int num_paths,
    int num_steps,
    double T,
    double K,
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    double geom_analytical,  // Pre-computed analytical geometric Asian price
    bool is_call,
    double* price,
    double* std_error,
    double* cv_beta
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

    double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    size_t num_randoms = static_cast<size_t>(num_paths) * num_steps;

    CudaMemory<double> d_Z_S(num_randoms);
    CudaMemory<double> d_U_v(num_randoms);
    CudaMemory<double> d_arith_payoffs(num_paths);
    CudaMemory<double> d_geom_payoffs(num_paths);

    CurandGenerator gen;
    gen.setSeed(seed);
    gen.generateNormalDouble(d_Z_S.get(), num_randoms, 0.0, 1.0);
    gen.generateUniformDouble(d_U_v.get(), num_randoms);

    asian_cv_payoff_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_arith_payoffs.get(),
        d_geom_payoffs.get(),
        num_paths, num_steps, T, K, S0, r, v0,
        kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift,
        d_Z_S.get(), d_U_v.get(), seed,
        is_call
    );
    CUDA_CHECK_KERNEL();

    // Copy payoffs to host for control variate computation
    std::vector<double> h_arith(num_paths);
    std::vector<double> h_geom(num_paths);

    CUDA_CHECK(cudaMemcpy(h_arith.data(), d_arith_payoffs.get(),
                          num_paths * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_geom.data(), d_geom_payoffs.get(),
                          num_paths * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute control variate statistics
    double sum_arith = 0.0, sum_geom = 0.0;
    double sum_arith_geom = 0.0, sum_geom_sq = 0.0;

    for (int i = 0; i < num_paths; ++i) {
        sum_arith += h_arith[i];
        sum_geom += h_geom[i];
        sum_arith_geom += h_arith[i] * h_geom[i];
        sum_geom_sq += h_geom[i] * h_geom[i];
    }

    double mean_arith = sum_arith / num_paths;
    double mean_geom = sum_geom / num_paths;
    double cov = sum_arith_geom / num_paths - mean_arith * mean_geom;
    double var_geom = sum_geom_sq / num_paths - mean_geom * mean_geom;

    double beta = cov / (var_geom + 1e-10);

    // Adjusted mean
    double adjusted_mean = mean_arith - beta * (mean_geom - geom_analytical);

    // Compute variance of adjusted payoffs
    double sum_adj_sq = 0.0;
    for (int i = 0; i < num_paths; ++i) {
        double adj = h_arith[i] - beta * (h_geom[i] - geom_analytical);
        sum_adj_sq += (adj - adjusted_mean) * (adj - adjusted_mean);
    }
    double var_adj = sum_adj_sq / (num_paths - 1);

    double discount = exp(-r * T);

    *price = adjusted_mean * discount;
    *std_error = sqrt(var_adj / num_paths) * discount;
    *cv_beta = beta;
}

/**
 * @brief Calculate Delta using finite difference
 */
void calculate_delta(
    int num_paths,
    int num_steps,
    double T,
    double K,
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    int payoff_type,
    double barrier,
    double rebate,
    double bump_pct,
    double* delta
) {
    double dS = S0 * bump_pct;
    double price_up, price_down, se_up, se_down;

    price_bates_qe(num_paths, num_steps, T, K, S0 + dS, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_up, &se_up);

    price_bates_qe(num_paths, num_steps, T, K, S0 - dS, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_down, &se_down);

    *delta = (price_up - price_down) / (2 * dS);
}

/**
 * @brief Calculate Gamma using finite difference
 */
void calculate_gamma(
    int num_paths,
    int num_steps,
    double T,
    double K,
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    int payoff_type,
    double barrier,
    double rebate,
    double bump_pct,
    double* gamma
) {
    double dS = S0 * bump_pct;
    double price_up, price_mid, price_down, se;

    price_bates_qe(num_paths, num_steps, T, K, S0 + dS, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_up, &se);

    price_bates_qe(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_mid, &se);

    price_bates_qe(num_paths, num_steps, T, K, S0 - dS, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_down, &se);

    *gamma = (price_up - 2 * price_mid + price_down) / (dS * dS);
}

/**
 * @brief Calculate Vega using finite difference on initial variance
 */
void calculate_vega(
    int num_paths,
    int num_steps,
    double T,
    double K,
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    int payoff_type,
    double barrier,
    double rebate,
    double bump_pct,
    double* vega
) {
    double dv = v0 * bump_pct;
    double price_up, price_down, se;

    price_bates_qe(num_paths, num_steps, T, K, S0, r, v0 + dv,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_up, &se);

    price_bates_qe(num_paths, num_steps, T, K, S0, r, v0 - dv,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_down, &se);

    // Vega per 1% vol change (v = sigma^2, so dsigma = dv / (2*sqrt(v)))
    double dsigma = dv / (2 * sqrt(v0));
    *vega = (price_up - price_down) / (2 * dsigma) * 0.01;
}

// ============================================================================
// Batch Pricing Kernel (Price Multiple Options)
// ============================================================================

/**
 * @brief KERNEL: Batch pricing - compute payoffs for multiple strikes
 *
 * Reuses simulated paths across multiple option specifications.
 * Memory efficient: stores only terminal values and statistics.
 */
__global__ void batch_payoff_kernel(
    const double* S_T,           // Terminal stock prices (num_paths)
    const double* max_S,         // Path maxima (num_paths)
    const double* min_S,         // Path minima (num_paths)
    const double* avg_S,         // Path averages (num_paths)
    double* payoffs,             // Output: (num_options * num_paths)
    const double* strikes,       // Option strikes (num_options)
    const int* payoff_types,     // PayoffType for each option (num_options)
    const double* barriers,      // Barrier levels (num_options)
    const double* rebates,       // Rebates (num_options)
    const int num_paths,
    const int num_options
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    // Load path data once
    double s_T = S_T[path_id];
    double s_max = max_S[path_id];
    double s_min = min_S[path_id];
    double s_avg = avg_S[path_id];

    // Compute payoff for each option
    for (int opt = 0; opt < num_options; ++opt) {
        double K = strikes[opt];
        int ptype = payoff_types[opt];
        double barrier = barriers[opt];
        double rebate = rebates[opt];

        double payoff = 0.0;

        switch (ptype) {
            case ASIAN_PUT:
                payoff = fmax(K - s_avg, 0.0);
                break;
            case ASIAN_CALL:
                payoff = fmax(s_avg - K, 0.0);
                break;
            case EUROPEAN_PUT:
                payoff = fmax(K - s_T, 0.0);
                break;
            case EUROPEAN_CALL:
                payoff = fmax(s_T - K, 0.0);
                break;
            case BARRIER_UP_OUT_CALL:
                payoff = (s_max >= barrier) ? rebate : fmax(s_T - K, 0.0);
                break;
            case BARRIER_DOWN_OUT_PUT:
                payoff = (s_min <= barrier) ? rebate : fmax(K - s_T, 0.0);
                break;
            case LOOKBACK_FIXED_CALL:
                payoff = fmax(s_max - K, 0.0);
                break;
            case LOOKBACK_FIXED_PUT:
                payoff = fmax(K - s_min, 0.0);
                break;
            default:
                payoff = fmax(s_T - K, 0.0);
        }

        payoffs[opt * num_paths + path_id] = payoff;
    }
}

/**
 * @brief KERNEL: Warp-level reduction using shuffle
 *
 * Uses __shfl_down_sync for efficient warp-level reduction.
 * Much faster than shared memory for small reductions.
 */
__device__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief KERNEL: Optimized reduction with warp primitives
 */
__global__ void reduce_sum_warp_kernel(
    const double* data,
    double* partial_sums,
    const int n
) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop for coalesced memory access
    double thread_sum = 0.0;
    for (unsigned int idx = i; idx < static_cast<unsigned int>(n); idx += blockDim.x * gridDim.x) {
        thread_sum += data[idx];
    }

    // Warp-level reduction first
    thread_sum = warp_reduce_sum(thread_sum);

    // Write warp results to shared memory
    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        sdata[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        thread_sum = (tid < num_warps) ? sdata[tid] : 0.0;
        thread_sum = warp_reduce_sum(thread_sum);

        if (tid == 0) {
            partial_sums[blockIdx.x] = thread_sum;
        }
    }
}

/**
 * @brief Price batch of options with shared path simulation
 */
void price_batch(
    int num_paths,
    int num_steps,
    int num_options,
    double T,
    const double* strikes,      // Array of strikes
    const int* payoff_types,    // Array of PayoffType
    const double* barriers,     // Array of barriers
    const double* rebates,      // Array of rebates
    double S0,
    double r,
    double v0,
    double kappa,
    double theta,
    double xi,
    double rho,
    double lambda_j,
    double mu_j,
    double sigma_j,
    double* prices,             // Output array
    double* std_errors          // Output array
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;

    double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    size_t num_randoms = static_cast<size_t>(num_paths) * num_steps;

    // Allocate memory for simulation
    CudaMemory<double> d_Z_S(num_randoms);
    CudaMemory<double> d_U_v(num_randoms);

    // Path statistics
    CudaMemory<double> d_S_T(num_paths);
    CudaMemory<double> d_max_S(num_paths);
    CudaMemory<double> d_min_S(num_paths);
    CudaMemory<double> d_avg_S(num_paths);

    // Option data on device
    CudaMemory<double> d_strikes(num_options);
    CudaMemory<int> d_payoff_types(num_options);
    CudaMemory<double> d_barriers(num_options);
    CudaMemory<double> d_rebates(num_options);

    // Copy option parameters
    CUDA_CHECK(cudaMemcpy(d_strikes.get(), strikes, num_options * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_payoff_types.get(), payoff_types, num_options * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_barriers.get(), barriers, num_options * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rebates.get(), rebates, num_options * sizeof(double), cudaMemcpyHostToDevice));

    // Generate random numbers
    CurandGenerator gen;
    gen.setSeed(seed);
    gen.generateNormalDouble(d_Z_S.get(), num_randoms, 0.0, 1.0);
    gen.generateUniformDouble(d_U_v.get(), num_randoms);

    // First pass: simulate paths and collect statistics
    // (Using fused kernel would be more efficient in production)
    CudaMemory<double> d_S_path(static_cast<size_t>(num_paths) * (num_steps + 1));

    bates_path_qe_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path.get(), nullptr,
        num_paths, num_steps, T, S0, r, v0,
        kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift,
        d_Z_S.get(), d_U_v.get(), seed
    );
    CUDA_CHECK_KERNEL();

    // Extract statistics (could be fused into path kernel for efficiency)
    barrier_payoff_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path.get(), d_S_T.get(),
        num_paths, num_steps,
        0.0, 0.0, 0.0, EUROPEAN_CALL  // Dummy values - just extracting S_T
    );
    CUDA_CHECK_KERNEL();

    // Batch payoff computation
    CudaMemory<double> d_payoffs(static_cast<size_t>(num_options) * num_paths);

    batch_payoff_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_T.get(), d_max_S.get(), d_min_S.get(), d_avg_S.get(),
        d_payoffs.get(),
        d_strikes.get(), d_payoff_types.get(), d_barriers.get(), d_rebates.get(),
        num_paths, num_options
    );
    CUDA_CHECK_KERNEL();

    // Reduce payoffs for each option
    int reduction_blocks = std::min(blocks_per_grid, 256);
    CudaMemory<double> d_partial_sums(reduction_blocks);
    CudaMemory<double> d_partial_sum_sqs(reduction_blocks);

    double discount = exp(-r * T);

    for (int opt = 0; opt < num_options; ++opt) {
        reduce_sum_sq_kernel<<<reduction_blocks, threads_per_block,
                              2 * threads_per_block * sizeof(double)>>>(
            d_payoffs.get() + opt * num_paths,
            d_partial_sums.get(), d_partial_sum_sqs.get(), num_paths
        );
        CUDA_CHECK_KERNEL();

        std::vector<double> h_partial_sums(reduction_blocks);
        std::vector<double> h_partial_sum_sqs(reduction_blocks);

        CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums.get(),
                              reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_partial_sum_sqs.data(), d_partial_sum_sqs.get(),
                              reduction_blocks * sizeof(double), cudaMemcpyDeviceToHost));

        double sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < reduction_blocks; ++i) {
            sum += h_partial_sums[i];
            sum_sq += h_partial_sum_sqs[i];
        }

        double mean = sum / num_paths;
        double variance = sum_sq / num_paths - mean * mean;

        prices[opt] = mean * discount;
        std_errors[opt] = sqrt(variance / num_paths) * discount;
    }
}

// ============================================================================
// Memory-Coalesced Path Statistics Kernel
// ============================================================================

/**
 * @brief KERNEL: Compute path statistics with coalesced memory access
 *
 * Uses Structure-of-Arrays (SoA) layout for optimal memory bandwidth.
 */
__global__ void compute_path_stats_coalesced(
    const double* S_path,        // SoA layout: all S[0], then all S[1], ...
    double* S_T,                 // Terminal values
    double* max_S,               // Path maxima
    double* min_S,               // Path minima
    double* sum_S,               // Path sums (for average)
    const int num_paths,
    const int num_steps
) {
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    // Coalesced reads: threads in a warp read consecutive memory
    double local_max = S_path[path_id];  // S[0] for this path
    double local_min = local_max;
    double local_sum = local_max;

    for (int step = 1; step <= num_steps; ++step) {
        // Coalesced: all threads read from step * num_paths + thread_id
        double S = S_path[step * num_paths + path_id];
        local_max = fmax(local_max, S);
        local_min = fmin(local_min, S);
        local_sum += S;
    }

    // Coalesced writes
    S_T[path_id] = S_path[num_steps * num_paths + path_id];
    max_S[path_id] = local_max;
    min_S[path_id] = local_min;
    sum_S[path_id] = local_sum;
}

/**
 * @brief Get GPU device properties
 */
void get_gpu_info(
    char* name,
    int* compute_major,
    int* compute_minor,
    size_t* total_memory,
    int* multiprocessors
) {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    strcpy(name, props.name);
    *compute_major = props.major;
    *compute_minor = props.minor;
    *total_memory = props.totalGlobalMem;
    *multiprocessors = props.multiProcessorCount;
}

}  // extern "C"
