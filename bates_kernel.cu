/**
 * @file bates_kernel.cu
 * @brief CUDA implementation of the Bates model for exotic option pricing
 *
 * Mathematical Model (Bates, 1996):
 * The Bates model combines Heston stochastic volatility with Merton jump-diffusion:
 *
 * Stock Price SDE:
 *   dS = (r - k_drift - 0.5*v)*dt + sqrt(v)*dW_S + J*dN_t
 *   where k_drift = lambda_j * (exp(mu_j + 0.5*sigma_j^2) - 1)
 *
 * Volatility SDE (Heston):
 *   dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
 *
 * Correlation: dW_S * dW_v = rho * dt
 *
 * Jump Process:
 *   N_t ~ Poisson(lambda_j * t)
 *   J ~ LogNormal(mu_j, sigma_j)
 *
 * References:
 * - Bates, D. (1996). "Jumps and Stochastic Volatility: Exchange Rate Processes
 *   Implicit in Deutsche Mark Options." Review of Financial Studies.
 * - Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic
 *   Volatility." Review of Financial Studies.
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
// CUDA Error Checking Utilities
// ============================================================================

/**
 * @brief Macro for CUDA error checking with file and line information
 * @param call The CUDA API call to check
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

/**
 * @brief Macro for cuRAND error checking
 * @param call The cuRAND API call to check
 */
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

/**
 * @brief Check for kernel launch errors
 */
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
// RAII Wrapper for GPU Memory Management
// ============================================================================

/**
 * @brief RAII wrapper for CUDA device memory
 * @tparam T The data type to allocate
 *
 * Ensures automatic cleanup of GPU memory on scope exit, preventing memory
 * leaks even when exceptions are thrown.
 */
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
            cudaFree(ptr_);  // Don't throw in destructor
            ptr_ = nullptr;
        }
    }

    // Disable copy
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Enable move
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
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    size_t size() const { return size_; }

    void memset(int value) {
        if (ptr_) {
            CUDA_CHECK(cudaMemset(ptr_, value, size_));
        }
    }

    void memsetAsync(int value, cudaStream_t stream) {
        if (ptr_) {
            CUDA_CHECK(cudaMemsetAsync(ptr_, value, size_, stream));
        }
    }

private:
    T* ptr_;
    size_t size_;
};

/**
 * @brief RAII wrapper for pinned host memory
 */
template <typename T>
class PinnedHostMemory {
public:
    explicit PinnedHostMemory(size_t count) : ptr_(nullptr), count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, count_ * sizeof(T)));
        }
    }

    ~PinnedHostMemory() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    PinnedHostMemory(const PinnedHostMemory&) = delete;
    PinnedHostMemory& operator=(const PinnedHostMemory&) = delete;

    PinnedHostMemory(PinnedHostMemory&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedHostMemory& operator=(PinnedHostMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() const { return ptr_; }
    size_t count() const { return count_; }

private:
    T* ptr_;
    size_t count_;
};

/**
 * @brief RAII wrapper for CUDA streams
 */
class CudaStream {
public:
    CudaStream() : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_;
};

/**
 * @brief RAII wrapper for cuRAND generator
 */
class CurandGenerator {
public:
    CurandGenerator() : gen_(nullptr) {
        CURAND_CHECK(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    }

    ~CurandGenerator() {
        if (gen_) {
            curandDestroyGenerator(gen_);
            gen_ = nullptr;
        }
    }

    // Disable copy
    CurandGenerator(const CurandGenerator&) = delete;
    CurandGenerator& operator=(const CurandGenerator&) = delete;

    // Enable move
    CurandGenerator(CurandGenerator&& other) noexcept : gen_(other.gen_) {
        other.gen_ = nullptr;
    }

    CurandGenerator& operator=(CurandGenerator&& other) noexcept {
        if (this != &other) {
            if (gen_) {
                curandDestroyGenerator(gen_);
            }
            gen_ = other.gen_;
            other.gen_ = nullptr;
        }
        return *this;
    }

    void setSeed(unsigned long long seed) {
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen_, seed));
    }

    void setStream(cudaStream_t stream) {
        CURAND_CHECK(curandSetStream(gen_, stream));
    }

    void generateNormalDouble(double* output, size_t n, double mean, double stddev) {
        CURAND_CHECK(curandGenerateNormalDouble(gen_, output, n, mean, stddev));
    }

    void generateNormalFloat(float* output, size_t n, float mean, float stddev) {
        CURAND_CHECK(curandGenerateNormal(gen_, output, n, mean, stddev));
    }

    curandGenerator_t get() const { return gen_; }

private:
    curandGenerator_t gen_;
};

// ============================================================================
// Input Validation
// ============================================================================

/**
 * @brief Validate all input parameters for the Bates model
 * @throws std::invalid_argument if any parameter is invalid
 */
inline void validate_bates_parameters(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    // Simulation parameters
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

    // Financial parameters
    if (T <= 0.0) {
        throw std::invalid_argument("Time to maturity T must be positive, got " + std::to_string(T));
    }
    if (K <= 0.0) {
        throw std::invalid_argument("Strike K must be positive, got " + std::to_string(K));
    }
    if (S0 <= 0.0) {
        throw std::invalid_argument("Initial stock price S0 must be positive, got " + std::to_string(S0));
    }

    // Heston volatility parameters
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

    // Feller condition warning (not enforced, but useful for stability)
    // 2*kappa*theta > xi^2 ensures variance stays positive

    // Jump parameters
    if (lambda_j < 0.0) {
        throw std::invalid_argument("Jump intensity lambda_j cannot be negative, got " + std::to_string(lambda_j));
    }
    if (sigma_j < 0.0) {
        throw std::invalid_argument("Jump volatility sigma_j cannot be negative, got " + std::to_string(sigma_j));
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief KERNEL 1: Bates model path generator
 *
 * Simulates stock price paths using the Bates model with:
 * - Heston stochastic volatility (mean-reverting variance)
 * - Merton-style log-normal jumps with Poisson arrival
 * - Correlated Brownian motions for stock and volatility
 *
 * Memory layout: Column-major (S_path[step * num_paths + path_id]) for coalesced access
 *
 * @param S_path Output array for stock price paths [num_steps+1, num_paths]
 * @param num_paths Number of Monte Carlo paths
 * @param num_steps Number of time steps
 * @param T Time to maturity
 * @param S0 Initial stock price
 * @param r Risk-free rate
 * @param v0 Initial variance
 * @param kappa Mean reversion speed
 * @param theta Long-term variance
 * @param xi Volatility of variance (vol-of-vol)
 * @param rho Correlation between stock and variance Brownian motions
 * @param lambda_j Jump intensity (expected jumps per year)
 * @param mu_j Mean of log jump size
 * @param sigma_j Std dev of log jump size
 * @param k_drift Drift adjustment for jump compensation
 * @param Z1_in Pre-generated standard normal samples for variance
 * @param Z2_in Pre-generated standard normal samples for stock
 * @param seed Random seed for Poisson jump generation
 */
__global__ void bates_path_generator_kernel(
    double* S_path,
    const int num_paths, const int num_steps, const double T,
    const double S0, const double r, double v0,
    const double kappa, const double theta, const double xi, const double rho,
    const double lambda_j, const double mu_j, const double sigma_j,
    const double k_drift,
    const double* Z1_in, const double* Z2_in,
    const unsigned long long seed)
{
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const double dt = T / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);

    // Initialize per-thread RNG state for Poisson jump generation
    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;

    // Store initial price (column-major layout for coalesced memory access)
    S_path[static_cast<size_t>(path_id)] = current_S;

    // Precompute correlation complement for Cholesky decomposition
    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        // Generate Poisson-distributed number of jumps
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);

        // Accumulate jump component (log-normal jumps)
        double jump_component = 0.0;
        if (num_jumps > 0) {
            for (unsigned int j = 0; j < num_jumps; ++j) {
                jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
            }
        }

        // Load pre-generated random numbers (column-major for coalescing).
        // Use 64-bit indexing to avoid overflow when num_steps * num_paths exceeds 2^31.
        const size_t idx = static_cast<size_t>(step) * static_cast<size_t>(num_paths) +
                           static_cast<size_t>(path_id);
        const double Z1 = Z1_in[idx];
        const double Z2 = Z2_in[idx];

        // Correlated Brownian motions via Cholesky decomposition:
        // W_v = Z1
        // W_S = rho * Z1 + sqrt(1 - rho^2) * Z2
        const double W_v = Z1;
        const double W_S = rho * W_v + rho_compl * Z2;

        // Ensure variance stays non-negative (full truncation scheme)
        const double v_positive = fmax(current_v, 0.0);
        const double sqrt_v = sqrt(v_positive);

        // Update variance (Heston model)
        // dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
        current_v += kappa * (theta - v_positive) * dt + xi * sqrt_v * W_v * sqrt_dt;

        // Update stock price (log-Euler scheme with jump)
        // dS/S = (r - k_drift - 0.5*v)*dt + sqrt(v)*dW_S + jump
        current_S *= exp((r - k_drift - 0.5 * v_positive) * dt +
                         sqrt_v * W_S * sqrt_dt + jump_component);

        // Store result using 64-bit indexing to avoid overflow.
        const size_t s_idx = static_cast<size_t>(step + 1) * static_cast<size_t>(num_paths) +
                             static_cast<size_t>(path_id);
        S_path[s_idx] = current_S;
    }
}

/**
 * @brief KERNEL 2: Asian put option payoff calculator
 *
 * Computes arithmetic average Asian put option payoffs:
 * Payoff = max(K - A, 0) where A = (1/n) * sum(S_t)
 *
 * @param S_path Input stock price paths [num_steps+1, num_paths]
 * @param payoffs Output payoff array [num_paths]
 * @param num_paths Number of paths
 * @param num_steps Number of time steps
 * @param K Strike price
 */
__global__ void asian_put_payoff_kernel(
    const double* S_path,
    double* payoffs,
    const int num_paths,
    const int num_steps,
    const double K)
{
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    // Compute arithmetic average along the path
    double path_sum = 0.0;
    for (int step = 0; step <= num_steps; ++step) {
        const size_t s_idx = static_cast<size_t>(step) * static_cast<size_t>(num_paths) +
                             static_cast<size_t>(path_id);
        path_sum += S_path[s_idx];
    }
    const double average_price = path_sum / static_cast<double>(num_steps + 1);

    // Asian put payoff
    payoffs[path_id] = fmax(K - average_price, 0.0);
}

/**
 * @brief KERNEL 2b: Bates path simulation with in-kernel Asian put payoff
 *
 * Generates Bates paths and computes the arithmetic average payoff without
 * storing the full path in device memory.
 *
 * @param payoffs Output payoff array [num_paths]
 * @param num_paths Number of paths
 * @param num_steps Number of time steps
 * @param T Time to maturity
 * @param K Strike price
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
 * @param k_drift Drift adjustment for jump compensation
 * @param Z1_in Pre-generated standard normal samples for variance
 * @param Z2_in Pre-generated standard normal samples for stock
 * @param seed Random seed for Poisson jump generation
 */
__global__ void bates_asian_put_payoff_kernel(
    double* payoffs,
    const int num_paths, const int num_steps, const double T, const double K,
    const double S0, const double r, double v0,
    const double kappa, const double theta, const double xi, const double rho,
    const double lambda_j, const double mu_j, const double sigma_j,
    const double k_drift,
    const double* Z1_in, const double* Z2_in,
    const unsigned long long seed)
{
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const double dt = T / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);

    // Initialize per-thread RNG state for Poisson jump generation
    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;
    double sum_S = current_S;

    // Precompute correlation complement for Cholesky decomposition
    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        // Generate Poisson-distributed number of jumps
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);

        // Accumulate jump component (log-normal jumps)
        double jump_component = 0.0;
        if (num_jumps > 0) {
            for (unsigned int j = 0; j < num_jumps; ++j) {
                jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
            }
        }

        // Load pre-generated random numbers (column-major for coalescing).
        // Use 64-bit indexing to avoid overflow when num_steps * num_paths exceeds 2^31.
        const size_t idx = static_cast<size_t>(step) * static_cast<size_t>(num_paths) +
                           static_cast<size_t>(path_id);
        const double Z1 = Z1_in[idx];
        const double Z2 = Z2_in[idx];

        // Correlated Brownian motions via Cholesky decomposition
        const double W_v = Z1;
        const double W_S = rho * W_v + rho_compl * Z2;

        // Ensure variance stays non-negative (full truncation scheme)
        const double v_positive = fmax(current_v, 0.0);
        const double sqrt_v = sqrt(v_positive);

        // Update variance (Heston model)
        current_v += kappa * (theta - v_positive) * dt + xi * sqrt_v * W_v * sqrt_dt;

        // Update stock price (log-Euler scheme with jump)
        current_S *= exp((r - k_drift - 0.5 * v_positive) * dt +
                         sqrt_v * W_S * sqrt_dt + jump_component);

        sum_S += current_S;
    }

    // Arithmetic average Asian put payoff
    const double average_price = sum_S / static_cast<double>(num_steps + 1);
    payoffs[path_id] = fmax(K - average_price, 0.0);
}

/**
 * @brief KERNEL 2c: Bates path simulation with in-kernel Asian put payoff (float)
 *
 * Uses FP32 intermediates to reduce memory usage while returning a float payoff.
 */
__global__ void bates_asian_put_payoff_kernel_f32(
    float* payoffs,
    const int num_paths, const int num_steps, const double T, const double K,
    const double S0, const double r, const double v0,
    const double kappa, const double theta, const double xi, const double rho,
    const double lambda_j, const double mu_j, const double sigma_j,
    const double k_drift,
    const float* Z1_in, const float* Z2_in,
    const unsigned long long seed)
{
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    const float dt = static_cast<float>(T / static_cast<double>(num_steps));
    const float sqrt_dt = sqrtf(dt);

    // Initialize per-thread RNG state for Poisson jump generation
    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    float current_S = static_cast<float>(S0);
    float current_v = static_cast<float>(v0);
    float sum_S = current_S;

    const float rho_f = static_cast<float>(rho);
    const float rho_compl = sqrtf(1.0f - rho_f * rho_f);

    const float k_drift_f = static_cast<float>(k_drift);
    const float r_f = static_cast<float>(r);
    const float kappa_f = static_cast<float>(kappa);
    const float theta_f = static_cast<float>(theta);
    const float xi_f = static_cast<float>(xi);
    const float mu_j_f = static_cast<float>(mu_j);
    const float sigma_j_f = static_cast<float>(sigma_j);

    for (int step = 0; step < num_steps; ++step) {
        const unsigned int num_jumps = curand_poisson(&rng_state,
                                                      static_cast<double>(lambda_j) * dt);

        float jump_component = 0.0f;
        if (num_jumps > 0) {
            for (unsigned int j = 0; j < num_jumps; ++j) {
                jump_component += mu_j_f + curand_normal(&rng_state) * sigma_j_f;
            }
        }

        const size_t idx = static_cast<size_t>(step) * static_cast<size_t>(num_paths) +
                           static_cast<size_t>(path_id);
        const float Z1 = Z1_in[idx];
        const float Z2 = Z2_in[idx];

        const float W_v = Z1;
        const float W_S = rho_f * W_v + rho_compl * Z2;

        const float v_positive = fmaxf(current_v, 0.0f);
        const float sqrt_v = sqrtf(v_positive);

        current_v += kappa_f * (theta_f - v_positive) * dt + xi_f * sqrt_v * W_v * sqrt_dt;

        current_S *= expf((r_f - k_drift_f - 0.5f * v_positive) * dt +
                          sqrt_v * W_S * sqrt_dt + jump_component);

        sum_S += current_S;
    }

    const float average_price = sum_S / static_cast<float>(num_steps + 1);
    payoffs[path_id] = fmaxf(static_cast<float>(K) - average_price, 0.0f);
}

/**
 * @brief KERNEL 3: Parallel reduction for computing mean
 *
 * Uses shared memory for fast block-level reduction, then atomic add
 * for global accumulation. Uses float for atomic compatibility.
 *
 * Performance notes:
 * - Tree-based reduction in shared memory minimizes bank conflicts
 * - Atomic operations only at block level, not thread level
 * - Float precision acceptable for sum of many values
 *
 * @param payoffs_double Input payoffs in double precision
 * @param final_result Output sum (atomic accumulation)
 * @param num_paths Number of values to reduce
 */
__global__ void reduce_average_kernel_float(
    const double* payoffs_double,
    float* final_result,
    const int num_paths)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to cover all paths, not just the first block range.
    double thread_sum = 0.0;
    for (unsigned int idx = i; idx < static_cast<unsigned int>(num_paths); idx += stride) {
        thread_sum += payoffs_double[idx];
    }

    // Load to shared memory (convert double to float)
    sdata[tid] = static_cast<float>(thread_sum);
    __syncthreads();

    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block leader writes to global memory
    if (tid == 0) {
        atomicAdd(final_result, sdata[0]);
    }
}

__global__ void reduce_average_kernel_float_input(
    const float* payoffs_float,
    float* final_result,
    const int num_paths)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float thread_sum = 0.0f;
    for (unsigned int idx = i; idx < static_cast<unsigned int>(num_paths); idx += stride) {
        thread_sum += payoffs_float[idx];
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
        atomicAdd(final_result, sdata[0]);
    }
}

// ============================================================================
// Main Orchestrator Function
// ============================================================================

static void validate_threads_per_block(int threads_per_block) {
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument("threads_per_block must be in [1, 1024], got " +
                                    std::to_string(threads_per_block));
    }
    if ((threads_per_block & (threads_per_block - 1)) != 0) {
        throw std::invalid_argument("threads_per_block must be a power of two, got " +
                                    std::to_string(threads_per_block));
    }
}

static void validate_num_streams(int num_streams) {
    if (num_streams <= 0 || num_streams > 32) {
        throw std::invalid_argument("num_streams must be in [1, 32], got " +
                                    std::to_string(num_streams));
    }
}

static int compute_max_paths_per_stream(size_t bytes_per_path, int num_streams) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const double safety = 0.85;
    size_t usable_bytes = static_cast<size_t>(static_cast<double>(free_bytes) * safety);
    if (bytes_per_path == 0 || num_streams <= 0) {
        return 1;
    }

    size_t max_paths_total = usable_bytes / bytes_per_path;
    if (max_paths_total < static_cast<size_t>(num_streams)) {
        throw std::runtime_error("Insufficient GPU memory for num_streams=" +
                                 std::to_string(num_streams));
    }
    size_t max_paths = max_paths_total / static_cast<size_t>(num_streams);
    if (max_paths < 1) {
        max_paths = 1;
    }
    if (max_paths > static_cast<size_t>(std::numeric_limits<int>::max())) {
        max_paths = static_cast<size_t>(std::numeric_limits<int>::max());
    }

    return static_cast<int>(max_paths);
}

static double price_bates_full_sequence_impl_double(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block,
    int num_streams
) {
    validate_bates_parameters(num_paths, num_steps, T, K, S0, r, v0,
                              kappa, theta, xi, rho, lambda_j, mu_j, sigma_j);
    validate_threads_per_block(threads_per_block);
    validate_num_streams(num_streams);

    // Compute jump-compensated drift: k = lambda * (E[J] - 1) where J = exp(mu + sigma^2/2)
    const double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);
    const unsigned long long seed_base = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    const size_t bytes_per_path = static_cast<size_t>(num_steps) * 2 * sizeof(double) + sizeof(double);
    int max_paths_per_stream = compute_max_paths_per_stream(bytes_per_path, num_streams);
    if (max_paths_per_stream > num_paths) {
        max_paths_per_stream = num_paths;
    }

    std::vector<CudaStream> streams;
    std::vector<CurandGenerator> generators;
    std::vector<CudaMemory<double>> d_Z1;
    std::vector<CudaMemory<double>> d_Z2;
    std::vector<CudaMemory<double>> d_payoffs;
    std::vector<CudaMemory<float>> d_sums;
    std::vector<int> chunk_sizes(num_streams, 0);

    streams.reserve(num_streams);
    generators.reserve(num_streams);
    d_Z1.reserve(num_streams);
    d_Z2.reserve(num_streams);
    d_payoffs.reserve(num_streams);
    d_sums.reserve(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        streams.emplace_back();
        generators.emplace_back();
        generators.back().setSeed(seed_base ^ static_cast<unsigned long long>(num_paths + i));
        generators.back().setStream(streams.back().get());
        d_Z1.emplace_back(static_cast<size_t>(num_steps) * max_paths_per_stream);
        d_Z2.emplace_back(static_cast<size_t>(num_steps) * max_paths_per_stream);
        d_payoffs.emplace_back(static_cast<size_t>(max_paths_per_stream));
        d_sums.emplace_back(1);
    }

    PinnedHostMemory<float> h_sums(static_cast<size_t>(num_streams));
    float* h_sums_ptr = h_sums.get();

    int remaining_paths = num_paths;
    long long batch_index = 0;
    double payoff_sum = 0.0;

    while (remaining_paths > 0) {
        int batch_paths = 0;
        int active_streams = 0;

        for (int s = 0; s < num_streams; ++s) {
            const int chunk_paths = std::min(remaining_paths - batch_paths, max_paths_per_stream);
            if (chunk_paths <= 0) {
                break;
            }
            chunk_sizes[s] = chunk_paths;
            batch_paths += chunk_paths;
            active_streams = s + 1;

            d_sums[s].memsetAsync(0, streams[s].get());
            generators[s].generateNormalDouble(
                d_Z1[s].get(), static_cast<size_t>(num_steps) * chunk_paths, 0.0, 1.0);
            generators[s].generateNormalDouble(
                d_Z2[s].get(), static_cast<size_t>(num_steps) * chunk_paths, 0.0, 1.0);

            const int blocks_per_grid = (chunk_paths + threads_per_block - 1) / threads_per_block;
            const unsigned long long seed = seed_base +
                                            static_cast<unsigned long long>(batch_index * num_streams + s);

            bates_asian_put_payoff_kernel<<<blocks_per_grid, threads_per_block, 0, streams[s].get()>>>(
                d_payoffs[s].get(), chunk_paths, num_steps, T, K,
                S0, r, v0, kappa, theta, xi, rho,
                lambda_j, mu_j, sigma_j, k_drift,
                d_Z1[s].get(), d_Z2[s].get(), seed
            );
            CUDA_CHECK_KERNEL();

            const int reduction_blocks = (blocks_per_grid < 128) ? blocks_per_grid : 128;
            reduce_average_kernel_float<<<reduction_blocks, threads_per_block,
                                          threads_per_block * sizeof(float), streams[s].get()>>>(
                d_payoffs[s].get(), d_sums[s].get(), chunk_paths
            );
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpyAsync(&h_sums_ptr[s], d_sums[s].get(),
                                       sizeof(float), cudaMemcpyDeviceToHost, streams[s].get()));
        }

        for (int s = 0; s < active_streams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s].get()));
            payoff_sum += static_cast<double>(h_sums_ptr[s]);
        }

        remaining_paths -= batch_paths;
        ++batch_index;
    }

    return (payoff_sum / static_cast<double>(num_paths)) * exp(-r * T);
}

static double price_bates_full_sequence_impl_float(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block,
    int num_streams
) {
    validate_bates_parameters(num_paths, num_steps, T, K, S0, r, v0,
                              kappa, theta, xi, rho, lambda_j, mu_j, sigma_j);
    validate_threads_per_block(threads_per_block);
    validate_num_streams(num_streams);

    const double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);
    const unsigned long long seed_base = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    const size_t bytes_per_path = static_cast<size_t>(num_steps) * 2 * sizeof(float) + sizeof(float);
    int max_paths_per_stream = compute_max_paths_per_stream(bytes_per_path, num_streams);
    if (max_paths_per_stream > num_paths) {
        max_paths_per_stream = num_paths;
    }

    std::vector<CudaStream> streams;
    std::vector<CurandGenerator> generators;
    std::vector<CudaMemory<float>> d_Z1;
    std::vector<CudaMemory<float>> d_Z2;
    std::vector<CudaMemory<float>> d_payoffs;
    std::vector<CudaMemory<float>> d_sums;

    streams.reserve(num_streams);
    generators.reserve(num_streams);
    d_Z1.reserve(num_streams);
    d_Z2.reserve(num_streams);
    d_payoffs.reserve(num_streams);
    d_sums.reserve(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        streams.emplace_back();
        generators.emplace_back();
        generators.back().setSeed(seed_base ^ static_cast<unsigned long long>(num_paths + 1315423911U * i));
        generators.back().setStream(streams.back().get());
        d_Z1.emplace_back(static_cast<size_t>(num_steps) * max_paths_per_stream);
        d_Z2.emplace_back(static_cast<size_t>(num_steps) * max_paths_per_stream);
        d_payoffs.emplace_back(static_cast<size_t>(max_paths_per_stream));
        d_sums.emplace_back(1);
    }

    PinnedHostMemory<float> h_sums(static_cast<size_t>(num_streams));
    float* h_sums_ptr = h_sums.get();

    int remaining_paths = num_paths;
    long long batch_index = 0;
    double payoff_sum = 0.0;

    while (remaining_paths > 0) {
        int batch_paths = 0;
        int active_streams = 0;

        for (int s = 0; s < num_streams; ++s) {
            const int chunk_paths = std::min(remaining_paths - batch_paths, max_paths_per_stream);
            if (chunk_paths <= 0) {
                break;
            }
            batch_paths += chunk_paths;
            active_streams = s + 1;

            d_sums[s].memsetAsync(0, streams[s].get());
            generators[s].generateNormalFloat(
                d_Z1[s].get(), static_cast<size_t>(num_steps) * chunk_paths, 0.0f, 1.0f);
            generators[s].generateNormalFloat(
                d_Z2[s].get(), static_cast<size_t>(num_steps) * chunk_paths, 0.0f, 1.0f);

            const int blocks_per_grid = (chunk_paths + threads_per_block - 1) / threads_per_block;
            const unsigned long long seed = seed_base +
                                            static_cast<unsigned long long>(batch_index * num_streams + s);

            bates_asian_put_payoff_kernel_f32<<<blocks_per_grid, threads_per_block, 0, streams[s].get()>>>(
                d_payoffs[s].get(), chunk_paths, num_steps, T, K,
                S0, r, v0, kappa, theta, xi, rho,
                lambda_j, mu_j, sigma_j, k_drift,
                d_Z1[s].get(), d_Z2[s].get(), seed
            );
            CUDA_CHECK_KERNEL();

            const int reduction_blocks = (blocks_per_grid < 128) ? blocks_per_grid : 128;
            reduce_average_kernel_float_input<<<reduction_blocks, threads_per_block,
                                               threads_per_block * sizeof(float), streams[s].get()>>>(
                d_payoffs[s].get(), d_sums[s].get(), chunk_paths
            );
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaMemcpyAsync(&h_sums_ptr[s], d_sums[s].get(),
                                       sizeof(float), cudaMemcpyDeviceToHost, streams[s].get()));
        }

        for (int s = 0; s < active_streams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s].get()));
            payoff_sum += static_cast<double>(h_sums_ptr[s]);
        }

        remaining_paths -= batch_paths;
        ++batch_index;
    }

    return (payoff_sum / static_cast<double>(num_paths)) * exp(-r * T);
}

/**
 * @brief Complete Bates model Asian put pricing pipeline
 *
 * Orchestrates the full GPU computation with optional chunking:
 * 1. Queries free GPU memory to size chunks
 * 2. Allocates per-chunk GPU memory (with RAII for automatic cleanup)
 * 3. Generates random numbers using cuRAND
 * 4. Simulates paths and computes Asian put payoffs in-kernel
 * 5. Reduces to get per-chunk payoff sums
 * 6. Aggregates and discounts to present value
 *
 * @return Discounted option price
 * @throws std::invalid_argument if parameters are invalid
 * @throws std::runtime_error if CUDA operations fail
 */
extern "C" double price_bates_full_sequence(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    return price_bates_full_sequence_impl_double(
        num_paths, num_steps, T, K,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j,
        256,
        1
    );
}

extern "C" double price_bates_full_sequence_with_block_size(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block
) {
    return price_bates_full_sequence_impl_double(
        num_paths, num_steps, T, K,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j,
        threads_per_block,
        1
    );
}

extern "C" double price_bates_full_sequence_with_config(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block, int num_streams
) {
    return price_bates_full_sequence_impl_double(
        num_paths, num_steps, T, K,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j,
        threads_per_block,
        num_streams
    );
}

extern "C" double price_bates_full_sequence_mixed_precision(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    return price_bates_full_sequence_impl_float(
        num_paths, num_steps, T, K,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j,
        256,
        1
    );
}

extern "C" double price_bates_full_sequence_mixed_precision_with_config(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    int threads_per_block, int num_streams
) {
    return price_bates_full_sequence_impl_float(
        num_paths, num_steps, T, K,
        S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j,
        threads_per_block,
        num_streams
    );
}
