#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cmath>
#include <chrono>

// KERNEL 1: Path Generator (in double precision)
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

    const double dt = T / (double)num_steps;
    const double sqrt_dt = sqrt(dt);
    curandState_t rng_state;
    curand_init(seed, path_id, 0, &rng_state);

    double current_S = S0;
    double current_v = v0;
    S_path[0 * num_paths + path_id] = current_S;
    const double rho_compl = sqrt(1.0 - rho * rho);

    for (int step = 0; step < num_steps; ++step) {
        const unsigned int num_jumps = curand_poisson(&rng_state, lambda_j * dt);
        double jump_component = 0.0;
        if (num_jumps > 0) {
            for (unsigned int j = 0; j < num_jumps; ++j) {
                jump_component += mu_j + curand_normal_double(&rng_state) * sigma_j;
            }
        }

        const double Z1 = Z1_in[step * num_paths + path_id];
        const double Z2 = Z2_in[step * num_paths + path_id];
        const double W_v = Z1;
        const double W_S = rho * W_v + rho_compl * Z2;
        const double v_positive = fmax(current_v, 0.0);
        const double sqrt_v = sqrt(v_positive);

        current_v += kappa * (theta - v_positive) * dt + xi * sqrt_v * W_v * sqrt_dt;
        current_S *= exp((r - k_drift - 0.5 * v_positive) * dt + sqrt_v * W_S * sqrt_dt + jump_component);
        S_path[(step + 1) * num_paths + path_id] = current_S;
    }
}

// KERNEL 2: Payoff Calculator (in double precision)
__global__ void asian_put_payoff_kernel(
    const double* S_path,
    double* payoffs,
    const int num_paths,
    const int num_steps,
    const double K)
{
    const int path_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_id >= num_paths) return;

    double path_sum = 0.0;
    for (int step = 0; step <= num_steps; ++step) {
        path_sum += S_path[step * num_paths + path_id];
    }
    double average_price = path_sum / (double)(num_steps + 1);
    payoffs[path_id] = fmax(K - average_price, 0.0);
}

// KERNEL 3: Parallel Reducer (in float precision to support atomicAdd)
__global__ void reduce_average_kernel_float(
    const double* payoffs_double,
    float* final_result,
    const int num_paths)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_paths) {
        sdata[tid] = (float)payoffs_double[i];
    } else {
        sdata[tid] = 0.0f;
    }
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

// The Main C-style Orchestrator Function
extern "C" double price_bates_full_sequence(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    double *d_S_path, *d_Z1, *d_Z2, *d_payoffs;
    float *d_final_price_sum_float;

    cudaMalloc(&d_S_path, (size_t)(num_steps + 1) * num_paths * sizeof(double));
    cudaMalloc(&d_Z1, (size_t)num_steps * num_paths * sizeof(double));
    cudaMalloc(&d_Z2, (size_t)num_steps * num_paths * sizeof(double));
    cudaMalloc(&d_payoffs, (size_t)num_paths * sizeof(double));
    cudaMalloc(&d_final_price_sum_float, sizeof(float));
    cudaMemset(d_final_price_sum_float, 0, sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, std::chrono::high_resolution_clock::now().time_since_epoch().count());
    curandGenerateNormalDouble(gen, d_Z1, num_steps * num_paths, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_Z2, num_steps * num_paths, 0.0, 1.0);

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;
    double k_drift = lambda_j * (exp(mu_j + 0.5 * sigma_j * sigma_j) - 1);
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    bates_path_generator_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path, num_paths, num_steps, T, S0, r, v0, kappa, theta, xi, rho,
        lambda_j, mu_j, sigma_j, k_drift, d_Z1, d_Z2, seed
    );

    asian_put_payoff_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_S_path, d_payoffs, num_paths, num_steps, K
    );

    const int reduction_blocks = 128;
    reduce_average_kernel_float<<<reduction_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        d_payoffs, d_final_price_sum_float, num_paths
    );

    float h_final_price_sum_float;
    cudaMemcpy(&h_final_price_sum_float, d_final_price_sum_float, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double final_price = ((double)h_final_price_sum_float / (double)num_paths) * exp(-r * T);

    curandDestroyGenerator(gen);
    cudaFree(d_S_path);
    cudaFree(d_Z1);
    cudaFree(d_Z2);
    cudaFree(d_payoffs);
    cudaFree(d_final_price_sum_float);

    return final_price;
}
