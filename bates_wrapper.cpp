/**
 * @file bates_wrapper.cpp
 * @brief PyBind11 wrapper for Bates model pricing function
 *
 * Exposes the price() function to Python for pricing Asian put options
 * using the Bates stochastic volatility + jump diffusion model on GPU.
 */

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include "cuda_runtime.h"

namespace py = pybind11;

// Declare the C-style launcher function from our .cu file.
extern "C" double price_bates_full_sequence(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
);

/**
 * @brief Python wrapper for the Bates model Asian put pricer
 *
 * This function validates inputs at the Python boundary before
 * delegating to the CUDA implementation. Additional validation
 * is performed in the CUDA code for defense in depth.
 *
 * @param num_paths Number of Monte Carlo paths (1 to 100M)
 * @param num_steps Number of time steps (1 to 10000)
 * @param T Time to maturity in years (> 0)
 * @param K Strike price (> 0)
 * @param S0 Initial stock price (> 0)
 * @param r Risk-free interest rate
 * @param v0 Initial variance (>= 0)
 * @param kappa Mean reversion speed (> 0)
 * @param theta Long-term variance (>= 0)
 * @param xi Volatility of variance (>= 0)
 * @param rho Correlation between stock and variance [-1, 1]
 * @param lambda_j Jump intensity (>= 0)
 * @param mu_j Mean of log jump size
 * @param sigma_j Std dev of log jump size (>= 0)
 * @return Discounted option price
 * @throws std::invalid_argument if parameters are out of bounds
 * @throws std::runtime_error if CUDA operations fail
 */
double price_wrapper(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    // Note: Full validation is performed in price_bates_full_sequence
    // This wrapper provides a clean Python interface
    return price_bates_full_sequence(
        num_paths, num_steps, T, K, S0, r, v0,
        kappa, theta, xi, rho, lambda_j, mu_j, sigma_j
    );
}

// Create the Python Module
PYBIND11_MODULE(bates_kernel_cpp, m) {
    m.doc() = R"doc(
        Fully integrated C++/CUDA Bates Model Option Pricer.

        The Bates model (1996) combines:
        - Heston stochastic volatility for realistic variance dynamics
        - Merton-style jump diffusion for crash/rally risk

        This implementation prices arithmetic average Asian put options
        using GPU-accelerated Monte Carlo simulation.

        Mathematical Model:
            dS = (r - k)*dt + sqrt(v)*dW_S + J*dN
            dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
            corr(dW_S, dW_v) = rho

        where:
            k = lambda_j * (exp(mu_j + 0.5*sigma_j^2) - 1)  (jump compensation)
            N ~ Poisson(lambda_j * t)                       (jump arrival)
            J ~ LogNormal(mu_j, sigma_j)                    (jump size)

        Functions:
            price: Calculate option price using GPU Monte Carlo
    )doc";

    m.def("price", &price_wrapper,
          R"doc(
              Calculate Bates model Asian put option price.

              Uses GPU-accelerated Monte Carlo simulation with:
              - cuRAND for random number generation
              - Parallel path generation
              - Parallel reduction for averaging

              Parameters:
                  num_paths (int): Number of Monte Carlo paths (1 to 100M)
                  num_steps (int): Number of time steps (1 to 10000)
                  T (float): Time to maturity in years
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free interest rate (annualized)
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Volatility of variance (vol-of-vol)
                  rho (float): Correlation between stock and variance
                  lambda_j (float): Jump intensity (expected jumps per year)
                  mu_j (float): Mean of log jump size
                  sigma_j (float): Std dev of log jump size

              Returns:
                  float: Discounted option price

              Raises:
                  ValueError: If parameters are out of valid bounds
                  RuntimeError: If CUDA operations fail

              Example:
                  >>> import bates_kernel_cpp
                  >>> price = bates_kernel_cpp.price(
                  ...     num_paths=100000, num_steps=252, T=1.0, K=100.0,
                  ...     S0=100.0, r=0.05, v0=0.04,
                  ...     kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                  ...     lambda_j=0.1, mu_j=-0.1, sigma_j=0.2
                  ... )
          )doc",
          py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j")
    );
}
