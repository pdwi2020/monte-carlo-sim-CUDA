/**
 * @file bates_wrapper_extended.cpp
 * @brief Extended PyBind11 wrapper for Bates model with QE scheme, barriers, and Greeks
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <map>
#include "cuda_runtime.h"

namespace py = pybind11;

// Declare C-style launcher functions from extended CUDA file
extern "C" {
    void price_bates_qe(
        int num_paths, int num_steps, double T, double K,
        double S0, double r, double v0,
        double kappa, double theta, double xi, double rho,
        double lambda_j, double mu_j, double sigma_j,
        int payoff_type, double barrier, double rebate,
        double* price, double* std_error
    );

    void price_asian_cv(
        int num_paths, int num_steps, double T, double K,
        double S0, double r, double v0,
        double kappa, double theta, double xi, double rho,
        double lambda_j, double mu_j, double sigma_j,
        double geom_analytical, bool is_call,
        double* price, double* std_error, double* cv_beta
    );

    void calculate_delta(
        int num_paths, int num_steps, double T, double K,
        double S0, double r, double v0,
        double kappa, double theta, double xi, double rho,
        double lambda_j, double mu_j, double sigma_j,
        int payoff_type, double barrier, double rebate,
        double bump_pct, double* delta
    );

    void calculate_gamma(
        int num_paths, int num_steps, double T, double K,
        double S0, double r, double v0,
        double kappa, double theta, double xi, double rho,
        double lambda_j, double mu_j, double sigma_j,
        int payoff_type, double barrier, double rebate,
        double bump_pct, double* gamma
    );

    void calculate_vega(
        int num_paths, int num_steps, double T, double K,
        double S0, double r, double v0,
        double kappa, double theta, double xi, double rho,
        double lambda_j, double mu_j, double sigma_j,
        int payoff_type, double barrier, double rebate,
        double bump_pct, double* vega
    );
}

// Payoff type enum matching CUDA kernel
enum class PayoffType : int {
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

// Result struct for pricing
struct PricingResult {
    double price;
    double std_error;
    double cv_beta;  // Control variate beta (0 if not used)

    PricingResult(double p, double se, double beta = 0.0)
        : price(p), std_error(se), cv_beta(beta) {}
};

// Result struct for Greeks
struct GreeksResult {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;

    GreeksResult() : delta(0), gamma(0), vega(0), theta(0), rho(0) {}
};

/**
 * @brief Validate Bates model parameters
 */
void validate_params(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    if (num_paths <= 0) throw std::invalid_argument("num_paths must be positive");
    if (num_paths > 100000000) throw std::invalid_argument("num_paths exceeds 100M limit");
    if (num_steps <= 0) throw std::invalid_argument("num_steps must be positive");
    if (num_steps > 10000) throw std::invalid_argument("num_steps exceeds 10000 limit");
    if (T <= 0) throw std::invalid_argument("T must be positive");
    if (K <= 0) throw std::invalid_argument("K must be positive");
    if (S0 <= 0) throw std::invalid_argument("S0 must be positive");
    if (v0 < 0) throw std::invalid_argument("v0 must be non-negative");
    if (kappa <= 0) throw std::invalid_argument("kappa must be positive");
    if (theta < 0) throw std::invalid_argument("theta must be non-negative");
    if (xi < 0) throw std::invalid_argument("xi must be non-negative");
    if (rho < -1 || rho > 1) throw std::invalid_argument("rho must be in [-1, 1]");
    if (lambda_j < 0) throw std::invalid_argument("lambda_j must be non-negative");
    if (sigma_j < 0) throw std::invalid_argument("sigma_j must be non-negative");
}

/**
 * @brief Price option with QE scheme
 */
PricingResult price_qe(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    const std::string& payoff_type_str,
    double barrier = 0.0,
    double rebate = 0.0
) {
    validate_params(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j);

    // Map string to enum
    static const std::map<std::string, int> payoff_map = {
        {"asian_put", 0},
        {"asian_call", 1},
        {"asian_geom_put", 2},
        {"asian_geom_call", 3},
        {"european_put", 4},
        {"european_call", 5},
        {"barrier_up_out_put", 6},
        {"barrier_up_in_put", 7},
        {"barrier_down_out_put", 8},
        {"barrier_down_in_put", 9},
        {"barrier_up_out_call", 10},
        {"barrier_up_in_call", 11},
        {"barrier_down_out_call", 12},
        {"barrier_down_in_call", 13},
        {"lookback_fixed_put", 14},
        {"lookback_fixed_call", 15}
    };

    auto it = payoff_map.find(payoff_type_str);
    if (it == payoff_map.end()) {
        throw std::invalid_argument("Unknown payoff type: " + payoff_type_str);
    }
    int payoff_type = it->second;

    double price, std_error;
    price_bates_qe(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price, &std_error);

    return PricingResult(price, std_error);
}

/**
 * @brief Price Asian option with control variates
 */
PricingResult price_asian_with_cv(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    double geom_analytical,
    bool is_call = false
) {
    validate_params(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j);

    double price, std_error, cv_beta;
    price_asian_cv(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   geom_analytical, is_call, &price, &std_error, &cv_beta);

    return PricingResult(price, std_error, cv_beta);
}

/**
 * @brief Calculate all Greeks
 */
GreeksResult compute_greeks(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j,
    const std::string& payoff_type_str,
    double barrier = 0.0,
    double rebate = 0.0,
    double bump_pct = 0.01
) {
    validate_params(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j);

    static const std::map<std::string, int> payoff_map = {
        {"asian_put", 0}, {"asian_call", 1},
        {"asian_geom_put", 2}, {"asian_geom_call", 3},
        {"european_put", 4}, {"european_call", 5},
        {"barrier_up_out_put", 6}, {"barrier_up_in_put", 7},
        {"barrier_down_out_put", 8}, {"barrier_down_in_put", 9},
        {"barrier_up_out_call", 10}, {"barrier_up_in_call", 11},
        {"barrier_down_out_call", 12}, {"barrier_down_in_call", 13},
        {"lookback_fixed_put", 14}, {"lookback_fixed_call", 15}
    };

    auto it = payoff_map.find(payoff_type_str);
    if (it == payoff_map.end()) {
        throw std::invalid_argument("Unknown payoff type: " + payoff_type_str);
    }
    int payoff_type = it->second;

    GreeksResult result;

    // Calculate Delta
    calculate_delta(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, bump_pct, &result.delta);

    // Calculate Gamma
    calculate_gamma(num_paths, num_steps, T, K, S0, r, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, bump_pct, &result.gamma);

    // Calculate Vega
    calculate_vega(num_paths, num_steps, T, K, S0, r, v0,
                  kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                  payoff_type, barrier, rebate, bump_pct, &result.vega);

    // Calculate Theta (time decay)
    double dt = 1.0 / 365.0;  // 1 day
    if (T > dt) {
        double price_now, price_later, se;
        price_bates_qe(num_paths, num_steps, T, K, S0, r, v0,
                       kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                       payoff_type, barrier, rebate, &price_now, &se);
        price_bates_qe(num_paths, num_steps, T - dt, K, S0, r, v0,
                       kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                       payoff_type, barrier, rebate, &price_later, &se);
        result.theta = price_later - price_now;
    }

    // Calculate Rho (interest rate sensitivity)
    double dr = 0.0001;  // 1 bp
    double price_up_r, price_down_r, se;
    price_bates_qe(num_paths, num_steps, T, K, S0, r + dr, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_up_r, &se);
    price_bates_qe(num_paths, num_steps, T, K, S0, r - dr, v0,
                   kappa, theta, xi, rho, lambda_j, mu_j, sigma_j,
                   payoff_type, barrier, rebate, &price_down_r, &se);
    result.rho = (price_up_r - price_down_r) / (2 * dr) * 0.01;  // per 1%

    return result;
}

/**
 * @brief Check Feller condition: 2*kappa*theta > xi^2
 */
bool check_feller_condition(double kappa, double theta, double xi) {
    return 2 * kappa * theta > xi * xi;
}

// Create the Python Module
PYBIND11_MODULE(bates_extended, m) {
    m.doc() = R"doc(
        Extended Bates Model Option Pricer with QE Scheme.

        Features:
        - Quadratic-Exponential (QE) discretization for variance (Andersen, 2008)
        - Multiple payoff types (Asian, European, Barrier, Lookback)
        - Control variates for Asian options
        - Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
        - High-performance CUDA implementation

        Mathematical Model (Bates, 1996):
            dS = (r - k)*dt + sqrt(v)*dW_S + J*dN
            dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
            corr(dW_S, dW_v) = rho

        QE Scheme Benefits:
        - Eliminates negative variance
        - Matches first two moments exactly
        - Better accuracy with fewer time steps
    )doc";

    // Pricing Result class
    py::class_<PricingResult>(m, "PricingResult")
        .def(py::init<double, double, double>(),
             py::arg("price"), py::arg("std_error"), py::arg("cv_beta") = 0.0)
        .def_readonly("price", &PricingResult::price)
        .def_readonly("std_error", &PricingResult::std_error)
        .def_readonly("cv_beta", &PricingResult::cv_beta)
        .def("__repr__", [](const PricingResult& r) {
            return "PricingResult(price=" + std::to_string(r.price) +
                   ", std_error=" + std::to_string(r.std_error) +
                   ", cv_beta=" + std::to_string(r.cv_beta) + ")";
        });

    // Greeks Result class
    py::class_<GreeksResult>(m, "GreeksResult")
        .def(py::init<>())
        .def_readonly("delta", &GreeksResult::delta)
        .def_readonly("gamma", &GreeksResult::gamma)
        .def_readonly("vega", &GreeksResult::vega)
        .def_readonly("theta", &GreeksResult::theta)
        .def_readonly("rho", &GreeksResult::rho)
        .def("__repr__", [](const GreeksResult& g) {
            return "GreeksResult(delta=" + std::to_string(g.delta) +
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) +
                   ", theta=" + std::to_string(g.theta) +
                   ", rho=" + std::to_string(g.rho) + ")";
        });

    // Main pricing function
    m.def("price", &price_qe,
          R"doc(
              Price option using Bates model with QE scheme.

              Parameters:
                  num_paths (int): Number of Monte Carlo paths
                  num_steps (int): Number of time steps
                  T (float): Time to maturity
                  K (float): Strike price
                  S0 (float): Initial stock price
                  r (float): Risk-free rate
                  v0 (float): Initial variance
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Vol of vol
                  rho (float): Correlation
                  lambda_j (float): Jump intensity
                  mu_j (float): Mean log jump size
                  sigma_j (float): Std log jump size
                  payoff_type (str): Payoff type (e.g., "asian_put", "barrier_down_out_call")
                  barrier (float): Barrier level (for barrier options)
                  rebate (float): Rebate (for barrier options)

              Returns:
                  PricingResult: Price and standard error

              Payoff Types:
                  - asian_put, asian_call
                  - asian_geom_put, asian_geom_call
                  - european_put, european_call
                  - barrier_up_out_put, barrier_up_in_put
                  - barrier_down_out_put, barrier_down_in_put
                  - barrier_up_out_call, barrier_up_in_call
                  - barrier_down_out_call, barrier_down_in_call
                  - lookback_fixed_put, lookback_fixed_call
          )doc",
          py::arg("num_paths"), py::arg("num_steps"),
          py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("payoff_type"),
          py::arg("barrier") = 0.0,
          py::arg("rebate") = 0.0
    );

    // Asian with control variates
    m.def("price_asian_cv", &price_asian_with_cv,
          R"doc(
              Price Asian option with control variates.

              Uses geometric Asian (which has closed-form) as control for
              arithmetic Asian, significantly reducing variance.

              Parameters:
                  ... (same as price)
                  geom_analytical (float): Pre-computed analytical geometric Asian price
                  is_call (bool): True for call, False for put

              Returns:
                  PricingResult: Price, std error, and CV beta
          )doc",
          py::arg("num_paths"), py::arg("num_steps"),
          py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("geom_analytical"),
          py::arg("is_call") = false
    );

    // Greeks calculation
    m.def("greeks", &compute_greeks,
          R"doc(
              Calculate all Greeks for an option.

              Parameters:
                  ... (same as price)
                  bump_pct (float): Bump percentage for finite difference (default 1%)

              Returns:
                  GreeksResult: Delta, Gamma, Vega, Theta, Rho
          )doc",
          py::arg("num_paths"), py::arg("num_steps"),
          py::arg("T"), py::arg("K"),
          py::arg("S0"), py::arg("r"), py::arg("v0"),
          py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
          py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j"),
          py::arg("payoff_type"),
          py::arg("barrier") = 0.0,
          py::arg("rebate") = 0.0,
          py::arg("bump_pct") = 0.01
    );

    // Utility function
    m.def("check_feller", &check_feller_condition,
          R"doc(
              Check if Feller condition is satisfied.

              The Feller condition (2*kappa*theta > xi^2) ensures that
              variance stays positive with high probability.

              Parameters:
                  kappa (float): Mean reversion speed
                  theta (float): Long-term variance
                  xi (float): Vol of vol

              Returns:
                  bool: True if Feller condition is satisfied
          )doc",
          py::arg("kappa"), py::arg("theta"), py::arg("xi")
    );
}
