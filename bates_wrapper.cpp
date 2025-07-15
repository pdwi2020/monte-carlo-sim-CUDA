#include <pybind11/pybind11.h>
#include "cuda_runtime.h"

namespace py = pybind11;

// Declare the C-style launcher function from our .cu file.
extern "C" double price_bates_full_sequence(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
);

// Our Python wrapper just calls the C function and returns the result.
double price_wrapper(
    int num_paths, int num_steps, double T, double K,
    double S0, double r, double v0,
    double kappa, double theta, double xi, double rho,
    double lambda_j, double mu_j, double sigma_j
) {
    return price_bates_full_sequence(
        num_paths, num_steps, T, K, S0, r, v0,
        kappa, theta, xi, rho, lambda_j, mu_j, sigma_j
    );
}

// Create the Python Module, exposing the function as "price"
PYBIND11_MODULE(bates_kernel_cpp, m) {
    m.doc() = "Fully integrated C++/CUDA Bates Pricer";
    m.def("price", &price_wrapper, "Calculates the full Bates price on the GPU",
        py::arg("num_paths"), py::arg("num_steps"), py::arg("T"), py::arg("K"),
        py::arg("S0"), py::arg("r"), py::arg("v0"),
        py::arg("kappa"), py::arg("theta"), py::arg("xi"), py::arg("rho"),
        py::arg("lambda_j"), py::arg("mu_j"), py::arg("sigma_j")
    );
}
