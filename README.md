# monte-carlo-sim-CUDA
Of course. This README is the culmination of all your hard work. It's designed to be the main document in your GitHub repository, telling the complete story of your project in a clear, compelling, and professional manner. It's written for an audience of professors on an admissions committee.

---

# Advanced Monte Carlo Methods for Exotic Option Pricing with CUDA Acceleration

**Author:** [Your Name]  
**Date:** [Current Date]  
**Contact:** [Your Email / LinkedIn / GitHub Profile]

## 1. Project Overview

This repository contains the source code and results for a master's-level research project in computational finance and applied mathematics. The primary goal was to develop a high-performance, extensible framework for pricing complex exotic options under advanced stochastic models.

The project systematically builds from foundational concepts to state-of-the-art numerical methods, demonstrating a comprehensive skillset across **financial modeling, numerical analysis, high-performance computing (HPC) with CUDA, and advanced software engineering in Python and C++.**

The key achievements of this project are:
*   **A flexible, object-oriented pricing framework** capable of handling multiple models (GBM, Heston, Bates), options (European, Asian), and pricing backends (CPU, CuPy-GPU, Custom C++ CUDA).
*   Implementation and validation of the **Heston Stochastic Volatility** and **Bates (Stochastic Volatility + Jumps)** models.
*   A rigorous analysis of advanced Monte Carlo techniques, including **Quasi-Monte Carlo (QMC)** and **Multilevel Monte Carlo (MLMC)**, showcasing a **~22x speedup** with MLMC.
*   Application of the framework to a practical risk management problem by calculating and visualizing **option Greeks** (the Vega surface).
*   Development of a **custom CUDA C++ kernel** to explore the performance trade-offs between high-level GPU libraries and low-level, hand-optimized code, yielding critical insights into modern HPC architecture.

---

## 2. Technical Stack

*   **Languages:** Python, C++, CUDA C++
*   **Libraries:** CuPy, NumPy, Matplotlib, SciPy, Pybind11
*   **HPC Platform:** NVIDIA V100 GPU on Paperspace Gradient
*   **Development Environment:** Custom Linux environment with CUDA Toolkit 12.x, `g++`, and `build-essential`.

---

## 3. Project Journey & Key Results

The project was developed in a series of logical stages, with each stage building upon the last to demonstrate a new concept or technique.

### Part 1: Foundational Pricing Engine & GPU Acceleration

A baseline Monte Carlo pricer for a European option under Geometric Brownian Motion (GBM) was developed. This served to validate the core simulation logic against the analytical Black-Scholes formula and establish a performance benchmark.

*   **Result:** The CuPy-based GPU implementation demonstrated a **~76x speedup** over the NumPy-based CPU implementation for 5 million paths, confirming the immense value of GPU acceleration for financial simulations.

### Part 2: Advanced Modeling - Heston & Asian Options

The framework was extended to price a path-dependent Asian option under the more realistic Heston Stochastic Volatility model.

*   **Result:** The GPU pricer successfully handled the more complex, coupled SDEs of the Heston model, achieving a **14.5x speedup** over the CPU. This demonstrated the framework's extensibility.

| Model         | Option  | Backend | Paths     | Time (s) | Speedup |
|---------------|---------|---------|-----------|----------|---------|
| Heston        | Asian   | CPU     | 1,000,000 | 21.37    | 1x      |
| **Heston**    | **Asian**   | **GPU**     | **1,000,000** | **1.47**     | **14.52x**  |

### Part 3: Advanced Numerical Methods - QMC & MLMC

State-of-the-art variance reduction techniques were implemented to improve simulation efficiency.

#### Convergence Analysis (MC vs. QMC)
A comparison between standard Monte Carlo, antithetic variates, and Quasi-Monte Carlo (using a Sobol sequence generator) was performed.

*   **Result:** The convergence plot shows that for this high-dimensional problem (d=100), QMC, while noisy, generally trended towards a faster convergence rate ($O(N^{-1})$) than standard MC ($O(N^{-0.5})$) for a large number of paths.



#### Multilevel Monte Carlo (MLMC)
A full MLMC pricer was implemented for the Asian option under the Heston model. The implementation journey revealed and resolved several critical numerical challenges, including floating-point instability and discretization bias at coarse levels, which were fixed by enforcing `float64` precision and introducing a `base_steps` parameter.

*   **Result:** The final, correct MLMC implementation demonstrated its theoretical power, achieving the target accuracy with a **21.56x speedup** over a highly optimized standard Monte Carlo method. The diagnostic plots show the classic MLMC behavior: variance decays rapidly across levels, allowing the algorithm to concentrate computational effort on cheaper, coarser simulations.

| Method                    | Target Error | Time (s) | Speedup  |
|---------------------------|--------------|----------|----------|
| Standard MC (Antithetic)  | 0.01         | 0.0437   | 1x       |
| **Multilevel MC (MLMC)**  | **0.01**     | **0.0020**   | **21.56x** |



### Part 4: Financial Application - Risk Analysis of Greeks

The pricer was extended to calculate option sensitivities (Greeks) using finite differences. The Vega surface was calculated and visualized, connecting the computational tool to a practical risk management application.

*   **Result:** The generated Vega surface correctly displays the expected financial behavior: Vega is highest for at-the-money, long-dated options and decays towards the wings and for shorter maturities. This demonstrates a complete understanding of the financial product's risk profile.



### Part 5: The Final Frontier - Bates Model & Custom CUDA C++ Kernel

To push the boundaries of both financial modeling and HPC, the project's final phase involved two major extensions.

#### Bates Model Implementation
The framework was extended to handle the Bates model (Heston + Jumps) to capture crash risk.

*   **Result:** The model was successfully validated. When pricing an out-of-the-money put, the Bates model yielded a price **~2.5x higher** than the Heston model (`$2.56` vs. `$1.02`), correctly quantifying the premium for "crash insurance." However, this realism came at a significant performance cost.

| Model                   | Option  | Price    | Time (s) |
|-------------------------|---------|----------|----------|
| Heston                  | OTM Put | 1.02     | 0.25     |
| **Bates (Crash Jumps)** | **OTM Put** | **2.56**     | **1.95**     |

#### Custom CUDA C++ Kernel Benchmark
To address the performance cost of the Bates model, a hand-optimized CUDA C++ kernel was developed and benchmarked against the high-level CuPy implementation.

*   **Final Result & Key Insight:** The custom kernel was benchmarked at `8.30s`, surprisingly **~8x slower** than the CuPy implementation at `1.09s`. A deep analysis revealed this was due to a **data transfer bottleneck**: the custom implementation required multiple GPU-CPU-GPU data transfers, whereas the CuPy version performed the entire iterative calculation without ever leaving the GPU's high-speed memory.

| Backend for Bates Model | Paths     | Time (s) | Speedup vs. CuPy |
|-------------------------|-----------|----------|------------------|
| **CuPy (High-Level GPU)** | **2,000,000** | **1.09**     | **1x**           |
| Custom C++ (Low-Level GPU)  | 2,000,000 | 8.30     | 0.13x            |

This final, counter-intuitive result provides the most advanced lesson of the project: **a naive low-level implementation is not inherently superior to a well-designed, high-level library that respects data locality.** It demonstrates a mature understanding of HPC architecture, where minimizing data movement is often more critical than raw computational optimization.

## 4. How to Run

1.  **Environment Setup:** This project was developed in a custom Linux environment on Paperspace. It requires Python 3.10+, the CUDA Toolkit, and a C++ compiler (`g++`). Install Python dependencies with `pip install -r requirements.txt`.
2.  **Build Custom Kernel:** To run the final benchmark, the custom C++ kernel must be compiled:
    ```bash
    chmod +x build.sh
    ./build.sh
    ```
3.  **Run the Notebook:** All code and analysis are contained within the main Jupyter Notebook: `Financial_Engineering_Project.ipynb`. The cells are numbered and can be run sequentially to reproduce all results.

## 5. Conclusion

This project successfully delivered a powerful, GPU-accelerated framework for modern quantitative finance. It spans the full stack from advanced mathematical models and numerical methods to low-level performance engineering. The journey through implementing and debugging methods like MLMC and the custom CUDA kernel provided deep insights into the practical challenges and trade-offs inherent in computational science. The final framework is not just a pricing engine but a robust tool for research, analysis, and risk management.
