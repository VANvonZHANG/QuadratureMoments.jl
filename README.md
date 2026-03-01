# QBMM.jl

[![Build Status](https://github.com/yourusername/QBMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/yourusername/QBMM.jl/actions)
[![Coverage](https://codecov.io/gh/yourusername/QBMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/yourusername/QBMM.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Quadrature-Based Moment Methods for Mesoscale Modeling in Julia.**

`QBMM.jl` is a high-performance Julia library designed for mesoscale modeling of polydisperse particulate and multiphase systems. It provides a comprehensive suite of moment-inversion algorithms (QMOM, EQMOM, CQMOM, ECQMOM, DQMOM) optimized for industrial-grade CFD applications, emphasizing numerical robustness, zero-allocation execution, and a unified API.

---

## 1. Installation

`QBMM.jl` is currently available via GitHub. You can install it using the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/QBMM.jl.git")
```

---

## 2. Project Positioning
In the simulation of multiphase flows (e.g., sprays, aerosols, crystallization), the evolution of the Number Density Function (NDF) is governed by the **Generalized Population Balance Equation (GPBE)**. `QBMM.jl` provides the mathematical "engine" to close the moment-transport equations derived from the GPBE, allowing users to reconstruct distributions from their transportable moments.

## 3. Mathematical Background
This library implements the core algorithms described in **Marchisio & Fox (2013)**. The primary challenge in QBMM is the **Moment Inversion Problem**: finding a quadrature approximation (weights $w_\alpha$ and nodes $\xi_\alpha$) that matches a set of known moments $m_k = \int \xi^k n(\xi) d\xi$.

### 3.1 Univariate QMOM Algorithms
- **Wheeler Algorithm**: A robust and stable method that builds a sequence of polynomials orthogonal to the NDF. The nodes are the roots of the $N$-th order orthogonal polynomial, which are found as the eigenvalues of a tri-diagonal **Jacobi matrix**.
- **Product-Difference (PD) Algorithm**: An alternative to Wheeler that uses a different recurrence relationship to construct the Jacobi matrix.

### 3.2 Multivariate QMOM
- **Conditional QMOM (CQMOM)**: Solves the multivariate moment problem by recursively decomposing the NDF into a sequence of conditional 1D distributions. It uses a **recursive deconvolution** approach.
- **Tensor-product QMOM**: Suitable for systems where internal coordinates are independent.
- **Brute-force QMOM**: Directly solves the nonlinear system using Newton-Raphson accelerated by **ForwardDiff.jl**.

### 3.3 Extended Quadrature (EQMOM)
Standard QMOM represents the NDF as a sum of Dirac delta functions. **EQMOM** extends this by using non-negative continuous kernel functions (Gaussian, Gamma, or Beta mixtures) with a bandwidth parameter $\sigma$.

### 3.4 Evolutionary Methods (DQMOM)
**Direct QMOM (DQMOM)** tracks the evolution of weights and nodes directly by solving a linear system derived from the moment-transport equations.

### 3.5 Robustness & Stability
- **Realizability**: Built-in Hankel-matrix-based validation for Stieltjes $[0, \infty)$ and Hamburger $(-\infty, \infty)$ supports.
- **McGraw Correction**: Repairs corrupted moments by maximizing the smoothness of $\ln(m_k)$.
- **Wright Correction**: Fallback log-normal reconstruction for highly corrupted sequences.

---

## 4. Core Features
- **Strict Zero-Allocation**: Core solvers utilize `StaticArrays.jl` and `Val{N}` static dispatch to ensure no heap allocations occur during inversion loops.
- **Unified API**: Every method implements `invert_moments(method, moments)`, returning a standardized `QuadratureResult`.
- **Numerical Robustness**: Adaptive rank reduction and moment repair algorithms.
- **Dual-Backend Support**: `NativeBackend()` (optimized $O(n^2)$ solvers) and `ExternalBackend()` (Standard Library).

---

## 5. Quick Start

### 1D QMOM (Wheeler Inversion)
```julia
using QBMM, StaticArrays

m = @SVector [1.0, 5.0, 26.0, 140.0] 
method = Wheeler(2)
res = invert_moments(method, m)
# res.weights -> [0.5, 0.5], res.nodes -> [4.0, 6.0]
```

---

## 6. Performance Guide
To achieve the best performance in high-frequency loops:
1. **Use `StaticArrays`**: Provide moments as `SVector` or `SMatrix`.
2. **Prefer `NativeBackend`**: Uses specialized $O(n^2)$ Björck-Pereyra solvers.
3. **Static Dispatch**: If $N$ is fixed, use the parametric constructor `Wheeler{N}()`.

---

## 7. Detailed API Documentation

### 7.1 The Unified Inversion Interface
`invert_moments(method::AbstractQBMM, moments::SArray; backend=NativeBackend())`
- **Returns**: `QuadratureResult{D, N, T}` containing `.weights`, `.nodes`, and optional `.sigmas`.

### 7.2 Solvers
- `Wheeler(N)`, `PD(N)`: Univariate solvers.
- `EQMOM(N, kernel)`: Continuous kernels (`GaussianKernel()`, `GammaKernel()`, `BetaKernel()`).
- `CQMOM(N_tuple)`, `ECQMOM(N_tuple, kernel)`: Multivariate solvers.
- `DQMOM(N)`: Utilities for direct tracking.

### 7.3 Robustness Tools
- `is_realizable(m; domain=:pos)`
- `mcgraw_correction(m)`

---

## 8. Citing
If you use `QBMM.jl` in your research, please cite the following work:

```bibtex
@book{marchisio2013computational,
  title={Computational Models for Polydisperse Particulate and Multiphase Systems},
  author={Marchisio, Daniele L and Fox, Rodney O},
  year={2013},
  publisher={Cambridge University Press}
}
```

---

## 9. Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, specifically regarding the **Zero-Allocation Policy**.

## 10. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
