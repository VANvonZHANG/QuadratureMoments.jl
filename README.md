# QBMM.jl

[![Build Status](https://github.com/VANvonZHANG/QBMM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/VANvonZHANG/QBMM.jl/actions)
[![Coverage](https://codecov.io/gh/VANvonZHANG/QBMM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/VANvonZHANG/QBMM.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-Performance Quadrature-Based Moment Methods for Mesoscale Modeling in Julia.**

`QBMM.jl` is a high-performance Julia library designed for mesoscale modeling of polydisperse particulate and multiphase systems. It provides a comprehensive suite of moment-inversion algorithms (QMOM, EQMOM, CQMOM, ECQMOM, DQMOM) optimized for industrial-grade CFD applications, emphasizing numerical robustness, zero-allocation execution, and a unified API.

---

## 1. Installation

`QBMM.jl` is currently available via GitHub. You can install it using the Julia package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/VANvonZHANG/QBMM.jl.git")
```

---

## 2. Project Positioning
In the simulation of multiphase flows (e.g., sprays, aerosols, crystallization), the evolution of the Number Density Function (NDF) is governed by the **Generalized Population Balance Equation (GPBE)**. `QBMM.jl` provides both the mathematical "engine" (moment inversion algorithms) and the physical closure models (source terms) to close the moment-transport equations derived from the GPBE. This allows CFD software to seamlessly reconstruct distributions and calculate the advection and reaction of the disperse phase.

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

### 3.6 Physical Source Terms (Closures)
The library provides zero-allocation calculation of the moment source terms ($S_k = dm_k/dt$) for various microscale physical processes:
- **Phase-space advection**: Continuous `ParticleGrowth` and `ParticleShrinkage` (evaporation/dissolution).
- **Point processes (0th, 1st, 2nd order)**: `Nucleation`, `Deposition`, `Breakage`, and `Aggregation`.
These physical processes can be intuitively superposed using the `+` operator, which resolves into a highly optimized, allocation-free loop at compile-time.

---

## 4. Core Features
- **Strict Zero-Allocation**: Both the core inversion solvers and the physical source term integrators utilize `StaticArrays.jl` and `@generated` static dispatch to ensure no heap allocations occur during inner loops.
- **Unified API**: Every inversion method implements `invert_moments(method, moments)`, and every physical process implements `compute_source_terms(physics, nodes, weights)`.
- **Numerical Robustness**: Adaptive rank reduction and moment repair algorithms.
- **Dual-Backend Support**: `NativeBackend()` (optimized $O(n^2)$ solvers with zero-allocation) and `ExternalBackend()` (Standard Library).

---

## 5. Quick Start

### 1D QMOM (Wheeler Inversion)
```julia
using QBMM, StaticArrays

m = @SVector [1.0, 5.0, 26.0, 140.0] 
method = Wheeler(2) # Or Wheeler{2}() for static dispatch
res = invert_moments(method, m)
# res.weights -> [0.5, 0.5], res.nodes -> [4.0, 6.0]
```

### Physical Superposition with DQMOM
Demonstrating the elegance of the physical-mathematical architecture:
```julia
using QBMM, StaticArrays

nodes = @SVector [1.0, 2.0, 3.0]
weights = @SVector [0.5, 0.3, 0.2]

# 1. Define physical kernels
growth = ParticleGrowth(xi -> 0.1 * xi)            # Size-dependent growth
agg    = Aggregation((xi, xj) -> 0.05 * (xi + xj)) # Collision kernel

# 2. Superpose physics (compiled into a single zero-allocation pass)
physics = growth + agg

# 3. Compute 2N source terms for DQMOM
S_k = compute_source_terms(physics, nodes, weights, Val(6))

# 4. Solve the DQMOM linear system for weight/node evolution rates
da, db = dqmom_solve(DQMOM(3), nodes, S_k)
```

---

## 6. Performance Guide
To achieve the best performance in high-frequency loops:
1. **Use `StaticArrays`**: Provide moments as `SVector` or `SMatrix`.
2. **Prefer `NativeBackend`**: Uses specialized $O(n^2)$ Björck-Pereyra and Hankel-based solvers for zero-allocation performance.
3. **Static Dispatch**: If $N$ is fixed, use the parametric constructor `Wheeler{N}()` to help the compiler unroll loops.

---

## 7. Detailed API Documentation

### 7.1 The Unified Inversion Interface
`invert_moments(method::AbstractQBMM, moments::SArray; backend=NativeBackend())`
- **Returns**: `QuadratureResult{D, N, T}` containing:
  - `.weights`: `SVector{N, T}` of quadrature weights.
  - `.nodes`: `SMatrix{N, D, T}` of quadrature nodes.
  - `.sigmas`: Optional bandwidth parameter (for EQMOM), otherwise `nothing`.

### 7.2 Solvers
- `Wheeler(N)`, `PD(N)`: Univariate solvers.
- `EQMOM(N, kernel)`: Continuous kernels (`GaussianKernel()`, `GammaKernel()`, `BetaKernel()`).
- `CQMOM(N_tuple)`, `ECQMOM(N_tuple, kernel)`: Multivariate solvers.
- `DQMOM(N)`: Utilities for direct tracking. `dqmom_solve(method, nodes, source_terms)` returns evolution rates `(da, db)`.

### 7.3 Robustness Tools
- `is_realizable(m; domain=:pos, backend=NativeBackend())`
- `mcgraw_correction(m; backend=NativeBackend())`

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
