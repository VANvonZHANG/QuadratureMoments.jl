# QBMM.jl Developer Guide & Contributing Guidelines

Welcome to the `QBMM.jl` developer community! This document provides a deep dive into the library's architecture, design philosophy, and the strict performance standards required for contributions.

---

## 1. Project Architecture Overview

`QBMM.jl` is organized into a layered architecture to separate low-level mathematical kernels from high-level physics-based solvers.

```text
src/
├── Core/           # Foundational types, Traits, and Kernel definitions.
├── Math/           # Low-level numerical utilities (Dual-Backend).
├── Solvers/        
│   ├── 1D/         # Atomic univariate inversion (Wheeler, PD, EQMOM).
│   ├── MultiD/     # Recursive multivariate solvers (CQMOM, ECQMOM).
│   └── Evolution/  # Direct tracking methods (DQMOM).
└── Tools/          # Robustness tools (Realizability, McGraw Correction).
```

### 1.1 Design Philosophy: Dual-Backend System
We separate mathematical operations into a **Dual-Backend** system located in `src/Math/`:

- **NativeBackend (Default)**: Optimized specifically for the small-scale matrices ($N < 15$) typical in QBMM. It uses `StaticArrays.jl` and $O(n^2)$ algorithms (like Björck-Pereyra for Vandermonde systems) to achieve **zero-allocation** and maximum speed.
- **ExternalBackend**: Uses standard Julia `LinearAlgebra` and generic `Array` types. This is intended for cross-verification, handling larger moment sets, or when `StaticArrays` compilation times become prohibitive.

### 1.2 Trait-Based Dispatch
We use an abstract type hierarchy and traits to ensure a unified API:
- `AbstractQBMM{D, N}`: Encodes the Dimension ($D$) and Node Count ($N$) into the type system, allowing the compiler to unroll loops and pre-allocate stack memory.
- `AbstractMathBackend`: Dispatches math utilities at compile-time without runtime overhead.

---

## 2. Performance Red-Line: Zero-Allocation Policy

`QBMM.jl` is designed to be used inside the inner loops of CFD solvers. Therefore, **heap allocations in the core inversion path are strictly prohibited.**

### 2.1 The Rules
When contributing to `Solvers/` or `Math/`:
1. **NO `zeros(n, m)` or `Vector(undef, n)`**: Use `zero(MVector{N, T})` or `SVector` instead.
2. **NO `push!()` or `append!()`**: Dimensions must be known via type parameters.
3. **NO Slicing**: Avoid `m[1:4]`, which creates a copy. Use `@views m[1:4]` or `SVector{4}(...)`.
4. **NO Type Instability**: Ensure that all variables have concrete types that the compiler can infer.

### 2.2 Verification
Every PR must include a benchmark check using `BenchmarkTools.jl`. A successful 1D/2D inversion should report **"0 allocations"**.

```julia
using BenchmarkTools, QBMM, StaticArrays
m = @SVector[...]
@btime invert_moments(Wheeler(3), $m) 
# Target: ~700ns, 0 samples, 0 bytes allocated
```

---

## 3. Extending the Library

### 3.1 Adding New Kernels (EQMOM)
To add a new continuous distribution kernel (e.g., Log-Normal):
1. **Define the type** in `src/Solvers/1D/eqmom.jl` (inheriting from `AbstractKernel`).
2. **Implement `compute_modified_moments`**: This function transforms raw moments into the orthogonal space of your kernel.
3. **Implement `_reconstruct_moment`**: Needed for the internal $\sigma$-optimization loop.

### 3.2 Recursive Decomposition Principle (CQMOM/ECQMOM)
The multivariate solvers use a **Recursive Deconvolution** strategy to avoid the $O(N^D)$ complexity of direct solvers.

**How it works:**
1. **Marginalization**: The $D$-dimensional moment tensor is projected onto the first dimension.
2. **1D Inversion**: The marginal 1D problem is solved using Wheeler or EQMOM to find $\xi_{1,\alpha}$ and $w_{1,\alpha}$.
3. **Conditional Moments**: For each node in the first dimension, we solve a **Transpose Vandermonde System** ($V^T \mathbf{c} = \mathbf{b}$) to extract the "conditional moments" of the remaining $D-1$ dimensions.
4. **Recursion**: Step 1 is repeated for the conditional moments until $D=1$.

**Developer Note**: When modifying `src/Solvers/MultiD/cqmom.jl`, ensure the `backend` is passed down through all recursive calls to maintain performance consistency.

---

## 4. Coding Standards
- **Parametric Types**: Favor `Val{N}` for static dispatch.
- **Documentation**: Use Docstrings for all exported functions, including mathematical references where applicable.
- **Tests**: New solvers must pass realizability tests and reconstruct known analytic distributions (e.g., mixtures of Gaussians).

---

Thank you for helping us make `QBMM.jl` the fastest moment-method library in the Julia ecosystem!
