# QuadratureMoments.jl Developer Guide & Contributing Guidelines

Welcome to the `QuadratureMoments.jl` developer community! This document provides a deep dive into the library's architecture, design philosophy, and the strict performance standards required for contributions.

---

## 1. Project Architecture Overview

`QuadratureMoments.jl` is organized into a layered architecture to separate low-level mathematical kernels from high-level physics-based solvers.

```text
src/
├── Core/           # Foundational types, Traits, Kernel definitions, and SourceTerms API.
├── Math/           # Low-level numerical utilities (Dual-Backend: Hankel, Vandermonde, Stirling).
├── SourceTerms/    # Physical closure models (Growth, Aggregation, Breakage, etc.).
├── Solvers/        
│   ├── 1D/         # Atomic univariate inversion (Wheeler, PD, EQMOM).
│   ├── MultiD/     # Recursive multivariate solvers (CQMOM, ECQMOM, Tensor, Brute).
│   └── Evolution/  # Direct tracking methods (DQMOM).
└── Tools/          # Robustness tools (Realizability, McGraw Correction).
```

### 1.1 Design Philosophy: Dual-Backend System
We separate mathematical operations into a **Dual-Backend** system located in `src/Math/`:

- **NativeBackend (Default)**: Optimized specifically for the small-scale matrices ($N < 20$) typical in QBMM. It uses `StaticArrays.jl` and specialized algorithms (like Björck-Pereyra for Vandermonde systems, Hankel-based PSD checks) to achieve **zero-allocation** and maximum speed.
- **ExternalBackend**: Uses standard Julia `LinearAlgebra` and generic `Array` types. This is intended for cross-verification, handling larger moment sets, or when `StaticArrays` compilation times become prohibitive.

### 1.2 Trait-Based Dispatch
We use an abstract type hierarchy and traits to ensure a unified API:
- `AbstractQBMM{D, N}`: Encodes the Dimension ($D$) and Node Count ($N$) into the type system, allowing the compiler to unroll loops and pre-allocate stack memory.
- `AbstractMathBackend`: Dispatches math utilities at compile-time without runtime overhead.
- `QuadratureResult{D, N, T}`: A standardized structure containing `.weights` (SVector), `.nodes` (SMatrix), and `.sigmas` (optional).

---

## 2. Performance Red-Line: Zero-Allocation Policy

`QuadratureMoments.jl` is designed to be used inside the inner loops of CFD solvers. Therefore, **heap allocations in the core inversion path are strictly prohibited.**

### 2.1 The Rules
When contributing to `Solvers/` or `Math/`:
1. **NO `zeros(n, m)` or `Vector(undef, n)`**: Use `zero(MMatrix{N, M, T})` or `SVector` instead for `NativeBackend`.
2. **NO `push!()` or `append!()`**: Dimensions must be known via type parameters.
3. **NO Slicing**: Avoid `m[1:4]`, which creates a copy. Use `@views m[1:4]` or `SVector{4}(...)`.
4. **NO Type Instability**: Ensure that all variables have concrete types that the compiler can infer. Avoid `any` or abstract containers in inner loops.

### 2.2 Verification
Every PR must include a benchmark check using `BenchmarkTools.jl` or `Test.@allocated`. A successful inversion should report **"0 allocations"** for static inputs.

```julia
using Test, QBMM, StaticArrays
m = @SVector[...]
method = Wheeler(2)
@test @allocated(invert_moments(method, m)) == 0
```

---

## 3. Extending the Library

### 3.1 Adding New Kernels (EQMOM)
To add a new continuous distribution kernel (e.g., Log-Normal):
1. **Define the type** in `src/Core/kernels.jl` (inheriting from `AbstractKernel`).
2. **Implement `compute_modified_moments`** in `src/Solvers/1D/eqmom.jl`: This function transforms raw moments into the orthogonal space of your kernel.
3. **Implement `_reconstruct_moment`**: Needed for the internal $\sigma$-optimization loop.

### 3.2 Recursive Decomposition Principle (CQMOM/ECQMOM)
The multivariate solvers use a **Recursive Deconvolution** strategy to avoid the $O(N^D)$ complexity of direct solvers.

**How it works:**
1. **Marginalization**: The $D$-dimensional moment tensor is projected onto the first dimension.
2. **1D Inversion**: The marginal 1D problem is solved using Wheeler or EQMOM to find $\xi_{1,\alpha}$ and $w_{1,\alpha}$.
3. **Conditional Moments**: For each node in the first dimension, we solve a **Transpose Vandermonde System** ($V^T \mathbf{c} = \mathbf{b}$) via `Math/vandermonde.jl` to extract the "conditional moments" of the remaining $D-1$ dimensions.
4. **Recursion**: Step 1 is repeated for the conditional moments until $D=1$.

**Developer Note**: When modifying `src/Solvers/MultiD/cqmom.jl`, ensure the `backend` is passed down through all recursive calls to maintain performance consistency.

### 3.3 Adding New Physical Source Terms
`QuadratureMoments.jl` uses a composition-based physics engine. To add a new microscale physical process (e.g., a specific collision kernel or reaction):

1. **Define the type** in `src/SourceTerms/` inheriting from `AbstractSourceTerm`.
2. **Implement `compute_source_terms`**:
   ```julia
   function compute_source_terms(source::MyNewPhysics, nodes::SVector{N, T}, weights::SVector{N, T}, ::Val{L}) where {N, T, L}
       # MUST return an SVector{L, T} using `ntuple` to ensure zero-allocation.
       return SVector{L, T}(ntuple(Val(L)) do idx
           k = idx - 1
           # ... physical integration logic ...
       end)
   end
   ```
3. **Composite Pattern**: The base architecture in `Core/source_terms_api.jl` overloads the `+` operator to create `CompositeSourceTerm`s. This uses `@generated` functions to unroll the summation at compile-time, allowing users to stack multiple physics models (`physics = growth + aggregation + breakage`) with absolutely zero runtime overhead. Always ensure your new source term returns an `SVector` to maintain this property.

---

## 4. Coding Standards
- **Parametric Types**: Favor `Val{N}` or `N` as a type parameter for static dispatch.
- **Documentation**: Use Docstrings for all exported functions, including mathematical references where applicable.
- **Tests**: New solvers must pass realizability tests and reconstruct known analytic distributions (e.g., mixtures of Gaussians).

---

Thank you for helping us make `QuadratureMoments.jl` the fastest moment-method library in the Julia ecosystem!
