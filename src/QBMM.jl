module QBMM

raw"""
    QBMM

A high-performance Julia library for Quadrature-Based Moment Methods.
Optimized for industrial-grade CFD solvers with zero-allocation execution paths.

See [README.md](README.md) for detailed usage and algorithms.
raw"""

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Roots
using Combinatorics

# --- 1. Core Infrastructure ---
include("Core/types.jl")
include("Core/kernels.jl")
export AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend
export QuadratureResult
export AbstractKernel, GaussianKernel, GammaKernel, BetaKernel

# --- 2. Math Utilities (Dual Backend) ---
include("Math/hankel.jl")
include("Math/stirling.jl")
include("Math/vandermonde.jl")
include("Math/moments_utils.jl")
export hankel_matrix, stirling2, solve_vandermonde, solve_vandermonde_transpose
export reconstruct_moment

# --- 3. 1D Solvers ---
include("Solvers/1D/wheeler.jl")
include("Solvers/1D/pd.jl")
include("Solvers/1D/eqmom.jl")
export Wheeler, PD, EQMOM

# --- 4. Multi-D Solvers ---
include("Solvers/MultiD/cqmom.jl")
include("Solvers/MultiD/ecqmom.jl")
include("Solvers/MultiD/tensor.jl")
include("Solvers/MultiD/brute.jl")
export CQMOM, ECQMOM, TensorQMOM, BruteQMOM

# --- 5. Evolution Solvers ---
include("Solvers/Evolution/dqmom.jl")
export DQMOM, dqmom_source_terms, dqmom_system_matrix, dqmom_matrix, dqmom_solve

# --- 6. Robustness Tools ---
include("Tools/realizability.jl")
include("Tools/correction.jl")
export is_realizable, mcgraw_correction

# --- 7. Main API ---

raw"""
    invert_moments(method::AbstractQBMM, moments; backend=NativeBackend()) -> QuadratureResult

Perform moment inversion using the specified QBMM algorithm.

# Arguments
- `method`: An instance of an `AbstractQBMM` solver (e.g., `Wheeler(3)`, `CQMOM((2,2))`).
- `moments`: A vector or matrix of transportable moments.
- `backend`: `NativeBackend()` (default, zero-allocation) or `ExternalBackend()`.

# Returns
A `QuadratureResult` structure containing weights, nodes, and optional sigmas.

# Examples
```julia
using QBMM, StaticArrays
m = @SVector [1.0, 5.0, 26.0, 140.0]
res = invert_moments(Wheeler(2), m)
```
raw"""
function invert_moments end
export invert_moments

end # module
