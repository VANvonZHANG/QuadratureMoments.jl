module QuadratureMoments

raw"""
    QuadratureMoments

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
include("Core/source_terms_api.jl")
export AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend
export QuadratureResult, n_active
export AbstractKernel, GaussianKernel, GammaKernel, BetaKernel
export AbstractSourceTerm, CompositeSourceTerm, compute_source_terms

# --- 2. Math Utilities (Dual Backend) ---
include("Math/hankel.jl")
include("Math/stirling.jl")
include("Math/vandermonde.jl")
include("Math/moments_utils.jl")
export hankel_matrix, stirling2, solve_vandermonde, solve_vandermonde_transpose
export reconstruct_moment

# --- 3. Physical Source Terms ---
include("SourceTerms/growth.jl")
include("SourceTerms/shrinkage.jl")
include("SourceTerms/aggregation.jl")
include("SourceTerms/breakage.jl")
include("SourceTerms/nucleation.jl")
include("SourceTerms/deposition.jl")
export ParticleGrowth, ParticleShrinkage, Aggregation, Breakage, Nucleation, Deposition

# --- 4. 1D Solvers ---
include("Solvers/1D/wheeler.jl")
include("Solvers/1D/pd.jl")
include("Solvers/1D/eqmom.jl")
include("Solvers/1D/eqmom_expansion.jl")
export Wheeler, PD, EQMOM, expand_quadrature

# --- 5. Multi-D Solvers ---
include("Solvers/MultiD/cqmom.jl")
include("Solvers/MultiD/ecqmom.jl")
include("Solvers/MultiD/tensor.jl")
include("Solvers/MultiD/brute.jl")
export CQMOM, ECQMOM, TensorQMOM, BruteQMOM

# --- 6. Evolution Solvers ---
include("Solvers/Evolution/dqmom.jl")
include("Solvers/Evolution/realizable_evolution.jl")
export DQMOM, dqmom_source_terms, dqmom_system_matrix, dqmom_matrix, dqmom_solve
export evolve_moments

# --- 7. Robustness Tools ---
include("Tools/realizability.jl")
include("Tools/correction.jl")
export is_realizable, mcgraw_correction

# --- 8. Analysis Tools ---
include("Analysis/Analysis.jl")
using .Analysis
export Analysis
export reconstruct_ndf, evaluate_kernel
export MomentComparison, compare_moments, verify_reconstruction
export abs_errors, rel_errors, max_abs_errors, max_rel_errors, verify

# --- 9. Main API ---

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
using QuadratureMoments, StaticArrays
m = @SVector [1.0, 5.0, 26.0, 140.0]
res = invert_moments(Wheeler(2), m)
```
raw"""
function invert_moments end
export invert_moments

end # module
