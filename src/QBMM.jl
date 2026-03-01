module QBMM

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Roots

# --- 1. Core Infrastructure ---
include("Core/types.jl")
export AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend
export QuadratureResult

# --- 2. Math Utilities (Dual Backend) ---
include("Math/hankel.jl")
include("Math/stirling.jl")
include("Math/vandermonde.jl")
export hankel_matrix, stirling2, solve_vandermonde

# --- 3. 1D Solvers ---
include("Solvers/1D/wheeler.jl")
include("Solvers/1D/pd.jl")
include("Solvers/1D/eqmom.jl")
export Wheeler, PD, EQMOM
export AbstractKernel, GaussianKernel, GammaKernel, BetaKernel

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
export invert_moments

end # module
