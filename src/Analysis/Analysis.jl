module Analysis

using ..QuadratureMoments
using StaticArrays
using LinearAlgebra
using SpecialFunctions

include("ndf_reconstruction.jl")
include("moment_verification.jl")

export reconstruct_ndf, evaluate_kernel
export MomentComparison, compare_moments, verify_reconstruction
export abs_errors, rel_errors, max_abs_errors, max_rel_errors, verify

end
