module QBMM

using LinearAlgebra
using StaticArrays

# 抽象类型
abstract type AbstractQBMM end

export AbstractQBMM
export wheeler_inversion, pd_inversion, is_realizable
export CQMOM, TensorQMOM, ECQMOM, BruteQMOM, invert_moments
export EQMOM, GaussianKernel, GammaKernel, BetaKernel
export dqmom_matrix, dqmom_solve

include("wheeler.jl")
include("pd.jl")
include("realizability.jl")
include("cqmom.jl")
include("tensor.jl")
include("brute.jl")
include("eqmom.jl")
include("ecqmom.jl")
include("dqmom.jl")

end # module
