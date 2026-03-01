module QBMM

using LinearAlgebra
using StaticArrays

# 抽象类型
abstract type AbstractQBMM end

export AbstractQBMM
export wheeler_inversion, is_realizable
export CQMOM, invert_moments
export EQMOM, GaussianKernel, GammaKernel, BetaKernel
export dqmom_matrix, dqmom_solve

include("wheeler.jl")
include("realizability.jl")
include("cqmom.jl")
include("eqmom.jl")
include("dqmom.jl")

end # module
