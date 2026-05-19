# src/Core/types.jl

raw"""
    AbstractMathBackend

Abstract type for dispatching different mathematical computation backends.
Used to separate high-performance kernels from generic implementations.
raw"""
abstract type AbstractMathBackend end

raw"""
    NativeBackend <: AbstractMathBackend

High-performance backend utilizing `StaticArrays.jl` and specialized \$O(n^2)\$ algorithms.
Optimized for small-scale systems (\$N < 20\$) with zero-allocation targets.
raw"""
struct NativeBackend <: AbstractMathBackend end

raw"""
    ExternalBackend <: AbstractMathBackend

Generic backend using standard Julia `LinearAlgebra` and standard `Array` types.
Intended for verification, large systems, or generic types.
raw"""
struct ExternalBackend <: AbstractMathBackend end

raw"""
    AbstractQBMM{D, N}

Base abstract type for all Quadrature-Based Moment Methods.

# Type Parameters
- `D::Int`: Dimension of the internal coordinate space.
- `N`: Number of quadrature nodes (can be `Int` or `Tuple` for multivariate).
raw"""
abstract type AbstractQBMM{D,N} end

raw"""
    QuadratureResult{D, N, T}

Standardized container for moment inversion results.

# Fields
- `weights::SVector{N, T}`: Quadrature weights, normalized to \$m_0\$.
- `nodes::SMatrix{N, D, T}`: Positions of quadrature nodes in D-dimensional space.
- `sigmas::Union{Nothing, SMatrix{N, D, T}}`: Bandwidth parameters for continuous kernels (EQMOM/ECQMOM).
raw"""
struct QuadratureResult{D,N,T}
    weights::SVector{N,T}
    nodes::SMatrix{N,D,T}
    sigmas::Union{Nothing,SMatrix{N,D,T}}
end

raw"""
    QuadratureResult(weights, nodes, sigmas=nothing)

Convenience constructor for `QuadratureResult`. Automatically infers dimensions and types.
raw"""
function QuadratureResult(w::SVector{N,T}, n::SMatrix{N,D,T}, s=nothing) where {D,N,T}
    return QuadratureResult{D,N,T}(w, n, s)
end
