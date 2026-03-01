# src/Core/types.jl

"""
    AbstractMathBackend

Abstract type for dispatching different mathematical computation backends.
Used to separate high-performance kernels from generic implementations.
"""
abstract type AbstractMathBackend end

"""
    NativeBackend <: AbstractMathBackend

High-performance backend utilizing `StaticArrays.jl` and specialized \$O(n^2)\$ algorithms.
Optimized for small-scale systems (\$N < 20\$) with zero-allocation targets.
"""
struct NativeBackend <: AbstractMathBackend end

"""
    ExternalBackend <: AbstractMathBackend

Generic backend using standard Julia `LinearAlgebra` and standard `Array` types.
Intended for verification, large systems, or generic types.
"""
struct ExternalBackend <: AbstractMathBackend end

"""
    AbstractQBMM{D, N}

Base abstract type for all Quadrature-Based Moment Methods.

# Type Parameters
- `D::Int`: Dimension of the internal coordinate space.
- `N`: Number of quadrature nodes (can be `Int` or `Tuple` for multivariate).
"""
abstract type AbstractQBMM{D, N} end

"""
    QuadratureResult{D, N, T}

Standardized container for moment inversion results.

# Fields
- `weights::SVector{N, T}`: Quadrature weights, normalized to \$m_0\$.
- `nodes::SMatrix{N, D, T}`: Positions of quadrature nodes in D-dimensional space.
- `sigmas::Union{Nothing, SMatrix{N, D, T}}`: Bandwidth parameters for continuous kernels (EQMOM/ECQMOM).
"""
struct QuadratureResult{D, N, T}
    weights::SVector{N, T}
    nodes::SMatrix{N, D, T}
    sigmas::Union{Nothing, SMatrix{N, D, T}}
end

"""
    QuadratureResult(weights, nodes, sigmas=nothing)

Convenience constructor for `QuadratureResult`. Automatically infers dimensions and types.
"""
function QuadratureResult(w::SVector{N, T}, n::SMatrix{N, D, T}, s=nothing) where {D, N, T}
    return QuadratureResult{D, N, T}(w, n, s)
end
