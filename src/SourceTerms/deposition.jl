# src/SourceTerms/deposition.jl

using StaticArrays
using ..QBMM: AbstractSourceTerm

raw"""
    Deposition{F} <: AbstractSourceTerm

A source term representing particle deposition, filtration, or disappearance 
from the domain (first-order point process with no birth).

# Type Parameters
- `F`: The type of the deposition rate function.

# Fields
- `rate_func::F`: A function `K(xi)` that returns the deposition/removal rate of a particle of size `xi`.
raw"""
struct Deposition{F} <: AbstractSourceTerm
    rate_func::F
end

raw"""
    compute_source_terms(dep::Deposition, nodes, weights, ::Val{L})

Compute the moment source terms due to particle deposition.
The source term for the \$k\$-th moment is strictly negative (sink):
\$S_k = - \sum_{i=1}^N w_i K_{dep}(\xi_i) \xi_i^k\$
raw"""
function compute_source_terms(
    dep::Deposition, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {N,T,L}
    return SVector{L,T}(
        ntuple(Val(L)) do idx
            k = idx - 1
            val = zero(T)
            for i in 1:N
                val -= weights[i] * dep.rate_func(nodes[i]) * (nodes[i]^k)
            end
            return val
        end,
    )
end
