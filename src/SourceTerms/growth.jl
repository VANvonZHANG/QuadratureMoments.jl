# src/SourceTerms/growth.jl

using StaticArrays
using ..QBMM: AbstractSourceTerm

raw"""
    ParticleGrowth{F} <: AbstractSourceTerm

A source term representing continuous particle growth.

# Type Parameters
- `F`: The type of the rate function.

# Fields
- `rate_func::F`: A function `G(xi)` that returns the growth rate of a particle of size `xi`.
raw"""
struct ParticleGrowth{F} <: AbstractSourceTerm
    rate_func::F
end

raw"""
    compute_source_terms(growth::ParticleGrowth, nodes, weights, ::Val{L})

Compute the moment source terms due to continuous particle growth.
The source term for the \$k\$-th moment is:
\$S_k = \sum_{i=1}^N w_i k \xi_i^{k-1} G(\xi_i)\$

Note that for \$k=0\$ (total number), the source term is zero.
raw"""
function compute_source_terms(
    growth::ParticleGrowth,
    nodes::SVector{N, T},
    weights::SVector{N, T},
    ::Val{L},
) where {N, T, L}
    return SVector{L, T}(ntuple(Val(L)) do idx
        k = idx - 1
        if k == 0
            return zero(T)
        else
            val = zero(T)
            for i in 1:N
                val += weights[i] * k * (nodes[i]^(k - 1)) * growth.rate_func(nodes[i])
            end
            return val
        end
    end)
end
