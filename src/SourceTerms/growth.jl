# src/SourceTerms/growth.jl

using StaticArrays
using ..QuadratureMoments: AbstractSourceTerm

raw"""
    ParticleGrowth{F} <: AbstractSourceTerm

A source term representing continuous particle growth (phase-space advection).

!!! note "Important Assumption: Zero Disappearance"
    The current implementation assumes that particles strictly grow (\$G(\xi) > 0\$) 
    or that the rate of shrinkage does not cause particles to completely disappear 
    (reach size 0). Thus, the zeroth moment source term is strictly zero (\$S_0 = 0\$). 
    If you need to model complete particle evaporation/dissolution (finite-time truncation), 
    this requires more complex flux-based numerical schemes rather than simple moment source terms.

# Type Parameters
- `F`: The type of the rate function.

# Fields
- `rate_func::F`: A function `G(xi)` that returns the continuous growth rate (\$d\xi/dt\$) of a particle of size `xi`.
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
    growth::ParticleGrowth, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {N,T,L}
    return SVector{L,T}(
        ntuple(Val(L)) do idx
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
        end,
    )
end
