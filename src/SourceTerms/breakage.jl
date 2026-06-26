# src/SourceTerms/breakage.jl

using StaticArrays
using ..QuadratureMoments: AbstractSourceTerm, AbstractCoord, MassBased, daughter_moment

raw"""
    Breakage{F, P, C} <: AbstractSourceTerm

A source term representing particle breakage.

# Type Parameters
- `F`: breakage frequency function type `b(xi)`.
- `P`: daughter distribution — a library object extending `daughter_moment`, or a plain `Function` `(k, xi) -> ...` (MassBased convention only).
- `C<:AbstractCoord`: coordinate convention (`MassBased` or `LengthBased`).

# Fields
- `frequency_func::F`: `b(xi)` returning the breakage frequency of a particle of size `xi`.
- `daughter::P`: daughter distribution; its k-th fragment moment is `daughter_moment(daughter, k, xi, coord)`.
- `coord::C`: coordinate convention, default `MassBased()`.

For a plain-function `daughter`, `MassBased` calls `daughter(k, xi)` (backward compatible);
`LengthBased` errors — use a library daughter object (`SymmetricFragmentation`, `Uniform`, …) instead.
"""
struct Breakage{F,P,C<:AbstractCoord} <: AbstractSourceTerm
    frequency_func::F
    daughter::P
    coord::C
    Breakage(f::F, daughter::P, coord::C=MassBased()) where {F,P,C<:AbstractCoord} = new{F,P,C}(f, daughter, coord)
end

raw"""
    compute_source_terms(breakage::Breakage, nodes, weights, ::Val{L})

``S_k = Σ_i w_i b(xi_i) daughter_moment(daughter, k, xi_i, coord) - Σ_i w_i b(xi_i) xi_i^k``.
"""
function compute_source_terms(
    breakage::Breakage{F,P,C}, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {F,P,C,N,T,L}
    return SVector{L,T}(
        ntuple(Val(L)) do idx
            k = idx - 1
            val = zero(T)
            for i in 1:N
                iszero(weights[i]) && continue
                freq = breakage.frequency_func(nodes[i])
                frag_moment = daughter_moment(breakage.daughter, k, nodes[i], breakage.coord)
                val += weights[i] * freq * (frag_moment - nodes[i]^k)
            end
            return val
        end,
    )
end
