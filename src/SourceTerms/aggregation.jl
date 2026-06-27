# src/SourceTerms/aggregation.jl

using StaticArrays
using ..QuadratureMoments: AbstractSourceTerm, AbstractCoord, MassBased, aggregation_birth

raw"""
    Aggregation{K, C} <: AbstractSourceTerm

A source term representing particle aggregation.

# Type Parameters
- `K`: type of the aggregation kernel, a callable `beta(xi, xj)`.
- `C<:AbstractCoord`: coordinate convention (`MassBased` or `LengthBased`).

# Fields
- `kernel::K`: `beta(xi, xj)` returning the aggregation rate between particles of size `xi` and `xj`.
- `coord::C`: coordinate convention governing the birth-term geometry. Defaults to `MassBased()`.

For `MassBased` the birth term is `(xi_i + xi_j)^k`; for `LengthBased` (xi = diameter) it is
`(xi_i^3 + xi_j^3)^(k/3)`. The death term `xi_i^k` is coordinate-independent.
raw"""
struct Aggregation{K,C<:AbstractCoord} <: AbstractSourceTerm
    kernel::K
    coord::C
    function Aggregation(kernel::K, coord::C=MassBased()) where {K,C<:AbstractCoord}
        return new{K,C}(kernel, coord)
    end
end

raw"""
    compute_source_terms(agg::Aggregation, nodes, weights, ::Val{L})

``S_k = 0.5 Σ_i Σ_j w_i w_j β(xi_i, xi_j) aggregation_birth(coord, xi_i, xi_j, k)
      - Σ_i Σ_j w_i w_j β(xi_i, xi_j) xi_i^k``.
raw"""
function compute_source_terms(
    agg::Aggregation{K,C}, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {K,C,N,T,L}
    return SVector{L,T}(
        ntuple(Val(L)) do idx
            k = idx - 1
            val = zero(T)
            for i in 1:N
                iszero(weights[i]) && continue
                for j in 1:N
                    iszero(weights[j]) && continue
                    β = agg.kernel(nodes[i], nodes[j])
                    val +=
                        weights[i] *
                        weights[j] *
                        β *
                        (
                            0.5 * aggregation_birth(agg.coord, nodes[i], nodes[j], k) -
                            nodes[i]^k
                        )
                end
            end
            return val
        end,
    )
end
