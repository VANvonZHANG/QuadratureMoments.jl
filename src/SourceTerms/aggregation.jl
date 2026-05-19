# src/SourceTerms/aggregation.jl

using StaticArrays
using ..QBMM: AbstractSourceTerm

raw"""
    Aggregation{K} <: AbstractSourceTerm

A source term representing particle aggregation.

# Type Parameters
- `K`: The type of the aggregation kernel function.

# Fields
- `kernel::K`: A function `beta(xi, xj)` that returns the aggregation rate between particles of size `xi` and `xj`.
raw"""
struct Aggregation{K} <: AbstractSourceTerm
    kernel::K
end

raw"""
    compute_source_terms(agg::Aggregation, nodes, weights, ::Val{L})

Compute the moment source terms due to particle aggregation.
The source term for the \$k\$-th moment is:
\$S_k = \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N w_i w_j \beta(\xi_i, \xi_j) (\xi_i + \xi_j)^k - \sum_{i=1}^N \sum_{j=1}^N w_i w_j \beta(\xi_i, \xi_j) \xi_i^k\$
raw"""
function compute_source_terms(
    agg::Aggregation, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {N,T,L}
    return SVector{L,T}(
        ntuple(Val(L)) do idx
            k = idx - 1
            val = zero(T)
            for i in 1:N
                for j in 1:N
                    beta = agg.kernel(nodes[i], nodes[j])
                    term1 =
                        0.5 * weights[i] * weights[j] * beta * ((nodes[i] + nodes[j])^k)
                    term2 = weights[i] * weights[j] * beta * (nodes[i]^k)
                    val += (term1 - term2)
                end
            end
            return val
        end,
    )
end
