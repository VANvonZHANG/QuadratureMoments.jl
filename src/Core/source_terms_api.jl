# src/Core/source_terms_api.jl

using StaticArrays

raw"""
    AbstractSourceTerm

Base abstract type for all physical source terms (e.g., growth, aggregation, breakage) 
that contribute to the evolution of the moments.
raw"""
abstract type AbstractSourceTerm end

raw"""
    compute_source_terms(source::AbstractSourceTerm, nodes::SVector, weights::SVector, ::Val{L}) -> SVector

Compute the source terms \$S_k = dm_k/dt\$ for the first `L` moments (from \$k=0\$ to \$L-1\$), 
given the current quadrature `nodes` and `weights`.

This method must be implemented by all concrete subtypes of `AbstractSourceTerm`.
It must return an `SVector{L, T}` to maintain zero-allocation guarantees.
raw"""
function compute_source_terms(
    source::AbstractSourceTerm, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {N,T,L}
    return error(
        "`compute_source_terms` is not implemented for source type: $(typeof(source))"
    )
end

# --- Composite Pattern for Stacking Physics ---

raw"""
    CompositeSourceTerm{T<:Tuple} <: AbstractSourceTerm

A composite type that holds multiple source terms. 
Evaluates by summing the contributions of all contained source terms.
raw"""
struct CompositeSourceTerm{T<:Tuple} <: AbstractSourceTerm
    sources::T
end

# Overload the `+` operator to allow easy stacking of physical processes
Base.:+(s1::AbstractSourceTerm, s2::AbstractSourceTerm) = CompositeSourceTerm((s1, s2))
function Base.:+(c::CompositeSourceTerm, s::AbstractSourceTerm)
    return CompositeSourceTerm((c.sources..., s))
end
function Base.:+(s::AbstractSourceTerm, c::CompositeSourceTerm)
    return CompositeSourceTerm((s, c.sources...))
end
function Base.:+(c1::CompositeSourceTerm, c2::CompositeSourceTerm)
    return CompositeSourceTerm((c1.sources..., c2.sources...))
end

raw"""
    compute_source_terms(comp::CompositeSourceTerm, nodes, weights, val_L)

Unrolls and sums the computation of all source terms in the composite at compile time.
raw"""
@generated function compute_source_terms(
    comp::CompositeSourceTerm{Sources},
    nodes::SVector{N,T},
    weights::SVector{N,T},
    val_L::Val{L},
) where {Sources,N,T,L}
    num_sources = length(Sources.parameters)

    if num_sources == 0
        return :(zero(SVector{L,T}))
    end

    expr = :(compute_source_terms(comp.sources[1], nodes, weights, val_L))
    for i in 2:num_sources
        expr = :($expr + compute_source_terms(comp.sources[$i], nodes, weights, val_L))
    end

    return expr
end
