# src/SourceTerms/nucleation.jl

using StaticArrays
using ..QuadratureMoments: AbstractSourceTerm

raw"""
    Nucleation{T} <: AbstractSourceTerm

A source term representing particle nucleation (zeroth-order point process).
This models the spontaneous formation of new particles from the continuous phase 
at a specific critical size.

# Type Parameters
- `T`: The floating-point type of the rates and sizes.

# Fields
- `rate::T`: The nucleation rate \$J_{nuc}\$ (number of particles formed per unit volume per unit time).
- `critical_size::T`: The size \$\xi_{nuc}\$ of the newly nucleated particles.
raw"""
struct Nucleation{T} <: AbstractSourceTerm
    rate::T
    critical_size::T
end

raw"""
    compute_source_terms(nuc::Nucleation, nodes, weights, ::Val{L})

Compute the moment source terms due to nucleation.
The source term for the \$k\$-th moment is:
\$S_k = J_{nuc} \cdot \xi_{nuc}^k\$
raw"""
function compute_source_terms(
    nuc::Nucleation{T_field}, nodes::SVector{N,T}, weights::SVector{N,T}, ::Val{L}
) where {T_field,N,T,L}
    # Use T for the final vector type to match the quadrature precision
    return SVector{L,T}(
        ntuple(Val(L)) do idx
            k = idx - 1
            return T(nuc.rate) * (T(nuc.critical_size)^k)
        end,
    )
end
