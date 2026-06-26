# src/Core/coordinates.jl

raw"""
    AbstractCoord

Abstract type for the physical interpretation of the internal coordinate ξ.
Determines the moment-geometry of aggregation birth and the daughter-distribution
form under breakage.
"""
abstract type AbstractCoord end

raw"""
    Mass <: AbstractCoord

Internal coordinate ξ is volume/mass-like: volume is additive, so aggregating
particles of size ξ_i, ξ_j yields size ξ_i + ξ_j. Birth geometry `(ξ_i + ξ_j)^k`.
Default convention (preserves the library's historical behavior).
"""
struct Mass <: AbstractCoord end

raw"""
    Length <: AbstractCoord

Internal coordinate ξ is a length (e.g. diameter): volume ∝ ξ³ is conserved, so
aggregating ξ_i, ξ_j yields size `(ξ_i³ + ξ_j³)^(1/3)`. Birth geometry `(ξ_i³ + ξ_j³)^(k/3)`.
"""
struct Length <: AbstractCoord end

raw"""
    aggregation_birth(coord, a1, a2, k)

k-th moment of the particle formed by aggregating two particles of internal
coordinate values `a1`, `a2`, under convention `coord`. Physics-independent
(the rate β is applied by the caller). Analog of OpenQBMM
`aggregationKernel::nodeSource` (length) / `massNodeSource` (mass).
"""
@inline aggregation_birth(::Mass, a1, a2, k) = (a1 + a2)^k
@inline aggregation_birth(::Length, a1, a2, k) = (a1^3 + a2^3)^(k / 3)

raw"""
    daughter_moment(d, k, xi, coord)

Protocol: k-th moment of the daughter distribution when a particle of value `xi`
breaks, under convention `coord`. Library daughter types extend this for both
`Mass` and `Length`. A plain `Function` `d` is accepted as a Mass-convention
fragment-moment function `d(k, xi)` (backward compat); `Length` with a plain
function errors — use a library daughter object instead.
"""
@inline daughter_moment(d::Function, k, xi, ::Mass) = d(k, xi)
@inline function daughter_moment(d::Function, k, xi, ::Length)
    return error(
        "daughter_moment: a plain-function daughter cannot infer its Length form. " *
        "Use a library daughter distribution (e.g. Symmetric()) instead."
    )
end
