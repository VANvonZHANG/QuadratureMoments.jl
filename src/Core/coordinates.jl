# src/Core/coordinates.jl

raw"""
    AbstractCoord

Abstract type for the physical interpretation of the internal coordinate ξ.
Determines the moment-geometry of aggregation birth and the daughter-distribution
form under breakage.
"""
abstract type AbstractCoord end

raw"""
    MassBased <: AbstractCoord

Internal coordinate ξ is volume/mass-like: volume is additive, so aggregating
particles of size ξ_i, ξ_j yields size ξ_i + ξ_j. Birth geometry `(ξ_i + ξ_j)^k`.
Default convention (preserves the library's historical behavior).
"""
struct MassBased <: AbstractCoord end

raw"""
    LengthBased <: AbstractCoord

Internal coordinate ξ is a length (e.g. diameter): volume ∝ ξ³ is conserved, so
aggregating ξ_i, ξ_j yields size `(ξ_i³ + ξ_j³)^(1/3)`. Birth geometry `(ξ_i³ + ξ_j³)^(k/3)`.
"""
struct LengthBased <: AbstractCoord end

raw"""
    aggregation_birth(coord, a1, a2, k)

k-th moment of the particle formed by aggregating two particles of internal
coordinate values `a1`, `a2`, under convention `coord`. Physics-independent
(the rate β is applied by the caller). Analog of OpenQBMM
`aggregationKernel::nodeSource` (length) / `massNodeSource` (mass).
"""
@inline aggregation_birth(::MassBased, a1, a2, k) = (a1 + a2)^k
@inline aggregation_birth(::LengthBased, a1, a2, k) = (a1^3 + a2^3)^(k / 3)

raw"""
    daughter_moment(d, k, xi, coord)

Protocol: k-th moment of the daughter distribution when a particle of value `xi`
breaks, under convention `coord`. Library daughter types extend this for both
`MassBased` and `LengthBased`. A plain `Function` `d` is accepted as a MassBased-convention
fragment-moment function `d(k, xi)` (backward compat); `LengthBased` with a plain
function errors — use a library daughter object instead.
"""
@inline daughter_moment(d::Function, k, xi, ::MassBased) = d(k, xi)
@inline function daughter_moment(d::Function, k, xi, ::LengthBased)
    return error(
        "daughter_moment: a plain-function daughter cannot infer its LengthBased form. " *
        "Use a library daughter distribution (e.g. SymmetricFragmentation()) instead."
    )
end
