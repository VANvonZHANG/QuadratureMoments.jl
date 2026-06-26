# src/Physics/daughter_distributions.jl
# Concrete daughter distributions. Each extends daughter_moment(d, k, xi, coord)
# for both MassBased and LengthBased. Transcribed verbatim from OpenQBMM:
#   MassBased   <- daughterDistribution::mDMass(order, abscissa)
#   LengthBased <- daughterDistribution::mD(order, abscissa)   (length via pow3)
using ..QuadratureMoments: daughter_moment, MassBased, LengthBased

raw"""Symmetric binary breakage: two equal daughters."""
struct Symmetric end
@inline daughter_moment(::Symmetric, k, xi, ::MassBased)   = 2^(1 - k) * xi^k
@inline daughter_moment(::Symmetric, k, xi, ::LengthBased) = 2^((3 - k) / 3) * xi^k

raw"""Uniform daughter distribution."""
struct Uniform end
@inline daughter_moment(::Uniform, k, xi, ::MassBased)   = 2 * xi^k / (k + 1)
@inline daughter_moment(::Uniform, k, xi, ::LengthBased) = 6 * xi^k / (k + 3)

raw"""Binary breakage with one-quarter mass ratio (daughters split 1:4 by mass)."""
struct OneQuarterMassRatio end
@inline daughter_moment(::OneQuarterMassRatio, k, xi, ::MassBased)   = (4^k + 1) * xi^k / 5^k
@inline daughter_moment(::OneQuarterMassRatio, k, xi, ::LengthBased) = (4^(k / 3) + 1) * xi^k / 5^(k / 3)

raw"""Erosion: one large fragment of size d0 plus eroded material (parameter d0)."""
struct Erosion{T}
    d0::T
end
@inline daughter_moment(e::Erosion, k, xi, ::MassBased)   = e.d0^k + (xi - e.d0)^k
@inline daughter_moment(e::Erosion, k, xi, ::LengthBased) = e.d0^k + (xi^3 - e.d0^3)^(k / 3)

raw"""Full fragmentation: daughters of size d0, count scaled by volume/mass ratio (parameter d0)."""
struct FullFragmentation{T}
    d0::T
end
@inline daughter_moment(f::FullFragmentation, k, xi, ::MassBased)   = (xi / f.d0) * f.d0^k
@inline daughter_moment(f::FullFragmentation, k, xi, ::LengthBased) = (xi / f.d0)^3 * f.d0^k
