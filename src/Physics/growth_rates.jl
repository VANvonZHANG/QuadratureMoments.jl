# src/Physics/growth_rates.jl
# Concrete growth-rate callables G(xi). Coordinate-agnostic — ParticleGrowth has
# no coord parameter (its moment source S_k = Σ w_i k xi_i^{k-1} G(xi_i) is a
# chain rule independent of coord convention).

raw"""Constant growth rate `G(xi) = G0`."""
struct ConstantGrowth{T}
    G0::T
end
(g::ConstantGrowth)(xi) = g.G0

raw"""Linear growth rate `G(xi) = k0 * xi`."""
struct LinearGrowth{T}
    k0::T
end
(g::LinearGrowth)(xi) = g.k0 * xi
