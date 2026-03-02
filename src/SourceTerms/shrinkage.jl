# src/SourceTerms/shrinkage.jl

using StaticArrays
using ..QBMM: AbstractSourceTerm

raw"""
    ParticleShrinkage{F, D} <: AbstractSourceTerm

A source term representing continuous particle shrinkage (e.g., evaporation, dissolution).

Unlike `ParticleGrowth` where particles strictly increase in size, shrinkage can cause 
particles to reach a size of zero and completely disappear from the system. This finite-time 
truncation results in a non-zero, negative source term for the total number density (\$S_0 < 0\$).

Because standard QMOM represents the distribution as a sum of Dirac delta functions, 
the exact evaluation of the boundary flux \$G(0)n(0)\$ is mathematically ill-posed. 
Therefore, this implementation requires the user to provide a specific closure model 
for the disappearance rate.

# Type Parameters
- `F`: The type of the rate function.
- `D`: The type of the zero-flux closure function.

# Fields
- `rate_func::F`: A function `G(xi)` that returns the continuous shrinkage rate (\$d\xi/dt < 0\$) of a particle of size `xi`.
- `zero_flux_func::D`: A function `f(nodes, weights)` that returns the rate of change of the total particle number (\$S_0 = dm_0/dt \le 0\$).
raw"""
struct ParticleShrinkage{F, D} <: AbstractSourceTerm
    rate_func::F
    zero_flux_func::D
end

raw"""
    compute_source_terms(shrink::ParticleShrinkage, nodes, weights, ::Val{L})

Compute the moment source terms due to continuous particle shrinkage.
- For \$k = 0\$: \$S_0\$ is computed using `zero_flux_func(nodes, weights)`.
- For \$k \ge 1\$: \$S_k = \sum_{i=1}^N w_i k \xi_i^{k-1} G(\xi_i)\$.
raw"""
function compute_source_terms(
    shrink::ParticleShrinkage,
    nodes::SVector{N, T},
    weights::SVector{N, T},
    ::Val{L},
) where {N, T, L}
    return SVector{L, T}(ntuple(Val(L)) do idx
        k = idx - 1
        if k == 0
            # S_0 is determined by the specific boundary flux closure model
            return T(shrink.zero_flux_func(nodes, weights))
        else
            val = zero(T)
            for i in 1:N
                val += weights[i] * k * (nodes[i]^(k - 1)) * shrink.rate_func(nodes[i])
            end
            return val
        end
    end)
end
