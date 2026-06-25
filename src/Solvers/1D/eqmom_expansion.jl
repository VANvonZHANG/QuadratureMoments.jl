# src/Solvers/1D/eqmom_expansion.jl

using StaticArrays
using FastGaussQuadrature
using ..QuadratureMoments: QuadratureResult
using ..QuadratureMoments: GaussianKernel, GammaKernel, BetaKernel

"""
    expand_quadrature(res::QuadratureResult, kernel::AbstractKernel, ::Val{M})

Expand an EQMOM `QuadratureResult` (continuous-kernel NDF with bandwidth `res.sigmas`)
into an `N*M`-node Dirac quadrature by replacing each primary node's kernel with its
`M`-point Gauss quadrature. Feed the result to `compute_source_terms` to make source
terms correctly σ-aware. Returns `(nodes::SVector{N*M}, weights::SVector{N*M})`.

If `res.sigmas` is `nothing` or `σ ≈ 0` (EQMOM collapsed to QMOM), the primary Dirac
quadrature is returned unchanged.
"""
function expand_quadrature end

function expand_quadrature(
    res::QuadratureResult{1,N,T}, ::GaussianKernel, ::Val{M}
) where {N,T,M}
    σ = res.sigmas === nothing ? zero(T) : res.sigmas[1]
    if abs(σ) < eps(T)
        return (
            SVector{N,T}(ntuple(i -> res.nodes[i, 1], Val(N))),
            res.weights,
        )
    end
    z, ν = FastGaussQuadrature.gausshermite(M)
    νsum = sum(ν)
    s2 = sqrt(2) * σ
    NT = N * M
    nodes = MVector{NT,T}(undef)
    weights = MVector{NT,T}(undef)
    for i in 1:N
        xi = res.nodes[i, 1]
        wi = res.weights[i]
        for j in 1:M
            idx = (i - 1) * M + j
            nodes[idx] = xi + s2 * z[j]
            weights[idx] = wi * ν[j] / νsum
        end
    end
    return SVector{NT,T}(nodes), SVector{NT,T}(weights)
end
