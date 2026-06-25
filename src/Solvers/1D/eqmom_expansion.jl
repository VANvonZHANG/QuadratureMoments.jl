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

# Gamma kernel: per primary node i, use an M-point Gauss-Laguerre rule with
# shape alpha_i = xi_i/sigma. Secondary nodes are sigma*t_j (rescaling the
# Laguerre nodes to the gamma kernel's scale), and weights are w_i*omega_j
# normalized by sum(omega) = Gamma(alpha_i) so no SpecialFunctions dep is needed.
function expand_quadrature(
    res::QuadratureResult{1,N,T}, ::GammaKernel, ::Val{M}
) where {N,T,M}
    σ = res.sigmas === nothing ? zero(T) : res.sigmas[1]
    if abs(σ) < eps(T)
        return (
            SVector{N,T}(ntuple(i -> res.nodes[i, 1], Val(N))),
            res.weights,
        )
    end
    NT = N * M
    nodes = MVector{NT,T}(undef)
    weights = MVector{NT,T}(undef)
    for i in 1:N
        α = res.nodes[i, 1] / σ          # gamma shape parameter for primary node i
        t, ω = FastGaussQuadrature.gausslaguerre(M, α - 1)   # per-primary-node rule
        ωsum = sum(ω)                    # = Gamma(α); normalizing avoids needing SpecialFunctions
        wi = res.weights[i]
        for j in 1:M
            idx = (i - 1) * M + j
            nodes[idx] = σ * t[j]
            weights[idx] = wi * ω[j] / ωsum
        end
    end
    return SVector{NT,T}(nodes), SVector{NT,T}(weights)
end
