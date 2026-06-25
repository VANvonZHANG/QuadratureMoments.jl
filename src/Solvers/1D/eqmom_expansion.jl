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
        return (SVector{N,T}(ntuple(i -> res.nodes[i, 1], Val(N))), res.weights)
    end
    z, ν = FastGaussQuadrature.gausshermite(M)
    νsum = sum(ν)
    s2 = sqrt(2) * σ
    NT = N * M
    nodes = MVector{NT,T}(undef)
    weights = MVector{NT,T}(undef)
    for i in 1:N
        if iszero(res.weights[i])
            for j in 1:M
                idx = (i - 1) * M + j
                nodes[idx] = zero(T)
                weights[idx] = zero(T)
            end
            continue
        end
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

# Beta kernel: per primary node i, use an M-point Gauss-Jacobi rule on [-1,1]
# and map nodes to [0,1] via u = (1+t)/2. The beta shape parameters are
# a_i = xi_i/sigma and b_i = (1-xi_i)/sigma. FastGaussQuadrature.gaussjacobi
# integrates int_{-1}^1 f(t) (1-t)^alpha (1+t)^beta dt. Substituting t = 2u-1:
#   (1-t) = 2(1-u), (1+t) = 2u  =>  weight = 2^(alpha+beta) (1-u)^alpha u^beta
# To obtain the Beta(a,b) PDF shape u^(a-1) (1-u)^(b-1), pass alpha = b-1,
# beta = a-1 (arguments swapped relative to the shape parameters). The constant
# factor 2^(a+b-2) cancels when normalizing weights by sum(kappa). Secondary
# nodes u_j are in [0,1]; weights are w_i*kappa_j/sum(kappa) (sum(kappa) equals
# 2^(a+b-1) B(a,b), so no SpecialFunctions dep is needed).
function expand_quadrature(
    res::QuadratureResult{1,N,T}, ::BetaKernel, ::Val{M}
) where {N,T,M}
    σ = res.sigmas === nothing ? zero(T) : res.sigmas[1]
    if abs(σ) < eps(T)
        return (SVector{N,T}(ntuple(i -> res.nodes[i, 1], Val(N))), res.weights)
    end
    NT = N * M
    nodes = MVector{NT,T}(undef)
    weights = MVector{NT,T}(undef)
    for i in 1:N
        if iszero(res.weights[i])
            for j in 1:M
                idx = (i - 1) * M + j
                nodes[idx] = zero(T)
                weights[idx] = zero(T)
            end
            continue
        end
        xi = res.nodes[i, 1]
        a = xi / σ                  # beta shape a_i = xi_i/sigma
        b = (1 - xi) / σ            # beta shape b_i = (1 - xi_i)/sigma
        # gaussjacobi on [-1,1]; pass (b-1, a-1) so the [0,1] weight is u^(a-1)(1-u)^(b-1).
        t, κ = FastGaussQuadrature.gaussjacobi(M, b - 1, a - 1)
        u = (1 .+ t) ./ 2           # map Jacobi nodes from [-1,1] to [0,1] (beta support)
        κsum = sum(κ)               # = 2^(a+b-1) B(a,b); normalizing avoids SpecialFunctions
        wi = res.weights[i]
        for j in 1:M
            idx = (i - 1) * M + j
            nodes[idx] = u[j]
            weights[idx] = wi * κ[j] / κsum
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
        return (SVector{N,T}(ntuple(i -> res.nodes[i, 1], Val(N))), res.weights)
    end
    NT = N * M
    nodes = MVector{NT,T}(undef)
    weights = MVector{NT,T}(undef)
    for i in 1:N
        if iszero(res.weights[i])
            for j in 1:M
                idx = (i - 1) * M + j
                nodes[idx] = zero(T)
                weights[idx] = zero(T)
            end
            continue
        end
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
