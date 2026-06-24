# src/Solvers/1D/eqmom.jl

using Roots
using LinearAlgebra
using StaticArrays
using ..QuadratureMoments:
    AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend, QuadratureResult
using ..QuadratureMoments: AbstractKernel, GaussianKernel, GammaKernel, BetaKernel
using ..QuadratureMoments: stirling2, reconstruct_moment, Wheeler

raw"""
    EQMOM{N, K} <: AbstractQBMM{1, N}

Extended Quadrature-Based Moment Method (1D).

Instead of using a sum of Dirac delta functions, EQMOM approximates the Number 
Density Function (NDF) using a set of non-negative continuous kernel functions 
(e.g., Gaussian, Gamma) with a bandwidth parameter \$\\sigma\$.

# Type Parameters
- `N::Int`: Number of primary quadrature nodes.
- `K<:AbstractKernel`: The continuous kernel type.
raw"""
struct EQMOM{N,K<:AbstractKernel} <: AbstractQBMM{1,N}
    kernel::K
end

raw"""
    EQMOM(N::Int)
    EQMOM(N::Int, kernel::AbstractKernel)

Constructors for the EQMOM solver. Defaults to `GaussianKernel()`.
raw"""
EQMOM(N::Int) = EQMOM{N,GaussianKernel}(GaussianKernel())
EQMOM(N::Int, kernel::K) where {K<:AbstractKernel} = EQMOM{N,K}(kernel)

# --- Modified Moments API ---

raw"""
    compute_modified_moments(m, σ, kernel, backend) -> SVector

Compute the modified moments \$m^*_k\$ by mapping raw moments into the orthogonal 
space of the specified kernel.

# Arguments
- `m::SVector`: The raw moment sequence.
- `σ::Float64`: The bandwidth parameter.
- `kernel::AbstractKernel`: The kernel type.
- `backend`: Mathematical backend to use.

# Returns
- A vector of modified moments.
raw"""
function compute_modified_moments(
    m::SVector{L,T}, σ::T, ::GaussianKernel, ::AbstractMathBackend
) where {L,T}
    m_star = MVector{L,T}(m)
    for k in 3:L # Index 1=m0, 2=m1, 3=m2
        k_val = k - 1
        term_sum = zero(T)
        for j in 1:floor(Int, k_val / 2)
            c = one(T)
            for m_idx in 0:(2j - 1)
                c *= (k_val - m_idx)
            end
            for m_idx in 1:j
                c /= (2 * m_idx)
            end
            term_sum += c * m_star[k - 2j] * σ^(2j)
        end
        m_star[k] -= term_sum
    end
    return SVector{L,T}(m_star)
end

function compute_modified_moments(
    m::SVector{L,T}, σ::T, ::GammaKernel, backend::AbstractMathBackend
) where {L,T}
    # Inverse gamma-kernel transform (moments -> modified moments).
    # Forward (reconstruct_moment) is m_k = sum_i w_i * prod_{r=0}^{k-1}(xi_i+r*sigma),
    # whose coefficients are UNSIGNED Stirling-first-kind. Its matrix-inverse gives
    # m*_k = sum_{j=0}^k (-1)^(k+j) * S(k,j) * sigma^(k-j) * m_j   (S = Stirling 2nd kind).
    # Matches OpenQBMM gammaEQMOM::momentsToMomentsStar.
    m_star = MVector{L,T}(undef)
    for k in 1:L
        k_val = k - 1
        val = zero(T)
        for j in 1:k
            j_val = j - 1
            s_coeff = stirling2(k_val, j_val, backend)
            val += (-1)^(k_val + j_val) * s_coeff * (σ^(k_val - j_val)) * m[j]
        end
        m_star[k] = val
    end
    return SVector{L,T}(m_star)
end

function compute_modified_moments(
    m::SVector{L,T}, σ::T, ::BetaKernel, backend::AbstractMathBackend
) where {L,T}
    m_prime = MVector{L,T}(m)
    for k in 2:L
        k_val = k - 1
        prod_val = one(T)
        for j in 0:(k_val - 1)
            prod_val *= (1 + j * σ)
        end
        m_prime[k] *= prod_val
    end
    return compute_modified_moments(SVector{L,T}(m_prime), σ, GammaKernel(), backend)
end

# --- Main API ---

raw"""
    invert_moments(method::EQMOM, m; backend=NativeBackend()) -> QuadratureResult

Perform 1D EQMOM inversion.

The algorithm searches for an optimal \$\\sigma \\in [0, \\sigma_{max}]\$ such that 
the reconstruction of the \$(2N+1)\$-th moment matches the target exactly.

# Arguments
- `method::EQMOM`: The EQMOM solver instance.
- `m::SVector`: Vector of at least \$2N+1\$ moments.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A `QuadratureResult` containing weights, primary nodes, and \$\\sigma\$ parameters.
raw"""
# Per-kernel upper bound on sigma for the bracketing search.
# Gaussian: std dev. Gamma: var/mean. Beta: p2/(1-p2).
# See OpenQBMM gammaEQMOM::sigmaMax and betaEQMOM::sigmaMax.
@inline _sigma_max(m0::T, m1::T, m2::T, ::GaussianKernel) where {T} =
    sqrt(max(eps(T), m2 / m0 - (m1 / m0)^2))

@inline _sigma_max(m0::T, m1::T, m2::T, ::GammaKernel) where {T} =
    max(zero(T), (m2 / m0 - (m1 / m0)^2) / (m1 / m0))

@inline function _sigma_max(m0::T, m1::T, m2::T, ::BetaKernel) where {T}
    num = m0 * m2 - m1 * m1
    den = m0 * m1 - m1 * m1
    den <= zero(T) && return zero(T)
    p2 = num / den
    p2 >= one(T) && return oftype(p2, 1e6)   # degenerate; let the search fall back to 0
    return p2 / (one(T) - p2)
end

function invert_moments(
    method::EQMOM{N,K}, m::SVector{L,T}; backend::AbstractMathBackend=NativeBackend()
) where {N,K,T,L}
    @assert L >= 2N + 1 "EQMOM requires at least 2N+1 moments"

    # 1. Per-kernel upper bound for sigma (OpenQBMM-matched formulas)
    m0, m1, m2 = m[1], m[2], m[3]
    σ_max = _sigma_max(m0, m1, m2, method.kernel)

    # 2. Optimization target
    function residual(σ)
        if σ < 0.0 || σ > σ_max
            return 1e10
        end

        m_slice = SVector{2N,T}(m[1:(2N)])
        m_star = compute_modified_moments(m_slice, σ, method.kernel, backend)

        # Internal Wheeler Inversion
        res = invert_moments(Wheeler{N}(), m_star; backend=backend)

        # Predict m[2N+1]
        m2N_pred = reconstruct_moment(res.nodes, res.weights, 2N, σ, method.kernel)
        return m2N_pred - m[2N + 1]
    end

    # 3. Solver
    σ_opt = zero(T)
    try
        r0 = residual(0.0)
        rmax = residual(σ_max)
        if abs(rmax) < 1e-10
            σ_opt = σ_max
        elseif r0 * rmax <= 0
            σ_opt = find_zero(residual, (0.0, σ_max), Bisection())
        else
            σ_opt = abs(r0) < abs(rmax) ? 0.0 : σ_max
        end
    catch
        σ_opt = 0.0
    end

    # 4. Final Construct
    m_slice_final = SVector{2N,T}(m[1:(2N)])
    m_star_final = compute_modified_moments(m_slice_final, σ_opt, method.kernel, backend)
    res_final = invert_moments(Wheeler{N}(), m_star_final; backend=backend)

    sigmas = SMatrix{N,1,T}(fill(σ_opt, N))
    return QuadratureResult(res_final.weights, res_final.nodes, sigmas)
end
