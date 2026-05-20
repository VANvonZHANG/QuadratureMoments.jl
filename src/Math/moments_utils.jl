# src/Math/moments_utils.jl

using StaticArrays
using ..QuadratureMoments: AbstractKernel, GaussianKernel, GammaKernel, BetaKernel

raw"""
    reconstruct_moment(nodes, weights, k, sigma, kernel) -> Float64

Reconstruct the \$k\$-th order moment from an EQMOM continuous distribution.

# Arguments
- `nodes::SMatrix`: The primary quadrature nodes \$\\xi_i\$.
- `weights::SVector`: The quadrature weights \$w_i\$.
- `k::Int`: The order of the moment to reconstruct.
- `sigma`: The bandwidth parameter \$\\sigma\$.
- `kernel::AbstractKernel`: The continuous kernel type (e.g., `GaussianKernel()`).

# Returns
- The reconstructed moment of type `T`.
raw"""
function reconstruct_moment(
    nodes::SMatrix{N,1,T}, weights::SVector{N,T}, k::Int, σ::T, ::GaussianKernel
) where {N,T}
    # E[X^k] for Gaussian kernel with mean x_i and variance sigma^2
    # sum_i w_i * sum_{j=0}^{floor(k/2)} (k! / (j!(k-2j)!)) * x_i^(k-2j) * (sigma^2/2)^j
    res = zero(T)
    for i in 1:N
        m_k = zero(T)
        for j in 0:floor(Int, k / 2)
            # Coefficient c = (k! / (j! (k-2j)!)) * (1/2)^j
            c = one(T)
            for m in 0:(2j - 1)
                c *= (k - m)
            end
            for m in 1:j
                c /= (2 * m)
            end
            m_k += c * (nodes[i, 1]^(k - 2j)) * σ^(2j)
        end
        res += weights[i] * m_k
    end
    return res
end

function reconstruct_moment(
    nodes::SMatrix{N,1,T}, weights::SVector{N,T}, k::Int, σ::T, ::GammaKernel
) where {N,T}
    # Gamma EQMOM: m_k = sum_i w_i * Prod_{j=0}^{k-1} (x_i + j * sigma)
    res = zero(T)
    for i in 1:N
        p = one(T)
        for j in 0:(k - 1)
            p *= (nodes[i, 1] + j * σ)
        end
        res += weights[i] * p
    end
    return res
end

function reconstruct_moment(
    nodes::SMatrix{N,1,T}, weights::SVector{N,T}, k::Int, σ::T, ::BetaKernel
) where {N,T}
    # Beta EQMOM: m_k = (sum_i w_i * Prod_{j=0}^{k-1} (x_i + j*sigma)) / Prod_{j=0}^{k-1} (1 + j*sigma)
    res = zero(T)
    den = one(T)
    for j in 0:(k - 1)
        den *= (1 + j * σ)
    end
    for i in 1:N
        p = one(T)
        for j in 0:(k - 1)
            p *= (nodes[i, 1] + j * σ)
        end
        res += weights[i] * (p / den)
    end
    return res
end
