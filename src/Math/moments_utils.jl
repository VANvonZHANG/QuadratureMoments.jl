# QBMM.jl/src/Math/moments_utils.jl

using StaticArrays
using ..QBMM: AbstractKernel, GaussianKernel, GammaKernel, BetaKernel

"""
    reconstruct_moment(nodes, weights, k, sigma, kernel) -> T

根据离散节点、权重和带宽，反构连续分布的第 k 阶矩。
"""
function reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::GaussianKernel) where {N, T}
    # E[X^k] for Gaussian kernel with mean x_i and variance sigma^2
    # sum_i w_i * sum_{j=0}^{floor(k/2)} (k! / (j!(k-2j)!)) * x_i^(k-2j) * (sigma^2/2)^j
    # 更简单的形式：利用正态分布的高阶原点矩公式
    res = zero(T)
    for i in 1:N
        m_k = zero(T)
        for j in 0:floor(Int, k/2)
            # 系数 c = (k! / (j! (k-2j)!)) * (1/2)^j
            c = one(T)
            for m in 0:(2j-1) c *= (k - m) end
            for m in 1:j c /= (2*m) end
            m_k += c * (nodes[i, 1]^(k-2j)) * σ^(2j)
        end
        res += weights[i] * m_k
    end
    return res
end

function reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::GammaKernel) where {N, T}
    # Gamma EQMOM: m_k = sum_i w_i * Prod_{j=0}^{k-1} (x_i + j * sigma)
    res = zero(T)
    for i in 1:N
        p = one(T)
        for j in 0:(k-1) p *= (nodes[i, 1] + j * σ) end
        res += weights[i] * p
    end
    return res
end

function reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::BetaKernel) where {N, T}
    # Beta EQMOM: m_k = (sum_i w_i * Prod_{j=0}^{k-1} (x_i + j*sigma)) / Prod_{j=0}^{k-1} (1 + j*sigma)
    res = zero(T)
    den = one(T)
    for j in 0:(k-1) den *= (1 + j * σ) end
    for i in 1:N
        p = one(T)
        for j in 0:(k-1) p *= (nodes[i, 1] + j * σ) end
        res += weights[i] * (p / den)
    end
    return res
end
