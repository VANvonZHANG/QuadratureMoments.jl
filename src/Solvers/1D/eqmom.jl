# src/Solvers/1D/eqmom.jl
using Roots
using LinearAlgebra
using StaticArrays

# --- Kernel Definitions ---
abstract type AbstractKernel end
struct GaussianKernel <: AbstractKernel end
struct GammaKernel <: AbstractKernel end
struct BetaKernel <: AbstractKernel end

"""
    EQMOM{D, N, K} <: AbstractQBMM{1, N}
"""
struct EQMOM{N, K<:AbstractKernel} <: AbstractQBMM{1, N}
    kernel::K
end

EQMOM(N::Int) = EQMOM{N, GaussianKernel}(GaussianKernel())
EQMOM(N::Int, kernel::K) where K<:AbstractKernel = EQMOM{N, K}(kernel)

# --- Modified Moments Computation ---

"""
    compute_modified_moments(m, σ, ::GaussianKernel, backend)
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::GaussianKernel, backend::AbstractMathBackend) where {L, T}
    m_star = MVector{L, T}(m)
    for k in 1:L
        k_idx = k - 1
        if k_idx < 2 continue end
        term_sum = zero(T)
        for j in 1:floor(Int, k_idx/2)
            # Gaussian modified moment coefficients
            c = one(T)
            for i in 0:(2j-1) c *= (k_idx - i) end
            for i in 1:j c /= (2*i) end
            term_sum += c * m_star[k-2j] * σ^(2j)
        end
        m_star[k] -= term_sum
    end
    return SVector{L, T}(m_star)
end

"""
    compute_modified_moments(m, σ, ::GammaKernel, backend)
    Uses Stirling numbers of the second kind.
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::GammaKernel, backend::AbstractMathBackend) where {L, T}
    m_star = MVector{L, T}(undef)
    for k in 1:L
        k_val = k - 1
        val = zero(T)
        for j in 1:k
            j_val = j - 1
            # 这里的修正逻辑应遵循: m_star_k = sum_{j=0}^k S(k, j) * sigma^(k-j) * m_j
            # 我们使用 Math/stirling.jl 中的实现
            s_coeff = stirling2(k_val, j_val, backend)
            val += s_coeff * (σ^(k_val - j_val)) * m[j]
        end
        m_star[k] = val
    end
    return SVector{L, T}(m_star)
end

"""
    compute_modified_moments(m, σ, ::BetaKernel, backend)
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::BetaKernel, backend::AbstractMathBackend) where {L, T}
    m_prime = MVector{L, T}(m)
    for k in 2:L
        k_val = k - 1
        prod_val = one(T)
        for j in 0:(k_val-1)
            prod_val *= (1 + j * σ)
        end
        m_prime[k] *= prod_val
    end
    return compute_modified_moments(SVector{L, T}(m_prime), σ, GammaKernel(), backend)
end

# --- Inversion Implementation ---

function invert_moments(
    method::EQMOM{N, K}, 
    m::SVector{L, T}; 
    backend::AbstractMathBackend = NativeBackend()
) where {N, K, T, L}
    
    @assert L >= 2N + 1 "EQMOM requires at least 2N+1 moments"
    
    # 1. Estimate σ_max based on variance
    m0, m1, m2 = m[1], m[2], m[3]
    var_total = max(eps(T), m2/m0 - (m1/m0)^2)
    σ_max = sqrt(var_total)
    
    # 2. Optimization target: find σ such that m[2N+1] is matched
    function residual(σ)
        if σ < 0.0 || σ > σ_max return 1e10 end
        
        m_slice = SVector{2N, T}(m[1:2N])
        m_star = compute_modified_moments(m_slice, σ, method.kernel, backend)
        
        # 调用基础 Wheeler 反演 (内部应使用 backend)
        # 注意：这里需要 Wheeler 支持带 backend 的调用
        res = invert_moments(Wheeler{N}(), m_star; backend=backend)
        
        # 重构第 2N+1 阶矩进行对比
        # 这里的 reconstruct_moment 逻辑暂时保留在内部或迁移至 Math
        m2N_pred = _reconstruct_moment(res.nodes, res.weights, 2N, σ, method.kernel)
        return m2N_pred - m[2N+1]
    end

    # 3. Solve for σ
    σ_opt = zero(T)
    try
        r0 = residual(0.0)
        rmax = residual(σ_max)
        if abs(rmax) < 1e-10
            σ_opt = σ_max
        elseif r0 * rmax <= 0
            σ_opt = find_zero(residual, (0.0, σ_max), Bisection())
        else
            # 启发式步进
            σ_opt = 0.0 # Fallback
        end
    catch
        σ_opt = 0.0
    end
    
    # 4. Final inversion with σ_opt
    m_slice_final = SVector{2N, T}(m[1:2N])
    m_star_final = compute_modified_moments(m_slice_final, σ_opt, method.kernel, backend)
    res_final = invert_moments(Wheeler{N}(), m_star_final; backend=backend)
    
    # 构造带宽矩阵 (N x 1)
    sigmas = SMatrix{N, 1, T}(fill(σ_opt, N))
    
    return QuadratureResult(res_final.weights, res_final.nodes, sigmas)
end

# 内部辅助：重构矩 (1D)
function _reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::GaussianKernel) where {N, T}
    res = zero(T)
    for i in 1:N
        m_k = zero(T)
        for j in 0:floor(Int, k/2)
            c = one(T)
            for m in 0:(2j-1) c *= (k - m) end
            for m in 1:j c /= (2*m) end
            m_k += c * (nodes[i, 1]^(k-2j)) * σ^(2j)
        end
        res += weights[i] * m_k
    end
    return res
end

function _reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::GammaKernel) where {N, T}
    res = zero(T)
    for i in 1:N
        p = one(T)
        for j in 0:(k-1) p *= (nodes[i, 1] + j * σ) end
        res += weights[i] * p
    end
    return res
end

function _reconstruct_moment(nodes::SMatrix{N, 1, T}, weights::SVector{N, T}, k::Int, σ::T, ::BetaKernel) where {N, T}
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
