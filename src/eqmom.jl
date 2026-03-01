using Roots
using LinearAlgebra
using StaticArrays

# 抽象核函数类型
abstract type AbstractKernel end
struct GaussianKernel <: AbstractKernel end
struct GammaKernel <: AbstractKernel end
struct BetaKernel <: AbstractKernel end

# EQMOM 结构体
struct EQMOM{N, K<:AbstractKernel} <: AbstractQBMM 
    kernel::K
end

EQMOM(N::Int) = EQMOM{N, GaussianKernel}(GaussianKernel())
EQMOM(N::Int, kernel::K) where K<:AbstractKernel = EQMOM{N, K}(kernel)

"""
    compute_modified_moments(m::SVector{L, T}, σ::T, ::GaussianKernel)
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::GaussianKernel) where {L, T}
    m_star = MVector{L, T}(m)
    for k in 2:L
        k_idx = k - 1
        term_sum = zero(T)
        for j in 1:floor(Int, k_idx/2)
            c = one(T)
            for i in 0:(2j-1)
                c *= (k_idx - i)
            end
            for i in 1:j
                c /= (2*i)
            end
            term_sum += c * m_star[k-2j] * σ^(2j)
        end
        m_star[k] -= term_sum
    end
    return SVector{L, T}(m_star)
end

"""
    compute_modified_moments(m::SVector{L, T}, σ::T, ::GammaKernel)
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::GammaKernel) where {L, T}
    m_star_res = MVector{L, T}(m)
    for k in 2:L
        k_val = k - 1
        val = zero(T)
        for j in 1:k
            j_val = j - 1
            s_coeff = get_stirling1(k_val, j_val)
            val += s_coeff * (σ^(k_val - j_val)) * m[j]
        end
        m_star_res[k] = val
    end
    return SVector{L, T}(m_star_res)
end

function get_stirling1(n::Int, k::Int)
    if k < 0 || k > n return 0.0 end
    if n == 0 return 1.0 end
    return _stirling1_recursive(n, k)
end

function _stirling1_recursive(n::Int, k::Int)
    if k == n return 1.0 end
    if k == 0 || k > n return 0.0 end
    # 简单的递归，对于小 n (n < 12) 性能尚可
    return _stirling1_recursive(n-1, k-1) - (n-1) * _stirling1_recursive(n-1, k)
end

"""
    compute_modified_moments(m::SVector{L, T}, σ::T, ::BetaKernel)
"""
function compute_modified_moments(m::SVector{L, T}, σ::T, ::BetaKernel) where {L, T}
    m_prime = MVector{L, T}(m)
    for k in 2:L
        k_val = k - 1
        prod_val = one(T)
        for j in 0:(k_val-1)
            prod_val *= (1 + j * σ)
        end
        m_prime[k] *= prod_val
    end
    return compute_modified_moments(SVector{L, T}(m_prime), σ, GammaKernel())
end

function reconstruct_moment(nodes::SVector{N, T}, weights::SVector{N, T}, k::Int, σ::T, ::GaussianKernel) where {N, T}
    res = zero(T)
    for i in 1:N
        m_k = zero(T)
        for j in 0:floor(Int, k/2)
            c = one(T)
            for m in 0:(2j-1) c *= (k - m) end
            for m in 1:j c /= (2*m) end
            m_k += c * (nodes[i]^(k-2j)) * σ^(2j)
        end
        res += weights[i] * m_k
    end
    return res
end

function reconstruct_moment(nodes::SVector{N, T}, weights::SVector{N, T}, k::Int, σ::T, ::GammaKernel) where {N, T}
    res = zero(T)
    for i in 1:N
        p = one(T)
        for j in 0:(k-1) p *= (nodes[i] + j * σ) end
        res += weights[i] * p
    end
    return res
end

function reconstruct_moment(nodes::SVector{N, T}, weights::SVector{N, T}, k::Int, σ::T, ::BetaKernel) where {N, T}
    res = zero(T)
    den = one(T)
    for j in 0:(k-1) den *= (1 + j * σ) end
    for i in 1:N
        p = one(T)
        for j in 0:(k-1) p *= (nodes[i] + j * σ) end
        res += weights[i] * (p / den)
    end
    return res
end

function invert_moments(method::EQMOM{N, K}, m::SVector{L, T}) where {N, K, T, L}
    @assert L >= 2N + 1 "EQMOM requires 2N+1 moments"
    
    m0, m1, m2 = m[1], m[2], m[3]
    var_total = max(1e-12, m2/m0 - (m1/m0)^2)
    σ_max = sqrt(var_total)
    
    function residual(σ)
        if σ < 0.0 return -1e10 end
        if σ > σ_max return 1e10 end
        
        m_slice = SVector{2N, T}(m[1:2N])
        m_star = compute_modified_moments(m_slice, σ, method.kernel)
        
        # 尝试反演并检查有效性
        nodes, weights = try
            res_nodes, res_weights = wheeler_inversion(m_star)
            # 检查是否有 NaN 或权重是否全为正
            if any(isnan, res_nodes) || any(isnan, res_weights) || any(w < -1e-10 for w in res_weights)
                return 1e10
            end
            res_nodes, res_weights
        catch
            return 1e10
        end
        
        m2N_pred = reconstruct_moment(nodes, weights, 2N, σ, method.kernel)
        return m2N_pred - m[2N+1]
    end

    # 寻找根
    σ_opt = 0.0
    # 检查边界
    r0 = residual(0.0)
    rmax = residual(σ_max)
    
    if isnan(r0) || isnan(rmax) || r0 * rmax > 0
        # 如果边界没有变号，尝试在中间寻找变号区间
        # 因为 realizability 边界可能小于 σ_max
        found = false
        n_steps = 50
        for i in 1:n_steps
            s1 = (i-1) * σ_max / n_steps
            s2 = i * σ_max / n_steps
            rs1 = residual(s1)
            rs2 = residual(s2)
            if rs1 * rs2 <= 0 && rs1 < 1e9 && rs2 < 1e9
                σ_opt = find_zero(residual, (s1, s2), Bisection())
                found = true
                break
            end
        end
        if !found
            # 尝试回退到 QMOM (σ=0)
            σ_opt = 0.0
        end
    else
        σ_opt = find_zero(residual, (0.0, σ_max), Bisection())
    end
    
    m_slice_final = SVector{2N, T}(m[1:2N])
    m_star_final = compute_modified_moments(m_slice_final, σ_opt, method.kernel)
    nodes, weights = wheeler_inversion(m_star_final)
    return nodes, weights, σ_opt
end
