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
    compute_modified_moments(m::AbstractVector{T}, σ::T, ::GaussianKernel)

根据带宽 σ 计算修正矩 m^*，用于后续调用 Wheeler 算法。
输入前 2N 个矩 (m_0 到 m_{2N-1})，返回对应的 2N 个修正矩。
"""
function compute_modified_moments(m::AbstractVector{T}, σ::T, ::GaussianKernel) where T
    N2 = length(m)
    m_star = copy(m)
    for k in 2:N2
        term_sum = zero(T)
        for j in 1:floor(Int, (k-1)/2)
            coeff = factorial(k-1) / (factorial(j) * factorial(k-1-2j) * 2^j)
            term_sum += coeff * m_star[k-2j] * σ^(2j)
        end
        m_star[k] -= term_sum
    end
    return m_star
end

"""
    compute_modified_moments(m::AbstractVector{T}, σ::T, ::GammaKernel)

根据带宽 σ 计算 Gamma 核的修正矩。其展开关系基于第一类无符号斯特林数。
"""
function compute_modified_moments(m::AbstractVector{T}, σ::T, ::GammaKernel) where T
    L = length(m)
    m_star = copy(m)
    
    # 预计算第一类无符号斯特林数 S(k, j)
    S = zeros(T, L, L)
    S[1, 1] = 1
    for k in 2:L
        for j in 1:k
            if j == 1
                S[k, j] = (k - 1) * S[k-1, j]
            elseif j == k
                S[k, j] = 1
            else
                S[k, j] = S[k-1, j-1] + (k - 1) * S[k-1, j]
            end
        end
    end
    
    for k in 3:L
        k_true = k - 1
        term_sum = zero(T)
        for j in 2:k-1
            j_true = j - 1
            term_sum += S[k_true, j_true] * σ^(k_true - j_true) * m_star[j]
        end
        m_star[k] = m[k] - term_sum
    end
    return m_star
end

"""
    compute_modified_moments(m::AbstractVector{T}, σ::T, ::BetaKernel)

根据带宽 σ 计算 Beta 核的修正矩。Beta 核在解析上可以通过缩放后复用 Gamma 核的逻辑来求得。
"""
function compute_modified_moments(m::AbstractVector{T}, σ::T, ::BetaKernel) where T
    L = length(m)
    m_scaled = copy(m)
    
    # 预先缩放 Beta 矩以匹配 Gamma 核的形式: m'_k = m_k * Π_{i=0}^{k-1}(1 + i*σ)
    D = one(T)
    for k in 1:L
        k_true = k - 1
        if k_true > 0
            D *= (1 + (k_true - 1) * σ)
        end
        m_scaled[k] *= D
    end
    
    # 使用缩放后的矩调用 Gamma 核逻辑
    return compute_modified_moments(m_scaled, σ, GammaKernel())
end

"""
    reconstruct_moment(nodes, weights, k_true, σ, kernel)

根据反演的节点、权重和选定的核函数带宽 σ，重构出真实的第 k_true 阶矩（通常用来比对 2N 阶矩残差）。
"""
function reconstruct_moment(nodes::AbstractVector{T}, weights::AbstractVector{T}, k::Int, σ::T, ::GaussianKernel) where T
    pred = zero(T)
    N = length(nodes)
    for a in 1:N
        term_sum = zero(T)
        for j in 0:floor(Int, k/2)
            coeff = factorial(k) / (factorial(j) * factorial(k-2j) * 2^j)
            term_sum += coeff * (nodes[a]^(k-2j)) * σ^(2j)
        end
        pred += weights[a] * term_sum
    end
    return pred
end

function reconstruct_moment(nodes::AbstractVector{T}, weights::AbstractVector{T}, k::Int, σ::T, ::GammaKernel) where T
    pred = zero(T)
    N = length(nodes)
    for a in 1:N
        term_prod = one(T)
        for i in 0:k-1
            term_prod *= (nodes[a] + i * σ)
        end
        pred += weights[a] * term_prod
    end
    return pred
end

function reconstruct_moment(nodes::AbstractVector{T}, weights::AbstractVector{T}, k::Int, σ::T, ::BetaKernel) where T
    pred = zero(T)
    N = length(nodes)
    
    D = one(T)
    for i in 0:k-1
        D *= (1 + i * σ)
    end
    
    for a in 1:N
        term_prod = one(T)
        for i in 0:k-1
            term_prod *= (nodes[a] + i * σ)
        end
        pred += weights[a] * (term_prod / D)
    end
    return pred
end

"""
    invert_moments(method::EQMOM{N, GaussianKernel}, m::AbstractVector{T})

执行 EQMOM 反演过程：
输入 2N+1 个矩序列 (m_0 到 m_{2N})。
寻找最优带宽 σ 使得系统闭合。
"""
function invert_moments(method::EQMOM{N, K}, m::AbstractVector{T}) where {N, K, T}
    @assert length(m) >= 2N + 1 "EQMOM with N=$N requires at least $(2N+1) moments."
    
    # σ 的搜索范围通常从 0 到 sqrt(Var)
    m0, m1, m2 = m[1], m[2], m[3]
    var_sys = max(0.0, m2/m0 - (m1/m0)^2)
    # 对于 Beta 分布，σ 可以更大，增加一个保底的上限 1.0 以扩大搜索网络
    max_σ = max(sqrt(var_sys), 1.0)
    
    if max_σ <= 1e-12
        nodes, weights = wheeler_inversion(m[1:2N])
        return nodes, weights, 0.0
    end

    # 残差函数: R(σ) = m_2N - \sum w_a * \mu_{2N}(\xi_a, σ)
    function residual(σ)
        # 1. 基于当前 σ 获取前 2N 个修正矩，并进行 Wheeler 反演
        m_star = compute_modified_moments(m[1:2N], σ, method.kernel)
        
        nodes, weights = try
            wheeler_inversion(m_star)
        catch
            return NaN
        end
        
        # 2. 根据节点和权重重建第 2N 阶矩 (m[2N+1])
        pred_m2N = reconstruct_moment(nodes, weights, 2N, σ, method.kernel)
        
        return pred_m2N - m[2N+1]
    end

    # 在 [0, max_σ] 之间使用二分法或类似的容错算法寻找根
    # 为了避免反演引发 NaN 导致报错，使用简单的网格搜索结合二分
    # 找到第一次发生符号变化的地方
    num_intervals = 20
    dσ = max_σ / num_intervals
    
    root_σ = 0.0
    for i in 0:(num_intervals-1)
        s1, s2 = i*dσ, (i+1)*dσ
        r1, r2 = residual(s1), residual(s2)
        if !isnan(r1) && !isnan(r2) && r1 * r2 <= 0
            # 找到符号交叉点，交给 find_zero
            root_σ = find_zero(residual, (s1, s2), Bisection())
            break
        end
    end
    
    if root_σ == 0.0
        # 默认回退（如果没有找到，返回退化解）
        nodes, weights = wheeler_inversion(m[1:2N])
        return nodes, weights, 0.0
    end

    # 使用求得的 root_σ 进行最终反演
    m_star_final = compute_modified_moments(m[1:2N], root_σ, method.kernel)
    nodes, weights = wheeler_inversion(m_star_final)
    
    return nodes, weights, root_σ
end
