using LinearAlgebra
using StaticArrays

"""
    is_realizable(m::AbstractVector{T}; domain=:pos) -> Bool

检查矩序列的实现性 (Realizability)。
domain: 
  - :all 表示 (-∞, ∞) [Hamburger moment problem]
  - :pos 表示 [0, ∞) [Stieltjes moment problem] (默认)
"""
function is_realizable(m::AbstractVector{T}; domain=:pos) where T
    L = length(m)
    if L < 2 return true end
    
    # --- 1. Hamburger Condition: H_n is positive semi-definite ---
    # n 是使得 2n <= L-1 的最大整数
    n = (L - 1) ÷ 2
    H = zeros(T, n+1, n+1)
    for i in 0:n, j in 0:n
        H[i+1, j+1] = m[i+j+1]
    end
    
    vals = eigvals(Symmetric(H))
    if any(vals .< -sqrt(eps(T)))
        return false
    end
    
    # 如果只检查全轴分布，到此为止
    if domain == :all
        return true
    end
    
    # --- 2. Stieltjes Condition: H_n^(1) is positive semi-definite ---
    # n_s 是使得 2n_s + 1 <= L-1 的最大整数
    n_s = (L - 2) ÷ 2
    H1 = zeros(T, n_s+1, n_s+1)
    for i in 0:n_s, j in 0:n_s
        H1[i+1, j+1] = m[i+j+2]
    end
    
    vals1 = eigvals(Symmetric(H1))
    if any(vals1 .< -sqrt(eps(T)))
        return false
    end
    
    return true
end

"""
    is_realizable(m::SVector{L, T}; domain=:pos)
"""
function is_realizable(m::SVector{L, T}; domain=:pos) where {L, T}
    return is_realizable(collect(m); domain=domain)
end
