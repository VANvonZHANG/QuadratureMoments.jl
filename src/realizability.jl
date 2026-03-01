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
    # n+1 = (L+1) ÷ 2
    N1 = (L + 1) ÷ 2
    H = zeros(T, N1, N1)
    @inbounds for j in 1:N1, i in 1:N1
        H[i, j] = m[i+j-1]
    end
    
    vals = eigvals(Symmetric(H))
    if any(vals .< -sqrt(eps(T)))
        return false
    end
    
    if domain == :all
        return true
    end
    
    # --- 2. Stieltjes Condition: H_n^(1) is positive semi-definite ---
    # n+1 = L ÷ 2
    N2 = L ÷ 2
    H1 = zeros(T, N2, N2)
    @inbounds for j in 1:N2, i in 1:N2
        H1[i, j] = m[i+j]
    end
    
    vals1 = eigvals(Symmetric(H1))
    if any(vals1 .< -sqrt(eps(T)))
        return false
    end
    
    return true
end

"""
    is_realizable(m::StaticVector{L, T}; domain=:pos)
"""
@inline function is_realizable(m::StaticVector{L, T}; domain=:pos) where {L, T}
    if L < 2 return true end
    
    # --- 1. Hamburger Condition ---
    N1 = (L + 1) ÷ 2
    H = SMatrix{N1, N1, T}(ntuple(k -> @inbounds(m[((k-1) % N1) + ((k-1) ÷ N1) + 1]), Val(N1*N1)))
    
    vals = eigvals(Symmetric(H))
    if any(vals .< -sqrt(eps(T)))
        return false
    end
    
    if domain == :all
        return true
    end
    
    # --- 2. Stieltjes Condition ---
    N2 = L ÷ 2
    H1 = SMatrix{N2, N2, T}(ntuple(k -> @inbounds(m[((k-1) % N2) + ((k-1) ÷ N2) + 2]), Val(N2*N2)))
    
    vals1 = eigvals(Symmetric(H1))
    if any(vals1 .< -sqrt(eps(T)))
        return false
    end
    
    return true
end
