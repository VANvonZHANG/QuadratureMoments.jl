# src/Tools/realizability.jl
using LinearAlgebra
using StaticArrays

"""
    is_realizable(m::AbstractVector{T}; domain=:pos, backend=NativeBackend()) -> Bool

检查矩序列的实现性 (Realizability)。
domain: 
  - :all 表示 (-∞, ∞) [Hamburger moment problem]
  - :pos 表示 [0, ∞) [Stieltjes moment problem] (默认)
"""
function is_realizable(m::AbstractVector{T}; domain=:pos, backend::AbstractMathBackend=NativeBackend()) where T
    L = length(m)
    if L < 2 return true end
    
    # --- 1. Hamburger Condition: H_n is positive semi-definite ---
    # n 是阶数，(n+1)x(n+1) 矩阵
    n_h = (L - 1) ÷ 2
    H = hankel_matrix(m, n_h, backend)
    
    vals = eigvals(Symmetric(H))
    if any(vals .< -sqrt(eps(T)))
        return false
    end
    
    if domain == :all
        return true
    end
    
    # --- 2. Stieltjes Condition: H_n^(1) is positive semi-definite ---
    n_s = (L - 2) ÷ 2
    # H1[i, j] = m[i+j] (1-based), 即从 m[2] 开始
    # 我们利用偏移后的切片调用 hankel_matrix
    m_shifted = @view m[2:end]
    H1 = hankel_matrix(m_shifted, n_s, backend)
    
    vals1 = eigvals(Symmetric(H1))
    if any(vals1 .< -sqrt(eps(T)))
        return false
    end
    
    return true
end

"""
    is_realizable(m::StaticVector{L, T}; domain=:pos, backend=NativeBackend())
"""
@inline function is_realizable(m::StaticVector{L, T}; domain=:pos, backend::AbstractMathBackend=NativeBackend()) where {L, T}
    if L < 2 return true end
    
    # --- 1. Hamburger Condition ---
    n_h = (L - 1) ÷ 2
    H = hankel_matrix(m, n_h, backend)
    
    vals = eigvals(Symmetric(H))
    if any(vals .< -sqrt(eps(T)))
        return false
    end
    
    if domain == :all
        return true
    end
    
    # --- 2. Stieltjes Condition ---
    n_s = (L - 2) ÷ 2
    # 对于 SVector，我们可以通过索引映射直接构建偏移的 Hankel
    # 相比 collect(view)，这种方式完全零分配
    H1 = SMatrix{n_s+1, n_s+1, T}(ntuple(k -> begin
        i = (k-1) % (n_s+1)
        j = (k-1) ÷ (n_s+1)
        m[i + j + 2] # 偏移 1
    end, Val((n_s+1)*(n_s+1))))
    
    vals1 = eigvals(Symmetric(H1))
    if any(vals1 .< -sqrt(eps(T)))
        return false
    end
    
    return true
end
