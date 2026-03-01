# QBMM.jl/src/Tools/realizability.jl
using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractMathBackend, NativeBackend, ExternalBackend, hankel_matrix

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
    
    # 1. Hamburger Condition: H_n is positive semi-definite
    n_h = (L - 1) ÷ 2
    H = hankel_matrix(m, n_h, backend; offset=0)
    
    if !is_psd(H, T)
        return false
    end
    
    if domain == :all
        return true
    end
    
    # 2. Stieltjes Condition: H_n^(1) is positive semi-definite
    n_s = (L - 2) ÷ 2
    H1 = hankel_matrix(m, n_s, backend; offset=1)
    
    return is_psd(H1, T)
end

"""
    is_realizable(m::StaticVector{L, T}; domain=:pos, backend=NativeBackend())
"""
@inline function is_realizable(m::StaticVector{L, T}; domain=:pos, backend::AbstractMathBackend=NativeBackend()) where {L, T}
    if L < 2 return true end
    
    # 1. Hamburger Condition
    n_h = (L - 1) ÷ 2
    H = hankel_matrix(m, n_h, backend; offset=0)
    
    if !is_psd(H, T)
        return false
    end
    
    if domain == :all
        return true
    end
    
    # 2. Stieltjes Condition
    n_s = (L - 2) ÷ 2
    H1 = hankel_matrix(m, n_s, backend; offset=1)
    
    return is_psd(H1, T)
end

# 内部辅助：检查半正定性 (PSD)
@inline function is_psd(H::AbstractMatrix{T}, ::Type{T}) where T
    # 对于小矩阵，eigvals 是最稳健的
    vals = eigvals(Symmetric(H))
    return all(vals .> -sqrt(eps(T)))
end
