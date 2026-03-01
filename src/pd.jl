using LinearAlgebra
using StaticArrays

"""
    PD <: AbstractQBMM
    
使用 Product-Difference 算法的 1D 矩反演。
"""
struct PD <: AbstractQBMM end

"""
    invert_moments(method::PD, m::AbstractVector)
"""
function invert_moments(::PD, m::AbstractVector{T}) where T
    return pd_inversion(m)
end

function invert_moments(::PD, m::SVector{L, T}) where {L, T}
    return pd_inversion(m)
end

"""
    pd_inversion(m::AbstractVector{T}) -> (nodes, weights)

使用 Product-Difference (PD) 算法执行单变量矩反演。
输入长度为 2N 的矩序列，返回 N 个节点和权重。
该方法为基于标准动态数组 (Array) 的实现。
"""
function pd_inversion(m::AbstractVector{T}) where T
    N = length(m) ÷ 2
    @assert length(m) == 2N "The number of moments must be even (2N)."
    
    P = zeros(T, 2N+1, 2N+1)
    
    P[1, 1] = one(T)
    for α in 1:2N
        P[α, 2] = (-1)^(α-1) * m[α]
    end
    
    for β in 3:2N+1
        for α in 1:(2N+2-β)
            P[α, β] = P[1, β-1] * P[α+1, β-2] - P[1, β-2] * P[α+1, β-1]
        end
    end
    
    ζ = zeros(T, 2N)
    # ζ[1] is implicitly 0
    for α in 2:2N
        ζ[α] = P[1, α+1] / (P[1, α] * P[1, α-1])
    end
    
    a = zeros(T, N)
    b = zeros(T, N-1)
    
    for α in 1:N
        a[α] = ζ[2α] + ζ[2α-1]
    end
    
    for α in 1:N-1
        # Use abs inside sqrt to handle small numerical noise correctly if it occurs
        b[α] = sqrt(abs(ζ[2α+1] * ζ[2α]))
    end
    
    J = SymTridiagonal(a, b)
    eigen_decomp = eigen(J)
    
    nodes = eigen_decomp.values
    weights = m[1] .* (eigen_decomp.vectors[1, :] .^ 2)
    
    return nodes, weights
end

"""
    pd_inversion(m::SVector{L, T}) -> (nodes, weights)

基于 `StaticArrays.jl` 优化的 PD 反演实现，针对较小规模的矩 (L <= 12 即 N <= 6)
设计，最大程度消除堆内存分配并提升性能。
"""
function pd_inversion(m::SVector{L, T}) where {L, T}
    return _pd_inversion(m, Val(L ÷ 2))
end

function _pd_inversion(m::SVector{L, T}, ::Val{N}) where {L, T, N}
    P = zero(MMatrix{L+1, L+1, T})
    
    P[1, 1] = one(T)
    for α in 1:L
        P[α, 2] = (-1)^(α-1) * m[α]
    end
    
    for β in 3:L+1
        for α in 1:(L+2-β)
            P[α, β] = P[1, β-1] * P[α+1, β-2] - P[1, β-2] * P[α+1, β-1]
        end
    end
    
    ζ = MVector{L, T}(undef)
    ζ[1] = zero(T)
    for α in 2:L
        ζ[α] = P[1, α+1] / (P[1, α] * P[1, α-1])
    end
    
    a = MVector{N, T}(undef)
    for α in 1:N
        a[α] = ζ[2α] + ζ[2α-1]
    end
    
    J_M = zero(MMatrix{N, N, T})
    for i in 1:N
        J_M[i, i] = a[i]
    end
    for i in 1:N-1
        off = sqrt(abs(ζ[2i+1] * ζ[2i]))
        J_M[i, i+1] = off
        J_M[i+1, i] = off
    end
    
    J = SMatrix{N, N, T}(J_M)
    eigen_decomp = eigen(Symmetric(J))
    
    nodes = SVector{N, T}(eigen_decomp.values)
    weights = SVector{N, T}(ntuple(i -> m[1] * eigen_decomp.vectors[1, i]^2, Val(N)))
    
    return nodes, weights
end
