# QBMM.jl/src/Math/hankel.jl

using LinearAlgebra
using StaticArrays

"""
    hankel_matrix(m, n, backend; offset=0) -> Matrix/SMatrix

构建 (n+1)x(n+1) 的汉克尔矩阵。
H[i, j] = m[i+j+1 + offset]
"""
function hankel_matrix(m::AbstractVector{T}, n::Int, ::ExternalBackend; offset=0) where T
    H = zeros(T, n+1, n+1)
    for i in 0:n, j in 0:n
        H[i+1, j+1] = m[i+j+1+offset]
    end
    return H
end

@inline function hankel_matrix(m::AbstractVector{T}, n::Int, ::NativeBackend; offset=0) where T
    # 动态向量的 Native 实现
    return SMatrix{n+1, n+1, T}(ntuple(k -> begin
        idx = (k-1) % (n+1) + (k-1) ÷ (n+1) + 1 + offset
        m[idx]
    end, Val((n+1)*(n+1))))
end

@inline function hankel_matrix(m::SVector{L, T}, n::Int, ::NativeBackend; offset=0) where {L, T}
    # 静态向量的 Native 实现 (极速)
    return SMatrix{n+1, n+1, T}(ntuple(k -> begin
        i = (k-1) % (n+1)
        j = (k-1) ÷ (n+1)
        m[i + j + 1 + offset]
    end, Val((n+1)*(n+1))))
end
