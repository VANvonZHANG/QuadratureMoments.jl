# src/Math/hankel.jl
using LinearAlgebra
using StaticArrays

"""
    hankel_matrix(m::AbstractVector{T}, n::Int, ::NativeBackend) -> SMatrix

原生实现：构建 (n+1)x(n+1) 的静态汉克尔矩阵。
H[i, j] = m[i+j-1] (0-based idx: i+j)
"""
@inline function hankel_matrix(m::AbstractVector{T}, n::Int, ::NativeBackend) where T
    # 使用 SMatrix 构造器，基于索引计算
    return SMatrix{n+1, n+1, T}(
        ntuple(i -> m[i], (n+1)*(n+1)) # 这种直接 ntuple 展开对小规模 N 极快
    )
end

# 针对 SVector 的更优版本
@inline function hankel_matrix(m::SVector{L, T}, n::Int, ::NativeBackend) where {L, T}
    return SMatrix{n+1, n+1, T}(ntuple(k -> begin
        i = (k-1) % (n+1)
        j = (k-1) ÷ (n+1)
        m[i + j + 1]
    end, Val((n+1)*(n+1))))
end

"""
    hankel_matrix(m::AbstractVector{T}, n::Int, ::ExternalBackend) -> Matrix

外部后端实现：构建标准 Matrix。
"""
function hankel_matrix(m::AbstractVector{T}, n::Int, ::ExternalBackend) where T
    H = zeros(T, n+1, n+1)
    for i in 0:n, j in 0:n
        H[i+1, j+1] = m[i+j+1]
    end
    return H
end
