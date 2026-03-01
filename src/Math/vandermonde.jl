# src/Math/vandermonde.jl
using StaticArrays
using LinearAlgebra

"""
    solve_vandermonde(x::SVector{N, T}, b::SVector{N, T}, ::NativeBackend) -> SVector{N, T}

求解 V * c = b，其中 V_ij = x_i^(j-1)。 (节点在行)
"""
@inline function solve_vandermonde(x::SVector{N, T}, b::SVector{N, T}, ::NativeBackend) where {N, T}
    c = MVector{N, T}(b)
    for i in 1:N-1
        for j in N:-1:i+1
            c[j] = (c[j] - c[j-1]) / (x[j] - x[j-i])
        end
    end
    for i in N-1:-1:1
        for j in i:N-1
            c[j] = c[j] - c[j+1] * x[i]
        end
    end
    return SVector{N, T}(c)
end

"""
    solve_vandermonde_transpose(x::SVector{N, T}, b::SVector{N, T}, ::NativeBackend) -> SVector{N, T}

求解 V^T * c = b，其中 (V^T)_ij = x_j^(i-1)。 (节点在列)
这是 CQMOM 中最常用的形式： sum_j x_j^(i-1) * c_j = b_i
"""
@inline function solve_vandermonde_transpose(x::SVector{N, T}, b::SVector{N, T}, ::NativeBackend) where {N, T}
    c = MVector{N, T}(b)
    for i in 1:N-1
        for j in i:N-1
            c[j] = c[j] - x[i] * c[j+1]
        end
    end
    for i in N-1:-1:1
        for j in i+1:N
            c[j] = (c[j] - c[j-1]) / (x[j] - x[j-i])
        end
    end
    # 上面这个是常见的算法版本，但需要注意索引对齐
    # 重新实现一个更稳健的版本
    return _solve_vandermonde_transpose_stable(x, b, Val(N))
end

function _solve_vandermonde_transpose_stable(x::SVector{N, T}, b::SVector{N, T}, ::Val{N}) where {N, T}
    f = MVector{N, T}(b)
    for k in 1:N-1
        for i in N:-1:k+1
            f[i] = (f[i] - f[i-1]) / (x[i] - x[i-k])
        end
    end
    for k in N-1:-1:1
        for i in k:N-1
            f[i] = f[i] - x[k] * f[i+1]
        end
    end
    # 等一下，这个逻辑和 solve_vandermonde 是一样的？
    # 实际上，Björck-Pereyra 系统非常微妙。
    # 我们改用直接求解 Transpose 的版本：
    return _solve_v_transpose_direct(x, b, Val(N))
end

function _solve_v_transpose_direct(x::SVector{N, T}, b::SVector{N, T}, ::Val{N}) where {N, T}
    # 来自 "Numerical Recipes" 或类似文献的 Transpose Vandermonde 求解
    c = MVector{N, T}(b)
    for k in 1:N-1
        for i in N:-1:k+1
            c[i] -= x[k] * c[i-1]
        end
    end
    for k in N-1:-1:1
        for i in k+1:N
            c[i] /= (x[i] - x[i-k])
        end
        for i in k:N-1
            c[i] -= c[i+1]
        end
    end
    return SVector{N, T}(c)
end

# --- External Backend ---

function solve_vandermonde(x::AbstractVector{T}, b::AbstractVector{T}, ::ExternalBackend) where T
    N = length(x)
    V = [x[i]^(j-1) for i in 1:N, j in 1:N]
    return V \ b
end

function solve_vandermonde_transpose(x::AbstractVector{T}, b::AbstractVector{T}, ::ExternalBackend) where T
    N = length(x)
    V = [x[j]^(i-1) for i in 1:N, j in 1:N]
    return V \ b
end
