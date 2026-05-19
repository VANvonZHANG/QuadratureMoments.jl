# src/Math/vandermonde.jl
using StaticArrays
using LinearAlgebra

raw"""
    solve_vandermonde(x::AbstractVector, b::AbstractVector, backend)

Solve the primal Vandermonde system \$V c = b\$, where \$V_{i,j} = x_i^{j-1}\$.

The primal system involves nodes assigned to rows. In the `NativeBackend`, this
is implemented using the \$O(N^2)\$ Björck-Pereyra algorithm for zero-allocation performance.

# Arguments
- `x::AbstractVector`: A vector of distinct nodes \$x_1, \dots, x_N\$.
- `b::AbstractVector`: The right-hand side vector.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A vector `c` of coefficients matching the type of `b`.
raw"""
@inline function solve_vandermonde(
    x::SVector{N,T}, b::SVector{N,T}, ::NativeBackend
) where {N,T}
    c = MVector{N,T}(b)
    for i in 1:(N - 1)
        for j in N:-1:(i + 1)
            c[j] = (c[j] - c[j - 1]) / (x[j] - x[j - i])
        end
    end
    for i in (N - 1):-1:1
        for j in i:(N - 1)
            c[j] = c[j] - c[j + 1] * x[i]
        end
    end
    return SVector{N,T}(c)
end

raw"""
    solve_vandermonde_transpose(x::AbstractVector, b::AbstractVector, backend)

Solve the dual (transposed) Vandermonde system \$V^T c = b\$, where \$(V^T)_{i,j} = x_j^{i-1}\$.

This is the most common form in QBMM (e.g., CQMOM), where \$\\sum_j x_j^{i-1} c_j = b_i\$.
In `NativeBackend`, this utilizes a stable \$O(N^2)\$ algorithm.

# Arguments
- `x::AbstractVector`: A vector of distinct nodes \$x_1, \dots, x_N\$.
- `b::AbstractVector`: The moment vector (RHS).
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A vector `c` of weights/coefficients matching the type of `b`.
raw"""
@inline function solve_vandermonde_transpose(
    x::SVector{N,T}, b::SVector{N,T}, ::NativeBackend
) where {N,T}
    return _solve_v_transpose_direct(x, b, Val(N))
end

function _solve_v_transpose_direct(x::SVector{N,T}, b::SVector{N,T}, ::Val{N}) where {N,T}
    # Stable O(N^2) Björck-Pereyra dual solver
    c = MVector{N,T}(b)
    for k in 1:(N - 1)
        for i in N:-1:(k + 1)
            c[i] -= x[k] * c[i - 1]
        end
    end
    for k in (N - 1):-1:1
        for i in (k + 1):N
            c[i] /= (x[i] - x[i - k])
        end
        for i in k:(N - 1)
            c[i] -= c[i + 1]
        end
    end
    return SVector{N,T}(c)
end

# --- External Backend ---

function solve_vandermonde(
    x::AbstractVector{T}, b::AbstractVector{T}, ::ExternalBackend
) where {T}
    N = length(x)
    V = [x[i]^(j - 1) for i in 1:N, j in 1:N]
    return V \ b
end

function solve_vandermonde_transpose(
    x::AbstractVector{T}, b::AbstractVector{T}, ::ExternalBackend
) where {T}
    N = length(x)
    V = [x[j]^(i - 1) for i in 1:N, j in 1:N]
    return V \ b
end
