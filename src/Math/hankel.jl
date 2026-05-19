# QBMM.jl/src/Math/hankel.jl

using LinearAlgebra
using StaticArrays

raw"""
    hankel_matrix(m, n, backend; offset=0)

Construct a \$(n+1) \times (n+1)\$ Hankel matrix from a moment sequence.

The matrix is defined as \$H_{i,j} = m_{i+j+offset}\$ (using 0-based indexing for moments).

# Arguments
- `m::AbstractVector`: The moment sequence \$m_0, m_1, \dots\$.
- `n::Int`: The order of the matrix (result is \$(n+1) \times (n+1)\$).
- `backend`: `NativeBackend()` or `ExternalBackend()`.
- `offset::Int`: Optional shift in the moment index (default 0).

# Returns
- A Hankel matrix of type `Matrix{T}` or `SMatrix{n+1, n+1, T}`.
raw"""
function hankel_matrix(m::AbstractVector{T}, n::Int, ::ExternalBackend; offset=0) where {T}
    H = zeros(T, n + 1, n + 1)
    for i in 0:n, j in 0:n
        H[i + 1, j + 1] = m[i + j + 1 + offset]
    end
    return H
end

@inline function hankel_matrix(
    m::AbstractVector{T}, n::Int, ::NativeBackend; offset=0
) where {T}
    # Dynamic vector Native implementation
    return SMatrix{n + 1,n + 1,T}(
        ntuple(
            k -> begin
                idx = (k - 1) % (n + 1) + (k - 1) ÷ (n + 1) + 1 + offset
                m[idx]
            end,
            Val((n + 1) * (n + 1)),
        ),
    )
end

@inline function hankel_matrix(
    m::SVector{L,T}, n::Int, ::NativeBackend; offset=0
) where {L,T}
    # Static vector Native implementation (maximum performance)
    return SMatrix{n + 1,n + 1,T}(
        ntuple(k -> begin
            i = (k - 1) % (n + 1)
            j = (k - 1) ÷ (n + 1)
            m[i + j + 1 + offset]
        end, Val((n + 1) * (n + 1)))
    )
end
