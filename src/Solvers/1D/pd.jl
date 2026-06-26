# src/Solvers/1D/pd.jl
using LinearAlgebra
using StaticArrays

raw"""
    PD{N} <: AbstractQBMM{1, N}
    
The Product-Difference (PD) algorithm for 1D moment inversion.

# Type Parameters
- `N::Int`: Number of quadrature nodes to reconstruct.
raw"""
struct PD{N} <: AbstractQBMM{1,N} end

raw"""
    PD(N::Int)

Constructor for the Product-Difference algorithm with `N` nodes.
raw"""
PD(N::Int) = PD{N}()

raw"""
    invert_moments(method::PD{N}, m; backend=NativeBackend()) -> QuadratureResult

Perform 1D Product-Difference inversion.

# Arguments
- `method::PD{N}`: The PD solver instance.
- `m::AbstractVector`: Vector of at least \$2N\$ moments.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A `QuadratureResult` containing weights and nodes.
raw"""
function invert_moments(
    method::PD{N}, m::AbstractVector{T}; backend::AbstractMathBackend=NativeBackend()
) where {N,T}
    nodes, weights = pd_inversion(m)
    return QuadratureResult(SVector{N,T}(weights), SMatrix{N,1,T}(nodes), nothing)
end

function invert_moments(
    method::PD{N}, m::SVector{L,T}; backend::AbstractMathBackend=NativeBackend()
) where {N,L,T}
    nodes, weights = _pd_inversion(m, Val(N))
    return QuadratureResult(weights, SMatrix{N,1,T}(nodes), nothing)
end

# --- Internal Implementations ---

function pd_inversion(m::AbstractVector{T}) where {T}
    N = length(m) ÷ 2
    P = zeros(T, 2N + 1, 2N + 1)

    P[1, 1] = one(T)
    for α in 1:(2N)
        P[α, 2] = (-1)^(α - 1) * m[α]
    end

    for β in 3:(2N + 1)
        for α in 1:(2N + 2 - β)
            P[α, β] = P[1, β - 1] * P[α + 1, β - 2] - P[1, β - 2] * P[α + 1, β - 1]
        end
    end

    ζ = zeros(T, 2N)
    for α in 2:(2N)
        ζ[α] = P[1, α + 1] / (P[1, α] * P[1, α - 1])
    end

    # Determine active rank: stop where the recurrence product goes non-positive
    # or non-finite (non-realizable Hankel determinant / 0-0 in the recurrence).
    # Mirror Wheeler's adaptive rank reduction instead of abs()-masking the sign.
    a = zeros(T, N)
    for α in 1:N
        a[α] = ζ[2α] + ζ[2α - 1]
    end
    actual_N = N
    b_off = zeros(T, N - 1)
    for α in 1:(N - 1)
        prod_ζ = ζ[2α + 1] * ζ[2α]
        if !isfinite(prod_ζ) || prod_ζ <= 0
            actual_N = α
            break
        end
        b_off[α] = sqrt(prod_ζ)
    end

    J = SymTridiagonal(a[1:actual_N], b_off[1:max(actual_N - 1, 0)])
    eigen_decomp = eigen(J)

    nodes = zeros(T, N)
    weights = zeros(T, N)
    nodes[1:actual_N] = eigen_decomp.values
    weights[1:actual_N] = m[1] .* (eigen_decomp.vectors[1, 1:actual_N] .^ 2)
    return nodes, weights
end

function _pd_inversion(m::SVector{L,T}, ::Val{N}) where {L,T,N}
    P = zero(MMatrix{L + 1,L + 1,T})

    P[1, 1] = one(T)
    for α in 1:L
        P[α, 2] = (-1)^(α - 1) * m[α]
    end

    for β in 3:(L + 1)
        for α in 1:(L + 2 - β)
            P[α, β] = P[1, β - 1] * P[α + 1, β - 2] - P[1, β - 2] * P[α + 1, β - 1]
        end
    end

    ζ = MVector{L,T}(undef)
    ζ[1] = zero(T)
    for α in 2:L
        ζ[α] = P[1, α + 1] / (P[1, α] * P[1, α - 1])
    end

    a = MVector{N,T}(undef)
    for α in 1:N
        a[α] = ζ[2α] + ζ[2α - 1]
    end

    # Adaptive rank reduction: truncate when the off-diagonal product goes
    # non-positive or non-finite (non-realizable). Mirror Wheeler's pattern:
    # zero-init the full MMatrix, fill only the active block, and eigen-decompose
    # the full SMatrix. The unused lower-right zero block yields eigenvalue 0
    # with eigenvectors whose first component is 0 → weight 0 emerges naturally.
    actual_N = N
    for i in 1:(N - 1)
        prod_ζ = ζ[2i + 1] * ζ[2i]
        if !isfinite(prod_ζ) || prod_ζ <= 0
            actual_N = i
            break
        end
    end

    J_M = zero(MMatrix{N,N,T})
    for i in 1:actual_N
        J_M[i, i] = a[i]
    end
    for i in 1:(actual_N - 1)
        off = sqrt(ζ[2i + 1] * ζ[2i])
        J_M[i, i + 1] = off
        J_M[i + 1, i] = off
    end

    J = SMatrix{N,N,T}(J_M)
    eigen_decomp = eigen(Symmetric(J))

    nodes = SVector{N,T}(eigen_decomp.values)
    weights = SVector{N,T}(ntuple(i -> m[1] * eigen_decomp.vectors[1, i]^2, Val(N)))

    return nodes, weights
end
