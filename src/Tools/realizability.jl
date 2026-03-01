# QBMM.jl/src/Tools/realizability.jl
using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractMathBackend, NativeBackend, ExternalBackend, hankel_matrix

raw"""
    is_realizable(m::AbstractVector; domain=:pos, backend=NativeBackend()) -> Bool

Check the realizability of a moment sequence \$m_0, m_1, \\dots, m_L\$.

A moment sequence is realizable if there exists a non-negative density 
function \$n(\\xi) \\ge 0\$ that produces these moments.

# Arguments
- `m::AbstractVector`: The moment sequence to check.
- `domain::Symbol`: The support of the NDF.
    - `:all`: Check on \$(-\\infty, \\infty)\$ (Hamburger moment problem).
    - `:pos`: Check on \$[0, \\infty)\$ (Stieltjes moment problem, default).
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- `true` if the sequence is realizable within the specified domain, `false` otherwise.
raw"""
function is_realizable(
    m::AbstractVector{T};
    domain=:pos,
    backend::AbstractMathBackend=NativeBackend(),
) where {T}
    L = length(m)
    if L < 2
        return true
    end

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

raw"""
    is_realizable(m::StaticVector; domain=:pos, backend=NativeBackend())
raw"""
@inline function is_realizable(
    m::StaticVector{L, T};
    domain=:pos,
    backend::AbstractMathBackend=NativeBackend(),
) where {L, T}
    if L < 2
        return true
    end

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

raw"""
    is_psd(H::AbstractMatrix, T) -> Bool

Internal helper to check if a matrix is Positive Semi-Definite (PSD).
Uses eigenvalue analysis with a tolerance of \$-\\sqrt{\\epsilon}\$.
raw"""
@inline function is_psd(H::AbstractMatrix{T}, ::Type{T}) where {T}
    # For small matrices, eigvals is robust
    vals = eigvals(Symmetric(H))
    return all(vals .> -sqrt(eps(T)))
end
