# QBMM.jl/src/Solvers/Evolution/dqmom.jl

using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend

"""
    DQMOM{N} <: AbstractQBMM{1, N}

Direct Quadrature Method of Moments for evolving weights and weighted nodes.
"""
struct DQMOM{N} <: AbstractQBMM{1, N} end

DQMOM(N::Int) = DQMOM{N}()

"""
    dqmom_matrix(nodes::AbstractVector, backend=NativeBackend()) -> Matrix/SMatrix

Constructs the 2Nx2N coefficient matrix A for the DQMOM system:
A * [da/dt; db/dt] = S
where a_i = w_i and b_i = w_i * xi_i.
"""
@inline dqmom_matrix(nodes::AbstractVector, backend::AbstractMathBackend=NativeBackend()) = _dqmom_matrix_dispatch(nodes, backend)

_dqmom_matrix_dispatch(nodes::SVector{N, T}, backend::NativeBackend) where {N, T} = _dqmom_matrix_native(nodes, Val(N))
_dqmom_matrix_dispatch(nodes::AbstractVector{T}, backend::ExternalBackend) where T = _dqmom_matrix_external(nodes)

function _dqmom_matrix_native(nodes::SVector{N, T}, ::Val{N}) where {N, T}
    # Matrix size is 2N x 2N
    A = MMatrix{2*N, 2*N, T}(undef)
    
    for k in 0:(2*N - 1)
        row = k + 1
        for i in 1:N
            col_a = i
            col_b = i + N
            
            # Part 1: Coeff for da_i/dt -> (1 - k) * xi_i^k
            if k == 0
                A[row, col_a] = one(T)
            else
                A[row, col_a] = (1 - k) * (nodes[i]^k)
            end
            
            # Part 2: Coeff for db_i/dt -> k * xi_i^(k-1)
            if k == 0
                A[row, col_b] = zero(T)
            elseif k == 1
                A[row, col_b] = one(T)
            else
                A[row, col_b] = k * (nodes[i]^(k - 1))
            end
        end
    end
    
    return SMatrix{2*N, 2*N, T}(A)
end

function _dqmom_matrix_external(nodes::AbstractVector{T}) where T
    N = length(nodes)
    A = zeros(T, 2*N, 2*N)
    
    for k in 0:(2*N - 1)
        row = k + 1
        for i in 1:N
            col_a = i
            col_b = i + N
            
            if k == 0
                A[row, col_a] = one(T)
                A[row, col_b] = zero(T)
            else
                A[row, col_a] = (1 - k) * (nodes[i]^k)
                if k == 1
                    A[row, col_b] = one(T)
                else
                    A[row, col_b] = k * (nodes[i]^(k - 1))
                end
            end
        end
    end
    return A
end

"""
    dqmom_solve(method, nodes, source_terms; backend=NativeBackend()) -> (da, db)

Solves the DQMOM system to obtain evolution rates of weights (da) and weighted nodes (db).
source_terms: S_k = d(m_k)/dt for k = 0 to 2N-1.
"""
function dqmom_solve(
    ::DQMOM{N}, 
    nodes::SVector{N, T}, 
    source_terms::SVector{L, T}; 
    backend::AbstractMathBackend = NativeBackend()
) where {N, L, T}
    @assert L == 2*N "Number of source terms (S_k) must be 2N."
    
    A = dqmom_matrix(nodes, backend)
    # Solve A * x = S
    x = A \ source_terms
    
    da = SVector{N, T}(ntuple(i -> x[i], Val(N)))
    db = SVector{N, T}(ntuple(i -> x[i+N], Val(N)))
    return da, db
end

function dqmom_solve(
    ::DQMOM{N}, 
    nodes::AbstractVector{T}, 
    source_terms::AbstractVector{T}; 
    backend::AbstractMathBackend = NativeBackend()
) where {N, T}
    @assert length(source_terms) == 2*N "Number of source terms must be 2N."
    
    A = dqmom_matrix(nodes, backend)
    x = A \ source_terms
    
    return x[1:N], x[N+1:2*N]
end
