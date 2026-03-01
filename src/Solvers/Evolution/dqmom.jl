# QBMM.jl/src/Solvers/Evolution/dqmom.jl

using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend

raw"""
    DQMOM{N} <: AbstractQBMM{1, N}

Direct Quadrature Method of Moments for tracking evolution of weights and nodes.

Instead of inverting moments at each step, DQMOM directly evolves the quadrature 
approximation by solving a linear system for the rates of change of weights 
and weighted nodes.

# Type Parameters
- `N::Int`: Number of quadrature nodes.
raw"""
struct DQMOM{N} <: AbstractQBMM{1, N} end

raw"""
    DQMOM(N::Int)

Constructor for the DQMOM solver.
raw"""
DQMOM(N::Int) = DQMOM{N}()

raw"""
    dqmom_matrix(nodes, backend=NativeBackend()) -> Matrix/SMatrix

Construct the \$2N \times 2N\$ coefficient matrix \$A\$ for the DQMOM system.

The system is defined as \$A \mathbf{x} = \mathbf{S}\$, where 
\$\mathbf{x} = [da/dt; db/dt]\$, \$a_i = w_i\$, and \$b_i = w_i \xi_i\$.

# Arguments
- `nodes::AbstractVector`: Current quadrature nodes \$\xi_i\$.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- The coefficient matrix \$A\$.
raw"""
@inline dqmom_matrix(
    nodes::AbstractVector,
    backend::AbstractMathBackend=NativeBackend(),
) = _dqmom_matrix_dispatch(nodes, backend)

_dqmom_matrix_dispatch(nodes::SVector{N, T}, backend::NativeBackend) where {N, T} =
    _dqmom_matrix_native(nodes, Val(N))
_dqmom_matrix_dispatch(nodes::AbstractVector{T}, backend::ExternalBackend) where {T} =
    _dqmom_matrix_external(nodes)

function _dqmom_matrix_native(nodes::SVector{N, T}, ::Val{N}) where {N, T}
    # Matrix size is 2N x 2N
    A = MMatrix{2 * N, 2 * N, T}(undef)

    for k in 0:(2 * N - 1)
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

    return SMatrix{2 * N, 2 * N, T}(A)
end

function _dqmom_matrix_external(nodes::AbstractVector{T}) where {T}
    N = length(nodes)
    A = zeros(T, 2 * N, 2 * N)

    for k in 0:(2 * N - 1)
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

raw"""
    dqmom_solve(method, nodes, source_terms; backend=NativeBackend()) -> (da, db)

Solve the DQMOM system to obtain evolution rates.

# Arguments
- `method::DQMOM{N}`: The DQMOM solver.
- `nodes::AbstractVector`: Current nodes \$\xi_i\$.
- `source_terms::AbstractVector`: Moment source terms \$S_k = dm_k/dt\$.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- `da`: Rates of change for weights \$w_i\$.
- `db`: Rates of change for weighted nodes \$(w_i \xi_i)\$.
raw"""
function dqmom_solve(
    ::DQMOM{N},
    nodes::SVector{N, T},
    source_terms::SVector{L, T};
    backend::AbstractMathBackend=NativeBackend(),
) where {N, L, T}
    @assert L == 2 * N "Number of source terms (S_k) must be 2N."

    A = dqmom_matrix(nodes, backend)
    # Solve A * x = S
    x = A \ source_terms

    da = SVector{N, T}(ntuple(i -> x[i], Val(N)))
    db = SVector{N, T}(ntuple(i -> x[i + N], Val(N)))
    return da, db
end

function dqmom_solve(
    ::DQMOM{N},
    nodes::AbstractVector{T},
    source_terms::AbstractVector{T};
    backend::AbstractMathBackend=NativeBackend(),
) where {N, T}
    @assert length(source_terms) == 2 * N "Number of source terms must be 2N."

    A = dqmom_matrix(nodes, backend)
    x = A \ source_terms

    return x[1:N], x[N + 1:(2 * N)]
end
