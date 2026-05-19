# src/Solvers/MultiD/brute.jl
using LinearAlgebra
using StaticArrays
using ForwardDiff

raw"""
    BruteQMOM{D, N} <: AbstractQBMM{D, N}

Brute-force multi-dimensional moment inversion.

This solver formulates the moment inversion problem as a highly nonlinear system 
of equations and solves it directly using a Newton-Raphson method accelerated by 
automatic differentiation (`ForwardDiff.jl`).

# Type Parameters
- `D::Int`: Dimensions of the coordinate space.
- `N::Int`: Number of multi-dimensional quadrature nodes.
raw"""
struct BruteQMOM{D,N} <: AbstractQBMM{D,N} end

raw"""
    BruteQMOM(D::Int, N::Int)

Constructor for the BruteQMOM solver.
raw"""
BruteQMOM(D::Int, N::Int) = BruteQMOM{D,N}()

raw"""
    invert_moments(method::BruteQMOM, m::SArray; kwargs...) -> QuadratureResult

Perform brute-force Newton-Raphson moment inversion.

# Arguments
- `method::BruteQMOM`: The BruteQMOM solver instance.
- `m::SArray`: The multi-dimensional moment tensor.
- `max_iter::Int`: Maximum Newton iterations (default 20).
- `tol::Float64`: Convergence tolerance (default `1e-10`).
- `backend`: Mathematical backend.

# Returns
- A `QuadratureResult` containing weights and nodes.
raw"""
function invert_moments(
    ::BruteQMOM{D,N},
    m::SArray{S,T,D};
    max_iter=20,
    tol=1e-10,
    backend::AbstractMathBackend=NativeBackend(),
) where {D,N,S,T}
    P = N * (D + 1)

    # 1. Initial Guess Generation
    x0_data = MVector{P,T}(undef)
    # Uniform weight distribution
    w0 = m[ntuple(i -> 1, Val(D))...] / N
    for i in 1:N
        x0_data[i] = w0
    end
    # Node positioning spread around the mean
    for i in 1:N
        for d in 1:D
            x0_data[N + (d - 1) * N + i] =
                (
                    m[ntuple(j -> (j == d ? 2 : 1), Val(D))...] /
                    m[ntuple(j -> 1, Val(D))...]
                ) * (0.8 + 0.4 * i / N)
        end
    end
    x = SVector{P,T}(x0_data)

    # 2. Residual Function Formulation
    S_tuple = ntuple(i -> S.parameters[i], Val(D))
    mom_indices = CartesianIndices(S_tuple)[1:P]

    function residual(curr_x::AbstractVector{V}) where {V}
        res = MVector{P,V}(undef)
        ws = curr_x[1:N]
        # nodes[alpha, d]
        nodes = reshape(curr_x[(N + 1):P], N, D)

        for p in 1:P
            idx = mom_indices[p]
            target = m[idx]
            calc = zero(V)
            for alpha in 1:N
                term = ws[alpha]
                for d in 1:D
                    k = idx.I[d] - 1
                    term *= (nodes[alpha, d]^k)
                end
                calc += term
            end
            res[p] = calc - target
        end
        return SVector{P,V}(res)
    end

    # 3. Newton-Raphson Loop with Auto-Diff
    for iter in 1:max_iter
        r = residual(x)
        if norm(r) < tol
            break
        end
        J = ForwardDiff.jacobian(residual, x)
        delta = J \ r
        x = x - delta
    end

    # 4. Assembly
    final_weights = SVector{N,T}(x[1:N])
    final_nodes = SMatrix{N,D,T}(reshape(x[(N + 1):P], N, D))

    return QuadratureResult(final_weights, final_nodes, nothing)
end
