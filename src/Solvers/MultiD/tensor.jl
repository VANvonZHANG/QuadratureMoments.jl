# src/Solvers/MultiD/tensor.jl
using LinearAlgebra
using StaticArrays

raw"""
    TensorQMOM{D, N_tuple, N_total} <: AbstractQBMM{D, N_total}

Tensor-product Quadrature-Based Moment Method.

A simplified multi-dimensional solver that assumes the joint NDF can be 
fully separated into independent marginal distributions. It performs 
1D inversions on each coordinate axis and constructs the final \$D\$-dimensional 
quadrature via a Cartesian product.

# Type Parameters
- `D::Int`: Dimensions of the coordinate space.
- `N_tuple::NTuple{D, Int}`: Nodes per dimension.
- `N_total::Int`: Total nodes in the result (`prod(N)`).
raw"""
struct TensorQMOM{D,N_tuple,N_total} <: AbstractQBMM{D,N_total} end

raw"""
    TensorQMOM(N::NTuple{D, Int})
    TensorQMOM(D::Int, N_per_dim::Int)

Constructors for the TensorQMOM solver.
raw"""
TensorQMOM(N::NTuple{D,Int}) where {D} = TensorQMOM{D,N,prod(N)}()
TensorQMOM(D::Int, N_per_dim::Int) = TensorQMOM(ntuple(i -> N_per_dim, D))

raw"""
    invert_moments(method::TensorQMOM, m::SArray; backend=NativeBackend()) -> QuadratureResult

Perform Tensor-product moment inversion.

# Arguments
- `method::TensorQMOM`: The TensorQMOM solver instance.
- `m::SArray`: The multi-dimensional moment tensor.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A `QuadratureResult` containing the combined weights and nodes.
raw"""
function invert_moments(
    method::TensorQMOM{D,N_tuple,N_total},
    m::SArray{S,T,D};
    backend::AbstractMathBackend=NativeBackend(),
) where {D,N_tuple,N_total,S,T}
    # Extract marginal moments for each dimension
    marginal_m = ntuple(Val(D)) do d
        Ld = 2 * N_tuple[d]
        return ntuple(k -> m[ntuple(i -> (i == d ? k : 1), Val(D))...], Val(Ld))
    end

    marginal_m_vecs = ntuple(d -> SVector{length(marginal_m[d]),T}(marginal_m[d]), Val(D))
    return invert_moments(method, marginal_m_vecs; backend=backend)
end

function invert_moments(
    ::TensorQMOM{D,N_tuple,N_total},
    marginal_m::NTuple{D,SVector{L,T}};
    backend::AbstractMathBackend=NativeBackend(),
) where {D,N_tuple,N_total,L,T}

    # 1. Independent 1D inversions
    res_1d = ntuple(Val(D)) do d
        N_d = N_tuple[d]
        return invert_moments(Wheeler{N_d}(), marginal_m[d]; backend=backend)
    end

    # 2. Cartesian product combination
    final_nodes = MMatrix{N_total,D,T}(undef)
    final_weights = MVector{N_total,T}(undef)

    indices = CartesianIndices(N_tuple)
    for (idx, cid) in enumerate(indices)
        w_prod = one(T)
        for d in 1:D
            node_idx = cid.I[d]
            final_nodes[idx, d] = res_1d[d].nodes[node_idx, 1]
            w_prod *= res_1d[d].weights[node_idx]
        end
        final_weights[idx] = w_prod
    end

    return QuadratureResult(
        SVector{N_total,T}(final_weights), SMatrix{N_total,D,T}(final_nodes), nothing
    )
end
