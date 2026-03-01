# QBMM.jl/src/Solvers/MultiD/cqmom.jl

using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend, QuadratureResult
using ..QBMM: Wheeler, solve_vandermonde_transpose

"""
    CQMOM{D, N, NT} <: AbstractQBMM{D, NT}

Multivariate Conditional Quadrature-Based Moment Method (Discrete).
D: Dimensions
N: NTuple of nodes per dimension
NT: Total nodes (prod(N))
"""
struct CQMOM{D, N, NT} <: AbstractQBMM{D, NT} end

# Constructors
function CQMOM(N::NTuple{D, Int}) where D
    return CQMOM{D, N, prod(N)}()
end

function CQMOM(D::Int, N_per_dim::Int)
    N = ntuple(_ -> N_per_dim, D)
    return CQMOM(N)
end

"""
    invert_moments(method::CQMOM, m::SArray; backend=NativeBackend()) -> QuadratureResult
"""
function invert_moments(
    ::CQMOM{D, N, NT}, 
    m::SArray{S, T, D, L}; 
    backend::AbstractMathBackend = NativeBackend()
) where {D, N, NT, S, T, L}
    
    # Run recursive conditional decomposition
    # Returns (nodes, weights) in (D, NT) shaped MMatrix/MVector
    res_tuple = _cqmom_recursive(Val(D), N, m, backend)
    
    # Standardize result to (NT, D) Matrix
    return QuadratureResult(
        SVector{NT, T}(res_tuple[2]), 
        SMatrix{NT, D, T}(res_tuple[1]'), 
        nothing
    )
end

# --- Base Case: D = 1 ---
function _cqmom_recursive(::Val{1}, N_tuple::NTuple{1, Int}, m::SVector{L, T}, backend::AbstractMathBackend) where {L, T}
    N1 = N_tuple[1]
    res = invert_moments(Wheeler{N1}(), m; backend=backend)
    
    # Return (D, N) transposed for recursive assembly
    return MMatrix{1, N1, T}(res.nodes'), MVector{N1, T}(res.weights)
end

# --- Recursive Step: D > 1 ---
function _cqmom_recursive(::Val{D}, N_tuple::NTuple{D, Int}, m::SArray{S, T, D, L}, backend::AbstractMathBackend) where {D, S, T, L}
    N1 = N_tuple[1]
    L1 = 2 * N1 # Minimum moments needed for 1D discrete inversion
    
    # 1. Marginal Inversion (Dimension 1)
    # Extract marginal moments m[k, 1, 1, ...]
    m1_vec = SVector{L1, T}(ntuple(k -> m[k, ntuple(_ -> 1, Val(D-1))...], Val(L1)))
    
    q1 = invert_moments(Wheeler{N1}(), m1_vec; backend=backend)
    xi1 = SVector{N1, T}(ntuple(i -> q1.nodes[i, 1], Val(N1)))
    w1 = q1.weights
    
    # 2. Deconvolution & Conditional Inversion
    N_rem = ntuple(i -> N_tuple[i+1], Val(D-1))
    NT_rem = prod(N_rem)
    S_rem = ntuple(i -> S.parameters[i+1], Val(D-1))
    L_rem = prod(S_rem)
    
    # For each node in dimension 1, solve a D-1 problem
    all_res = ntuple(Val(N1)) do alpha
        # Construct conditional moment tensor for node alpha
        cond_m_data = ntuple(Val(L_rem)) do idx
            # Map linear index to D-1 Cartesian coordinates
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            
            # Extract mixed moments m[1:N1, coords_rem...]
            # CQMOM uses N1 moments to decompose dimension 1
            b_vec = SVector{N1, T}(ntuple(k -> m[k, coords_rem...], Val(N1)))
            
            # Solve Transpose Vandermonde to get weights*conditional_moments
            # sum_j x1_j^(i-1) * (w1_j * m_cond_j) = b_i
            c_vals = solve_vandermonde_transpose(xi1, b_vec, backend)
            
            # Normalize by weight w1[alpha]
            return c_vals[alpha] / w1[alpha]
        end
        
        # Recurse on the remaining D-1 dimensions
        cond_tensor = SArray{Tuple{S_rem...}, T, D-1, L_rem}(cond_m_data)
        return _cqmom_recursive(Val(D-1), N_rem, cond_tensor, backend)
    end
    
    # 3. Assemble results
    NT_total = N1 * NT_rem
    final_nodes = MMatrix{D, NT_total, T}(undef)
    final_weights = MVector{NT_total, T}(undef)
    
    for alpha in 1:N1
        nodes_rem, weights_rem = all_res[alpha]
        for beta in 1:NT_rem
            idx = (alpha - 1) * NT_rem + beta
            
            # assembly logic
            final_nodes[1, idx] = xi1[alpha]
            final_weights[idx] = w1[alpha] * weights_rem[beta]
            
            for d in 2:D
                final_nodes[d, idx] = nodes_rem[d-1, beta]
            end
        end
    end
    
    return final_nodes, final_weights
end
