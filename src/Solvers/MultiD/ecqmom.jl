# src/Solvers/MultiD/ecqmom.jl
using LinearAlgebra
using StaticArrays

"""
    ECQMOM{D, N_tuple, N_total, K} <: AbstractQBMM{D, N_total}
"""
struct ECQMOM{D, N_tuple, N_total, K<:AbstractKernel} <: AbstractQBMM{D, N_total}
    kernel::K
end

ECQMOM(N::NTuple{D, Int}, kernel::K=GaussianKernel()) where {D, K} = ECQMOM{D, N, prod(N), K}(kernel)
ECQMOM(D::Int, N_per_dim::Int, kernel::K=GaussianKernel()) where K = ECQMOM(ntuple(i -> N_per_dim, D), kernel)

"""
    invert_moments(method::ECQMOM, m::SArray; backend=NativeBackend())
"""
function invert_moments(
    method::ECQMOM{D, N_tuple, N_total, K}, 
    m::SArray{S, T, D}; 
    backend::AbstractMathBackend = NativeBackend()
) where {D, N_tuple, N_total, K, S, T}
    
    nodes_m, weights_v, sigmas_m = _ecqmom_recursive(Val(D), N_tuple, method.kernel, m, backend)
    
    return QuadratureResult(weights_v, SMatrix{N_total, D, T}(nodes_m'), SMatrix{N_total, D, T}(sigmas_m'))
end

# --- 递归基准情况 (Base Case): D = 1 ---
function _ecqmom_recursive(::Val{1}, N_tuple::NTuple{1, Int}, kernel::K, m::SVector{L, T}, backend::AbstractMathBackend) where {K, L, T}
    N1 = N_tuple[1]
    res = invert_moments(EQMOM{N1, K}(kernel), m; backend=backend)
    return res.nodes', res.weights, res.sigmas'
end

# --- 递归步骤 (Recursive Step): D > 1 ---
function _ecqmom_recursive(::Val{D}, N_tuple::NTuple{D, Int}, kernel::K, m::SArray{S, T, D}, backend::AbstractMathBackend) where {D, K, S, T}
    N1 = N_tuple[1]
    L1 = 2 * N1 + 1
    
    # 1. 边缘矩反演
    m1_tuple = ntuple(i -> m[ntuple(d -> (d == 1 ? i : 1), Val(D))...], Val(L1))
    m1_vec = SVector{L1, T}(m1_tuple)
    
    q1 = invert_moments(EQMOM{N1, K}(kernel), m1_vec; backend=backend)
    xi1 = SVector{N1, T}(ntuple(i -> q1.nodes[i, 1], Val(N1)))
    w1 = q1.weights
    σ1 = q1.sigmas[1, 1] 
    
    # 2. 递归处理后续维度
    S_rem = ntuple(i -> S.parameters[i+1], Val(D-1))
    L_rem = prod(S_rem)
    total_nodes_rem = prod(N_tuple[2:end])
    total_nodes = N1 * total_nodes_rem
    
    all_res = ntuple(Val(N1)) do alpha
        cond_m_data = ntuple(Val(L_rem)) do idx
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            b_vec = SVector{N1, T}(ntuple(i -> m[i, coords_rem...], Val(N1)))
            
            # 第一维去卷积 (Modified Moments)
            m_star = compute_modified_moments(b_vec, σ1, kernel, backend)
            
            # 解 Vandermonde 系统 (Transpose)
            c_prime = solve_vandermonde_transpose(xi1, m_star, backend)
            return c_prime[alpha] / w1[alpha]
        end
        
        cond_m_tensor = SArray{Tuple{S_rem...}, T, D-1, L_rem}(cond_m_data)
        return _ecqmom_recursive(Val(D-1), N_tuple[2:end], kernel, cond_m_tensor, backend)
    end
    
    # 3. 组合结果
    final_nodes = MMatrix{D, total_nodes, T}(undef)
    final_weights = MVector{total_nodes, T}(undef)
    final_sigmas = MMatrix{D, total_nodes, T}(undef)
    
    for alpha in 1:N1
        nodes_rem, weights_rem, sigmas_rem = all_res[alpha]
        for beta in 1:total_nodes_rem
            idx = (alpha - 1) * total_nodes_rem + beta
            final_nodes[1, idx] = xi1[alpha]
            final_sigmas[1, idx] = σ1
            for d in 2:D
                final_nodes[d, idx] = nodes_rem[d-1, beta]
                final_sigmas[d, idx] = sigmas_rem[d-1, beta]
            end
            final_weights[idx] = w1[alpha] * weights_rem[beta]
        end
    end
    
    return SMatrix{D, total_nodes, T}(final_nodes), SVector{total_nodes, T}(final_weights), SMatrix{D, total_nodes, T}(final_sigmas)
end
