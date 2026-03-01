# src/Solvers/MultiD/cqmom.jl
using LinearAlgebra
using StaticArrays

"""
    CQMOM{D, N_tuple, N_total} <: AbstractQBMM{D, N_total}
"""
struct CQMOM{D, N_tuple, N_total} <: AbstractQBMM{D, N_total} end

# 外部构造函数进行参数计算
CQMOM(N::NTuple{D, Int}) where D = CQMOM{D, N, prod(N)}()
CQMOM(D::Int, N_per_dim::Int) = CQMOM(ntuple(i -> N_per_dim, D))

"""
    invert_moments(method::CQMOM, m::SArray; backend=NativeBackend())
"""
function invert_moments(
    ::CQMOM{D, N_tuple, N_total}, 
    m::SArray{S, T, D}; 
    backend::AbstractMathBackend = NativeBackend()
) where {D, N_tuple, N_total, S, T}
    
    res = _cqmom_recursive(Val(D), N_tuple, m, backend)
    nodes_m, weights_v = res
    return QuadratureResult(weights_v, SMatrix{N_total, D, T}(nodes_m'), nothing)
end

# --- 递归基准情况 (Base Case): D = 1 ---
function _cqmom_recursive(::Val{1}, N_tuple::NTuple{1, Int}, m::SVector{L, T}, backend::AbstractMathBackend) where {L, T}
    N1 = N_tuple[1]
    q1 = invert_moments(Wheeler{N1}(), m; backend=backend)
    return q1.nodes', q1.weights
end

# --- 递归步骤 (Recursive Step): D > 1 ---
function _cqmom_recursive(::Val{D}, N_tuple::NTuple{D, Int}, m::SArray{S, T, D}, backend::AbstractMathBackend) where {D, S, T}
    N1 = N_tuple[1]
    L1 = 2 * N1
    
    # 1. 第一维边缘矩
    m1_tuple = ntuple(i -> m[ntuple(d -> (d == 1 ? i : 1), Val(D))...], Val(L1))
    m1_vec = SVector{L1, T}(m1_tuple)
    
    # 2. 第一维反演
    q1 = invert_moments(Wheeler{N1}(), m1_vec; backend=backend)
    xi1 = SVector{N1, T}(ntuple(i -> q1.nodes[i, 1], Val(N1)))
    w1 = q1.weights
    
    # 3. 准备 Vandermonde 求解
    S_rem = ntuple(i -> S.parameters[i+1], Val(D-1))
    L_rem = prod(S_rem)
    total_nodes_rem = prod(N_tuple[2:end])
    total_nodes = N1 * total_nodes_rem
    
    all_res = ntuple(Val(N1)) do alpha
        cond_m_data = ntuple(Val(L_rem)) do idx
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            b_vec = SVector{N1, T}(ntuple(i -> m[i, coords_rem...], Val(N1)))
            
            # 使用 Math/vandermonde.jl 的高效 Transpose 求解器
            c_prime = solve_vandermonde_transpose(xi1, b_vec, backend)
            return c_prime[alpha] / w1[alpha] 
        end
        
        cond_m_tensor = SArray{Tuple{S_rem...}, T, D-1, L_rem}(cond_m_data)
        return _cqmom_recursive(Val(D-1), N_tuple[2:end], cond_m_tensor, backend)
    end
    
    # 4. 组合结果
    final_nodes = MMatrix{D, total_nodes, T}(undef)
    final_weights = MVector{total_nodes, T}(undef)
    
    for alpha in 1:N1
        nodes_rem, weights_rem = all_res[alpha]
        for beta in 1:total_nodes_rem
            idx = (alpha - 1) * total_nodes_rem + beta
            final_nodes[1, idx] = xi1[alpha]
            for d in 2:D
                final_nodes[d, idx] = nodes_rem[d-1, beta]
            end
            final_weights[idx] = w1[alpha] * weights_rem[beta]
        end
    end
    
    return SMatrix{D, total_nodes, T}(final_nodes), SVector{total_nodes, T}(final_weights)
end
