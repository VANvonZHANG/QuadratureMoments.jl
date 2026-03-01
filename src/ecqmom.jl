using LinearAlgebra
using StaticArrays

"""
    ECQMOM{D, N, K}
    
扩展条件矩反演 (Extended Conditional Quadrature Method of Moments)。
结合了 CQMOM 的多维条件分解和 EQMOM 的连续核函数。
D: 维度 (Integer)。
N: 每一维的节点数 (NTuple{D, Int})。
K: 核函数类型 (AbstractKernel)。
"""
struct ECQMOM{D, N, K<:AbstractKernel} <: AbstractQBMM
    kernel::K
end

# 构造函数
ECQMOM(D::Int, N::Int, kernel::K=GaussianKernel()) where K = ECQMOM{D, ntuple(i -> N, D), K}(kernel)
ECQMOM(N::NTuple{D, Int}, kernel::K=GaussianKernel()) where {D, K} = ECQMOM{D, N, K}(kernel)

"""
    invert_moments(method::ECQMOM{D, N, K}, m::SArray)
    
ECQMOM 的入口。输入张量 m 的尺寸应为 (2N1+1) x (2N2+1) x ...
返回节点矩阵、权重向量和带宽矩阵 sigma_matrix (D x TotalNodes)。
"""
function invert_moments(method::ECQMOM{D, N, K}, m::SArray{S, T, D}) where {D, N, K, S, T}
    return _ecqmom_recursive(Val(D), N, method.kernel, m)
end

# --- 递归基准情况 (Base Case): D = 1 ---
function _ecqmom_recursive(::Val{1}, N_tuple::NTuple{1, Int}, kernel::K, m::SVector{L, T}) where {K, L, T}
    method_1d = EQMOM{N_tuple[1], K}(kernel)
    nodes_1d, weights_1d, σ = invert_moments(method_1d, m)
    
    # 返回节点向量、权重向量，以及该层特有的带宽 (单个值包装在 SVector 中)
    return nodes_1d, weights_1d, SVector{1, T}(σ)
end

# --- 递归步骤 (Recursive Step): D > 1 ---
function _ecqmom_recursive(::Val{D}, N_tuple::NTuple{D, Int}, kernel::K, m::SArray{S, T, D}) where {D, K, S, T}
    # 1. 提取当前第一维的边缘矩
    N1 = N_tuple[1]
    L1 = 2 * N1 + 1
    
    m1_tuple = ntuple(i -> m[ntuple(d -> (d == 1 ? i : 1), Val(D))...], Val(L1))
    m1_vec = SVector{L1, T}(m1_tuple)
    
    # 2. 本维 EQMOM 反演
    method_1d = EQMOM{N1, K}(kernel)
    xi1, w1, σ1 = invert_moments(method_1d, m1_vec)
    
    # 3. 构造 Vandermonde 求解器 (基于修正矩)
    V_M = zero(MMatrix{N1, N1, T})
    for i in 1:N1
        for alpha in 1:N1
            V_M[i, alpha] = w1[alpha] * (xi1[alpha]^(i-1))
        end
    end
    V_fact = lu(SMatrix{N1, N1, T}(V_M))
    
    # 4. 递归处理后续维度
    total_nodes_remaining = prod(N_tuple[2:end])
    total_nodes = N1 * total_nodes_remaining
    
    S_rem = ntuple(i -> S.parameters[i+1], Val(D-1))
    L_rem = prod(S_rem)
    
    all_res = ntuple(Val(N1)) do alpha
        # 解 Vandermonde 系统获取修正后的条件矩张量
        cond_m_data = ntuple(Val(L_rem)) do idx
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            b_tuple = ntuple(i -> m[i, coords_rem...], Val(N1))
            b_vec = SVector{N1, T}(b_tuple)
            
            # 第一维去卷积
            m_star_slice = compute_modified_moments(b_vec, σ1, kernel)
            return (V_fact \ m_star_slice)[alpha]
        end
        
        cond_m_tensor = SArray{Tuple{S_rem...}, T, D-1, L_rem}(cond_m_data)
        return _ecqmom_recursive(Val(D-1), N_tuple[2:end], kernel, cond_m_tensor)
    end
    
    # 5. 组合结果
    final_nodes_m = zero(MMatrix{D, total_nodes, T})
    final_weights_v = zero(MVector{total_nodes, T})
    final_sigmas = zero(MMatrix{D, total_nodes, T})
    
    for alpha in 1:N1
        nodes_rem, weights_rem, sigmas_rem = all_res[alpha]
        for beta in 1:total_nodes_remaining
            idx = (alpha - 1) * total_nodes_remaining + beta
            
            # 第一维结果
            final_nodes_m[1, idx] = xi1[alpha]
            final_sigmas[1, idx] = σ1
            
            # 后续维度结果
            if D == 2
                # sigmas_rem 是 SVector{1, T}
                final_nodes_m[2, idx] = nodes_rem[beta]
                final_sigmas[2, idx] = sigmas_rem[1]
            else
                # sigmas_rem 是 SMatrix{(D-1), total_nodes_rem}
                for d in 2:D
                    final_nodes_m[d, idx] = nodes_rem[d-1, beta]
                    final_sigmas[d, idx] = sigmas_rem[d-1, beta]
                end
            end
            
            # 组合权重
            final_weights_v[idx] = w1[alpha] * weights_rem[beta]
        end
    end
    
    return SMatrix{D, total_nodes, T}(final_nodes_m), SVector{total_nodes, T}(final_weights_v), SMatrix{D, total_nodes, T}(final_sigmas)
end
