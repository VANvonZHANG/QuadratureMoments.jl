using LinearAlgebra
using StaticArrays

"""
    CQMOM{D, N}
    
多变量条件矩反演 (Conditional Quadrature Method of Moments)。
D: 维度 (Integer)。
N: 每一维的节点数 (NTuple{D, Int})。
"""
struct CQMOM{D, N} <: AbstractQBMM end

# 构造函数简化
CQMOM(D::Int, N::Int) = CQMOM{D, ntuple(i -> N, D)}()
CQMOM(N::NTuple{D, Int}) where D = CQMOM{D, N}()

"""
    invert_moments(method::CQMOM{D, N}, m::SArray)
    
CQMOM 的统一入口。输入 D 维矩张量 (SArray)，返回节点矩阵和权重向量。
"""
function invert_moments(::CQMOM{D, N}, m::SArray{S, T, D}) where {D, N, S, T}
    return _cqmom_recursive(Val(D), N, m)
end

# --- 递归基准情况 (Base Case): D = 1 ---
function _cqmom_recursive(::Val{1}, N_tuple::NTuple{1, Int}, m::SVector{L, T}) where {L, T}
    # 直接调用 Wheeler 算法进行单变量反演
    nodes_1d, weights_1d = wheeler_inversion(m)
    
    # 返回 (D x TotalNodes) 的节点矩阵
    N_nodes = N_tuple[1]
    final_nodes = reshape(nodes_1d, 1, N_nodes)
    return final_nodes, weights_1d
end

# --- 递归步骤 (Recursive Step): D > 1 ---
function _cqmom_recursive(::Val{D}, N_tuple::NTuple{D, Int}, m::SArray{S, T, D}) where {D, S, T}
    # 1. 提取第一维的边缘矩
    N1 = N_tuple[1]
    L1 = 2 * N1
    
    m1_tuple = ntuple(i -> m[ntuple(d -> (d == 1 ? i : 1), Val(D))...], Val(L1))
    m1_vec = SVector{L1, T}(m1_tuple)
    
    # 2. 第一维反演
    xi1, w1 = wheeler_inversion(m1_vec)
    
    # 3. 构造 Vandermonde 矩阵 V
    V_M = zero(MMatrix{N1, N1, T})
    for i in 1:N1
        for alpha in 1:N1
            V_M[i, alpha] = w1[alpha] * (xi1[alpha]^(i-1))
        end
    end
    V = SMatrix{N1, N1, T}(V_M)
    V_fact = lu(V)
    
    # 4. 计算剩余维度的条件矩并递归
    total_nodes_remaining = prod(N_tuple[2:end])
    total_nodes = N1 * total_nodes_remaining
    
    # 获取剩余维度的 S 尺寸元组
    S_rem = ntuple(i -> S.parameters[i+1], Val(D-1))
    L_rem = prod(S_rem)
    
    all_res = ntuple(Val(N1)) do alpha
        cond_m_data = ntuple(Val(L_rem)) do idx
            # 使用线性索引手动模拟 CartesianIndices
            # 因为 CartesianIndices(S_rem) 在 S_rem 为元组时可能无法直接工作
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            
            b_tuple = ntuple(i -> m[i, coords_rem...], Val(N1))
            b_vec = SVector{N1, T}(b_tuple)
            
            # 解方程并取第 alpha 个解
            return (V_fact \ b_vec)[alpha]
        end
        
        cond_m_tensor = SArray{Tuple{S_rem...}, T, D-1, L_rem}(cond_m_data)
        return _cqmom_recursive(Val(D-1), N_tuple[2:end], cond_m_tensor)
    end
    
    # 5. 组合结果
    final_nodes_m = zero(MMatrix{D, total_nodes, T})
    final_weights_v = zero(MVector{total_nodes, T})
    
    for alpha in 1:N1
        nodes_rem, weights_rem = all_res[alpha]
        for beta in 1:total_nodes_remaining
            idx = (alpha - 1) * total_nodes_remaining + beta
            final_nodes_m[1, idx] = xi1[alpha]
            for d in 2:D
                final_nodes_m[d, idx] = nodes_rem[d-1, beta]
            end
            final_weights_v[idx] = w1[alpha] * weights_rem[beta]
        end
    end
    
    return SMatrix{D, total_nodes, T}(final_nodes_m), SVector{total_nodes, T}(final_weights_v)
end
