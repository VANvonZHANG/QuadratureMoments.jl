"""
    CQMOM{D, N}
    
多变量条件矩反演 (Conditional Quadrature Method of Moments)。
D: 维度。
N: 每一维的节点数 (Tuple{Int1, Int2, ...})。
"""
struct CQMOM{D, N} end

"""
    invert_moments(method::CQMOM{2, N}, m::AbstractMatrix{T}) -> (nodes, weights)

CQMOM 2D 实现的原型。
输入 m 为 (2N1) x (2N2) 的矩矩阵。
返回: nodes (Matrix, D x TotalNodes), weights (Vector)
"""
function invert_moments(::CQMOM{2, N}, m::AbstractMatrix{T}) where {N, T}
    N1, N2 = N
    # 1. 第一维边缘反演 (Marginal 1)
    # 提取 m[i, 1] 即 m_{i, 0}
    m1 = m[:, 1]
    xi1, w1 = wheeler_inversion(m1)
    
    total_nodes = N1 * N2
    final_nodes = zeros(T, 2, total_nodes)
    final_weights = zeros(T, total_nodes)
    
    # 2. 构造第一维的 Vandermonde 矩阵 V
    # V_{i, alpha} = w_{1, alpha} * xi_{1, alpha}^(i-1)
    V = zeros(T, N1, N1)
    for i in 1:N1
        for alpha in 1:N1
            V[i, alpha] = w1[alpha] * (xi1[alpha]^(i-1))
        end
    end
    
    # 3. 对每一列 j (第二维的矩阶数)，求解条件矩 m_{j | alpha}
    # m_{i, j} = sum_alpha (V_{i, alpha} * m_{j | alpha})
    # 我们有 2*N2 个这样的系统 (j = 0 to 2N2-1)
    # 注意: j=0 时条件矩应为 1 (因为 m_{i,0} 已经由边缘分布提供)
    
    # 预分配条件矩矩阵 [alpha, j]
    cond_moments = zeros(T, N1, 2*N2)
    
    # 使用 LU 分解加速批量求解
    V_fact = lu(V)
    for j in 1:2*N2
        # 解 V * x = m[:, j]
        cond_moments[:, j] = V_fact \ m[1:N1, j]
    end
    
    # 4. 对每个第一维节点 alpha，反演其条件矩序列
    for alpha in 1:N1
        m_cond = cond_moments[alpha, :]
        xi2, w2 = wheeler_inversion(m_cond)
        
        for beta in 1:N2
            idx = (alpha - 1) * N2 + beta
            final_nodes[1, idx] = xi1[alpha]
            final_nodes[2, idx] = xi2[beta]
            final_weights[idx] = w1[alpha] * w2[beta]
        end
    end
    
    return final_nodes, final_weights
end
