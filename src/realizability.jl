"""
    is_realizable(m::AbstractVector{T}) -> Bool

检查矩序列的实现性（Realizability）。
对于单变量分布，必要且充分条件是 Hankel 矩阵 H_n = [m_{i+j}] 为正定矩阵。
"""
function is_realizable(m::AbstractVector{T}) where T
    L = length(m)
    N = (L ÷ 2) - 1
    # 构建 Hankel 矩阵 (N+1)x(N+1)
    # H[i, j] = m[i+j-1]
    H = zeros(T, N+1, N+1)
    for i in 1:N+1
        for j in 1:N+1
            idx = i + j - 2 + 1 # Julia 1-based indexing
            H[i, j] = m[idx]
        end
    end
    
    # 对于 AbstractVector，Cholesky 失败可能意味着它是半正定的
    # 在 QBMM 中，半正定（奇异）通常对应于 Delta 函数分布，也是可实现的
    try
        vals = eigvals(Symmetric(H))
        return all(vals .> -sqrt(eps(T)))
    catch e
        return false
    end
end

"""
    is_realizable(m::SVector{L, T}) -> Bool

针对小规模矩序列优化的静态数组实现。
"""
function is_realizable(m::SVector{L, T}) where {L, T}
    N_plus_1 = L ÷ 2
    
    # 使用 MMatrix 构造以避免分配
    H_M = zeros(MMatrix{N_plus_1, N_plus_1, T})
    for i in 1:N_plus_1
        for j in 1:N_plus_1
            idx = i + j - 2 + 1
            H_M[i, j] = m[idx]
        end
    end
    
    H = SMatrix{N_plus_1, N_plus_1, T}(H_M)
    
    # 对于小矩阵，检查特征值是否全部大于零（带微小裕量）
    try
        # 对于边界情况（如单点分布），最小特征值可能接近 0
        vals = eigvals(Symmetric(H))
        return all(vals .> -sqrt(eps(T))) 
    catch e
        return false
    end
end
