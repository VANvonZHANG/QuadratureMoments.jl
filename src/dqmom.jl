using LinearAlgebra
using StaticArrays

"""
    dqmom_matrix(nodes::SVector{N, T}) -> SMatrix{2N, 2N, T}

为直接积分矩方法 (DQMOM) 构建线性系统系数矩阵 A。
矩阵 A 的维度为 2N x 2N，它由节点 (nodes) 决定。
方程式形式： A * [a; b] = S
其中 a_i = dw_i/dt, b_i = d(w_i * xi_i)/dt。
"""
function dqmom_matrix(nodes::SVector{N, T}) where {N, T}
    A_elements = zeros(MMatrix{2N, 2N, T})
    
    for k in 0:(2N - 1)
        row = k + 1
        for i in 1:N
            col_a = i
            col_b = i + N
            
            # 第一部分：对应 a_i 的系数 (1 - k) * xi_i^k
            if k == 0
                A_elements[row, col_a] = one(T)
            else
                A_elements[row, col_a] = (1 - k) * (nodes[i]^k)
            end
            
            # 第二部分：对应 b_i 的系数 k * xi_i^(k-1)
            if k == 0
                A_elements[row, col_b] = zero(T)
            elseif k == 1
                A_elements[row, col_b] = one(T)
            else
                A_elements[row, col_b] = k * (nodes[i]^(k - 1))
            end
        end
    end
    
    return SMatrix{2N, 2N, T}(A_elements)
end

"""
    dqmom_matrix(nodes::AbstractVector{T}) -> Matrix{T}

为通用动态数组构建 DQMOM 矩阵。
"""
function dqmom_matrix(nodes::AbstractVector{T}) where T
    N = length(nodes)
    A = zeros(T, 2N, 2N)
    
    for k in 0:(2N - 1)
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

"""
    dqmom_solve(nodes::AbstractVector{T}, source_terms::AbstractVector{T}) -> (a, b)

给定节点和由物理模型计算出的前 2N 阶矩源项 S_k，
求解 DQMOM 线性系统，返回权重演化率 a 和加权节点演化率 b。
"""
function dqmom_solve(nodes::SVector{N, T}, source_terms::SVector{L, T}) where {N, L, T}
    @assert L == 2N "Number of source terms must be exactly 2N."
    A = dqmom_matrix(nodes)
    # 对于小规模固定矩阵，使用 \ (会调用 LU 分解的优化版本)
    x = A \ source_terms
    
    a = SVector{N, T}(x[1:N])
    b = SVector{N, T}(x[N+1:2N])
    return a, b
end

function dqmom_solve(nodes::AbstractVector{T}, source_terms::AbstractVector{T}) where T
    N = length(nodes)
    @assert length(source_terms) == 2N "Number of source terms must be exactly 2N."
    A = dqmom_matrix(nodes)
    
    # 动态数组的大型系统求解
    x = A \ source_terms
    
    return x[1:N], x[N+1:2N]
end
