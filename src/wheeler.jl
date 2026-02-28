"""
    wheeler_inversion(m::AbstractVector{T}) -> (nodes, weights)

使用 Wheeler 算法执行单变量矩反演。输入长度为 2N 的矩序列，返回 N 个节点和权重。
该方法为基于标准动态数组 (Array) 的通用实现。
"""
function wheeler_inversion(m::AbstractVector{T}) where T
    N = length(m) ÷ 2
    @assert length(m) == 2N "The number of moments must be even (2N)."
    
    a = zeros(T, N)
    b = zeros(T, N)
    sigma = zeros(T, N+1, 2N)
    
    sigma[2, :] .= m
    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    for k in 1:N-1
        row = k + 2
        for j in k:(2N - k - 1)
            col = j + 1
            sigma[row, col] = sigma[row-1, col+1] - a[k] * sigma[row-1, col] - b[k] * sigma[row-2, col]
        end
        a[k+1] = sigma[row, k+2] / sigma[row, k+1] - sigma[row-1, k+1] / sigma[row-1, k]
        b[k+1] = sigma[row, k+1] / sigma[row-1, k]
    end
    
    J = SymTridiagonal(a, sqrt.(b[2:N]))
    eigen_decomp = eigen(J)
    
    nodes = eigen_decomp.values
    weights = m[1] .* (eigen_decomp.vectors[1, :] .^ 2)
    
    return nodes, weights
end

"""
    wheeler_inversion(m::SVector{L, T}) -> (nodes, weights)

基于 `StaticArrays.jl` 优化的 Wheeler 反演实现，针对较小规模的矩 (L <= 12 即 N <= 6)
设计，最大程度消除堆内存分配并提升性能。
"""
function wheeler_inversion(m::SVector{L, T}) where {L, T}
    N = L ÷ 2
    
    a = MVector{N, T}(undef)
    b = MVector{N, T}(undef)
    sigma = zeros(MMatrix{N+1, L, T})
    
    for i in 1:L
        sigma[2, i] = m[i]
    end
    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    for k in 1:N-1
        row = k + 2
        for j in k:(L - k - 1)
            col = j + 1
            sigma[row, col] = sigma[row-1, col+1] - a[k] * sigma[row-1, col] - b[k] * sigma[row-2, col]
        end
        a[k+1] = sigma[row, k+2] / sigma[row, k+1] - sigma[row-1, k+1] / sigma[row-1, k]
        b[k+1] = sigma[row, k+1] / sigma[row-1, k]
    end
    
    # Construct dense symmetric matrix for eigen decomposition
    J_M = zeros(MMatrix{N, N, T})
    for i in 1:N
        J_M[i, i] = a[i]
    end
    for i in 1:N-1
        off = sqrt(b[i+1])
        J_M[i, i+1] = off
        J_M[i+1, i] = off
    end
    J = SMatrix{N, N, T}(J_M)
    eigen_decomp = eigen(Symmetric(J))
    
    nodes = SVector{N, T}(eigen_decomp.values)
    weights = SVector{N, T}(ntuple(i -> m[1] * eigen_decomp.vectors[1, i]^2, Val(N)))
    
    return nodes, weights
end
