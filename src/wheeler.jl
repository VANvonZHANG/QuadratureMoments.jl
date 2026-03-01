using LinearAlgebra
using StaticArrays

"""
    Wheeler <: AbstractQBMM
    
使用 Adaptive Wheeler 算法的 1D 矩反演。
"""
struct Wheeler <: AbstractQBMM end

"""
    invert_moments(method::Wheeler, m::AbstractVector)
"""
function invert_moments(::Wheeler, m::AbstractVector{T}) where T
    return wheeler_inversion(m)
end

function invert_moments(::Wheeler, m::SVector{L, T}) where {L, T}
    return wheeler_inversion(m)
end

"""
    wheeler_inversion(m::AbstractVector{T}; tol=1e-14) -> (nodes, weights)

使用 Adaptive Wheeler 算法执行单变量矩反演。
如果矩序列退化（如方差为0），算法会自动降低节点数 N 并返回精确解。
"""
function wheeler_inversion(m::AbstractVector{T}; tol=1e-14) where T
    N_max = length(m) ÷ 2
    @assert length(m) >= 2N_max "Moments count must be at least 2N."
    
    # 提前检查 m0
    if m[1] <= tol
        return T[], T[]
    end

    a = zeros(T, N_max)
    b = zeros(T, N_max)
    # sigma 表: [row, col] -> sigma[k, i]
    # 书中索引: sigma_{k, i}, k 是行(0..N), i 是列(k..2N-k-1)
    # 这里使用 1-based: row 1..N+1, col 1..2N
    sig = zeros(T, N_max + 1, 2 * N_max)
    
    # 初始化
    for i in 1:2*N_max
        sig[1, i] = 0.0 # sigma_{-1, i}
        sig[2, i] = m[i] # sigma_{0, i}
    end
    
    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    actual_N = N_max
    for k in 1:N_max-1
        row = k + 2
        # 计算 sigma_{k, i}
        for i in k:(2*N_max - k - 1)
            col = i + 1
            sig[row, col] = sig[row-1, col+1] - a[k] * sig[row-1, col] - b[k] * sig[row-2, col]
        end
        
        # 检查分母是否过小 (Adaptive 逻辑)
        if abs(sig[row, k+1]) < tol || abs(sig[row-1, k]) < tol
            actual_N = k
            break
        end
        
        a[k+1] = sig[row, k+2] / sig[row, k+1] - sig[row-1, k+1] / sig[row-1, k]
        b[k+1] = sig[row, k+1] / sig[row-1, k]
        
        # 检查 b 是否为负 (数值不稳定性)
        if b[k+1] < 0
            # 尝试修复或截断
            if b[k+1] > -tol
                b[k+1] = 0.0
            else
                actual_N = k
                break
            end
        end
    end
    
    # 构建实际维度的 Jacobi 矩阵
    J = SymTridiagonal(a[1:actual_N], sqrt.(abs.(b[2:actual_N])))
    eigen_decomp = eigen(J)
    
    nodes = eigen_decomp.values
    weights = m[1] .* (eigen_decomp.vectors[1, :] .^ 2)
    
    return nodes, weights
end

"""
    wheeler_inversion(m::SVector{L, T}; tol=1e-14)
    
静态分派版本，默认调用自适应逻辑。
"""
function wheeler_inversion(m::SVector{L, T}; tol=1e-14) where {L, T}
    return _adaptive_wheeler_static(m, Val(L ÷ 2), tol)
end

function _adaptive_wheeler_static(m::SVector{L, T}, ::Val{N_max}, tol::T) where {L, T, N_max}
    a = MVector{N_max, T}(undef)
    b = MVector{N_max, T}(undef)
    sig = zero(MMatrix{N_max+1, L, T})
    
    for i in 1:L
        sig[2, i] = m[i]
    end
    
    if abs(m[1]) < tol
        return zero(SVector{N_max, T}), zero(SVector{N_max, T})
    end

    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    actual_N = N_max
    for k in 1:N_max-1
        row = k + 2
        for i in k:(L - k - 1)
            col = i + 1
            sig[row, col] = sig[row-1, col+1] - a[k] * sig[row-1, col] - b[k] * sig[row-2, col]
        end
        
        # Adaptive Check
        if abs(sig[row, k+1]) < tol || abs(sig[row-1, k]) < tol
            actual_N = k
            break
        end
        
        new_a = sig[row, k+2] / sig[row, k+1] - sig[row-1, k+1] / sig[row-1, k]
        new_b = sig[row, k+1] / sig[row-1, k]
        
        if new_b < -tol
            actual_N = k
            break
        elseif new_b < 0
            new_b = 0.0
        end
        
        a[k+1] = new_a
        b[k+1] = new_b
    end
    
    # 为了保持静态返回类型一致，我们填充剩余节点为非常大的值且权重为0
    # 或者对于 StaticArrays，我们依然返回 N_max 长度，但失效节点权重设为0
    # 另一种做法是根据 actual_N 进行二次分派，但这会增加编译开销
    
    # 这里我们采用“降阶填充”策略：前 actual_N 个是真实解
    J_dense = zero(MMatrix{N_max, N_max, T})
    for i in 1:actual_N
        J_dense[i, i] = a[i]
    end
    for i in 1:actual_N-1
        val = sqrt(abs(b[i+1]))
        J_dense[i, i+1] = val
        J_dense[i+1, i] = val
    end
    # 对于 i > actual_N，保持对角线为 0 或某个标记值，非对角线为 0
    # 这样 eigen 分解后，这些“虚假”节点会出现在 0 附近
    
    eigen_decomp = eigen(Symmetric(SMatrix{N_max, N_max, T}(J_dense)))
    
    # 权重处理：只有对应前 actual_N 个特征值的特征向量分量才有意义
    # 但实际上，eigen(Symmetric(J)) 对于分块对角矩阵会自动处理
    # 我们需要确保返回的结果中，只有 actual_N 个节点的权重非零
    
    # 更加优雅的做法：
    # 如果 actual_N < N_max，我们将那些多出来的权重强行设为 0
    # 注意：Symmetric(J_dense) 的前 actual_N 行列是独立的。
    
    nodes = SVector{N_max, T}(eigen_decomp.values)
    raw_weights = ntuple(i -> m[1] * eigen_decomp.vectors[1, i]^2, Val(N_max))
    
    # 逻辑过滤：由于我们只填充了前 actual_N 的矩阵，
    # 只有前 actual_N 个特征向量是有意义的。
    # 我们需要找到哪些特征值是来自有效部分的。
    # 实际上在对称三对角阵中，这会自动工作。
    
    return nodes, SVector{N_max, T}(raw_weights)
end
