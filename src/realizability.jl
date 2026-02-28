"""
    is_realizable(m::AbstractVector{T}) -> Bool

检查给定的一维矩序列是否满足实现性 (Realizability) 条件。
通过构建 Hankel 矩阵并验证其是否为正定矩阵来进行判断。
"""
function is_realizable(m::AbstractVector{T}) where T
    L = length(m)
    N = (L + 1) ÷ 2
    H = zeros(T, N, N)
    for i in 1:N
        for j in 1:N
            H[i, j] = m[i + j - 1]
        end
    end
    return isposdef(Symmetric(H))
end

"""
    is_realizable(m::SVector{L, T}) -> Bool

`is_realizable` 的 `StaticArrays` 零内存分配优化实现。
"""
function is_realizable(m::SVector{L, T}) where {L, T}
    N = (L + 1) ÷ 2
    # 构造 SMatrix：按列优先存储
    H_data = ntuple(Val(N * N)) do idx
        j = (idx - 1) ÷ N + 1
        i = (idx - 1) % N + 1
        m[i + j - 1]
    end
    H = SMatrix{N, N, T}(H_data)
    return isposdef(Symmetric(H))
end
