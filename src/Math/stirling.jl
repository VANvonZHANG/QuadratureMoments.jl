# src/Math/stirling.jl
using StaticArrays

# 预计算 Stirling 2 矩阵 (仅针对核心计算 D=1..10)
# S(n, k) = k*S(n-1, k) + S(n-1, k-1)
# 用于 EQMOM 矩修正： m_k* = sum_{j=0}^k S(k, j) * sigma^(k-j) * m_j
const STIRLING2_TABLE = SMatrix{11, 11, Int}([
    1 0 0 0 0 0 0 0 0 0 0;
    0 1 0 0 0 0 0 0 0 0 0;
    0 1 1 0 0 0 0 0 0 0 0;
    0 1 3 1 0 0 0 0 0 0 0;
    0 1 7 6 1 0 0 0 0 0 0;
    0 1 15 25 10 1 0 0 0 0 0;
    0 1 31 90 65 15 1 0 0 0 0;
    0 1 63 301 350 140 21 1 0 0 0;
    0 1 127 966 1701 1050 266 28 1 0 0;
    0 1 255 3025 7770 6951 2646 462 36 1 0;
    0 1 511 9330 34105 42525 22827 5880 750 45 1
])

"""
    stirling2(n::Int, k::Int, ::NativeBackend) -> Int
查询预计算好的静态表。
"""
@inline function stirling2(n::Int, k::Int, ::NativeBackend)
    if n < 0 || k < 0 || n > 10 || k > 10
        return 0
    end
    return STIRLING2_TABLE[n+1, k+1]
end

"""
    stirling2(n::Int, k::Int, ::ExternalBackend) -> Int
"""
function stirling2(n::Int, k::Int, ::ExternalBackend)
    # 如果已安装 Combinatorics.jl，则调用。
    # 这里为了保持零分配，如果没安装可以退回到 Native。
    # 用户可以在 ExternalBackend 下手动安装 Combinatorics。
    return stirling2(n, k, NativeBackend())
end
