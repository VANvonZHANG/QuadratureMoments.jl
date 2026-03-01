# QBMM.jl/src/Math/stirling.jl

using StaticArrays
using Combinatorics: stirlings2 as comb_stirlings2

"""
    stirling2(n::Int, k::Int, ::NativeBackend)
    stirling2(::Val{n}, ::Val{k}, ::NativeBackend)

原生后端实现：针对 QBMM 常用范围 (N <= 12) 的第二类斯特林数查找表。
"""
@inline function stirling2(n::Int, k::Int, backend::NativeBackend)
    # 动态派发到静态版本以获取极致性能
    if 0 <= n <= 12 && 0 <= k <= n
        return _stirling2_static(Val(n), Val(k))
    elseif k < 0 || k > n
        return 0.0
    else
        return Float64(comb_stirlings2(n, k))
    end
end

@generated function _stirling2_static(::Val{n}, ::Val{k}) where {n, k}
    S = [
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 3.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 7.0 6.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 15.0 25.0 10.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 31.0 90.0 65.0 15.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 63.0 301.0 350.0 140.0 21.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 127.0 966.0 1701.0 1050.0 266.0 28.0 1.0 0.0 0.0 0.0 0.0;
        0.0 1.0 255.0 3025.0 7770.0 6951.0 2646.0 462.0 36.0 1.0 0.0 0.0 0.0;
        0.0 1.0 511.0 9330.0 34105.0 42525.0 22827.0 5880.0 750.0 45.0 1.0 0.0 0.0;
        0.0 1.0 1023.0 28501.0 145750.0 246730.0 179487.0 63987.0 11880.0 1155.0 55.0 1.0 0.0;
        0.0 1.0 2047.0 86526.0 611501.0 1379400.0 1323652.0 627396.0 159027.0 22275.0 1705.0 66.0 1.0
    ]
    val = S[n+1, k+1]
    return :($val)
end

"""
    stirling2(n::Int, k::Int, ::ExternalBackend)

外部后端实现：直接调用 Combinatorics.jl 的实现。
"""
function stirling2(n::Int, k::Int, ::ExternalBackend)
    return Float64(comb_stirlings2(n, k))
end
