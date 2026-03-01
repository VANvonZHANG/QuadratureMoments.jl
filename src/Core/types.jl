# src/Core/types.jl

"""
    AbstractMathBackend
    NativeBackend <: AbstractMathBackend
    ExternalBackend <: AbstractMathBackend

用于派发不同的数学计算后端。
"""
abstract type AbstractMathBackend end
struct NativeBackend <: AbstractMathBackend end   # 极致性能，StaticArrays 实现
struct ExternalBackend <: AbstractMathBackend end # 官方/通用库实现

"""
    AbstractQBMM{D, N}

所有 QBMM 算法的基类。
D: 维度，N: 节点数 (可以是整数或 Tuple)。
"""
abstract type AbstractQBMM{D, N} end

"""
    QuadratureResult{D, N, T}

统一的反演结果结构体。
- weights: 节点权重，大小为 N
- nodes: 节点位置，大小为 (N, D)
- sigmas: 连续核的带宽，大小为 (N, D)，离散核为 Nothing
"""
struct QuadratureResult{D, N, T}
    weights::SVector{N, T}
    nodes::SMatrix{N, D, T}
    sigmas::Union{Nothing, SMatrix{N, D, T}}
end

# 针对 N 是 Tuple 的情况 (如 CQMOM) 的辅助构造函数
# 将元组展开为总节点数
function QuadratureResult(w::SVector{N, T}, n::SMatrix{N, D, T}, s=nothing) where {D, N, T}
    return QuadratureResult{D, N, T}(w, n, s)
end
