using LinearAlgebra
using StaticArrays

"""
    TensorQMOM{D, N}
    
张量积矩反演 (Tensor-product Quadrature Method of Moments)。
通过对每一维独立求积并进行笛卡尔积组合得到多维节点。
D: 维度 (Integer)。
N: 每一维的节点数 (NTuple{D, Int})。
"""
struct TensorQMOM{D, N} <: AbstractQBMM end

# 构造函数
TensorQMOM(D::Int, N::Int) = TensorQMOM{D, ntuple(i -> N, D)}()
TensorQMOM(N::NTuple{D, Int}) where D = TensorQMOM{D, N}()

"""
    invert_moments(method::TensorQMOM{D, N}, marginal_moments::NTuple{D, SVector})
    
直接输入各维度的边缘矩向量，返回张量积组合后的节点和权重。
这是 TensorQMOM 最常用的调用方式。
"""
function invert_moments(::TensorQMOM{D, N}, marginal_m::NTuple{D, SVector{L, T}}) where {D, N, L, T}
    # 1. 独立反演每一维
    # 使用 ntuple 确保在编译期展开
    res_1d = ntuple(Val(D)) do d
        # 使用第 d 维的边缘矩进行 Wheeler 反演
        # 注意：每一维的节点数 N[d] 可能不同
        # 这里 wheeler_inversion 会自动推导 N
        wheeler_inversion(marginal_m[d])
    end
    
    # 提取各维节点和权重
    nodes_1d = ntuple(d -> res_1d[d][1], Val(D))
    weights_1d = ntuple(d -> res_1d[d][2], Val(D))
    
    # 2. 笛卡尔积组合
    total_nodes = prod(N)
    final_nodes_m = zero(MMatrix{D, total_nodes, T})
    final_weights_v = zero(MVector{total_nodes, T})
    
    # 使用 CartesianIndices 遍历 N1 x N2 x ... x ND 的所有组合
    indices = CartesianIndices(N)
    for (idx, cid) in enumerate(indices)
        w_prod = one(T)
        for d in 1:D
            node_idx = cid.I[d]
            # 填充节点矩阵：第 d 维，第 idx 个组合节点
            final_nodes_m[d, idx] = nodes_1d[d][node_idx]
            # 累乘权重
            w_prod *= weights_1d[d][node_idx]
        end
        final_weights_v[idx] = w_prod
    end
    
    return SMatrix{D, total_nodes, T}(final_nodes_m), SVector{total_nodes, T}(final_weights_v)
end

"""
    invert_moments(method::TensorQMOM{D, N}, m::SArray)
    
输入 D 维矩张量，仅提取其边缘矩进行张量积反演。
"""
function invert_moments(method::TensorQMOM{D, N}, m::SArray{S, T, D}) where {D, N, S, T}
    # 提取每一维的边缘矩 [m_{0,0,...}, m_{k,0,...}]
    marginal_m = ntuple(Val(D)) do d
        # 对于第 d 维，我们需要提取 2*N[d] 个矩
        Ld = 2 * N[d]
        ntuple(k -> m[ntuple(i -> (i == d ? k : 1), Val(D))...], Val(Ld))
    end
    
    # 转换为 SVector 并调用主逻辑
    marginal_m_vecs = ntuple(d -> SVector{length(marginal_m[d]), T}(marginal_m[d]), Val(D))
    return invert_moments(method, marginal_m_vecs)
end
