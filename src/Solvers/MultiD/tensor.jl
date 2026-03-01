# src/Solvers/MultiD/tensor.jl
using LinearAlgebra
using StaticArrays

"""
    TensorQMOM{D, N_tuple, N_total} <: AbstractQBMM{D, N_total}
"""
struct TensorQMOM{D, N_tuple, N_total} <: AbstractQBMM{D, N_total} end

# 构造函数
TensorQMOM(N::NTuple{D, Int}) where D = TensorQMOM{D, N, prod(N)}()
TensorQMOM(D::Int, N_per_dim::Int) = TensorQMOM(ntuple(i -> N_per_dim, D))

"""
    invert_moments(method::TensorQMOM, marginal_moments::NTuple{D, SVector}; backend=NativeBackend())
"""
function invert_moments(
    ::TensorQMOM{D, N_tuple, N_total}, 
    marginal_m::NTuple{D, SVector{L, T}}; 
    backend::AbstractMathBackend = NativeBackend()
) where {D, N_tuple, N_total, L, T}
    
    # 1. 独立反演每一维
    res_1d = ntuple(Val(D)) do d
        # 使用第 d 维的边缘矩进行 Wheeler 反演
        # 这里需要注意的是每个维度的 N 可能不同
        N_d = N_tuple[d]
        invert_moments(Wheeler{N_d}(), marginal_m[d]; backend=backend)
    end
    
    # 2. 笛卡尔积组合
    final_nodes = MMatrix{N_total, D, T}(undef)
    final_weights = MVector{N_total, T}(undef)
    
    indices = CartesianIndices(N_tuple)
    for (idx, cid) in enumerate(indices)
        w_prod = one(T)
        for d in 1:D
            node_idx = cid.I[d]
            # 填充节点矩阵：第 idx 个组合节点，第 d 维
            final_nodes[idx, d] = res_1d[d].nodes[node_idx, 1]
            # 累乘权重
            w_prod *= res_1d[d].weights[node_idx]
        end
        final_weights[idx] = w_prod
    end
    
    return QuadratureResult(SVector{N_total, T}(final_weights), SMatrix{N_total, D, T}(final_nodes), nothing)
end

"""
    invert_moments(method::TensorQMOM, m::SArray; backend=NativeBackend())
"""
function invert_moments(method::TensorQMOM{D, N_tuple, N_total}, m::SArray{S, T, D}; backend::AbstractMathBackend=NativeBackend()) where {D, N_tuple, N_total, S, T}
    # 提取每一维的边缘矩
    marginal_m = ntuple(Val(D)) do d
        Ld = 2 * N_tuple[d]
        ntuple(k -> m[ntuple(i -> (i == d ? k : 1), Val(D))...], Val(Ld))
    end
    
    marginal_m_vecs = ntuple(d -> SVector{length(marginal_m[d]), T}(marginal_m[d]), Val(D))
    return invert_moments(method, marginal_m_vecs; backend=backend)
end
