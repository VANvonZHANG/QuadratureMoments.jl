# QBMM.jl/test/test_dqmom_forwarddiff.jl
using Test
using QBMM
using StaticArrays
using ForwardDiff
using LinearAlgebra

@testset "DQMOM ForwardDiff Integration" begin
    # 此测试验证 DQMOM 求解器是否能与 ForwardDiff 配合使用，
    # 这在需要对 DQMOM 演化率进行雅可比计算（如刚性 ODE 求解）时非常重要。

    # 模拟一个简单的源项函数，它依赖于节点
    function compute_source_terms(nodes::SVector{N,T}) where {N,T}
        # S_k = k * m_{k-1} * (some_growth_rate)
        # 简化模型：S_k = k * sum(nodes)
        return SVector{2N,T}(ntuple(k -> T(k-1) * sum(nodes), Val(2N)))
    end

    function dqmom_rhs(v::SVector{L,T}) where {L,T}
        N = L ÷ 2
        # v = [w1, w2, w1*xi1, w2*xi2]
        weights = SVector{N,T}(ntuple(i -> v[i], Val(N)))
        weighted_nodes = SVector{N,T}(ntuple(i -> v[i + N], Val(N)))

        # 提取节点 xi_i = (w_i * xi_i) / w_i
        nodes = SVector{N,T}(ntuple(i -> weighted_nodes[i] / (weights[i] + eps(T)), Val(N)))

        # 计算源项
        S = compute_source_terms(nodes)

        # 求解 DQMOM 系统得到 [dw/dt; d(w*xi)/dt]
        # 注意：这里需要传入 DQMOM(N)
        da, db = dqmom_solve(DQMOM(N), nodes, S)

        return vcat(da, db)
    end

    # 初始化状态
    v0 = SVector{4,Float64}(0.5, 0.5, 0.5*1.0, 0.5*2.0)

    # 1. 基本运行检查
    rhs0 = dqmom_rhs(v0)
    @test length(rhs0) == 4

    # 2. 雅可比检查 (ForwardDiff)
    # 如果代码包含类型不稳定或非通用类型，这里会报错
    Jac = ForwardDiff.jacobian(dqmom_rhs, v0)
    @test size(Jac) == (4, 4)
    @test !any(isnan.(Jac))

    @testset "Dual-Backend dispatch inside AD" begin
        # 验证 ExternalBackend 是否也能在 AD 中工作
        function dqmom_rhs_ext(v::Vector{T}) where {T}
            N = length(v) ÷ 2
            weights = v[1:N]
            wnodes = v[(N + 1):end]
            nodes = wnodes ./ (weights .+ eps(T))
            S = collect(compute_source_terms(SVector{N,T}(nodes)))
            da, db = dqmom_solve(DQMOM(N), nodes, S, backend=ExternalBackend())
            return [da; db]
        end

        v0_vec = [0.5, 0.5, 0.5, 1.0]
        Jac_ext = ForwardDiff.jacobian(dqmom_rhs_ext, v0_vec)
        @test size(Jac_ext) == (4, 4)
    end
end
