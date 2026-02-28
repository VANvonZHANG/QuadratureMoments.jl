# QBMM.jl 核心测试套件
using Test
using LinearAlgebra
using StaticArrays

# 注意：目前 wheeler.jl 尚未实现，测试将作为驱动开发的“红灯”阶段。
@testset "Wheeler Algorithm Verification" begin
    # 已知分布测试：均值为 1，方差为 0.1 的正态分布
    # 前 4 个解析矩：m0=1, m1=1, m2=1.1, m3=1.3 (对于 N=2 的测试)
    # 计算公式：m_k = E[X^k]
    # m0 = 1
    # m1 = μ = 1
    # m2 = σ² + μ² = 0.1 + 1.0 = 1.1
    # m3 = μ³ + 3μσ² = 1.0 + 3*1*0.1 = 1.3
    
    m = @SVector [1.0, 1.0, 1.1, 1.3]
    
    # 预测应得到的节点和权重 (N=2)
    # 对于 N=2，理论上的节点应该分布在均值 1 的两侧
    
    @testset "StaticArrays Version" begin
        nodes, weights = wheeler_inversion(m)
        @test sum(weights) ≈ 1.0
        @test dot(weights, nodes) ≈ 1.0
        @test dot(weights, nodes.^2) ≈ 1.1
        @test dot(weights, nodes.^3) ≈ 1.3
    end
    
    @testset "Base Array Version" begin
        m_base = [1.0, 1.0, 1.1, 1.3]
        nodes, weights = wheeler_inversion(m_base)
        @test sum(weights) ≈ 1.0
        @test dot(weights, nodes) ≈ 1.0
        @test dot(weights, nodes.^2) ≈ 1.1
        @test dot(weights, nodes.^3) ≈ 1.3
    end
end
