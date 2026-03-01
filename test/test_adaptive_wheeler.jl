# QBMM.jl/test/test_adaptive_wheeler.jl
using Test
using QBMM
using StaticArrays
using LinearAlgebra

@testset "Adaptive Wheeler Inversion" begin
    # 此测试验证 Wheeler 算法在处理“退化矩序列”时的自适应能力。
    # 例如：当分布实际上只有一个 Delta 函数时，Wheeler(2) 应该自动降阶为 1。
    
    @testset "Degenerate Case: Single Node (Delta Distribution)" begin
        # 实际分布：w=1.0, xi=5.0
        # 矩序列：m_k = 1.0 * 5.0^k
        m = @SVector [1.0, 5.0, 25.0, 125.0] # 2 节点所需的 4 个矩
        
        # 尝试使用 2 节点 Wheeler 反演
        method = Wheeler(2)
        res = invert_moments(method, m)
        
        # 自适应逻辑应返回 1 个有效节点
        # 由于静态数组长度固定，我们会检查权重分布
        @test count(w -> w > 1e-10, res.weights) == 1
        @test sum(res.weights) ≈ 1.0
        
        # 有效节点的位置应该是 5.0
        valid_idx = findfirst(w -> w > 1e-10, res.weights)
        @test res.nodes[valid_idx, 1] ≈ 5.0
    end

    @testset "Near-Degenerate Case: Numerical Noise" begin
        # 构造一个几乎退化但带有极小扰动的序列
        # 这种序列会导致 Wheeler 算法中的 sigma 表项或 b_k 极小
        m = SVector{6, Float64}(1.0, 2.0, 4.0, 8.0, 16.0, 32.000000000001)
        
        method = Wheeler(3)
        res = invert_moments(method, m)
        
        # 应自动降阶处理
        @test count(w -> w > 1e-8, res.weights) < 3
        @test sum(res.weights) ≈ 1.0
    end
end
