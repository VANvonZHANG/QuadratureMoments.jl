using QBMM
using Test
using StaticArrays

@testset "Adaptive Wheeler Inversion" begin
    @testset "Degenerate Case: Single Node (Delta Distribution)" begin
        # 一个完美的 Delta 函数分布：x=5.0, w=1.0
        # 即使我们要求 N=3，它也应该降阶返回精确的一个节点
        m = SVector{6, Float64}(1.0, 5.0, 25.0, 125.0, 625.0, 3125.0)
        
        nodes, weights = wheeler_inversion(m)
        
        # 应该有一个节点在 5.0，权重为 1.0。其余权重应接近 0。
        @test any(isapprox.(nodes, 5.0, atol=1e-10))
        @test isapprox(sum(weights), 1.0, atol=1e-10)
        
        # 过滤掉权重接近 0 的节点
        valid_indices = findall(w -> w > 1e-10, weights)
        @test length(valid_indices) == 1
        @test isapprox(nodes[valid_indices[1]], 5.0, atol=1e-10)
    end

    @testset "Near-Degenerate Case: Numerical Noise" begin
        # 模拟非常小的方差
        mu = 10.0
        sigma_sq = 1e-12
        # m0=1, m1=10, m2=100+1e-12, m3=1000+3*10*1e-12...
        m = SVector{4, Float64}(1.0, 10.0, 100.0 + sigma_sq, 1000.0 + 3*10*sigma_sq)
        
        nodes, weights = wheeler_inversion(m)
        @test isapprox(sum(weights), 1.0, atol=1e-10)
        @test isapprox(sum(weights .* nodes) / sum(weights), 10.0, atol=1e-10)
    end
end
