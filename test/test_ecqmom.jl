using QBMM
using Test
using StaticArrays
using LinearAlgebra

@testset "ECQMOM (Extended CQMOM)" begin
    @testset "2D Case: Mixed Orders" begin
        # 2D 独立分布
        # 第一维 N1=2: 0.4*N(4,1) + 0.6*N(6,1) -> 5 个矩
        # 第二维 N2=1: 1.0*N(10, 0.25) -> 3 个矩
        
        mx = [1.0, 5.2, 29.0, 170.8, 1051.0] # 2N+1 = 5
        my = [1.0, 10.0, 100.25] # 2N+1 = 3
        
        # 矩张量尺寸 5 x 3
        m_data = zeros(5, 3)
        for i in 1:5, j in 1:3
            m_data[i, j] = mx[i] * my[j]
        end
        
        m_static = SMatrix{5, 3, Float64}(m_data)
        
        # 定义 N=(2, 1)
        method = ECQMOM((2, 1), GaussianKernel())
        nodes, weights, sigmas = invert_moments(method, m_static)
        
        # 节点数应为 2 * 1 = 2
        @test size(nodes) == (2, 2)
        @test isapprox(sum(weights), 1.0, atol=1e-3)
        
        # 带宽验证
        @test isapprox(sigmas[1, 1], 1.0, atol=1.1e-1)
        @test isapprox(sigmas[2, 1], 0.5, atol=1.1e-1)
        
        # 节点验证
        @test any(isapprox.(nodes[1, :], 4.0, atol=0.2))
        @test any(isapprox.(nodes[1, :], 6.0, atol=0.2))
        @test all(isapprox.(nodes[2, :], 10.0, atol=0.2))
    end
end
