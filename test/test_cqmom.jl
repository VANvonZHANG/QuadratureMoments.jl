using QBMM
using Test
using StaticArrays
using LinearAlgebra

@testset "CQMOM Recursive Implementation" begin
    @testset "2D Case: Independent Variables" begin
        # 两个独立的正态分布: x ~ N(1, 0.1), y ~ N(2, 0.2)
        N = (2, 2)

        # 计算 2D 独立分布的矩: m[i, j] = m_x[i] * m_y[j]
        mx = [1.0, 1.0, 1.1, 1.3] # 0..3 阶矩
        my = [1.0, 2.0, 4.2, 9.2] # 0..3 阶矩

        m_data = zeros(4, 4)
        for i in 1:4, j in 1:4
            m_data[i, j] = mx[i] * my[j]
        end

        m_static = SMatrix{4,4,Float64}(m_data)

        method = CQMOM(N)
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (4, 2)
        @test length(weights) == 4

        # 验证矩重构
        for i in 1:2, j in 1:2
            pred = sum(weights[k] * nodes[k, 1]^(i-1) * nodes[k, 2]^(j-1) for k in 1:4)
            @test isapprox(pred, m_data[i, j], atol=1e-8)
        end
    end

    @testset "3D Case: Low Order" begin
        N = (1, 1, 1)
        m_data = zeros(2, 2, 2)
        v1, v2, v3 = 1.5, 2.5, 3.5
        for i in 0:1, j in 0:1, k in 0:1
            m_data[i + 1, j + 1, k + 1] = v1^i * v2^j * v3^k
        end

        m_static = SArray{Tuple{2,2,2},Float64,3,8}(m_data)

        method = CQMOM(N)
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (1, 3)
        @test isapprox(vec(nodes[1, :]), [1.5, 2.5, 3.5], atol=1e-8)
        @test isapprox(weights[1], 1.0, atol=1e-8)
    end
end
