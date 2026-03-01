using QBMM
using Test
using StaticArrays
using LinearAlgebra

@testset "Brute-force QMOM" begin
    @testset "2D Case: Convergence" begin
        # N=2 节点，D=2 维度 -> 变量数 P = 2 * (2+1) = 6
        # 我们需要至少 6 个矩。使用 4x4 矩张量（16个矩）提取前 6 个。
        
        # 预设解：
        # w = [0.4, 0.6]
        # xi_1 = [1.0, 2.0]
        # xi_2 = [3.0, 4.0]
        ws_true = [0.4, 0.6]
        x1_true = [1.0, 2.0]
        x2_true = [3.0, 4.0]
        
        m_data = zeros(4, 4)
        for i in 1:4, j in 1:4
            val = 0.0
            for alpha in 1:2
                val += ws_true[alpha] * (x1_true[alpha]^(i-1)) * (x2_true[alpha]^(j-1))
            end
            m_data[i, j] = val
        end
        m_static = SMatrix{4, 4, Float64}(m_data)
        
        method = BruteQMOM(2, 2)
        nodes, weights = invert_moments(method, m_static)
        
        @test size(nodes) == (2, 2)
        @test isapprox(sum(weights), 1.0, atol=1e-5)
        
        # 验证矩重构 (对低阶矩应极度精确)
        for i in 1:2, j in 1:2
            pred = sum(weights[alpha] * nodes[1, alpha]^(i-1) * nodes[2, alpha]^(j-1) for alpha in 1:2)
            @test isapprox(pred, m_data[i, j], atol=1e-8)
        end
    end
end
