using QBMM
using Test
using StaticArrays
using LinearAlgebra

@testset "Brute-force QMOM" begin
    @testset "2D Case: Convergence" begin
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
        m_static = SMatrix{4,4,Float64}(m_data)

        method = BruteQMOM(2, 2)
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (2, 2)
        @test isapprox(sum(weights), 1.0, atol=1e-5)

        # Verify moment reconstruction
        for i in 1:2, j in 1:2
            pred = sum(
                weights[alpha] * nodes[alpha, 1]^(i-1) * nodes[alpha, 2]^(j-1) for
                alpha in 1:2
            )
            @test isapprox(pred, m_data[i, j], atol=1e-8)
        end
    end
end
