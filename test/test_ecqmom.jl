using QuadratureMoments
using Test
using StaticArrays
using LinearAlgebra

@testset "ECQMOM (Extended CQMOM)" begin
    @testset "2D Case: Mixed Orders" begin
        # 2D independent distribution
        mx = [1.0, 5.2, 29.0, 170.8, 1051.0] # 2N+1 = 5
        my = [1.0, 10.0, 100.25] # 2N+1 = 3

        m_data = zeros(5, 3)
        for i in 1:5, j in 1:3
            m_data[i, j] = mx[i] * my[j]
        end

        m_static = SMatrix{5,3,Float64}(m_data)

        method = ECQMOM((2, 1), GaussianKernel())
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights
        sigmas = res.sigmas

        @test size(nodes) == (2, 2)
        @test isapprox(sum(weights), 1.0, atol=1e-3)

        # Bandwidth verification (sigma[k, d])
        @test isapprox(sigmas[1, 1], 1.0, atol=1.5e-1)
        @test isapprox(sigmas[1, 2], 0.5, atol=1.5e-1)

        # Node verification
        @test any(isapprox.(nodes[:, 1], 4.0, atol=0.2))
        @test any(isapprox.(nodes[:, 1], 6.0, atol=0.2))
        @test all(isapprox.(nodes[:, 2], 10.0, atol=0.2))
    end
end
