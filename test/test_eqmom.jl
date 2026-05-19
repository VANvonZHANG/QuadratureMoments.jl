using QBMM
using Test
using StaticArrays
using LinearAlgebra

@testset "EQMOM Kernel Extensions" begin
    @testset "Gaussian Kernel (Mixture)" begin
        m = SVector{5,Float64}(1.0, 5.2, 29.0, 170.8, 1051.0)

        method = EQMOM(2, GaussianKernel())
        res = invert_moments(method, m)
        nodes = res.nodes
        weights = res.weights
        σ = res.sigmas[1, 1]

        @test size(nodes) == (2, 1)
        @test isapprox(σ, 1.0, atol=1e-2)
        @test isapprox(sum(weights), 1.0, atol=1e-3)
    end

    @testset "Gamma Kernel (Skewed Distribution)" begin
        m = SVector{5,Float64}(1.0, 4.0, 24.0, 192.0, 1920.0)

        method = EQMOM(2, GammaKernel())
        res = invert_moments(method, m)
        nodes = res.nodes
        weights = res.weights
        σ = res.sigmas[1, 1]

        @test size(nodes) == (2, 1)
        @test σ > 0.0
    end

    @testset "Beta Kernel" begin
        m = SVector{5,Float64}(1.0, 0.5, 0.3, 0.2, 0.15)

        method = EQMOM(2, BetaKernel())
        res = invert_moments(method, m)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (2, 1)
        @test all(0.0 .<= nodes .<= 1.0)
    end
end
