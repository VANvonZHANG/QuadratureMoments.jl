using Test
using QuadratureMoments
using StaticArrays

@testset "Physics kernel library" begin
    @testset "Aggregation kernels" begin
        @test Constant(1.2)(2.0, 3.0) == 1.2
        @test Sum(1.0)(2.0, 3.0) == 5.0
        @test Brownian(1.0)(2.0, 3.0) ≈ (2.0 + 3.0)^2 / (2.0 * 3.0)
    end

    @testset "Aggregation(Brownian, LengthBased) end-to-end" begin
        nodes = @SVector [1.0, 2.0, 4.0]
        weights = @SVector [0.4, 0.35, 0.25]
        agg = Aggregation(Brownian(0.5), LengthBased())
        S = compute_source_terms(agg, nodes, weights, Val(4))
        @test all(isfinite, S)
        # Volume (m_3) conserved under LengthBased aggregation for any symmetric kernel:
        @test S[4] ≈ 0.0 atol=1e-10
    end
end
