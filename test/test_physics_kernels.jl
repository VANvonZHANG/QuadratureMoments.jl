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

    @testset "Daughter distributions (MassBased / LengthBased table)" begin
        # Symmetric
        @test daughter_moment(SymmetricFragmentation(), 1, 4.0, MassBased())   ≈ 4.0          # 2^0 * 4
        @test daughter_moment(SymmetricFragmentation(), 0, 4.0, MassBased())   ≈ 2.0          # two daughters
        @test daughter_moment(SymmetricFragmentation(), 3, 4.0, LengthBased()) ≈ 4.0^3        # volume conserved
        # Uniform
        @test daughter_moment(Uniform(), 1, 4.0, MassBased()) ≈ 4.0              # 2*4/(1+1)
        @test daughter_moment(Uniform(), 0, 4.0, MassBased()) ≈ 2.0              # 2*1/(0+1)
        # OneQuarterMassRatio
        @test daughter_moment(OneQuarterMassRatio(), 1, 5.0, MassBased()) ≈ 5.0  # (4+1)*5/5
        # Erosion (needs d0)
        @test daughter_moment(Erosion(1.0), 1, 4.0, MassBased())   ≈ 1.0 + 3.0
        @test daughter_moment(Erosion(1.0), 3, 4.0, LengthBased()) ≈ 1.0 + (4.0^3 - 1.0^3)
        # FullFragmentation (needs d0)
        @test daughter_moment(FullFragmentation(2.0), 0, 4.0, MassBased()) ≈ 4.0 / 2.0
    end

    @testset "Breakage(Constant, SymmetricFragmentation, LengthBased) end-to-end" begin
        nodes = @SVector [1.0, 2.0, 4.0]
        weights = @SVector [0.4, 0.35, 0.25]
        brk = Breakage(xi -> 1.5, SymmetricFragmentation(), LengthBased())
        S = compute_source_terms(brk, nodes, weights, Val(4))
        # Symmetric binary breakage conserves volume (m_3) under LengthBased:
        @test S[4] ≈ 0.0 atol=1e-10
        # Number: each break turns 1 particle into 2, so S_0 = b0 * (2 - 1) * m0 = b0*m0
        @test S[1] ≈ 1.5 * sum(weights) atol=1e-10
    end

    @testset "Growth rates" begin
        @test ConstantGrowth(0.5)(3.0) == 0.5
        @test LinearGrowth(0.2)(3.0) ≈ 0.6

        nodes = @SVector [1.0, 2.0, 3.0]
        weights = @SVector [0.5, 0.3, 0.2]
        gr = ParticleGrowth(ConstantGrowth(0.5))
        S = compute_source_terms(gr, nodes, weights, Val(4))
        @test S[1] ≈ 0.0 atol=1e-10                 # number conserved
        @test S[2] ≈ 0.5 * sum(weights) atol=1e-10  # S_1 = G0 * m0
    end
end
