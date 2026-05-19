using QBMM
using Test
using StaticArrays

@testset "McGraw Moment Correction" begin
    @testset "Exercise 3.4: Corrupted Set" begin
        # Input corrupted moment sequence
        m = SVector{4,Float64}(1.0, 5.0, 26.0, 101.0)

        # Verify the initial state is non-realizable
        @test !is_realizable(m)

        # Perform correction
        m_corr = mcgraw_correction(m)

        # Verify the result is realizable
        @test is_realizable(m_corr)

        # Verify that N=2 Wheeler inversion can now execute successfully
        # (Previously it would error or produce NaN)
        res = invert_moments(Wheeler(2), m_corr)
        @test length(res.weights) == 2
        @test isapprox(sum(res.weights), 1.0, atol=1e-8)
    end

    @testset "Already Realizable Set" begin
        m = SVector{4,Float64}(1.0, 5.0, 26.0, 140.0)
        m_corr = mcgraw_correction(m)
        @test m_corr == m # There should be no change
    end
end
