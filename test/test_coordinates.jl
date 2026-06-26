using Test
using QuadratureMoments
using StaticArrays

@testset "Coordinate conventions" begin
    @testset "aggregation_birth" begin
        @test aggregation_birth(MassBased(), 2.0, 3.0, 1) == 5.0
        @test aggregation_birth(MassBased(), 2.0, 3.0, 2) == 25.0
        @test aggregation_birth(LengthBased(), 2.0, 3.0, 3) == 2.0^3 + 3.0^3
        @test aggregation_birth(LengthBased(), 2.0, 3.0, 0) == 1.0
    end

    @testset "daughter_moment Function fallback" begin
        f(k, xi) = 2^(1 - k) * xi^k
        @test daughter_moment(f, 1, 4.0, MassBased()) == 4.0
        @test daughter_moment(f, 0, 4.0, MassBased()) == 2.0
        @test_throws ErrorException daughter_moment(f, 1, 4.0, LengthBased())
    end

    @testset "LengthBased aggregation conserves volume (k=3)" begin
        for (a, b) in [(1.0, 2.0), (3.0, 5.0), (0.5, 0.7)]
            @test aggregation_birth(LengthBased(), a, b, 3) ≈ a^3 + b^3
        end
    end
end
