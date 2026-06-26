using Test
using QuadratureMoments

# StaticArrays (used by sibling test files included before/after this one in
# runtests.jl) also exports `Length`, which would clash with our coordinate
# type. Pin our meanings locally so the test is robust under both the
# standalone runner and `Pkg.test()`.
const Mass = QuadratureMoments.Mass
const Length = QuadratureMoments.Length

@testset "Coordinate conventions" begin
    @testset "aggregation_birth" begin
        @test aggregation_birth(Mass(), 2.0, 3.0, 1) == 5.0
        @test aggregation_birth(Mass(), 2.0, 3.0, 2) == 25.0
        @test aggregation_birth(Length(), 2.0, 3.0, 3) == 2.0^3 + 3.0^3
        @test aggregation_birth(Length(), 2.0, 3.0, 0) == 1.0
    end

    @testset "daughter_moment Function fallback" begin
        f(k, xi) = 2^(1 - k) * xi^k
        @test daughter_moment(f, 1, 4.0, Mass()) == 4.0
        @test daughter_moment(f, 0, 4.0, Mass()) == 2.0
        @test_throws ErrorException daughter_moment(f, 1, 4.0, Length())
    end

    @testset "Length aggregation conserves volume (k=3)" begin
        for (a, b) in [(1.0, 2.0), (3.0, 5.0), (0.5, 0.7)]
            @test aggregation_birth(Length(), a, b, 3) ≈ a^3 + b^3
        end
    end
end
