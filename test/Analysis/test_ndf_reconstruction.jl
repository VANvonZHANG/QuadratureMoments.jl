using QuadratureMoments
using QuadratureMoments.Analysis
using StaticArrays
using Test

@testset "NDF Reconstruction" begin
    @testset "Gaussian kernel evaluation" begin
        # Standard normal at mean
        @test isapprox(
            evaluate_kernel(GaussianKernel(), 0.0, 0.0, 1.0), 1.0 / sqrt(2π), atol=1e-10
        )
        # Integral should be ~1
        xs = range(-5.0, 5.0, length=1000)
        dx = step(xs)
        integral = sum(evaluate_kernel(GaussianKernel(), x, 0.0, 1.0) for x in xs) * dx
        @test isapprox(integral, 1.0, atol=1e-3)
    end

    @testset "reconstruct_ndf with EQMOM Gaussian" begin
        # Known normal distribution moments
        m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0]
        res = invert_moments(EQMOM(2, GaussianKernel()), m)
        @test res.sigmas !== nothing

        ξ_range = range(0.0, 10.0, length=200)
        ndf = reconstruct_ndf(res, ξ_range, GaussianKernel())

        # NDF should be non-negative
        @test all(ndf .>= 0)

        # Integral should approximate m₀ = 1.0
        dξ = step(ξ_range)
        integral = sum(ndf) * dξ
        @test isapprox(integral, 1.0, atol=0.05)

        # Mean should approximate m₁/m₀ = 5.0
        mean_val = sum(ξ_range .* ndf) * dξ / integral
        @test isapprox(mean_val, 5.0, atol=0.1)
    end

    @testset "reconstruct_ndf without sigmas throws" begin
        m = @SVector [1.0, 5.0, 26.0, 140.0]
        res = invert_moments(Wheeler(2), m)
        @test res.sigmas === nothing
        @test_throws ArgumentError reconstruct_ndf(res, 0.0:0.1:1.0, GaussianKernel())
    end
end
