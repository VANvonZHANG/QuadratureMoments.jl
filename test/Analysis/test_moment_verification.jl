using QuadratureMoments
using QuadratureMoments.Analysis
using StaticArrays
using Test

@testset "Moment Verification" begin
    @testset "compare_moments" begin
        t_check = [0.0, 1.0, 2.0]
        num_fn(t) = [1.0 + t, 2.0 + 2t, 3.0 + 3t]
        ref_fn(t) = [1.0 + t, 2.0 + 2t, 3.0 + 3t]

        mc = compare_moments(t_check, num_fn, ref_fn; n_moments = 3)
        @test mc.times == t_check
        @test size(mc.numerical) == (3, 3)
        @test size(mc.reference) == (3, 3)

        # Exact match -> zero errors
        @test all(abs_errors(mc) .== 0)
        @test all(rel_errors(mc) .== 0)
        @test all(max_abs_errors(mc) .== 0)
        @test verify(mc; atol = [1e-10, 1e-10, 1e-10])
    end

    @testset "compare_moments with mismatch" begin
        bad_fn(t) = [1.0]  # returns 1 moment, expected 2
        @test_throws DimensionMismatch compare_moments(
            [0.0], bad_fn, t -> [1.0, 2.0]; n_moments = 2
        )
    end

    @testset "verify_reconstruction" begin
        # PD(4) on Gaussian moments
        m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]
        res = invert_moments(PD(4), m)
        results = verify_reconstruction(res, Vector(m); tol = 1e-8)

        @test length(results) == 8
        for (k, r) in enumerate(results)
            @test r.order == k - 1
            @test r.pass == true
            @test r.rel_err < 1e-8
        end
    end

    @testset "verify with tolerances" begin
        t_check = [0.0, 1.0]
        num_fn(t) = [1.0 + 0.1t, 2.0 + 0.2t]
        ref_fn(t) = [1.0, 2.0]
        mc = compare_moments(t_check, num_fn, ref_fn; n_moments = 2)

        @test verify(mc; atol = [0.2, 0.2])
        @test !verify(mc; atol = [0.001, 0.001], rtol = [0.001, 0.001])
    end
end
