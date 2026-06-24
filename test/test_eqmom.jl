using QuadratureMoments
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
        res = invert_moments(EQMOM(2, GammaKernel()), m)
        @test size(res.nodes) == (2, 1)
        @test res.sigmas[1] > 0.0
        for k in 0:4
            rec_mk = QuadratureMoments.reconstruct_moment(
                res.nodes, res.weights, k, res.sigmas[1], GammaKernel()
            )
            @test isapprox(rec_mk, m[k + 1]; atol=1e-6)
        end
    end

    @testset "Beta Kernel" begin
        # Moments of a single beta node xi=0.5, sigma=0.25 — exactly realizable
        # (analogous to the gamma testset above). The original hand-picked
        # 0.15 for m_4 was not a valid beta-kernel moment; corrected to 1/7.
        m = SVector{5,Float64}(1.0, 0.5, 0.3, 0.2, 0.14285714285714285)
        res = invert_moments(EQMOM(2, BetaKernel()), m)
        @test size(res.nodes) == (2, 1)
        @test all(0.0 .<= res.nodes .<= 1.0)
        for k in 0:4
            rec_mk = QuadratureMoments.reconstruct_moment(
                res.nodes, res.weights, k, res.sigmas[1], BetaKernel()
            )
            @test isapprox(rec_mk, m[k + 1]; atol=1e-6)
        end
    end
end

# --- Round-trip correctness tests (added to catch modified-moment transform bugs) ---
# Ground truth = analytic 2-component kernel mixture. mu_k(xi_i) is the DEFINITION
# of the k-th kernel moment; the inversion+reconstruction pipeline must reproduce it.

_gamma_moment(xi, k, σ) = k == 0 ? one(xi) : prod(xi + j * σ for j in 0:(k-1))

@testset "Gamma Kernel Round-Trip (2-node mixture)" begin
    w_true = [0.4, 0.6]
    xi_true = [2.0, 5.0]
    σ_true = 0.5
    m = SVector{5,Float64}(
        sum(w_true[i] * _gamma_moment(xi_true[i], k, σ_true) for i in 1:2) for k in 0:4
    )

    res = invert_moments(EQMOM(2, GammaKernel()), m)

    # sigma must be recovered
    @test isapprox(res.sigmas[1, 1], σ_true; atol=1e-6)
    # nodes & weights recovered up to permutation (sort by node)
    rec = sort(collect(zip(vec(res.nodes), collect(res.weights))); by=first)
    @test isapprox(first.(rec), sort(xi_true); atol=1e-6)
    @test isapprox(last.(rec), w_true; atol=1e-6)
    # ALL input moments must be reproduced (not just m_4)
    for k in 0:4
        rec_mk = QuadratureMoments.reconstruct_moment(
            res.nodes, res.weights, k, res.sigmas[1], GammaKernel()
        )
        @test isapprox(rec_mk, m[k + 1]; atol=1e-8)
    end
end

_beta_moment(xi, k, σ) =
    k == 0 ? one(xi) :
    _gamma_moment(xi, k, σ) / prod(1 + j * σ for j in 0:(k - 1))

@testset "Beta Kernel Round-Trip (2-node mixture)" begin
    w_true = [0.5, 0.5]
    xi_true = [0.2, 0.6]
    σ_true = 0.1
    m = SVector{5,Float64}(
        sum(w_true[i] * _beta_moment(xi_true[i], k, σ_true) for i in 1:2) for k in 0:4
    )

    res = invert_moments(EQMOM(2, BetaKernel()), m)

    @test isapprox(res.sigmas[1, 1], σ_true; atol=1e-6)
    rec = sort(collect(zip(vec(res.nodes), collect(res.weights))); by=first)
    @test isapprox(first.(rec), sort(xi_true); atol=1e-6)
    @test isapprox(last.(rec), w_true; atol=1e-6)
    for k in 0:4
        rec_mk = QuadratureMoments.reconstruct_moment(
            res.nodes, res.weights, k, res.sigmas[1], BetaKernel()
        )
        @test isapprox(rec_mk, m[k + 1]; atol=1e-8)
    end
end
