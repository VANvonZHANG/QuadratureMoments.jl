using Test, QuadratureMoments, StaticArrays, LinearAlgebra

# Analytic Gaussian raw moment E[X^k] for X ~ N(mu, sigma^2) — independent reference.
_double_factorial(n) = n <= 0 ? 1 : prod(1:2:n)
_gauss_moment(mu, k, sig) =
    sum(binomial(k, 2l) * _double_factorial(2l - 1) * sig^(2l) * mu^(k - 2l) for l in 0:div(k, 2))

@testset "expand_quadrature: Gaussian" begin
    # Known 2-Gaussian-mixture NDF.
    xi_true = [2.0, 5.0]; w_true = [0.4, 0.6]; σ_true = 0.5
    m = SVector{5,Float64}(
        sum(w_true[i] * _gauss_moment(xi_true[i], k, σ_true) for i in 1:2) for k in 0:4
    )
    res = invert_moments(EQMOM(2, GaussianKernel()), m)
    @test isapprox(res.sigmas[1], σ_true; atol=1e-6)   # sanity (Task 1 of the EQMOM plan)

    M = 8
    nodes_exp, weights_exp = expand_quadrature(res, GaussianKernel(), Val(M))
    @test length(nodes_exp) == 2 * M
    # (a) Normalization: expanded weights sum to m0.
    @test isapprox(sum(weights_exp), m[1]; atol=1e-10)
    # (b) Moment reproduction vs ANALYTIC mixture moments (independent of the library),
    #     exact up to the Gauss order 2M-1 = 15. This is the core σ-awareness check:
    #     a Dirac (unexpanded) quadrature would FAIL this for k >= 2 when σ > 0.
    #     rtol=1e-10 still rejects the Dirac (which errs ~1.5% at k=2); the atol=1e-8
    #     floor covers the small-moment (k=0,1) cases. High-k moments reach O(1e10),
    #     so an rtol is required for the test to be satisfiable at machine precision.
    for k in 0:(2M - 1)
        expanded_mk = sum(weights_exp[j] * nodes_exp[j]^k for j in 1:(2M))
        analytic_mk = sum(w_true[i] * _gauss_moment(xi_true[i], k, σ_true) for i in 1:2)
        @test isapprox(expanded_mk, analytic_mk; atol=1e-8, rtol=1e-10)
    end
    # (c) End-to-end pipeline: constant-kernel aggregation source S_0 = -0.5*beta0*m0^2,
    #     S_1 = 0 (holds for any realizable NDF). Confirms expand -> compute_source_terms plumb together.
    beta0 = 1.3
    S = compute_source_terms(Aggregation((x, y) -> beta0), nodes_exp, weights_exp, Val(2))
    @test isapprox(S[1], -0.5 * beta0 * m[1]^2; atol=1e-9)
    @test isapprox(S[2], 0.0; atol=1e-9)
end
