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

@testset "expand_quadrature: Gamma" begin
    # testset-local to avoid colliding with the top-level _gamma_moment in test_eqmom.jl
    # (runtests.jl includes both files; a top-level redefinition here would be
    # include-order-dependent and fragile).
    _gamma_moment(xi, k, σ) = k == 0 ? one(xi) : prod(xi + r * σ for r in 0:(k - 1))

    xi_true = [2.0, 5.0]; w_true = [0.4, 0.6]; σ_true = 0.5
    m = SVector{5,Float64}(
        sum(w_true[i] * _gamma_moment(xi_true[i], k, σ_true) for i in 1:2) for k in 0:4
    )
    res = invert_moments(EQMOM(2, GammaKernel()), m)
    @test isapprox(res.sigmas[1], σ_true; atol=1e-6)

    M = 8
    nodes_exp, weights_exp = expand_quadrature(res, GammaKernel(), Val(M))
    @test length(nodes_exp) == 2 * M
    @test isapprox(sum(weights_exp), m[1]; atol=1e-10)
    # Expanded quadrature must be non-negative (gamma support is [0, inf)).
    @test all(nodes_exp .> 0.0)
    # Moment reproduction vs analytic gamma-mixture moments, k = 0..2M-1.
    # atol=1e-7 per the brief; rtol=1e-9 is required because gamma raw moments
    # grow very fast with k (here the k=15 mixture moment is O(1e10)), so a pure
    # atol is unsatisfiable at machine precision for high-k terms. rtol=1e-9
    # still distinguishes the gamma kernel from a Dirac quadrature: the Dirac
    # approximation errs by ~σ^2*(k choose 2)*xi^(k-2) at order k>=2, which for
    # this test's parameters is far larger than 1e-9 relative error.
    for k in 0:(2M - 1)
        expanded_mk = sum(weights_exp[j] * nodes_exp[j]^k for j in 1:(2M))
        analytic_mk = sum(w_true[i] * _gamma_moment(xi_true[i], k, σ_true) for i in 1:2)
        @test isapprox(expanded_mk, analytic_mk; atol=1e-7, rtol=1e-9)
    end
end

@testset "expand_quadrature: Beta" begin
    # testset-local helpers (per clarification 3): the gamma testset above scopes its
    # `_gamma_moment` INSIDE its @testset, so it is not visible here. Define both
    # locally to avoid colliding with the top-level helpers in test_eqmom.jl.
    _gamma_moment(xi, k, σ) = k == 0 ? one(xi) : prod(xi + r * σ for r in 0:(k - 1))
    _beta_moment(xi, k, σ) =
        k == 0 ? one(xi) : _gamma_moment(xi, k, σ) / prod(1 + r * σ for r in 0:(k - 1))

    xi_true = [0.2, 0.6]; w_true = [0.5, 0.5]; σ_true = 0.1
    m = SVector{5,Float64}(
        sum(w_true[i] * _beta_moment(xi_true[i], k, σ_true) for i in 1:2) for k in 0:4
    )
    res = invert_moments(EQMOM(2, BetaKernel()), m)
    @test isapprox(res.sigmas[1], σ_true; atol=1e-6)

    M = 8
    nodes_exp, weights_exp = expand_quadrature(res, BetaKernel(), Val(M))
    @test length(nodes_exp) == 2 * M
    @test isapprox(sum(weights_exp), m[1]; atol=1e-10)
    # Beta support is [0,1].
    @test all(0.0 .<= nodes_exp .<= 1.0)
    # Moment reproduction vs analytic beta-mixture moments, k = 0..2M-1.
    # atol=1e-8 per the brief; rtol=1e-9 added for safety/consistency with the
    # Gaussian and Gamma testsets. Beta moments on [0,1] are bounded (<=1), so
    # the atol alone is in fact satisfiable here; the rtol is harmless. This
    # still distinguishes the beta kernel from a Dirac quadrature: the Dirac
    # approximation errs by ~sigma^2*(k choose 2)*xi^(k-2) at order k>=2, which
    # for these parameters (xi~0.4, sigma=0.1) is O(1e-3) at k=2 — far above 1e-9.
    for k in 0:(2M - 1)
        expanded_mk = sum(weights_exp[j] * nodes_exp[j]^k for j in 1:(2M))
        analytic_mk = sum(w_true[i] * _beta_moment(xi_true[i], k, σ_true) for i in 1:2)
        @test isapprox(expanded_mk, analytic_mk; atol=1e-8, rtol=1e-9)
    end
end

@testset "expand_quadrature: degenerate primary node (single-component NDF)" begin
    # Local moment helpers (scoped inside the Gamma/Beta testsets above, not visible here).
    _gamma_moment_local(xi, k, σ) = k == 0 ? one(xi) : prod(xi + r * σ for r in 0:(k - 1))
    _beta_moment_local(xi, k, σ) =
        k == 0 ? one(xi) : _gamma_moment_local(xi, k, σ) / prod(1 + r * σ for r in 0:(k - 1))

    # --- Gamma: single-component NDF -> EQMOM(2) returns 1 real primary + 1
    # zero-padded degenerate slot. expand_quadrature must not crash on the
    # degenerate slot (alpha = xi_i/sigma = 0 -> gausslaguerre(M,-1) DomainError).
    xi_true = [4.0]; w_true = [1.0]; σ_true = 1.0
    m = SVector{5,Float64}(
        sum(w_true[i] * _gamma_moment_local(xi_true[i], k, σ_true) for i in 1:1) for k in 0:4
    )
    res = invert_moments(EQMOM(2, GammaKernel()), m)
    # At least one primary weight is ~0 (degenerate slot).
    @test count(iszero, res.weights) >= 1 || minimum(abs.(res.weights)) < 1e-8
    M = 8
    # Must not throw:
    nodes_exp, weights_exp = expand_quadrature(res, GammaKernel(), Val(M))
    @test all(isfinite, nodes_exp)
    @test all(isfinite, weights_exp)
    @test isapprox(sum(weights_exp), m[1]; atol=1e-10)   # mass preserved

    # --- Beta: same degeneracy class. xi_true in (0,1) so the beta node is valid.
    xi_true_beta = [0.4]; w_true_beta = [1.0]; σ_true_beta = 1.0
    m_beta = SVector{5,Float64}(
        sum(w_true_beta[i] * _beta_moment_local(xi_true_beta[i], k, σ_true_beta) for i in 1:1)
        for k in 0:4
    )
    res_beta = invert_moments(EQMOM(2, BetaKernel()), m_beta)
    @test count(iszero, res_beta.weights) >= 1 || minimum(abs.(res_beta.weights)) < 1e-8
    nodes_b, weights_b = expand_quadrature(res_beta, BetaKernel(), Val(M))
    @test all(isfinite, nodes_b)
    @test all(isfinite, weights_b)
end
