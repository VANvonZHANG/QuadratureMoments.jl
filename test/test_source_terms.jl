# test/test_source_terms.jl

using Test
using QuadratureMoments
using StaticArrays

@testset "SourceTerms and Physical Closures" begin
    # Initial state
    nodes = @SVector [1.0, 2.0, 3.0]
    weights = @SVector [0.5, 0.3, 0.2]
    m0 = sum(weights)

    @testset "Constant Aggregation Kernel" begin
        beta_0 = 1.2
        const_agg = Aggregation((xi, xj) -> beta_0)

        S_k = compute_source_terms(const_agg, nodes, weights, Val(4))

        # S_0 = -0.5 * beta_0 * m0^2
        @test S_k[1] ≈ -0.5 * beta_0 * (m0^2) atol=1e-10
        # S_1 = 0.0 (Mass conservation)
        @test S_k[2] ≈ 0.0 atol=1e-10
    end

    @testset "Constant Rate Growth" begin
        G_0 = 0.5
        const_growth = ParticleGrowth(xi -> G_0)

        S_k = compute_source_terms(const_growth, nodes, weights, Val(4))

        # S_0 = 0.0 (Number conservation)
        @test S_k[1] ≈ 0.0 atol=1e-10
        # S_1 = G_0 * m0
        @test S_k[2] ≈ G_0 * m0 atol=1e-10
    end

    @testset "Symmetric Binary Breakage" begin
        b0 = 1.5
        # frequency: constant b0
        # fragment moment: 2 * (xi/2)^k = 2^(1-k) * xi^k
        sym_break = Breakage(xi -> b0, (k, xi) -> (2.0^(1 - k)) * (xi^k))

        S_k = compute_source_terms(sym_break, nodes, weights, Val(4))

        # S_0 = b0 * m0
        @test S_k[1] ≈ b0 * m0 atol=1e-10
        # S_1 = 0.0 (Mass conservation)
        @test S_k[2] ≈ 0.0 atol=1e-10
    end

    @testset "Uniform Binary Breakage" begin
        b1 = 2.0
        # frequency: proportional to xi, i.e., b1 * xi
        # fragment moment: integral_0^xi v^k * (2/xi) dv = (2/(k+1)) * xi^k
        uni_break = Breakage(xi -> b1 * xi, (k, xi) -> (2.0 / (k + 1.0)) * (xi^k))

        S_k = compute_source_terms(uni_break, nodes, weights, Val(4))

        # S_1 = 0.0 (Mass conservation)
        @test S_k[2] ≈ 0.0 atol=1e-10
    end

    @testset "Nucleation" begin
        J_nuc = 1.5
        xi_nuc = 0.1
        nuc = Nucleation(J_nuc, xi_nuc)

        S_k = compute_source_terms(nuc, nodes, weights, Val(4))

        # S_k = J_nuc * xi_nuc^k
        @test S_k[1] ≈ J_nuc atol=1e-10
        @test S_k[2] ≈ J_nuc * xi_nuc atol=1e-10
        @test S_k[3] ≈ J_nuc * (xi_nuc^2) atol=1e-10
    end

    @testset "Deposition" begin
        K_dep = 0.5
        # Constant deposition rate
        dep = Deposition(xi -> K_dep)

        S_k = compute_source_terms(dep, nodes, weights, Val(4))

        # S_0 = - sum(w_i * K_dep) = - K_dep * m0
        @test S_k[1] ≈ -K_dep * m0 atol=1e-10

        # S_1 = - sum(w_i * K_dep * xi_i) = - K_dep * m1
        m1 = sum(weights .* nodes)
        @test S_k[2] ≈ -K_dep * m1 atol=1e-10
    end

    @testset "Particle Shrinkage (Evaporation)" begin
        G_0 = -0.5
        flux_0 = -0.1
        # Shrinkage requires an explicit model for the boundary flux at size zero
        shrink = ParticleShrinkage(xi -> G_0, (n, w) -> flux_0)

        S_k = compute_source_terms(shrink, nodes, weights, Val(4))

        # S_0 = user_defined_flux (flux_0)
        @test S_k[1] ≈ flux_0 atol=1e-10

        # S_1 = sum(w_i * 1 * xi_i^0 * G_0) = G_0 * m0
        @test S_k[2] ≈ G_0 * m0 atol=1e-10
    end

    @testset "Composite SourceTerms (Superposition)" begin
        beta_0 = 1.2
        G_0 = 0.5

        agg = Aggregation((xi, xj) -> beta_0)
        growth = ParticleGrowth(xi -> G_0)

        total_physics = agg + growth

        S_k_total = compute_source_terms(total_physics, nodes, weights, Val(4))
        S_k_agg = compute_source_terms(agg, nodes, weights, Val(4))
        S_k_growth = compute_source_terms(growth, nodes, weights, Val(4))

        # Test linearity/superposition
        for k in 1:4
            @test S_k_total[k] ≈ S_k_agg[k] + S_k_growth[k] atol=1e-10
        end
    end

    @testset "Source terms robust to degenerate (zero-weight) nodes" begin
        # A rank-1 quadrature padded into a 2-node result: slot 2 is (node=0, weight=0).
        # A kernel singular at 0 (Brownian-like 1/xi) must NOT produce NaN.
        weights = SVector{2,Float64}(1.0, 0.0)
        nodes = SVector{2,Float64}(3.0, 0.0)           # second slot is the degenerate pad
        sing_kernel(xi, xj) = 1.0 / (xi + xj)          # singular at (0,0)
        agg = Aggregation(sing_kernel)
        S = compute_source_terms(agg, nodes, weights, Val(4))
        @test all(isfinite, S)
        # Only slot 1 contributes; S_k == 0.5*w1^2*kern*(2*3)^k - w1^2*kern*3^k.
        for k in 0:3
            expected = 0.5 * 1.0^2 * (1 / 6.0) * 6.0^k - 1.0^2 * (1 / 6.0) * 3.0^k
            @test isapprox(S[k + 1], expected; atol=1e-10)
        end
    end

    @testset "LengthBased-based Aggregation (volume conservation)" begin
        beta_0 = 1.2
        agg_len = Aggregation((xi, xj) -> beta_0, LengthBased())

        S_k = compute_source_terms(agg_len, nodes, weights, Val(4))
        @test size(S_k) == (4,)
        # S_3 (k=3, volume moment) == 0  — volume conserved under LengthBased aggregation
        @test S_k[4] ≈ 0.0 atol=1e-10
        # S_0 is coordinate-independent (number): -0.5*beta_0*m0^2
        @test S_k[1] ≈ -0.5 * beta_0 * (m0^2) atol=1e-10
        # S_1 is NOT generally zero under LengthBased (mass moment not conserved)
        @test abs(S_k[2]) > 1e-6
    end

    @testset "Backward compat: old-form == explicit MassBased()" begin
        # Aggregation(f) defaults to MassBased(); must equal explicit MassBased() and the
        # hand-computed mass-based source.
        beta_0 = 1.2
        S_old = compute_source_terms(Aggregation((xi, xj) -> beta_0), nodes, weights, Val(2))
        S_new = compute_source_terms(Aggregation((xi, xj) -> beta_0, MassBased()), nodes, weights, Val(2))
        @test S_old == S_new
        @test S_old[1] ≈ -0.5 * beta_0 * m0^2 atol=1e-12

        # Breakage(f, fn) defaults to MassBased(); plain-function daughter via fallback.
        b0 = 1.5
        Sb_old = compute_source_terms(Breakage(xi -> b0, (k, xi) -> 2^(1 - k) * xi^k), nodes, weights, Val(2))
        Sb_new = compute_source_terms(Breakage(xi -> b0, (k, xi) -> 2^(1 - k) * xi^k, MassBased()), nodes, weights, Val(2))
        @test Sb_old == Sb_new
    end

    @testset "coord-aware source composes with expand_quadrature" begin
        # Gaussian EQMOM expands nodes over (-inf, inf), which can include negative
        # values. Only MassBased (birth (xi_i+xi_j)^k, defined for all reals) is valid
        # on them; LengthBased birth (xi_i^3+xi_j^3)^(k/3) needs non-negative nodes
        # (diameters) and is exercised on positive nodes in test_physics_kernels.jl.
        m = @SVector [1.0, 5.2, 29.0, 170.8, 1051.0]
        res = invert_moments(EQMOM(2, GaussianKernel()), m)
        en, ew = expand_quadrature(res, GaussianKernel(), Val(8))
        S_mass = compute_source_terms(Aggregation(Constant(1.0), MassBased()), en, ew, Val(4))
        @test all(isfinite, S_mass)
        @test S_mass[2] ≈ 0.0 atol=1e-6    # mass conserved under MassBased aggregation
    end
end
