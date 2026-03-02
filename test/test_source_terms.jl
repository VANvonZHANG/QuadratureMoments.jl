# test/test_source_terms.jl

using Test
using QBMM
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
        sym_break = Breakage(
            xi -> b0,
            (k, xi) -> (2.0^(1 - k)) * (xi^k)
        )
        
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
        uni_break = Breakage(
            xi -> b1 * xi,
            (k, xi) -> (2.0 / (k + 1.0)) * (xi^k)
        )
        
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
end
