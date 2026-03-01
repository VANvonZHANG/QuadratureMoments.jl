using QBMM
using Test
using StaticArrays

@testset "EQMOM Kernel Extensions" begin
    @testset "Gaussian Kernel (Mixture)" begin
        # N=2, 需要 5 个矩
        # 混合分布: 0.4 * N(4, 1) + 0.6 * N(6, 1)
        # σ_true = 1.0
        # m0 = 1.0
        # m1 = 0.4*4 + 0.6*6 = 5.2
        # m2 = 0.4*(4^2+1) + 0.6*(6^2+1) = 0.4*17 + 0.6*37 = 29.0
        # m3 = 0.4*(4^3+3*4*1) + 0.6*(6^3+3*6*1) = 0.4*76 + 0.6*234 = 30.4 + 140.4 = 170.8
        # m4 = 0.4*(4^4+6*4^2*1+3*1) + 0.6*(6^4+6*6^2*1+3*1) = 0.4*(256+96+3) + 0.6*(1296+216+3) 
        #    = 0.4*355 + 0.6*1515 = 142.0 + 909.0 = 1051.0
        m = SVector{5, Float64}(1.0, 5.2, 29.0, 170.8, 1051.0)
        
        method = EQMOM(2, GaussianKernel())
        nodes, weights, σ = invert_moments(method, m)
        
        @test length(nodes) == 2
        @test isapprox(σ, 1.0, atol=1e-2)
        @test isapprox(sum(weights), 1.0, atol=1e-3)
    end

    @testset "Gamma Kernel (Skewed Distribution)" begin
        # 使用 Gamma 分布的解析矩进行测试
        # k=2.0 (shape), θ=2.0 (scale) -> mean = 4.0, var = 8.0
        # 矩公式: m_n = θ^n * Γ(k+n)/Γ(k) = θ^n * (k+n-1)...k
        # m0 = 1
        # m1 = 2 * 2 = 4
        # m2 = 2^2 * (2*3) = 24
        # m3 = 2^3 * (2*3*4) = 192
        # m4 = 2^4 * (2*3*4*5) = 1920
        m = SVector{5, Float64}(1.0, 4.0, 24.0, 192.0, 1920.0)
        
        method = EQMOM(2, GammaKernel())
        nodes, weights, σ = invert_moments(method, m)
        
        @test length(nodes) == 2
        @test σ > 0.0
        # 重构验证
        m2_pred = sum(weights[i] * (nodes[i] + 0.0)*(nodes[i] + σ) for i in 1:2)
        @test isapprox(m2_pred, 24.0, atol=1e-3)
    end

    @testset "Beta Kernel" begin
        # 简单测试 Beta 核的反演稳定性
        # mean=0.5, var=0.05
        m = SVector{5, Float64}(1.0, 0.5, 0.3, 0.2, 0.15)
        
        method = EQMOM(2, BetaKernel())
        nodes, weights, σ = invert_moments(method, m)
        
        @test length(nodes) == 2
        @test all(0.0 .<= nodes .<= 1.0)
    end
end
