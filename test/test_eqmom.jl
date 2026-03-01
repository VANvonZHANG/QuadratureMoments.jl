using Test
using LinearAlgebra
using StaticArrays
using QBMM

@testset "EQMOM (Gaussian Kernel) Verification" begin
    # 生成均值为 1，方差为 0.1 的正态分布的矩
    # 期望 EQMOM 能够处理这些矩并返回节点、权重和优化的带宽 σ
    # 对于 N=2，我们需要 2N+1 = 5 个矩: m0, m1, m2, m3, m4
    # m0 = 1
    # m1 = μ = 1
    # m2 = σ² + μ² = 0.1 + 1.0 = 1.1
    # m3 = μ³ + 3μσ² = 1.0 + 3*1*0.1 = 1.3
    # m4 = μ⁴ + 6μ²σ² + 3σ⁴ = 1.0 + 0.6 + 0.03 = 1.63
    m = @SVector [1.0, 1.0, 1.1, 1.3, 1.63]
    
    eqmom_method = EQMOM(2) # N=2 的 EQMOM，默认高斯核
    
    nodes, weights, sigma = invert_moments(eqmom_method, m)
    
    # 检查返回格式
    @test length(nodes) == 2
    @test length(weights) == 2
    
    # 零阶矩应守恒 (总概率/权重之和为 1)
    @test sum(weights) ≈ 1.0
    
    # σ 应该被成功找到且有物理意义
    @test sigma >= 0.0
end

@testset "EQMOM (Gamma Kernel) Verification" begin
    # 构造两个带有相同尺度参数的 Gamma 分布的混合，以测试 N=2 的 GammaKernel EQMOM
    # 分布 1: 形状 α1=10, 尺度 θ=0.1, 权重 w1=0.5 -> 均值 ξ1=1.0
    # 分布 2: 形状 α2=20, 尺度 θ=0.1, 权重 w2=0.5 -> 均值 ξ2=2.0
    # 期望反演出的结果应该为: nodes = [1.0, 2.0], weights = [0.5, 0.5], σ = 0.1
    m = @SVector [1.0, 1.5, 2.65, 5.28, 11.484]
    
    eqmom_method = EQMOM(2, GammaKernel())
    
    nodes, weights, sigma = invert_moments(eqmom_method, m)
    
    @test length(nodes) == 2
    @test length(weights) == 2
    @test sum(weights) ≈ 1.0
    
    # 因为存在寻根和精度误差，使用 isapprox 并设定较宽松的 tolerance
    @test isapprox(sigma, 0.1, atol=1e-3)
    @test isapprox(nodes[1], 1.0, atol=1e-2) || isapprox(nodes[2], 1.0, atol=1e-2)
    @test isapprox(nodes[2], 2.0, atol=1e-2) || isapprox(nodes[1], 2.0, atol=1e-2)
end

@testset "EQMOM (Beta Kernel) Verification" begin
    # 构造两个带有相同带宽参数的 Beta 分布混合，以测试 N=2 的 BetaKernel EQMOM
    # 选取 σ = 0.1 (即 α_i + β_i = 10)
    # 分布 1: 均值 ξ1=0.2 (α1=2, β1=8), 权重 w1=0.5
    # 分布 2: 均值 ξ2=0.8 (α2=8, β2=2), 权重 w2=0.5
    # m_k 经过推导得出：
    # m0 = 1.0
    # m1 = 0.5
    # m2 = 0.39 / 1.1
    # m3 = 0.372 / 1.32
    # m4 = 0.402 / 1.716
    m = @SVector [1.0, 0.5, 0.39/1.1, 0.372/1.32, 0.402/1.716]
    
    eqmom_method = EQMOM(2, BetaKernel())
    nodes, weights, sigma = invert_moments(eqmom_method, m)
    
    @test length(nodes) == 2
    @test length(weights) == 2
    @test sum(weights) ≈ 1.0
    
    # 验证反演精度
    @test isapprox(sigma, 0.1, atol=1e-3)
    @test isapprox(nodes[1], 0.2, atol=1e-2) || isapprox(nodes[2], 0.2, atol=1e-2)
    @test isapprox(nodes[2], 0.8, atol=1e-2) || isapprox(nodes[1], 0.8, atol=1e-2)
end
