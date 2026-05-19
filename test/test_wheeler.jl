# QBMM.jl 核心测试套件
using QBMM
using Test
using LinearAlgebra
using StaticArrays

@testset "Wheeler Algorithm Verification" begin
    m = @SVector [1.0, 1.0, 1.1, 1.3]

    @testset "StaticArrays Version" begin
        res = invert_moments(Wheeler(2), m)
        nodes = vec(res.nodes) # 使用 vec 解决 dot 维度不匹配
        weights = res.weights

        @test sum(weights) ≈ 1.0
        @test dot(weights, nodes) ≈ 1.0
        @test dot(weights, nodes .^ 2) ≈ 1.1
        @test dot(weights, nodes .^ 3) ≈ 1.3
    end

    @testset "Base Array Version" begin
        m_base = [1.0, 1.0, 1.1, 1.3]
        res = invert_moments(Wheeler(2), m_base)
        nodes = vec(res.nodes)
        weights = res.weights

        @test sum(weights) ≈ 1.0
        @test dot(weights, nodes) ≈ 1.0
        @test dot(weights, nodes .^ 2) ≈ 1.1
        @test dot(weights, nodes .^ 3) ≈ 1.3
    end
end
