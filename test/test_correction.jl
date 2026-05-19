using QBMM
using Test
using StaticArrays

@testset "McGraw Moment Correction" begin
    @testset "Exercise 3.4: Corrupted Set" begin
        # 输入损坏的矩序列
        m = SVector{4,Float64}(1.0, 5.0, 26.0, 101.0)

        # 验证初始状态是不可实现的
        @test !is_realizable(m)

        # 执行修正
        m_corr = mcgraw_correction(m)

        # 验证结果可实现
        @test is_realizable(m_corr)

        # 验证 N=2 的 Wheeler 反演现在可以成功执行
        # (之前会报错或产生 NaN)
        res = invert_moments(Wheeler(2), m_corr)
        @test length(res.weights) == 2
        @test isapprox(sum(res.weights), 1.0, atol=1e-8)
    end

    @testset "Already Realizable Set" begin
        m = SVector{4,Float64}(1.0, 5.0, 26.0, 140.0)
        m_corr = mcgraw_correction(m)
        @test m_corr == m # 不应有任何改变
    end
end
