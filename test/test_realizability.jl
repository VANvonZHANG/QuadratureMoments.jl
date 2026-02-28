using Test
using QBMM
using StaticArrays

@testset "Realizability Check" begin
    @testset "Valid Moments" begin
        # 均值为 1，方差为 0.1 的正态分布，必然是合法的
        m_valid = @SVector [1.0, 1.0, 1.1, 1.3]
        @test is_realizable(m_valid) == true
        
        m_base = [1.0, 1.0, 1.1, 1.3]
        @test is_realizable(m_base) == true
    end
    
    @testset "Invalid Moments" begin
        # 构造非法的矩序列：方差为负 (m2 < m1^2)，m0=1, m1=1, m2=0.9
        m_invalid = @SVector [1.0, 1.0, 0.9, 1.3]
        @test is_realizable(m_invalid) == false
        
        m_invalid_base = [1.0, 1.0, 0.9, 1.3]
        @test is_realizable(m_invalid_base) == false
    end
end
