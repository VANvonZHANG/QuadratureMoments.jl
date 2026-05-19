using Test
using QBMM
using StaticArrays
using LinearAlgebra

@testset "Moment Realizability Check" begin
    # 1. Physically valid moments (normal distribution with mean 1 and variance 0.1)
    # m0=1, m1=1, m2=1.1, m3=1.3, m4=1.63, m5=2.2 (approx)
    m_valid = @SVector [1.0, 1.0, 1.1, 1.3]
    @test is_realizable(m_valid) == true

    # 2. Physically invalid moments (violating monotonicity or positive definiteness)
    # E.g., m2 < m1^2 (negative variance)
    m_invalid = @SVector [1.0, 1.0, 0.5, 0.2]
    @test is_realizable(m_invalid) == false

    # 3. Edge case: single-point distribution (node at 1.0)
    # m0=1, m1=1, m2=1, m3=1...
    # This is numerically borderline positive-definite
    m_boundary = @SVector [1.0, 1.0, 1.0, 1.0]
    @test is_realizable(m_boundary) == true
end
