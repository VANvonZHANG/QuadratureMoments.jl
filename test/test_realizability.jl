using Test
using QBMM
using StaticArrays
using LinearAlgebra

@testset "Moment Realizability Check" begin
    # 1. 物理合理的矩 (均值为 1, 方差为 0.1 的正态分布)
    # m0=1, m1=1, m2=1.1, m3=1.3, m4=1.63, m5=2.2 (approx)
    m_valid = @SVector [1.0, 1.0, 1.1, 1.3]
    @test is_realizable(m_valid) == true

    # 2. 物理不合理的矩 (破坏单调性或正定性)
    # 比如 m2 < m1^2 (方差为负)
    m_invalid = @SVector [1.0, 1.0, 0.5, 0.2]
    @test is_realizable(m_invalid) == false

    # 3. 边界情况：单点分布 (节点在 1.0)
    # m0=1, m1=1, m2=1, m3=1... 
    # 这在数值上是临界正定的
    m_boundary = @SVector [1.0, 1.0, 1.0, 1.0]
    @test is_realizable(m_boundary) == true
end
