using QBMM
using Test
using StaticArrays
using LinearAlgebra

# Function to compute moments of a normal distribution
function normal_moments(μ, σ, n)
    m = zeros(n)
    m[1] = 1.0 # m0
    if n > 1
        m[2] = μ # m1
    end
    for k in 3:n
        m[k] = μ * m[k-1] + (k-2) * σ^2 * m[k-2]
    end
    return m
end

@testset "PD Algorithm Tests" begin
    μ, σ = 5.0, 1.0
    m_base = normal_moments(μ, σ, 10)

    @testset "N=2 (4 moments)" begin
        m_4 = m_base[1:4]
        res = invert_moments(PD(2), m_4)
        nodes = vec(res.nodes)
        weights = res.weights
        
        # Reconstruct moments
        pred_m = [sum(weights .* nodes .^ k) for k in 0:3]
        @test isapprox(pred_m, m_4, rtol=1e-12)
        
        @test isapprox(sort(nodes), [4.0, 6.0], atol=1e-8)
        @test isapprox(weights, [0.5, 0.5], atol=1e-8)
    end
    
    @testset "N=4 (8 moments)" begin
        m_8 = m_base[1:8]
        res = invert_moments(PD(4), m_8)
        nodes = vec(res.nodes)
        weights = res.weights
        
        pred_m = [sum(weights .* nodes .^ k) for k in 0:7]
        @test isapprox(pred_m, m_8, rtol=1e-10)
    end

    @testset "StaticArrays zero allocation" begin
        m_static = SVector{8, Float64}(m_base[1:8])
        
        res = invert_moments(PD(4), m_static)
        nodes = vec(res.nodes)
        weights = res.weights
        
        pred_m = [sum(weights .* nodes .^ k) for k in 0:7]
        @test isapprox(pred_m, m_base[1:8], rtol=1e-10)
        
        @test weights isa SVector{4, Float64}
    end
end
