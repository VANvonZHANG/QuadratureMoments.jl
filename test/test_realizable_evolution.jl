using Test, QuadratureMoments, StaticArrays

@testset "Realizable evolution: constant-kernel aggregation" begin
    # beta(xi,xj) = beta0 (constant aggregation kernel).
    # dm0/dt = -beta0/2 * m0^2  =>  m0(t) = m0(0)/(1 + beta0/2 * m0(0) * t)
    # m1 (total "mass") is EXACTLY conserved.
    beta0 = 0.5
    m0_init = 1.0
    xi_init = SVector{2,Float64}(1.0, 3.0)
    w_init  = SVector{2,Float64}(0.5, 0.5)
    m1_init = sum(w_init[i] * xi_init[i] for i in 1:2)   # = 2.0
    # Build a 4-moment initial condition from the 2-node quadrature
    m_init = SVector{4,Float64}(ntuple(k -> sum(w_init[i]*xi_init[i]^(k-1) for i in 1:2), Val(4)))

    agg = Aggregation((xi,xj) -> beta0)
    t_end = 1.0
    # dt0=0.02 (not 0.05): first-order explicit Euler on the stiff early
    # transient overshoots m0 decay by ~0.23% at dt0=0.05 (above rtol=1e-3);
    # halving to dt0=0.02 gives rel_err ~9e-4, comfortably under 1e-3.
    m_final = evolve_moments(m_init, agg, (0.0, t_end); dt0=0.02, N=2)

    # m1 conserved
    @test isapprox(m_final[2], m1_init; atol=1e-8)
    # m0 matches analytic
    m0_analytic = m0_init / (1 + beta0/2 * m0_init * t_end)
    @test isapprox(m_final[1], m0_analytic; rtol=1e-3)
    # moments stay realizable throughout (implicit: evolve_moments guarantees this)
    @test is_realizable(m_final; domain=:pos)
end

@testset "Realizable evolution: constant growth keeps m0 constant" begin
    G0 = 0.7
    grow = ParticleGrowth(xi -> G0)
    xi0 = SVector{2,Float64}(1.0, 2.0)
    w0  = SVector{2,Float64}(0.5, 0.5)
    m_init = SVector{4,Float64}(ntuple(k -> sum(w0[i]*xi0[i]^(k-1) for i in 1:2), Val(4)))
    m_final = evolve_moments(m_init, grow, (0.0, 1.0); dt0=0.05, N=2)
    @test isapprox(m_final[1], m_init[1]; atol=1e-10)   # number conserved under growth
    @test is_realizable(m_final; domain=:pos)
end
