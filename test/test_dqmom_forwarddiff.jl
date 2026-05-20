# test/test_dqmom_forwarddiff.jl
using Test
using QuadratureMoments
using StaticArrays
using ForwardDiff
using LinearAlgebra

@testset "DQMOM ForwardDiff Integration" begin
    # This test verifies whether the DQMOM solver can work with ForwardDiff,
    # which is important when Jacobian computation of DQMOM evolution rates is needed (e.g., for stiff ODE solving).

    # Simulate a simple source term function that depends on nodes
    function compute_source_terms(nodes::SVector{N,T}) where {N,T}
        # S_k = k * m_{k-1} * (some_growth_rate)
        # Simplified model: S_k = k * sum(nodes)
        return SVector{2N,T}(ntuple(k -> T(k-1) * sum(nodes), Val(2N)))
    end

    function dqmom_rhs(v::SVector{L,T}) where {L,T}
        N = L ÷ 2
        # v = [w1, w2, w1*xi1, w2*xi2]
        weights = SVector{N,T}(ntuple(i -> v[i], Val(N)))
        weighted_nodes = SVector{N,T}(ntuple(i -> v[i + N], Val(N)))

        # Extract nodes xi_i = (w_i * xi_i) / w_i
        nodes = SVector{N,T}(ntuple(i -> weighted_nodes[i] / (weights[i] + eps(T)), Val(N)))

        # Compute source terms
        S = compute_source_terms(nodes)

        # Solve the DQMOM system to get [dw/dt; d(w*xi)/dt]
        # Note: DQMOM(N) needs to be passed here
        da, db = dqmom_solve(DQMOM(N), nodes, S)

        return vcat(da, db)
    end

    # Initialize state
    v0 = SVector{4,Float64}(0.5, 0.5, 0.5*1.0, 0.5*2.0)

    # 1. Basic execution check
    rhs0 = dqmom_rhs(v0)
    @test length(rhs0) == 4

    # 2. Jacobian check (ForwardDiff)
    # If the code contains type instability or non-generic types, this will error
    Jac = ForwardDiff.jacobian(dqmom_rhs, v0)
    @test size(Jac) == (4, 4)
    @test !any(isnan.(Jac))

    @testset "Dual-Backend dispatch inside AD" begin
        # Verify that ExternalBackend also works inside AD
        function dqmom_rhs_ext(v::Vector{T}) where {T}
            N = length(v) ÷ 2
            weights = v[1:N]
            wnodes = v[(N + 1):end]
            nodes = wnodes ./ (weights .+ eps(T))
            S = collect(compute_source_terms(SVector{N,T}(nodes)))
            da, db = dqmom_solve(DQMOM(N), nodes, S, backend=ExternalBackend())
            return [da; db]
        end

        v0_vec = [0.5, 0.5, 0.5, 1.0]
        Jac_ext = ForwardDiff.jacobian(dqmom_rhs_ext, v0_vec)
        @test size(Jac_ext) == (4, 4)
    end
end
