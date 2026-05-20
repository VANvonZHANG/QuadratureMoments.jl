# test/test_adaptive_wheeler.jl
using Test
using QuadratureMoments
using StaticArrays
using LinearAlgebra

@testset "Adaptive Wheeler Inversion" begin
    # This test verifies the Wheeler algorithm's adaptive capability when handling "degenerate moment sequences."
    # E.g., when the distribution is actually a single Delta function, Wheeler(2) should automatically reduce to order 1.

    @testset "Degenerate Case: Single Node (Delta Distribution)" begin
        # Actual distribution: w=1.0, xi=5.0
        # Moment sequence: m_k = 1.0 * 5.0^k
        m = @SVector [1.0, 5.0, 25.0, 125.0] # 4 moments needed for 2 nodes

        # Attempting 2-node Wheeler inversion
        method = Wheeler(2)
        res = invert_moments(method, m)

        # Adaptive logic should return 1 valid node
        # Since static array length is fixed, we check the weight distribution
        @test count(w -> w > 1e-10, res.weights) == 1
        @test sum(res.weights) ≈ 1.0

        # The valid node position should be 5.0
        valid_idx = findfirst(w -> w > 1e-10, res.weights)
        @test res.nodes[valid_idx, 1] ≈ 5.0
    end

    @testset "Near-Degenerate Case: Numerical Noise" begin
        # Construct a nearly degenerate sequence with minimal perturbation
        # Such a sequence causes sigma table entries or b_k in the Wheeler algorithm to be extremely small
        m = SVector{6,Float64}(1.0, 2.0, 4.0, 8.0, 16.0, 32.000000000001)

        method = Wheeler(3)
        res = invert_moments(method, m)

        # Should be handled with automatic order reduction
        @test count(w -> w > 1e-8, res.weights) < 3
        @test sum(res.weights) ≈ 1.0
    end
end
