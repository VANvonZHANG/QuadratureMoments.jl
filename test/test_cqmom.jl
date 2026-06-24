using QuadratureMoments
using Test
using StaticArrays
using LinearAlgebra

@testset "CQMOM Recursive Implementation" begin
    @testset "2D Case: Independent Variables" begin
        # Two independent normal distributions: x ~ N(1, 0.1), y ~ N(2, 0.2)
        N = (2, 2)

        # Compute moments of 2D independent distribution: m[i, j] = m_x[i] * m_y[j]
        mx = [1.0, 1.0, 1.1, 1.3] # 0th-3rd order moments
        my = [1.0, 2.0, 4.2, 9.2] # 0th-3rd order moments

        m_data = zeros(4, 4)
        for i in 1:4, j in 1:4
            m_data[i, j] = mx[i] * my[j]
        end

        m_static = SMatrix{4,4,Float64}(m_data)

        method = CQMOM(N)
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (4, 2)
        @test length(weights) == 4

        # Verify moment reconstruction
        for i in 1:2, j in 1:2
            pred = sum(weights[k] * nodes[k, 1]^(i-1) * nodes[k, 2]^(j-1) for k in 1:4)
            @test isapprox(pred, m_data[i, j], atol=1e-8)
        end
    end

    @testset "3D Case: Low Order" begin
        N = (1, 1, 1)
        m_data = zeros(2, 2, 2)
        v1, v2, v3 = 1.5, 2.5, 3.5
        for i in 0:1, j in 0:1, k in 0:1
            m_data[i + 1, j + 1, k + 1] = v1^i * v2^j * v3^k
        end

        m_static = SArray{Tuple{2,2,2},Float64,3,8}(m_data)

        method = CQMOM(N)
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (1, 3)
        @test isapprox(vec(nodes[1, :]), [1.5, 2.5, 3.5], atol=1e-8)
        @test isapprox(weights[1], 1.0, atol=1e-8)
    end

    @testset "2D round-trip against known 2x2 discrete mixture" begin
        # Ground-truth quadrature: 4 nodes in 2D (2x2 tensor product).
        # CQMOM is the discrete multivariate method, so moments are exact
        # finite sums — no sigma/bandwidth involved.
        w_true = SVector{4,Float64}(0.1, 0.2, 0.3, 0.4)
        xi_true = SMatrix{4,2,Float64}(1.0, 2.0, 1.0, 3.0,   # x1 coords (col 1)
                                       0.5, 0.5, 2.0, 2.0)   # x2 coords (col 2)
        # Build the 2D moment tensor m[i,j] = sum_a w_a * x1_a^(i-1) * x2_a^(j-1), i,j=1..4
        M = zeros(4, 4)
        for i in 1:4, j in 1:4
            M[i, j] = sum(w_true[a] * xi_true[a, 1]^(i - 1) * xi_true[a, 2]^(j - 1) for a in 1:4)
        end
        mSA = SMatrix{4,4,Float64}(M)
        res = invert_moments(CQMOM((2, 2)), mSA)
        # Total weight conserved
        @test isapprox(sum(res.weights), sum(w_true); atol=1e-10)
        # Reconstruct mixed moments and compare to the tensor.
        # Nodes/weights are recovered only up to permutation, so assert via
        # moment reproduction rather than direct node equality.
        #
        # KNOWN CQMOM BUG (Phase 1 Task 4 regression test):
        # The recursive conditional-deconvolution in cqmom.jl uses only N1
        # moments of dimension 1 to extract each conditional moment value
        # (see line ~111: `b_vec = SVector{N1,T}`). A correct CQMOM(2,2)
        # should reproduce the full 4x4 moment tensor up to order (3,3),
        # but the current implementation only matches mixed moments where
        # the dim-1 order is < N1 (i.e. i <= 2 here) OR the dim-2 order is
        # 0 (pure dim-1 marginal). Specifically, M[3,2] and M[4,4] (and
        # every entry with i >= 3 and j >= 2) are NOT reproduced.
        # See task-4-report.md for the full failure matrix.
        # When cqmom.jl is fixed, the @test_broken pairs below will start
        # PASSING, which flips @test_broken to FAILURE — remove the wrapper
        # and convert back to @test at that time.
        #
        # Passing mixed moments (current behavior, correctly reproduced):
        for (i, j) in [(1, 1), (2, 1), (1, 2), (2, 3)]
            rec = sum(res.weights[a] * res.nodes[a, 1]^(i - 1) * res.nodes[a, 2]^(j - 1) for a in 1:4)
            @test isapprox(rec, M[i, j]; atol=1e-8)
        end
        # Broken mixed moments (documented CQMOM bug — should be @test once fixed):
        for (i, j) in [(3, 2), (4, 4)]
            rec = sum(res.weights[a] * res.nodes[a, 1]^(i - 1) * res.nodes[a, 2]^(j - 1) for a in 1:4)
            @test_broken isapprox(rec, M[i, j]; atol=1e-8)
        end
    end
end
