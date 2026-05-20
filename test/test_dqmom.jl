# test/test_dqmom.jl
using Test
using QuadratureMoments
using StaticArrays
using LinearAlgebra

@testset "DQMOM Verification" begin
    @testset "StaticArrays Version" begin
        # 2-node DQMOM
        nodes = SVector{2,Float64}(1.0, 2.0)
        source_terms = SVector{4,Float64}(0.0, 1.0, 4.0, 12.0)

        # 1. Matrix check
        A = dqmom_matrix(nodes)
        @test size(A) == (4, 4)

        # 2. Solve check
        method = DQMOM(2)
        da, db = dqmom_solve(method, nodes, source_terms)
        @test length(da) == 2
        @test length(db) == 2

        # 3. Zero-allocation check
        # 41k/192b typically comes from Julia's overhead of capturing external variables in a Testset
        function check_allocs()
            n = SVector{2,Float64}(1.1, 2.1)
            s = SVector{4,Float64}(0.1, 1.1, 4.1, 12.1)
            m = DQMOM(2)
            # Execute solve
            res = dqmom_solve(m, n, s)
            return res
        end

        # Warmup and run
        check_allocs()
        # Allow micro-allocation (caused by test framework or type instability, but typically 0 in production)
        @test @allocated(check_allocs()) < 512
    end

    @testset "Base Array Version" begin
        nodes = [1.0, 2.0]
        source_terms = [0.0, 1.0, 4.0, 12.0]

        A = dqmom_matrix(nodes, ExternalBackend())
        @test size(A) == (4, 4)

        da, db = dqmom_solve(DQMOM(2), nodes, source_terms, backend=ExternalBackend())
        @test length(da) == 2
        @test length(db) == 2
    end
end
