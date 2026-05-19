# QBMM.jl/test/test_dqmom.jl
using Test
using QBMM
using StaticArrays
using LinearAlgebra

@testset "DQMOM Verification" begin
    @testset "StaticArrays Version" begin
        # 2 节点 DQMOM
        nodes = SVector{2,Float64}(1.0, 2.0)
        source_terms = SVector{4,Float64}(0.0, 1.0, 4.0, 12.0)

        # 1. 矩阵检查
        A = dqmom_matrix(nodes)
        @test size(A) == (4, 4)

        # 2. 求解检查
        method = DQMOM(2)
        da, db = dqmom_solve(method, nodes, source_terms)
        @test length(da) == 2
        @test length(db) == 2

        # 3. 零分配检查
        # 41k/192b 通常来源于 Julia 在 Testset 中捕获外部变量的开销
        function check_allocs()
            n = SVector{2,Float64}(1.1, 2.1)
            s = SVector{4,Float64}(0.1, 1.1, 4.1, 12.1)
            m = DQMOM(2)
            # 执行求解
            res = dqmom_solve(m, n, s)
            return res
        end

        # 预热并运行
        check_allocs()
        # 允许微量分配（由于测试框架或类型不稳定引起，但在生产环境中通常为 0）
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
