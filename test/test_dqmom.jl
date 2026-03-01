using Test
using LinearAlgebra
using StaticArrays
using QBMM

@testset "DQMOM Verification" begin
    @testset "StaticArrays Version" begin
        # 假设存在 N=2 的分布
        nodes = @SVector [1.0, 2.0]
        
        # 构建 DQMOM 矩阵
        A = dqmom_matrix(nodes)
        
        # 校验矩阵维度
        @test size(A) == (4, 4)
        
        # 校验矩阵第一行 (k=0): 应为 a_1 + a_2 = S_0 (因此全 1 和全 0)
        # 即 [1.0, 1.0, 0.0, 0.0]
        @test A[1, 1:2] ≈ [1.0, 1.0]
        @test A[1, 3:4] ≈ [0.0, 0.0]
        
        # 校验矩阵第二行 (k=1): 对应 a 的系数为 (1-1)*xi^1 = 0, 对应 b 的系数为 1*xi^0 = 1
        # 即 [0.0, 0.0, 1.0, 1.0]
        @test A[2, 1:2] ≈ [0.0, 0.0]
        @test A[2, 3:4] ≈ [1.0, 1.0]
        
        # 给定一些任意的伪物理源项 (S_0 到 S_3)
        S = @SVector [0.1, -0.2, 0.5, 1.0]
        
        a, b = dqmom_solve(nodes, S)
        
        # 验证解的正确性: A * [a; b] 应当精确恢复 S
        x = vcat(a, b)
        @test A * x ≈ S
    end
    
    @testset "Base Array Version" begin
        nodes = [1.0, 2.0, 3.0]
        N = length(nodes)
        A = dqmom_matrix(nodes)
        @test size(A) == (6, 6)
        
        # 纯随机源项测试
        S = [0.1, -0.5, 1.2, 0.0, 3.1, -1.1]
        a, b = dqmom_solve(nodes, S)
        
        x = vcat(a, b)
        @test A * x ≈ S
    end
end
