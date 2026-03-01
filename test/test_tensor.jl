using QBMM
using Test
using StaticArrays

@testset "Tensor-product QMOM" begin
    @testset "2D Independent Variables" begin
        # 两个独立变量: x ~ {1.0, 2.0} w={0.5, 0.5}, y ~ {3.0, 4.0} w={0.5, 0.5}
        # 边缘矩
        mx = SVector{4, Float64}(1.0, 1.5, 2.5, 4.5) # 对应节点 [1, 2], 权重 [0.5, 0.5]
        my = SVector{4, Float64}(1.0, 3.5, 12.5, 45.5) # 对应节点 [3, 4], 权重 [0.5, 0.5]
        
        # 1. 测试直接输入边缘矩的接口
        method = TensorQMOM((2, 2))
        nodes, weights = invert_moments(method, (mx, my))
        
        @test size(nodes) == (2, 4)
        @test length(weights) == 4
        @test sum(weights) ≈ 1.0
        
        # 验证节点组合是否为 (1,3), (2,3), (1,4), (2,4) 的某种排列
        # 由于 CartesianIndices 顺序固定，我们可以直接检查
        @test isapprox(nodes[:, 1], [1.0, 3.0])
        @test isapprox(nodes[:, 2], [2.0, 3.0])
        @test isapprox(nodes[:, 3], [1.0, 4.0])
        @test isapprox(nodes[:, 4], [2.0, 4.0])
        @test all(weights .≈ 0.25)
    end

    @testset "3D Tensor from Full Array" begin
        # 3D 情况，每一维 N=1
        v1, v2, v3 = 1.0, 2.0, 3.0
        m_data = zeros(2, 2, 2)
        for i in 0:1, j in 0:1, k in 0:1
            m_data[i+1, j+1, k+1] = v1^i * v2^j * v3^k
        end
        m_static = SArray{Tuple{2, 2, 2}, Float64, 3, 8}(m_data)
        
        method = TensorQMOM((1, 1, 1))
        nodes, weights = invert_moments(method, m_static)
        
        @test size(nodes) == (3, 1)
        @test isapprox(nodes[:, 1], [1.0, 2.0, 3.0])
        @test weights[1] ≈ 1.0
    end
end
