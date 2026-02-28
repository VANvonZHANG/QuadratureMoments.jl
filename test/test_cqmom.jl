using Test
using QBMM
using LinearAlgebra

@testset "CQMOM 2D Prototype" begin
    # 设计一个简单的 2D 矩测试
    # 假设节点在 (1, 1) 和 (2, 2)，权重各为 0.5
    # N1 = 2, N2 = 1 (或者 N1=1, N2=2)
    # 我们使用 N1=2, N2=1 这样输入矩阵是 4x2
    
    n1, n2 = 2, 1
    m = zeros(2*n1, 2*n2)
    
    # 解析矩计算: m_{i,j} = sum_alpha w_alpha * xi_1^i * xi_2^j
    # alpha=1: w=0.5, xi=(1,1)
    # alpha=2: w=0.5, xi=(2,2)
    for i in 0:2*n1-1
        for j in 0:2*n2-1
            m[i+1, j+1] = 0.5 * (1.0^i * 1.0^j) + 0.5 * (2.0^i * 2.0^j)
        end
    end
    
    method = CQMOM{2, (n1, n2)}()
    nodes, weights = invert_moments(method, m)
    
    @test sum(weights) ≈ 1.0
    # 验证节点
    # 注意特征值求解可能顺序不同，我们通过排序或求和来验证
    @test sort(nodes[1, :]) ≈ [1.0, 2.0]
    @test sort(nodes[2, :]) ≈ [1.0, 2.0]
    @test all(weights .≈ 0.5)
end
