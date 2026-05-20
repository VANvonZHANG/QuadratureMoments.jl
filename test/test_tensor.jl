using QuadratureMoments
using Test
using StaticArrays

@testset "Tensor-product QMOM" begin
    @testset "2D Independent Variables" begin
        # Two independent variables: x ~ {1.0, 2.0} w={0.5, 0.5}, y ~ {3.0, 4.0} w={0.5, 0.5}
        mx = SVector{4,Float64}(1.0, 1.5, 2.5, 4.5)
        my = SVector{4,Float64}(1.0, 3.5, 12.5, 45.5)

        method = TensorQMOM((2, 2))
        res = invert_moments(method, (mx, my))
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (4, 2)
        @test length(weights) == 4
        @test sum(weights) ≈ 1.0

        # Verify node combinations are (1,3), (2,3), (1,4), (2,4)
        @test isapprox(vec(nodes[1, :]), [1.0, 3.0])
        @test isapprox(vec(nodes[2, :]), [2.0, 3.0])
        @test isapprox(vec(nodes[3, :]), [1.0, 4.0])
        @test isapprox(vec(nodes[4, :]), [2.0, 4.0])
        @test all(weights .≈ 0.25)
    end

    @testset "3D Tensor from Full Array" begin
        v1, v2, v3 = 1.0, 2.0, 3.0
        m_data = zeros(2, 2, 2)
        for i in 0:1, j in 0:1, k in 0:1
            m_data[i + 1, j + 1, k + 1] = v1^i * v2^j * v3^k
        end
        m_static = SArray{Tuple{2,2,2},Float64,3,8}(m_data)

        method = TensorQMOM((1, 1, 1))
        res = invert_moments(method, m_static)
        nodes = res.nodes
        weights = res.weights

        @test size(nodes) == (1, 3)
        @test isapprox(vec(nodes[1, :]), [1.0, 2.0, 3.0])
        @test weights[1] ≈ 1.0
    end
end
