# QBMM.jl/test/test_stirling.jl
using Test
using QBMM
using StaticArrays

@testset "Stirling Numbers Math Backend" begin
    # 1. NativeBackend (Lookup Table)
    @test stirling2(3, 2, NativeBackend()) == 3.0
    @test stirling2(4, 2, NativeBackend()) == 7.0
    @test stirling2(5, 3, NativeBackend()) == 25.0
    @test stirling2(10, 5, NativeBackend()) == 42525.0
    @test stirling2(12, 11, NativeBackend()) == 66.0
    
    # 边界情况
    @test stirling2(0, 0, NativeBackend()) == 1.0
    @test stirling2(5, 0, NativeBackend()) == 0.0
    @test stirling2(5, 6, NativeBackend()) == 0.0
    @test stirling2(5, -1, NativeBackend()) == 0.0

    # 超出范围 (n=13) 自动回退
    @test stirling2(13, 2, NativeBackend()) == 4095.0

    # 2. ExternalBackend (Combinatorics.jl)
    @test stirling2(3, 2, ExternalBackend()) == 3.0
    @test stirling2(10, 5, ExternalBackend()) == 42525.0
    
    # 3. 性能测试（验证 NativeBackend 零分配）
    # 使用函数封装以完全隔离测试环境
    function test_alloc()
        # 使用本地不可变对象
        val_n = Val(5)
        val_k = Val(3)
        res = QBMM._stirling2_static(val_n, val_k)
        return res
    end
    
    # 预热
    test_alloc()
    # 在许多 Julia 版本中，@allocated 在顶层可能会有微量分配，但在函数内更准确
    @test @allocated(test_alloc()) == 0
end
