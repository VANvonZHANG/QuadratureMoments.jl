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
    # 注意：在测试中我们需要确保编译器不会因为循环而产生额外的装箱
    # 使用静态分发的接口进行基准测试
    function bench_stirling_static()
        s = 0.0
        for i in 1:100
            # 使用常量进行测试
            s += QBMM._stirling2_static(Val(5), Val(3))
        end
        return s
    end
    
    # 预热并测试
    bench_stirling_static()
    @test @allocated(bench_stirling_static()) == 0
end
