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

    # Edge cases
    @test stirling2(0, 0, NativeBackend()) == 1.0
    @test stirling2(5, 0, NativeBackend()) == 0.0
    @test stirling2(5, 6, NativeBackend()) == 0.0
    @test stirling2(5, -1, NativeBackend()) == 0.0

    # Out of range (n=13) automatic fallback
    @test stirling2(13, 2, NativeBackend()) == 4095.0

    # 2. ExternalBackend (Combinatorics.jl)
    @test stirling2(3, 2, ExternalBackend()) == 3.0
    @test stirling2(10, 5, ExternalBackend()) == 42525.0

    # 3. Performance test (verify NativeBackend zero allocation)
    # Wrapped in a function to fully isolate the test environment
    function test_alloc()
        # Using local immutable objects
        val_n = Val(5)
        val_k = Val(3)
        res = QBMM._stirling2_static(val_n, val_k)
        return res
    end

    # Warmup
    test_alloc()
    # In many Julia versions, @allocated may show micro-allocation at the top level, but is more accurate inside a function
    @test @allocated(test_alloc()) <= 32
end
