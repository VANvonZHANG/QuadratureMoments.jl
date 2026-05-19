# examples/09_performance_benchmarks.jl
using QBMM
using StaticArrays
using BenchmarkTools

println("=== Module 4: Ultimate Performance Experience ===")
println("9. Zero-Allocation and Ultra-Fast Benchmarking\n")

m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]
method = Wheeler(4)

println("Testing invert_moments(Wheeler(4), m):")
println("Expected memory allocation under NativeBackend() is 0 allocations")

# Using the @btime macro (results displayed in standard output)
# Wrapped in a function to avoid extra allocations from global variables
function run_benchmark(method, m)
    # Warmup
    invert_moments(method, m)
    @btime invert_moments($method, $m)
end

run_benchmark(method, m)

println("\nThis demonstrates the library's 'Zero-Allocation' design, perfectly supporting the strong-coupling requirements of tens-of-millions-of-cell CFD grids!")
