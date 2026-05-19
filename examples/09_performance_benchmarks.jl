# examples/09_performance_benchmarks.jl
using QBMM
using StaticArrays
using BenchmarkTools

println("=== 模块四：极致性能体验 ===")
println("9. 零内存分配与极速基准测试
")

m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]
method = Wheeler(4)

println("测试 invert_moments(Wheeler(4), m):")
println("预期在 NativeBackend() 下分配内存为 0 allocations")

# 利用 @btime 宏（会在标准输出里显示结果）
# 将其放入函数中以避免全局变量带来的额外分配
function run_benchmark(method, m)
    # 预热
    invert_moments(method, m)
    @btime invert_moments($method, $m)
end

run_benchmark(method, m)

println("
这体现了库的 'Zero-Allocation' 设计，完美支持千万级网格的 CFD 强耦合要求！")
