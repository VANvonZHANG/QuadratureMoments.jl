using BenchmarkTools
using StaticArrays
using QBMM

# 创建 Benchmark 组
const suite = BenchmarkGroup()

suite["Wheeler Inversion"] = BenchmarkGroup()
suite["PD Inversion"] = BenchmarkGroup()
suite["DQMOM Solve"] = BenchmarkGroup()

# ---------------------------------------------------------
# 1. Wheeler Algorithm Benchmarks (QMOM)
# ---------------------------------------------------------

# 基于标准正态分布的矩
mu = 0.0
sigma = 1.0
N = 3 # 3 nodes, 6 moments
base_moments = Float64[
    1.0, 
    mu, 
    mu^2 + sigma^2, 
    mu^3 + 3*mu*sigma^2, 
    mu^4 + 6*mu^2*sigma^2 + 3*sigma^4, 
    mu^5 + 10*mu^3*sigma^2 + 15*mu*sigma^4
]
static_moments = SVector{2N, Float64}(base_moments)

suite["Wheeler Inversion"]["Base Array"] = @benchmarkable wheeler_inversion($base_moments)
suite["Wheeler Inversion"]["Static Array"] = @benchmarkable wheeler_inversion($static_moments)

# PD Inversion
suite["PD Inversion"]["Base Array"] = @benchmarkable pd_inversion($base_moments)
suite["PD Inversion"]["Static Array"] = @benchmarkable pd_inversion($static_moments)

# ---------------------------------------------------------
# 2. DQMOM Benchmarks
# ---------------------------------------------------------

# N = 3 
dq_nodes_base = [1.0, 2.0, 3.0]
dq_source_base = [0.1, -0.5, 1.2, 0.0, 3.1, -1.1]

dq_nodes_static = SVector{3, Float64}(dq_nodes_base)
dq_source_static = SVector{6, Float64}(dq_source_base)

suite["DQMOM Solve"]["Base Array"] = @benchmarkable dqmom_solve($dq_nodes_base, $dq_source_base)
suite["DQMOM Solve"]["Static Array"] = @benchmarkable dqmom_solve($dq_nodes_static, $dq_source_static)

# ---------------------------------------------------------
# 运行并打印报告
# ---------------------------------------------------------

println("=========================================================")
println("    QBMM.jl Performance Benchmarks (N=3 Nodes)")
println("=========================================================
")

# 执行跑分预热
tune!(suite)
results = run(suite, verbose = true)

println("
=========================================================")
println("    Detailed Benchmark Results")
println("=========================================================
")

println("--- Wheeler Inversion (QMOM) ---")
println("1. Base Array Allocation & Time:")
display(results["Wheeler Inversion"]["Base Array"])
println("
")

println("2. Static Array Allocation & Time:")
display(results["Wheeler Inversion"]["Static Array"])
println("\n")

println("--- PD Inversion (QMOM) ---")
println("1. Base Array Allocation & Time:")
display(results["PD Inversion"]["Base Array"])
println("\n")

println("2. Static Array Allocation & Time:")
display(results["PD Inversion"]["Static Array"])
println("\n")

println("--- DQMOM Linear System Solve ---")
println("1. Base Array Allocation & Time:")
display(results["DQMOM Solve"]["Base Array"])
println("
")

println("2. Static Array Allocation & Time:")
display(results["DQMOM Solve"]["Static Array"])
println("
")

# 输出对比比例
w_base_med = median(results["Wheeler Inversion"]["Base Array"]).time
w_stat_med = median(results["Wheeler Inversion"]["Static Array"]).time
w_speedup = w_base_med / w_stat_med

pd_base_med = median(results["PD Inversion"]["Base Array"]).time
pd_stat_med = median(results["PD Inversion"]["Static Array"]).time
pd_speedup = pd_base_med / pd_stat_med

dq_base_med = median(results["DQMOM Solve"]["Base Array"]).time
dq_stat_med = median(results["DQMOM Solve"]["Static Array"]).time
dq_speedup = dq_base_med / dq_stat_med

println("=========================================================")
println("    Speedup Summary (StaticArray vs Base Array)")
println("=========================================================")
println("Wheeler Algorithm Speedup: $(round(w_speedup, digits=2))x")
println("PD Algorithm Speedup:      $(round(pd_speedup, digits=2))x")
println("DQMOM Solve Speedup:       $(round(dq_speedup, digits=2))x")
println("=========================================================")
