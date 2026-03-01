using BenchmarkTools
using StaticArrays
using QBMM
using LinearAlgebra

# 创建 Benchmark 组
const suite = BenchmarkGroup()

suite["Wheeler Inversion"] = BenchmarkGroup()
suite["PD Inversion"] = BenchmarkGroup()
suite["EQMOM (Gaussian)"] = BenchmarkGroup()
suite["CQMOM (2D)"] = BenchmarkGroup()
suite["ECQMOM (2D)"] = BenchmarkGroup()
suite["TensorQMOM (2D)"] = BenchmarkGroup()
suite["DQMOM Solve"] = BenchmarkGroup()

# ---------------------------------------------------------
# 1. Univariate Inversion (N=3)
# ---------------------------------------------------------
mu, sigma = 5.0, 1.0
N_1d = 3
# 6 moments for QMOM/PD, 7 for EQMOM
m_1d_base = [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26146.0]
m_1d_static_6 = SVector{6, Float64}(m_1d_base[1:6])
m_1d_static_7 = SVector{7, Float64}(m_1d_base[1:7])

suite["Wheeler Inversion"]["Static Array"] = @benchmarkable wheeler_inversion($m_1d_static_6)
suite["PD Inversion"]["Static Array"] = @benchmarkable pd_inversion($m_1d_static_6)
suite["EQMOM (Gaussian)"]["Static Array"] = @benchmarkable invert_moments($(EQMOM(3, GaussianKernel())), $m_1d_static_7)

# ---------------------------------------------------------
# 2. Multivariate Inversion (2D, N=(2,2))
# ---------------------------------------------------------
# N1=2, N2=2 -> Total 4 nodes
# CQMOM needs (2N1)x(2N2) = 4x4 moments
# ECQMOM needs (2N1+1)x(2N2+1) = 5x5 moments
mx = [1.0, 5.2, 29.0, 170.8, 1051.0]
my = [1.0, 10.0, 100.25, 1007.5, 10150.5625]

m_4x4 = SMatrix{4, 4, Float64}([mx[i]*my[j] for i in 1:4, j in 1:4])
m_5x5 = SMatrix{5, 5, Float64}([mx[i]*my[j] for i in 1:5, j in 1:5])

suite["CQMOM (2D)"]["Recursive Static"] = @benchmarkable invert_moments($(CQMOM((2,2))), $m_4x4)
suite["ECQMOM (2D)"]["Recursive Static"] = @benchmarkable invert_moments($(ECQMOM((2,1))), $(SMatrix{5, 3, Float64}(m_5x5[1:5, 1:3])))
suite["TensorQMOM (2D)"]["Static"] = @benchmarkable invert_moments($(TensorQMOM((2,2))), $m_4x4)

# ---------------------------------------------------------
# 3. DQMOM Benchmarks
# ---------------------------------------------------------
dq_nodes_static = SVector{3, Float64}(1.0, 2.0, 3.0)
dq_source_static = SVector{6, Float64}(0.1, -0.5, 1.2, 0.0, 3.1, -1.1)

suite["DQMOM Solve"]["Static Array"] = @benchmarkable dqmom_solve($dq_nodes_static, $dq_source_static)

# ---------------------------------------------------------
# 运行并打印简明报告
# ---------------------------------------------------------

println("=========================================================")
println("    QBMM.jl High-Performance Benchmark Suite")
println("=========================================================")

tune!(suite)
results = run(suite, verbose = false)

println("\n--- Univariate (1D, N=3) ---")
print("Wheeler Inversion:  ")
display(median(results["Wheeler Inversion"]["Static Array"]))
print("\nPD Inversion:       ")
display(median(results["PD Inversion"]["Static Array"]))
print("\nEQMOM (Gaussian):   ")
display(median(results["EQMOM (Gaussian)"]["Static Array"]))

println("\n\n--- Multivariate (2D) ---")
print("CQMOM (Recursive):  ")
display(median(results["CQMOM (2D)"]["Recursive Static"]))
print("\nECQMOM (Recursive): ")
display(median(results["ECQMOM (2D)"]["Recursive Static"]))
print("\nTensorQMOM:         ")
display(median(results["TensorQMOM (2D)"]["Static"]))

println("\n\n--- Direct Method ---")
print("DQMOM Solve (N=3):  ")
display(median(results["DQMOM Solve"]["Static Array"]))

println("\n=========================================================")
println("Note: All benchmarks use StaticArrays (Zero Allocations goal)")
println("=========================================================")
