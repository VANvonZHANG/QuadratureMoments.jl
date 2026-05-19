# examples/06_composite_source_terms.jl
using QBMM
using StaticArrays

println("=== 模块三：物理源项与动态演化 ===")
println("6. DQMOM 与物理源项的模块化组合\n")

N = 3
weights = @SVector [0.3, 0.4, 0.3]
nodes = @SVector [1.0, 2.0, 3.0]

println("初始状态:")
println("Weights: ", weights)
println("Nodes: ", nodes)
println()

# 定义并组合物理源项
growth = ParticleGrowth(x -> 0.1)                # 恒定生长率 0.1
aggregation = Aggregation((x, y) -> 0.05) # 恒定核聚并 0.05
physics = growth + aggregation

println("配置物理模型: ParticleGrowth(0.1) + Aggregation(0.05)\n")

# 计算前 2N 个矩的源项 S_k
S_k = compute_source_terms(physics, nodes, weights, Val(2N))

println("合并后的矩源项 S_k:")
println(S_k)
println()

# DQMOM 瞬态求解
method = DQMOM(N)
da, db = dqmom_solve(method, nodes, S_k)

println("--- DQMOM 瞬时演化速率 ---")
println("权重变化率 da/dt: ", da)
println("加权节点变化率 db/dt (b = w * ξ): ", db)
