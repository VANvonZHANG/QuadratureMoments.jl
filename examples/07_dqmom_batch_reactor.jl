# examples/07_dqmom_batch_reactor.jl
using QBMM
using StaticArrays
using Printf

println("=== 模块三：物理源项与动态演化 ===")
println("7. 间歇反应器的时间演化模拟 (DQMOM Batch Reactor) - RK4 积分\n")

# 为保持轻量级（无外部依赖），我们直接实现 RK4 积分
function dqmom_rhs(u, method, physics)
    N = div(length(u), 2)
    weights = SVector{N}(u[1:N])
    weighted_nodes = SVector{N}(u[(N + 1):2N])

    # 还原节点，并加入安全保护防止除以0
    nodes = SVector{N}(weighted_nodes ./ max.(weights, 1e-12))

    # 计算物理源项与演化速率
    S_k = compute_source_terms(physics, nodes, weights, Val(2N))
    da, db = dqmom_solve(method, nodes, S_k)

    return vcat(da, db)
end

# 初始化问题
N = 3
w0 = @SVector [0.2, 0.6, 0.2]
x0 = @SVector [1.0, 2.0, 3.0]
b0 = SVector{N}(w0 .* x0)
u = vcat(w0, b0)

# 物理模型：纯生长 G(x) = 0.5
physics = ParticleGrowth(x -> 0.5)
method = DQMOM(N)

dt = 0.1
t_end = 2.0
steps = Int(t_end / dt)

println("初始节点: ", x0)
println("纯生长模型 G=0.5，理论上所有节点在 t=2.0 时应增加 1.0\n")

for i in 1:steps
    # RK4 step
    k1 = dqmom_rhs(u, method, physics)
    k2 = dqmom_rhs(u .+ 0.5 * dt * k1, method, physics)
    k3 = dqmom_rhs(u .+ 0.5 * dt * k2, method, physics)
    k4 = dqmom_rhs(u .+ dt * k3, method, physics)
    global u = u .+ (dt / 6.0) * (k1 .+ 2k2 .+ 2k3 .+ k4)
end

w_final = u[1:N]
x_final = u[(N + 1):2N] ./ max.(w_final, 1e-12)

println("积分到 t=2.0 后的结果:")
println("权重: ", w_final)
println("节点: ", x_final)
println("可以看到，权重保持不变，各个节点正好向右平移了近似 1.0。完美符合解析解！")
