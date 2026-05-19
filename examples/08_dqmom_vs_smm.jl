# examples/08_dqmom_vs_smm.jl
using QBMM
using StaticArrays

println("=== 模块三：物理源项与动态演化 ===")
println("8. 演化稳定性终极对决：DQMOM vs Standard QMOM (SMM)\n")

function smm_rhs(m, method, physics)
    N = div(length(m), 2)
    # 首先反演节点和权重
    res = invert_moments(method, m)
    # 若返回的节点不是1D向量，尝试转换（Wheeler应返回向量）
    nodes = SVector{N}(vec(res.nodes))
    weights = SVector{N}(res.weights)
    # 计算矩源项
    return compute_source_terms(physics, nodes, weights, Val(2N))
end

N = 3
w0 = @SVector [0.2, 0.6, 0.2]
x0 = @SVector [1.0, 2.0, 3.0]
m0 = SVector{2N}(sum(w0[i] * x0[i]^k for i in 1:N) for k in 0:(2N - 1))

# 剧烈物理过程 (常数核聚并)
physics = Aggregation((x, y) -> 1.0)
method = Wheeler(N)

println("开始 SMM (标准矩方法) 时间积分...")
println("SMM 通过演化矩向量，然后在每个时间步使用 Wheeler 算法反演。")
println("这极有可能因为高阶矩快速增长带来的数值截断误差，导致某一步数据不可实现从而触发崩溃。\n")

dt = 0.05
m_current = m0
success_smm = true
for step in 1:20
    try
        dm = smm_rhs(m_current, method, physics)
        global m_current = m_current .+ dt .* dm
    catch e
        println("=> SMM 在第 $step 步崩溃！\n错误信息: $e")
        println("=> 原因: m 已经无法满足 Stieltjes 条件，失去可实现性。")
        global success_smm = false
        break
    end
end
if success_smm
    println("=> SMM 惊险跑完了全程。")
end

println("\n---------------------------------------------------------")
println("作为对比，同样的物理和步长，DQMOM 直接演化节点和权重：\n")
u_current = vcat(w0, SVector{N}(w0 .* x0))
dqmom_method = DQMOM(N)

success_dqmom = true
for step in 1:20
    try
        w = SVector{N}(u_current[1:N])
        b = SVector{N}(u_current[(N + 1):2N])
        nodes = SVector{N}(b ./ max.(w, 1e-12))
        S_k = compute_source_terms(physics, nodes, w, Val(2N))
        da, db = dqmom_solve(dqmom_method, nodes, S_k)
        global u_current = u_current .+ dt .* vcat(da, db)
    catch e
        println("=> DQMOM 崩溃: ", e)
        global success_dqmom = false
        break
    end
end

if success_dqmom
    println("=> DQMOM 极其稳定地跑完了 20 步！")
    println("=> 结论：由于 DQMOM 不涉及“从高阶矩到节点”的反复非线性反演，其在时间积分时的鲁棒性远超 SMM。")
end
