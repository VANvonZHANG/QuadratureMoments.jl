# examples/05_adaptive_wheeler.jl
using QBMM
using StaticArrays
using Printf

println("=== 模块二：鲁棒性与异常处理 ===")
println("5. 极值情况与自适应降阶 (Adaptive Wheeler)
")

# 构造一个退化的矩序列 (仅由2个离散节点生成的矩序列)
# 节点 x = [1.0, 2.0], 权重 w = [0.5, 0.5]
nodes = [1.0, 2.0]
weights = [0.5, 0.5]
m_deg_array = [sum(weights[i] * nodes[i]^k for i in 1:2) for k in 0:7]
m_degenerate = SVector{8,Float64}(m_deg_array...)

println("退化矩序列 (理论上只支持 N=2):")
println(m_degenerate)
println()

println("尝试请求 N=4 的 Wheeler 反演 (自适应框架应该将其安全降级到 N=2)...")

method_w = Wheeler(4)
res = invert_moments(method_w, m_degenerate)

println("
--- 降阶后的结果 ---")
for i in 1:4
    @printf(
        "节点 %d: %8.4f, 权重: %8.4f
",
        i,
        res.nodes[i],
        res.weights[i]
    )
end
println("
可以看到，多余的两个节点权重被安全地置为了 0.0，且没有引发除零崩溃。")
