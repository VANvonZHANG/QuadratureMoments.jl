# examples/01_univariate_inversion.jl
using QBMM
using StaticArrays
using Printf

println("=== 模块一：静态分布的反演算法 ===")
println("1. 单变量反演验证 (Univariate Inversion)
")

# 正态分布的矩（均值 μ=5.0, 标准差 σ=1.0）
m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]

println("输入矩序列: ", m)
println()

# 1. 使用 Wheeler 算法 (N=4)
method_w = Wheeler(4)
res_w = invert_moments(method_w, m)

println("--- Wheeler 算法结果 (N=4) ---")
for i in 1:4
    @printf(
        "节点 %d: %8.4f, 权重: %8.4f
",
        i,
        res_w.nodes[i],
        res_w.weights[i]
    )
end
println()

# 2. 使用 PD 算法 (N=4)
method_pd = PD(4)
res_pd = invert_moments(method_pd, m)

println("--- PD 算法结果 (N=4) ---")
for i in 1:4
    @printf(
        "节点 %d: %8.4f, 权重: %8.4f
",
        i,
        res_pd.nodes[i],
        res_pd.weights[i]
    )
end
println()

println("预期节点应接近: 2.6656, 4.2580, 5.7420, 7.3344")
println("预期权重应接近: 0.0459, 0.4541, 0.4541, 0.0459")
