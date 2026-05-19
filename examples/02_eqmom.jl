# examples/02_eqmom.jl
using QBMM
using StaticArrays
using Printf

println("=== 模块一：静态分布的反演算法 ===")
println("2. 连续分布重建 (EQMOM)
")

# 具有一定宽度的高斯核组成的矩序列 (混合正态分布)
m_extended = @SVector [1.0, 5.2, 29.0, 170.8, 1051.0]

println("输入矩序列: ", m_extended)
println()

# 使用 N=2 的高斯核 EQMOM (因为给定了5个矩)
method_eq = EQMOM(2, GaussianKernel())
res_eq = invert_moments(method_eq, m_extended)

println("--- EQMOM (Gaussian Kernel, N=2) 结果 ---")
for i in 1:2
    @printf(
        "节点 (均值) %d: %8.4f, 权重: %8.4f
",
        i,
        res_eq.nodes[i],
        res_eq.weights[i]
    )
end
@printf(
    "全局带宽参数 σ: %8.4f
",
    res_eq.sigmas[1]
)
println()
println("对比标准 QMOM（所有概率质量集中在离散节点上），EQMOM 提供了一个具有非零 σ 的平滑分布的近似。")
