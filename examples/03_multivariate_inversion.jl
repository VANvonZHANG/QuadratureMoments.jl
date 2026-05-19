# examples/03_multivariate_inversion.jl
using QBMM
using StaticArrays
using Printf

println("=== 模块一：静态分布的反演算法 ===")
println("3. 多维 CQMOM 与 Brute 算法对比
")

# 构建 2D 矩张量 (两个独立分布的乘积作为演示)
# x ~ N(1, 0.1), y ~ N(2, 0.2)
mx = [1.0, 1.0, 1.1, 1.3] 
my = [1.0, 2.0, 4.2, 9.2] 

m_data = zeros(4, 4)
for i in 1:4, j in 1:4
    m_data[i, j] = mx[i] * my[j]
end
m_2d = SMatrix{4, 4, Float64}(m_data)

println("输入 2D 矩张量 (4x4):")
display(m_2d)
println("
")

# 1. CQMOM 反演
method_cq = CQMOM((2, 2))
res_cq = invert_moments(method_cq, m_2d)

println("--- CQMOM 递归降维结果 ---")
for i in 1:4
    @printf("节点 %d: [%8.4f, %8.4f], 权重: %8.4f
", i, res_cq.nodes[i, 1], res_cq.nodes[i, 2], res_cq.weights[i])
end
println()

# 2. BruteQMOM 反演
method_brute = BruteQMOM(2, 4) # 2维, 共4个节点
try
    res_brute = invert_moments(method_brute, m_2d)
    println("--- BruteQMOM 直接非线性求解结果 ---")
    for i in 1:4
        @printf("节点 %d: [%8.4f, %8.4f], 权重: %8.4f
", i, res_brute.nodes[i, 1], res_brute.nodes[i, 2], res_brute.weights[i])
    end
catch e
    println("BruteQMOM 求解失败 (这在非凸优化中常见，故更显 CQMOM 的重要性): ", e)
end
