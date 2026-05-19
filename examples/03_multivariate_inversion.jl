# examples/03_multivariate_inversion.jl
using QBMM
using StaticArrays
using Printf

println("=== Module 1: Inversion Algorithms for Static Distributions ===")
println("3. Multivariate CQMOM vs Brute Algorithm Comparison\n")

# Construct 2D moment tensor (product of two independent distributions as demonstration)
# x ~ N(1, 0.1), y ~ N(2, 0.2)
mx = [1.0, 1.0, 1.1, 1.3]
my = [1.0, 2.0, 4.2, 9.2]

m_data = zeros(4, 4)
for i in 1:4, j in 1:4
    m_data[i, j] = mx[i] * my[j]
end
m_2d = SMatrix{4,4,Float64}(m_data)

println("Input 2D moment tensor (4x4):")
display(m_2d)
println("\n")

# 1. CQMOM inversion
method_cq = CQMOM((2, 2))
res_cq = invert_moments(method_cq, m_2d)

println("--- CQMOM recursive dimension-reduction results ---")
for i in 1:4
    @printf(
        "Node %d: [%8.4f, %8.4f], Weight: %8.4f\n",
        i,
        res_cq.nodes[i, 1],
        res_cq.nodes[i, 2],
        res_cq.weights[i]
    )
end
println()

# 2. BruteQMOM inversion
method_brute = BruteQMOM(2, 4) # 2D, 4 nodes total
try
    res_brute = invert_moments(method_brute, m_2d)
    println("--- BruteQMOM direct nonlinear solve results ---")
    for i in 1:4
        @printf(
            "Node %d: [%8.4f, %8.4f], Weight: %8.4f\n",
            i,
            res_brute.nodes[i, 1],
            res_brute.nodes[i, 2],
            res_brute.weights[i]
        )
    end
catch e
    println("BruteQMOM solve failed (common in non-convex optimization, highlighting the importance of CQMOM): ", e)
end
