# examples/01_univariate_inversion.jl
using QuadratureMoments
using StaticArrays
using Printf

println("=== Module 1: Inversion Algorithms for Static Distributions ===")
println("1. Univariate Inversion Verification\n")

# Moments of a normal distribution (mean mu=5.0, std sigma=1.0)
m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]

println("Input moment sequence: ", m)
println()

# 1. Using the Wheeler algorithm (N=4)
method_w = Wheeler(4)
res_w = invert_moments(method_w, m)

println("--- Wheeler algorithm results (N=4) ---")
for i in 1:4
    @printf(
        "Node %d: %8.4f, Weight: %8.4f\n",
        i,
        res_w.nodes[i],
        res_w.weights[i]
    )
end
println()

# 2. Using the PD algorithm (N=4)
method_pd = PD(4)
res_pd = invert_moments(method_pd, m)

println("--- PD algorithm results (N=4) ---")
for i in 1:4
    @printf(
        "Node %d: %8.4f, Weight: %8.4f\n",
        i,
        res_pd.nodes[i],
        res_pd.weights[i]
    )
end
println()

println("Expected nodes should be close to: 2.6656, 4.2580, 5.7420, 7.3344")
println("Expected weights should be close to: 0.0459, 0.4541, 0.4541, 0.0459")
