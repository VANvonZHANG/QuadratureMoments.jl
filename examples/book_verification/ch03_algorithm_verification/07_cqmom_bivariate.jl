# examples/book_verification/ch03_algorithm_verification/07_cqmom_bivariate.jl
#
# CQMOM for a bivariate Gaussian distribution.
# Book reference: Exercise 3.8 from Marchisio & Fox (2013),
#   "Computational Models for Polydisperse Particulate and Multiphase Systems."
#
# We consider a bivariate distribution built from two independent normals:
#   xi1 ~ N(mu1=1, sigma1=0.1)  =>  variance = sigma1^2 = 0.01
#   xi2 ~ N(mu2=2, sigma2=0.2)  =>  variance = sigma2^2 = 0.04
#
# The joint moments factor: m_{i,j} = E[xi1^i * xi2^j] = mx_{i+1} * my_{j+1}.
#
# Two configurations are compared:
#   (a) N = (2, 2) => 4-point quadrature from a 4x4 moment matrix
#   (b) N = (3, 3) => 9-point quadrature from a 6x6 moment matrix
#
# For both, we verify that the reconstructed moments match the exact values.

using QuadratureMoments
using StaticArrays
using LinearAlgebra
using Printf

println("=== Book Verification: Exercise 3.8 ===")
println("CQMOM for Bivariate Gaussian\n")

# ---------------------------------------------------------------------------
# 1. Raw moments of the marginal distributions
# ---------------------------------------------------------------------------

# xi1 ~ N(1, 0.01)  (mean=1, variance=0.01)
# mx[k] = E[xi1^(k-1)] for k = 1..6  (moments of order 0..5)
mx = [1.0, 1.0, 1.01, 1.03, 1.0606, 1.1030]

# xi2 ~ N(2, 0.04)  (mean=2, variance=0.04)
# my[k] = E[xi2^(k-1)] for k = 1..6  (moments of order 0..5)
my = [1.0, 2.0, 4.04, 8.24, 16.976, 35.752]

# Build the full 6x6 bivariate moment tensor  m[i,j] = mx[i]*my[j]
m_data = zeros(6, 6)
for i in 1:6, j in 1:6
    m_data[i, j] = mx[i] * my[j]
end

println("Full 6x6 bivariate moment tensor:")
println("  (rows: order in xi1, columns: order in xi2)\n")
for i in 1:6
    @printf("  order %d: [", i - 1)
    for j in 1:6
        @printf("%10.4f", m_data[i, j])
        if j < 6
            print(", ")
        end
    end
    println("]")
end
println()

# ---------------------------------------------------------------------------
# 2. CQMOM with N = (2, 2)  =>  4-point quadrature
# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Case (a): CQMOM with N = (2, 2)  =>  4 quadrature nodes")
println("=" ^ 60)

m_4 = SMatrix{4, 4, Float64}(m_data[1:4, 1:4])
res4 = invert_moments(CQMOM((2, 2)), m_4)

K4 = length(res4.weights)
println("\nQuadrature nodes and weights:")
for k in 1:K4
    @printf("  Node %d: (xi1 = %12.6f, xi2 = %12.6f),  weight = %12.6f\n",
        k, res4.nodes[k, 1], res4.nodes[k, 2], res4.weights[k])
end

# Verification: reconstruct moments up to order (1,1) from the quadrature
println("\nMoment reconstruction verification (up to order 1 in each dim):")
all_pass4 = true
for i in 0:1, j in 0:1
    pred = sum(
        res4.weights[k] * res4.nodes[k, 1]^i * res4.nodes[k, 2]^j for k in 1:K4
    )
    exact = m_data[i + 1, j + 1]
    rel_err = abs(pred - exact) / max(abs(exact), 1e-15)
    status = rel_err < 1e-8 ? "PASS" : "FAIL"
    if status == "FAIL"
        global all_pass4 = false
    end
    @printf("  m(%d,%d): predicted = %12.6f, exact = %12.6f, rel_err = %8.2e  [%s]\n",
        i, j, pred, exact, rel_err, status)
end
println()

# ---------------------------------------------------------------------------
# 3. CQMOM with N = (3, 3)  =>  9-point quadrature
# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Case (b): CQMOM with N = (3, 3)  =>  9 quadrature nodes")
println("=" ^ 60)

m_6 = SMatrix{6, 6, Float64}(m_data)
res9 = invert_moments(CQMOM((3, 3)), m_6)

K9 = length(res9.weights)
println("\nQuadrature nodes and weights:")
for k in 1:K9
    @printf("  Node %d: (xi1 = %12.6f, xi2 = %12.6f),  weight = %12.6f\n",
        k, res9.nodes[k, 1], res9.nodes[k, 2], res9.weights[k])
end

# Verification: reconstruct moments up to order (2,2) from the quadrature
println("\nMoment reconstruction verification (up to order 2 in each dim):")
all_pass9 = true
for i in 0:2, j in 0:2
    pred = sum(
        res9.weights[k] * res9.nodes[k, 1]^i * res9.nodes[k, 2]^j for k in 1:K9
    )
    exact = m_data[i + 1, j + 1]
    rel_err = abs(pred - exact) / max(abs(exact), 1e-15)
    status = rel_err < 1e-8 ? "PASS" : "FAIL"
    if status == "FAIL"
        global all_pass9 = false
    end
    @printf("  m(%d,%d): predicted = %12.6f, exact = %12.6f, rel_err = %8.2e  [%s]\n",
        i, j, pred, exact, rel_err, status)
end
println()

# ---------------------------------------------------------------------------
# 3b. Visualization: CQMOM 4-point vs 9-point comparison
# ---------------------------------------------------------------------------
try
    using Plots
    gr()

    # Distribution parameters (from section 1 above)
    μ_x, σ_x = 1.0, 0.1
    μ_y, σ_y = 2.0, 0.2

    # True bivariate NDF contours
    ξ1_range = range(μ_x - 3σ_x, μ_x + 3σ_x, length=50)
    ξ2_range = range(μ_y - 3σ_y, μ_y + 3σ_y, length=50)
    ndf_2d = [exp(-(ξ1 - μ_x)^2 / (2 * σ_x^2)) / sqrt(2π * σ_x^2) *
              exp(-(ξ2 - μ_y)^2 / (2 * σ_y^2)) / sqrt(2π * σ_y^2)
              for ξ2 in ξ2_range, ξ1 in ξ1_range]

    p1 = contour(ξ1_range, ξ2_range, ndf_2d, fill=false, color=:gray, lw=1,
                 title="CQMOM N=(2,2): 4 nodes", xlabel="ξ₁", ylabel="ξ₂")
    scatter!(p1, res4.nodes[:, 1], res4.nodes[:, 2],
             ms=res4.weights .* 300, color=:red, label="Nodes", markerstrokewidth=2)

    p2 = contour(ξ1_range, ξ2_range, ndf_2d, fill=false, color=:gray, lw=1,
                 title="CQMOM N=(3,3): 9 nodes", xlabel="ξ₁", ylabel="ξ₂",
                 ylims=(μ_y - 3σ_y, μ_y + 3σ_y))
    # Filter out near-zero-weight outlier nodes for display
    mask9 = res9.weights .> 1e-6
    scatter!(p2, res9.nodes[mask9, 1], res9.nodes[mask9, 2],
             ms=res9.weights[mask9] .* 300, color=:red, label="Nodes", markerstrokewidth=2)

    p = plot(p1, p2, layout=(1, 2), size=(900, 400))
    mkpath(joinpath(@__DIR__, "..", "output"))
    savefig(p, joinpath(@__DIR__, "..", "output", "ch03_07_cqmom_bivariate.png"))
    println("\n  Plot saved to output/ch03_07_cqmom_bivariate.png")
catch e
    @show e
    println("\n  (Install Plots.jl to generate plots)")
end

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
@printf("  CQMOM N=(2,2)  [4 nodes] : %s\n", all_pass4 ? "PASS" : "FAIL")
@printf("  CQMOM N=(3,3)  [9 nodes] : %s\n", all_pass9 ? "PASS" : "FAIL")
println()
if all_pass4 && all_pass9
    println("  Overall: PASS  --  All moment reconstructions within tolerance.")
else
    println("  Overall: FAIL  --  Some moment reconstructions exceed tolerance.")
end
