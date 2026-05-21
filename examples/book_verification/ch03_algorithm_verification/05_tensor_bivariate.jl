# examples/book_verification/ch03_algorithm_verification/05_tensor_bivariate.jl
#
# Book reference: Exercise 3.6 from
#   D. L. Marchisio and R. O. Fox, "Computational Models for Polydisperse
#   Particulate and Multiphase Systems," Cambridge University Press, 2013.
#
# Tensor-product QMOM for a bivariate distribution (M=2, N1=N2=2).
# The joint NDF is assumed to be a product of two independent normal
# distributions:  xi1 ~ N(1, 0.1^2),  xi2 ~ N(2, 0.2^2).
#
# The tensor-product approach performs 1D inversions on each marginal
# and then forms the 4 = 2 x 2 quadrature nodes via Cartesian product.

using QuadratureMoments
using StaticArrays
using LinearAlgebra
using Printf

println("============================================================")
println(" Exercise 3.6: Tensor-Product QMOM for Bivariate Distribution")
println("============================================================")
println()

# --- Construct the moment tensor for two independent normals ---
# xi1 ~ N(mu1=1, sigma1^2=0.01)  =>  raw moments [1, 1, 1.01, 1.03]
# xi2 ~ N(mu2=2, sigma2^2=0.04)  =>  raw moments [1, 2, 4.04, 8.24]
#
# Because the distributions are independent, mixed moments factor:
#   m_{i,j} = E[xi1^i * xi2^j] = m1_i * m2_j

mx = [1.0, 1.0, 1.01, 1.03]   # raw moments of xi1
my = [1.0, 2.0, 4.04, 8.24]    # raw moments of xi2

m_data = zeros(4, 4)
for i in 1:4, j in 1:4
    m_data[i, j] = mx[i] * my[j]
end
m_static = SMatrix{4, 4, Float64}(m_data)

println("Marginal moments of xi1 ~ N(1, 0.01): ", mx)
println("Marginal moments of xi2 ~ N(2, 0.04): ", my)
println()
println("Input moment tensor (4 x 4):")
for i in 1:4
    @printf("  [%8.4f  %8.4f  %8.4f  %8.4f]\n", m_data[i, 1], m_data[i, 2], m_data[i, 3], m_data[i, 4])
end
println()

# --- Tensor-product QMOM inversion ---
# (2, 2) means N1=2 nodes in dim 1, N2=2 nodes in dim 2, giving 4 total nodes.
method = TensorQMOM((2, 2))
res = invert_moments(method, m_static)

println("--- Tensor-product QMOM results (4 nodes) ---")
for k in 1:4
    @printf(
        "  Node %d: xi1 = %10.6f, xi2 = %10.6f, weight = %10.6f\n",
        k,
        res.nodes[k, 1],
        res.nodes[k, 2],
        res.weights[k]
    )
end
println()

# --- Visualization ---
try
    using Plots
    gr()

    # True bivariate NDF: n(xi1,xi2) = n_x(xi1) * n_y(xi2)
    # where n_x = N(1, 0.1), n_y = N(2, 0.2)
    mu_x, sigma_x = 1.0, 0.1
    mu_y, sigma_y = 2.0, 0.2

    xi1_range = range(0.5, 1.5, length=50)
    xi2_range = range(1.3, 2.7, length=50)
    ndf_2d = [exp(-(xi1 - mu_x)^2 / (2 * sigma_x^2)) / sqrt(2pi * sigma_x^2) *
              exp(-(xi2 - mu_y)^2 / (2 * sigma_y^2)) / sqrt(2pi * sigma_y^2)
              for xi2 in xi2_range, xi1 in xi1_range]

    p = contour(xi1_range, xi2_range, ndf_2d, fill=false, color=:gray, lw=1,
                xlabel="ξ₁", ylabel="ξ₂",
                title="Ex 3.6: Tensor-Product QMOM (2,2)")
    scatter!(res.nodes[:, 1], res.nodes[:, 2],
             ms=res.weights .* 200, color=:red, label="Nodes",
             markerstrokewidth=2)
    mkpath(joinpath(@__DIR__, "..", "output"))
    savefig(p, joinpath(@__DIR__, "..", "output", "ch03_05_tensor_bivariate.png"))
    println("\n  Plot saved to output/ch03_05_tensor_bivariate.png")
catch e
    @show e
    println("\n  (Install Plots.jl to generate plots)")
end
println()

# --- Verification: reconstruct moments from quadrature and compare ---
println("--- Moment reconstruction verification ---")
let max_rel_err = 0.0, all_pass = true, tol = 1.0e-8
    for i in 0:2, j in 0:2
        pred = sum(res.weights[k] * res.nodes[k, 1]^i * res.nodes[k, 2]^j for k in 1:4)
        exact = m_data[i + 1, j + 1]
        abs_err = abs(pred - exact)
        rel_err = abs_err / (abs(exact) + eps())
        max_rel_err = max(max_rel_err, rel_err)
        status = rel_err < tol ? "PASS" : "FAIL"
        if rel_err >= tol
            all_pass = false
        end
        @printf(
            "  m_{%d,%d}: exact = %12.6f, reconstructed = %12.6f, rel_err = %8.2e  [%s]\n",
            i,
            j,
            exact,
            pred,
            rel_err,
            status
        )
    end
    println()

    @printf("Maximum relative error: %.2e\n", max_rel_err)
    @printf("Tolerance:             %.2e\n", tol)
    println()
    println(all_pass ? "PASS" : "FAIL")
end
