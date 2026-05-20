# examples/book_verification/ch03_algorithm_verification/06_tensor_trivariate.jl
#
# Book reference:
#   Marchisio, D. L. & Fox, R. O. (2013).
#   Computational Models for Polydisperse Particulate and Multiphase Systems.
#   Exercise 3.7 -- Tensor-product QMOM for a trivariate distribution.
#
# Description:
#   This script verifies the Tensor-product QMOM (TensorQMOM) algorithm for a
#   3D distribution constructed from three independent delta distributions:
#       delta(V1 - 1.5) * delta(V2 - 2.5) * delta(V3 - 3.5)
#   With N1 = N2 = N3 = 2 (M = 3 dimensions, 2 nodes per dimension), the
#   algorithm produces 2*2*2 = 8 quadrature nodes and weights. Since each
#   coordinate has a single delta distribution, the exact moment tensor is
#   m_{i,j,k} = (1.5)^i * (2.5)^j * (3.5)^k and the quadrature should
#   recover the exact distribution.

using QuadratureMoments
using StaticArrays
using LinearAlgebra
using Printf

println("============================================================")
println(" Exercise 3.7: Tensor-Product QMOM for Trivariate Distribution")
println("   Marchisio & Fox (2013)")
println("============================================================")
println()

# --- Problem setup ---
# Three independent delta distributions at distinct abscissae
v1 = 1.5  # abscissa in dimension 1
v2 = 2.5  # abscissa in dimension 2
v3 = 3.5  # abscissa in dimension 3

N1, N2, N3 = 2, 2, 2  # nodes per dimension

println("Delta distribution abscissae:")
@printf("   v1 = %.4f,  v2 = %.4f,  v3 = %.4f\n", v1, v2, v3)
println("Nodes per dimension: N1=$N1, N2=$N2, N3=$N3")
println("Total expected nodes: $(N1 * N2 * N3)")
println()

# --- Build the 4x4x4 moment tensor ---
# For a delta(V - c), the raw moment of order k is c^k.
# The multivariate moment tensor entry m_{i,j,k} = v1^i * v2^j * v3^k
# with i, j, k in {0, 1, 2, 3}, stored as a 4x4x4 array.
# TensorQMOM with N_tuple=(2,2,2) requires 2*N = 4 moments per dimension
# (orders 0 through 2*N-1 = 3) to perform the 1D Wheeler inversion.
L = 4  # number of moment orders per dimension (2 * N1)
m_data = zeros(L, L, L)
for i in 0:(L - 1), j in 0:(L - 1), k in 0:(L - 1)
    m_data[i + 1, j + 1, k + 1] = v1^i * v2^j * v3^k
end

println("Input moment tensor m_{i,j,k} (4x4x4), selected entries:")
for order in 0:3
    @printf("   m[%d,0,0] = %.6f  (v1^%d)\n", order, m_data[order + 1, 1, 1], order)
    @printf("   m[0,%d,0] = %.6f  (v2^%d)\n", order, m_data[1, order + 1, 1], order)
    @printf("   m[0,0,%d] = %.6f  (v3^%d)\n", order, m_data[1, 1, order + 1], order)
end
println()

# Convert to a StaticArray for the solver
m_static = SArray{Tuple{4, 4, 4}, Float64, 3, 64}(m_data)

# --- Perform tensor-product QMOM inversion ---
method = TensorQMOM((2, 2, 2))
res = invert_moments(method, m_static)

println("--- TensorQMOM inversion results ---")
println("Number of nodes: ", length(res.weights))
println()

# Sort nodes by weight (descending) for cleaner display
order = sortperm(res.weights; rev=true)

for idx in order
    @printf(
        "   Node %d: [v1=%10.6f, v2=%10.6f, v3=%10.6f], Weight = %12.6f\n",
        idx,
        res.nodes[idx, 1],
        res.nodes[idx, 2],
        res.nodes[idx, 3],
        res.weights[idx]
    )
end
println()

# --- Verification: reconstruct moments from quadrature ---
# Verify all moments with total order <= 3 (the unique-determinacy range
# for 2 nodes per dimension). This includes 20 moment entries.
println("--- Moment reconstruction verification (total order <= 3) ---")

all_pass, max_rel_err, n_checked = let res = res, m_data = m_data
    local _all_pass = true
    local _max_rel_err = 0.0
    local _n_checked = 0

    for i in 0:3, j in 0:3, k in 0:3
        total_order = i + j + k
        if total_order > 3
            continue
        end
        exact = m_data[i + 1, j + 1, k + 1]
        pred = sum(
            res.weights[n] *
            res.nodes[n, 1]^i *
            res.nodes[n, 2]^j *
            res.nodes[n, 3]^k for n in 1:length(res.weights)
        )
        rel_err = abs(pred - exact) / (abs(exact) + 1e-30)
        _max_rel_err = max(_max_rel_err, rel_err)
        _n_checked += 1

        status = rel_err < 1e-8 ? "PASS" : "FAIL"
        if status == "FAIL"
            _all_pass = false
        end
        @printf(
            "   m[%d,%d,%d]: exact = %12.6f, pred = %12.6f, rel_err = %.2e  [%s]\n",
            i,
            j,
            k,
            exact,
            pred,
            rel_err,
            status
        )
    end

    (_all_pass, _max_rel_err, _n_checked)
end

println()
@printf("Checked %d moments. Maximum relative error: %.2e\n", n_checked, max_rel_err)
println()

# --- Expected value checks ---
# With delta distributions, 7 of 8 weights should be zero (or near-zero),
# and one node should be at exactly (v1, v2, v3) with weight 1.0.
# Alternatively, the TensorQMOM decomposition may produce multiple nodes
# with the same abscissa in each dimension if N > 1 per dimension.

# Check that total weight sums to 1.0
total_weight = sum(res.weights)
weight_err = abs(total_weight - 1.0)
println("Weight conservation check:")
@printf("   Sum of weights = %.10f (error = %.2e)\n", total_weight, weight_err)
weight_pass = weight_err < 1e-8
all_pass = all_pass && weight_pass
println()

# Check first moments (means) against expected delta values
println("First moment (mean) checks:")
@printf("   E[v1]: exact = %.6f, pred = %.6f\n", v1, sum(res.weights .* res.nodes[:, 1]))
@printf("   E[v2]: exact = %.6f, pred = %.6f\n", v2, sum(res.weights .* res.nodes[:, 2]))
@printf("   E[v3]: exact = %.6f, pred = %.6f\n", v3, sum(res.weights .* res.nodes[:, 3]))
println()

# --- Summary ---
println("============================================================")
if all_pass
    println(" RESULT: PASS -- All moment reconstructions verified.")
else
    println(" RESULT: FAIL -- Some moment reconstructions exceeded tolerance.")
end
println("============================================================")
