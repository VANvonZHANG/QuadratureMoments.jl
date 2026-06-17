# 02_wheeler_zero_mean.jl
#
# Book reference: Exercise 3.2 from
#   Marchisio & Fox (2013), "Computational Models for Polydisperse
#   Particulate and Multiphase Systems", Cambridge University Press.
#
# Demonstrates the Wheeler algorithm on a Normal(0,1) distribution where
# the Product-Difference (PD) algorithm fails due to the first moment
# being zero (m_1 = 0 causes division by zero in the PD recurrence).
# With N(0,1) all odd moments are exactly zero.
#
# Expected results:
#   Nodes   ~ {-2.3344, -0.7420, 0.7420, 2.3344}  (symmetric about zero)
#   Weights ~ {0.0459, 0.4541, 0.4541, 0.0459}

using QuadratureMoments
using StaticArrays
using Printf

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

N = 4                     # number of quadrature nodes
Nmom = 2 * N              # moments required by the inversion algorithms

# Raw moments of N(mu=0, sigma=1) for k=0..7:
#   Even moments:  m_{2k} = (2k-1)!!  (double factorial)
#   Odd moments:   m_{2k+1} = 0
m = @SVector [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0]

println("=== Wheeler Algorithm Verification: Gaussian N(0,1) ===")
println("Reference: Marchisio & Fox (2013), Exercise 3.2")
println("Number of nodes N = ", N)
println("Input moments (k=0..7): ", m)
println()

# ---------------------------------------------------------------------------
# Show that the PD algorithm fails on zero-mean input
# ---------------------------------------------------------------------------

println("--- Attempting PD Algorithm (expected to fail) ---")
method_pd = PD(N)
pd_failed = true  # default: expect failure
try
    res_pd = invert_moments(method_pd, m)
    nodes_pd = vec(res_pd.nodes)
    weights_pd = res_pd.weights
    println("  PD unexpectedly succeeded -- this should not happen.")
    println("  Nodes:  ", nodes_pd)
    println("  Weights:", weights_pd)
    global pd_failed = false
catch e
    println("  PD threw an error (as expected):")
    println("  ", e)
end
println()

# ---------------------------------------------------------------------------
# Run the Wheeler algorithm (should succeed)
# ---------------------------------------------------------------------------

println("--- Wheeler Algorithm Results ---")
method_w = Wheeler(N)
res = invert_moments(method_w, m)

nodes = vec(res.nodes)
weights = res.weights

for i in 1:N
    @printf("  Node %d: %10.4f    Weight: %10.4f\n", i, nodes[i], weights[i])
end
println()

# ---------------------------------------------------------------------------
# Compare with book values
# ---------------------------------------------------------------------------

book_nodes = [-2.3344, -0.7420, 0.7420, 2.3344]
book_weights = [0.0459, 0.4541, 0.4541, 0.0459]

println("--- Comparison with Book Values (Exercise 3.2) ---")
tol_book = 1e-2  # book prints 4 decimal places

all_book_match = true
for i in 1:N
    n_ok = abs(nodes[i] - book_nodes[i]) < tol_book
    w_ok = abs(weights[i] - book_weights[i]) < tol_book
    if !n_ok || !w_ok
        global all_book_match = false
    end
    @printf(
        "  Node %d: computed=%10.4f  book=%10.4f  %s\n",
        i,
        nodes[i],
        book_nodes[i],
        n_ok ? "PASS" : "FAIL"
    )
    @printf(
        "  Wt   %d: computed=%10.4f  book=%10.4f  %s\n",
        i,
        weights[i],
        book_weights[i],
        w_ok ? "PASS" : "FAIL"
    )
end
println()

# ---------------------------------------------------------------------------
# Symmetry verification
# ---------------------------------------------------------------------------
# For N(0,1), the quadrature nodes and weights must exhibit exact
# reflection symmetry: nodes[i] = -nodes[N+1-i] and weights[i] = weights[N+1-i].

tol_sym = 1e-12
all_sym_pass = true

println("--- Symmetry Check (zero-mean Gaussian) ---")
for i in 1:div(N, 2)
    j = N + 1 - i
    node_sym_ok = abs(nodes[i] + nodes[j]) < tol_sym
    wt_sym_ok = abs(weights[i] - weights[j]) < tol_sym
    if !node_sym_ok || !wt_sym_ok
        global all_sym_pass = false
    end
    @printf(
        "  Pair (%d,%d): node_sum=%.2e  wt_diff=%.2e  %s\n",
        i,
        j,
        abs(nodes[i] + nodes[j]),
        abs(weights[i] - weights[j]),
        (node_sym_ok && wt_sym_ok) ? "PASS" : "FAIL"
    )
end
println()

# ---------------------------------------------------------------------------
# Moment reconstruction verification
# ---------------------------------------------------------------------------
# With N nodes the quadrature has degree of accuracy 2N-1 = 7.
# Moments k=0..7 should be reconstructed exactly within numerical precision.

tol_moment = 1e-8
all_moments_pass = true

println("--- Moment Reconstruction (degree of accuracy = 2N-1 = 7) ---")
for k in 0:7
    pred = sum(weights .* nodes .^ k)
    exact_val = m[k + 1]
    if abs(exact_val) > tol_moment
        rel_err = abs(pred - exact_val) / abs(exact_val)
    else
        # For zero moments (odd k), use absolute error
        rel_err = abs(pred - exact_val)
    end
    ok = rel_err < tol_moment
    if !ok
        global all_moments_pass = false
    end
    @printf(
        "  m_%d: exact=%12.1f  quad=%12.6f  err=%.2e  %s\n",
        k,
        exact_val,
        pred,
        rel_err,
        ok ? "PASS" : "FAIL"
    )
end
println()

# ---------------------------------------------------------------------------
# Weight normalization check
# ---------------------------------------------------------------------------

w_sum = sum(weights)
w_norm_err = abs(w_sum - 1.0)
w_norm_ok = w_norm_err < 1e-10

println("--- Weight Normalization ---")
@printf("  Sum of weights: %.15f\n", w_sum)
@printf("  |sum - 1| = %.2e  %s\n", w_norm_err, w_norm_ok ? "PASS" : "FAIL")
println()

# ---------------------------------------------------------------------------
# Visualization: Wheeler(4) zero-mean NDF with PD failure annotation
# ---------------------------------------------------------------------------

try
    using Plots
    gr()

    ξ_range = range(-4.0, 4.0; length=200)
    ndf_true = [exp(-ξ^2 / 2) / sqrt(2π) for ξ in ξ_range]

    p = plot(
        ξ_range,
        ndf_true;
        label="True NDF N(0,1)",
        lw=2,
        color=:blue,
        xlabel="ξ",
        ylabel="n(ξ)",
        title="Exercise 3.2: Wheeler(4) — Zero Mean",
        legend=:topright,
    )
    for i in eachindex(nodes)
        plot!(
            [nodes[i], nodes[i]],
            [0, weights[i] * 3];
            color=:red,
            lw=2,
            label=(i==1 ? "Weights (×3)" : false),
        )
        scatter!([nodes[i]], [weights[i] * 3]; color=:red, ms=6, label=false)
    end
    annotate!(-3.2, 0.35, text("PD fails (m₁=0)", 10, :left, :darkred))
    mkpath(joinpath(@__DIR__, "..", "output"))
    savefig(p, joinpath(@__DIR__, "..", "output", "ch03_02_wheeler_zero_mean.png"))
    println("\n  Plot saved to output/ch03_02_wheeler_zero_mean.png")
catch
    println("\n  (Install Plots.jl to generate plots)")
end

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

all_pass = pd_failed && all_book_match && all_sym_pass && all_moments_pass && w_norm_ok

println("========================================")
if all_pass
    println("  ALL CHECKS PASSED")
else
    println("  SOME CHECKS FAILED")
    if !pd_failed
        println("  - PD failure detection: FAILED (PD should have errored)")
    end
    if !all_book_match
        println("  - Book value comparison: FAILED")
    end
    if !all_sym_pass
        println("  - Symmetry check: FAILED")
    end
    if !all_moments_pass
        println("  - Moment reconstruction: FAILED")
    end
    if !w_norm_ok
        println("  - Weight normalization: FAILED")
    end
end
println("========================================")
