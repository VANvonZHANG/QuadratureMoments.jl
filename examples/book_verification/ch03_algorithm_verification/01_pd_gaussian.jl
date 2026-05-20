# 01_pd_gaussian.jl
#
# Book reference: Exercise 3.1 from
#   Marchisio & Fox (2013), "Computational Models for Polydisperse
#   Particulate and Multiphase Systems", Cambridge University Press.
#
# Demonstrates the Product-Difference (PD) algorithm on a Normal(5,1)
# distribution using N=4 quadrature nodes (requires 8 raw moments).
#
# Expected results (Table 3.1, Eq. 3.18):
#   Nodes   ~ {2.6656, 4.2580, 5.7420, 7.3344}
#   Weights ~ {0.0459, 0.4541, 0.4541, 0.0459}

using QuadratureMoments
using StaticArrays
using Printf

function main()
    # -----------------------------------------------------------------------
    # Problem setup
    # -----------------------------------------------------------------------

    N = 4                     # number of quadrature nodes

    # Raw moments of N(mu=5, sigma=1):
    #   m_0 = 1
    #   m_k = mu * m_{k-1} + (k-1) * sigma^2 * m_{k-2}
    m = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]

    println("=== PD Algorithm Verification: Gaussian N(5,1) ===")
    println("Reference: Marchisio & Fox (2013), Exercise 3.1, Eq. 3.18")
    println("Number of nodes N = ", N)
    println("Input moments (k=0..7): ", m)
    println()

    # -----------------------------------------------------------------------
    # Run the PD algorithm
    # -----------------------------------------------------------------------

    method = PD(N)
    res = invert_moments(method, m)

    nodes = vec(res.nodes)
    weights = res.weights

    println("--- PD Algorithm Results ---")
    for i in 1:N
        @printf("  Node %d: %10.4f    Weight: %10.4f\n", i, nodes[i], weights[i])
    end
    println()

    # -----------------------------------------------------------------------
    # Compare with book values
    # -----------------------------------------------------------------------

    book_nodes  = [2.6656, 4.2580, 5.7420, 7.3344]
    book_weights = [0.0459, 0.4541, 0.4541, 0.0459]

    println("--- Comparison with Book Values (Eq. 3.18) ---")
    tol_book = 1e-2  # book prints 4 decimal places

    all_book_match = true
    for i in 1:N
        n_ok = abs(nodes[i] - book_nodes[i]) < tol_book
        w_ok = abs(weights[i] - book_weights[i]) < tol_book
        if !n_ok || !w_ok
            all_book_match = false
        end
        @printf(
            "  Node %d: computed=%10.4f  book=%10.4f  %s\n",
            i, nodes[i], book_nodes[i], n_ok ? "PASS" : "FAIL"
        )
        @printf(
            "  Wt   %d: computed=%10.4f  book=%10.4f  %s\n",
            i, weights[i], book_weights[i], w_ok ? "PASS" : "FAIL"
        )
    end
    println()

    # -----------------------------------------------------------------------
    # Moment reconstruction verification
    # -----------------------------------------------------------------------
    # With N nodes the quadrature has degree of accuracy 2N-1 = 7.
    # Moments k=0..7 should be reconstructed exactly; m_8 should have error.

    m8_exact = 991760.0  # exact 8th raw moment of N(5,1)

    tol_moment = 1e-10
    all_moments_pass = true

    println("--- Moment Reconstruction (degree of accuracy = 2N-1 = 7) ---")
    for k in 0:7
        pred = sum(weights .* nodes .^ k)
        rel_err = abs(pred - m[k+1]) / abs(m[k+1])
        ok = rel_err < tol_moment
        if !ok
            all_moments_pass = false
        end
        @printf("  m_%d: exact=%12.1f  quad=%12.1f  rel_err=%.2e  %s\n",
            k, m[k+1], pred, rel_err, ok ? "PASS" : "FAIL")
    end

    # m_8: expect non-zero error since degree of accuracy is 7
    k = 8
    m8_pred = sum(weights .* nodes .^ k)
    m8_rel_err = abs(m8_pred - m8_exact) / abs(m8_exact)
    m8_fail = m8_rel_err > tol_moment  # this SHOULD fail (nonzero error is expected)
    @printf("  m_%d: exact=%12.1f  quad=%12.1f  rel_err=%.2e  %s (expected nonzero)\n",
        k, m8_exact, m8_pred, m8_rel_err,
        m8_fail ? "PASS" : "FAIL")
    println()

    # -----------------------------------------------------------------------
    # Weight normalization check
    # -----------------------------------------------------------------------

    w_sum = sum(weights)
    w_norm_err = abs(w_sum - 1.0)
    w_norm_ok = w_norm_err < 1e-10

    println("--- Weight Normalization ---")
    @printf("  Sum of weights: %.15f\n", w_sum)
    @printf("  |sum - 1| = %.2e  %s\n", w_norm_err, w_norm_ok ? "PASS" : "FAIL")
    println()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    all_pass = all_book_match && all_moments_pass && w_norm_ok

    println("========================================")
    if all_pass
        println("  ALL CHECKS PASSED")
    else
        println("  SOME CHECKS FAILED")
        if !all_book_match
            println("  - Book value comparison: FAILED")
        end
        if !all_moments_pass
            println("  - Moment reconstruction: FAILED")
        end
        if !w_norm_ok
            println("  - Weight normalization: FAILED")
        end
    end
    println("========================================")
end

main()
