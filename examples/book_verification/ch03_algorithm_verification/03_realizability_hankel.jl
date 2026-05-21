#!/usr/bin/env julia
# examples/book_verification/ch03_algorithm_verification/03_realizability_hankel.jl
#
# Book Reference:
#   Marchisio, D. L. and Fox, R. O. (2013).
#   Computational Models for Polydisperse Particulate and Multiphase Systems.
#   Cambridge University Press.
#
# Exercise 3.3: Hankel-Hadamard realizability check and difference table of ln(m_k).
#
# This script demonstrates the Hankel-Hadamard realizability criterion for the
# Stieltjes moment problem. A moment sequence m_0, ..., m_{2N} is realizable
# (i.e., corresponds to a non-negative density on [0, inf)) if and only if
# the Hankel determinants D_{0,N} and D_{1,N} are both positive.
#
# Additionally, for a log-normal distribution the second-order differences of
# ln(m_k) are all positive, which is a necessary (but not sufficient) condition.

using QuadratureMoments
using StaticArrays
using LinearAlgebra
using Printf

# ============================================================
# Helper: build an (n+1)x(n+1) Hankel matrix manually
# H[i,j] = m[i+j+offset] with 0-based moment indexing
# ============================================================
function build_hankel(m, n, offset=0)
    return [m[i + j + 1 + offset] for i in 0:n, j in 0:n]
end

# ============================================================
# Helper: print the difference table of ln(m_k)
# ============================================================
function print_difference_table(m, label)
    L = length(m)
    ln_m = [log(m[k]) for k in 1:L]

    @printf("\n--- Difference table of ln(m_k) for %s ---\n", label)
    @printf("  k    ln(m_k)       d1           d2\n")
    @printf("  ---  ----------    ----------    ----------\n")

    # First-order differences
    d1 = [ln_m[k + 1] - ln_m[k] for k in 1:(L - 1)]

    # Second-order differences
    d2 = [d1[k + 1] - d1[k] for k in 1:(L - 2)]

    for k in 1:L
        if k <= L
            @printf("  %2d   %10.6f", k - 1, ln_m[k])
        end
        if k <= L - 1
            @printf("    %10.6f", d1[k])
        else
            @printf("               ")
        end
        if k <= L - 2
            @printf("    %10.6f", d2[k])
        end
        println()
    end

    all_pos = all(d2 .> 0)
    @printf("  All second-order differences positive? %s\n", all_pos ? "YES" : "NO")
    return all_pos
end

# ============================================================
# Helper: evaluate Hankel determinants manually
# ============================================================
function print_hankel_determinants(m, label)
    # 4x4 Hankel matrices for 8 moments (N=3)
    D03 = build_hankel(m, 3, 0)  # Delta_{0,3}
    D13 = build_hankel(m, 3, 1)  # Delta_{1,3}

    det_D03 = det(D03)
    det_D13 = det(D13)

    @printf("\n--- Hankel determinants for %s ---\n", label)
    @printf("  Delta_{0,3} = det(H_0) = %.6f\n", det_D03)
    @printf("  Delta_{1,3} = det(H_1) = %.6f\n", det_D13)
    @printf("  Delta_{0,3} > 0 ? %s\n", det_D03 > 0 ? "YES" : "NO")
    @printf("  Delta_{1,3} > 0 ? %s\n", det_D13 > 0 ? "YES" : "NO")

    return det_D03, det_D13
end

# ============================================================
# Main
# ============================================================

println("=" ^ 68)
println("  Exercise 3.3: Hankel-Hadamard Realizability Check")
println("  Book: Marchisio & Fox (2013)")
println("=" ^ 68)

all_pass = true

# ============================================================
# Test Case 1: Realizable -- N(5, 1) moments
# ============================================================
println()
println("=" ^ 68)
println("  Test Case 1: Realizable moment sequence (Normal N(5,1))")
println("=" ^ 68)

m_realizable = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]
m_real = collect(m_realizable)

@printf("\nMoments: %s\n", m_real)

# Print difference table
d2_ok = print_difference_table(m_real, "realizable set")

# Print Hankel determinants
det_D03, det_D13 = print_hankel_determinants(m_real, "realizable set")

# Expected values from the book
@printf("\nExpected: Delta_{0,3} = 12,  Delta_{1,3} = 5736\n")

# Verify against known values
tol = 1.0
D03_ok = abs(det_D03 - 12.0) < tol
D13_ok = abs(det_D13 - 5736.0) < tol

@printf("  Delta_{0,3} == 12   ? %s (diff = %.6f)\n", D03_ok ? "YES" : "NO", abs(det_D03 - 12.0))
@printf("  Delta_{1,3} == 5736 ? %s (diff = %.6f)\n", D13_ok ? "YES" : "NO", abs(det_D13 - 5736.0))

# Use the library function
result1 = is_realizable(m_realizable)
@printf("\n  is_realizable(SVector{8}(m)) = %s\n", result1)
@printf("  Expected: true\n")

test1_pass = result1 == true && D03_ok && D13_ok && d2_ok
@printf("  Test Case 1: %s\n", test1_pass ? "PASS" : "FAIL")
if !test1_pass
    all_pass = false
end

# ============================================================
# Test Case 2: Unrealizable -- m_2 corrupted from 26 to 25
# ============================================================
println()
println("=" ^ 68)
println("  Test Case 2: Unrealizable moment sequence (m_2 corrupted: 26 -> 25)")
println("=" ^ 68)

m_unrealizable = @SVector [1.0, 5.0, 25.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]
m_unreal = collect(m_unrealizable)

@printf("\nMoments: %s\n", m_unreal)
@printf("(m_2 changed from 26 to 25)\n")

# Print difference table
d2_ok2 = print_difference_table(m_unreal, "unrealizable set")

# Print Hankel determinants
det_D03_u, det_D13_u = print_hankel_determinants(m_unreal, "unrealizable set")

# Verify determinants are negative
D03_neg = det_D03_u < 0
D13_neg = det_D13_u < 0

@printf("\n  Delta_{0,3} < 0 ? %s\n", D03_neg ? "YES" : "NO")
@printf("  Delta_{1,3} < 0 ? %s\n", D13_neg ? "YES" : "NO")

# Use the library function
result2 = is_realizable(m_unrealizable)
@printf("\n  is_realizable(SVector{8}(m)) = %s\n", result2)
@printf("  Expected: false\n")

test2_pass = result2 == false && D03_neg && D13_neg
@printf("  Test Case 2: %s\n", test2_pass ? "PASS" : "FAIL")
if !test2_pass
    all_pass = false
end

# --- Visualization ---
try
    using Plots
    gr()

    # Collect Hankel determinants for both moment sets
    hankel_dets_real = [det_D03, det_D13]
    hankel_dets_unreal = [det_D03_u, det_D13_u]

    # Compute 2nd differences of ln(m_k) for both sets
    function second_diffs(m)
        ln_m = [log(x) for x in m]
        d1 = [ln_m[k+1] - ln_m[k] for k in 1:(length(m)-1)]
        d2 = [d1[k+1] - d1[k] for k in 1:(length(d1)-1)]
        return d2
    end
    d2_real = second_diffs(m_real)
    d2_unreal = second_diffs(m_unreal)

    # Left panel: Hankel determinants
    k_hankel = collect(0:length(hankel_dets_real)-1)
    p1 = bar(k_hankel .- 0.15, hankel_dets_real, bar_width=0.3, label="Realizable",
             color=:blue, xlabel="Determinant index", ylabel="Value",
             title="Hankel Determinants")
    bar!(k_hankel .+ 0.15, hankel_dets_unreal, bar_width=0.3, label="Corrupted", color=:red)
    hline!([0], color=:gray, ls=:dash, label=false)

    # Right panel: 2nd differences of ln(m_k)
    k_diff = collect(0:length(d2_real)-1)
    p2 = bar(k_diff .- 0.15, d2_real, bar_width=0.3, label="Realizable (d2>0)",
             color=:blue, xlabel="Order k", ylabel="d2[ln(m_k)]",
             title="ln(m_k) 2nd Differences")
    bar!(k_diff .+ 0.15, d2_unreal, bar_width=0.3, label="Corrupted", color=:red)
    hline!([0], color=:gray, ls=:dash, label=false)

    p = plot(p1, p2, layout=(1,2), size=(900,400))
    output_dir = joinpath(@__DIR__, "..", "output")
    mkpath(output_dir)
    savefig(p, joinpath(output_dir, "ch03_03_realizability.png"))
    println("\n  Plot saved to output/ch03_03_realizability.png")
catch e
    @show e
    println("\n  (Install Plots.jl to generate plots)")
end

# ============================================================
# Summary
# ============================================================
println()
println("=" ^ 68)
if all_pass
    println("  ALL TESTS PASSED")
else
    println("  SOME TESTS FAILED")
end
println("=" ^ 68)
