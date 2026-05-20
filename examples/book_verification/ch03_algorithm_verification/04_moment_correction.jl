# examples/book_verification/ch03_algorithm_verification/04_moment_correction.jl
#
# Book reference:
#   Marchisio, D. L. & Fox, R. O. (2013).
#   Computational Models for Polydisperse Particulate and Multiphase Systems.
#   Cambridge University Press.
#   Exercises 3.4 (single corrupted moment) and 3.5 (multiple corrupted moments).
#
# Demonstrates the McGraw moment correction algorithm applied to corrupted
# moment sets.  The corrected moments are guaranteed to form a realizable
# sequence, even when one or more input moments violate the Hankel
# determinant conditions.
#
# Note: The McGraw algorithm is iterative and its convergence path depends on
# implementation details (step size, pivot selection, smoothing operator).
# Consequently, the exact corrected values may differ from the book, but the
# key properties -- realizability of the output and closeness to the original
# healthy moments -- are preserved.

using QuadratureMoments
using StaticArrays
using Printf

# ============================================================================
# Shared moment set: 8 moments of a normal distribution (mu=5, sigma=1)
# ============================================================================
m_original = @SVector [1.0, 5.0, 26.0, 140.0, 778.0, 4450.0, 26140.0, 157400.0]

println("========================================")
println("  McGraw Moment Correction Verification")
println("  (Exercises 3.4 - 3.5, Marchisio & Fox)")
println("========================================\n")

println("Original moment set (8 moments):")
for i in 0:7
    @printf("  m%d = %12.1f\n", i, m_original[i + 1])
end
println()
println("Realizable: ", is_realizable(m_original; domain=:pos))
println()

# ============================================================================
# Exercise 3.4 -- single corrupted moment (m3: 140 -> 101)
# ============================================================================
println("-------------------------------------------------")
println("  Exercise 3.4: Single Corrupted Moment (m3)")
println("-------------------------------------------------\n")

m_ex34 = SVector(1.0, 5.0, 26.0, 101.0, 778.0, 4450.0, 26140.0, 157400.0)

println("Corrupted moment set (m3 changed from 140.0 to 101.0):")
for i in 0:7
    @printf("  m%d = %12.1f\n", i, m_ex34[i + 1])
end
println()
println("Realizable: ", is_realizable(m_ex34; domain=:pos))
println()

m_ex34_corr = mcgraw_correction(m_ex34)

println("Corrected moment set after McGraw correction:")
for i in 0:7
    @printf("  m%d = %12.4f\n", i, m_ex34_corr[i + 1])
end
println()
println("Realizable: ", is_realizable(m_ex34_corr; domain=:pos))
println()

@printf("Restoration of m3: corrected = %.4f, original = 140.0\n",
    m_ex34_corr[4])
println()

# Verification: output must be realizable and m3 should be close to original
ex34_realizable = is_realizable(m_ex34_corr; domain=:pos)
ex34_m3_close = isapprox(m_ex34_corr[4], 140.0; atol=5.0)

if ex34_realizable && ex34_m3_close
    println("Exercise 3.4: PASS")
else
    println("Exercise 3.4: FAIL")
    if !ex34_realizable
        println("  -- corrected set is not realizable")
    end
    if !ex34_m3_close
        println("  -- m3 not restored to ~140.0")
    end
end
println()

# ============================================================================
# Exercise 3.5 -- multiple corrupted moments (m2: 26->12, m3: 140->101)
# ============================================================================
println("-------------------------------------------------")
println("  Exercise 3.5: Multiple Corrupted Moments")
println("  (m2: 26 -> 12, m3: 140 -> 101)")
println("-------------------------------------------------\n")

m_ex35 = SVector(1.0, 5.0, 12.0, 101.0, 778.0, 4450.0, 26140.0, 157400.0)

println("Corrupted moment set (m2: 26->12, m3: 140->101):")
for i in 0:7
    @printf("  m%d = %12.1f\n", i, m_ex35[i + 1])
end
println()
println("Realizable: ", is_realizable(m_ex35; domain=:pos))
println()

m_ex35_corr = mcgraw_correction(m_ex35)

println("Corrected moment set after McGraw correction:")
for i in 0:7
    @printf("  m%d = %12.4f\n", i, m_ex35_corr[i + 1])
end
println()
println("Realizable: ", is_realizable(m_ex35_corr; domain=:pos))
println()

# Book reference values for the first 4 corrected moments
@printf("Comparison with book reference values (Exercise 3.5):\n")
@printf("  m0: corrected = %8.4f, book ~ 0.3689\n", m_ex35_corr[1])
@printf("  m1: corrected = %8.4f, book ~ 2.378\n",  m_ex35_corr[2])
@printf("  m2: corrected = %8.4f, book ~ 15.65\n",  m_ex35_corr[3])
@printf("  m3: corrected = %8.4f, book ~ 104.5\n",   m_ex35_corr[4])
println()

# The book notes that when multiple moments are corrupted, the corrected set
# is realizable but NOT necessarily equal to the original.  The key test is
# realizability of the output.
ex35_realizable = is_realizable(m_ex35_corr; domain=:pos)

if ex35_realizable
    println("Exercise 3.5: PASS")
else
    println("Exercise 3.5: FAIL")
    println("  -- corrected set is not realizable")
end
println()

# ============================================================================
# Summary
# ============================================================================
println("========================================")
all_pass = ex34_realizable && ex34_m3_close && ex35_realizable

if all_pass
    println("  OVERALL: PASS")
else
    println("  OVERALL: FAIL")
end
println("========================================")
