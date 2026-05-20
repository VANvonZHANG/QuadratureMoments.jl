# examples/04_realizability_correction.jl
using QuadratureMoments
using StaticArrays

println("=== Module 2: Robustness and Error Handling ===")
println("4. Moment Realizability and McGraw Correction\n")

m_good = @SVector [1.0, 5.0, 26.0, 140.0]
println("Original healthy sequence m: ", m_good)
println("Realizability check: ", is_realizable(m_good; domain=:pos))
println()

# Corrupted data
m_bad = @SVector [1.0, 5.0, 26.0, 101.0]
println("Corrupted sequence m_bad (m3 changed to 101.0): ", m_bad)
println("Realizability check: ", is_realizable(m_bad; domain=:pos))
println()

# Perform McGraw correction
m_corr = mcgraw_correction(m_bad)
println("McGraw-corrected sequence: ", m_corr)
println("Realizability check: ", is_realizable(m_corr; domain=:pos))
