# examples/source_terms_physics.jl

using QBMM
using StaticArrays
using Printf

println("=== QBMM.jl Physical Source Terms & DQMOM Example ===")
println("Replicating textbook examples for Pure Aggregation and Particle Growth.
")

# 1. Provide an arbitrary set of current nodes and weights
# In a real CFD simulation, these are obtained from the DQMOM transport equations
nodes = @SVector [1.0, 2.0, 3.0]
weights = @SVector [0.5, 0.3, 0.2]
m0 = sum(weights) # Current total number density = 1.0

println("Current State:")
println("  Nodes:   ", nodes)
println("  Weights: ", weights)
println(
    "  Total Number Density (m0): ",
    m0,
    "
",
)

# ==========================================
# Case 1: Constant Aggregation Kernel
# ==========================================
println("--- Case 1: Constant Aggregation ---")
beta_0 = 1.2
const_agg = Aggregation((xi, xj) -> beta_0)

# Compute source terms for the first 4 moments (k=0, 1, 2, 3)
S_k_agg = compute_source_terms(const_agg, nodes, weights, Val(4))

println("Computed Source Terms S_k: ", S_k_agg)
println("Theoretical S_0: ", -0.5 * beta_0 * (m0^2))
println("Theoretical S_1: 0.0 (Mass Conservation)")
@assert isapprox(S_k_agg[1], -0.5 * beta_0 * (m0^2), atol=1e-10)
@assert isapprox(S_k_agg[2], 0.0, atol=1e-10)
println("  => Match verified!
")

# ==========================================
# Case 2: Constant Rate Particle Growth
# ==========================================
println("--- Case 2: Constant Rate Growth ---")
G_0 = 0.5
const_growth = ParticleGrowth(xi -> G_0)

S_k_growth = compute_source_terms(const_growth, nodes, weights, Val(4))

println("Computed Source Terms S_k: ", S_k_growth)
println("Theoretical S_0: 0.0 (Number Conservation)")
println("Theoretical S_1: ", G_0 * m0)
@assert isapprox(S_k_growth[1], 0.0, atol=1e-10)
@assert isapprox(S_k_growth[2], G_0 * m0, atol=1e-10)
println("  => Match verified!
")

# ==========================================
# Case 3: Symmetric Binary Breakage
# ==========================================
println("--- Case 3: Symmetric Binary Breakage ---")
b_0 = 1.5
sym_break = Breakage(xi -> b_0, (k, xi) -> (2.0^(1.0 - k)) * (xi^k))

S_k_break = compute_source_terms(sym_break, nodes, weights, Val(4))

println("Computed Source Terms S_k: ", S_k_break)
println("Theoretical S_0: ", b_0 * m0)
println("Theoretical S_1: 0.0 (Mass Conservation)")
@assert isapprox(S_k_break[1], b_0 * m0, atol=1e-10)
@assert isapprox(S_k_break[2], 0.0, atol=1e-10)
println("  => Match verified!\n")

# ==========================================
# Case 4: Superposition & DQMOM Evolution
# ==========================================
println("--- Case 4: Superposition & DQMOM Evolution ---")
# Combine physics with the overloaded + operator
total_physics = const_growth + const_agg + sym_break

# DQMOM with N=3 requires 2N = 6 source terms
S_total = compute_source_terms(total_physics, nodes, weights, Val(6))

# Solve the DQMOM system to get the evolution rates of weights (da) and weighted abscissas (db)
da, db = dqmom_solve(DQMOM(3), nodes, S_total)

println("Total combined S_k (first 6 moments):")
for k in 0:5
    @printf(
        "  S_%d = %10.5f
",
        k,
        S_total[k + 1]
    )
end
println("
DQMOM Evolution Rates:")
println("  da/dt (Weight rates): ", da)
println("  db/dt (Weighted node rates): ", db)
println("
=== Example Completed Successfully ===")
