# examples/06_composite_source_terms.jl
using QBMM
using StaticArrays

println("=== Module 3: Physical Source Terms and Dynamic Evolution ===")
println("6. DQMOM with Modular Physical Source Terms\n")

N = 3
weights = @SVector [0.3, 0.4, 0.3]
nodes = @SVector [1.0, 2.0, 3.0]

println("Initial state:")
println("Weights: ", weights)
println("Nodes: ", nodes)
println()

# Define and compose physical source terms
growth = ParticleGrowth(x -> 0.1)                # Constant growth rate 0.1
aggregation = Aggregation((x, y) -> 0.05) # Constant kernel aggregation 0.05
physics = growth + aggregation

println("Configuring physical model: ParticleGrowth(0.1) + Aggregation(0.05)\n")

# Compute source terms S_k for the first 2N moments
S_k = compute_source_terms(physics, nodes, weights, Val(2N))

println("Combined moment source terms S_k:")
println(S_k)
println()

# DQMOM transient solve
method = DQMOM(N)
da, db = dqmom_solve(method, nodes, S_k)

println("--- DQMOM instantaneous evolution rate ---")
println("Weight rate of change da/dt: ", da)
println("Weighted-node rate of change db/dt (b = w * xi): ", db)
