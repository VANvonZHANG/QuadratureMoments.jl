# examples/02_eqmom.jl
using QuadratureMoments
using StaticArrays
using Printf

println("=== Module 1: Inversion Algorithms for Static Distributions ===")
println("2. Continuous Distribution Reconstruction (EQMOM)\n")

# Moment sequence composed of Gaussian kernels with nonzero width (mixture of normals)
m_extended = @SVector [1.0, 5.2, 29.0, 170.8, 1051.0]

println("Input moment sequence: ", m_extended)
println()

# Using N=2 Gaussian-kernel EQMOM (since 5 moments are given)
method_eq = EQMOM(2, GaussianKernel())
res_eq = invert_moments(method_eq, m_extended)

println("--- EQMOM (Gaussian Kernel, N=2) results ---")
for i in 1:2
    @printf("Node (mean) %d: %8.4f, Weight: %8.4f\n", i, res_eq.nodes[i], res_eq.weights[i])
end
@printf("Global bandwidth parameter sigma: %8.4f\n", res_eq.sigmas[1])
println()
println(
    "Compared with standard QMOM (where all probability mass is concentrated at discrete nodes), EQMOM provides an approximation of a smooth distribution with nonzero sigma.",
)
