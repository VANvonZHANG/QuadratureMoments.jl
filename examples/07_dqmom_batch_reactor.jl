# examples/07_dqmom_batch_reactor.jl
using QBMM
using StaticArrays
using Printf

println("=== Module 3: Physical Source Terms and Dynamic Evolution ===")
println("7. Batch Reactor Time Evolution Simulation (DQMOM) - RK4 Integration\n")

# To stay lightweight (no external dependencies), we implement RK4 integration directly
function dqmom_rhs(u, method, physics)
    N = div(length(u), 2)
    weights = SVector{N}(u[1:N])
    weighted_nodes = SVector{N}(u[(N + 1):2N])

    # Recover nodes, with safety guard against division by zero
    nodes = SVector{N}(weighted_nodes ./ max.(weights, 1e-12))

    # Compute physical source terms and evolution rate
    S_k = compute_source_terms(physics, nodes, weights, Val(2N))
    da, db = dqmom_solve(method, nodes, S_k)

    return vcat(da, db)
end

# Initialize the problem
N = 3
w0 = @SVector [0.2, 0.6, 0.2]
x0 = @SVector [1.0, 2.0, 3.0]
b0 = SVector{N}(w0 .* x0)
u = vcat(w0, b0)

# Physical model: pure growth G(x) = 0.5
physics = ParticleGrowth(x -> 0.5)
method = DQMOM(N)

dt = 0.1
t_end = 2.0
steps = Int(t_end / dt)

println("Initial nodes: ", x0)
println("Pure growth model G=0.5; theoretically all nodes should increase by 1.0 at t=2.0\n")

for i in 1:steps
    # RK4 step
    k1 = dqmom_rhs(u, method, physics)
    k2 = dqmom_rhs(u .+ 0.5 * dt * k1, method, physics)
    k3 = dqmom_rhs(u .+ 0.5 * dt * k2, method, physics)
    k4 = dqmom_rhs(u .+ dt * k3, method, physics)
    global u = u .+ (dt / 6.0) * (k1 .+ 2k2 .+ 2k3 .+ k4)
end

w_final = u[1:N]
x_final = u[(N + 1):2N] ./ max.(w_final, 1e-12)

println("Results after integrating to t=2.0:")
println("Weights: ", w_final)
println("Nodes: ", x_final)
println("As can be seen, the weights remain unchanged and each node shifted right by approximately 1.0. This perfectly matches the analytical solution!")
