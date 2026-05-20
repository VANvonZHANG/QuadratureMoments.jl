# examples/08_dqmom_vs_smm.jl
using QuadratureMoments
using StaticArrays

println("=== Module 3: Physical Source Terms and Dynamic Evolution ===")
println("8. Evolution Stability Showdown: DQMOM vs Standard QMOM (SMM)\n")

function smm_rhs(m, method, physics)
    N = div(length(m), 2)
    # First invert for nodes and weights
    res = invert_moments(method, m)
    # If the returned nodes are not a 1D vector, attempt conversion (Wheeler should return a vector)
    nodes = SVector{N}(vec(res.nodes))
    weights = SVector{N}(res.weights)
    # Compute moment source terms
    return compute_source_terms(physics, nodes, weights, Val(2N))
end

N = 3
w0 = @SVector [0.2, 0.6, 0.2]
x0 = @SVector [1.0, 2.0, 3.0]
m0 = SVector{2N}(sum(w0[i] * x0[i]^k for i in 1:N) for k in 0:(2N - 1))

# Intense physical process (constant kernel aggregation)
physics = Aggregation((x, y) -> 1.0)
method = Wheeler(N)

println("Starting SMM (Standard Method of Moments) time integration...")
println("SMM evolves the moment vector, then uses the Wheeler algorithm to invert at each time step.")
println("This is very likely to crash at some step due to numerical truncation errors from rapid growth of high-order moments, making the data non-realizable.\n")

dt = 0.05
m_current = m0
success_smm = true
for step in 1:20
    try
        dm = smm_rhs(m_current, method, physics)
        global m_current = m_current .+ dt .* dm
    catch e
        println("=> SMM crashed at step $step!\nError message: $e")
        println("=> Cause: m can no longer satisfy the Stieltjes condition; realizability is lost.")
        global success_smm = false
        break
    end
end
if success_smm
    println("=> SMM narrowly completed the full run.")
end

println("\n---------------------------------------------------------")
println("For comparison, with the same physics and step size, DQMOM directly evolves nodes and weights:\n")
u_current = vcat(w0, SVector{N}(w0 .* x0))
dqmom_method = DQMOM(N)

success_dqmom = true
for step in 1:20
    try
        w = SVector{N}(u_current[1:N])
        b = SVector{N}(u_current[(N + 1):2N])
        nodes = SVector{N}(b ./ max.(w, 1e-12))
        S_k = compute_source_terms(physics, nodes, w, Val(2N))
        da, db = dqmom_solve(dqmom_method, nodes, S_k)
        global u_current = u_current .+ dt .* vcat(da, db)
    catch e
        println("=> DQMOM crashed: ", e)
        global success_dqmom = false
        break
    end
end

if success_dqmom
    println("=> DQMOM completed all 20 steps with extreme stability!")
    println("=> Conclusion: Because DQMOM does not involve repeated nonlinear inversion from high-order moments to nodes, its robustness during time integration far exceeds that of SMM.")
end
