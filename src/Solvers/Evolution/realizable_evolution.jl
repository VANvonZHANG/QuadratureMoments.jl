# src/Solvers/Evolution/realizable_evolution.jl
using StaticArrays
using LinearAlgebra

# NOTE: invert_moments, is_realizable, and mcgraw_correction are defined in later
# sections of QuadratureMoments.jl (Section 7 Tools / Section 9 API), after this
# file is included. An explicit `using ..QuadratureMoments:` import would fail at
# precompile (UndefVarError) because those names don't exist yet at this include
# site. Bare references are intentional: Julia resolves them at call time, once the
# full module is loaded.

"""
    evolve_moments(m0, source, tspan; dt0, N, tol=1e-12, max_substeps=8, domain=:pos)

Evolve the moment vector `m0` over `tspan=(t0,tf)` under `dm/dt = S(m)`, where the
moment source `S` is obtained by reinverting `m` to an `N`-node Wheeler quadrature
and calling `compute_source_terms(source, nodes, weights, Val(L))`.

Realizability-aware: after each candidate Euler step, if the updated moments are
not realizable on `domain`, the step is rejected and `dt` halved (up to
`max_substeps`); if still unrealizable, `mcgraw_correction` is applied as a
projection. Returns the final moment vector (same length/type as `m0`).
"""
function evolve_moments(
    m0::SVector{L,T},
    source::AbstractSourceTerm,
    tspan::Tuple{<:Real,<:Real};
    dt0::Real,
    N::Int,
    tol::Real=1e-12,
    max_substeps::Int=8,
    domain::Symbol=:pos,
) where {L,T}
    t0, tf = tspan
    m = m0
    t = T(t0)
    dt = T(dt0)
    tf = T(tf)
    while t < tf
        dt = min(dt, tf - t)
        S = _source_for_moments(m, source, N)
        accepted = false
        for _ in 1:max_substeps
            m_trial = m + dt .* S
            if is_realizable(m_trial; domain=domain)
                m = m_trial
                t += dt
                accepted = true
                break
            end
            dt = dt / 2
            (dt < tol) && break
        end
        if !accepted
            # project back to the realizable cone and advance with a tiny step
            m = mcgraw_correction(m + dt .* S; max_iter=20)
            # mcgraw preserves m0 only approximately; clamp t forward to avoid infinite loop
            t += dt > tol ? dt : tol
            dt = T(dt0)   # reset for the next macro step
        end
    end
    return m
end

# Reinvert m to an N-node Wheeler quadrature, compute the moment source S_k, k=0..L-1.
@inline function _source_for_moments(m::SVector{L,T}, source, N::Int) where {L,T}
    res = invert_moments(Wheeler{N}(), m)
    # res.weights / res.nodes are length N (possibly zero-padded); source terms skip zeros.
    return compute_source_terms(source, SVector{N,T}(res.nodes), res.weights, Val(L))
end
