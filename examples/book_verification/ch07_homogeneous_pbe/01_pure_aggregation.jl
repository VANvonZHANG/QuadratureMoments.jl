# 01_pure_aggregation.jl
#
# Book reference: Section 7.4.1 from
#   Marchisio & Fox (2013), "Computational Models for Polydisperse
#   Particulate and Multiphase Systems", Cambridge University Press.
#
# Solves a spatially homogeneous population balance equation (PBE)
# with pure aggregation using QMOM + ODE time integration, comparing
# against the analytical (self-similar) solution.
#
# Physics:
#   - Constant aggregation kernel beta(V,V') = beta_0 = 1.0
#   - Initial condition: monodisperse distribution at V = 1.0 (delta function)
#   - Moments: m_k(0) = 1.0^k = 1.0 for all k
#
# Analytical solution:
#   m_0(t) = m_0(0) / (1 + 0.5*beta_0*m_0(0)*t)    -- number density decreases
#   m_1(t) = m_1(0) = 1.0                             -- mass conserved
#   m_2(t) = m_2(0) * (1 + 0.5*beta_0*m_0(0)*t)^2   -- second moment grows

# -----------------------------------------------------------------------
# Dependency check
# -----------------------------------------------------------------------

try
    using OrdinaryDiffEq
catch
    println("ERROR: OrdinaryDiffEq.jl is required but not installed.")
    println("Run: using Pkg; Pkg.add(\"OrdinaryDiffEq\")")
    exit(1)
end

using QuadratureMoments
using QuadratureMoments.Analysis
using StaticArrays
using LinearAlgebra
using Printf

# Optional dependency for plotting
const _PLOTS_AVAILABLE = try
    using Plots
    true
catch
    false
end

include(joinpath(@__DIR__, "..", "utils", "book_reporting.jl"))

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------

const beta_0    = 1.0      # constant aggregation kernel coefficient
const N_quad    = 4        # number of quadrature nodes
const n_moments = 6        # moments tracked: m_0 through m_5
const t_final   = 2.0      # end time

# -----------------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------------

function analytical_moments(t::Float64)
    tau = 1.0 + 0.5 * beta_0 * t   # m_0(0) = 1.0
    m0 = 1.0 / tau
    m1 = 1.0
    m2 = 1.0 * tau^2
    return (m0, m1, m2)
end

# -----------------------------------------------------------------------
# Aggregation source term via quadrature
# -----------------------------------------------------------------------
# For moment k, the aggregation source is:
#   S_k = 0.5 * sum_i sum_j w_i*w_j * beta_0 * [(xi_i+xi_j)^k - xi_i^k - xi_j^k]
# This is the standard form: birth - death.

function aggregation_source(m::MVector{n_moments, Float64}) where {}
    # Clamp moments to prevent negative values from ODE solver drift
    m_safe = MVector{n_moments, Float64}(max(mi, 1e-30) for mi in m)

    # Invert moments to obtain quadrature
    res = invert_moments(Wheeler(N_quad), SVector{n_moments, Float64}(m_safe))
    nodes = vec(res.nodes)    # SVector{N_quad, Float64}
    weights = res.weights     # SVector{N_quad, Float64}

    # Compute source terms: birth - death for each moment order k = 0..n_moments-1
    S = MVector{n_moments, Float64}(zeros(n_moments))
    for k in 0:(n_moments - 1)
        val = 0.0
        for i in 1:N_quad
            for j in 1:N_quad
                wi = weights[i]
                wj = weights[j]
                xi = nodes[i]
                xj = nodes[j]
                # birth term: 0.5 * w_i * w_j * beta_0 * (xi + xj)^k
                birth = 0.5 * wi * wj * beta_0 * (xi + xj)^k
                # death term: w_i * w_j * beta_0 * xi^k
                death = wi * wj * beta_0 * xi^k
                val += (birth - death)
            end
        end
        S[k + 1] = val
    end
    return S
end

# -----------------------------------------------------------------------
# ODE right-hand side
# -----------------------------------------------------------------------

function ode_rhs(m::MVector{n_moments, Float64}, p, t)
    return aggregation_source(m)
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

function main()
    println("=== Pure Aggregation PBE: QMOM Verification ===")
    println("Reference: Marchisio & Fox (2013), Section 7.4.1")
    println()
    println("Kernel:    beta(V,V') = ", beta_0, " (constant)")
    println("Quadrature: N_quad = ", N_quad)
    println("Moments:    m_0 .. m_", n_moments - 1, " (n_moments = ", n_moments, ")")
    println("Time span:  [0, ", t_final, "]")
    println()

    # -------------------------------------------------------------------
    # Initial condition: monodisperse at V = 1.0
    # m_k(0) = 1.0^k = 1.0 for all k
    # -------------------------------------------------------------------

    m0_init = MVector{n_moments, Float64}(ones(n_moments))

    println("Initial moments (monodisperse at V=1.0):")
    for k in 0:(n_moments - 1)
        @printf("  m_%d = %.6f\n", k, m0_init[k + 1])
    end
    println()

    # -------------------------------------------------------------------
    # Solve ODE
    # -------------------------------------------------------------------

    tspan = (0.0, t_final)
    prob = ODEProblem(ode_rhs, m0_init, tspan)
    sol = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-12)

    # Extract solution at desired output times
    t_out = [0.0, 0.5, 1.0, 1.5, 2.0]

    # -------------------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------------------

    mc = compare_moments(t_out, t -> sol(t)[1:3], t -> collect(analytical_moments(t)); n_moments = 3)
    print_comparison_table(mc, ["m₀", "m₁", "m₂"])

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------

    if _PLOTS_AVAILABLE
        try
            Plots.gr()

            # Build dense time arrays for plotting
            t_dense = range(tspan[1], tspan[2], length = 100)

            # Build moment matrices
            n_mom_plot = 3
            m_num = Matrix{Float64}(undef, length(t_dense), n_mom_plot)
            m_ref = Matrix{Float64}(undef, length(t_dense), n_mom_plot)
            for (i, t) in enumerate(t_dense)
                m_num[i, :] .= sol(t)[1:n_mom_plot]
                m_ref[i, :] .= collect(analytical_moments(t))[1:n_mom_plot]
            end

            # NDF reconstruction at snapshot times
            ξ_range = range(0.5, 3.5, length = 200)
            snapshot_times = [0.0, 0.5, 1.0, 1.5, 2.0]
            ndfs = Vector{Float64}[]
            for t_snap in snapshot_times
                m_at_t = sol(t_snap)
                m_snap = SVector{n_moments, Float64}(max(mi, 1e-30) for mi in m_at_t)
                n_eq = (n_moments - 1) ÷ 2
                res_eq = invert_moments(EQMOM(n_eq, GaussianKernel()), m_snap)
                ndf = reconstruct_ndf(res_eq, ξ_range, GaussianKernel())
                push!(ndfs, ndf)
            end

            # Plot
            p = plot_pbe_summary(t_dense, m_num, ξ_range, ndfs, snapshot_times; exact = m_ref)
            mkpath(joinpath(@__DIR__, "..", "output"))
            savefig(p, output_path("ch07_01_aggregation.png"))
            println("\n  Plot saved to output/ch07_01_aggregation.png")
        catch e
            @show e
            println("\n  (Plot generation failed)")
        end
    else
        println("\n  (Install Plots.jl to generate plots)")
    end

    # -------------------------------------------------------------------
    # Pass/fail criteria
    # -------------------------------------------------------------------

    max_errs = max_rel_errors(mc)
    pass = verify(mc; atol = [1e-4, 1e-6, Inf])
    print_verification_banner(pass, max_errs, [1e-4, 1e-6, Inf], ["m₀", "m₁", "m₂"])
end

main()
