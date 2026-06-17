# 03_growth_nucleation.jl
#
# Book reference: Section 7.4.1 from
#   Marchisio & Fox (2013), "Computational Models for Polydisperse
#   Particulate and Multiphase Systems", Cambridge University Press.
#
# Solves a spatially homogeneous population balance equation (PBE) with
# constant particle growth and nucleation using QMOM coupled with ODE
# time integration, comparing against the analytical solution.
#
# Physics:
#   - Constant growth rate: G0 = 0.5 (particles grow in size continuously)
#   - Constant nucleation rate: J = 1.0, at size V_nuc = 0.1
#   - Start from near-empty distribution
#
# Analytical solution (method of characteristics, starting from empty):
#   m0(t) = J * t                              -- number grows linearly
#   m1(t) = J * V_nuc * t + J * G0 * t^2 / 2  -- volume grows quadratically
#
# Growth source term for moment k (phase-space advection, Leibniz rule):
#   S_k_growth = k * G0 * sum_i w_i * node_i^(k-1)
#   Note: S_0_growth = 0 (no change in number due to growth)
#
# Nucleation source term for moment k:
#   S_k_nuc = J * V_nuc^k  (new particles appear at size V_nuc)
#
# Total: S_k = k * G0 * sum_i w_i * node_i^(k-1) + J * V_nuc^k

try
    using OrdinaryDiffEq
catch
    println(
        "OrdinaryDiffEq is required. Install with: using Pkg; Pkg.add(\"OrdinaryDiffEq\")"
    )
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

function main()
    # -----------------------------------------------------------------------
    # Problem parameters
    # -----------------------------------------------------------------------

    G0 = 0.5     # constant growth rate
    J = 1.0     # constant nucleation rate
    V_nuc = 0.1     # nucleation size
    N_quad = 4       # number of quadrature nodes (needs 2*N_quad = 8 moments)
    N_mom = 8       # number of moments tracked (m0..m7)
    tspan = (0.01, 3.0)  # start at t=0.01 to avoid degenerate quadrature

    # Near-empty initial condition (avoid exactly zero for quadrature)
    # Use a small monodisperse distribution at V ~ 0.1 (near nucleation size)
    # to ensure realizability: m_k = m0 * V_mean^k
    V_init = 0.1
    m0_init = MVector{8,Float64}(undef)
    for k in 0:7
        m0_init[k + 1] = 0.01 * V_init^k
    end

    println("=== Growth + Nucleation PBE: QMOM Verification ===")
    println("Reference: Marchisio & Fox (2013), Section 7.4.1")
    println()
    println("Parameters:")
    @printf("  Growth rate    G0    = %.2f\n", G0)
    @printf("  Nucleation rate J    = %.1f\n", J)
    @printf("  Nucleation size V_nuc = %.2f\n", V_nuc)
    @printf("  Quadrature nodes  N  = %d\n", N_quad)
    @printf("  Moments tracked     = %d (m0..m7)\n", N_mom)
    @printf("  Time span           = [%.2f, %.1f]\n", tspan[1], tspan[2])
    println()
    println("Initial moments (near-empty):")
    for k in 0:(N_mom - 1)
        @printf("  m_%d = %.2e\n", k, m0_init[k + 1])
    end
    println()

    # Pack parameters into a tuple for the ODE RHS
    params = (G0, J, V_nuc, N_quad, N_mom)

    # -----------------------------------------------------------------------
    # Right-hand side: compute growth + nucleation source terms via QMOM
    # -----------------------------------------------------------------------

    function rhs!(dm, m, p, t)
        G0_val = p[1]
        J_val = p[2]
        Vnuc = p[3]
        Nq = p[4]
        Nm = p[5]

        # Clamp moments to avoid negative/zero values before inversion
        m_safe = MVector{8,Float64}(undef)
        for i in 1:8
            m_safe[i] = max(m[i], 1e-30)
        end

        # Correct moments to ensure realizability
        m_corr = mcgraw_correction(SVector{8,Float64}(m_safe))
        m_safe = MVector{8,Float64}(m_corr)

        # Moment inversion: recover quadrature nodes and weights
        res = invert_moments(Wheeler(Nq), m_safe)
        nodes = vec(res.nodes)
        weights = res.weights

        # Compute growth + nucleation source terms for each moment k
        for k in 0:(Nm - 1)
            # Growth term: S_k_growth = k * G0 * sum_j w_j * node_j^(k-1)
            if k == 0
                S_growth = 0.0  # no change in number due to growth
            else
                S_growth = 0.0
                for j in 1:Nq
                    S_growth += weights[j] * (nodes[j] ^ (k - 1))
                end
                S_growth *= k * G0_val
            end

            # Nucleation term: S_k_nuc = J * V_nuc^k
            S_nuc = J_val * (Vnuc ^ k)

            dm[k + 1] = S_growth + S_nuc
        end

        return nothing
    end

    # -----------------------------------------------------------------------
    # Solve ODE
    # -----------------------------------------------------------------------

    println("Solving ODE with Tsit5() ...")
    prob = ODEProblem(rhs!, m0_init, tspan, params)
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12, dense=true)
    println("  Status: ", sol.retcode)
    println()

    # -----------------------------------------------------------------------
    # Analytical solution
    # -----------------------------------------------------------------------
    # m0(t) = J * t  (starting from near-empty, ignoring initial m0 ~ 0.01)
    # m1(t) = J * V_nuc * t + J * G0 * t^2 / 2  (approximate, ignoring initial)
    #
    # Note: The analytical solution assumes starting from an empty distribution.
    # Since our initial condition is near-empty but not exactly zero, and we
    # start at t=0.01, there is a small systematic offset. The comparison is
    # therefore approximate.

    analytical_moments(t) = [J * t, J * V_nuc * t + J * G0 * t^2 / 2]

    # -----------------------------------------------------------------------
    # Comparison table at selected time points
    # -----------------------------------------------------------------------

    t_check = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    mc = compare_moments(t_check, t -> sol(t)[1:2], analytical_moments; n_moments=2)
    print_comparison_table(mc, ["m₀", "m₁"])

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------

    if _PLOTS_AVAILABLE
        try
            Plots.gr()

            # Build dense time arrays for plotting
            t_dense = range(tspan[1], tspan[2]; length=100)

            # Build moment matrices
            n_mom_plot = 2
            m_num = Matrix{Float64}(undef, length(t_dense), n_mom_plot)
            m_ref = Matrix{Float64}(undef, length(t_dense), n_mom_plot)
            for (i, t) in enumerate(t_dense)
                m_num[i, :] .= sol(t)[1:n_mom_plot]
                m_ref[i, :] .= analytical_moments(t)[1:n_mom_plot]
            end

            # NDF reconstruction at snapshot times
            ξ_range = range(0, 2; length=200)
            snapshot_times = range(tspan[1], tspan[2]; length=6)[2:end]
            ndfs = Vector{Float64}[]
            for t_snap in snapshot_times
                m_at_t = sol(t_snap)
                m_snap = SVector{N_mom,Float64}(max(mi, 1e-30) for mi in m_at_t)
                n_eq = (N_mom - 1) ÷ 2
                res_eq = invert_moments(EQMOM(n_eq, GaussianKernel()), m_snap)
                ndf = reconstruct_ndf(res_eq, ξ_range, GaussianKernel())
                push!(ndfs, ndf)
            end

            # Plot
            p = plot_pbe_summary(t_dense, m_num, ξ_range, ndfs, snapshot_times; exact=m_ref)
            mkpath(joinpath(@__DIR__, "..", "output"))
            savefig(p, output_path("ch07_03_growth_nucleation.png"))
            println("\n  Plot saved to output/ch07_03_growth_nucleation.png")
        catch e
            @show e
            println("\n  (Plot generation failed)")
        end
    else
        println("\n  (Install Plots.jl to generate plots)")
    end

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    max_errs = max_abs_errors(mc)
    pass = verify(mc; atol=[5e-3, 5e-3])
    return print_verification_banner(pass, max_errs, [5e-3, 5e-3], ["m₀", "m₁"])
end

main()
