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
    println("OrdinaryDiffEq is required. Install with: using Pkg; Pkg.add(\"OrdinaryDiffEq\")")
    exit(1)
end

using QuadratureMoments
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

function main()
    # -----------------------------------------------------------------------
    # Problem parameters
    # -----------------------------------------------------------------------

    G0     = 0.5     # constant growth rate
    J      = 1.0     # constant nucleation rate
    V_nuc  = 0.1     # nucleation size
    N_quad = 4       # number of quadrature nodes (needs 2*N_quad = 8 moments)
    N_mom  = 8       # number of moments tracked (m0..m7)
    tspan  = (0.01, 3.0)  # start at t=0.01 to avoid degenerate quadrature

    # Near-empty initial condition (avoid exactly zero for quadrature)
    # Use a small monodisperse distribution at V ~ 0.1 (near nucleation size)
    # to ensure realizability: m_k = m0 * V_mean^k
    V_init = 0.1
    m0_init = MVector{8, Float64}(undef)
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
        G0_val  = p[1]
        J_val   = p[2]
        Vnuc    = p[3]
        Nq      = p[4]
        Nm      = p[5]

        # Clamp moments to avoid negative/zero values before inversion
        m_safe = MVector{8, Float64}(undef)
        for i in 1:8
            m_safe[i] = max(m[i], 1e-30)
        end

        # Correct moments to ensure realizability
        m_corr = mcgraw_correction(SVector{8, Float64}(m_safe))
        m_safe = MVector{8, Float64}(m_corr)

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
    sol = solve(prob, Tsit5(); reltol = 1e-10, abstol = 1e-12, dense = true)
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

    # -----------------------------------------------------------------------
    # Comparison table at selected time points
    # -----------------------------------------------------------------------

    t_check = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    println("--- Comparison: QMOM vs Analytical Solution ---")
    println()
    @printf("  %6s  %12s  %12s  %12s  %12s  %12s\n",
            "t", "m0_QMOM", "m0_exact", "m0_err",
            "m1_QMOM", "m1_exact")
    @printf("  %6s  %12s  %12s  %12s  %12s  %12s\n",
            "", "", "", "", "", "")
    println("  ", "-"^78)

    max_m0_err = 0.0
    max_m1_err = 0.0

    for t in t_check
        m_computed = sol(t)
        m0_num = m_computed[1]
        m1_num = m_computed[2]

        m0_exact = J * t
        m1_exact = J * V_nuc * t + J * G0 * t^2 / 2

        m0_err = abs(m0_num - m0_exact)
        m1_err = abs(m1_num - m1_exact)

        max_m0_err = max(max_m0_err, m0_err)
        max_m1_err = max(max_m1_err, m1_err)

        @printf("  %6.2f  %12.6f  %12.6f  %12.2e  %12.6f  %12.6f\n",
                t, m0_num, m0_exact, m0_err, m1_num, m1_exact)
    end

    println()

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------

    if _PLOTS_AVAILABLE
        try
            Plots.gr()

        # Left panel: moment time evolution
        t_dense = range(tspan[1], tspan[2], length=100)
        m0_qmom = [sol(t)[1] for t in t_dense]
        m1_qmom = [sol(t)[2] for t in t_dense]
        m0_exact = [J * t for t in t_dense]
        m1_exact = [J * V_nuc * t + J * G0 * t^2 / 2 for t in t_dense]

        p1 = plot(t_dense, m0_qmom, label="m₀ QMOM", lw=2, color=:blue,
                  xlabel="Time t", ylabel="Moment value", title="Moment Evolution")
        plot!(t_dense, m0_exact, label="m₀ Exact", ls=:dash, color=:blue)
        plot!(t_dense, m1_qmom, label="m₁ QMOM", lw=2, color=:green)
        plot!(t_dense, m1_exact, label="m₁ Exact", ls=:dash, color=:green)

        # Right panel: NDF reconstruction via EQMOM
        ξ_range = range(0, 2, length=200)
        n_snap = 5
        t_snaps = range(tspan[1], tspan[2], length=n_snap+1)[2:end]
        colors_snap = [:blue, :green, :orange, :red, :purple]

        p2 = plot(title="NDF at Snapshots", xlabel="ξ (Volume)", ylabel="n(ξ)")
        for (idx, t_snap) in enumerate(t_snaps)
            m_at_t = sol(t_snap)
            m_snap = SVector{N_mom, Float64}(max(mi, 1e-30) for mi in m_at_t)
            n_eq = (N_mom - 1) ÷ 2  # EQMOM needs 2*n_eq+1 <= N_mom
            res_eq = invert_moments(EQMOM(n_eq, GaussianKernel()), m_snap)
            σ = res_eq.sigmas[1]
            ndf = zeros(length(ξ_range))
            for (i, ξ) in enumerate(ξ_range)
                for α in 1:length(res_eq.weights)
                    ndf[i] += res_eq.weights[α] * exp(-(ξ - res_eq.nodes[α])^2 / (2σ^2)) / (σ * sqrt(2π))
                end
            end
            plot!(p2, ξ_range, ndf, label="t=$(round(t_snap, digits=2))", lw=2,
                  color=colors_snap[idx])
        end

        p = plot(p1, p2, layout=(1,2), size=(1000,400))
        mkpath(joinpath(@__DIR__, "..", "output"))
        savefig(p, joinpath(@__DIR__, "..", "output", "ch07_03_growth_nucleation.png"))
        println("\n  Plot saved to output/ch07_03_growth_nucleation.png")
    catch e
        @show e
        println("\n  (Install Plots.jl to generate plots)")
    end
    end

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    tol_m0 = 5e-3
    tol_m1 = 5e-3

    m0_pass = max_m0_err < tol_m0
    m1_pass = max_m1_err < tol_m1

    println("--- Error Summary ---")
    @printf("  max |m0_err| = %.2e  (tol = %.0e)  %s\n",
            max_m0_err, tol_m0, m0_pass ? "PASS" : "FAIL")
    @printf("  max |m1_err| = %.2e  (tol = %.0e)  %s\n",
            max_m1_err, tol_m1, m1_pass ? "PASS" : "FAIL")
    println()

    all_pass = m0_pass && m1_pass

    println("========================================")
    if all_pass
        println("  PASS")
    else
        println("  FAIL")
        if !m0_pass
            @printf("  - m0 (number density): max_err = %.2e > tol = %.0e\n",
                    max_m0_err, tol_m0)
        end
        if !m1_pass
            @printf("  - m1 (volume concentration): max_err = %.2e > tol = %.0e\n",
                    max_m1_err, tol_m1)
        end
    end
    println("========================================")
end

main()
