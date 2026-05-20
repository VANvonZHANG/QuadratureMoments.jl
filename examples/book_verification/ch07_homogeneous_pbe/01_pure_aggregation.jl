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
using StaticArrays
using LinearAlgebra
using Printf

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

    println("=== Moment Comparison: QMOM vs Exact ===")
    println()

    @printf("%-6s  |  %-30s  |  %-30s  |  %-30s\n",
        "t", "m_0 (QMOM / Exact / RelErr)", "m_1 (QMOM / Exact / RelErr)",
        "m_2 (QMOM / Exact / RelErr)")
    println(repeat("-", 106))

    max_m0_err = 0.0
    max_m1_err = 0.0
    max_m2_err = 0.0

    for t in t_out
        m_num = sol(t)
        m0_num = m_num[1]
        m1_num = m_num[2]
        m2_num = m_num[3]

        (m0_ex, m1_ex, m2_ex) = analytical_moments(t)

        err_m0 = abs(m0_num - m0_ex) / max(abs(m0_ex), 1e-30)
        err_m1 = abs(m1_num - m1_ex) / max(abs(m1_ex), 1e-30)
        err_m2 = abs(m2_num - m2_ex) / max(abs(m2_ex), 1e-30)

        max_m0_err = max(max_m0_err, err_m0)
        max_m1_err = max(max_m1_err, err_m1)
        max_m2_err = max(max_m2_err, err_m2)

        @printf("%-6.2f  |  %8.6f / %8.6f / %.2e  |  %8.6f / %8.6f / %.2e  |  %8.6f / %8.6f / %.2e\n",
            t,
            m0_num, m0_ex, err_m0,
            m1_num, m1_ex, err_m1,
            m2_num, m2_ex, err_m2)
    end
    println()

    # -------------------------------------------------------------------
    # Max relative errors
    # -------------------------------------------------------------------

    println("=== Maximum Relative Errors ===")
    @printf("  max |m_0_err| = %.2e\n", max_m0_err)
    @printf("  max |m_1_err| = %.2e\n", max_m1_err)
    @printf("  max |m_2_err| = %.2e\n", max_m2_err)
    println()

    # -------------------------------------------------------------------
    # Pass/fail criteria
    # -------------------------------------------------------------------

    pass_m0 = max_m0_err < 1e-4
    pass_m1 = max_m1_err < 1e-6

    println("=== Verification ===")
    @printf("  m_0 max rel err < 1e-4 : %s  (actual: %.2e)\n",
        pass_m0 ? "PASS" : "FAIL", max_m0_err)
    @printf("  m_1 max rel err < 1e-6 : %s  (actual: %.2e)\n",
        pass_m1 ? "PASS" : "FAIL", max_m1_err)
    println()

    if pass_m0 && pass_m1
        println("PASS")
    else
        println("FAIL")
        if !pass_m0
            println("  m_0 error exceeds tolerance")
        end
        if !pass_m1
            println("  m_1 error exceeds tolerance")
        end
    end
end

main()
