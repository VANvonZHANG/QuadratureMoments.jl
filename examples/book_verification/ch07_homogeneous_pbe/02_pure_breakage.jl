# 02_pure_breakage.jl
#
# Book reference: Section 7.4.1 from
#   Marchisio & Fox (2013), "Computational Models for Polydisperse
#   Particulate and Multiphase Systems", Cambridge University Press.
#
# Solves a spatially homogeneous population balance equation (PBE) with
# pure breakage using QMOM + ODE time integration, comparing against the
# analytical solution.
#
# Physics:
#   - Constant breakage rate: b(V) = b0 = 0.5
#   - Uniform binary daughter distribution: N(V|V') = 2/V'
#   - Initial condition: monodisperse at V = 2.0, i.e. n(V,0) = delta(V - 2)
#
# Analytical solution:
#   m0(t) = m0(0) * exp(b0 * t)   -- number density grows (particles break)
#   m1(t) = m1(0) = 2.0           -- mass is conserved
#
# Breakage source term for moment k (uniform binary daughters, constant b0):
#   S_k = b0 * sum_j w_j * (2/(k+1) - 1) * node_j^k

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

const b0         = 0.5    # constant breakage rate
const V0         = 2.0    # initial monodisperse volume
const N_quad     = 4      # number of quadrature nodes
const n_moments  = 6      # moments tracked: m_0 through m_5
const t_final    = 2.0    # end time

# -----------------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------------

function analytical_moments(t::Float64)
    m0 = 1.0 * exp(b0 * t)   # m0(0) = V0^0 = 1.0
    m1 = V0                    # mass conservation
    return (m0, m1)
end

# -----------------------------------------------------------------------
# Breakage source term via quadrature
# -----------------------------------------------------------------------
# For moment k, with uniform binary daughter distribution N(V|V') = 2/V'
# and constant breakage rate b(V) = b0:
#
#   Birth:  integral_V^infty b(V') * (2/V') * V^k * n(V') dV'
#         = sum_j w_j * b0 * (2/(k+1)) * node_j^k
#
#   Death:  -integral b(V) * V^k * n(V) dV
#         = -sum_i w_i * b0 * node_i^k
#
#   S_k = b0 * sum_j w_j * (2/(k+1) - 1) * node_j^k

function breakage_source(m::MVector{n_moments, Float64})
    # Clamp moments to prevent negative values from ODE solver drift
    m_safe = MVector{n_moments, Float64}(max(mi, 1e-30) for mi in m)

    # Invert moments to obtain quadrature
    res = invert_moments(Wheeler(N_quad), SVector{n_moments, Float64}(m_safe))
    nodes = vec(res.nodes)    # SVector{N_quad, Float64}
    weights = res.weights     # SVector{N_quad, Float64}

    # Compute source terms for each moment order k = 0..n_moments-1
    S = MVector{n_moments, Float64}(zeros(n_moments))
    for k in 0:(n_moments - 1)
        coeff = 2.0 / (k + 1) - 1.0
        val = 0.0
        for j in 1:N_quad
            val += weights[j] * coeff * (nodes[j]^k)
        end
        S[k + 1] = b0 * val
    end
    return S
end

# -----------------------------------------------------------------------
# ODE right-hand side
# -----------------------------------------------------------------------

function ode_rhs(m::MVector{n_moments, Float64}, p, t)
    return breakage_source(m)
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

function main()
    println("=== Pure Breakage PBE: QMOM Verification ===")
    println("Reference: Marchisio & Fox (2013), Section 7.4.1")
    println()
    println("Breakage rate:  b(V) = ", b0, " (constant)")
    println("Daughter dist:  N(V|V') = 2/V' (uniform binary)")
    println("Quadrature:     N_quad = ", N_quad)
    println("Moments:        m_0 .. m_", n_moments - 1, " (n_moments = ", n_moments, ")")
    println("Time span:      [0, ", t_final, "]")
    println()

    # -------------------------------------------------------------------
    # Initial condition: monodisperse at V = 2.0
    # m_k(0) = V0^k
    # -------------------------------------------------------------------

    m0_init = MVector{n_moments, Float64}(undef)
    for k in 0:(n_moments - 1)
        m0_init[k + 1] = V0^k
    end

    println("Initial moments (monodisperse at V=", V0, "):")
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

    println("=== Moment Comparison: QMOM vs Analytical ===")
    println()

    @printf("%-6s  |  %-30s  |  %-30s\n",
        "t", "m_0 (QMOM / Exact / Err)", "m_1 (QMOM / Exact / Err)")
    println(repeat("-", 74))

    max_m0_err = 0.0
    max_m1_err = 0.0

    for t in t_out
        m_num = sol(t)
        m0_num = m_num[1]
        m1_num = m_num[2]

        (m0_ex, m1_ex) = analytical_moments(t)

        err_m0 = abs(m0_num - m0_ex)
        err_m1 = abs(m1_num - m1_ex)

        max_m0_err = max(max_m0_err, err_m0)
        max_m1_err = max(max_m1_err, err_m1)

        @printf("%-6.2f  |  %8.6f / %8.6f / %.2e  |  %8.6f / %8.6f / %.2e\n",
            t,
            m0_num, m0_ex, err_m0,
            m1_num, m1_ex, err_m1)
    end
    println()

    # -------------------------------------------------------------------
    # Max absolute errors
    # -------------------------------------------------------------------

    println("=== Maximum Absolute Errors ===")
    @printf("  max |m_0_err| = %.2e\n", max_m0_err)
    @printf("  max |m_1_err| = %.2e\n", max_m1_err)
    println()

    # -------------------------------------------------------------------
    # Pass/fail criteria
    # -------------------------------------------------------------------

    pass_m0 = max_m0_err < 1e-4
    pass_m1 = max_m1_err < 1e-6

    println("=== Verification ===")
    @printf("  m_0 max abs err < 1e-4 : %s  (actual: %.2e)\n",
        pass_m0 ? "PASS" : "FAIL", max_m0_err)
    @printf("  m_1 max abs err < 1e-6 : %s  (actual: %.2e)\n",
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
