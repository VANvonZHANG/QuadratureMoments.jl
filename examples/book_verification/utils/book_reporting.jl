# Shared reporting utilities for book verification examples

using QuadratureMoments.Analysis
using Printf

"""
    print_comparison_table(mc::MomentComparison, moment_names)

Print formatted comparison table of numerical vs reference moments.
"""
function print_comparison_table(mc::MomentComparison, moment_names::Vector{String})
    println("=== Moment Comparison: QMOM vs Exact ===")
    println()

    n_mom = length(moment_names)
    header_parts = ["$(name) (Num / Ref / Err)" for name in moment_names]
    header = join(header_parts, "  |  ")
    @printf("%-6s  |  %s\n", "t", header)
    println(repeat("-", 6 + 4 + length(header)))

    for (i, t) in enumerate(mc.times)
        parts = String[]
        for k in 1:n_mom
            num = mc.numerical[i, k]
            ref = mc.reference[i, k]
            err = abs(num - ref)
            push!(parts, @sprintf("%8.6f / %8.6f / %.2e", num, ref, err))
        end
        @printf("%-6.2f  |  %s\n", t, join(parts, "  |  "))
    end
    return println()
end

"""
    print_verification_banner(passed::Bool, max_errors, tolerances, moment_names)

Print PASS/FAIL banner with per-moment error details.
"""
function print_verification_banner(
    passed::Bool,
    max_errors::Vector{<:Real},
    tolerances::Vector{<:Real},
    moment_names::Vector{String}=["m_$i" for i in eachindex(max_errors)],
)
    println("=== Verification ===")
    for (i, (err, tol)) in enumerate(zip(max_errors, tolerances))
        name = moment_names[i]
        pass_i = err < tol
        @printf(
            "  %s max err < %.0e : %s  (actual: %.2e)\n",
            name,
            tol,
            pass_i ? "PASS" : "FAIL",
            err
        )
    end
    println()

    println("========================================")
    if passed
        println("  PASS")
    else
        println("  FAIL")
        for (i, (err, tol)) in enumerate(zip(max_errors, tolerances))
            if err >= tol
                @printf("  - %s: max_err = %.2e >= tol = %.0e\n", moment_names[i], err, tol)
            end
        end
    end
    return println("========================================")
end

"""
    output_path(filename) -> String

Return path to the shared output directory for book verification plots.
"""
output_path(filename::String) = joinpath(@__DIR__, "..", "output", filename)
