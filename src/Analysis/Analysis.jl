module Analysis

using ..QuadratureMoments

include("ndf_reconstruction.jl")
include("moment_verification.jl")

export reconstruct_ndf, evaluate_kernel
export MomentComparison, compare_moments, verify_reconstruction
export abs_errors, rel_errors, max_abs_errors, max_rel_errors, verify

# Plotting functions — methods provided by QuadratureMomentsPlotsExt when Plots.jl is loaded
function plot_moment_evolution end
function plot_ndf_snapshots end
function plot_pbe_summary end
function plot_moment_comparison end
function plot_quadrature_nodes end
function plot_ndf_reconstruction end
export plot_moment_evolution, plot_ndf_snapshots, plot_pbe_summary
export plot_moment_comparison, plot_quadrature_nodes, plot_ndf_reconstruction

end
