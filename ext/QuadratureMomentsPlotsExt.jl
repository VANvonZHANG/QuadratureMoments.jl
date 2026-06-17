module QuadratureMomentsPlotsExt

using QuadratureMoments
using QuadratureMoments.Analysis
using Plots

import QuadratureMoments.Analysis:
    plot_moment_evolution,
    plot_ndf_snapshots,
    plot_pbe_summary,
    plot_moment_comparison,
    plot_quadrature_nodes,
    plot_ndf_reconstruction

include("pbe_plots.jl")
include("quadrature_plots.jl")
include("correction_plots.jl")

end
