# Tests for QuadratureMomentsPlotsExt (conditional on Plots availability)

if !isdefined(Main, :_PLOTS_AVAILABLE)
    _PLOTS_AVAILABLE = try
        using Plots
        true
    catch
        false
    end
end

if _PLOTS_AVAILABLE
    using QuadratureMoments
    using QuadratureMoments.Analysis
    using StaticArrays
    using Test

    @testset "Plots Extension" begin
        @testset "plot_moment_evolution" begin
            t = 0.0:0.1:1.0
            m = hcat(ones(length(t)), 2.0 .* ones(length(t)))
            p = plot_moment_evolution(t, m)
            @test p isa Plots.Plot
        end

        @testset "plot_ndf_snapshots" begin
            ξ = 0.0:0.1:1.0
            ndfs = [sin.(ξ), cos.(ξ)]
            times = [0.0, 1.0]
            p = plot_ndf_snapshots(ξ, ndfs, times)
            @test p isa Plots.Plot
        end

        @testset "plot_pbe_summary" begin
            t = 0.0:0.1:1.0
            m = hcat(ones(length(t)), 2.0 .* ones(length(t)))
            ξ = 0.0:0.1:1.0
            ndfs = [sin.(ξ)]
            p = plot_pbe_summary(t, m, ξ, ndfs, [0.0])
            @test p isa Plots.Plot
        end

        @testset "plot_quadrature_nodes" begin
            m = @SVector [1.0, 5.0, 26.0, 140.0]
            res = invert_moments(Wheeler(2), m)
            p = plot_quadrature_nodes(res)
            @test p isa Plots.Plot
        end

        @testset "plot_moment_comparison" begin
            sets = [[1.0, 2.0, 3.0], [1.1, 1.9, 3.2], [1.0, 2.0, 3.0]]
            p = plot_moment_comparison(sets; labels=["A", "B", "C"])
            @test p isa Plots.Plot
        end
    end
else
    @info "Plots.jl not available, skipping extension tests"
end
