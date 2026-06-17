function plot_quadrature_nodes(
    res::QuadratureResult{1};
    true_pdf::Union{Nothing,Function}=nothing,
    ξ_range::Union{Nothing,AbstractVector}=nothing,
    show_weights::Bool=true,
    title::String="Quadrature Nodes",
)
    nodes = vec(res.nodes)
    weights = res.weights

    if ξ_range === nothing
        ξ_min = minimum(nodes) - 1.0
        ξ_max = maximum(nodes) + 1.0
        ξ_range_plot = range(ξ_min, ξ_max; length=200)
    else
        ξ_range_plot = ξ_range
    end

    p = plot(; xlabel="ξ", ylabel="n(ξ)", title=title, legend=:topright)

    if true_pdf !== nothing
        pdf_vals = [true_pdf(ξ) for ξ in ξ_range_plot]
        plot!(p, ξ_range_plot, pdf_vals; label="True PDF", lw=2, color=:blue)
    end

    if show_weights
        w_max = maximum(weights)
        scale = w_max > 0 ? 3.0 / w_max : 1.0
        for i in eachindex(nodes)
            y_top = weights[i] * scale
            plot!(
                [nodes[i], nodes[i]],
                [0, y_top];
                color=:red,
                lw=2,
                label=i == 1 ? "Nodes" : false,
            )
            scatter!([nodes[i]], [y_top]; color=:red, ms=6, label=false)
        end
    else
        scatter!(
            nodes, zeros(length(nodes)); ms=8, color=:red, label="Nodes", markershape=:vline
        )
    end

    return p
end

function plot_ndf_reconstruction(
    ξ_range::AbstractVector,
    ndf::AbstractVector;
    true_pdf::Union{Nothing,Function}=nothing,
    title::String="NDF Reconstruction",
)
    p = plot(; xlabel="ξ", ylabel="n(ξ)", title=title, legend=:topright)
    plot!(p, ξ_range, ndf; label="Reconstructed", lw=2, color=:blue)
    if true_pdf !== nothing
        plot!(
            p, ξ_range, [true_pdf(ξ) for ξ in ξ_range]; label="True", ls=:dash, color=:green
        )
    end
    return p
end
