function plot_moment_comparison(
    moment_sets::Vector{<:AbstractVector};
    labels::Union{Nothing,Vector{String}} = nothing,
    colors = nothing,
    yscale::Symbol = :log10,
    title::String = "Moment Comparison",
)
    n_sets = length(moment_sets)
    n_sets > 0 || error("moment_sets cannot be empty")
    if colors !== nothing && length(colors) < n_sets
        error("colors vector length ($(length(colors))) < number of sets ($n_sets)")
    end
    if labels !== nothing && length(labels) < n_sets
        error("labels vector length ($(length(labels))) < number of sets ($n_sets)")
    end
    n_moments = length(moment_sets[1])

    for (i, ms) in enumerate(moment_sets)
        length(ms) == n_moments || error("Moment set $i has $(length(ms)) elements, expected $n_moments")
    end

    k_labels = ["m$k" for k in 0:(n_moments - 1)]
    w = 0.8 / n_sets
    offset_range = range(-(n_sets - 1) * w / 2, (n_sets - 1) * w / 2, length = n_sets)

    p = plot(xlabel = "Moment order", ylabel = "Value", title = title, yscale = yscale)
    default_colors = palette(:default)

    for (idx, ms) in enumerate(moment_sets)
        col = colors !== nothing ? colors[idx] : default_colors[idx]
        lab = labels !== nothing ? labels[idx] : "Set $idx"
        x_pos = collect(1:n_moments) .+ offset_range[idx]
        bar!(p, x_pos, ms, bar_width = w, label = lab, color = col)
    end

    plot!(p, xticks = (1:n_moments, k_labels))
    return p
end
