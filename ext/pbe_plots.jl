function plot_moment_evolution(
    t::AbstractVector,
    moments::AbstractMatrix;
    exact::Union{Nothing,AbstractMatrix} = nothing,
    labels = nothing,
    colors = nothing,
    title::String = "Moment Evolution",
)
    n_mom = size(moments, 2)
    if colors !== nothing && length(colors) < n_mom
        error("colors vector length ($(length(colors))) < number of moments ($n_mom)")
    end
    if labels !== nothing && length(labels) < n_mom
        error("labels vector length ($(length(labels))) < number of moments ($n_mom)")
    end
    p = plot(xlabel = "Time t", ylabel = "Moment value", title = title)

    default_colors = palette(:default)
    for k in 1:n_mom
        col = colors !== nothing ? colors[k] : default_colors[k]
        lab = labels !== nothing ? labels[k] : "m$(k - 1)"
        plot!(p, t, moments[:, k], label = lab, lw = 2, color = col)
        if exact !== nothing
            plot!(p, t, exact[:, k], label = "$(lab) exact", ls = :dash, color = col)
        end
    end
    return p
end

function plot_ndf_snapshots(
    ξ_range::AbstractVector,
    ndfs::AbstractVector{<:AbstractVector},
    times::AbstractVector;
    colors = nothing,
    title::String = "NDF at Snapshots",
)
    n_snaps = length(ndfs)
    if colors !== nothing && length(colors) < n_snaps
        error("colors vector length ($(length(colors))) < number of snapshots ($n_snaps)")
    end

    p = plot(xlabel = "ξ (Volume)", ylabel = "n(ξ)", title = title)
    default_colors = palette(:default)

    for (idx, (ndf, t)) in enumerate(zip(ndfs, times))
        col = colors !== nothing ? colors[idx] : default_colors[idx]
        plot!(p, ξ_range, ndf, label = "t=$(round(t, digits=2))", lw = 2, color = col)
    end
    return p
end

function plot_pbe_summary(
    t_dense::AbstractVector,
    moments::AbstractMatrix,
    ξ_range::AbstractVector,
    ndfs::AbstractVector{<:AbstractVector},
    snap_times::AbstractVector;
    exact::Union{Nothing,AbstractMatrix} = nothing,
    size::Tuple{Int,Int} = (1000, 400),
    colors = nothing,
    labels = nothing,
    title1::String = "Moment Evolution",
    title2::String = "NDF at Snapshots",
)
    p1 = plot_moment_evolution(t_dense, moments; exact = exact, labels = labels, colors = colors, title = title1)
    p2 = plot_ndf_snapshots(ξ_range, ndfs, snap_times; colors = colors, title = title2)
    plot(p1, p2, layout = (1, 2), size = size)
end
