struct MomentComparison{T}
    times::Vector{T}
    numerical::Matrix{T}
    reference::Matrix{T}
end

function compare_moments(
    t_check::AbstractVector{T},
    numerical_fn,
    reference_fn;
    n_moments::Int,
) where {T}
    n_t = length(t_check)
    numerical = Matrix{T}(undef, n_t, n_moments)
    reference = Matrix{T}(undef, n_t, n_moments)

    for (i, t) in enumerate(t_check)
        num = numerical_fn(t)
        ref = reference_fn(t)
        length(num) == n_moments || throw(DimensionMismatch(
            "numerical_fn returned $(length(num)) moments, expected $n_moments"
        ))
        length(ref) == n_moments || throw(DimensionMismatch(
            "reference_fn returned $(length(ref)) moments, expected $n_moments"
        ))
        numerical[i, :] .= num
        reference[i, :] .= ref
    end

    return MomentComparison(Vector(t_check), numerical, reference)
end

function verify_reconstruction(
    res::QuadratureResult,
    moments::AbstractVector{<:Real};
    tol::Real = 1e-10,
)
    nodes = vec(res.nodes)
    weights = res.weights
    n = length(moments)
    T = promote_type(eltype(nodes), eltype(weights), eltype(moments))
    results = Vector{NamedTuple}(undef, n)

    for k in 0:(n - 1)
        pred = sum(weights .* nodes .^ k)
        exact = moments[k + 1]
        rel_err = abs(pred - exact) / max(abs(exact), eps(T))
        results[k + 1] = (
            order = k,
            predicted = pred,
            exact = exact,
            rel_err = rel_err,
            pass = rel_err < tol,
        )
    end
    return results
end

abs_errors(mc::MomentComparison) = abs.(mc.numerical .- mc.reference)

function rel_errors(mc::MomentComparison{T}) where {T}
    err = abs_errors(mc)
    ref = mc.reference
    err ./ max.(abs.(ref), eps(T))
end

max_abs_errors(mc::MomentComparison) = vec(maximum(abs_errors(mc), dims = 1))
max_rel_errors(mc::MomentComparison) = vec(maximum(rel_errors(mc), dims = 1))

function verify(
    mc::MomentComparison;
    atol::AbstractVector{<:Real},
    rtol::AbstractVector{<:Real} = fill(Inf, length(atol)),
)
    ma = max_abs_errors(mc)
    mr = max_rel_errors(mc)
    all((ma[i] < atol[i]) || (mr[i] < rtol[i]) for i in eachindex(atol))
end
