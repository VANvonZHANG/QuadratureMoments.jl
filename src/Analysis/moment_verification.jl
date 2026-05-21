raw"""
    MomentComparison{T}

Store the comparison between numerical and reference moments over time.

# Fields
- `times::Vector{T}`: Time instants at which moments were evaluated.
- `numerical::Matrix{T}`: Numerical moments (rows = time, cols = moment order).
- `reference::Matrix{T}`: Reference moments (rows = time, cols = moment order).
"""
struct ReconstructionCheck{T}
    order::Int
    predicted::T
    exact::T
    rel_err::T
    pass::Bool
end

struct MomentComparison{T}
    times::Vector{T}
    numerical::Matrix{T}
    reference::Matrix{T}
end

raw"""
    compare_moments(t_check, numerical_fn, reference_fn; n_moments)

Evaluate `numerical_fn(t)` and `reference_fn(t)` at each `t` in `t_check`
and return a [`MomentComparison`](@ref).

Both functions must return vectors of length `n_moments`.
"""
function compare_moments(
    t_check::AbstractVector,
    numerical_fn,
    reference_fn;
    n_moments::Int,
)
    t_check = float.(t_check)
    T = eltype(t_check)
    isempty(t_check) && return MomentComparison(T[], Matrix{T}(undef, 0, n_moments), Matrix{T}(undef, 0, n_moments))
    n_t = length(t_check)
    sample_num = numerical_fn(first(t_check))
    sample_ref = reference_fn(first(t_check))
    ElT = promote_type(T, eltype(sample_num), eltype(sample_ref))
    numerical = Matrix{ElT}(undef, n_t, n_moments)
    reference = Matrix{ElT}(undef, n_t, n_moments)

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

raw"""
    verify_reconstruction(res, moments; tol=1e-10)

Check that the quadrature result `res` reproduces each moment in `moments`
up to relative tolerance `tol`. Returns a vector of [`ReconstructionCheck`](@ref)
with fields `order`, `predicted`, `exact`, `rel_err`, and `pass`.
"""
function verify_reconstruction(
    res::QuadratureResult,
    moments::AbstractVector{<:Real};
    tol::Real = 1e-10,
)
    nodes = vec(res.nodes)
    weights = res.weights
    n = length(moments)
    T = promote_type(eltype(nodes), eltype(weights), eltype(moments))
    results = Vector{ReconstructionCheck{T}}(undef, n)

    for k in 0:(n - 1)
        pred = sum(weights .* nodes .^ k)
        exact = moments[k + 1]
        rel_err = abs(pred - exact) / max(abs(exact), eps(T))
        results[k + 1] = ReconstructionCheck{T}(k, pred, exact, rel_err, rel_err < tol)
    end
    return results
end

raw"""
    abs_errors(mc::MomentComparison)

Return the element-wise absolute error `abs.(numerical .- reference)`.
"""
abs_errors(mc::MomentComparison) = abs.(mc.numerical .- mc.reference)

raw"""
    rel_errors(mc::MomentComparison)

Return the element-wise relative error using [`abs_errors`](@ref).
"""
function rel_errors(mc::MomentComparison{T}) where {T}
    err = abs_errors(mc)
    ref = mc.reference
    err ./ max.(abs.(ref), eps(T))
end

raw"""
    max_abs_errors(mc::MomentComparison)

Return the maximum absolute error for each moment order across all times.
"""
function max_abs_errors(mc::MomentComparison)
    isempty(mc.times) && return zeros(eltype(mc.times), size(mc.numerical, 2))
    vec(maximum(abs_errors(mc), dims = 1))
end

raw"""
    max_rel_errors(mc::MomentComparison)

Return the maximum relative error for each moment order across all times.
"""
function max_rel_errors(mc::MomentComparison)
    isempty(mc.times) && return zeros(eltype(mc.times), size(mc.numerical, 2))
    vec(maximum(rel_errors(mc), dims = 1))
end

raw"""
    verify(mc; atol, rtol=fill(Inf, length(atol)))

Check that every moment satisfies either an absolute or a relative error bound.
Returns `true` if all moments pass.
"""
function verify(
    mc::MomentComparison;
    atol::AbstractVector{<:Real},
    rtol::AbstractVector{<:Real} = fill(Inf, length(atol)),
)
    isempty(mc.times) && return true
    ma = max_abs_errors(mc)
    mr = max_rel_errors(mc)
    all((ma[i] < atol[i]) || (mr[i] < rtol[i]) for i in eachindex(atol))
end
