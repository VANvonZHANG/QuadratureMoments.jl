using SpecialFunctions

raw"""
    evaluate_kernel(kernel, ξ, node, σ)

Evaluate a kernel function at abscissa `ξ` centered at `node` with bandwidth `σ`.
"""
function evaluate_kernel end

raw"""
    evaluate_kernel(::GaussianKernel, ξ, node, σ)

Gaussian (normal) kernel: ``\\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\!\\left(-\\frac{(\\xi - \\text{node})^2}{2\\sigma^2}\\right)``.
"""
function evaluate_kernel(::GaussianKernel, ξ::Real, node::Real, σ::Real)
    exp(-(ξ - node)^2 / (2 * σ^2)) / (σ * sqrt(2π))
end

raw"""
    evaluate_kernel(::GammaKernel, ξ, node, σ)

Gamma kernel with shape ``\\alpha = (\\mu/\\sigma)^2`` and scale ``\\theta = \\sigma^2/\\mu``.
"""
function evaluate_kernel(::GammaKernel, ξ::Real, node::Real, σ::Real)
    ξ > 0 || return zero(promote_type(typeof(ξ), typeof(node), typeof(σ)))
    μ = node
    α = (μ / σ)^2
    θ = σ^2 / μ
    # Use log-domain computation to avoid overflow
    log_pdf = (α - 1) * log(ξ) - ξ / θ - α * log(θ) - loggamma(α)
    exp(log_pdf)
end

raw"""
    evaluate_kernel(::BetaKernel, ξ, node, σ)

Beta kernel on ``[0,1]`` with parameters derived from mean `node` and variance `σ^2`.
"""
function evaluate_kernel(::BetaKernel, ξ::Real, node::Real, σ::Real)
    (0 < ξ < 1) || return zero(promote_type(typeof(ξ), typeof(node), typeof(σ)))
    μ = node
    ν = max(μ * (1 - μ) / σ^2 - 1, one(σ))
    α = μ * ν
    β = (1 - μ) * ν
    # Guard against zero alpha or beta
    α > 0 && β > 0 || return zero(promote_type(typeof(ξ), typeof(node), typeof(σ)))
    ξ^(α - 1) * (1 - ξ)^(β - 1) / beta(α, β)
end

raw"""
    reconstruct_ndf(res, ξ_range, kernel)

Reconstruct the number density function (NDF) from a 1-D quadrature result `res`
on the abscissa grid `ξ_range` using the specified `kernel`.

Requires `res.sigmas !== nothing` (e.g. from EQMOM or ECQMOM).
"""
function reconstruct_ndf(
    res::QuadratureResult{1,N,T},
    ξ_range::AbstractVector{<:Real},
    kernel::AbstractKernel,
) where {N,T}
    sigmas = res.sigmas
    if sigmas === nothing
        throw(ArgumentError(
            "QuadratureResult has no kernel bandwidth (sigmas === nothing). " *
            "Use EQMOM or ECQMOM for continuous NDF reconstruction."
        ))
    end

    σ = sigmas[1]
    ndf = zeros(promote_type(T, eltype(ξ_range)), length(ξ_range))

    for (i, ξ) in enumerate(ξ_range)
        for α in eachindex(res.weights)
            ndf[i] += res.weights[α] * evaluate_kernel(kernel, ξ, res.nodes[α], σ)
        end
    end
    return ndf
end
