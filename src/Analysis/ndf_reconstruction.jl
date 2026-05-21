using SpecialFunctions

function evaluate_kernel(::GaussianKernel, ξ::Real, node::Real, σ::Real)
    exp(-(ξ - node)^2 / (2 * σ^2)) / (σ * sqrt(2π))
end

function evaluate_kernel(::GammaKernel, ξ::Real, node::Real, σ::Real)
    μ = node
    α = (μ / σ)^2
    θ = σ^2 / μ
    ξ^(α - 1) * exp(-ξ / θ) / (θ^α * gamma(α))
end

function evaluate_kernel(::BetaKernel, ξ::Real, node::Real, σ::Real)
    μ = node
    ν = max(μ * (1 - μ) / σ^2 - 1, 1.0)
    α = μ * ν
    β = (1 - μ) * ν
    ξ^(α - 1) * (1 - ξ)^(β - 1) / beta(α, β)
end

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
