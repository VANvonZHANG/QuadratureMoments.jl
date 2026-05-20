# src/Core/kernels.jl

raw"""
    AbstractKernel

Base abstract type for all continuous kernel functions used in EQMOM/ECQMOM.
raw"""
abstract type AbstractKernel end

raw"""
    GaussianKernel <: AbstractKernel

Represents a Gaussian (Normal) distribution kernel.
Used for distributions defined on \$(\\infty, \\infty)\$.
raw"""
struct GaussianKernel <: AbstractKernel end

raw"""
    GammaKernel <: AbstractKernel

Represents a Gamma distribution kernel.
Used for distributions defined on \$[0, \\infty)\$.
raw"""
struct GammaKernel <: AbstractKernel end

raw"""
    BetaKernel <: AbstractKernel

Represents a Beta distribution kernel.
Used for distributions bounded within \$[0, 1]\$.
raw"""
struct BetaKernel <: AbstractKernel end

# Reserved for future multi-dimensional extensions:
# struct JointGaussianKernel <: AbstractKernel end
