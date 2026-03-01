# QBMM.jl/src/Core/kernels.jl

"""
    AbstractKernel

Base abstract type for all continuous kernel functions used in EQMOM/ECQMOM.
"""
abstract type AbstractKernel end

"""
    GaussianKernel <: AbstractKernel

Represents a Gaussian (Normal) distribution kernel.
Used for distributions defined on \$(\\infty, \\infty)\$.
"""
struct GaussianKernel <: AbstractKernel end

"""
    GammaKernel <: AbstractKernel

Represents a Gamma distribution kernel.
Used for distributions defined on \$[0, \\infty)\$.
"""
struct GammaKernel <: AbstractKernel end

"""
    BetaKernel <: AbstractKernel

Represents a Beta distribution kernel.
Used for distributions bounded within \$[0, 1]\$.
"""
struct BetaKernel <: AbstractKernel end

# Reserved for future multi-dimensional extensions:
# struct JointGaussianKernel <: AbstractKernel end
