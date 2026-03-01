# QBMM.jl/src/Core/kernels.jl

"""
    AbstractKernel
    GaussianKernel, GammaKernel, BetaKernel
    
用于 EQMOM/ECQMOM 的连续核函数基类。
"""
abstract type AbstractKernel end
struct GaussianKernel <: AbstractKernel end
struct GammaKernel <: AbstractKernel end
struct BetaKernel <: AbstractKernel end

# 为以后多维扩展预留：
# struct JointGaussianKernel <: AbstractKernel end
