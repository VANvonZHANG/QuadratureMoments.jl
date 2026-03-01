# src/Solvers/1D/wheeler.jl
using LinearAlgebra
using StaticArrays

"""
    Wheeler{N} <: AbstractQBMM{1, N}

Wheeler 算法 (1D 矩反演)。
N 为期望的节点数。
"""
struct Wheeler{N} <: AbstractQBMM{1, N} end
Wheeler(N::Int) = Wheeler{N}()

"""
    invert_moments(method::Wheeler{N}, m::SVector{L, T}; backend=NativeBackend())

1D Wheeler 反演实现，支持自适应降维。
"""
function invert_moments(
    method::Wheeler{N}, 
    m::SVector{L, T}; 
    backend::AbstractMathBackend = NativeBackend()
) where {N, L, T}
    
    # 调用自适应静态核心
    nodes, weights = _adaptive_wheeler_static(m, Val(N), backend)
    
    return QuadratureResult(weights, SMatrix{N, 1, T}(nodes), nothing)
end

function invert_moments(
    method::Wheeler{N}, 
    m::AbstractVector{T}; 
    backend::AbstractMathBackend = NativeBackend()
) where {N, T}
    
    nodes, weights = _wheeler_dynamic(m, N, backend)
    
    return QuadratureResult(SVector{N, T}(weights), SMatrix{N, 1, T}(nodes), nothing)
end

# --- Internal Implementations ---

function _wheeler_dynamic(m::AbstractVector{T}, N_max::Int, backend::AbstractMathBackend; tol=1e-14) where T
    @assert length(m) >= 2N_max "Moments count must be at least 2N."
    
    if m[1] <= tol
        return zeros(T, N_max), zeros(T, N_max)
    end

    a = zeros(T, N_max)
    b = zeros(T, N_max)
    sig = zeros(T, N_max + 2, 2 * N_max + 1)
    
    for i in 1:2*N_max
        sig[1, i] = 0.0
        sig[2, i] = m[i]
    end
    
    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    actual_N = N_max
    for k in 1:N_max-1
        row = k + 2
        for i in k:(2*N_max - k - 1)
            col = i + 1
            sig[row, col] = sig[row-1, col+1] - a[k] * sig[row-1, col] - b[k] * sig[row-2, col]
        end
        
        if abs(sig[row, k+1]) < tol || abs(sig[row-1, k]) < tol
            actual_N = k
            break
        end
        
        a[k+1] = sig[row, k+2] / sig[row, k+1] - sig[row-1, k+1] / sig[row-1, k]
        b[k+1] = sig[row, k+1] / sig[row-1, k]
        
        if b[k+1] < -tol
            actual_N = k
            break
        elseif b[k+1] < 0
            b[k+1] = 0.0
        end
    end
    
    # Build Jacobi Matrix
    J = SymTridiagonal(a[1:actual_N], sqrt.(abs.(b[2:actual_N])))
    eigen_decomp = eigen(J)
    
    # Fill to N_max for consistent return size
    nodes = zeros(T, N_max)
    weights = zeros(T, N_max)
    nodes[1:actual_N] = eigen_decomp.values
    weights[1:actual_N] = m[1] .* (eigen_decomp.vectors[1, :] .^ 2)
    
    return nodes, weights
end

function _adaptive_wheeler_static(m::SVector{L, T}, ::Val{N_max}, backend::AbstractMathBackend; tol=1e-14) where {L, T, N_max}
    a = MVector{N_max, T}(undef)
    b = MVector{N_max, T}(undef)
    sig = zero(MMatrix{N_max+2, L+1, T})
    
    for i in 1:L
        sig[2, i] = m[i]
    end
    
    if abs(m[1]) < tol
        return zero(SVector{N_max, T}), zero(SVector{N_max, T})
    end

    a[1] = m[2] / m[1]
    b[1] = m[1]
    
    actual_N = N_max
    for k in 1:N_max-1
        row = k + 2
        for i in k:(L - k - 1)
            col = i + 1
            sig[row, col] = sig[row-1, col+1] - a[k] * sig[row-1, col] - b[k] * sig[row-2, col]
        end
        
        if abs(sig[row, k+1]) < tol || abs(sig[row-1, k]) < tol
            actual_N = k
            break
        end
        
        new_a = sig[row, k+2] / sig[row, k+1] - sig[row-1, k+1] / sig[row-1, k]
        new_b = sig[row, k+1] / sig[row-1, k]
        
        if new_b < -tol
            actual_N = k
            break
        elseif new_b < 0
            new_b = 0.0
        end
        
        a[k+1] = new_a
        b[k+1] = new_b
    end
    
    J_dense = zero(MMatrix{N_max, N_max, T})
    for i in 1:actual_N
        J_dense[i, i] = a[i]
    end
    for i in 1:actual_N-1
        val = sqrt(abs(b[i+1]))
        J_dense[i, i+1] = val
        J_dense[i+1, i] = val
    end
    
    eigen_decomp = eigen(Symmetric(SMatrix{N_max, N_max, T}(J_dense)))
    
    nodes = SVector{N_max, T}(eigen_decomp.values)
    weights = SVector{N_max, T}(ntuple(i -> m[1] * eigen_decomp.vectors[1, i]^2, Val(N_max)))
    
    return nodes, weights
end
