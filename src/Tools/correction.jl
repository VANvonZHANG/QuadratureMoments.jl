using LinearAlgebra
using StaticArrays

"""
    mcgraw_correction(m::AbstractVector{T}; max_iter=20) -> m_corrected
    
McGraw 矩修正算法优化版。
"""
function mcgraw_correction(m::AbstractVector{T}; max_iter=20) where T
    L = length(m)
    if L < 4 return m end
    
    m_curr = collect(m)
    m0_orig = m_curr[1]
    
    for iter in 1:max_iter
        if is_realizable(m_curr)
            m_curr[1] = m0_orig
            return m_curr
        end
        
        ln_m = log.(max.(m_curr, 1e-30))
        d3 = [ln_m[j+3] - 3*ln_m[j+2] + 3*ln_m[j+1] - ln_m[j] for j in 1:L-3]
        sum_d3_2 = dot(d3, d3)
        sum_d3_2 = max(1e-15, sum_d3_2)
        
        best_k = -1
        max_cos2 = -1.0
        best_ln_ck = 0.0
        
        for k in 2:L
            dot_val = 0.0
            bk_norm2 = 0.0
            # bk coefficients for j: [1, -3, 3, -1] for j in [k-3, k-2, k-1, k]
            coeffs = (1.0, -3.0, 3.0, -1.0)
            for (idx, j) in enumerate((k-3, k-2, k-1, k))
                if 1 <= j <= L-3
                    dot_val += d3[j] * coeffs[idx]
                    bk_norm2 += coeffs[idx]^2
                end
            end
            
            if bk_norm2 > 1e-15
                cos2 = (dot_val^2) / (sum_d3_2 * bk_norm2)
                if cos2 > max_cos2
                    max_cos2 = cos2
                    best_k = k
                    best_ln_ck = - dot_val / bk_norm2
                end
            end
        end
        
        if best_k != -1
            m_curr[best_k] *= exp(0.8 * best_ln_ck)
        else
            break
        end
    end
    
    if !is_realizable(m_curr)
        return _wright_fallback(m_curr)
    end
    return m_curr
end

"""
    mcgraw_correction(m::StaticVector{L, T}; max_iter=20)
"""
@inline function mcgraw_correction(m::StaticVector{L, T}; max_iter=20) where {L, T}
    if L < 4 return m end
    
    m_curr = MVector{L, T}(m)
    m0_orig = m_curr[1]
    
    for iter in 1:max_iter
        if is_realizable(m_curr)
            m_curr[1] = m0_orig
            return SVector{L, T}(m_curr)
        end
        
        ln_m = ntuple(i -> log(max(m_curr[i], 1e-30)), Val(L))
        # d3[j] = ln_m[j+3] - 3*ln_m[j+2] + 3*ln_m[j+1] - ln_m[j]
        d3 = ntuple(j -> ln_m[j+3] - 3*ln_m[j+2] + 3*ln_m[j+1] - ln_m[j], Val(L-3))
        
        sum_d3_2 = 0.0
        for j in 1:L-3
            sum_d3_2 += d3[j]^2
        end
        sum_d3_2 = max(1e-15, sum_d3_2)
        
        best_k = -1
        max_cos2 = -1.0
        best_ln_ck = 0.0
        
        for k in 2:L
            dot_val = 0.0
            bk_norm2 = 0.0
            
            # j=k-3
            j1 = k-3
            if 1 <= j1 <= L-3
                dot_val += d3[j1] * 1.0
                bk_norm2 += 1.0
            end
            # j=k-2
            j2 = k-2
            if 1 <= j2 <= L-3
                dot_val += d3[j2] * (-3.0)
                bk_norm2 += 9.0
            end
            # j=k-1
            j3 = k-1
            if 1 <= j3 <= L-3
                dot_val += d3[j3] * 3.0
                bk_norm2 += 9.0
            end
            # j=k
            j4 = k
            if 1 <= j4 <= L-3
                dot_val += d3[j4] * (-1.0)
                bk_norm2 += 1.0
            end
            
            if bk_norm2 > 1e-15
                cos2 = (dot_val^2) / (sum_d3_2 * bk_norm2)
                if cos2 > max_cos2
                    max_cos2 = cos2
                    best_k = k
                    best_ln_ck = - dot_val / bk_norm2
                end
            end
        end
        
        if best_k != -1
            m_curr[best_k] *= exp(0.8 * best_ln_ck)
        else
            break
        end
    end
    
    if !is_realizable(m_curr)
        return _wright_fallback_static(m_curr)
    end
    return SVector{L, T}(m_curr)
end

function _wright_fallback(m::AbstractVector{T}) where T
    L = length(m)
    m_res = copy(m)
    m0, m1, m2 = m[1], m[2], m[3]
    var_term = m2*m0 / (m1^2)
    if var_term <= 1.0
        for k in 1:L-1
            m_res[k+1] = m0 * (m1/m0)^k
        end
    else
        sig2 = log(var_term)
        mu = log(m1/m0) - 0.5*sig2
        for k in 3:L-1
            m_res[k+1] = m0 * exp(k*mu + 0.5*(k^2)*sig2)
        end
    end
    return m_res
end

@inline function _wright_fallback_static(m::StaticVector{L, T}) where {L, T}
    m0, m1, m2 = m[1], m[2], m[3]
    var_term = m2*m0 / (m1^2)
    if var_term <= 1.0
        return SVector{L, T}(ntuple(k -> m0 * (m1/m0)^(k-1), Val(L)))
    else
        sig2 = log(var_term)
        mu = log(m1/m0) - 0.5*sig2
        return SVector{L, T}(ntuple(k -> k <= 3 ? m[k] : m0 * exp((k-1)*mu + 0.5*((k-1)^2)*sig2), Val(L)))
    end
end
