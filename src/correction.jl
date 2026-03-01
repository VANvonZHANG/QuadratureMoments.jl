using LinearAlgebra
using StaticArrays

"""
    mcgraw_correction(m::AbstractVector{T}; max_iter=20, tol=1e-10) -> m_corrected
    
McGraw 矩修正算法优化版。
- 强制保持 m0 (质量守恒) 不变。
- 增加迭代次数以提高在实现性边界附近的表现。
"""
function mcgraw_correction(m::AbstractVector{T}; max_iter=20, tol=1e-10) where T
    L = length(m)
    if L < 4
        return m
    end
    
    m_curr = collect(m)
    
    # 记录原始 m0 以防数值漂移
    m0_orig = m_curr[1]
    
    for iter in 1:max_iter
        if is_realizable(m_curr)
            # 确保 m0 绝对一致
            m_curr[1] = m0_orig
            return m_curr
        end
        
        ln_m = log.(max.(m_curr, 1e-30))
        d3 = diff(diff(diff(ln_m)))
        
        best_k = -1
        max_cos2 = -1.0
        best_ln_ck = 0.0
        
        # 排除 k=1 (即 m0)，从 k=2 开始修正
        for k in 2:L
            bk = _compute_d3_response(L, k)
            dot_val = dot(d3, bk)
            bk_norm2 = dot(bk, bk)
            
            if bk_norm2 > 1e-15
                cos2 = (dot_val^2) / (max(1e-15, dot(d3, d3)) * bk_norm2)
                if cos2 > max_cos2
                    max_cos2 = cos2
                    best_k = k
                    best_ln_ck = - dot_val / bk_norm2
                end
            end
        end
        
        if best_k != -1
            # 应用欠松弛以增强稳定性 (书中提到可使用松弛)
            m_curr[best_k] *= exp(0.8 * best_ln_ck)
        else
            break
        end
    end
    
    # 如果 McGraw 失败，作为最后的兜底，尝试 Wright 修正 (即强制 Log-normal)
    if !is_realizable(m_curr)
        # 简单的 Wright 兜底：使用 m0, m1, m2 构造 Log-normal 并填充后续矩
        return _wright_fallback(m_curr)
    end
    
    return m_curr
end

function _compute_d3_response(L, k)
    e = zeros(L)
    e[k] = 1.0
    return diff(diff(diff(e)))
end

function _wright_fallback(m::AbstractVector{T}) where T
    L = length(m)
    m_res = copy(m)
    m0, m1, m2 = m[1], m[2], m[3]
    
    # 估计 log-normal 参数
    # λk = m0 * exp(k*mu + k^2*sigma^2/2)
    # log(m1/m0) = mu + sigma^2/2
    # log(m2/m0) = 2*mu + 2*sigma^2
    # -> sigma^2 = log(m2*m0 / m1^2)
    # -> mu = log(m1/m0) - sigma^2/2
    
    var_term = m2*m0 / (m1^2)
    if var_term <= 1.0
        # 如果 var <= 0，退化为 Delta 函数
        for k in 1:L-1
            m_res[k+1] = m0 * (m1/m0)^k
        end
    else
        sig2 = log(var_term)
        mu = log(m1/m0) - 0.5*sig2
        for k in 3:L-1 # 保持 m0, m1, m2，修正 m3 及以后
            m_res[k+1] = m0 * exp(k*mu + 0.5*(k^2)*sig2)
        end
    end
    return m_res
end

function mcgraw_correction(m::SVector{L, T}; max_iter=20) where {L, T}
    return SVector{L, T}(mcgraw_correction(collect(m); max_iter=max_iter))
end
