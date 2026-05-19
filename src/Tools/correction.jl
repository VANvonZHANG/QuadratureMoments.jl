# QBMM.jl/src/Tools/correction.jl

using LinearAlgebra
using StaticArrays
using ..QBMM: AbstractMathBackend, NativeBackend, ExternalBackend, is_realizable

raw"""
    mcgraw_correction(m::AbstractVector; max_iter=20, backend=NativeBackend()) -> corrected_m

Correct a moment sequence \$m_0, m_1, \\dots, m_L\$ to a realizable region.

Uses the McGraw algorithm to maximize the smoothness of \$\\ln(m_k)\$ by 
iteratively adjusting moments that contribute most to non-realizability. 
Falls back to the `wright_fallback` algorithm if the sequence remains 
non-realizable after `max_iter`.

# Arguments
- `m::AbstractVector`: Input moment sequence.
- `max_iter::Int`: Maximum iterations for the smoothing loop (default 20).
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A realizable moment sequence of the same length and type.
raw"""
function mcgraw_correction(
    m::AbstractVector{T}; max_iter=20, backend::AbstractMathBackend=NativeBackend()
) where {T}
    L = length(m)
    if L < 4
        return m
    end

    # Dynamic version dispatch
    m_curr = (backend isa NativeBackend) ? MVector{L,T}(m) : collect(m)
    m0_orig = m_curr[1]

    for iter in 1:max_iter
        if is_realizable(m_curr; backend=backend)
            m_curr[1] = m0_orig
            return (backend isa NativeBackend) ? SVector{L,T}(m_curr) : m_curr
        end

        # ln_m[k]
        ln_m = [log(max(m_curr[i], 1e-30)) for i in 1:L]
        # d3[j] = 3rd order difference
        d3 = [ln_m[j + 3] - 3 * ln_m[j + 2] + 3 * ln_m[j + 1] - ln_m[j] for j in 1:(L - 3)]

        sum_d3_2 = dot(d3, d3)
        if sum_d3_2 < 1e-15
            break
        end

        best_k = -1
        max_cos2 = -1.0
        best_ln_ck = 0.0

        # Optimization: Find the index k that is "most responsible" for the non-smoothness
        for k in 2:L
            dot_val = 0.0
            bk_norm2 = 0.0
            # Coefficients from the 3rd-order difference operator
            # j=k-3, k-2, k-1, k
            for (idx, coeff) in enumerate((1.0, -3.0, 3.0, -1.0))
                j = (k - 3) + idx - 1
                if 1 <= j <= (L - 3)
                    dot_val += d3[j] * coeff
                    bk_norm2 += coeff^2
                end
            end

            if bk_norm2 > 1e-15
                cos2 = (dot_val^2) / (sum_d3_2 * bk_norm2)
                if cos2 > max_cos2
                    max_cos2 = cos2
                    best_k = k
                    best_ln_ck = -dot_val / bk_norm2
                end
            end
        end

        if best_k != -1
            m_curr[best_k] *= exp(0.8 * best_ln_ck)
        else
            break
        end
    end

    # Final fallback if McGraw fails to reach the realizable region
    if !is_realizable(m_curr; backend=backend)
        return wright_fallback(m_curr, backend)
    end

    return (backend isa NativeBackend) ? SVector{L,T}(m_curr) : m_curr
end

raw"""
    mcgraw_correction(m::StaticVector; max_iter=20, backend=NativeBackend())
raw"""
@inline function mcgraw_correction(
    m::StaticVector{L,T}; max_iter=20, backend::AbstractMathBackend=NativeBackend()
) where {L,T}
    if L < 4
        return m
    end

    m_curr = MVector{L,T}(m)
    m0_orig = m_curr[1]

    for iter in 1:max_iter
        if is_realizable(m_curr; backend=backend)
            m_curr[1] = m0_orig
            return SVector{L,T}(m_curr)
        end

        # Zero-allocation ln_m and d3 using ntuple
        ln_m = ntuple(i -> log(max(m_curr[i], 1e-30)), Val(L))
        d3 = ntuple(
            j -> ln_m[j + 3] - 3 * ln_m[j + 2] + 3 * ln_m[j + 1] - ln_m[j], Val(L - 3)
        )

        sum_d3_2 = 0.0
        for j in 1:(L - 3)
            sum_d3_2 += d3[j]^2
        end

        if sum_d3_2 < 1e-15
            break
        end

        best_k = -1
        max_cos2 = -1.0
        best_ln_ck = 0.0

        for k in 2:L
            dot_val = 0.0
            bk_norm2 = 0.0

            # Unrolled 3rd-order diff contribution
            # j=k-3
            j1 = k - 3
            if 1 <= j1 <= (L - 3)
                dot_val += d3[j1] * 1.0
                bk_norm2 += 1.0
            end
            # j=k-2
            j2 = k - 2
            if 1 <= j2 <= (L - 3)
                dot_val += d3[j2] * (-3.0)
                bk_norm2 += 9.0
            end
            # j=k-1
            j3 = k - 1
            if 1 <= j3 <= (L - 3)
                dot_val += d3[j3] * 3.0
                bk_norm2 += 9.0
            end
            # j=k
            j4 = k
            if 1 <= j4 <= (L - 3)
                dot_val += d3[j4] * (-1.0)
                bk_norm2 += 1.0
            end

            if bk_norm2 > 1e-15
                cos2 = (dot_val^2) / (sum_d3_2 * bk_norm2)
                if cos2 > max_cos2
                    max_cos2 = cos2
                    best_k = k
                    best_ln_ck = -dot_val / bk_norm2
                end
            end
        end

        if best_k != -1
            m_curr[best_k] *= exp(0.8 * best_ln_ck)
        else
            break
        end
    end

    if !is_realizable(m_curr; backend=backend)
        return wright_fallback(m_curr, backend)
    end
    return SVector{L,T}(m_curr)
end

raw"""
    wright_fallback(m, backend)

Reconstruct moments using a log-normal distribution (Wright algorithm).

This fallback is triggered when iterative optimization fails to restore 
realizability. It uses the first three moments to build a log-normal 
distribution and calculates higher-order moments from it.

# Arguments
- `m`: Input moment sequence.
- `backend`: Backend to use for computation.
raw"""
function wright_fallback(m::AbstractVector{T}, ::ExternalBackend) where {T}
    L = length(m)
    m_res = copy(m)
    m0, m1, m2 = m[1], m[2], m[3]
    var_term = m2 * m0 / (m1^2)

    if var_term <= 1.0
        # Fallback to monodisperse
        for k in 1:(L - 1)
            m_res[k + 1] = m0 * (m1 / m0)^k
        end
    else
        # Log-normal reconstruction
        sig2 = log(var_term)
        mu = log(m1 / m0) - 0.5 * sig2
        for k in 3:(L - 1) # Keep m0, m1, m2
            m_res[k + 1] = m0 * exp(k * mu + 0.5 * (k^2) * sig2)
        end
    end
    return m_res
end

@inline function wright_fallback(m::StaticVector{L,T}, ::NativeBackend) where {L,T}
    m0, m1, m2 = m[1], m[2], m[3]
    var_term = m2 * m0 / (m1^2)
    if var_term <= 1.0
        return SVector{L,T}(ntuple(k -> m0 * (m1 / m0)^(k - 1), Val(L)))
    else
        sig2 = log(var_term)
        mu = log(m1 / m0) - 0.5 * sig2
        return SVector{L,T}(
            ntuple(
                k -> k <= 3 ? m[k] : m0 * exp((k - 1) * mu + 0.5 * ((k - 1)^2) * sig2),
                Val(L),
            ),
        )
    end
end

@inline function wright_fallback(m::MVector{L,T}, backend::NativeBackend) where {L,T}
    return wright_fallback(SVector{L,T}(m), backend)
end
