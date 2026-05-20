# src/Solvers/MultiD/ecqmom.jl

using LinearAlgebra
using StaticArrays
using ..QuadratureMoments:
    AbstractQBMM, AbstractMathBackend, NativeBackend, ExternalBackend, QuadratureResult
using ..QuadratureMoments: AbstractKernel, GaussianKernel, GammaKernel, BetaKernel
using ..QuadratureMoments: EQMOM, solve_vandermonde_transpose, compute_modified_moments

raw"""
    ECQMOM{D, N, NT, K} <: AbstractQBMM{D, NT}

Extended Conditional Quadrature-Based Moment Method.

A multivariate solver combining the continuous kernel approach of EQMOM 
with the recursive conditional decomposition of CQMOM.

# Type Parameters
- `D::Int`: Dimensions of the coordinate space.
- `N::NTuple{D, Int}`: Nodes per dimension.
- `NT::Int`: Total number of resulting nodes (`prod(N)`).
- `K<:AbstractKernel`: The continuous kernel type.
raw"""
struct ECQMOM{D,N,NT,K<:AbstractKernel} <: AbstractQBMM{D,NT}
    kernel::K
end

raw"""
    ECQMOM(N::NTuple{D, Int}, kernel=GaussianKernel())
    ECQMOM(D::Int, N_per_dim::Int, kernel=GaussianKernel())

Constructors for the ECQMOM solver.
raw"""
function ECQMOM(N::NTuple{D,Int}, kernel::K=GaussianKernel()) where {D,K<:AbstractKernel}
    return ECQMOM{D,N,prod(N),K}(kernel)
end

function ECQMOM(
    D::Int, N_per_dim::Int, kernel::K=GaussianKernel()
) where {K<:AbstractKernel}
    N = ntuple(_ -> N_per_dim, D)
    return ECQMOM(N, kernel)
end

raw"""
    invert_moments(method::ECQMOM, m; backend=NativeBackend()) -> QuadratureResult

Perform multivariate ECQMOM inversion.

# Arguments
- `method::ECQMOM`: The ECQMOM solver instance.
- `m::SArray`: The multi-dimensional moment tensor.
- `backend`: `NativeBackend()` or `ExternalBackend()`.

# Returns
- A `QuadratureResult` containing weights, \$D\$-dimensional primary nodes, and \$\\sigma\$ parameters.
raw"""
function invert_moments(
    method::ECQMOM{D,N,NT,K},
    m::SArray{S,T,D,L};
    backend::AbstractMathBackend=NativeBackend(),
) where {D,N,NT,K,S,T,L}

    # Run recursive deconvolution
    res_tuple = _ecqmom_recursive(Val(D), N, method.kernel, m, backend)

    return QuadratureResult(
        SVector{NT,T}(res_tuple[2]),
        SMatrix{NT,D,T}(res_tuple[1]'),
        SMatrix{NT,D,T}(res_tuple[3]'),
    )
end

# --- Base Case: D = 1 ---
function _ecqmom_recursive(
    ::Val{1},
    N_tuple::NTuple{1,Int},
    kernel::K,
    m::SVector{L,T},
    backend::AbstractMathBackend,
) where {K,L,T}
    N1 = N_tuple[1]
    res = invert_moments(EQMOM(N1, kernel), m; backend=backend)

    return MMatrix{1,N1,T}(res.nodes'),
    MVector{N1,T}(res.weights),
    MMatrix{1,N1,T}(res.sigmas')
end

# --- Recursive Step: D > 1 ---
function _ecqmom_recursive(
    ::Val{D},
    N_tuple::NTuple{D,Int},
    kernel::K,
    m::SArray{S,T,D,L},
    backend::AbstractMathBackend,
) where {D,K,S,T,L}
    N1 = N_tuple[1]
    L1 = 2 * N1 + 1 # Moments needed for 1D inversion

    # 1. Marginal Inversion (Dimension 1)
    m1_vec = SVector{L1,T}(ntuple(k -> m[k, ntuple(_ -> 1, Val(D - 1))...], Val(L1)))

    q1 = invert_moments(EQMOM(N1, kernel), m1_vec; backend=backend)
    xi1 = SVector{N1,T}(ntuple(i -> q1.nodes[i, 1], Val(N1)))
    w1 = q1.weights
    σ1 = q1.sigmas[1, 1]

    # 2. Deconvolution & Conditional Inversion
    N_rem = ntuple(i -> N_tuple[i + 1], Val(D - 1))
    NT_rem = prod(N_rem)
    S_rem = ntuple(i -> S.parameters[i + 1], Val(D - 1))
    L_rem = prod(S_rem)

    all_res = ntuple(Val(N1)) do alpha
        cond_m_data = ntuple(Val(L_rem)) do idx
            coords_rem = Tuple(CartesianIndices(S_rem)[idx])
            b_vec = SVector{N1,T}(ntuple(k -> m[k, coords_rem...], Val(N1)))

            # 1D Modified Moment deconvolution
            m_star = compute_modified_moments(b_vec, σ1, kernel, backend)

            c_vals = solve_vandermonde_transpose(xi1, m_star, backend)
            return c_vals[alpha] / w1[alpha]
        end

        cond_tensor = SArray{Tuple{S_rem...},T,D - 1,L_rem}(cond_m_data)
        return _ecqmom_recursive(Val(D - 1), N_rem, kernel, cond_tensor, backend)
    end

    # 3. Assemble results
    NT_total = N1 * NT_rem
    final_nodes = MMatrix{D,NT_total,T}(undef)
    final_weights = MVector{NT_total,T}(undef)
    final_sigmas = MMatrix{D,NT_total,T}(undef)

    for alpha in 1:N1
        nodes_rem, weights_rem, sigmas_rem = all_res[alpha]
        for beta in 1:NT_rem
            idx = (alpha - 1) * NT_rem + beta

            final_nodes[1, idx] = xi1[alpha]
            final_sigmas[1, idx] = σ1
            final_weights[idx] = w1[alpha] * weights_rem[beta]

            for d in 2:D
                final_nodes[d, idx] = nodes_rem[d - 1, beta]
                final_sigmas[d, idx] = sigmas_rem[d - 1, beta]
            end
        end
    end

    return final_nodes, final_weights, final_sigmas
end
