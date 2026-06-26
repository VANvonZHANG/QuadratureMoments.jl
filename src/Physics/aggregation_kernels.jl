# src/Physics/aggregation_kernels.jl
# Concrete aggregation-rate kernels Ka(xi, xj). Each is a callable value object.
# Coordinate-agnostic in signature (xi is the internal-coordinate value); the
# coord convention (MassBased/LengthBased) lives on the source term, not the kernel.

raw"""
    Constant{T}

Constant aggregation kernel `Ka(xi, xj) = beta0`. The simplest Marchisio & Fox baseline.
"""
struct Constant{T}
    beta0::T
end
(k::Constant)(xi, xj) = k.beta0

raw"""
    Sum{T}

Sum (additive) aggregation kernel `Ka(xi, xj) = beta0 * (xi + xj)`.
"""
struct Sum{T}
    beta0::T
end
(k::Sum)(xi, xj) = k.beta0 * (xi + xj)

raw"""
    Brownian{T}

Brownian aggregation kernel `Ka(xi, xj) = ko * (xi + xj)^2 / (xi * xj)`, where `ko`
folds the physical prefactor (e.g. `2 k_B T / (3 mu)`). Singular as `xi*xj -> 0`;
the source term's zero-weight degenerate-node guards handle collapsed nodes.
"""
struct Brownian{T}
    ko::T
end
(k::Brownian)(xi, xj) = k.ko * (xi + xj)^2 / (xi * xj)
