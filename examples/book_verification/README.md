# Book Verification Examples

Verification examples reproduced from:
> Marchisio, D. L., & Fox, R. O. (2013). *Computational Models for Polydisperse Particulate and Multiphase Systems*. Cambridge University Press.

## Chapter 3: Algorithm Verification

| Script | Book Reference | Description |
|--------|---------------|-------------|
| `ch03/01_pd_gaussian.jl` | Exercise 3.1 | PD algorithm with Gaussian N(5,1) |
| `ch03/02_wheeler_zero_mean.jl` | Exercise 3.2 | Wheeler with Gaussian N(0,1), zero mean |
| `ch03/03_realizability_hankel.jl` | Exercise 3.3 | Hankel-Hadamard realizability check |
| `ch03/04_moment_correction.jl` | Exercises 3.4-3.5 | McGraw/Wright moment correction |
| `ch03/05_tensor_bivariate.jl` | Exercise 3.6 | Tensor-product QMOM, bivariate |
| `ch03/06_tensor_trivariate.jl` | Exercise 3.7 | Tensor-product QMOM, trivariate |
| `ch03/07_cqmom_bivariate.jl` | Exercise 3.8 | CQMOM, bivariate Gaussian |

## Chapter 7: Homogeneous PBE

| Script | Book Reference | Description |
|--------|---------------|-------------|
| `ch07/01_pure_aggregation.jl` | §7.4.1 | Constant-kernel aggregation, analytical solution |
| `ch07/02_pure_breakage.jl` | §7.4.1 | Constant-rate breakage, analytical solution |
| `ch07/03_growth_nucleation.jl` | §7.4.1 | Constant growth + nucleation |

## Running

```bash
# Chapter 3 examples (no extra dependencies)
julia examples/book_verification/ch03_algorithm_verification/01_pd_gaussian.jl

# Chapter 7 examples (requires OrdinaryDiffEq.jl)
using Pkg; Pkg.add("OrdinaryDiffEq")
julia examples/book_verification/ch07_homogeneous_pbe/01_pure_aggregation.jl
```
