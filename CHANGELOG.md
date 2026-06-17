# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Analysis` submodule with zero-dependency post-processing tools: `reconstruct_ndf` (NDF from EQMOM/ECQMOM results via Gaussian, Gamma, and Beta kernels), `MomentComparison` + `compare_moments`, `verify_reconstruction`, and error-metric accessors (`abs_errors`, `rel_errors`, `max_abs_errors`, `max_rel_errors`, `verify`)
- `QuadratureMomentsPlotsExt` Package Extension: `plot_pbe_summary`, `plot_moment_evolution`, `plot_ndf_snapshots`, `plot_quadrature_nodes`, `plot_ndf_reconstruction`, `plot_moment_comparison` (activates automatically when users load Plots.jl)
- Unit tests for the Analysis core and conditional tests for the Plots extension
- Shared reporting utilities (`print_comparison_table`, `print_verification_banner`, `output_path`) for book verification examples

### Changed

- Moved `Plots` from a hard dependency to a weak dependency, declared via the `QuadratureMomentsPlotsExt` extension; users without Plots installed are unaffected
- Added `SpecialFunctions` dependency for kernel PDF evaluation
- Refactored five book verification examples (Ch07 ×3, Ch03 ×2) to use the new Analysis API, eliminating ~200 lines of duplicated plotting/post-processing code

## [0.1.0] - 2026-05-20

### Added

- Wheeler algorithm for univariate moment inversion with StaticArrays optimization
- Moment realizability check via Hankel matrix eigen-decomposition
- 2D CQMOM with recursive moment inversion and Vandermonde system solver
- EQMOM kernels (Gaussian, Gamma, Beta) and DQMOM solvers
- Recursive CQMOM, PD algorithm, and extended EQMOM kernels
- Tensor-product QMOM with zero-allocation StaticArrays design
- Extended Conditional QMOM (ECQMOM)
- Brute-force QMOM with Automatic Differentiation (ForwardDiff)
- Adaptive Wheeler inversion to handle degenerate cases
- McGraw moment correction and enhanced realizability checks
- SourceTerms architecture for physical population balance closures
- Breakage source term with analytical test cases
- Nucleation and deposition source terms
- Particle shrinkage source term for evaporation/dissolution
- Performance benchmark suite covering all inversion algorithms

### Changed

- Refactored to layered architecture with unified high-performance API
- Optimized EQMOM and math utilities into zero-allocation hot paths
- Translated all Chinese comments and strings to English

### Fixed

- CI pipeline failures resolved; test matrix simplified for reliability
- JuliaFormatter compatibility fixed and blue style applied across codebase

[unreleased]: https://github.com/VANvonZHANG/QuadratureMoments.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/VANvonZHANG/QuadratureMoments.jl/releases/tag/v0.1.0
