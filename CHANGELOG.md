# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[unreleased]: https://github.com/VANvonZHANG/QBMM.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/VANvonZHANG/QBMM.jl/releases/tag/v0.1.0
