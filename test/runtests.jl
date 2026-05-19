using Test
using QBMM

@testset "QBMM.jl" begin
    include("test_wheeler.jl")
    include("test_adaptive_wheeler.jl")
    include("test_pd.jl")
    include("test_realizability.jl")
    include("test_stirling.jl")
    include("test_cqmom.jl")
    include("test_ecqmom.jl")
    include("test_eqmom.jl")
    include("test_tensor.jl")
    include("test_brute.jl")
    include("test_dqmom.jl")
    include("test_dqmom_forwarddiff.jl")
    include("test_correction.jl")
    include("test_source_terms.jl")
end
