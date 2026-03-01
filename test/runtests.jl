using Test
using QBMM

@testset "QBMM.jl" begin
    include("test_wheeler.jl")
    include("test_pd.jl")
    include("test_realizability.jl")
    include("test_stirling.jl")
    include("test_cqmom.jl")
    include("test_ecqmom.jl")
    include("test_eqmom.jl")
    include("test_tensor.jl")
    include("test_brute.jl")
    include("test_dqmom.jl")
end
