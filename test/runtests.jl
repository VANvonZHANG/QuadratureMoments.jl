using Test
using QBMM

@testset "QBMM.jl" begin
    include("test_wheeler.jl")
    include("test_pd.jl")
    include("test_realizability.jl")
    include("test_cqmom.jl")
    include("test_eqmom.jl")
    include("test_dqmom.jl")
    include("test_dqmom_forwarddiff.jl")
end
