using Test
using QBMM

@testset "QBMM.jl" begin
    include("test_wheeler.jl")
    include("test_realizability.jl")
end
