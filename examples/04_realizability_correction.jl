# examples/04_realizability_correction.jl
using QBMM
using StaticArrays

println("=== 模块二：鲁棒性与异常处理 ===")
println("4. 矩可实现性与 McGraw 修正
")

m_good = @SVector [1.0, 5.0, 26.0, 140.0]
println("原始健康序列 m: ", m_good)
println("可实现性检查: ", is_realizable(m_good, domain=:pos))
println()

# 污染数据
m_bad = @SVector [1.0, 5.0, 26.0, 101.0]
println("受污染序列 m_bad (m3 改为 101.0): ", m_bad)
println("可实现性检查: ", is_realizable(m_bad, domain=:pos))
println()

# 执行 McGraw 修正
m_corr = mcgraw_correction(m_bad)
println("McGraw 修正后的序列: ", m_corr)
println("可实现性检查: ", is_realizable(m_corr, domain=:pos))
