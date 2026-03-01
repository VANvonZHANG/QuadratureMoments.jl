using Test
using LinearAlgebra
using StaticArrays
using ForwardDiff
using QBMM

@testset "DQMOM ForwardDiff Integration" begin
    #=
    在这个测试中，我们将演示如何结合 DQMOM 和 ForwardDiff 来自动计算非线性系统源项的雅可比矩阵(Jacobian)。
    这对于 DQMOM 在 CFD 中使用全隐式时间推进（Implicit Time Integration）极其重要。
    
    假设状态变量向量为 v = [w_1, ..., w_N, ζ_1, ..., ζ_N]^T
    其中 ζ_i = w_i * ξ_i 为加权节点。
    =#
    
    N = 2
    # 定义物理模型源项生成器：纯生长模型 (Constant Growth Rate G = 1.5) 和 线性衰减项 (Death rate = 0.1)
    function compute_source_terms(w::SVector{2, T}, xi::SVector{2, T}) where {T}
        G = T(1.5)
        D = T(0.1)
        
        S_elements = zeros(MVector{4, T})
        for k in 0:3
            sum_val = zero(T)
            for i in 1:2
                # 衰减项: -D * w_i * xi_i^k
                term1 = -D * w[i] * (xi[i]^k)
                
                # 生长项: k * w_i * xi_i^(k-1) * G
                term2 = zero(T)
                if k > 0
                    term2 = k * w[i] * (xi[i]^(k-1)) * G
                end
                
                sum_val += term1 + term2
            end
            S_elements[k+1] = sum_val
        end
        return SVector{4, T}(S_elements)
    end
    
    # 定义 DQMOM 右端项 F(v) = dv/dt = [dw/dt; dζ/dt] = [a; b]
    function dqmom_rhs(v::SVector{4, T}) where {T}
        # 提取权重 w 和加权节点 ζ
        w = SVector{2, T}(v[1:2])
        zeta = SVector{2, T}(v[3:4])
        
        # 计算原始节点 ξ
        xi = zeta ./ w
        
        # 获取物理源项 S
        S = compute_source_terms(w, xi)
        
        # 求解 DQMOM 线性系统获取演化率 a 和 b
        a, b = dqmom_solve(xi, S)
        
        return vcat(a, b)
    end
    
    # 构造一个初始状态 v0
    # 假设权重 w = [0.4, 0.6], 节点 xi = [1.0, 2.0] -> 加权节点 zeta = [0.4, 1.2]
    v0 = @SVector [0.4, 0.6, 0.4, 1.2]
    
    # 1. 计算当前的演化率 dv/dt
    dvdt = dqmom_rhs(v0)
    @test length(dvdt) == 4
    @test eltype(dvdt) == Float64
    
    # 2. 使用 ForwardDiff 计算 dv/dt 相对 v 的雅可比矩阵 J = dF / dv
    # 因为 dqmom_rhs 中包含了节点求解、矩阵拼装、矩阵求逆（\）等全部过程
    # ForwardDiff 能够利用 Dual numbers 穿透追踪所有这些操作的导数。
    J = ForwardDiff.jacobian(dqmom_rhs, v0)
    
    @test size(J) == (4, 4)
    @test eltype(J) == Float64
    
    # 我们也可以使用有限差分（Finite Difference）来验证 ForwardDiff 的结果
    epsilon = 1e-7
    J_fd = zeros(4, 4)
    for j in 1:4
        v_plus = SVector{4, Float64}(v0 .+ epsilon .* (1:4 .== j))
        v_minus = SVector{4, Float64}(v0 .- epsilon .* (1:4 .== j))
        J_fd[:, j] .= (dqmom_rhs(v_plus) .- dqmom_rhs(v_minus)) ./ (2 * epsilon)
    end
    
    # 检查 ForwardDiff 计算的雅可比矩阵与有限差分结果是否高度吻合
    @test J ≈ J_fd rtol=1e-5
end
