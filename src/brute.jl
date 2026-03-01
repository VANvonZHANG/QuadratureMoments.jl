using LinearAlgebra
using StaticArrays
using ForwardDiff

"""
    BruteQMOM{D, N}
    
直接法矩反演 (Brute-force QMOM)。
通过 Newton-Raphson 直接解非线性方程组获取节点和权重。
D: 维度 (Integer)。
N: 节点总数 (Integer)。
"""
struct BruteQMOM{D, N} <: AbstractQBMM end

# 构造函数
BruteQMOM(D::Int, N::Int) = BruteQMOM{D, N}()

"""
    invert_moments(method::BruteQMOM{D, N}, moments::SArray; max_iter=20, tol=1e-10)
    
Brute-force QMOM 入口。
变量排布 x = [w1, ..., wN, xi_1,1, ..., xi_D,N]
总变量数 P = N * (D + 1)
"""
function invert_moments(::BruteQMOM{D, N}, m::SArray{S, T, D}; max_iter=20, tol=1e-10) where {D, N, S, T}
    P = N * (D + 1)
    
    # 1. 初始猜测 (Initial Guess)
    # 这里可以使用张量积 QMOM 或简单的随机/均匀分布作为起点
    # 为了演示，我们使用一个简单的启发式起点
    x0_data = MVector{P, T}(undef)
    # 权重初始化为 m0 / N
    w0 = m[ntuple(i->1, Val(D))...] / N
    for i in 1:N x0_data[i] = w0 end
    # 节点坐标初始化为均值附近的微扰
    for i in 1:N
        for d in 1:D
            x0_data[N + (d-1)*N + i] = (m[ntuple(j->(j==d ? 2 : 1), Val(D))...] / m[ntuple(j->1, Val(D))...]) * (0.8 + 0.4*i/N)
        end
    end
    x = SVector{P, T}(x0_data)

    # 2. 定义残差函数 f(x)
    # 我们需要匹配 P 个矩。通常选择低阶矩组合。
    # 这里我们预先选定 P 个矩索引 (可以使用动态生成的，但为了性能固定一套策略)
    # 简单策略：按线性索引顺序取前 P 个矩
    S_tuple = ntuple(i -> S.parameters[i], Val(D))
    mom_indices = CartesianIndices(S_tuple)[1:P]
    
    function residual(curr_x::AbstractVector{V}) where V
        res = MVector{P, V}(undef)
        # 提取权重和节点
        ws = curr_x[1:N]
        # nodes[d, alpha]
        nodes = reshape(curr_x[N+1:P], N, D)
        
        for p in 1:P
            idx = mom_indices[p]
            target = m[idx]
            # 计算当前 x 对应的矩 m_idx = sum w * prod(xi^k)
            calc = zero(V)
            for alpha in 1:N
                term = ws[alpha]
                for d in 1:D
                    k = idx.I[d] - 1
                    term *= (nodes[alpha, d]^k)
                end
                calc += term
            end
            res[p] = calc - target
        end
        return SVector{P, V}(res)
    end

    # 3. Newton-Raphson 迭代
    for iter in 1:max_iter
        # 计算残差和 Jacobian
        r = residual(x)
        if norm(r) < tol
            break
        end
        
        # 利用 ForwardDiff 计算 Jacobian
        # 对于 P <= 12 (N=4, D=2)，SMatrix 性能极佳
        J = ForwardDiff.jacobian(residual, x)
        
        # 步进：x = x - J \ r
        # 使用 \ 运算符（在 StaticArrays 下会自动生成高效代码）
        delta = J \ r
        x = x - delta
    end

    # 4. 整理结果并返回
    final_weights = x[1:N]
    final_nodes = reshape(x[N+1:P], N, D)' # 转置为 D x N 形式
    
    return SMatrix{D, N, T}(final_nodes), SVector{N, T}(final_weights)
end
