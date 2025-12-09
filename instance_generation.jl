#=
module InstanceGen

using Random, LinearAlgebra
using JuMP, MosekTools
# import Pkg; Pkg.add("JSON")
using JSON  
# ---------------------------
# objective generator
# ---------------------------
"Return (Q0, q0) with entries ~ Uniform(-scale, scale). Q0 is symmetrized."
function rand_objective(n; seed::Int, scale::Real=100.0, integer::Bool=true)
    rng = MersenneTwister(seed)

    A  = rand(rng, n, n) .* (2*scale) .- scale
    Q0 = 0.5 .* (A .+ A')                 # symmetric Float64
    q0 = rand(rng, n) .* (2*scale) .- scale

    if integer
        return round.(Int, Q0), round.(Int, q0)
    else
        return Q0, q0
    end
end
# ---------------------------
# A that positively spans ℝⁿ and b = A*xstar + δ
# ---------------------------
"""
Construct A whose rows form a positive spanning set of ℝⁿ and b = A*xstar + δ.
"""
function build_bounded_A(; n::Integer, mode::Symbol=:minimal, seed::Union{Nothing,Int}=nothing)
    seed !== nothing && Random.seed!(seed)
    R = round.(Float64.(randn(n, n)); digits=1)
    V = Matrix(qr(R).Q)[:, 1:n]
    A = if mode === :minimal
        vcat(V', (-sum(V; dims=2))')
    elseif mode === :symmetric
        vcat(V', -V')
    else
        error("unknown mode = $mode (use :minimal, :symmetric)")
    end
    xstar = fill(1.0/n, n)
    δ = 1.0
    b = A * xstar .+ δ
    return A, b
end

# ---------------------------
# Tight bounds and Big-M via 2n LPs
# ---------------------------
function bounds_via_LP(A::AbstractMatrix, b::AbstractVector; H=nothing, h=nothing, optimizer=MosekTools.Optimizer)
    _, n = size(A)
    ℓ = zeros(n); u = zeros(n)
    for i in 1:n
        model = Model(optimizer); @variable(model, x[1:n])
        @constraint(model, A * x .<= b)
        if H !== nothing && h !== nothing; @constraint(model, H * x .== h); end
        @objective(model, Min, x[i]); optimize!(model); ℓ[i] = value(x[i])

        model = Model(optimizer); @variable(model, x2[1:n])
        @constraint(model, A * x2 .<= b)
        if H !== nothing && h !== nothing; @constraint(model, H * x2 .== h); end
        @objective(model, Max, x2[i]); optimize!(model); u[i] = value(x2[i])
    end
    M = Diagonal(max.(abs.(ℓ), abs.(u)))
    return ℓ, u, M
end

# ---------------------------
# Box and simplex generators
# ---------------------------
function box_instance(n; low=-10.0, high=10.0, min_width=0.5, seed=nothing)
    @assert low < high
    seed !== nothing && Random.seed!(seed)
    ℓ = low .+ (high - low - min_width) .* rand(n)
    u = ℓ  .+ min_width .+ (high .- (ℓ .+ min_width)) .* rand(n)
    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(u, -ℓ)
    M = Diagonal(max.(abs.(ℓ), abs.(u)))
    return A, b, M, ℓ, u
end

function simplex_instance(n)
    A = -Matrix{Float64}(I, n, n)
    b = zeros(n)
    H = ones(1, n)
    h = [1.0]
    M = I(n)
    return A, b, H, h, M
end

# ---------------------------
# Dict packer
# ---------------------------
function pack(; n, rho=3.0, Q0=nothing, q0=nothing,
               Qi=nothing, qi=nothing, ri=nothing,
               Pi=nothing, pi=nothing, si=nothing,
               A=nothing, b=nothing, H=nothing, h=nothing, M=I(n))
    return Dict(
        "n"=>n, "rho"=>rho,
        "Q0"=>Q0, "q0"=>q0,
        "Qi"=>Qi, "qi"=>qi, "ri"=>ri,
        "Pi"=>Pi, "pi"=>pi, "si"=>si,
        "A"=>A, "b"=>b, "H"=>H, "h"=>h,
        "M"=>M
    )
end

# ---------------------------
# Build 8 instances (now assigns Q0,q0 inside)
# ---------------------------
function build_instances(; objective_scale::Real=100.0, objective_seeds=1:8)
    insts = Vector{Dict}(undef, 8)

    # 1) Unit simplex (n=7) etx=1 x>=0
    n1, rho1 = 7, 4.0
    A1, b1, H1, h1, M1 = simplex_instance(n1)
    insts[1] = pack(; n=n1, rho=rho1, A=A1, b=b1, H=H1, h=h1, M=M1)

    # 2) Box (n=4)
    n2, rho2 = 4, 3.0
    A2, b2, M2, ℓ2, ū2 = box_instance(n2; seed=42)
    insts[2] = pack(; n=n2, rho=rho2, A=A2, b=b2, M=M2)

    # 3) Box + 2 equalities (consistent with midpoint)
    xmid3 = 0.5 .* (ℓ2 .+ ū2)
    H3 = randn(2, n2); h3 = vec(H3 * xmid3)
    insts[3] = pack(; n=n2, rho=rho2, A=A2, b=b2, H=H3, h=h3, M=M2)

    # 4) A-bounded (minimal) (n=5)
    n4a, rho4a = 5, 4.0
    A4a, b4a = build_bounded_A(n=n4a, mode=:minimal, seed=7)
    _, _, M4a = bounds_via_LP(A4a, b4a)
    insts[4] = pack(; n=n4a, rho=rho4a, A=A4a, b=b4a, M=M4a)

    # 5) A-bounded (symmetric) (n=4)
    n4b, rho4b = 4, 3.0
    A4b, b4b = build_bounded_A(n=n4b, mode=:symmetric, seed=7)
    _, _, M4b = bounds_via_LP(A4b, b4b)
    insts[5] = pack(; n=n4b, rho=rho4b, A=A4b, b=b4b, M=M4b)

    # 6) Box + one quadratic inequality
    Q6 = 0.5 .* (randn(n2, n2) .+ randn(n2, n2)') .* 0.5
    q6 = 0.2 .* randn(n2)
    xmid6 = 0.5 .* (ℓ2 .+ ū2)
    r6 = xmid6' * Q6 * xmid6 + dot(q6, xmid6) + 0.2
    insts[6] = pack(; n=n2, rho=rho2, A=A2, b=b2, Qi=(Q6,), qi=(q6,), ri=(r6,), M=M2)

    # 7) Box + one quadratic equality
    P7 = 0.5 .* (randn(n2, n2) .+ randn(n2, n2)') .* 0.5
    p7 = 0.2 .* randn(n2)
    s7 = xmid6' * P7 * xmid6 + dot(p7, xmid6)
    insts[7] = pack(; n=n2, rho=rho2, A=A2, b=b2, Pi=(P7,), pi=(p7,), si=(s7,), M=M2)

    # 8) Box + both + one linear equality
    Qqi1 = zeros(n2, n2); Qqi1[1,1]=2.0; Qqi1[2,2]=2.0; Qqi1[1,2]=-2.0; Qqi1[2,1]=-2.0
    qqi1 = zeros(n2); rqi1 = -0.5
    Qqi2, qqi2, rqi2 = Q6, q6, r6
    Pqe,  pqe,  sqe  = P7, p7, s7
    H8 = [1.0 -1.0 0.0 0.0]; h8 = [0.0]
    insts[8] = pack(; n=n2, rho=rho2, A=A2, b=b2,
                     Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                     Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                     H=H8, h=h8, M=M2)

    # --- assign Q0, q0 here ---
    @assert length(objective_seeds) ≥ length(insts) "need ≥ one seed per instance"
    for (i, inst) in enumerate(insts)
        n = inst["n"]
        Q0, q0 = rand_objective(n; seed=objective_seeds[i], scale=objective_scale, integer=true)
        inst["Q0"] = Q0
        inst["q0"] = q0
    end

    return insts
end
instances = build_instances(objective_scale=100.0, objective_seeds=1:8)
instances[1]["Q0"]


"Save a Vector{Dict} of instances to a pretty JSON file."
function save_instances_json(path::AbstractString, insts::Vector{Dict})
    open(path, "w") do io
        JSON.print(io, insts, 2)  # 2 = pretty indent
    end
end

"Load instances back from JSON."
function load_instances_json(path::AbstractString)::Vector{Dict{String,Any}}
    return JSON.parsefile(path)   # matrices come back as arrays-of-arrays (ok)
end

# Example
instances = InstanceGen.build_instances(; objective_seeds=1:8, objective_scale=10.0)
save_instances_json("instances.json", instances)
instances2 = load_instances_json("instances.json")

end # module

=#








module InstanceGen

using Random, LinearAlgebra
using JuMP, MosekTools
using JSON

const MOI = JuMP.MOI

# ---------------------------
# Integer / basic helpers
# ---------------------------
rand_vec_int(rng::AbstractRNG, n::Int; vals=-10:10) = rand(rng, vals, n)

function symmetrize_int!(Q::AbstractMatrix)
    Q .= (Q .+ Q') .÷ 2  # integer division
    return Q
end

"Random symmetric integer matrix with even entries."
function rand_symm_int_even(rng::AbstractRNG, n::Int; vals=-10:10)
    Q = rand(rng, vals, n, n)
    symmetrize_int!(Q)
    return 2 .* Q
end

"Random positive integer slack δ ∈ [lo, hi]."
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

# ---------------------------
# Eigenvalue-controlled symmetric matrices
# ---------------------------
"""
rand_symm_from_eigs(rng, n; num_neg, nonneg_vals, neg_vals)

Builds a symmetric matrix Q = Qmat * Diag(eigs) * Qmat'
with integer eigenvalues:
- num_neg entries drawn from neg_vals (default -10:-1),
- the rest from nonneg_vals (default 0:10).
"""
function rand_symm_from_eigs(rng::AbstractRNG, n::Int;
                             num_neg::Int=0,
                             nonneg_vals::UnitRange{Int}=0:10,
                             neg_vals::UnitRange{Int}=-10:-1)
    @assert 0 ≤ num_neg ≤ n
    num_nonneg = n - num_neg

    negs    = num_neg == 0     ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0  ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R        = randn(rng, n, n)
    Qfac, _  = qr(R)
    Qmat     = Matrix(Qfac)
    Λ        = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

"""
rand_objective(n; rng, num_neg, ...)

Objective matrix Q0 with integer eigenvalues, num_neg of them negative.
q0 is an integer vector.
"""
function rand_objective(n::Int; rng::AbstractRNG,
                        num_neg::Int=0,
                        nonneg_vals::UnitRange{Int}=0:10,
                        neg_vals::UnitRange{Int}=-10:-1,
                        qvals=-10:10)
    Q0 = rand_symm_from_eigs(rng, n;
                             num_neg=num_neg,
                             nonneg_vals=nonneg_vals,
                             neg_vals=neg_vals)
    q0 = rand_vec_int(rng, n; vals=qvals)
    return Q0, q0
end

# ---------------------------
# A that positively spans R^n
# ---------------------------
function build_bounded_A(; n::Int, mode::Symbol=:minimal, rng::AbstractRNG)
    if mode === :minimal
        A = vcat(Matrix{Int}(I, n, n), -ones(Int, 1, n))
    elseif mode === :symmetric
        A = vcat(Matrix{Int}(I, n, n), -Matrix{Int}(I, n, n))
    else
        error("unknown mode = $mode (use :minimal or :symmetric)")
    end
    xhat = rand_vec_int(rng, n; vals=-2:2)
    δ    = rand_slack(rng)
    b    = A * xhat .+ δ
    return A, b, collect(xhat)
end

# ---------------------------
# Bounds via LP
# ---------------------------
function bounds_via_LP(A::AbstractMatrix, b::AbstractVector;
                       H::Union{Nothing,AbstractMatrix}=nothing,
                       h::Union{Nothing,AbstractVector}=nothing,
                       optimizer = MosekTools.Optimizer)
    _, n = size(A)
    ℓ = zeros(Float64, n)
    u = zeros(Float64, n)
    for i in 1:n
        # lower bound
        model = Model(optimizer)
        @variable(model, x[1:n])
        @constraint(model, A * x .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x .== h)
        end
        @objective(model, Min, x[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP lower bound not optimal"
        ℓ[i] = value.(x)[i]

        # upper bound
        model = Model(optimizer)
        @variable(model, x2[1:n])
        @constraint(model, A * x2 .<= b)
        if H !== nothing && h !== nothing
            @constraint(model, H * x2 .== h)
        end
        @objective(model, Max, x2[i])
        optimize!(model)
        @assert termination_status(model) == MOI.OPTIMAL "LP upper bound not optimal"
        u[i] = value.(x2)[i]
    end
    return ℓ, u
end

# ---------------------------
# Big-M from bounds
# ---------------------------
"""
bigM_from_bounds(ℓ, u; scale, common)

If common=true:
    returns a single vector M with M_i ≥ max(|ℓ_i|, |u_i|), scaled by 'scale'.

If common=false:
    returns (Mminus, Mplus) with:
        Mminus_i ≥ max(0, -ℓ_i), Mplus_i ≥ max(0, u_i),
    both scaled by 'scale'.
"""
function bigM_from_bounds(ℓ::AbstractVector, u::AbstractVector;
                          scale::Real = 1.0,
                          common::Bool = true)
    @assert length(ℓ) == length(u)
    @assert scale ≥ 1.0 "scale should be ≥ 1.0 to keep Big-M safe"

    if common
        base = max.(abs.(ℓ), abs.(u))
        return scale .* base
    else
        Mminus = scale .* max.(0.0, -ℓ)
        Mplus  = scale .* max.(0.0,  u)
        return Mminus, Mplus
    end
end

# ---------------------------
# Simplex and box instances
# ---------------------------
"""
Simplex:  x ≥ 0,  eᵀx = 1.

Returns (A, b, H, h, M) with:
- A x ≤ b   encoding x ≥ 0 (so A = -I, b = 0),
- H x = h   encoding eᵀx = 1,
- M        from LP-based bounds.
"""
function simplex_instance(n::Int; optimizer=MosekTools.Optimizer)
    A = -Matrix{Float64}(I, n, n)
    b = zeros(Float64, n)
    H = ones(Float64, 1, n)
    h = [1.0]
    ℓ, u = bounds_via_LP(A, b; H=H, h=h, optimizer=optimizer)
    M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)
    return A, b, H, h, M
end

"""
Box: ℓ ≤ x ≤ u with integer bounds.

Returns (A, b, M, ℓ, u, xhat) where:
- A x ≤ b encodes ℓ ≤ x ≤ u,
- M is a Big-M vector from bounds,
- xhat is a random integer point in the box.
"""
function box_instance(n::Int; rng::AbstractRNG, low::Int=-8, high::Int=8, min_width::Int=3)
    @assert low < high
    ℓ = zeros(Int, n)
    u = zeros(Int, n)
    for i in 1:n
        a = rand(rng, low:(high-min_width))
        w = rand(rng, min_width:(high-a))
        ℓ[i] = a
        u[i] = a + w
    end
    A = vcat(Matrix{Int}(I, n, n), -Matrix{Int}(I, n, n))
    b = vcat(u, -ℓ)
    ℓf = Float64.(ℓ)
    uf = Float64.(u)
    M  = bigM_from_bounds(ℓf, uf; scale=1.0, common=true)
    xhat = [rand(rng, ℓ[i]:u[i]) for i in 1:n]
    return A, b, M, ℓ, u, xhat
end

# ---------------------------
# Quadratic rows with anchor
# ---------------------------
"""
Quadratic inequality:

    0.5 xᵀ Q x + qᵀ x + r ≤ 0

constructed so that g(x̄) = -δ < 0 at the anchor x̄, with δ > 0.

num_neg controls the number of negative eigenvalues of Q.
If num_neg = 0, Q is PSD (inequality convex in x).
"""
function make_qineq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector{<:Integer};
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:10,
    neg_vals::UnitRange{Int} = -10:-1,
    qvals::UnitRange{Int} = -10:10,
    slack_lo::Int = 1,
    slack_hi::Int = 5,
)
    n = length(xbar)

    Q = rand_symm_from_eigs(rng, n;
                            num_neg     = num_neg,
                            nonneg_vals = nonneg_vals,
                            neg_vals    = neg_vals)
    q = rand_vec_int(rng, n; vals = qvals)

    g0 = 0.5 * (xbar' * Q * xbar) + dot(q, xbar)
    δ  = rand_slack(rng; lo = slack_lo, hi = slack_hi)

    r = -g0 - δ

    return Matrix{Float64}(Q), Float64.(q), Float64(r)
end

"""
Quadratic equality:

    0.5 xᵀ P x + pᵀ x + s = 0

constructed so that g(x̄) = 0 at the anchor x̄.

Any genuinely quadratic equality (with P ≠ 0) defines a nonconvex feasible set.
"""
function make_qeq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector{<:Integer};
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:10,
    neg_vals::UnitRange{Int} = -10:-1,
    pvals::UnitRange{Int} = -3:3,
)
    n = length(xbar)

    P = rand_symm_from_eigs(rng, n;
                            num_neg     = num_neg,
                            nonneg_vals = nonneg_vals,
                            neg_vals    = neg_vals)
    p = rand_vec_int(rng, n; vals = pvals)

    g0 = 0.5 * (xbar' * P * xbar) + dot(p, xbar)
    s  = -g0

    return Matrix{Float64}(P), Float64.(p), Float64(s)
end

# ---------------------------
# Dict packer
# ---------------------------
function pack(; n::Int, rho::Real=3.0, Q0=nothing, q0=nothing,
               Qi=nothing, qi=nothing, ri=nothing,
               Pi=nothing, pi=nothing, si=nothing,
               A=nothing, b=nothing, H=nothing, h=nothing,
               M::Union{Nothing,AbstractVector}=nothing,
               seed::Union{Nothing,Int}=nothing,
               base_tag::AbstractString = "",
               tag::AbstractString = "CVX")
    if M === nothing
        M = ones(Float64, n)
    end
    d = Dict{String,Any}(
        "n"        => n,
        "rho"      => rho,
        "Q0"       => Q0,
        "q0"       => q0,
        "Qi"       => Qi,
        "qi"       => qi,
        "ri"       => ri,
        "Pi"       => Pi,
        "pi"       => pi,
        "si"       => si,
        "A"        => A,
        "b"        => b,
        "H"        => H,
        "h"        => h,
        "M"        => M,
        "base_tag" => String(base_tag),
        "tag"      => String(tag)
    )
    if seed !== nothing
        d["seed"] = seed
    end
    return d
end

# ---------------------------
# Constraint-type helpers
# ---------------------------
has_lin_ineq(d::Dict{String,Any}) = get(d, "A", nothing) !== nothing
has_lin_eq(d::Dict{String,Any})   = get(d, "H", nothing) !== nothing
has_qineq(d::Dict{String,Any})    = get(d, "Qi", nothing) !== nothing
has_qeq(d::Dict{String,Any})      = get(d, "Pi", nothing) !== nothing

"""
Fallback convexity check (not used for tagging, but kept for sanity checks).
"""
function convex_tag_eig(d::Dict{String,Any}; tol::Real=1e-6)
    Q0 = get(d, "Q0", nothing)
    Qi = get(d, "Qi", nothing)
    Pi = get(d, "Pi", nothing)

    if (Qi !== nothing && length(Qi) > 0) || (Pi !== nothing && length(Pi) > 0)
        return "NCVX"
    end
    Q0 === nothing && return "NCVX"

    vals = eigvals(Symmetric(Matrix(Q0)))
    return minimum(vals) ≥ -tol ? "CVX" : "NCVX"
end

"Build human-readable instance id string."
function make_id(d::Dict{String,Any})
    n    = Int(d["n"])
    rho  = Int(d["rho"])
    seed = get(d, "seed", "?")
    base = get(d, "base_tag", "")

    parts = String[]
    push!(parts, "n$(n)")
    push!(parts, "rho$(rho)")
    base != "" && push!(parts, String(base))

    has_lin_ineq(d) && push!(parts, "LI")
    has_lin_eq(d)   && push!(parts, "LE")
    has_qineq(d)    && push!(parts, "QI")
    has_qeq(d)      && push!(parts, "QE")

    tag = get(d, "tag", nothing)
    if tag === nothing
        tag = convex_tag_eig(d)
    end
    push!(parts, String(tag))
    push!(parts, "seed$(seed)")

    return join(parts, "_")
end

# ---------------------------
# Build a small test suite
# ---------------------------
function build_instances(; objective_scale::Real=10.0, optimizer=MosekTools.Optimizer)
    insts = Vector{Dict{String,Any}}(undef, 8)

    # 1) Simplex (n=7), convex objective (num_neg=0)
    let i = 1, rng = MersenneTwister(1)
        is_cvx = true
        n   = 7
        rho = 4.0
        A, b, H, h, M = simplex_instance(n; optimizer=optimizer)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="S", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 2) Box (n=4), convex objective (num_neg=0)
    let i = 2, rng = MersenneTwister(2)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 3) Box + 2 linear equalities, convex objective
    let i = 3, rng = MersenneTwister(3)
        is_cvx = true
        n   = 15
        rho = 10.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)
        H = rand(rng, -2:2, 2, n)
        h = H * xhat

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 4) A-bounded minimal (positive spanning), convex objective
    let i = 4, rng = MersenneTwister(4)
        is_cvx = true
        n   = 5
        rho = 4.0
        A, b, xhat = build_bounded_A(n=n, mode=:minimal, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)

        num_neg_q0 = 0
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 5) A-bounded symmetric, nonconvex objective (num_neg > 0)
    let i = 5, rng = MersenneTwister(5)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, xhat = build_bounded_A(n=n, mode=:symmetric, rng=rng)
        ℓ, u = bounds_via_LP(A, b; optimizer=optimizer)
        M    = bigM_from_bounds(ℓ, u; scale=1.0, common=true)

        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 6) Box + one quadratic inequality
    #    - QI convex (num_neg=0),
    #    - objective nonconvex (num_neg_q0 > 0) → overall NCVX.
    let i = 6, rng = MersenneTwister(6)
        is_cvx = true
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        # convex QI
        Q6, q6, r6 = make_qineq_with_anchor(rng, xhat; num_neg=0)

        # nonconvex objective
        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)
        if num_neg_q0 > 0
            is_cvx = false
        end

        # presence of a genuinely quadratic inequality is still convex here
        # (Q PSD), so only the objective breaks convexity.

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Q6,), qi=(q6,), ri=(r6,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 7) Box + one quadratic equality (always nonconvex unless linear)
    let i = 7, rng = MersenneTwister(7)
        is_cvx = false
        n   = 4
        rho = 3.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        P7, p7, s7 = make_qeq_with_anchor(rng, xhat; num_neg=0)

        num_neg_q0 = 1
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Pi=(P7,), pi=(p7,), si=(s7,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 8) Box + two q-inequalities + one q-equality + one linear equality
    let i = 8, rng = MersenneTwister(8)
        is_cvx = false
        n   = 10
        rho = 8.0
        A, b, M, ℓ, u, xhat = box_instance(n; rng=rng)

        # First QI: explicit small indefinite-ish matrix (no eigencontrol here)
        Qqi1 = zeros(Float64, n, n)
        Qqi1[1,1] = 2.0; Qqi1[2,2] = 2.0; Qqi1[1,2] = -2.0; Qqi1[2,1] = -2.0
        qqi1 = zeros(Float64, n)
        τ1   = rand(rng, 0:2)
        g01  = 0.5 * (xhat' * Qqi1 * xhat) + dot(qqi1, xhat)
        rqi1 = -g01 - τ1

        # Second QI: eigenvalue-controlled (can choose num_neg≥0)
        Qqi2, qqi2, rqi2 = make_qineq_with_anchor(rng, xhat; num_neg=1)

        # One quadratic equality
        Pqe, pqe, sqe = make_qeq_with_anchor(rng, xhat; num_neg=0)

        # One linear equality Hx = h
        H = rand(rng, 1, n)          
        h = H * xhat                


        num_neg_q0 = 2
        Q0, q0 = rand_objective(n; rng=rng, num_neg=num_neg_q0)

        tag = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                 Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                 H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", tag=tag)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # Take instance 1 and scale bigM by the factor of 10
    let base_idx = 1  # istersen 2,3,... başka bir instance da seçebilirsin
        base = insts[base_idx]

        new_inst = deepcopy(base)

        # Scale M
        new_inst["M"] = 10.0 .* base["M"]  
        # change ID
        new_inst["id"] = string(base["id"], "_M10")

        # add to the list
        push!(insts, new_inst)
    end


    return insts
end

# ---------------------------
# JSON I/O
# ---------------------------
function save_instances_json(path::AbstractString, insts)
    open(path, "w") do io
        JSON.print(io, insts, 2)
    end
end

function load_instances_json(path::AbstractString)::Vector{Dict{String,Any}}
    return JSON.parsefile(path)
end

end # module

using .InstanceGen

insts = InstanceGen.build_instances()
InstanceGen.save_instances_json("instances.json", insts)

println("Saved $(length(insts)) instances to instances.json")

