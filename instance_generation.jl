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

# instances.json dosyasını bu script’in olduğu klasöre koy:
instances_path = joinpath(@__DIR__, "instances.json")

InstanceGen.save_instances_json(instances_path, insts)

println("Saved $(length(insts)) instances to $(instances_path)")





module UserInstanceGen

using Random, LinearAlgebra

export generate_instance

# ---------------------------
# Basic random helpers
# ---------------------------

rand_vec_int(rng::AbstractRNG, n::Int; vals = -5:5) = rand(rng, vals, n)

"Random positive integer slack δ ∈ [lo, hi]."
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

"""
rand_symm_from_eigs(rng, n; num_neg, nonneg_vals, neg_vals)

Builds a symmetric matrix Q = Qmat * Diag(eigs) * Qmat'
with:
- num_neg eigenvalues drawn from neg_vals (negative integers),
- the remaining from nonneg_vals (nonnegative integers).
Only the signs matter for convexity / nonconvexity.
"""
function rand_symm_from_eigs(
    rng::AbstractRNG,
    n::Int;
    num_neg::Int,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
)
    @assert 0 ≤ num_neg ≤ n "num_neg must be between 0 and n"
    num_nonneg = n - num_neg

    negs    = num_neg == 0    ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0 ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R = randn(rng, n, n)
    Qfac, _ = qr(R)
    Qmat = Matrix(Qfac)
    Λ = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

"""
Quadratic inequality:

    0.5 xᵀ Q x + qᵀ x + r ≤ 0

with Q built to have num_neg negative eigenvalues, and constructed so that
g(xbar) = -δ < 0 at the anchor xbar.
"""
function make_qineq_with_eigs(
    rng::AbstractRNG,
    xbar::AbstractVector;
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
    qvals::UnitRange{Int} = -5:5,
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

constructed so that g(xbar) = 0 at the anchor xbar. We do not control
eigenvalues here explicitly; QE is always treated as nonconvex.
"""
function make_qeq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector;
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
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

"""
Random objective:

    0.5 xᵀ Q0 x + q0ᵀ x

with controlled number of negative eigenvalues.
"""
function rand_objective(
    rng::AbstractRNG,
    n::Int;
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
    qvals::UnitRange{Int} = -5:5,
)
    Q0 = rand_symm_from_eigs(rng, n;
                             num_neg     = num_neg,
                             nonneg_vals = nonneg_vals,
                             neg_vals    = neg_vals)
    q0 = rand_vec_int(rng, n; vals = qvals)
    return Matrix{Float64}(Q0), Float64.(q0)
end

# ---------------------------
# Big-M from bounds
# ---------------------------

"""
Compute Big-M variants from componentwise bounds ell ≤ x ≤ u.

- M_common[i] = scale * max(|ell[i]|, |u[i]|)
- M_minus[i]  = scale * max(0, -ell[i])
- M_plus[i]   = scale * max(0,  u[i])
"""
function bigM_from_bounds(ell::AbstractVector, u::AbstractVector; scale::Real = 1.0)
    @assert length(ell) == length(u)
    @assert scale ≥ 1.0 "scale should be ≥ 1.0 to keep Big-M safe"

    base    = max.(abs.(ell), abs.(u))
    M       = scale .* base
    Mminus  = scale .* max.(0.0, -ell)
    Mplus   = scale .* max.(0.0,  u)

    return M, Mminus, Mplus
end

# ---------------------------
# Anchor generators
# ---------------------------

"Generate a sparse anchor xbar with ‖xbar‖₀ ≤ ρ in ℝⁿ (general case)."
function generate_sparse_anchor(rng::AbstractRNG, n::Int, rho::Int;
                                vals::UnitRange{Int} = -3:3)
    x = zeros(Float64, n)
    rho == 0 && return x

    k = rand(rng, 1:rho)                     # support size
    idx = randperm(rng, n)[1:k]
    for i in idx
        v = 0
        while v == 0
            v = rand(rng, vals)
        end
        x[i] = float(v)
    end
    return x
end

"Generate an anchor xbar on the unit simplex with ‖xbar‖₀ ≤ ρ."
function generate_simplex_anchor(rng::AbstractRNG, n::Int, rho::Int)
    @assert rho ≥ 1 "On the simplex we need at least one nonzero variable (ρ ≥ 1)."
    s = rand(rng, 1:min(rho, n))             # support size on simplex
    idx = randperm(rng, n)[1:s]
    x   = zeros(Float64, n)
    vals = rand(rng, s)
    vals ./= sum(vals)
    for (j, i) in enumerate(idx)
        x[i] = vals[j]
    end
    return x
end

# ---------------------------
# Bounding constructors
# ---------------------------

"""
Simplex bounds: x ≥ 0, eᵀ x = 1.

Returns:
- A, b for x ≥ 0 encoded as -I x ≤ 0,
- H, h for eᵀ x = 1,
- ell, u as componentwise bounds [0,1].
"""
function build_simplex_bounds(n::Int)
    A   = -Matrix{Float64}(I, n, n)
    b   = zeros(Float64, n)
    H   = ones(Float64, 1, n)
    h   = [1.0]
    ell = zeros(Float64, n)
    u   = ones(Float64, n)
    return A, b, H, h, ell, u
end

"""
Box bounds: ℓ ≤ x ≤ u, constructed around xbar.

We choose random positive margins around xbar so that xbar is strictly inside the box.

Returns:
- A, b encoding ℓ ≤ x ≤ u as [I; -I] x ≤ [u; -ℓ],
- ell, u.
"""
function build_box_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    min_margin::Real = 1.0,
    max_margin::Real = 3.0,
)
    n = length(xbar)
    ell = zeros(Float64, n)
    u   = zeros(Float64, n)
    for i in 1:n
        m_low = rand(rng) * (max_margin - min_margin) + min_margin
        m_up  = rand(rng) * (max_margin - min_margin) + min_margin
        ell[i] = xbar[i] - m_low
        u[i]   = xbar[i] + m_up
    end
    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(u, -ell)
    return A, b, ell, u
end

"""
Linear polytope bounds via LI only (no box/simplex/QI/QE).

We build a system:
    x_i ≤ u_i   (i = 1,…,n)
    -∑_j x_j ≤ b_last
with b_last chosen so that xbar is strictly feasible.

This set is bounded because the rows {e₁,…,eₙ, -e} positively span ℝⁿ.

We then optionally add extra LI constraints (already counted in n_LI).
Returns:
- A, b,
- ell, u as safe (possibly loose) componentwise bounds.
"""
function build_linear_polytope_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector,
    n_LI::Int;
    min_margin::Real = 1.0,
    max_margin::Real = 3.0,
)
    n = length(xbar)

    # choose upper bounds around xbar
    u = zeros(Float64, n)
    for i in 1:n
        margin = rand(rng) * (max_margin - min_margin) + min_margin
        u[i] = xbar[i] + abs(margin)
    end

    δ = 1.0
    b_last = -sum(xbar) + δ     # -∑ x ≤ b_last ⇒ ∑ x ≥ -b_last

    # base constraints: x_i ≤ u_i, and -∑ x ≤ b_last
    A = Matrix{Float64}(undef, 0, n)
    b = Float64[]

    A = vcat(A, Matrix{Float64}(I, n, n))
    b = vcat(b, u)

    A = vcat(A, (-ones(Float64, 1, n)))
    b = vcat(b, b_last)

    base_n_LI = n + 1
    if n_LI > base_n_LI
        extra = n_LI - base_n_LI
        for _ in 1:extra
            a = randn(rng, n)
            δk = rand(rng) * 1.0 + 0.5
            bk = dot(a, xbar) + δk
            A = vcat(A, (a'))
            b = vcat(b, bk)
        end
    end

    # safe lower bounds from the base structure
    ell = zeros(Float64, n)
    for i in 1:n
        ell[i] = -b_last - sum(u[j] for j in 1:n if j != i)
    end

    return A, b, ell, u
end

"""
Quadratic ball inequality around xbar:

    0.5 xᵀ Q x + qᵀ x + r ≤ 0

with Q = I, representing ‖x - xbar‖ ≤ R.

We choose R > 0 so that xbar is strictly inside the ball.
Returns:
- Q, q, r,
- ell, u as componentwise bounds [xbar_i - R, xbar_i + R].
"""
function build_qi_ball_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    radius_margin::Real = 1.0,
)
    n = length(xbar)
    R = norm(xbar) + radius_margin
    Q = Matrix{Float64}(I, n, n)
    q = -collect(xbar)
    r = 0.5 * (dot(xbar, xbar) - R^2)   # g(x) = 0.5||x-xbar||² - 0.5R²

    ell = xbar .- R
    u   = xbar .+ R

    return Q, q, r, ell, u
end

"""
Quadratic ball equality around xbar:

    0.5 xᵀ P x + pᵀ x + s = 0

representing ‖x - xbar‖² = R² with P = 2I.

Returns:
- P, p, s,
- ell, u as componentwise bounds [xbar_i - R, xbar_i + R].
"""
function build_qe_ball_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    radius_margin::Real = 1.0,
)
    n = length(xbar)
    R = norm(xbar) + radius_margin
    P = 2.0 .* Matrix{Float64}(I, n, n)
    p = -2.0 .* collect(xbar)
    s = dot(xbar, xbar) - R^2

    ell = xbar .- R
    u   = xbar .+ R

    return P, p, s, ell, u
end

# ---------------------------
# Spectral control for Q0 and QI
# ---------------------------

"""
Build (neg_obj, neg_QI) from user-specified neg_eig_counts (or default),
and check consistency with want_convex, n_QI, n, and (optionally) QI-ball bounding.

Returns:
- neg_obj::Int
- neg_QI::Vector{Int} of length n_QI
"""
function build_neg_eig_counts(
    n::Int,
    n_QI::Int,
    want_convex::Bool,
    neg_eig_counts::Union{Nothing,Vector{Int}},
)
    # Default pattern if not provided
    if neg_eig_counts === nothing
        if want_convex
            # all PSD
            return 0, fill(0, n_QI)
        else
            # introduce nonconvexity via the objective if dimension allows
            if n > 1
                return 1, fill(0, n_QI)
            else
                # in 1D we cannot have 1 ≤ k < n, so keep everything 0;
                # nonconvexity (if any) must come from quadratic equalities.
                return 0, fill(0, n_QI)
            end
        end
    end

    # If provided, check length and ranges
    counts = neg_eig_counts::Vector{Int}
    expected_len = 1 + n_QI
    @assert length(counts) == expected_len "neg_eig_counts must have length 1 + n_QI."

    for k in counts
        @assert 0 ≤ k < n "Each entry in neg_eig_counts must satisfy 0 ≤ k < n."
    end

    neg_obj = counts[1]
    neg_QI  = counts[2:end]

    if want_convex
        @assert all(counts .== 0) "In convex mode, all entries of neg_eig_counts must be zero."
    else
        @assert any(counts .> 0) "In nonconvex mode, at least one entry in neg_eig_counts must be strictly positive."
    end

    return neg_obj, neg_QI
end

# ---------------------------
# Main instance generator
# ---------------------------

"""
generate_instance(...)

Build a single QCQP instance according to user-specified counts and convexity options.

Arguments (all keyword):
- n::Int: dimension
- rho::Int: cardinality bound (‖x‖₀ ≤ ρ)
- n_LI::Int: number of user linear inequalities (LI)
- n_LE::Int: number of user linear equalities (LE)
- n_QI::Int: number of quadratic inequalities (QI)
- n_QE::Int: number of quadratic equalities (QE)
- want_convex::Bool: whether the continuous quadratic part should be convex
- unit_simplex::Bool: if true, include unit simplex constraints
- box_constraint::Bool: if true, include a box constraint
- neg_eig_counts::Union{Nothing,Vector{Int}}: optional spectral pattern
- seed::Int: RNG seed
- bigM_scale::Real: scaling factor for Big-M (≥ 1)

Returns:
- inst::Dict{String,Any}, containing Q0, q0, Qi, qi, ri, Pi, pi, si,
  A, b, H, h, ell, u, M, M_minus, M_plus, and metadata.
"""
function generate_instance(;
    n::Int,
    rho::Int,
    n_LI::Int,
    n_LE::Int,
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    unit_simplex::Bool = false,
    box_constraint::Bool = false,
    neg_eig_counts::Union{Nothing,Vector{Int}} = nothing,
    seed::Int = 1,
    bigM_scale::Real = 1.0,
)
    rng = MersenneTwister(seed)

    # -----------------------
    # Step 0: basic checks
    # -----------------------
    @assert n ≥ 1 "n must be at least 1."
    @assert 0 ≤ rho ≤ n "rho must satisfy 0 ≤ rho ≤ n."
    @assert n_LI ≥ 0 && n_LE ≥ 0 && n_QI ≥ 0 && n_QE ≥ 0

    # -----------------------
    # Step 1: choose bounding type
    # -----------------------
    # Priority: simplex > box > QI-ball > QE-ball > LI-poly
    bounding_type::Symbol
    if unit_simplex
        bounding_type = :simplex
        @assert rho ≥ 1 "With unit_simplex=true, we need ρ ≥ 1 (simplex points have at least one nonzero)."
    elseif box_constraint
        bounding_type = :box
    elseif n_QI > 0
        bounding_type = :qi_ball
    elseif n_QE > 0
        bounding_type = :qe_ball
        @assert !want_convex "Quadratic equalities are only allowed in nonconvex mode."
    else
        bounding_type = :li_poly
        @assert n_QI == 0 && n_QE == 0 "In LI-only mode we require n_QI = n_QE = 0."
        @assert n_LI > 0 "At least one linear inequality is required if there are no QI/QE and no simplex/box."
        @assert n_LI ≥ n + 1 "We require n_LI ≥ n+1 to build a bounded polytope from LI only."
    end

    # -----------------------
    # Step 2: spectral control
    # -----------------------
    neg_obj, neg_QI = build_neg_eig_counts(n, n_QI, want_convex, neg_eig_counts)

    # If we use the first QI as a ball for bounding, it must be PSD (no negative eigs)
    if bounding_type == :qi_ball && n_QI ≥ 1
        @assert neg_QI[1] == 0 "When using the first QI as a ball, its neg_eig_count must be 0."
    end

    # In convex mode we do not allow genuinely quadratic equalities
    if want_convex
        @assert n_QE == 0 "Quadratic equalities are not allowed in convex mode."
    end

    # -----------------------
    # Step 3: anchor xbar
    # -----------------------
    xbar = if bounding_type == :simplex
        generate_simplex_anchor(rng, n, rho)
    else
        generate_sparse_anchor(rng, n, rho)
    end

    # -----------------------
    # Step 4: build bounding constraints and ell,u
    # -----------------------
    A = Matrix{Float64}(undef, 0, n)
    b = Float64[]
    H = Matrix{Float64}(undef, 0, n)
    h = Float64[]
    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    # We also need slots for one "bounding" QI or QE if applicable
    Qi = Vector{Matrix{Float64}}()
    qi = Vector{Vector{Float64}}()
    ri = Float64[]

    Pi = Vector{Matrix{Float64}}()
    pi_vecs = Vector{Vector{Float64}}()
    si = Float64[]

    # For LI-only bounding, n_LI is fully consumed by the bounding polytope
    use_LI_for_bounding = (bounding_type == :li_poly)

    if bounding_type == :simplex
        A0, b0, H0, h0, ell0, u0 = build_simplex_bounds(n)
        A = vcat(A, A0)
        b = vcat(b, b0)
        H = vcat(H, H0)
        h = vcat(h, h0)
        ell .= ell0
        u   .= u0

    elseif bounding_type == :box
        A0, b0, ell0, u0 = build_box_bounds(rng, xbar)
        A = vcat(A, A0)
        b = vcat(b, b0)
        ell .= ell0
        u   .= u0

    elseif bounding_type == :qi_ball
        Qb, qb, rb, ell0, u0 = build_qi_ball_bounds(rng, xbar)
        push!(Qi, Qb)
        push!(qi, qb)
        push!(ri, rb)
        ell .= ell0
        u   .= u0

    elseif bounding_type == :qe_ball
        Pb, pb, sb, ell0, u0 = build_qe_ball_bounds(rng, xbar)
        push!(Pi, Pb)
        push!(pi_vecs, pb)
        push!(si, sb)
        ell .= ell0
        u   .= u0

    elseif bounding_type == :li_poly
        A0, b0, ell0, u0 = build_linear_polytope_bounds(rng, xbar, n_LI)
        A = vcat(A, A0)
        b = vcat(b, b0)
        ell .= ell0
        u   .= u0
    end

    # -----------------------
    # Step 5: add extra LI/LE (if any)
    # -----------------------

    # Extra LI: only if we did NOT already use n_LI for bounding
    if !use_LI_for_bounding && n_LI > 0
        for _ in 1:n_LI
            a_int = rand(rng, -2:2, n)
            while all(a_int .== 0)
                a_int = rand(rng, -2:2, n)
            end
            a = Float64.(a_int)
            δk = rand(rng) * 1.0 + 0.5
            bk = dot(a, xbar) + δk
            A = vcat(A, a')
            b = vcat(b, bk)
        end
    end

    # Extra LE: always generated according to n_LE
    if n_LE > 0
        H_extra = Matrix{Float64}(undef, n_LE, n)
        h_extra = zeros(Float64, n_LE)
        for r in 1:n_LE
            v_int = rand(rng, -2:2, n)
            while all(v_int .== 0)
                v_int = rand(rng, -2:2, n)
            end
            v = Float64.(v_int)
            H_extra[r, :] .= v
            h_extra[r] = dot(v, xbar)
        end
        H = vcat(H, H_extra)
        h = vcat(h, h_extra)
    end

    # -----------------------
    # Step 6: add remaining QI/QE (beyond bounding ones)
    # -----------------------

    # We may have already used the first QI as a ball in qi_ball case
    start_qi_idx = (bounding_type == :qi_ball) ? 2 : 1

    if n_QI ≥ 1
        @assert length(neg_QI) == n_QI
    end

    for j in start_qi_idx:n_QI
        num_neg = neg_QI[j]
        Qj, qj, rj = make_qineq_with_eigs(rng, xbar; num_neg = num_neg)
        push!(Qi, Qj)
        push!(qi, qj)
        push!(ri, rj)
    end

    # Quadratic equalities: we may already have used the first QE as a ball
    start_qe_idx = (bounding_type == :qe_ball) ? 2 : 1
    for _ in start_qe_idx:n_QE
        Pj, pj, sj = make_qeq_with_anchor(rng, xbar)
        push!(Pi, Pj)
        push!(pi_vecs, pj)
        push!(si, sj)
    end

    # -----------------------
    # Step 7: objective
    # -----------------------
    Q0, q0 = rand_objective(rng, n; num_neg = neg_obj)

    # -----------------------
    # Step 8: Big-M from ell,u
    # -----------------------
    M_common, M_minus, M_plus = bigM_from_bounds(ell, u; scale = bigM_scale)

    # -----------------------
    # Step 9: pack into Dict
    # -----------------------
    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = (length(Qi) == 0) ? nothing : Qi
    inst["qi"] = (length(qi) == 0) ? nothing : qi
    inst["ri"] = (length(ri) == 0) ? nothing : ri

    inst["Pi"] = (length(Pi) == 0) ? nothing : Pi
    inst["pi"] = (length(pi_vecs) == 0) ? nothing : pi_vecs
    inst["si"] = (length(si) == 0) ? nothing : si

    inst["A"] = (size(A, 1) == 0) ? nothing : A
    inst["b"] = (length(b) == 0) ? nothing : b
    inst["H"] = (size(H, 1) == 0) ? nothing : H
    inst["h"] = (length(h) == 0) ? nothing : h

    inst["ell"]     = ell
    inst["u"]       = u
    inst["M"]       = M_common
    inst["M_minus"] = M_minus
    inst["M_plus"]  = M_plus

    inst["xbar"]          = xbar
    inst["want_convex"]   = want_convex
    inst["unit_simplex"]  = unit_simplex
    inst["box_constraint"]= box_constraint
    inst["neg_eig_counts"] = neg_eig_counts
    inst["seed"]          = seed

    tag = want_convex ? "CVX" : "NCVX"
    id = "n$(n)_rho$(rho)_LI$(n_LI)_LE$(n_LE)_QI$(n_QI)_QE$(n_QE)_$(tag)_seed$(seed)"
    inst["id"] = id

    return inst
end

end # module
