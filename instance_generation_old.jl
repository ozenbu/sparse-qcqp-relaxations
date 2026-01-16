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
"""
pack(...)

Stores all instance data in a Dict. Meta fields:

- "base_tag"   ∈ {"S","B","P",""}  : base geometry (simplex, box, polytope)
- "convexity" ∈ {"CVX","NCVX"}     : continuous quadratic part type
"""
function pack(; n::Int, rho::Real=3.0, Q0=nothing, q0=nothing,
               Qi=nothing, qi=nothing, ri=nothing,
               Pi=nothing, pi=nothing, si=nothing,
               A=nothing, b=nothing, H=nothing, h=nothing,
               M::Union{Nothing,AbstractVector}=nothing,
               seed::Union{Nothing,Int}=nothing,
               base_tag::AbstractString = "",
               convexity::AbstractString = "CVX")
    if M === nothing
        M = ones(Float64, n)
    end
    d = Dict{String,Any}(
        "n"         => n,
        "rho"       => rho,
        "Q0"        => Q0,
        "q0"        => q0,
        "Qi"        => Qi,
        "qi"        => qi,
        "ri"        => ri,
        "Pi"        => Pi,
        "pi"        => pi,
        "si"        => si,
        "A"         => A,
        "b"         => b,
        "H"         => H,
        "h"         => h,
        "M"         => M,
        "base_tag"  => String(base_tag),
        "convexity" => String(convexity),
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

    # convexity label: prefer "convexity", fall back to old "tag" for compatibility,
    # otherwise infer from eigenvalues
    cvx = get(d, "convexity", nothing)
    if cvx === nothing
        cvx = get(d, "tag", nothing)  # backwards compatibility
    end
    if cvx === nothing
        cvx = convex_tag_eig(d)
        d["convexity"] = String(cvx)
    end
    push!(parts, String(cvx))

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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="S", convexity=cvx)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=cvx)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=cvx)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 4) A-bounded minimal (positive spanning), convex objective  → treat as polytope (P)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=cvx)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 5) A-bounded symmetric, nonconvex objective (num_neg > 0)  → also polytope (P)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=cvx)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 6) Box + one quadratic inequality (convex QI, nonconvex objective)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Q6,), qi=(q6,), ri=(r6,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=cvx)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Pi=(P7,), pi=(p7,), si=(s7,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=cvx)
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

        cvx = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                 Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                 H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=cvx)
        d["id"] = make_id(d)
        insts[i] = d
    end

    # 9) Take instance 1 and scale Big-M by the factor of 10
    let base_idx = 1
        base = insts[base_idx]
        new_inst = deepcopy(base)

        # Scale M
        new_inst["M"] = 10.0 .* base["M"]
        # change ID with explicit Big-M suffix
        new_inst["id"] = string(base["id"], "_Mx10")

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
















module UserInstanceGen

using Random, LinearAlgebra
using JuMP, MosekTools

const MOI = JuMP.MOI

export generate_instance

# ============================
# Basic random helpers
# ============================

"Random integer vector of length n with entries in `vals`."
rand_vec_int(rng::AbstractRNG, n::Int; vals = -5:5) = rand(rng, vals, n)

"Random positive integer slack δ in [lo, hi]."
rand_slack(rng::AbstractRNG; lo::Int = 1, hi::Int = 5) = rand(rng, lo:hi)

"""
Random symmetric matrix with prescribed numbers of negative eigenvalues.

Builds Q = Qmat * Diag(eigs) * Qmat', where:
- `num_neg` eigenvalues are drawn from `neg_vals` (negative integers),
- the remaining eigenvalues are drawn from `nonneg_vals` (nonnegative integers).
Only the sign pattern matters for convexity.
"""
function rand_symm_from_eigs(
    rng::AbstractRNG,
    n::Int;
    num_neg::Int,
    nonneg_vals::UnitRange{Int} = 1:5,
    neg_vals::UnitRange{Int} = -5:-1,
)
    @assert 0 ≤ num_neg ≤ n "num_neg must be between 0 and n"
    num_nonneg = n - num_neg

    negs    = num_neg == 0    ? Int[] : rand(rng, neg_vals,    num_neg)
    nonnegs = num_nonneg == 0 ? Int[] : rand(rng, nonneg_vals, num_nonneg)

    eigs = vcat(negs, nonnegs)
    eigs = eigs[randperm(rng, length(eigs))]

    R = randn(rng, n, n)
    F = qr(R)
    Qmat = Matrix(F.Q)
    Λ = Diagonal(Float64.(eigs))

    return Qmat * Λ * Qmat'
end

"Random quadratic objective: 0.5 x'Q0 x + q0'x, with controlled negative eigenvalues."
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

"""
Quadratic inequality:

    g(x) = 0.5 x'Qx + q'x + r ≤ 0

constructed so that:
- Q has `num_neg` negative eigenvalues,
- g(xbar) = -δ < 0 at the anchor `xbar`.
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

    h(x) = 0.5 x'P x + p'x + s = 0

constructed so that h(xbar) = 0. We do not control eigenvalues here; QE is always
treated as nonconvex at the feasible-set level when P ≠ 0.
"""
function make_qeq_with_anchor(
    rng::AbstractRNG,
    xbar::AbstractVector;
    nonneg_vals::UnitRange{Int} = 0:5,
    neg_vals::UnitRange{Int} = -5:-1,
    pvals::UnitRange{Int} = -3:3,
)
    n = length(xbar)
    P = rand_symm_from_eigs(rng, n;
                            num_neg     = 0,
                            nonneg_vals = nonneg_vals,
                            neg_vals    = neg_vals)
    p = rand_vec_int(rng, n; vals = pvals)

    g0 = 0.5 * (xbar' * P * xbar) + dot(p, xbar)
    s  = -g0

    return Matrix{Float64}(P), Float64.(p), Float64(s)
end

# ============================
# Big-M from bounds
# ============================

"""
Compute Big-M vectors from componentwise bounds ell ≤ x ≤ u.

- M[i]       = scale * max(|ell[i]|, |u[i]|)
- M_minus[i] = scale * max(0, -ell[i])
- M_plus[i]  = scale * max(0,  u[i])
"""
function bigM_from_bounds(ell::AbstractVector, u::AbstractVector; scale::Real = 1.0)
    @assert length(ell) == length(u)
    @assert scale ≥ 1.0 "scale should be ≥ 1.0."

    base   = max.(abs.(ell), abs.(u))
    M      = scale .* base
    Mminus = scale .* max.(0.0, -ell)
    Mplus  = scale .* max.(0.0,  u)

    return M, Mminus, Mplus
end

# ============================
# Anchor generators
# ============================

"Generate a sparse anchor xbar with ‖xbar‖₀ ≤ rho in R^n."
function generate_sparse_anchor(
    rng::AbstractRNG,
    n::Int,
    rho::Int;
    vals::UnitRange{Int} = -3:3,
)
    x = zeros(Float64, n)
    rho == 0 && return x

    k = rand(rng, 1:rho)            # support size
    idx = randperm(rng, n)[1:k]
    for i in idx
        v = 0
        while v == 0
            v = rand(rng, vals)
        end
        x[i] = Float64(v)
    end
    return x
end

"Generate an anchor xbar on the unit simplex with support size at most rho."
function generate_simplex_anchor(rng::AbstractRNG, n::Int, rho::Int)
    @assert rho ≥ 1 "On the unit simplex we need rho ≥ 1 (at least one nonzero entry)."
    s = rand(rng, 1:min(rho, n))    # support size
    idx = randperm(rng, n)[1:s]

    x = zeros(Float64, n)
    vals = rand(rng, s)
    vals ./= sum(vals)

    for (j, i) in enumerate(idx)
        x[i] = vals[j]
    end
    return x
end

# ============================
# Base linear bounded regions
# ============================

"""
Unit simplex base:

    x >= 0,  e'x = 1.

Returns:
- A, b for x >= 0 encoded as -I x <= 0,
- H, h for e'x = 1,
- ell, u as [0,1] bounds.
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
Box base: a random box around xbar.

We choose random positive margins around each coordinate of xbar so that
    ell[i] < xbar[i] < u[i],
and encode ell <= x <= u as [I; -I] x <= [u; -ell].

Returns:
- A, b,
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
General polytope base using LI only.

We build a bounded polytope of the form:
    x_i <= u_i  (i = 1,...,n)
    -sum_j x_j <= b_last

where u and b_last are chosen so that xbar is strictly feasible.
The normals {e_1,..., e_n, -e} positively span R^n, so the region is bounded.

Returns:
- A, b,
- ell, u as safe (possibly conservative) bounds derived from these constraints.
"""
function build_polytope_bounds(
    rng::AbstractRNG,
    xbar::AbstractVector;
    min_margin::Real = 1.0,
    max_margin::Real = 3.0,
)
    n = length(xbar)

    # Upper bounds x_i <= u_i
    u = zeros(Float64, n)
    for i in 1:n
        margin = rand(rng) * (max_margin - min_margin) + min_margin
        u[i] = xbar[i] + abs(margin)
    end

    # Constraint -sum x <= b_last so that xbar is strictly feasible
    δ = 1.0
    b_last = -sum(xbar) + δ

    # A has n+1 rows: n for x_i <= u_i, 1 for -sum x <= b_last
    A = zeros(Float64, n + 1, n)
    b = zeros(Float64, n + 1)

    for i in 1:n
        A[i, i] = 1.0
        b[i]    = u[i]
    end
    A[n+1, :] .= -1.0
    b[n+1]    = b_last

    # Lower bounds from -sum x <= b_last and x_j <= u_j for j != i
    ell = zeros(Float64, n)
    for i in 1:n
        ell[i] = -b_last - sum(u[j] for j in 1:n if j != i)
    end

    return A, b, ell, u
end

# ============================
# Vertex LP + cutting helpers
# ============================

"""
Solve a small LP to get a vertex v of {x : A x <= b, H x = h} with a random
linear objective c'x, and return both v and c.
"""
function find_vertex_lp(
    rng::AbstractRNG,
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n

    model = Model(optimizer)
    @variable(model, x[1:n])

    if m > 0
        @constraint(model, A * x .<= b)
    end
    if size(H, 1) > 0
        @constraint(model, H * x .== h)
    end

    # random nonzero objective c
    c_int = rand(rng, -2:2, n)
    while all(c_int .== 0)
        c_int = rand(rng, -2:2, n)
    end
    c = Float64.(c_int)

    @objective(model, Min, dot(c, x))
    optimize!(model)

    status = termination_status(model)
    @assert status == MOI.OPTIMAL "LP for vertex is not optimal: $status"

    v = [value(x[i]) for i in 1:n]
    return v, c
end

"Build inequality a'x <= beta from direction c so that xbar is feasible and v is cut off."
function build_cut_from_c(
    xbar::AbstractVector,
    v::AbstractVector,
    c::AbstractVector;
    tol_same::Float64 = 1e-8,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(c) == n

    αx = dot(c, xbar)
    αv = dot(c, v)

    if isapprox(αx, αv; atol = tol_same)
        error("Cannot build cut: c'xbar ≈ c'v for this c.")
    end

    # orient so that αv > αx
    if αv < αx
        c = -c
        αv = -αv
        αx = -αx
    end

    if αv <= αx + tol_same
        error("Cannot orient cut: c'xbar and c'v too close.")
    end

    θ = 0.5   # mid-point; can randomize in (0,1) if desired
    β = αx + θ * (αv - αx)

    # Then c'xbar < β < c'v, so xbar feasible, v infeasible.
    return c, β
end

"Build a linear equality c'x = beta that keeps xbar feasible and cuts off v."
function build_equality_from_c(
    xbar::AbstractVector,
    v::AbstractVector,
    c::AbstractVector;
    tol_same::Float64 = 1e-8,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(c) == n

    αx = dot(c, xbar)
    αv = dot(c, v)

    if isapprox(αx, αv; atol = tol_same)
        error("Cannot build a nontrivial equality: c'xbar ≈ c'v for this c.")
    end

    β = αx
    # Equality is c'x = β: xbar satisfies it, v does not.

    return c, β
end

# ============================
# Extra linear constraints
# ============================

"Add n_extra linear inequalities via vertex cuts (nonredundant, keep xbar feasible)."
function add_extra_LI_vertex_cuts!(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector,
    n_extra::Int;
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    for k in 1:n_extra
        println()
        println("---- Extra linear inequality $k / $n_extra ----")
        println("Current A size: ", size(A))

        # 1) find a vertex and the objective that generated it
        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Anchor x̄:")
        println(xbar)
        println("Vertex v (LP optimum):")
        println(v)
        println("Objective c used to obtain v:")
        println(c)

        # avoid degenerate case v ≈ xbar
        if norm(v .- xbar, Inf) < 1e-6
            println("Vertex is too close to anchor; trying alternative random objectives.")
            found = false
            for _try in 1:10
                v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                if norm(v2 .- xbar, Inf) ≥ 1e-6
                    v = v2
                    c = c2
                    found = true
                    println("Replaced vertex with:")
                    println(v)
                    println("New objective c:")
                    println(c)
                    break
                end
            end
            if !found
                @warn "Polytope seems to have collapsed to xbar; cannot add more nonredundant LI."
                return A, b
            end
        end

        # 2) build inequality a'x <= beta from c
        try
            a, β = build_cut_from_c(xbar, v, c)

            axbar = dot(a, xbar)
            av    = dot(a, v)

            println("New inequality: a' x ≤ β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", axbar, "   (should be < β)")
            println("  a' v  = ", av,    "   (should be > β)")
            println("  a' x̄ - β = ", axbar - β)
            println("  a' v  - β = ", av - β)

            # 3) append to A, b
            A = vcat(A, a')
            b = vcat(b, β)

            println("  -> Inequality accepted. New A size: ", size(A))
        catch e
            @warn "Failed to construct inequality for this vertex: $e. Stopping LI enrichment."
            return A, b
        end
    end

    return A, b
end

"Add n_extra linear equalities via vertex LPs using c'x = c'xbar."
function add_extra_LE_vertex_cuts!(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector,
    n_extra::Int;
    optimizer = MosekTools.Optimizer,
    rank_tol::Float64 = 1e-8,
)
    n = length(xbar)
    for k in 1:n_extra
        println()
        println("---- Extra linear equality $k / $n_extra ----")
        println("Current H size: ", size(H))

        # 1) Get a vertex v and objective c from the current polytope
        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Anchor x̄:")
        println(xbar)
        println("Vertex v (LP optimum):")
        println(v)
        println("Objective c used to obtain v:")
        println(c)

        # Avoid degenerate case v ≈ xbar
        if norm(v .- xbar, Inf) < 1e-6
            println("Vertex is too close to anchor; trying alternative random objectives.")
            found = false
            for _try in 1:10
                v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                if norm(v2 .- xbar, Inf) ≥ 1e-6
                    v = v2
                    c = c2
                    found = true
                    println("Replaced vertex with:")
                    println(v)
                    println("New objective c:")
                    println(c)
                    break
                end
            end
            if !found
                @warn "Polytope seems to have collapsed to xbar; cannot add more nonredundant LE."
                return H, h
            end
        end

        # 2) Build equality from c: c'x = beta with beta = c'xbar
        try
            a, beta = build_equality_from_c(xbar, v, c)
            println("Candidate equality: a' x = beta")
            println("  a = ", a)
            println("  beta = ", beta)

            axbar = dot(a, xbar)
            av    = dot(a, v)
            println("  a' x̄ = ", axbar, "   (should be = beta)")
            println("  a' v  = ", av,    "   (should be ≠ beta)")
            println("  a' x̄ - beta = ", axbar - beta)
            println("  a' v  - beta = ", av - beta)

            # 3) Check nonredundancy via rank(H)
            rank_H = (size(H, 1) == 0) ? 0 : rank(H; atol = rank_tol)
            H_new  = vcat(H, a')
            rank_new = rank(H_new; atol = rank_tol)

            println("  rank(H)       = ", rank_H)
            println("  rank([H; a']) = ", rank_new)

            if rank_new == rank_H
                println("  -> This equality is (numerically) redundant; trying another vertex/objective.")
                redundant = true
                for _try in 1:5
                    v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                    if norm(v2 .- xbar, Inf) < 1e-6
                        continue
                    end
                    try
                        a2, beta2 = build_equality_from_c(xbar, v2, c2)
                        H_new2 = vcat(H, a2')
                        rank_new2 = rank(H_new2; atol = rank_tol)
                        if rank_new2 > rank_H
                            println("  -> Found nonredundant equality after retry.")
                            a, beta = a2, beta2
                            axbar = dot(a, xbar)
                            av    = dot(a, v2)
                            println("  New a = ", a)
                            println("  New beta = ", beta)
                            println("  New a' x̄ = ", axbar)
                            println("  New a' v = ", av)
                            redundant = false
                            H_new = H_new2
                            break
                        end
                    catch
                        continue
                    end
                end
                if redundant
                    @warn "Could not find a nonredundant equality; stopping LE enrichment."
                    return H, h
                end
            end

            # 4) Accept equality: update H, h
            H = H_new
            h = vcat(h, beta)

            println("  -> Equality accepted. New H size: ", size(H))
        catch e
            @warn "Failed to construct equality for this vertex: $e. Stopping LE enrichment."
            return H, h
        end
    end

    return H, h
end

# ============================
# Spectral control for Q0 and QI
# ============================

"""
Build (neg_obj, neg_QI) from user-specified neg_eig_counts (or defaults),
and check consistency with convex / nonconvex mode and dimension.

Arguments:
- n: dimension
- n_QI: number of quadratic inequalities
- n_QE: number of quadratic equalities
- want_convex: convex mode flag
- neg_eig_counts: optional vector (k0, k1, ..., k_{n_QI})

Returns:
- neg_obj::Int
- neg_QI::Vector{Int} of length n_QI
"""
function build_neg_eig_counts(
    n::Int,
    n_QI::Int,
    n_QE::Int,
    want_convex::Bool,
    neg_eig_counts::Union{Nothing,Vector{Int}},
)
    # No vector provided: choose defaults
    if neg_eig_counts === nothing
        if want_convex
            return 0, fill(0, n_QI)
        else
            if n ≥ 2
                # introduce nonconvexity via the objective
                return 1, fill(0, n_QI)
            else
                # n = 1; we cannot have 1 ≤ k < n.
                # Nonconvexity must then come from QE (if any).
                if n_QE > 0
                    return 0, fill(0, n_QI)
                else
                    error("Cannot realize a nonconvex instance in dimension n=1 without QE or explicit neg_eig_counts.")
                end
            end
        end
    end

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
        if all(counts .== 0) && n_QE == 0
            error("In nonconvex mode, at least one entry of neg_eig_counts must be positive (or n_QE > 0).")
        end
    end

    return neg_obj, neg_QI
end

# ============================
# Main instance generator
# ============================

"""
generate_instance(...)

Build a single QCQP instance according to a user specification.

Keyword arguments:
- n::Int: dimension (n ≥ 1)
- rho::Int: cardinality bound for x (‖x‖₀ ≤ rho, with 0 ≤ rho ≤ n)
- base_type::Symbol: geometry of the base bounded region; one of
    :simplex, :box, :poly
- n_LI::Int: number of additional linear inequalities (beyond the base region)
- n_LE::Int: number of linear equalities
- n_QI::Int: number of quadratic inequalities
- n_QE::Int: number of quadratic equalities
- want_convex::Bool: whether the continuous quadratic part should be convex
- neg_eig_counts::Union{Nothing,Vector{Int}}:
    optional vector (k0, k1, ..., k_{n_QI}) controlling negative eigenvalues
    of Q0 and each QI
- seed::Int: RNG seed
- bigM_scale::Real: factor ≥ 1.0 used in Big-M computation

Returns:
- inst::Dict{String,Any} holding all data matrices and metadata.
"""
function generate_instance(;
    n::Int,
    rho::Int,
    base_type::Symbol = :box,
    n_LI::Int = 0,
    n_LE::Int = 0,
    n_QI::Int = 0,
    n_QE::Int = 0,
    want_convex::Bool = true,
    neg_eig_counts::Union{Nothing,Vector{Int}} = nothing,
    seed::Int = 1,
    bigM_scale::Real = 1.0,
    optimizer = MosekTools.Optimizer,
)
    rng = MersenneTwister(seed)

    # --- basic checks ---
    @assert n ≥ 1 "n must be at least 1."
    @assert 0 ≤ rho ≤ n "rho must satisfy 0 ≤ rho ≤ n."
    @assert n_LI ≥ 0 && n_LE ≥ 0 && n_QI ≥ 0 && n_QE ≥ 0
    @assert bigM_scale ≥ 1.0 "bigM_scale must be ≥ 1.0."
    @assert base_type in (:simplex, :box, :poly) "base_type must be :simplex, :box, or :poly."

    if base_type == :simplex
        @assert rho ≥ 1 "With base_type = :simplex we need rho ≥ 1."
    end

    if want_convex
        @assert n_QE == 0 "Quadratic equalities are only allowed in nonconvex mode."
    end

    # --- spectral pattern for objective and QI ---
    neg_obj, neg_QI = build_neg_eig_counts(n, n_QI, n_QE, want_convex, neg_eig_counts)

    # --- anchor xbar ---
    xbar = if base_type == :simplex
        generate_simplex_anchor(rng, n, rho)
    else
        generate_sparse_anchor(rng, n, rho)
    end

    # --- base linear region ---
    A = zeros(Float64, 0, n)
    b = Float64[]
    H = zeros(Float64, 0, n)
    h = Float64[]
    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    if base_type == :simplex
        A0, b0, H0, h0, ell0, u0 = build_simplex_bounds(n)
        A = vcat(A, A0)
        b = vcat(b, b0)
        H = vcat(H, H0)
        h = vcat(h, h0)
        ell .= ell0
        u   .= u0
    elseif base_type == :box
        A0, b0, ell0, u0 = build_box_bounds(rng, xbar)
        A = vcat(A, A0)
        b = vcat(b, b0)
        ell .= ell0
        u   .= u0
    else
        A0, b0, ell0, u0 = build_polytope_bounds(rng, xbar)
        A = vcat(A, A0)
        b = vcat(b, b0)
        ell .= ell0
        u   .= u0
    end

    # --- extra LE via vertex-based equalities ---
    if n_LE > 0
        println()
        println("=== Linear equalities via vertex LPs ===")
        println("Base type      : ", base_type)
        println("Dimension n    : ", n)
        println("Anchor x̄:")
        println(xbar)
        println("Number of extra LE constraints to add: ", n_LE)
        H, h = add_extra_LE_vertex_cuts!(rng, A, b, H, h, xbar, n_LE; optimizer = optimizer)
        println("=== Finished adding extra LE constraints ===")
        println()
    end

    # --- extra LI via vertex cuts ---
    if n_LI > 0
        println()
        println("=== Linear inequalities via vertex cuts ===")
        println("Base type      : ", base_type)
        println("Dimension n    : ", n)
        println("Anchor x̄:")
        println(xbar)
        println("Number of extra LI constraints to add: ", n_LI)
        A, b = add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, n_LI; optimizer = optimizer)
        println("=== Finished adding extra LI constraints ===")
        println()
    end

    # --- quadratic inequalities ---
    Qi_list = Vector{Matrix{Float64}}()
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    if n_QI > 0
        @assert length(neg_QI) == n_QI
        for i in 1:n_QI
            num_neg = neg_QI[i]
            Q_i, q_i, r_i = make_qineq_with_eigs(rng, xbar; num_neg = num_neg)
            push!(Qi_list, Q_i)
            push!(qi_list, q_i)
            push!(ri_list, r_i)
        end
    end

    # --- quadratic equalities (nonconvex mode only) ---
    Pi_list   = Vector{Matrix{Float64}}()
    pi_list   = Vector{Vector{Float64}}()
    si_list   = Float64[]

    if n_QE > 0
        @assert !want_convex "Quadratic equalities are only allowed in nonconvex mode."
        for _ in 1:n_QE
            P_j, p_j, s_j = make_qeq_with_anchor(rng, xbar)
            push!(Pi_list, P_j)
            push!(pi_list, p_j)
            push!(si_list, s_j)
        end
    end

    # --- objective ---
    Q0, q0 = rand_objective(rng, n; num_neg = neg_obj)

    # --- Big-M from bounds ---
    M_common, M_minus, M_plus = bigM_from_bounds(ell, u; scale = bigM_scale)

    # --- pack into Dict ---
    inst = Dict{String,Any}()

    inst["n"]   = n
    inst["rho"] = rho

    inst["Q0"] = Q0
    inst["q0"] = q0

    inst["Qi"] = isempty(Qi_list) ? nothing : Tuple(Qi_list)
    inst["qi"] = isempty(qi_list) ? nothing : Tuple(qi_list)
    inst["ri"] = isempty(ri_list) ? nothing : ri_list

    inst["Pi"] = isempty(Pi_list) ? nothing : Tuple(Pi_list)
    inst["pi"] = isempty(pi_list) ? nothing : Tuple(pi_list)
    inst["si"] = isempty(si_list) ? nothing : si_list

    inst["A"] = size(A, 1) == 0 ? nothing : A
    inst["b"] = isempty(b)      ? nothing : b
    inst["H"] = size(H, 1) == 0 ? nothing : H
    inst["h"] = isempty(h)      ? nothing : h

    inst["ell"]     = ell
    inst["u"]       = u
    inst["M"]       = M_common
    inst["M_minus"] = M_minus
    inst["M_plus"]  = M_plus

    inst["xbar"]                 = xbar
    inst["base_type"]            = String(Symbol(base_type))
    inst["want_convex"]          = want_convex
    inst["neg_eig_counts_input"] = neg_eig_counts
    inst["neg_eig_counts_used"]  = vcat(neg_obj, neg_QI)
    inst["seed"]                 = seed

    tag = want_convex ? "CVX" : "NCVX"
    id  = "n$(n)_rho$(rho)_LI$(n_LI)_LE$(n_LE)_QI$(n_QI)_QE$(n_QE)_$(tag)_seed$(seed)"
    inst["id"] = id

    return inst
end

end # module UserInstanceGen


using .InstanceGen
using .UserInstanceGen

# 1) Existing small instances from InstanceGen
insts = InstanceGen.build_instances()

# 2) New instance from UserInstanceGen (n = 50, convex simplex)
inst50 = UserInstanceGen.generate_instance(
    n = 50,
    rho = 10,
    base_type = :simplex,
    n_LI = 3,      # make this > 0 to see the step-by-step cuts
    n_LE = 0,
    n_QI = 0,
    n_QE = 0,
    want_convex = true,
    neg_eig_counts = nothing,  # defaults to all zeros (PSD objective)
    seed = 101,
    bigM_scale = 1.0,
)

# 3) Append new instance to the existing list
push!(insts, inst50)

# 4) Save everything into a single JSON file next to this script
instances_path = joinpath(@__DIR__, "instances.json")
InstanceGen.save_instances_json(instances_path, insts)

println("Saved $(length(insts)) instances to $(instances_path)")

using .UserInstanceGen

println("===== SMALL TEST INSTANCE: n=3, base_type=:poly, n_LI=2 =====")

inst_small = UserInstanceGen.generate_instance(
    n = 3,
    rho = 2,
    base_type = :poly,
    n_LI = 2,    # two extra LIs via vertex cuts
    n_LE = 0,
    n_QI = 0,
    n_QE = 0,
    want_convex = true,
    neg_eig_counts = nothing,   # objective PSD by default in convex mode
    seed = 2026,
    bigM_scale = 1.0,
)

println("----- Final instance summary -----")
println("id        = ", inst_small["id"])
println("n         = ", inst_small["n"])
println("rho       = ", inst_small["rho"])
println("base_type = ", inst_small["base_type"])
println("A size    = ", inst_small["A"] === nothing ? "none" : size(inst_small["A"]))
println("b length  = ", inst_small["b"] === nothing ? "none" : length(inst_small["b"]))
println("H size    = ", inst_small["H"] === nothing ? "none" : size(inst_small["H"]))
println("h length  = ", inst_small["h"] === nothing ? "none" : length(inst_small["h"]))
println("ell       = ", inst_small["ell"])
println("u         = ", inst_small["u"])
println("xbar      = ", inst_small["xbar"])
println("==============================================")
