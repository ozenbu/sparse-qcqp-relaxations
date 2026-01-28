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
               convexity::AbstractString = "CVX",
               n_LI::Int = 0, n_LE::Int = 0,
               n_QI::Int = 0, n_QE::Int = 0,
               bigM_scale::Real = 1.0)
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
        "convexity" => String(convexity),
        "n_LI"     => n_LI,
        "n_LE"     => n_LE,
        "n_QI"     => n_QI,
        "n_QE"     => n_QE,
        "bigM_scale" => bigM_scale,
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

"Build human-readable instance id string, with counts and Big-M scale."
function make_id(d::Dict{String,Any})
    n    = Int(d["n"])
    rho  = Int(round(d["rho"]))
    base = get(d, "base_tag", "")

    # Check if we have explicit counts (from UserInstanceGen or future extensions)
    has_counts = haskey(d, "n_LI") || haskey(d, "n_LE") || haskey(d, "n_QI") || haskey(d, "n_QE")
    n_LI = has_counts ? get(d, "n_LI", 0) : 0
    n_LE = has_counts ? get(d, "n_LE", 0) : 0
    n_QI = has_counts ? get(d, "n_QI", 0) : 0
    n_QE = has_counts ? get(d, "n_QE", 0) : 0

    convexity = get(d, "convexity", nothing)
    if convexity === nothing
        convexity = convex_tag_eig(d)
    end

    parts = String[]
    push!(parts, "n$(n)")
    push!(parts, "rho$(rho)")
    base != "" && push!(parts, String(base))

    if has_counts
        if n_LI > 0
            push!(parts, "LI$(n_LI)")
        elseif has_lin_ineq(d)
            push!(parts, "LI")
        end
        if n_LE > 0
            push!(parts, "LE$(n_LE)")
        elseif has_lin_eq(d)
            push!(parts, "LE")
        end
        if n_QI > 0
            push!(parts, "QI$(n_QI)")
        elseif has_qineq(d)
            push!(parts, "QI")
        end
        if n_QE > 0
            push!(parts, "QE$(n_QE)")
        elseif has_qeq(d)
            push!(parts, "QE")
        end
    else
        has_lin_ineq(d) && push!(parts, "LI")
        has_lin_eq(d)   && push!(parts, "LE")
        has_qineq(d)    && push!(parts, "QI")
        has_qeq(d)      && push!(parts, "QE")
    end

    push!(parts, String(convexity))

    bigM_scale = get(d, "bigM_scale", 1.0)
    if abs(bigM_scale - 1.0) > 1e-8
        k = round(bigM_scale; digits=6)
        if isapprox(k, round(k); atol=1e-8)
            push!(parts, "Mx$(Int(round(k)))")
        else
            push!(parts, "Mx$(k)")
        end
    end

    seed = get(d, "seed", nothing)
    if seed !== nothing
        push!(parts, "seed$(seed)")
    end

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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                  Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                  seed=i, base_tag="S", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="P", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Q6,), qi=(q6,), ri=(r6,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Pi=(P7,), pi=(p7,), si=(s7,),
                 M=M, Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
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

        convexity = is_cvx ? "CVX" : "NCVX"
        d = pack(; n=n, rho=rho, A=A, b=b,
                 Qi=(Qqi1, Qqi2), qi=(qqi1, qqi2), ri=(rqi1, rqi2),
                 Pi=(Pqe,), pi=(pqe,), si=(sqe,),
                 H=H, h=h, M=M,
                 Q0=Q0 .* objective_scale, q0=q0 .* objective_scale,
                 seed=i, base_tag="B", convexity=convexity)
        insts[i] = d
    end

    # 9) Take instance 1 and scale Big-M by a factor of 10
    base_idx = 1
    base = insts[base_idx]
    new_inst = deepcopy(base)
    new_inst["M"] = 10.0 .* base["M"]
    new_inst["bigM_scale"] = get(base, "bigM_scale", 1.0) * 10.0
    # seed aynı kalıyor; ID içinde sadece Mx10 farkı olacak
    push!(insts, new_inst)


    # ID'leri tek formatta oluştur
    for d in insts
        d["id"] = make_id(d)
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

end # module InstanceGen









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
Only the sign pattern matters for convexity (PSD vs indefinite).
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
    scale_Q::Real = 1.0,
    scale_q::Real = 1.0,
)
    Q0 = rand_symm_from_eigs(rng, n;
                             num_neg     = num_neg,
                             nonneg_vals = nonneg_vals,
                             neg_vals    = neg_vals)
    q0 = rand_vec_int(rng, n; vals = qvals)
    return scale_Q .* Matrix{Float64}(Q0), scale_q .* Float64.(q0)
end

"""
Quadratic inequality:

    g(x) = 0.5 x'Qx + q'x + r ≤ 0

constructed so that:
- Q has `num_neg` negative eigenvalues,
- g(xbar) = -δ < 0 at the anchor `xbar`.

For convex QI that we use in bounds, num_neg=0 and nonneg_vals=1:5
⇒ Q is strongly convex (λ_min >= 1).
"""
function make_qineq_with_eigs(
    rng::AbstractRNG,
    xbar::AbstractVector;
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 1:5,
    neg_vals::UnitRange{Int} = -5:-1,
    qvals::UnitRange{Int} = -5:5,
    slack_lo::Int = 1,
    slack_hi::Int = 5,
    scale_Q::Real = 1.0,
    scale_q::Real = 1.0,
)
    n = length(xbar)
    Q_raw = rand_symm_from_eigs(rng, n;
                                num_neg     = num_neg,
                                nonneg_vals = nonneg_vals,
                                neg_vals    = neg_vals)
    Q = scale_Q .* Matrix{Float64}(Q_raw)
    q = scale_q .* Float64.(rand_vec_int(rng, n; vals = qvals))

    g0 = 0.5 * (xbar' * Q * xbar) + dot(q, xbar)
    δ  = rand_slack(rng; lo = slack_lo, hi = slack_hi)

    r = -g0 - δ

    return Q, q, Float64(r)
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
    num_neg::Int = 0,
    nonneg_vals::UnitRange{Int} = 1:5,
    neg_vals::UnitRange{Int} = -5:-1,
    pvals::UnitRange{Int} = -3:3,
    scale_P::Real = 1.0,
    scale_p::Real = 1.0,
)
    n = length(xbar)

    P_raw = rand_symm_from_eigs(rng, n;
                                num_neg     = num_neg,
                                nonneg_vals = nonneg_vals,
                                neg_vals    = neg_vals)
    P = scale_P .* Matrix{Float64}(P_raw)
    p = scale_p .* Float64.(rand_vec_int(rng, n; vals = pvals))

    g0 = 0.5 * (xbar' * P * xbar) + dot(p, xbar)
    s  = -g0

    return P, p, Float64(s)
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
# Vertex LP helpers
# ============================

"""
Solve a small LP to get a vertex v of {x : A x <= b, H x = h}
with a random linear objective c'x.

Returns (v, c).
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
    @assert size(H, 2) == n || size(H, 1) == 0

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

"Given xbar, v, and a normal a, build a halfspace a'x <= β that keeps xbar and cuts off v."
function build_cut_halfspace_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector;
    θ::Real = 0.5,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    αx = dot(a, xbar)
    αv = dot(a, v)

    if isapprox(αx, αv; atol=1e-8)
        error("Cannot separate: projections coincide (a'xbar ≈ a'v).")
    end

    # ensure αv > αx by flipping sign if needed
    if αv ≤ αx
        a = -a
        αx = -αx
        αv = -αv
    end

    if αv ≤ αx + 1e-8
        error("Cannot separate safely: a'v not strictly larger than a'xbar.")
    end

    β = αx + θ * (αv - αx)

    return a, β, αx, αv
end

"Given xbar, v and normal a, build an equality a'x = β passing through xbar but not v."
function build_equality_from_normal(
    xbar::AbstractVector,
    v::AbstractVector,
    a::AbstractVector,
)
    n = length(xbar)
    @assert length(v) == n
    @assert length(a) == n

    β = dot(a, xbar)
    αv = dot(a, v)

    if isapprox(β, αv; atol=1e-8)
        error("Cannot build nonredundant equality: a'xbar ≈ a'v.")
    end

    return a, β, β, αv
end

# ============================
# Extra linear constraints
# ============================

"""
Add n_extra linear inequalities via vertex cuts (nonredundant, keep xbar feasible),
using the objective vector c of the vertex LP as the normal direction.

Each new inequality is of the form a'x <= β with:
- a' x̄ < β,
- a' v  > β,
so x̄ remains feasible, while the chosen vertex v is eliminated.
"""
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

        # 1) find a vertex and its objective direction c
        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v:")
        println(v)

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
                    break
                end
            end
            if !found
                @warn "Polytope seems to have collapsed to xbar; cannot add more nonredundant LI."
                return A, b
            end
        end

        # 2) build halfspace using a = c (may be flipped internally)
        try
            a, β, αx, αv = build_cut_halfspace_from_normal(xbar, v, c; θ = 0.5)

            println("New inequality: a' x ≤ β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", αx, "   (should be < β)")
            println("  a' v  = ", αv, "   (should be > β)")
            println("  a' x̄ - β = ", αx - β)
            println("  a' v  - β = ", αv - β)

            # 3) append to A, b
            A = vcat(A, a')
            b = vcat(b, β)
        catch err
            @warn "Failed to construct separating inequality from c; skipping this LI. Error: $err"
        end
    end

    return A, b
end

"""
Add n_LE linear equalities of the form a'x = β using vertex + objective vector.

Each new equality:
- passes through the anchor x̄ (so x̄ remains feasible),
- cuts off at least one vertex used to build it.
"""
function add_extra_LE!(
    rng::AbstractRNG,
    A::Matrix{Float64},
    b::Vector{Float64},
    H::Matrix{Float64},
    h::Vector{Float64},
    xbar::AbstractVector,
    n_LE::Int;
    optimizer = MosekTools.Optimizer,
)
    n = length(xbar)
    if n_LE == 0
        return H, h
    end

    for k in 1:n_LE
        println()
        println("---- Extra linear equality $k / $n_LE ----")

        v, c = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
        println("Vertex v used for equality:")
        println(v)

        # avoid degenerate case v ≈ xbar or cᵀv ≈ cᵀx̄
        ok = false
        for _try in 1:10
            if norm(v .- xbar, Inf) < 1e-6 || isapprox(dot(c, v), dot(c, xbar); atol = 1e-8)
                v2, c2 = find_vertex_lp(rng, A, b, H, h; optimizer = optimizer)
                v, c = v2, c2
            else
                ok = true
                break
            end
        end
        if !ok
            @warn "Could not find a vertex with distinct projection for equality; skipping this LE."
            continue
        end

        try
            a, β, αx, αv = build_equality_from_normal(xbar, v, c)

            println("New equality: a' x = β")
            println("  a = ", a)
            println("  β = ", β)
            println("  a' x̄ = ", αx, "   (x̄ lies on the hyperplane)")
            println("  a' v  = ", αv, "   (vertex does NOT lie on the hyperplane)")
            println("  a' x̄ - β = ", αx - β)
            println("  a' v  - β = ", αv - β)

            H = vcat(H, a')
            h = vcat(h, β)
        catch err
            @warn "Failed to construct separating equality from c; skipping this LE. Error: $err"
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
# Bounds (for Big-M): linear + convex quadratic ineq
# ============================

"""
Compute coordinate-wise bounds (ell, u) of the linear region:

    P_lin := { x : A x <= b, H x = h }.

Used when there are no convex quadratic inequalities.
"""
function bounds_from_linear_part(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n || size(H, 1) == 0

    use_eq = (size(H, 1) > 0)

    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    for i in 1:n
        # lower bound on x[i]
        model = Model(optimizer)
        @variable(model, x[1:n])
        if m > 0
            @constraint(model, A * x .<= b)
        end
        if use_eq
            @constraint(model, H * x .== h)
        end
        @objective(model, Min, x[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Lower-bound LP for x[$i] not optimal: $status"
        ell[i] = value(x[i])

        # upper bound on x[i]
        model = Model(optimizer)
        @variable(model, x2[1:n])
        if m > 0
            @constraint(model, A * x2 .<= b)
        end
        if use_eq
            @constraint(model, H * x2 .== h)
        end
        @objective(model, Max, x2[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Upper-bound LP for x[$i] not optimal: $status"
        u[i] = value(x2[i])
    end

    return ell, u
end

"""
Compute coordinate-wise bounds (ell, u) of

    P_cvx := { x : A x <= b, H x = h, g_j(x) <= 0 for j in convex_qi_indices },

where each g_j(x) = 0.5 x'Q_j x + q_j'x + r_j is convex
(Q_j is PSD, and here actually strongly convex because eigenvalues are ≥ 1).

Nonconvex quadratic inequalities and all quadratic equalities are ignored here.
"""
function bounds_with_convex_quadratics(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real},
    Qi_list::Vector{Matrix{Float64}},
    qi_list::Vector{Vector{Float64}},
    ri_list::Vector{Float64},
    convex_qi_indices::Vector{Int};
    optimizer = MosekTools.Optimizer,
)
    m, n = size(A)
    @assert length(b) == m
    @assert size(H, 2) == n || size(H, 1) == 0

    use_eq = (size(H, 1) > 0)

    ell = zeros(Float64, n)
    u   = zeros(Float64, n)

    for i in 1:n
        # ----- lower bound: min x[i] -----
        model = Model(optimizer)
        @variable(model, x[1:n])

        if m > 0
            @constraint(model, A * x .<= b)
        end
        if use_eq
            @constraint(model, H * x .== h)
        end

        # convex quadratic inequalities
        for j in convex_qi_indices
            Q = Qi_list[j]
            q = qi_list[j]
            r = ri_list[j]
            @constraint(model,
                0.5 * sum(Q[p, q_] * x[p] * x[q_] for p in 1:n, q_ in 1:n) +
                sum(q[p] * x[p] for p in 1:n) +
                r <= 0.0
            )
        end

        @objective(model, Min, x[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Lower-bound QP for x[$i] not optimal: $status"
        ell[i] = value(x[i])

        # ----- upper bound: max x[i] -----
        model = Model(optimizer)
        @variable(model, x2[1:n])

        if m > 0
            @constraint(model, A * x2 .<= b)
        end
        if use_eq
            @constraint(model, H * x2 .== h)
        end

        for j in convex_qi_indices
            Q = Qi_list[j]
            q = qi_list[j]
            r = ri_list[j]
            @constraint(model,
                0.5 * sum(Q[p, q_] * x2[p] * x2[q_] for p in 1:n, q_ in 1:n) +
                sum(q[p] * x2[p] for p in 1:n) +
                r <= 0.0
            )
        end

        @objective(model, Max, x2[i])
        optimize!(model)
        status = termination_status(model)
        @assert status == MOI.OPTIMAL "Upper-bound QP for x[$i] not optimal: $status"
        u[i] = value(x2[i])
    end

    return ell, u
end


"""
Clean bounds (ell, u) for nicer numerics, in a *safe* way for Big-M:

- Round ell[i] *downwards* and u[i] *upwards* to the given number of digits.
- No extra slack; we already go outward.
"""
function clean_bounds!(
    ell::AbstractVector,
    u::AbstractVector;
    digits::Int = 4,
)
    @assert length(ell) == length(u)
    fac = 10.0 ^ digits

    for i in eachindex(ell)
        # skip non-finite just in case
        if !isfinite(ell[i]) || !isfinite(u[i])
            continue
        end

        # outward rounding to `digits` decimals
        ell[i] = floor(ell[i] * fac) / fac
        u[i]   = ceil( u[i] * fac) / fac

        # tiny values → snap to exactly 0.0 for cosmetics
        if abs(ell[i]) < 10.0^(-digits)
            ell[i] = 0.0
        end
        if abs(u[i]) < 10.0^(-digits)
            u[i] = 0.0
        end
    end

    return ell, u
end

"""
Compute bounds (ell, u) for Big-M:

- If there are no quadratic inequalities, or none of them is convex (neg_QI[j] > 0),
  fall back to purely linear bounds.

- Otherwise, include all convex quadratic inequalities (neg_QI[j] == 0)
  when computing the bounds.
"""
function bounds_for_bigM(
    A::AbstractMatrix{<:Real},
    b::AbstractVector{<:Real},
    H::AbstractMatrix{<:Real},
    h::AbstractVector{<:Real},
    Qi_list::Vector{Matrix{Float64}},
    qi_list::Vector{Vector{Float64}},
    ri_list::Vector{Float64},
    neg_QI::Vector{Int};
    optimizer = MosekTools.Optimizer,
)
    if isempty(Qi_list)
        return bounds_from_linear_part(A, b, H, h; optimizer = optimizer)
    end

    convex_qi_indices = [j for j in 1:length(Qi_list) if neg_QI[j] == 0]

    if isempty(convex_qi_indices)
        # all quadratic inequalities are nonconvex → ignore them for bounds
        return bounds_from_linear_part(A, b, H, h; optimizer = optimizer)
    end

    return bounds_with_convex_quadratics(
        A, b, H, h,
        Qi_list, qi_list, ri_list,
        convex_qi_indices;
        optimizer = optimizer,
    )
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
- optimizer: JuMP optimizer type (default MosekTools.Optimizer)

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

    # --- extra LE (first fix the affine subspace) ---
    if n_LE > 0
        H, h = add_extra_LE!(rng, A, b, H, h, xbar, n_LE; optimizer = optimizer)
    end

    # --- extra LI via vertex cuts (nonredundant) ---
    if n_LI > 0
        println()
        println("=== Linear enrichment with vertex cuts ===")
        println("Base type      : ", base_type)
        println("Dimension n    : ", n)
        println("Anchor x̄:")
        println(xbar)
        println("Number of extra LI constraints to add: ", n_LI)
        A, b = add_extra_LI_vertex_cuts!(rng, A, b, H, h, xbar, n_LI; optimizer = optimizer)
        println("=== Finished adding extra LI constraints ===")
        println()
    end

    # --- quadratic inequalities (built before bounds) ---
    Qi_list = Vector{Matrix{Float64}}()
    qi_list = Vector{Vector{Float64}}()
    ri_list = Float64[]

    if n_QI > 0
        @assert length(neg_QI) == n_QI
        for i in 1:n_QI
            num_neg = neg_QI[i]
            Q_i, q_i, r_i = make_qineq_with_eigs(rng, xbar; num_neg = num_neg,
                                                 scale_Q = 1.0, scale_q = 1.0)
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
            # we keep num_neg=0 here; QE is nonconvex anyway as a set
            P_j, p_j, s_j = make_qeq_with_anchor(rng, xbar; num_neg = 0,
                                                 scale_P = 1.0, scale_p = 1.0)
            push!(Pi_list, P_j)
            push!(pi_list, p_j)
            push!(si_list, s_j)
        end
    end

    # --- bounds for Big-M: linear + convex quadratic inequalities ---
    ell, u = bounds_for_bigM(A, b, H, h, Qi_list, qi_list, ri_list, neg_QI; optimizer = optimizer)

    # numeric cleaning: outward rounding for safe Big-M
    clean_bounds!(ell, u; digits = 4)

    println("Final bounds (linear + convex quadratic inequalities) for Big-M (cleaned):")
    println("  ell = ", ell)
    println("  u   = ", u)

    # --- objective ---
    Q0, q0 = rand_objective(rng, n; num_neg = neg_obj,
                            scale_Q = 1.0, scale_q = 1.0)

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

    inst["xbar"]                  = xbar
    inst["base_type"]             = String(Symbol(base_type))
    inst["want_convex"]           = want_convex
    inst["neg_eig_counts_input"]  = neg_eig_counts
    inst["neg_eig_counts_used"]   = vcat(neg_obj, neg_QI)
    inst["seed"]                  = seed

    # base geometry tag S/B/P
    base_tag = base_type == :simplex ? "S" :
               base_type == :box     ? "B" :
               base_type == :poly    ? "P" : ""
    inst["base_tag"] = base_tag

    # convexity string (InstanceGen ile uyum)
    convexity = want_convex ? "CVX" : "NCVX"
    inst["convexity"] = convexity

    # meta: kaç ekstra alsın
    inst["n_LI"] = n_LI
    inst["n_LE"] = n_LE
    inst["n_QI"] = n_QI
    inst["n_QE"] = n_QE
    inst["bigM_scale"] = bigM_scale

    # --- ID: n{n}_rho{rho}_{S/B/P}_LI3_LE1_QI2_QE1_{CVX/NCVX}_Mx10_seed{seed}
    parts = String[]
    push!(parts, "n$(n)")
    push!(parts, "rho$(rho)")
    base_tag != "" && push!(parts, base_tag)

    # LI / LE / QI / QE - sayılı format
    if n_LI > 0
        push!(parts, "LI$(n_LI)")
    elseif size(A, 1) > 0
        push!(parts, "LI")
    end

    if n_LE > 0
        push!(parts, "LE$(n_LE)")
    elseif size(H, 1) > 0
        push!(parts, "LE")
    end

    if n_QI > 0
        push!(parts, "QI$(n_QI)")
    elseif !isempty(Qi_list)
        push!(parts, "QI")
    end

    if n_QE > 0
        push!(parts, "QE$(n_QE)")
    elseif !isempty(Pi_list)
        push!(parts, "QE")
    end

    push!(parts, convexity)

    if abs(bigM_scale - 1.0) > 1e-8
        k = round(bigM_scale; digits=6)
        if isapprox(k, round(k); atol=1e-8)
            push!(parts, "Mx$(Int(round(k)))")
        else
            push!(parts, "Mx$(k)")
        end
    end

    push!(parts, "seed$(seed)")
    inst["id"] = join(parts, "_")

    return inst
end

end # module UserInstanceGen











using .InstanceGen
using .UserInstanceGen

# ------------------------------------------------
# Special small rational box instance (EU ≠ IU)
# ------------------------------------------------
function make_small_rational_instance()
    Q0 = [
        3//10000    127//1250   79//2500    867//10000;
        127//1250   1//500      1001//10000 1059//10000;
        79//2500    1001//10000 -1//2000    -703//10000;
        867//10000  1059//10000 -703//10000 -1063//10000
    ]
    Q0d = Q0 .* 20000

    q0  = [-1973//10000, -2535//10000, -1967//10000, -973//10000]
    q0d = q0 .* 20000

    A2 = [ 1.0   0.0   0.0   0.0
           0.0   1.0   0.0   0.0
           0.0   0.0   1.0   0.0
           0.0   0.0   0.0   1.0
          -1.0   0.0   0.0   0.0
           0.0  -1.0   0.0   0.0
           0.0   0.0  -1.0   0.0
           0.0   0.0   0.0  -1.0 ]

    b2 = fill(1.0, 8)   # -1 ≤ x_i ≤ 1

    n   = 4
    rho = 3.0

    d = Dict{String,Any}()

    d["n"]   = n
    d["rho"] = rho

    d["Q0"] = Float64.(Q0d)
    d["q0"] = Float64.(q0d)

    d["Qi"] = nothing
    d["qi"] = nothing
    d["ri"] = nothing

    d["Pi"] = nothing
    d["pi"] = nothing
    d["si"] = nothing

    d["A"] = A2
    d["b"] = b2
    d["H"] = nothing
    d["h"] = nothing

    # Box [-1,1]^4 → Big-M = 1 is enough
    d["M"] = ones(Float64, n)

    # Metadata so the rest of the pipeline understands it
    d["base_tag"]   = "B"        # box
    d["convexity"]  = "NCVX"     # we know EU/IU differ at RLT level
    d["n_LI"]       = 0          # only the box, no extra linear constraints
    d["n_LE"]       = 0
    d["n_QI"]       = 0
    d["n_QE"]       = 0
    d["bigM_scale"] = 1.0
    d["seed"]       = 0          # artificial seed, just a placeholder

    # Manual ID so it is easy to spot in plots / tables
    d["id"] = "n4_rho3_B_LI_NCVX_rational"

    return d
end

function build_all_instances(; bigM_scale::Real = 1.0)
    # 0) Special rational box instance with EU ≠ IU at plain RLT
    special = make_small_rational_instance()

    # 1) Old small instances (8 + 1 scaled copy) → 9 instances
    base_insts = InstanceGen.build_instances()

    # start list with the special instance
    insts = vcat([special], base_insts)

    # 2) New user-specified instances (UserInstanceGen) → 11 more
    # Total will be 1 + 9 + 11 = 21.

    # 2.1  Large simplex instance: n=50, rho=10, 3 extra LI, only linear
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 50,
            rho = 10,
            base_type = :simplex,
            n_LI = 3,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 101,
            bigM_scale = bigM_scale,
        )
    )

    # 2.2  Simplex, n=10, convex, 1 quadratic inequality (PSD)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 10,
            rho = 4,
            base_type = :simplex,
            n_LI = 0,
            n_LE = 0,
            n_QI = 1,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 102,
            bigM_scale = bigM_scale,
        )
    )

    # 2.3  Box, convex, 2 extra LI (vertex cuts), no QE/QI
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 2,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 103,
            bigM_scale = bigM_scale,
        )
    )

    # 2.4  Box, convex, 2 LI + 2 LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 2,
            n_LE = 2,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 104,
            bigM_scale = bigM_scale,
        )
    )

    # 2.5  Box, nonconvex only via objective (no QI/QE)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 8,
            rho = 3,
            base_type = :box,
            n_LI = 0,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = false,
            neg_eig_counts = [1],   # one negative eigenvalue in Q0
            seed = 105,
            bigM_scale = bigM_scale,
        )
    )

    # 2.6  Polytope base, convex, 2 LI, no QI/QE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 2,
            n_LE = 0,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 106,
            bigM_scale = bigM_scale,
        )
    )

    # 2.7  Polytope base, nonconvex via objective + 1 convex QI
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 2,
            n_LE = 0,
            n_QI = 1,
            n_QE = 0,
            want_convex = false,
            neg_eig_counts = [1, 0],   # Q0 indefinite, Q1 PSD
            seed = 107,
            bigM_scale = bigM_scale,
        )
    )

    # 2.8  Polytope base, 1 quadratic equality (nonconvex via QE)
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 6,
            rho = 4,
            base_type = :poly,
            n_LI = 1,
            n_LE = 1,
            n_QI = 0,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [0],      # Q0 PSD, nonconvexity from QE
            seed = 108,
            bigM_scale = bigM_scale,
        )
    )

    # 2.9  Box, medium size n=20, convex, 3 LI + 2 LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 20,
            rho = 5,
            base_type = :box,
            n_LI = 3,
            n_LE = 2,
            n_QI = 0,
            n_QE = 0,
            want_convex = true,
            neg_eig_counts = nothing,
            seed = 109,
            bigM_scale = bigM_scale,
        )
    )

    # 2.10 Box, n=20, nonconvex, 1 QI + 1 QE + some LI/LE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 20,
            rho = 5,
            base_type = :box,
            n_LI = 2,
            n_LE = 2,
            n_QI = 1,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [1, 0],   # Q0 indefinite, QI PSD
            seed = 110,
            bigM_scale = bigM_scale,
        )
    )

    # 2.11 Polytope, n=15, mixed: LI + LE + 2 QI + 1 QE
    push!(insts,
        UserInstanceGen.generate_instance(
            n = 15,
            rho = 7,
            base_type = :poly,
            n_LI = 3,
            n_LE = 1,
            n_QI = 2,
            n_QE = 1,
            want_convex = false,
            neg_eig_counts = [1, 1, 0],  # Q0 indef, QI1 indef, QI2 PSD
            seed = 111,
            bigM_scale = bigM_scale,
        )
    )

    @assert length(insts) == 21 "We expect exactly 21 instances."

    return insts
end

# --- Build & save ---

insts = build_all_instances(; bigM_scale = 1.0)
@assert length(insts) == 21  # sanity check

instances_path = joinpath(@__DIR__, "instances.json")
InstanceGen.save_instances_json(instances_path, insts)

println("Saved $(length(insts)) instances to $(instances_path)")
println("Instance IDs:")
for d in insts
    println("  ", d["id"])
end
