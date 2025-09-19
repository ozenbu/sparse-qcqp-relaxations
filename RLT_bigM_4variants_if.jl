println(">>> Script loaded")

using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface
using Printf          
const MOI = MathOptInterface


# ---------- helpers ----------
# Python's as_list: None → empty, vector/tuple → itself, scalar → 1-tuple
function as_list(x)
    if x === nothing
        return ()
    elseif x isa AbstractVector || x isa Tuple
        return x
    else
        return (x,)
    end
end

# Outer product helper (column × row). Works for numeric vectors and JuMP variable arrays.
outer_xy(a, b) = a * transpose(b)

# ---------- core ----------
"""
    build_and_solve(data; variant="EXACT", build_only=false, optimizer=Gurobi.Optimizer)

Solve either the exact MIQCQP ("EXACT") or the first-level RLT relaxations ("E","EU","I","IU").
If `build_only=true`, returns the model without solving.

`data` keys (mimics your Python dict):
- "n"::Int, "rho"::Real
- "Q0"::Matrix, "q0"::Vector
- optional quadratic rows: "Qi","qi","ri" (each can be `nothing` or a collection)
- optional equality rows:  "Pi","pi","si" (each can be `nothing` or a collection)
- optional linear:         "A","b" (A is ℓ×n, b is length ℓ; if missing, treated as 0 rows)
- optional equalities:     "H","h" (H is η×n, h is length η; if missing, treated as 0 rows)
- Big-M:                   "M" (vector of Mᵢ or diagonal matrix)

Returns:
- EXACT → (:OPTIMAL, obj, x, u) on success
- E*/I* → (:OPTIMAL, obj, x, u, X, R, U) on success
- Otherwise one of (:UNBOUNDED, :INF_OR_UNBD, :INFEASIBLE, :OTHER, …) with placeholders
"""

function unpack_qcqp_data(data::Dict)
    n   = data["n"]
    ρ   = data["rho"]

    Q0  = data["Q0"]
    q0  = data["q0"]

    Qi  = as_list(get(data, "Qi", nothing))
    qi  = as_list(get(data, "qi", nothing))
    ri  = as_list(get(data, "ri", nothing))
    Pi  = as_list(get(data, "Pi", nothing))
    pi  = as_list(get(data, "pi", nothing))
    si  = as_list(get(data, "si", nothing))

    A = get(data, "A", nothing)
    b = get(data, "b", nothing)
    if A === nothing
        A = zeros(0, n)
        b = zeros(0)
    end
    ℓ = size(A, 1)

    H = get(data, "H", nothing)
    h = get(data, "h", nothing)
    if H === nothing
        H = zeros(0, n)
        h = zeros(0)
    end
    η = size(H, 1)

    # Big-M: accept vector or matrix; normalize to Diagonal
    Mraw  = data["M"]
    Mdiag = ndims(Mraw) == 1 ? Mraw : diag(Mraw)
    M     = Diagonal(Mdiag)

    e = ones(n)

    return (; n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e)
end

function build_and_solve(data; variant::String="EXACT", build_only::Bool=false, optimizer=Gurobi.Optimizer)
    vars = unpack_qcqp_data(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e = vars

    # -- model --
    m = Model(optimizer)
    set_name(m, "SQCQP_RLT")

    # decision variables
    @variable(m, x[1:n])
    if variant == "EXACT"
        @variable(m, u[1:n], Bin)
    else
        @variable(m, u[1:n]) # u ∈ (−∞, ∞)

    end
    @variable(m, X[1:n, 1:n], Symmetric)
    @variable(m, R[1:n, 1:n])
    @variable(m, U[1:n, 1:n], Symmetric)

    # objective
    if variant == "EXACT"
        @objective(m, Min, 0.5 * x' * Q0 * x + q0' * x)
        set_optimizer_attribute(m, "NonConvex", 2)  # allow general Q
    else
        @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + q0' * x)
    end

    # original linear constraints
    if ℓ > 0
        @constraint(m, A * x .<= b)      # Ax ≤ b
    end
    if η > 0
        @constraint(m, H * x .== h)      # Hx = h
    end

    # Big-M links: -M u ≤ x ≤ M u
    @constraint(m, -M * u .<= x)
    @constraint(m,  x .<= M * u)

    # relaxations / lifted part
    if variant != "EXACT"
        # quadratic ≤ 0 rows
        for (Qmat, qvec, rterm) in zip(Qi, qi, ri)
            @constraint(m, 0.5 * sum(Qmat[i,j]*X[i,j] for i=1:n, j=1:n) + qvec' * x + rterm <= 0)
        end
        # quadratic == 0 rows
        for (Pmat, pvec, sterm) in zip(Pi, pi, si)
            @constraint(m, 0.5 * sum(Pmat[i,j]*X[i,j] for i=1:n, j=1:n) + pvec' * x + sterm == 0)
        end

        # diag(U) = u
        @constraint(m, [i=1:n], U[i,i] == u[i])

        # H X = h xᵀ,  H R = h uᵀ
        if η > 0
            @constraint(m, H * X .== outer_xy(h, x))
            @constraint(m, H * R .== outer_xy(h, u))
        end

        # Big-M McCormick blocks (elementwise):
        # M U M − M Rᵀ − R M + X ≥ 0
        @constraint(m, M * U * M .- M * R' .- R * M .+ X .>= 0)
        # M U M + M Rᵀ + R M + X ≥ 0
        @constraint(m, M * U * M .+ M * R' .+ R * M .+ X .>= 0)
        # M U M + M Rᵀ − R M − X ≥ 0
        @constraint(m, M * U * M .+ M * R' .- R * M .- X .>= 0)

        # RLT from A x ≤ b
        if ℓ > 0
            # (b − A x)(b − A x)ᵀ ≥ 0  elementwise
            @constraint(m, A * X * A' .- A * (x * b') .- (b * x') * A' .+ (b * b') .>= 0)

            # (b − A x)(M u − x)ᵀ ≥ 0
            @constraint(m, A * X .- (b * x') .- A * R * M .+ (b * u') * M .>= 0)

            # (b − A x)(M u + x)ᵀ ≥ 0
            @constraint(m, -A * X .+ (b * x') .- A * R * M .+ (b * u') * M .>= 0)
        end

        # equality vs inequality cases
        if startswith(variant, "E")
            # eᵀu = ρ,  Re = ρ x,  Ue = ρ u
            @constraint(m, sum(u) == ρ)
            @constraint(m, R * e .== ρ .* x)
            

            # Big-M × (ρ − eᵀu)ᵀ
            # (M u − x)(ρ − eᵀu)ᵀ ≥ 0
            @constraint(m, ρ .* (M * u) .- ρ .* x .- M * U * e .+ R * e .>= 0)
            # (M u + x)(ρ − eᵀu)ᵀ ≥ 0
            @constraint(m, ρ .* (M * u) .+ ρ .* x .- M * U * e .- R * e .>= 0)
            @constraint(m, U * e .== ρ .* u)
        else
            # eᵀu ≤ ρ
            @constraint(m, sum(u) <= ρ)

            # self-RLT: (ρ − eᵀu)^2 ≥ 0  →  ρ² − 2ρ·(eᵀu) + eᵀ U e ≥ 0
            @constraint(m, ρ^2 - 2ρ * sum(u) + dot(e, U * e) >= 0)

            # (b − A x)(ρ − eᵀu)ᵀ ≥ 0
            if ℓ > 0
                @constraint(m, ρ .* b .- (b * u') * e .- ρ .* (A * x) .+ A * R * e .>= 0)
            end

            # Big-M × (ρ − eᵀu)ᵀ
            # (M u − x)(ρ − eᵀu)ᵀ ≥ 0
            @constraint(m, ρ .* (M * u) .- ρ .* x .- M * U * e .+ R * e .>= 0)
            # (M u + x)(ρ − eᵀu)ᵀ ≥ 0
            @constraint(m, ρ .* (M * u) .+ ρ .* x .- M * U * e .- R * e .>= 0)
        end

        # variants with "U": add u ≤ e and its RLTs
        if endswith(variant, "U")
            @constraint(m, u .<= 1)  # u ≤ e

            eeT = ones(n, n)
            # (e − u)(e − u)ᵀ ≥ 0  →  U − u eᵀ − e uᵀ + e eᵀ ≥ 0
            @constraint(m, U .- (u * e') .- (e * u') .+ eeT .>= 0)

            if ℓ > 0
                # (b − A x)(e − u)ᵀ ≥ 0
                @constraint(m, (b * e') .- (b * u') .- (A * x) * e' .+ A * R .>= 0)
            end
            # (M u − x)(e − u)ᵀ ≥ 0
            @constraint(m, M * (u * e') .- x * e' .- M * U .+ R .>= 0)
            # (M u + x)(e − u)ᵀ ≥ 0
            @constraint(m, M * (u * e') .+ x * e' .- M * U .- R .>= 0)

            # I2-only: (ρ − eᵀu)(e − u)ᵀ ≥ 0  →  ρ e − ρ u − (eᵀu) e + U e ≥ 0
            if startswith(variant, "I")
                @constraint(m, ρ .* e .- ρ .* u .- sum(u) .* e .+ U * e .>= 0)
            end
        end
    end

    # build-only path
    if build_only
        return m
    end

    # solve
    optimize!(m)
    term = termination_status(m)

    if term == MOI.OPTIMAL
        xval = value.(x); uval = value.(u)
        if variant == "EXACT"
            return (:OPTIMAL, objective_value(m), xval, uval)
        else
            Xv = value.(X); Rv = value.(R); Uv = value.(U)

            # Optional: switch to simplex+crossover if you need duals (LP-like export)
            set_optimizer_attribute(m, "Method", 0)
            set_optimizer_attribute(m, "Crossover", 1)
            optimize!(m)

            return (:OPTIMAL, objective_value(m), xval, uval, Xv, Rv, Uv)
        end
    elseif term == MOI.DUAL_INFEASIBLE
        return (:UNBOUNDED, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
        return (:INF_OR_UNBD, nothing, nothing, nothing, nothing, nothing, nothing)
    elseif term == MOI.INFEASIBLE
        return (:INFEASIBLE, nothing, nothing, nothing, nothing, nothing, nothing)
    else
        return (:OTHER, term, nothing, nothing, nothing, nothing, nothing)
    end
end

# ---------- inspect ----------
"""
    inspect_model(data, variant)

Builds (does not solve) the model, writes an LP file, and prints bounds and counts.
"""
function inspect_model(data, variant::String; optimizer=Gurobi.Optimizer)
    m = build_and_solve(data; variant=variant, build_only=true, optimizer=optimizer)
    fname = "$(variant)_model.lp"
    write_to_file(m, fname)
    println("\n$fname written.\n")

    println("Variable bounds:")
    for v in all_variables(m)
        lb = has_lower_bound(v) ? lower_bound(v) : "none"
        ub = has_upper_bound(v) ? upper_bound(v) : "none"
        println("  $(name(v))  LB = $(lb)   UB = $(ub)")
    end

    println("\nConstraints:")
    for con in all_constraints(m; include_variable_in_set_constraints = true)
        println("  ", con)
    end
    println()
    println("Number of constraints: ", num_constraints(m; count_variable_in_set_constraints = true))
    return nothing
end


function demo()
    data = Dict(
        "n"  => 4,
        "rho"=> 3.0,
        "Q0" => zeros(4,4),
        "q0" => [-1.0, 0.0, 0.0, 0.0],
        "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
        "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
        "A"  => nothing, "b"=>nothing,
        "H"  => nothing, "h"=>nothing,
        "M"  => I(4)
    )

    for var in ("EXACT","E","EU","I","IU")
        res = build_and_solve(data; variant=var)
        if res[1] == :OPTIMAL
            if var == "EXACT"
                _, obj, x, u = res
                @printf("%-5s → obj %.3f, x=%s, u=%s\n\n", var, obj, x, u)
            else
                _, obj, x, u, X, R, U = res
                @printf("%-5s → obj %.3f, x=%s, u=%s\n", var, obj, x, u)
                println("X =\n", X, "\nR =\n", R, "\nU =\n", U, "\n")
            end
        else
            @printf("%-5s → status = %s\n", var, string(res[1]))
        end
    end

    inspect_model(data, "IU")
end

# if abspath(PROGRAM_FILE) == @__FILE__
#     demo()
# end


demo()