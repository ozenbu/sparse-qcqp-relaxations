module SDPBigM

using JuMP, LinearAlgebra
using MosekTools
using Parameters

using ..RLTBigM: prepare_instance

"Add the lifted PSD block"
function add_SDP_lift!(model::Model, x, u, X, R, U; n::Int)
    @variable(model, Z[1:(2n+1), 1:(2n+1)], PSD)

    @constraint(model, Z[1, 1] == 1)
    @constraint(model, [i = 1:n], Z[i+1, 1] == x[i])
    @constraint(model, [j = 1:n], Z[1, j+1] == x[j])
    @constraint(model, Z[2:1+n, 2:1+n] == X)
    @constraint(model, Z[2:1+n, n+2:n+1+n] == R)
    @constraint(model, [j = 1:n], Z[1, n + 1 + j] == u[j])
    @constraint(model, Z[n+2:2n+1, n+2:2n+1] .== U)
    @constraint(model, Z[n+2:2n+1, 2:1+n] .== R')

    return Z
end

function build_SDP_model(data::Dict{String, Any}; variant::String="E", build_only::Bool=false, optimizer=MosekTools.Optimizer)
    vars = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e = vars

    model = Model(optimizer)

    # Decision variables
    @variable(model, x[1:n])
    @variable(model, u[1:n])
    @variable(model, X[1:n, 1:n], Symmetric)
    @variable(model, U[1:n, 1:n], Symmetric)
    @variable(model, R[1:n, 1:n])

    # Objective
    @objective(model, Min, 0.5 * dot(Q0, X) + dot(q0, x))

    # Quadratic inequalities
    for (Q, q, r) in zip(Qi, qi, ri)
        @constraint(model, 0.5 * dot(Q, X) + dot(q, x) + r <= 0)
    end

    # Quadratic equalities
    for (P, p, s) in zip(Pi, pi, si)
        @constraint(model, 0.5 * dot(P, X) + dot(p, x) + s == 0)
    end

    # Linear constraints
    if ℓ > 0
        @constraint(model, A * x .<= b)
    end
    if η > 0
        @constraint(model, H * x .== h)
    end

    # Big-M links
    @constraint(model, [i=1:n],  x[i] <=  M[i,i] * u[i])
    @constraint(model, [i=1:n], -x[i] <=  M[i,i] * u[i])

    # Cardinality (E or I)
    if variant == "E"
        @constraint(model, sum(u) == ρ)
    elseif variant == "I"
        @constraint(model, sum(u) <= ρ)
    else
        error("Unknown variant: $variant")
    end

    # diag(U) = u
    @constraint(model, [i=1:n], U[i, i] == u[i])

    # SDP lifting block (modularized)
    add_SDP_lift!(model, x, u, X, R, U; n)

    return model
end

function demo_SDP()
    println("Setting up SDP demo with test instance...")
    Q0 = [
            0.0003   0.1016   0.0316   0.0867;
            0.1016   0.0020   0.1001   0.1059;
            0.0316   0.1001  -0.0005  -0.0703;
            0.0867   0.1059  -0.0703  -0.1063
        ]
    q0 = [-0.1973, -0.2535, -0.1967, -0.0973]
    data_test = Dict(
        "n"   => 4,
        "rho" => 3.0,
        "Q0"  => Q0,
        "q0"  => q0,
        "Qi"  => nothing, "qi" => nothing, "ri" => nothing,
        "Pi"  => nothing, "pi" => nothing, "si" => nothing,
        "A"   => nothing, "b"  => nothing,
        "H"   => nothing, "h"  => nothing,
        "M"   => I(4)
    )

    for variant in ["E", "I"]
        println("\n🔹 Solving SDP variant: $variant")
        model = build_SDP_model(data_test; variant=variant)

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            println("✅ Objective = ", objective_value(model))
            println("x = ", value.(model[:x]))
            println("u = ", value.(model[:u]))
        else
            println("❌ Optimization failed for variant $variant")
            println(termination_status(model))
        end
    end
end


end

#=
if isinteractive() 
    using .SDPBigM
    SDPBigM.demo_SDP()
end
=#