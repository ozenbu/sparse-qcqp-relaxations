using JuMP, LinearAlgebra
using MosekTools
include("RLT_bigM_4variants.jl") 

function build_SDP_model(data::Dict{String, Any}; variant::String="E", build_only::Bool=false, optimizer=MosekTools.Optimizer)
    # Unpack problem data
    vars = unpack_qcqp_data(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e = vars

    model = Model(optimizer)

    # Decision variablesdemo
    @variable(model, x[1:n])
    @variable(model, u[1:n])
    @variable(model, X[1:n, 1:n], Symmetric)
    @variable(model, U[1:n, 1:n], Symmetric)
    @variable(model, R[1:n, 1:n])

    # Objective function
    @objective(model, Min, 0.5 * dot(Q0, X) + dot(q0, x))

    # Quadratic inequality constraints: Qi·X + qiᵗx + ri ≤ 0
    for (Q, q, r) in zip(Qi, qi, ri)
        @constraint(model, 0.5 * dot(Q, X) + dot(q, x) + r <= 0)
    end

    # Quadratic equality constraints: Pi·X + piᵗx + si = 0
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

    # Variable bounds: -Mu ≤ x ≤ Mu
    @constraint(model, [i=1:n],  x[i] <=  M[i,i] * u[i])
    @constraint(model, [i=1:n], -x[i] <=  M[i,i] * u[i])

    # Cardinality constraints
    if variant == "E"
        @constraint(model, sum(u) == ρ)
    
    elseif variant == "I"
        @constraint(model, sum(u) <= ρ)
   
    else
        error("Unknown variant: $variant")
    end

    # diag(U) = u
    @constraint(model, [i=1:n], U[i, i] == u[i])

    # Build lifted PSD matrix Z ∈ S^{2n+1}
    @variable(model, Z[1:(2n+1), 1:(2n+1)], PSD)

    @constraint(model, Z[1, 1] == 1)
    @constraint(model, [i = 1:n], Z[i+1, 1] == x[i])   # Z[2:1+n, 1] == x
    @constraint(model, [j = 1:n], Z[1, j+1] == x[j])   # Z[1, 2:1+n] == x'
    @constraint(model, Z[2:1+n, 2:1+n] == X)
    @constraint(model, Z[2:1+n, n+2:n+1+n] == R)
    @constraint(model, [j = 1:n], Z[1, n + 1 + j] == u[j])
    @constraint(model, Z[n+2:2n+1, n+2:2n+1] .== U)
    @constraint(model, Z[n+2:2n+1, 2:1+n] .== R')

    return model
end

function demo_SDP()
    println("Setting up SDP demo with test instance...")

    data = Dict(
        "n"   => 4,
        "rho" => 3.0,
        "Q0"  => zeros(4, 4),
        "q0"  => [-1.0, 0.0, 0.0, 0.0],
        "Qi"  => nothing, "qi" => nothing, "ri" => nothing,
        "Pi"  => nothing, "pi" => nothing, "si" => nothing,
        "A"   => nothing, "b"  => nothing,
        "H"   => nothing, "h"  => nothing,
        "M"   => I(4)  # identity as Big-M
    )

    for variant in ["E", "I"]
        println("\n🔹 Solving SDP variant: $variant")
        model = build_SDP_model(data; variant=variant)

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            println("✅ Objective = ", objective_value(model))
            println("x = ", value.(model[:x]))
            println("u = ", value.(model[:u]))
        else
            println("❌ Optimization failed for variant $variant")
        end
    end
end


demo_SDP()