module SDPBigM

using JuMP, LinearAlgebra
using MosekTools
using Parameters
using MathOptInterface
const MOI = MathOptInterface

# re-use instance parsing from RLTBigM
using ..RLTBigM: prepare_instance

"Add the lifted PSD block"
function add_SDP_lift!(model::Model, x, u, X, R, U; n::Int)
    # Z = [ 1   x'      u'
    #       x   X       R
    #       u   R'      U ]
    @variable(model, Z[1:(2n+1), 1:(2n+1)], Symmetric)
    @constraint(model, Z .== [1.0 x' u'; x X R; u R' U])
    @constraint(model, Z in PSDCone())
    return Z
end

function build_SDP_model(data::Dict{String, Any};
                         variant::String="E",
                         build_only::Bool=false,
                         optimizer=MosekTools.Optimizer)

    vars = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si,
            A, b, ℓ, H, h, η, Mminus, Mplus, e = vars

    model = Model(optimizer)

    # Decision variables
    @variable(model, x[1:n])
    @variable(model, u[1:n])
    @variable(model, X[1:n, 1:n], Symmetric)
    @variable(model, U[1:n, 1:n], Symmetric)
    @variable(model, R[1:n, 1:n])

    # Objective: 0.5⟨Q0, X⟩ + q0ᵀx
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

    # Big-M links: -M⁻ u ≤ x ≤ M⁺ u
    @constraint(model, [i=1:n],  x[i] <=  Mplus[i,i] * u[i])
    @constraint(model, [i=1:n], -x[i] <=  Mminus[i,i] * u[i])

    # Cardinality (E or I)
    if variant == "E"
        @constraint(model, sum(u) == ρ)
    elseif variant == "I"
        @constraint(model, sum(u) <= ρ)
    else
        error("Unknown variant: $variant (must be \"E\" or \"I\")")
    end

    # diag(U) = u
    @constraint(model, [i=1:n], U[i, i] == u[i])

    # SDP lifting block (modularized)
    add_SDP_lift!(model, x, u, X, R, U; n=n)

    if build_only
        return model
    end

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
    Q0 = Q0 * 20000
    q0 = [-0.1973, -0.2535, -0.1967, -0.0973] * 20000

    data_test = Dict(
        "n"   => 4,
        "rho" => 3.0,
        "Q0"  => Q0,
        "q0"  => q0,
        "Qi"  => nothing, "qi" => nothing, "ri" => nothing,
        "Pi"  => nothing, "pi" => nothing, "si" => nothing,
        "A"   => nothing, "b"  => nothing,
        "H"   => nothing, "h"  => nothing,
        # Here we use the old-style single M;
        # internally this becomes Mminus = Mplus = I(4)
        "M"   => I(4)
        # If you want asymmetric bounds, you can instead do:
        # "Mminus" => Diagonal([2.0, 1.0, 1.0, 0.5]),
        # "Mplus"  => Diagonal([5.0, 3.0, 2.0, 1.0])
    )

    for variant in ["E", "I"]
        println("\n🔹 Solving SDP variant: $variant")
        model = build_SDP_model(data_test; variant=variant)

        optimize!(model)

        status = termination_status(model)
        println("Status = ", status)

        if status == MOI.OPTIMAL
            println("Objective = ", objective_value(model))

            # Access variables by name from the model
            x_var = model[:x]
            u_var = model[:u]

            println("x = ", value.(x_var))
            println("u = ", value.(u_var))
        else
            println("Optimization failed for variant $variant")
        end
    end
end

end  # module SDPBigM

#=
if isinteractive()
    using .SDPBigM
    SDPBigM.demo_SDP()
end
=#

