module RLT_SDP_Combo
using MosekTools
using JuMP, LinearAlgebra, Parameters
const MOI = JuMP.MOI


# Bring in the building blocks
using ..RLTBigM: prepare_instance, add_FC!, add_FE!, add_FI!, add_FU!, add_FIU!
using ..SDPBigM: add_SDP_lift!

function build_RLT_SDP_model(
    data::Dict{String,Any};
    variant::String = "EU",
    optimizer=MosekTools.Optimizer,
    build_only::Bool = false
)
    # Unpack instance
    params = prepare_instance(data)
    @unpack n, ρ, Q0, q0, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e = params

    m = Model(optimizer)

    # Decision variables
    @variable(m, x[1:n])
    @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    # Objective: 0.5*⟨Q0,X⟩ + q0ᵀx
    @objective(m, Min, 0.5 * sum(Q0[i,j]*X[i,j] for i=1:n, j=1:n) + sum(q0[i]*x[i] for i=1:n))

    # Core RLT block
    add_FC!(m, x, u, X, R, U, params)

    # E vs I
    if startswith(variant, "E")
        add_FE!(m, x, u, X, R, U, params)
    elseif startswith(variant, "I")
        add_FI!(m, x, u, X, R, U, params)
    else
        error("Unknown variant: $variant (expected prefix 'E' or 'I')")
    end

    # U extras (+ I×U coupling for IU)
    if endswith(variant, "U")
        add_FU!(m, x, u, X, R, U, params)
        if startswith(variant, "I")
            add_FIU!(m, u, U, params)
        end
    end

    # SDP lifting block (does not add original constraints; only PSD-Z and link equalities)
    add_SDP_lift!(m, x, u, X, R, U; n=n)

    if build_only
            return m
    end

    optimize!(m)
    return m
end

"""
    solve_RLT_SDP(data; variant="EU", optimizer)

Helper: builds and solves, returns (status, obj, x, u, X, R, U).
"""
function solve_RLT_SDP(
    data::Dict{String,Any};
    variant::String = "EU",
    optimizer=MosekTools.Optimizer
)
    m = build_RLT_SDP_model(data; variant=variant, optimizer=optimizer, build_only=false)
    st = termination_status(m)

    if st == MOI.OPTIMAL || st == MOI.LOCALLY_OPTIMAL
        x = value.(m[:x]); u = value.(m[:u])
        X = value.(m[:X]); R = value.(m[:R]); U = value.(m[:U])
        return (st, objective_value(m), x, u, X, R, U)
    else
        return (st, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

function demo_RLT_SDP_Combo()
    
    Q0 = [
            0.0003   0.1016   0.0316   0.0867;
            0.1016   0.0020   0.1001   0.1059;
            0.0316   0.1001  -0.0005  -0.0703;
            0.0867   0.1059  -0.0703  -0.1063
        ]
    q0 = [-0.1973, -0.2535, -0.1967, -0.0973]
    data = Dict(
        "n"=>4, "rho"=>3.0,
        "Q0"=> Q0,
        "q0"=> q0,
        "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
        "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
        "A"=>nothing,"b"=>nothing,
        "H"=>nothing,"h"=>nothing,
        "M"=>I(4)
    )
    for v in ("E","EU","I","IU")
        m = build_RLT_SDP_model(data; variant=v, optimizer=MosekTools.Optimizer)
        optimize!(m)
        println("variant=$v → ", termination_status(m), "  obj=", objective_value(m))
    end
end

end # module

if isinteractive() 
    using .RLT_SDP_Combo
    RLT_SDP_Combo.demo_RLT_SDP_Combo()
end