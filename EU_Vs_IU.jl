using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface

include("RLT_bigM_4variants.jl")      # primal RLTs
include("RLT_duals_bigM_4variants.jl")# dual RLTs

using .RLTBigM: prepare_instance, add_FC!, add_FI!, add_FU!, add_FIU!
using .RLTDuals: add_dual_common_vars!, add_dual_FE_vars!, add_dual_FU_vars!,
                 add_Cx!, add_CX!, add_Cu!, add_CR!, add_CU!

function build_gap_model(data; optimizer=Gurobi.Optimizer, qmax)
    params0 = prepare_instance(data)
    m, k = length(params0.Qi), length(params0.Pi)
    params = merge(params0,(;m,k))
    n = params.n

    model = Model(optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_attribute(model, "TimeLimit", 300)

    # q0 is a decision vector but must be bounded/normalized
    @variable(model, q0_var[1:n],Int)
    @constraint(model, -qmax .<= q0_var)
    @constraint(model,  q0_var .<= qmax)

    # ---------- IU primal block ----------
    @variable(model, x[1:n])
    @variable(model, u[1:n])
    @variable(model, X[1:n,1:n], Symmetric)
    @variable(model, R[1:n,1:n])
    @variable(model, U[1:n,1:n], Symmetric)

    add_FC!(model, x,u,X,R,U,params)
    add_FI!(model, x,u,X,R,U,params)
    add_FU!(model, x,u,X,R,U,params)
    add_FIU!(model, u,U,params)

    # IU objective uses q0_var
    iu_obj = @expression(model, 0.5*sum(params.Q0[i,j]*X[i,j] for i=1:n, j=1:n) + dot(q0_var, x))

    # ---------- EU dual block ----------
    # IMPORTANT: feed q0_var into EU via a params "override"
    paramsEU = merge(params, (; q0 = q0_var))

    common = add_dual_common_vars!(model, paramsEU)
    multipliers = Dict(pairs(common))
    merge!(multipliers, Dict(pairs(add_dual_FE_vars!(model, paramsEU))))
    merge!(multipliers, Dict(pairs(add_dual_FU_vars!(model, paramsEU))))

    add_Cx!(model, paramsEU, paramsEU.M, multipliers; variant="EU")
    add_CX!(model, paramsEU, multipliers)
    add_Cu!(model, paramsEU, paramsEU.M, multipliers; variant="EU")
    add_CR!(model, paramsEU, paramsEU.M, multipliers; variant="EU")
    add_CU!(model, paramsEU, paramsEU.M, multipliers; variant="EU")

    @unpack ρ,b,h,e = paramsEU
    eu_obj = sum(multipliers[:α][i]*paramsEU.ri[i] for i=1:length(paramsEU.Qi); init=0.0) +
             sum(multipliers[:β][j]*paramsEU.si[j] for j=1:length(paramsEU.Pi); init=0.0) -
             dot(multipliers[:μ], b) - dot(multipliers[:λ],h) -
             0.5*b'*multipliers[:Θ]*b - ρ*multipliers[:σ] -
             dot(multipliers[:κ],e) - dot(e,multipliers[:Ω]'*b) - 0.5*dot(e,multipliers[:Λ]*e)

    @objective(model, Max, eu_obj - iu_obj)

    return model, q0_var
end

# ----------------- Run model ------------------
data = Dict(
    "n"=>4, "rho"=>3.0,
    "Q0"=>zeros(4,4),
    "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
    "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
    "A"=>nothing,"b"=>nothing,
    "H"=>nothing,"h"=>nothing,
    "M"=>I(4)
)

n   = 4
ρ   = 3.0
ℓ   = [-1.0, -0.5, -2.0, -0.3]   # lower bounds
ū   = [ 1.0,  0.8,  0.7,  2.5]   # upper bounds
Mvec = max.(abs.(ℓ), abs.(ū))    # Big-M not tighter than box
Mmat = Diagonal(Mvec)

A = [I(n); -I(n)]
b = vcat(ū, -ℓ)
H = nothing
h = nothing

data3 = Dict(
    "n"  => n,
    "rho"=> ρ,
    "Q0" => zeros(n,n),
    "Qi" => nothing, "qi"=>nothing, "ri"=>nothing,
    "Pi" => nothing, "pi"=>nothing, "si"=>nothing,
    "A"  => A, "b"=> b,
    "H"  => H, "h"=> h,
    "M"  => Mmat
)


model, q0_var = build_gap_model(data3; qmax=100)
# Write to LP for inspection
write_to_file(model, "gap_model.lp")
optimize!(model)

if has_values(model)
    println("Gap (EU_dual - IU_primal) = ", objective_value(model))
    println("q0 found = ", value.(q0_var))
else
    println("No feasible solution found.")
end
