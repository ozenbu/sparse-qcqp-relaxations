module IU_Hyperplane

using LinearAlgebra
using JuMP
using Gurobi
using Parameters
const MOI = JuMP.MOI   # <-- add this

include("RLT_bigM_4variants.jl")
include("RLT_duals_bigM_4variants.jl")

using .RLTBigM: prepare_instance, build_and_solve
using .RLTDuals:
    add_dual_common_vars!, add_dual_FI_vars!, add_dual_FU_vars!, add_dual_FIU_vars!,
    add_Cx!, add_CX!, add_Cu!, add_CR!, add_CU!

function find_IU_obj_coef(data::Dict, xhat::AbstractVector, Xhat::AbstractMatrix;
                          coeff_bound::Float64 = 1e2,
                          optimizer = Gurobi.Optimizer,
                          verbose::Bool=false)
    p0 = prepare_instance(data)
    m, k = length(p0.Qi), length(p0.Pi)
    params = merge(p0, (; m, k))

    n = params.n
    @assert length(xhat) == n
    @assert size(Xhat) == (n, n)
    e = params.e
    model = Model(optimizer)
    set_silent(model)
    verbose && set_optimizer_attribute(model, "OutputFlag", 1)


    B = coeff_bound
    @variable(model, -B .<= q0[1:n] .<= B)
    @variable(model, -B .<= Q0[1:n,1:n] .<= B, Symmetric)

    params2 = merge(params, (; q0 = q0, Q0 = Q0))

    common = add_dual_common_vars!(model, params2)
    multipliers = Dict(pairs(common))
    merge!(multipliers, Dict(pairs(add_dual_FI_vars!(model, common.σ, params2))))
    merge!(multipliers, Dict(pairs(add_dual_FU_vars!(model, params2))))
    merge!(multipliers, Dict(pairs(add_dual_FIU_vars!(model, params2))))

    add_Cx!(model, params2, params2.M, multipliers; variant="IU")
    add_CX!(model, params2, multipliers)
    add_Cu!(model, params2, params2.M, multipliers; variant="IU")
    add_CR!(model, params2, params2.M, multipliers; variant="IU")
    add_CU!(model, params2, params2.M, multipliers; variant="IU")

    @unpack ρ, b, h, e = params2
    dual_obj =
        sum(multipliers[:α][i] * params2.ri[i] for i in 1:m; init=0.0) +
        sum(multipliers[:β][j] * params2.si[j] for j in 1:k; init=0.0) -
        dot(multipliers[:μ], b) - dot(multipliers[:λ], h) -
        0.5 * b' * multipliers[:Θ] * b - ρ * multipliers[:σ]

    dual_obj -= ρ * dot(multipliers[:ϖ], b) + ρ^2 * multipliers[:ζ]
    dual_obj -= dot(multipliers[:κ], e) + dot(e, multipliers[:Ω]' * b) + 0.5 * dot(e, multipliers[:Λ] * e)
    dual_obj -= ρ * dot(multipliers[:ι], e)

    primal_val = @expression(model,
        0.5 * sum(Q0[i,j]*Xhat[i,j] for i=1:n, j=1:n) + sum(q0[i]*xhat[i] for i=1:n))

    @constraint(model, primal_val == dual_obj)

    @objective(model, Min, primal_val*2 + 2*primal_val)
    set_optimizer_attribute(model, "NonConvex", 2)

    optimize!(model)
    status = termination_status(model)
    if status == MOI.OPTIMAL
        return value.(Q0), value.(q0), value(primal_val), status
    else
        return nothing, nothing, nothing, status
    end
end


A = [ 1.0  0.0  0.0  0.0
    0.0  1.0  0.0  0.0
    0.0  0.0  1.0  0.0
    0.0  0.0  0.0  1.0
    -1.0  0.0  0.0  0.0
    0.0 -1.0  0.0  0.0
    0.0  0.0 -1.0  0.0
    0.0  0.0  0.0 -1.0 ]

b = [1.0, 1.0, 1.0, 1.0,   1.0, 1.0, 1.0, 1.0]
data = Dict(
    "n"=>4, "rho"=>3.0,
    "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
    "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
    "A"=>A,"b"=>b,
    "H"=>nothing,"h"=>nothing,
    "M"=>2*I(4)
)

xhat = [0.43401942119225423, 0.5339274947763765, 0.6451099803289744, 0.5252845535748437]
Xhat = [0.43401942119225423 0.0 0.0791294015212286 0.0;
        0.0 0.5339274947763765 0.18222762123382907 0.05921204835122018;
        0.0791294015212286 0.18222762123382907 0.6451099803289744 0.5252845535748437;
        0.0 0.05921204835122018 0.5252845535748437 0.5252845535748437]




xhat =[0.5, 0.5, 0.5, 0.0]  
Xhat = [
  0.5    0.0  0.25  -0.5;
  0.0    0.5  0.0   -0.5;
  0.25   0.0  0.5    0.5;
 -0.5   -0.5  0.5    1.0]


xhat =[0.5, 0.5, 0.5, 0.5]  
Xhat = [ 0.5  0.5  0.0  0.0
 0.5  0.5  0.0  0.0
 0.0  0.0  0.5  0.0
 0.0  0.0  0.0  0.5]


# ---- solve meta-LP for (Q0, q0) ----
Q0, q0, obj_val, stat = IU_Hyperplane.find_IU_obj_coef(
    data, xhat, Xhat;
    coeff_bound = 1e1,
    optimizer = Gurobi.Optimizer,
    verbose = false,
)
println("Meta-LP status = $stat")
println("Support value at (x̂, X̂) = $obj_val")
println("Q0 = ", round.(Q0; digits=4))
println("q0 = ", round.(q0; digits=4))

end # module
