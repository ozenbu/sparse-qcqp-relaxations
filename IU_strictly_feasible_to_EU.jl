using LinearAlgebra
using Parameters
using JuMP
using Gurobi
using MathOptInterface
using Random
include("RLT_bigM_4variants.jl")
using .RLTBigM: prepare_instance, add_FC!, add_FI!, add_FU!, add_FIU!, add_FE!

# --- tiny printer like your snippet ---
function dump_vars(x, u, X, R, U; label="")
    if !isempty(label) println(label) end
    println("x=", x, "  u=", u)
    println("X =")
    show(stdout, "text/plain", X)
    println("\n")
    println("R =")
    show(stdout, "text/plain", R)
    println("\n")
    println("U =")
    show(stdout, "text/plain", U)
    println("\n")
end

"Step 1: sample x̂ in (0,1)^n with eᵀx̂ < ρ"
function sample_xhat(n::Int, ρ::Float64; seed::Union{Nothing,Int}=nothing)
    if seed !== nothing; Random.seed!(seed); end
    r = rand(n)                     # (0,1)
    margin = rand()                 # in [0,1) and ρ ≥ 1 ⇒ margin < ρ
    α = min(1.0, (ρ - margin) / sum(r))
    return α .* r
end

"Step 2: Minimize eᵀu subject to IU-feasibility with x fixed to x̂ --"
function solve_step2_IU(data::Dict, xhat::AbstractVector; optimizer=Gurobi.Optimizer, verbose=false)
    params = prepare_instance(data)
    n = params.n
    @assert length(xhat) == n

    "Introduce variables"
    m = Model(optimizer); JuMP.set_silent(m); verbose && set_optimizer_attribute(m,"OutputFlag",1)
    @variable(m, x[1:n]); @variable(m, u[1:n])
    @variable(m, X[1:n,1:n], Symmetric); @variable(m, R[1:n,1:n]); @variable(m, U[1:n,1:n], Symmetric)

    @constraint(m, x .== xhat)         # x = x̂
    add_FC!(m, x, u, X, R, U, params)  # common block
    add_FI!(m, x, u, X, R, U, params)  # I
    add_FU!(m, x, u, X, R, U, params)  # U
    add_FIU!(m, u, U, params)          # I×U coupling

    @objective(m, Min, sum(u))         # min eᵀu
    write_to_file(m, "step2.lp")       # LP format
    optimize!(m)

    st = termination_status(m)
    return st == MOI.OPTIMAL ?
        (:OPTIMAL, value.(x), value.(u), value.(X), value.(R), value.(U), objective_value(m)) :
        (st, nothing, nothing, nothing, nothing, nothing, nothing)
end

"Step 3 (EU): keep x = x̂ and X = X̂; set u = û + s, s ≥ 0; maximize min s_j"
function solve_step3_EU(data::Dict, xhat::AbstractVector, Xhat::AbstractMatrix;
                        optimizer=Gurobi.Optimizer, verbose=false, tol=1e-3)
    params = prepare_instance(data)
    n = params.n
    @assert length(xhat) == n && size(Xhat) == (n,n) 

    m2 = Model(optimizer); JuMP.set_silent(m2); verbose && set_optimizer_attribute(m2,"OutputFlag",1)
    @variable(m2, x[1:n]); @variable(m2, u[1:n]);
    @variable(m2, X[1:n,1:n], Symmetric); @variable(m2, R[1:n,1:n]); @variable(m2, U[1:n,1:n], Symmetric)

    @constraint(m2, x - xhat .<= ones(n)*tol)
    @constraint(m2, x - xhat .>= -ones(n)*tol)

    @constraint(m2, X - Xhat .<= ones(n,n)*tol)
    @constraint(m2, X - Xhat .>= -ones(n,n)*tol)

    add_FC!(m2, x, u, X, R, U, params)
    add_FE!(m2, x, u, X, R, U, params)
    add_FU!(m2, x, u, X, R, U, params)

    @objective(m2, Max, 0)
    write_to_file(m2, "step3.lp")
    optimize!(m2)
    compute_conflict!(m2)

    if get_attribute(m2, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        iis_model, _ = copy_conflict(m2)
        print(iis_model)
    end
    st = termination_status(m2)
    return st == MOI.OPTIMAL ?
        (:OPTIMAL, value.(x), value.(u), value.(X), value.(R), value.(U), objective_value(m2)) :
        (st, nothing, nothing, nothing, nothing, nothing, nothing)  
end

"Step1→Step2→Step3"
function run_three_steps(data::Dict; xhat::Union{Nothing,AbstractVector}=nothing,
    Xhat::Union{Nothing,AbstractMatrix}=nothing, seed=7, verbose=false)
    
    p = prepare_instance(data)
    if xhat === nothing && Xhat === nothing
        xhat = sample_xhat(p.n, p.ρ; seed=seed)
    end
    println("Step1: x̂=", xhat, "  sum(x̂)=", sum(xhat), " < ρ=", p.ρ)

    if Xhat === nothing
        st2, x_fix, uhat, Xhat, Rhat, Uhat, obj2 = solve_step2_IU(data, xhat; verbose)
        if st2 == :OPTIMAL
            println("Step2(IU): status=$st2,  eᵀû=$(obj2)")
            dump_vars(x_fix, uhat, Xhat, Rhat, Uhat; label="— Step 2 solution —")
        else
            println("Step2(IU): status=$st2")
            println("→ Step 2 is not OPTIMAL (status=$st2). IU with x fixed appears infeasible or non-optimal; stopping.")
            return (; status2=st2)
        end
    end
    println("X̂=", Xhat)


    st3, xEU, uEU, XEU, REU, UEU, objEU = solve_step3_EU(data, xhat, Xhat; verbose=false)
    if st3 == :OPTIMAL
        println("Step3(EU): status=$st3,   sum(uEU)=", sum(uEU), " (should equal ρ)")
        dump_vars(xEU, uEU, XEU, REU, UEU; label="— Step 3 solution —")
        return (; xhat, Xhat, uEU, REU, UEU, status3=st3)
    else
        println("Step3(EU): status=$st3")
        println("→ Step 3 is INFEASIBLE for EU with x and X fixed.")
        println("  This shows (x̂, X̂) is in IU’s projection but not in EU’s projection under the fixed-X test.")
        return (; xhat, Xhat, status3=st3)
    end
end



data1 = Dict(
    "n"=>4, "rho"=>3.0,
    "Q0"=>zeros(4,4), "q0"=>[-1.0,0.0,0.0,0.0],
    "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
    "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
    "A"=>nothing,"b"=>nothing,
    "H"=>nothing,"h"=>nothing,
    "M"=>I(4)
)

data_test = Dict(
"n"  => 4,
"rho"=> 3.0,
"Q0" => [-50.0  50.0  50.0  50.0;
            50.0 -50.0  50.0  50.0;
            50.0  50.0 -50.0   0.0;
            50.0  50.0   0.0   0.0],
"q0" => [-100.0, -100.0, -100.0, -75.0],
"Qi" => nothing, "qi" => nothing, "ri" => nothing,
"Pi" => nothing, "pi" => nothing, "si" => nothing,
"A"  => nothing, "b"  => nothing,
"H"  => nothing, "h"  => nothing,
"M"  => I(4)
)


xhat =[0.5, 0.5, 0.5, 0.0]  
Xhat = [
  0.5    0.0  0.25  -0.5;
  0.0    0.5  0.0   -0.5;
  0.25   0.0  0.5    0.5;
 -0.5   -0.5  0.5    1.0]



st3, xEU, uEU, XEU, REU, UEU, objEU = solve_step3_EU(data_test, xhat, Xhat; verbose=false)
if st3 == :OPTIMAL
    println("Step3(EU): status=$st3,   sum(uEU)=", sum(uEU), " (should equal ρ)")
    dump_vars(xEU, uEU, XEU, REU, UEU; label="— Step 3 solution —")
    return (; xhat, Xhat, uEU, REU, UEU, status3=st3)
else
    println("Step3(EU): status=$st3")
    println("→ Step 3 is INFEASIBLE for EU with x and X fixed.")
    println("  This shows (x̂, X̂) is in IU’s projection but not in EU’s projection under the fixed-X test.")
    return (; xhat, Xhat, status3=st3)
end



# Testing if etx = rho/2 => EU infeasible 
println("Testing if etx = rho/2 => EU infeasible ")
xhat = [0.5, 0.5, 0.5, 0.5, 0.0]  
Xhat = nothing
data_test = Dict(
"n"  => 5,
"rho"=> 4.0,
"Qi" => nothing, "qi" => nothing, "ri" => nothing,
"Pi" => nothing, "pi" => nothing, "si" => nothing,
"A"  => nothing, "b"  => nothing,
"H"  => nothing, "h"  => nothing,
"M"  => I(5)
)

res = run_three_steps(data_test; xhat=xhat, Xhat=Xhat, seed=nothing, verbose=false)
