module RLTDuals

using JuMP, LinearAlgebra, Parameters, Gurobi
const MOI = JuMP.MOI
using ..RLTBigM: prepare_instance 

# --- helpers for empty JuMP arrays ---
const VR = JuMP.VariableRef
zero_vec(n::Int) = Vector{VR}(undef, n)
zero_mat(m::Int, n::Int) = Array{VR}(undef, m, n)

# ---------------- multipliers ----------------
function add_dual_common_vars!(model, params)
    @unpack n, m, k, ℓ, η = params
    
    @variable(model, γ[1:n] >= 0)       # x <= Mu
    @variable(model, δ[1:n] >= 0)       # -Mu <= x
    @variable(model, σ)                 # cardinality
    @variable(model, τ[1:n])            # Uii = ui
    @variable(model, ΓUU[1:n,1:n] >= 0) # big M RLT
    @variable(model, ΓLL[1:n,1:n] >= 0)
    @variable(model, ΓUL[1:n,1:n] >= 0)
    @constraint(model, ΓUU .== ΓUU')    # symmetry 
    @constraint(model, ΓLL .== ΓLL')

    # quadratic inequality
    if m > 0
        @variable(model, α[1:m] >= 0)
    else
        α = zero_vec(0)
    end
    
    # quadratic equality
    if k > 0
        @variable(model, β[1:k])
    else
        β = zero_vec(0)
    end

    if ℓ > 0
        @variable(model, μ[1:ℓ] >= 0)          # Ax <= b
        @variable(model, Θ[1:ℓ,1:ℓ] >= 0)      # Ax <= b self-RLT 
        @variable(model, ΓAU[1:ℓ,1:n] >= 0)    # Ax <= b and big M RLTs
        @variable(model, ΓAL[1:ℓ,1:n] >= 0)
        @constraint(model, Θ .== Θ')           # symmetry
    else
        μ   = zero_vec(0)
        Θ   = zero_mat(0, 0)
        ΓAU = zero_mat(0, n)
        ΓAL = zero_mat(0, n)
    end

    if η > 0
        @variable(model, λ[1:η])
        @variable(model, Φ[1:η,1:n])
        @variable(model, Ξ[1:η,1:n])
    else
        λ   = zero_vec(0)
        Φ   = zero_mat(0, n)
        Ξ   = zero_mat(0, n)
    end

    return (; α,β,μ,λ,γ,δ,σ,τ, Θ,Φ,Ξ, ΓAU,ΓAL, ΓUU,ΓLL,ΓUL)
end

function add_dual_FE_vars!(model, params)
    @unpack n = params
    @variable(model, ϕ[1:n])  # cardinality and x
    @variable(model, ψ[1:n])  # cardinality and u
    return (; ϕ, ψ)
end

function add_dual_FI_vars!(model, σ, params)
    @unpack n, ℓ = params
    @constraint(model, σ >= 0)

    # cardinality and Ax <= b
    if ℓ > 0
        @variable(model, ϖ[1:ℓ] >= 0)
    else
        ϖ = zero_vec(0)
    end
    
    @variable(model, χp[1:n] >= 0)  # cardinality and big M
    @variable(model, χm[1:n] >= 0)
    @variable(model, ζ >= 0)        # cardinality self-RLT
    return (; ϖ, χp, χm, ζ)
end

function add_dual_FU_vars!(model, params)
    @unpack n, ℓ = params

    @variable(model, κ[1:n] >= 0)         # u <= e
    @variable(model, Λ[1:n,1:n] >= 0)     # u <= e self-RLT
    @constraint(model, Λ .== Λ') 

    # u <= e and Ax <= b
    if ℓ > 0
        @variable(model, Ω[1:ℓ,1:n] >= 0)
    else
        Ω = zero_mat(0, n)
    end

    @variable(model, Πp[1:n,1:n] >= 0)    # u <= e and big M
    @variable(model, Πm[1:n,1:n] >= 0)
    return (; κ,Λ,Ω,Πp,Πm)
end

function add_dual_FIU_vars!(model, params)
    @unpack n = params
    @variable(model, ι[1:n] >= 0)
    return (; ι)
end


# ---------------- stationarity blocks ----------------
function add_Cx!(model, params, M, multipliers; variant)
    @unpack n, ρ, q0, qi, pi, A, b, H, h, e, ℓ, η, m, k = params
    @unpack α,β,μ,λ,γ,δ,Θ,Φ,ΓAU,ΓAL = multipliers
    Cx = @expression(model, q0 + γ - δ)
    for i in 1:m
        Cx .+= α[i]*qi[i]
    end
    for j in 1:k
        Cx .+= β[j]*pi[j]
    end
    if ℓ > 0
        Cx .+= A' * μ + A' * Θ * b + (ΓAU' - ΓAL') * b
    end
    if η > 0
        Cx .+= H' * λ - Φ' * h
    end
    if variant in ["E","EU"]
        Cx .+= -ρ * multipliers[:ϕ]
    end
    if variant in ["EU","IU"]
        U = multipliers
        Cx .+= -A' * (U[:Ω] * e) + (U[:Πp] - U[:Πm]) * e
    end
    if variant in ["I","IU"]
        Ivars = multipliers
        Cx .+= ρ*A'*Ivars[:ϖ] + ρ*(Ivars[:χp] - Ivars[:χm])
    end
    @constraint(model, Cx .== 0)
end

function add_CX!(model, params, multipliers)
    @unpack n, Q0, Qi, Pi, A, H, m, k, ℓ, η = params
    @unpack α,β,Θ,ΓAU,ΓAL,ΓUU,ΓLL,ΓUL,Φ = multipliers
    CX = @expression(model, ΓUU + ΓLL - ΓUL - ΓUL')
    for i in 1:m
        CX .+= -α[i]*Qi[i]
    end
    for j in 1:k
        CX .+= -β[j]*Pi[j]
    end
    if ℓ > 0
        CX .+= A' * Θ * A + A' * (ΓAU-ΓAL) + (ΓAU'-ΓAL')*A
    end
    if η > 0
        CX .+= H' * Φ - Φ' * H
    end
    @constraint(model, CX .== Q0)
end

function add_Cu!(model, params, M, multipliers; variant)
    @unpack n, ρ, A, b, H, h, e, ℓ, η = params
    @unpack γ,δ,σ,τ,ΓAU,ΓAL,Ξ = multipliers
    Mmat = Matrix(M)
    Cu = @expression(model, -Mmat*(γ+δ) + σ*e - τ)
    if ℓ > 0
        Cu .+= -Mmat*(ΓAU' + ΓAL')*b
    end
    if η > 0
        Cu .+= -Ξ' * h
    end
    if variant in ["E","EU"]
        Cu .+= -ρ * multipliers[:ψ]
    end
    if variant in ["EU","IU"]
        U = multipliers
        Cu .+= U[:κ] + U[:Ω]'*b - Mmat*(U[:Πp] + U[:Πm])*e + U[:Λ]*e
    end
    if variant in ["I","IU"]
        Ivars = multipliers
        Cu .+= -ρ*Mmat*(Ivars[:χp]+Ivars[:χm]) + 2ρ*Ivars[:ζ]*e
        if ℓ > 0
            Cu .+= (sum(Ivars[:ϖ][r]*b[r] for r=1:ℓ))*e
        end
    end
    if variant=="IU"
        ι = multipliers[:ι]
        Cu .+= ρ*ι + (sum(ι))*e
    end
    @constraint(model, Cu .== 0)
end

function add_CR!(model, params, M, multipliers; variant)
    @unpack n, A, H, e, ℓ, η = params
    @unpack ΓUU,ΓLL,ΓUL,ΓAU,ΓAL,Ξ = multipliers
    Mmat = Matrix(M)
    CR = @expression(model, (ΓUU-ΓLL+ΓUL-ΓUL')*Mmat)
    if ℓ > 0
        CR .+= A' * (ΓAU+ΓAL) * Mmat
    end
    if η > 0
        CR .+= H' * Ξ
    end
    if variant in ["E","EU"]
        CR .+= multipliers[:ϕ]*e'
    end
    if variant in ["EU","IU"]
        U = multipliers
        CR .+= -A'*U[:Ω] - U[:Πp] + U[:Πm]
    end
    if variant in ["I","IU"]
        Ivars = multipliers
        CR .+= -A'*Ivars[:ϖ]*e' - (Ivars[:χp]-Ivars[:χm])*e'
    end
    @constraint(model, CR .== 0)
end

function add_CU!(model, params, M, multipliers; variant)
    @unpack n, e = params
    @unpack ΓUU,ΓLL,ΓUL,τ = multipliers
    Mmat = Matrix(M)
    CU = @expression(model, -0.5*(Mmat*(ΓUU+ΓLL+ΓUL+ΓUL')*Mmat))
    for i=1:n
        CU[i,i] += τ[i]
    end
    if variant in ["E","EU"]
        CU .+= 0.5*(multipliers[:ψ]*e' + e*multipliers[:ψ]')
    end
    if variant in ["EU","IU"]
        U = multipliers
        CU .+= -0.5*U[:Λ] + 0.5*(Mmat*(U[:Πp]+U[:Πm]) + (U[:Πp]'+U[:Πm]')*Mmat)
    end
    if variant in ["I","IU"]
        Ivars = multipliers
        CU .+= -Ivars[:ζ]*(e*e') + 0.5*(Mmat*(Ivars[:χp]+Ivars[:χm])*e' + e*(Ivars[:χp]+Ivars[:χm])'*Mmat)
    end
    if variant=="IU"
        ι = multipliers[:ι]
        CU .+= -0.5*(ι*e' + e*ι')
    end
    @constraint(model, CU .== 0)
end

# ---------------- build dual ----------------
function build_dual(data::Dict; variant="E", solve=false, verbose=false)
    params0 = prepare_instance(data)
    m, k = length(params0.Qi), length(params0.Pi)
    params = merge(params0,(;m,k))
    model = Model(Gurobi.Optimizer)
    set_silent(model); verbose && set_optimizer_attribute(model,"OutputFlag",1)

    common = add_dual_common_vars!(model, params)
    multipliers = Dict(Symbol(k)=>getfield(common,Symbol(k)) for k in keys(common))

    if variant in ["E","EU"]
        Evars = add_dual_FE_vars!(model, params)
        merge!(multipliers, Dict(pairs(Evars)))
    end
    if variant in ["I","IU"]
        Ivars = add_dual_FI_vars!(model, common.σ, params)
        merge!(multipliers, Dict(pairs(Ivars)))
    end
    if variant in ["EU","IU"]
        Uvars = add_dual_FU_vars!(model, params)
        merge!(multipliers, Dict(pairs(Uvars)))
    end
    if variant=="IU"
        IUvars = add_dual_FIU_vars!(model, params)
        merge!(multipliers, Dict(pairs(IUvars)))
    end

    add_Cx!(model, params, params.M, multipliers; variant)
    add_CX!(model, params, multipliers)
    add_Cu!(model, params, params.M, multipliers; variant)
    add_CR!(model, params, params.M, multipliers; variant)
    add_CU!(model, params, params.M, multipliers; variant)

    @unpack ρ,b,h,e = params
    obj = sum(multipliers[:α][i]*params.ri[i] for i=1:m; init=0.0) +
          sum(multipliers[:β][j]*params.si[j] for j=1:k; init=0.0) -
          dot(multipliers[:μ], b) - dot(multipliers[:λ],h) - 0.5*b'*multipliers[:Θ]*b - ρ*multipliers[:σ]
    if variant in ["I","IU"]
        obj -= ρ*dot(multipliers[:ϖ],b) + ρ^2*multipliers[:ζ]
    end
    if variant in ["EU","IU"]
        obj -= dot(multipliers[:κ],e) + dot(e,multipliers[:Ω]'*b) + 0.5*dot(e,multipliers[:Λ]*e)
    end
    if variant=="IU"
        obj -= ρ*dot(multipliers[:ι],e)
    end
    @objective(model, Max, obj)

    if solve
        optimize!(model)
        term = termination_status(model)
        if term==MOI.OPTIMAL
            println("dual($variant) obj=", objective_value(model))
        else
            println("dual($variant) status=",term)
        end
    end
    return model
end

# ---------------- demo ----------------
function demo_duals()
    println("Running dual demo...")
    data = Dict("n"=>4,"rho"=>3.0,
        "Q0"=>zeros(4,4),"q0"=>[-1.,0,0,0],
        "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
        "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
        "A"=>nothing,"b"=>nothing,
        "H"=>nothing,"h"=>nothing,
        "M"=>I(4))

    
    q0_opt =[-1.,0,0,0]
    data = Dict(
    "n"=>4, "rho"=>3.0,
    "Q0"=>zeros(4,4),
    "q0"=>q0_opt,
    "Qi"=>nothing,"qi"=>nothing,"ri"=>nothing,
    "Pi"=>nothing,"pi"=>nothing,"si"=>nothing,
    "A"=>nothing,"b"=>nothing,
    "H"=>nothing,"h"=>nothing,
    "M"=>I(4)
    )
    for v in ["E","EU","I","IU"]
        build_dual(data;variant=v,solve=true)
    end
end
#demo_duals()
end # module
