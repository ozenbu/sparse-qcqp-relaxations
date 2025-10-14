using LinearAlgebra
using Printf

"""
    load_dual_txt(path) -> Dict{Symbol,Any}

Reads a TXT produced like: `name = 1.23` or `name = [ ... ; ... ]`.
Returns a Dict mapping :name => value (scalars / vectors / matrices).
"""
function load_dual_txt(path::AbstractString)
    d = Dict{Symbol,Any}()
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        pos = findfirst(==( '=' ), line)
        pos === nothing && continue
        name = Symbol(strip(line[1:pos-1]))
        rhs  = strip(line[pos+1:end])
        val = try
            Base.invokelatest(eval, Meta.parse(rhs))  # assumes Julia-literal RHS
        catch
            rhs
        end
        d[name] = val
    end
    return d
end


function compute_EU(Q0::AbstractMatrix, q0::AbstractVector, ρ::Real,
                    M::AbstractMatrix, e::AbstractVector,
                    mult::Dict{Symbol,Any}; tol::Real=1e-7)

    γ, δ, σ, τ    = mult[:γ], mult[:δ], mult[:σ], mult[:τ]
    ΓUU, ΓLL, ΓUL = mult[:ΓUU], mult[:ΓLL], mult[:ΓUL]
    ϕ, ψ          = mult[:ϕ], mult[:ψ]
    κ, Λ          = mult[:κ], mult[:Λ]
    Πp, Πm        = mult[:Πp], mult[:Πm]

    CX = ΓUU .+ ΓLL .- ΓUL .- ΓUL' .- Q0
    Cx = q0 .+ γ .- δ .- ρ .* ϕ .+ (Πp .- Πm) * e
    Cu = -(M*(γ .+ δ)) .+ σ .* e .- τ .- ρ .* ψ .+ κ .- (M*(Πp .+ Πm))*e .+ Λ*e
    CR = (ΓUU .- ΓLL .+ ΓUL .- ΓUL')*M .+ ϕ*e' .- Πp .+ Πm
    CU = -0.5 .* (M*(ΓUU .+ ΓLL .+ ΓUL .+ ΓUL')*M) .+
          Diagonal(vec(τ)) .+ 0.5 .* (ψ*e' .+ e*ψ') .- 0.5 .* Λ .+
          0.5 .* (M*(Πp .+ Πm) .+ (Πp' .+ Πm')*M)

    nonneg_min = Dict{Symbol,Float64}(
        :γ   => minimum(γ),
        :δ   => minimum(δ),
        :κ   => minimum(κ),
        :Λ   => minimum(Λ),
        :Πp  => minimum(Πp),
        :Πm  => minimum(Πm),
        :ΓUU => minimum(ΓUU),
        :ΓLL => minimum(ΓLL),
        :ΓUL => minimum(ΓUL),
    )

    sym_norm = Dict{Symbol,Any}(
        :ΓUU => ΓUU - ΓUU',
        :ΓLL => ΓLL - ΓLL',
        :Λ   => Λ   - Λ',
    )

    return (; CX, Cx, Cu, CR, CU, nonneg_min, sym_norm)
end


# ------------------------------- RUN & PRINT ----------------------------------

# ---- INPUTS ----
path = "dual_EU_20251014_170533.txt"
Q0 = [
    0.0003   0.1016   0.0316   0.0867;
    0.1016   0.0020   0.1001   0.1059;
    0.0316   0.1001  -0.0005  -0.0703;
    0.0867   0.1059  -0.0703  -0.1063
]
q0 = [-0.1973, -0.2535, -0.1967, -0.0973]
ρ  = 3.0
M  = I(4)
e  = ones(4)

fmt = "%.9g"                   # runtime format string is OK now
F = Printf.Format(fmt)         # precompile format once
printn(x) = (Printf.format(stdout, F, x); nothing)

mult = DualCheck.load_dual_txt(path)
out  = DualCheck.compute_EU(Q0, q0, ρ, M, e, mult)

#  print dual feasibility for EU
println("1) DUAL FEASIBILITY FOR EU SOLUTION")
let
    println("CX =")
    for i in axes(out.CX,1)
        for j in axes(out.CX,2)
            printn(out.CX[i,j]); print("  ")
        end
        println()
    end
    println()

    println("Cx =")
    print("[ ")
    for v in out.Cx
        printn(v); print(" ")
    end
    println("]\n")

    println("Cu =")
    print("[ ")
    for v in out.Cu
        printn(v); print(" ")
    end
    println("]\n")

    println("CR =")
    for i in axes(out.CR,1)
        for j in axes(out.CR,2)
            printn(out.CR[i,j]); print("  ")
        end
        println()
    end
    println()

    println("CU =")
    for i in axes(out.CU,1)
        for j in axes(out.CU,2)
            printn(out.CU[i,j]); print("  ")
        end
        println()
    end
    println()
end

# ---- print nonneg_min ----
println("nonneg_min =")
for k in sort!(collect(keys(out.nonneg_min)); by=String)
    print("  ", rpad(String(k), 6), " : ")
    printn(out.nonneg_min[k])
    println()
end
println()

# ---- print sym_norm (matrices A - Aᵀ) ----
println("sym_norm =")
for k in sort!(collect(keys(out.sym_norm)); by=String)
    V = out.sym_norm[k]
    println("  $(String(k)) =")
    for i in axes(V,1)
        for j in axes(V,2)
            print("    "); printn(V[i,j]); print("  ")
        end
        println()
    end
    println()
end


println("2) PRIMAL FEASIBILITY FOR IU SOLUTION")
function check_IU_bool(x::AbstractVector, u::AbstractVector,
                       X::AbstractMatrix, R::AbstractMatrix, U::AbstractMatrix;
                       rho::Real=3.0, M=I, e::AbstractVector=ones(length(u)),
                       tol::Real=1e-8, verbose::Bool=true)

    n  = length(u)
    M  = Matrix(M)
    e  = collect(e)

    # FC
    ok_diagU = maximum(abs.(diag(U) .- u)) ≤ tol
    ok_bigM1 = minimum(M*U*M .- M*R' .- R*M .+ X) ≥ -tol
    ok_bigM2 = minimum(M*U*M .+ M*R' .+ R*M .+ X) ≥ -tol
    ok_bigM3 = minimum(M*U*M .+ M*R' .- R*M .- X) ≥ -tol
    ok_box1  = minimum(M*u .- x)                 ≥ -tol   # x ≤ Mu
    ok_box2  = minimum(M*u .+ x)                 ≥ -tol   # -Mu ≤ x

    # I
    ok_sumu  = (rho - sum(u))                    ≥ -tol   # e'u ≤ rho
    ok_qcard = (rho^2 - 2rho*sum(u) + e' * U * e) ≥ -tol
    ok_Ixp   = minimum(rho .* (M*u) .- rho .* x .- M*U*e .+ R*e) ≥ -tol
    ok_Ixm   = minimum(rho .* (M*u) .+ rho .* x .- M*U*e .- R*e) ≥ -tol

    # U
    ok_ulee  = minimum(1 .- u)                   ≥ -tol
    eeT      = ones(n,n)
    ok_Uself = minimum(U .- (u*e') .- (e*u') .+ eeT) ≥ -tol
    ok_Uxp   = minimum(M*(u*e') .- x*e' .- M*U .+ R) ≥ -tol
    ok_Uxm   = minimum(M*(u*e') .+ x*e' .- M*U .- R) ≥ -tol

    # IU
    ok_IU    = minimum(rho .* e .- rho .* u .- (sum(u)) .* e .+ U*e) ≥ -tol

    result = (
        diagU_eq_u = ok_diagU,
        bigM1 = ok_bigM1, bigM2 = ok_bigM2, bigM3 = ok_bigM3,
        box1 = ok_box1, box2 = ok_box2,
        sumu_le_rho = ok_sumu, quad_card = ok_qcard,
        Ixp = ok_Ixp, Ixm = ok_Ixm,
        u_le_e = ok_ulee, Uself = ok_Uself, Uxp = ok_Uxp, Uxm = ok_Uxm,
        IU = ok_IU,
    )

    if verbose
        println("== IU primal feasibility (tol = ", tol, ") ==")
        for (k, v) in pairs(result)          # <-- fix: iterate key=>value pairs
            println(rpad(String(k), 14), ": ", v)
        end
    end
    return result
end

# Your instance
x = [0.5, 0.5, 0.5, 0.0]
u = [0.5, 0.5, 0.5, 1.0]
X = [ 0.5   0.0  0.25  -0.5;
      0.0   0.5  0.0   -0.5;
      0.25  0.0  0.5    0.5;
     -0.5  -0.5  0.5    1.0 ]
R = [ 0.5   0.0  0.25  0.5;
      0.0   0.5  0.0   0.5;
      0.25  0.0  0.5   0.5;
     -0.5  -0.5  0.5   0.0 ]
U = [ 0.5   0.0  0.25  0.5;
      0.0   0.5  0.0   0.5;
      0.25  0.0  0.5   0.5;
      0.5   0.5  0.5   1.0 ]

check_IU_bool(x,u,X,R,U; rho=3.0, M=I(4), e=ones(4), tol=1e-9, verbose=true)
