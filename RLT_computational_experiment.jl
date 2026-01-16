using JSON
using JuMP, MosekTools
using Combinatorics          # for combinations
using LinearAlgebra
using Main.RLTBigM           # RLTBigM.build_and_solve, prepare_instance, add_FC!, add_FE!, add_FU!
using Main.RLT_SDP_Batch: normalize_instance!
const MOI = JuMP.MOI

# ----------------------------------------------------------------------
# Helper: pretty-print matrices with rounding
# ----------------------------------------------------------------------
function print_matrix_rounded(name::AbstractString, M; digits::Int = 4)
    println("\n", name, " (rounded to ", digits, " digits) =")
    Mround = round.(M; digits = digits)
    show(stdout, "text/plain", Mround)
    println()
end

# ----------------------------------------------------------------------
# Build all 0–1 support vectors v^S with |S| = rho
# ----------------------------------------------------------------------
function all_support_vectors(n::Int, rho::Int)
    vs = Vector{Vector{Float64}}()
    for S in combinations(1:n, rho)
        v = zeros(Float64, n)
        v[S] .= 1.0
        push!(vs, v)
    end
    return vs
end

# ----------------------------------------------------------------------
# Feasibility model with x*, X*, u* fixed
#
# Modes:
#   mode = :S_psd
#       - variables: R, U, S (S is PSD matrix)
#       - constraint: U = u*u' + S
#
#   mode = :lambda
#       - variables: R, U, λ
#       - constraints: λ ≥ 0, sum(λ) = 1,
#                      U = Σ_s λ_s v^s (v^s)'
#
# In both modes:
#   - FC, FE, FU are added using fixed x*, u*, X* as data.
#   - If enforce_R_xu = true, we impose |R - x*u'| ≤ tol elementwise.
# ----------------------------------------------------------------------
function build_feas_model(
    data::Dict{String,Any},
    x_fix::Vector{Float64},
    u_fix::Vector{Float64},
    X_fix::Matrix{Float64};
    mode::Symbol = :S_psd,        # :S_psd or :lambda
    enforce_R_xu::Bool = false,
    tol::Float64 = 1e-6,
)
    params = RLTBigM.prepare_instance(data)
    n = params.n
    ρ = params.ρ

    m = Model(MosekTools.Optimizer)
    set_silent(m)

    # Variables (only R, U; x, u, X are fixed numerically)
    @variable(m, R[1:n,1:n])
    @variable(m, U[1:n,1:n], Symmetric)

    # Original RLT blocks, using fixed x,u,X as numeric data
    RLTBigM.add_FC!(m, x_fix, u_fix, X_fix, R, U, params)
    RLTBigM.add_FE!(m, x_fix, u_fix, X_fix, R, U, params)
    RLTBigM.add_FU!(m, x_fix, u_fix, X_fix, R, U, params)

    if mode == :S_psd
        # U = u*u' + S, with S PSD
        @variable(m, S[1:n,1:n], PSD)
        uuT = u_fix * transpose(u_fix)   # constant matrix
        @constraint(m, U .== uuT .+ S)

    elseif mode == :lambda
        # Lambda representation: all v^S with |S| = ρ
        vs = all_support_vectors(n, Int(round(ρ)))
        S_count = length(vs)

        @variable(m, λ[1:S_count] >= 0)
        @constraint(m, sum(λ) == 1)

        # U = Σ_s λ_s v^s (v^s)'
        for i in 1:n, j in 1:n
            @constraint(m, U[i,j] == sum(λ[s] * vs[s][i] * vs[s][j] for s in 1:S_count))
        end
    else
        error("Unknown mode = $mode. Use :S_psd or :lambda.")
    end

    if enforce_R_xu
        # Target outer product R_target = x* u*'
        R_target = x_fix * transpose(u_fix)
        # Impose |R - R_target| ≤ tol elementwise
        @constraint(m, R .<= R_target .+ tol)
        @constraint(m, R .>= R_target .- tol)
    end

    @objective(m, Min, 0.0)   # pure feasibility

    return m
end

# ----------------------------------------------------------------------
# Driver: load instances, pick one, solve EU, then test feasibility
# under a chosen U-structure and optional R ≈ x*u'
# ----------------------------------------------------------------------

# Load and normalize all instances
raw = JSON.parsefile("instances.json")
insts = [normalize_instance!(deepcopy(d)) for d in raw]

println("Total number of instances: ", length(insts))

# Choose which instance and which mode to test
instance_index = 13          # change this index if you want a different instance
mode          = :lambda         # choose :S_psd or :lambda
enforce_R_xu  = false          # set true if you want |R - x*u'| ≤ tol

data = insts[instance_index]
println("Selected instance index = ", instance_index,
        ", id = ", get(data, "id", "unknown"))

# Solve EU to get x*, u*, X*, R*, U*
res_EU = RLTBigM.build_and_solve(data; variant="EU")
status_EU = res_EU[1]
@assert status_EU == :OPTIMAL "EU variant is not optimal"

_, obj_EU, x_star, u_star, X_star, R_star, U_star = res_EU
println("EU objective = ", obj_EU)

println("\n=== x_star (rounded to 4 digits) ===")
println(round.(x_star; digits=4))

println("\n=== u_star (rounded to 4 digits) ===")
println(round.(u_star; digits=4))

println("\n=== X_star (rounded to 4 digits) ===")
show(stdout, "text/plain", round.(X_star; digits=4))
println()

# Build feasibility model with x*, X*, u* fixed and selected mode
m_feas = build_feas_model(
    data,
    x_star,
    u_star,
    X_star;
    mode = mode,
    enforce_R_xu = enforce_R_xu,
    tol = 1e-6,
)

optimize!(m_feas)

feas_status = termination_status(m_feas)
println("feasibility model status = ", feas_status)

if feas_status in (MOI.OPTIMAL, MOI.FEASIBLE_POINT)
    U_new = value.(m_feas[:U])
    R_new = value.(m_feas[:R])

    # Differences
    diffU     = maximum(abs.(U_new .- U_star))
    diffR     = maximum(abs.(R_new .- R_star))
    R_xu      = x_star * transpose(u_star)
    diffR_xu  = maximum(abs.(R_new .- R_xu))

    println("==== U comparison ====")
    println("max |U_new - U_star| = ", round(diffU; digits = 8))
    print_matrix_rounded("U_star (from EU)", U_star; digits = 4)
    print_matrix_rounded("U_new (from feas model)", U_new; digits = 4)

    println("\n==== R comparison ====")
    println("max |R_new - R_star| = ", round(diffR; digits = 8))
    println("max |R_new - x*u'|   = ", round(diffR_xu; digits = 8))
    print_matrix_rounded("R_star (from EU)", R_star; digits = 4)
    print_matrix_rounded("R_new (from feas model)", R_new; digits = 4)
    print_matrix_rounded("R_xu = x_star * u_star'", R_xu; digits = 4)

    if mode == :S_psd
        S_new = value.(m_feas[:S])
        print_matrix_rounded("S_new (U - u*u')", S_new; digits = 4)
    end

    if mode == :lambda
        λ_new = value.(m_feas[:λ])
        println("\nλ_new (first 20 entries, rounded to 4 digits) = ",
                round.(λ_new[1:min(end,20)]; digits = 4))
    end
end




























# ----------------------------------------------------------------------
# Main driver: run S_psd test on ALL instances
#
# For each instance:
#   1) Solve EU to get (x*, u*, X*, R*, U*)
#   2) Build feasibility model with mode = :S_psd
#   3) Check if there exists S PSD s.t. U = u*u' + S under all EU constraints
# ----------------------------------------------------------------------
function run_S_psd_all(; enforce_R_xu::Bool = false, tol::Float64 = 1e-6)
    # Load and normalize all instances
    raw = JSON.parsefile("instances.json")
    insts = [normalize_instance!(deepcopy(d)) for d in raw]

    println("Total number of instances: ", length(insts))

    feasible_count = 0
    infeasible_count = 0
    skipped_count = 0

    for (idx, data) in enumerate(insts)
        id = get(data, "id", "unknown")
        println("\n==============================")
        println("Instance index = $idx, id = $id")
        println("==============================")

        # Solve EU to get x*, u*, X*, R*, U*
        res_EU = RLTBigM.build_and_solve(data; variant = "EU")
        status_EU = res_EU[1]
        println("EU status = ", status_EU)

        if status_EU != :OPTIMAL
            println("Skipping S_psd test because EU is not OPTIMAL.")
            skipped_count += 1
            continue
        end

        _, obj_EU, x_star, u_star, X_star, R_star, U_star = res_EU
        println("EU objective = ", obj_EU)

        # Build S_psd feasibility model
        m_feas = build_feas_model(
            data,
            x_star,
            u_star,
            X_star;
            mode = :S_psd,
            enforce_R_xu = enforce_R_xu,
            tol = tol,
        )
        optimize!(m_feas)

        feas_status = termination_status(m_feas)
        println("S_psd feasibility model status = ", feas_status)

        if feas_status in (MOI.OPTIMAL, MOI.FEASIBLE_POINT)
            feasible_count += 1

            U_new = value.(m_feas[:U])
            R_new = value.(m_feas[:R])
            S_new = value.(m_feas[:S])

            # Differences (for diagnostics)
            diffU     = maximum(abs.(U_new .- U_star))
            diffR     = maximum(abs.(R_new .- R_star))
            R_xu      = x_star * transpose(u_star)
            diffR_xu  = maximum(abs.(R_new .- R_xu))

            println("  max |U_new - U_star| = ", round(diffU; digits = 8))
            println("  max |R_new - R_star| = ", round(diffR; digits = 8))
            println("  max |R_new - x*u'|   = ", round(diffR_xu; digits = 8))

            # Optional: check smallest eigenvalue of S_new
            ev = eigvals(Symmetric(S_new))
            println("  min eigenvalue(S_new) = ", round(minimum(ev); digits = 8))
        else
            infeasible_count += 1
        end
    end

    println("\n========== S_psd SUMMARY ==========")
    println("Feasible S_psd-model instances   : ", feasible_count)
    println("Infeasible S_psd-model instances : ", infeasible_count)
    println("Skipped (EU not OPTIMAL)         : ", skipped_count)
    println("Total                            : ", length(insts))
    println("==================================")
end

# Run the driver
run_S_psd_all()











