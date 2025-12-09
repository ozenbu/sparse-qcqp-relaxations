# clean_instance_hunter.jl
using Random, LinearAlgebra, Printf
using JuMP, Gurobi
using .RLTBigM   # make sure RLT_bigM_4variants.jl is included first

# ---------- pretty fractions ----------
fraction(x; denom=10) = round(x*denom)/denom  # e.g. denom=10 -> tenths, 20 -> twentieths
fracm(A; denom=10) = map(a -> fraction(a; denom=denom), A)

function print_instance(Q0, q0, A, b; denom=10)
    @printf("Q0 = [\n")
    for i in axes(Q0,1)
        for j in axes(Q0,2)
            @printf("  %g%s", fraction(Q0[i,j]; denom=denom), j==size(Q0,2) ? "" : "  ")
        end
        println()
    end
    println("]\n")
    @printf("q0 = [ "); for j=1:length(q0) @printf("%g%s", fraction(q0[j]; denom=denom), j==length(q0) ? "" : "  ") end; println(" ]\n")

    if size(A,1) > 0
        println("A = [")
        for i in 1:size(A,1)
            for j in 1:size(A,2)
                @printf("  %g%s", fraction(A[i,j]; denom=denom), j==size(A,2) ? "" : "  ")
            end
            println()
        end
        println("]")
        @printf("b = [ "); for i=1:length(b) @printf("%g%s", fraction(b[i]; denom=denom), i==length(b) ? "" : "  ") end; println(" ]\n")
    end
end

# ---------- simple, meaningful linear constraints ----------
# Always include box |x_i| ≤ M_i. Optionally add K extra small-coeff rows.
function make_linear_bounds(n; Mvec=ones(n), extra_K=2, rng=Random.GLOBAL_RNG)
    Abox = [Matrix(I(n)); -Matrix(I(n))]
    bbox = vcat(Mvec, Mvec)

    # extra constraints: rows with entries in {-1,0,1}, rhs in {0, 1}
    Aextra = Matrix{Float64}(undef, 0, n); bextra = Float64[]
    for _ in 1:extra_K
        row = rand(rng, (-1:1), n)
        if all(row .== 0)     # skip zero row
            continue
        end
        # normalize to avoid scaling weirdness: make gcd=1 implicitly via small ints
        push!(Aextra, float.(row)')
        push!(bextra, rand(rng, [0.0, 1.0]))
    end

    A = vcat(Abox, Aextra)
    b = vcat(bbox, bextra)
    return A, b
end

# ---------- clean random (Q0,q0) ----------
function make_clean_Q(n; q_range=-3:3, q_denom=2, Q_range=-2:2, Q_denom=10, rng=Random.GLOBAL_RNG)
    # symmetric Q0 with entries from multiples of 1/Q_denom
    S = rand(rng, Q_range, n, n) ./ Q_denom
    Q0 = 0.5*(S + S')
    # small linear term with halves / quarters etc.
    q0 = rand(rng, q_range, n) ./ q_denom
    return Q0, q0
end

# ---------- evaluate EU & IU on one instance ----------
function eval_gap(data; optimizer=Gurobi.Optimizer)
    res_EU = RLTBigM.build_and_solve(data; variant="EU", optimizer=optimizer)
    res_IU = RLTBigM.build_and_solve(data; variant="IU", optimizer=optimizer)
    if res_EU[1] != :OPTIMAL || res_IU[1] != :OPTIMAL
        return nothing
    end
    EU_obj = res_EU[2]
    IU_obj = res_IU[2]
    return (; EU_obj, IU_obj, gap = EU_obj - IU_obj,
             x_EU = res_EU[3], u_EU = res_EU[4],
             x_IU = res_IU[3], u_IU = res_IU[4])
end

# ---------- search loop ----------
"""
hunt_clean_instance(; n=4, ρ=3.0, Mvec=ones(n), trials=2000, seed=42,
                       q_range=-3:3, q_denom=2, Q_range=-2:2, Q_denom=10,
                       extra_K=2, target_gap=0.02, denom_print=10)

Looks for an instance with small rational (Q0,q0) and linear A,b such that EU_obj - IU_obj ≥ target_gap.
Prints the first found instance in nice fractional form and returns a NamedTuple with details.
"""
function hunt_clean_instance(; n=4, ρ=3.0, Mvec=ones(n), trials=2000, seed=42,
                              q_range=-3:3, q_denom=2, Q_range=-2:2, Q_denom=10,
                              extra_K=2, target_gap=0.02, denom_print=10)

    rng = MersenneTwister(seed)
    Mmat = Diagonal(Mvec)

    for t in 1:trials
        Q0, q0 = make_clean_Q(n; q_range, q_denom, Q_range, Q_denom, rng)
        A, b   = make_linear_bounds(n; Mvec, extra_K, rng)

        data = Dict(
            "n"=>n, "rho"=>ρ, "Q0"=>Q0, "q0"=>q0,
            "Qi"=>nothing, "qi"=>nothing, "ri"=>nothing,
            "Pi"=>nothing, "pi"=>nothing, "si"=>nothing,
            "A"=>A, "b"=>b,
            "H"=>nothing, "h"=>nothing,
            "M"=>Mmat
        )

        res = eval_gap(data)
        res === nothing && continue
        if res.gap ≥ target_gap
            println("Found instance at trial $t with gap = $(res.gap): EU=$(res.EU_obj), IU=$(res.IU_obj)\n")
            print_instance(Q0, q0, A, b; denom=denom_print)
            return (; data, res...)
        end
    end
    println("No instance found with gap ≥ $target_gap in $trials trials. Try increasing trials or ranges.")
    return nothing
end

 include("RLT_bigM_4variants.jl")
 out = hunt_clean_instance(n=4, ρ=3.0, Mvec=ones(4),
                           trials=5000, seed=123,
                           q_range=-3:3, q_denom=2,    # q0 in multiples of 0.5
                           Q_range=-2:2, Q_denom=10,   # Q0 in multiples of 0.1
                           extra_K=2, target_gap=0.03, denom_print=10)
 if out !== nothing
     # sanity: print the (x,u) sums to see EU hits ρ while IU can be < ρ
     @printf("sum(u)_EU = %.6f, sum(u)_IU = %.6f\n", sum(out.u_EU), sum(out.u_IU))
 end
