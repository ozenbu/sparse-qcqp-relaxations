module PSDviolation

using JuMP, Gurobi, LinearAlgebra


"""
Build S = [1  x';  
           x  X].
"""
function build_S(x::AbstractVector, X::AbstractMatrix)
    n = length(x)
    @assert size(X) == (n,n)
    S = Matrix{float(eltype(X))}(undef, n+1, n+1)
    S[1,1] = 1
    S[1,2:end] = x
    S[2:end,1] = x
    S[2:end,2:end] = X
    return Symmetric(S)
end

"""
    eig_S(x, X) -> (vals, vecs)
Eigenvalues/vectors of S = [1 x'; x X] (ascending).
"""
function eig_S(x::AbstractVector, X::AbstractMatrix)
    F = eigen(build_S(x,X))
    return (F.values, F.vectors)
end

end # module


using .PSDviolation

x = [0.5, 0.5, 0.5, 0.0]
X = [0.5 0.0 0.25 -0.5;
     0.0 0.5 0.0  -0.5;
     0.25 0.0 0.5  0.5;
     -0.5 -0.5 0.5 1.0]


vals, vecs = PSDviolation.eig_S(x, X)
println("negative eigenalue = ",vals[1])
print("corresponding eigenvector",vecs[:, 1])
v1= vecs[:, 1]  # n + 1 

v1 = round.(Float64.(v1); digits=4) # round to 4 decimals

α1 = v1[1]                        # scalar
d1 = v1[2:end]                    # d (length n vector)

# Build Q0 and q0
Q0t1 = 2.0 .* (d1 * d1')            # size n×n
q0t1 = 2.0 .* (α1 .* d1)             # size n
Q0t1 = round.(Float64.(Q0t1); digits=4)
q0t1 = round.(Float64.(q0t1); digits=4)
# (optional) print
println("alpha = ", α1)
println("d = ", d1)
println("Q0 =\n", Q0t1)
println("q0 = ", q0t1)




# Step 2: Obtaining integer PSD violating directions


function integer_witness_direction(x::AbstractVector, X::AbstractMatrix; B::Int=10)
    n = length(x)
    @assert size(X) == (n, n)

    m = Model(optimizer_with_attributes(Gurobi.Optimizer,
                                        "NonConvex" => 2,   # allow nonconvex quad obj/cons
                                        "OutputFlag" => 1)) # set 0 to silence

    @variable(m, alpha, Int)
    @variable(m, d[1:n], Int)

    @constraint(m, -B <= alpha <= B)
    @constraint(m, -B .<= d .<= B)

    # Optional but convenient: forbid the zero vector (needs NonConvex=2)
    @constraint(m, alpha^2 + sum(d[i]^2 for i in 1:n) >= 1)

    @objective(m, Min, dot(d, X*d) + 2 * alpha * dot(x, d))

    optimize!(m)

    st = termination_status(m)
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_OPTIMAL
        α     = round(Int, value(alpha))
        dstar = round.(Int, value.(d))
        objv  = objective_value(m)
        vSv   = α^2 + 2*α*dot(x, dstar) + dot(dstar, X*dstar)
        return (st, α, dstar, objv, vSv)
    else
        return (st, nothing, nothing, nothing, nothing)
    end
end


x = [0.5, 0.5, 0.5, 0.0]
X = [0.5 0.0 0.25 -0.5;
      0.0 0.5 0.0  -0.5;
      0.25 0.0 0.5  0.5;
     -0.5 -0.5 0.5  1.0]
status, α2, d2, objv, vSv = integer_witness_direction(x, X; B=10)
println((status, α2, d2, objv, vSv))

Q0t2 = 2.0 .* (d2 * d2')            # size n×n
q0t2 = 2.0 .* (α2 .* d2)   