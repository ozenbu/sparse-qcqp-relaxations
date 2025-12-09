# --- add these imports at the top of the module if not present ---
using Polyhedra, CDDLib

# ========== indexing helpers ==========
struct IndexMap
    x::UnitRange{Int}
    X::UnitRange{Int}
    u::UnitRange{Int}
    R::UnitRange{Int}
    U::UnitRange{Int}
end
vecindex(i,j,n) = (j-1)*n + i                # column-major
function make_index_map(n)
    nx, nX, nu, nR, nU = n, n*n, n, n*n, n*n
    o = 0
    x = o+1:o+nx;  o += nx
    X = o+1:o+nX;  o += nX
    u = o+1:o+nu;  o += nu
    R = o+1:o+nR;  o += nR
    U = o+1:o+nU;  o += nU
    return IndexMap(x,X,u,R,U), o
end
push_le!(A,b,row,β) = (push!(A,row); push!(b,β))
push_eq!(C,d,row,δ) = (push!(C,row); push!(d,δ))

# ========== build H-rep from your RLT blocks ==========
"""
    build_Hrep_RLT(params; variant="EU")

Given `params = prepare_instance(data)`, returns a Polyhedra polyhedron `P`
for the RLT-(EU/IU/...) model and the index map `idx`. ρ is used as a scalar.
Variables order: z = [x; vec(X); u; vec(R); vec(U)] (full vecs; symmetry via equalities).
"""
function build_Hrep_RLT(params; variant::String="EU")
    @unpack n, ρ, Qi, qi, ri, Pi, pi, si, A, b, ℓ, H, h, η, M, e = params

    idx, nvars = make_index_map(n)
    Arows = Vector{Vector{Float64}}();  brows = Float64[]
    Crows = Vector{Vector{Float64}}();  drows = Float64[]
    zrow() = zeros(Float64, nvars)

    # (0) Symmetry: X_ij = X_ji, U_ij = U_ji
    for i in 1:n, j in i+1:n
        r = zrow(); r[idx.X[vecindex(i,j,n)]] =  1; r[idx.X[vecindex(j,i,n)]] = -1; push_eq!(Crows,drows,r,0)
        r = zrow(); r[idx.U[vecindex(i,j,n)]] =  1; r[idx.U[vecindex(j,i,n)]] = -1; push_eq!(Crows,drows,r,0)
    end

    # (1) Original linear rows: A x ≤ b, H x = h
    for p in 1:ℓ
        r = zrow(); for j in 1:n r[idx.x[j]] = A[p,j] end; push_le!(Arows,brows,r,b[p])
    end
    for p in 1:η
        r = zrow(); for j in 1:n r[idx.x[j]] = H[p,j] end; push_eq!(Crows,drows,r,h[p])
    end

    # (2) Big-M links: -M u ≤ x ≤ M u, diag(U)=u
    for i in 1:n
        r = zrow(); r[idx.x[i]] =  1; r[idx.u[i]] = -M[i,i]; push_le!(Arows,brows,r,0.0)
        r = zrow(); r[idx.x[i]] = -1; r[idx.u[i]] = -M[i,i]; push_le!(Arows,brows,r,0.0)
        r = zrow(); r[idx.U[vecindex(i,i,n)]] = 1; r[idx.u[i]] -= 1; push_eq!(Crows,drows,r,0.0)
    end

    # (3) H-block lift: H*X = h x', H*R = h u'
    for p in 1:η, j in 1:n
        r = zrow()
        for i in 1:n r[idx.X[vecindex(i,j,n)]] += H[p,i] end
        r[idx.x[j]] -= h[p]; push_eq!(Crows,drows,r,0.0)
    end
    for p in 1:η, j in 1:n
        r = zrow()
        for i in 1:n r[idx.R[vecindex(i,j,n)]] += H[p,i] end
        r[idx.u[j]] -= h[p]; push_eq!(Crows,drows,r,0.0)
    end

    # (4) McCormick-style trio (entrywise ≥ 0)
    for i in 1:n, j in 1:n
        # G1 = MUM - MR' - RM + X ≥ 0  ⇒  -G1 ≤ 0
        r = zrow()
        r[idx.X[vecindex(i,j,n)]] -= 1
        r[idx.U[vecindex(i,j,n)]] -= M[i,i]*M[j,j]
        for k in 1:n
            r[idx.R[vecindex(j,k,n)]] += M[i,i]
            r[idx.R[vecindex(i,k,n)]] += M[k,k]
        end
        push_le!(Arows, brows, r, 0.0)

        # G2 = MUM + MR' + RM + X ≥ 0  ⇒  -G2 ≤ 0
        r = zrow()
        r[idx.X[vecindex(i,j,n)]] -= 1
        r[idx.U[vecindex(i,j,n)]] -= M[i,i]*M[j,j]
        for k in 1:n
            r[idx.R[vecindex(j,k,n)]] -= M[i,i]
            r[idx.R[vecindex(i,k,n)]] -= M[k,k]
        end
        push_le!(Arows, brows, r, 0.0)

        # G3 = MUM + MR' - RM - X ≥ 0  ⇒  -G3 ≤ 0
        r = zrow()
        r[idx.X[vecindex(i,j,n)]] += 1   # -( -X ) ≤ 0
        r[idx.U[vecindex(i,j,n)]] -= M[i,i]*M[j,j]
        for k in 1:n
            r[idx.R[vecindex(j,k,n)]] -= M[i,i]
            r[idx.R[vecindex(i,k,n)]] += M[k,k]
        end
        push_le!(Arows, brows, r, 0.0)
    end

    # (5) A-based RLT blocks (as in your add_FC!)
    if ℓ > 0
        # (5a) A X A' - A(x b') - (b x')A' + (b b') ≥ 0   (entrywise)
        for p in 1:ℓ, q in 1:ℓ
            r = zrow()
            for i in 1:n, j in 1:n r[idx.X[vecindex(i,j,n)]] += A[p,i]*A[q,j] end
            for j in 1:n r[idx.x[j]] -= A[p,j]*b[q] end
            for i in 1:n r[idx.x[i]] -= b[p]*A[q,i] end
            # move constant bb' to RHS
            push_le!(Arows, brows, (-1).*r, -b[p]*b[q])
        end
        # (5b) A X - (b x') - A R M + (b u')M ≥ 0
        for p in 1:ℓ, j in 1:n
            r = zrow()
            for i in 1:n r[idx.X[vecindex(i,j,n)]] += A[p,i] end
            r[idx.x[j]] -= b[p]
            for i in 1:n, t in 1:n r[idx.R[vecindex(i,t,n)]] -= A[p,i]*M[t,t] end
            r[idx.u[j]] += b[p]*M[j,j]
            push_le!(Arows, brows, (-1).*r, 0.0)
        end
        # (5c) -A X + (b x') - A R M + (b u')M ≥ 0
        for p in 1:ℓ, j in 1:n
            r = zrow()
            for i in 1:n r[idx.X[vecindex(i,j,n)]] -= A[p,i] end
            r[idx.x[j]] += b[p]
            for i in 1:n, t in 1:n r[idx.R[vecindex(i,t,n)]] -= A[p,i]*M[t,t] end
            r[idx.u[j]] += b[p]*M[j,j]
            push_le!(Arows, brows, (-1).*r, 0.0)
        end
    end

    # (6) EU / IU linkers
    if startswith(variant, "E")
        # sum(u) = ρ ;  R e = ρ x ;  U e = ρ u
        r = zrow(); for i in 1:n r[idx.u[i]] = 1 end; push_eq!(Crows,drows,r,ρ)
        for i in 1:n
            r = zrow(); for j in 1:n r[idx.R[vecindex(i,j,n)]] += 1 end
            r[idx.x[i]] -= ρ; push_eq!(Crows,drows,r,0.0)
        end
        for i in 1:n
            r = zrow(); for j in 1:n r[idx.U[vecindex(i,j,n)]] += 1 end
            r[idx.u[i]] -= ρ; push_eq!(Crows,drows,r,0.0)
        end
    else
        # IU core
        r = zrow(); for i in 1:n r[idx.u[i]] = 1 end; push_le!(Arows,brows,r,ρ)
        # ρ^2 - 2ρ∑u + e'Ue ≥ 0  ⇒  2ρ∑u - ∑∑U ≤ ρ^2
        r = zrow(); for i in 1:n r[idx.u[i]] += 2ρ end
        for i in 1:n, j in 1:n r[idx.U[vecindex(i,j,n)]] -= 1 end
        push_le!(Arows, brows, r, ρ^2)
        if ℓ > 0
            # ρ b - (b u')e - ρ(Ax) + A R e ≥ 0 (ℓ rows)
            for p in 1:ℓ
                r = zrow()
                for i in 1:n r[idx.x[i]] += ρ*A[p,i] end
                for j in 1:n r[idx.u[j]] -= b[p] end
                for i in 1:n, j in 1:n r[idx.R[vecindex(i,j,n)]] -= A[p,i] end
                push_le!(Arows, brows, r, ρ*b[p])
            end
        end
    end

    # (7) U-addons if needed: mirror your add_FU! / add_FIU! similarly (optional)
    # (kept out for brevity — you can paste 1:1 from your JuMP logic following the same pattern)

    # (8) Lifted quadratic rows that only involve (x,X): keep as-is
    for (Q,q,rhs) in zip(Qi,qi,ri)
        r = zrow()
        for i in 1:n, j in 1:n r[idx.X[vecindex(i,j,n)]] += 0.5*Q[i,j] end
        for j in 1:n r[idx.x[j]] += q[j] end
        push_le!(Arows,brows,r,-rhs)
    end
    for (P,p,s) in zip(Pi,pi,si)
        r = zrow()
        for i in 1:n, j in 1:n r[idx.X[vecindex(i,j,n)]] += 0.5*P[i,j] end
        for j in 1:n r[idx.x[j]] += p[j] end
        push_eq!(Crows,drows,r,-s)
    end

    # Build polyhedron
    A = vcat(Arows...); b = brows
    h = (length(Crows)>0) ? hrep(A, b, linear(vcat(Crows...), drows)) : hrep(A, b)
    P = polyhedron(h, CDDLib.Library())
    return P, idx
end

# ========== projection helpers ==========
"Project to (x, X) by eliminating indices for u, R, U."
function project_to_xX(params; variant="EU", method=:block)
    P, idx = build_Hrep_RLT(params; variant=variant)
    elim = sort!(vcat(collect(idx.u), collect(idx.R), collect(idx.U)))
    algo = method==:fm ? FourierMotzkin() : BlockElimination()
    Pproj = eliminate(P, elim, algo)
    return Pproj, idx
end

"Quick debug: print counts of halfspaces/equalities in the projected polyhedron."
function describe_projection(Pproj)
    H = hrep(Pproj)
    println("halfspaces = ", length(halfspaces(H)))
    println("equalities = ", length(hyperplanes(H)))
    return nothing
end


# You already have: params = prepare_instance(data_dict)
params  = prepare_instance(data_test4)      # e.g., from your demo
Pproj, idx = project_to_xX(params; variant="IU", method=:block)
describe_projection(Pproj)

# If you want to optimize a linear form cᵀ[x;vec(X)] over the projection:
# (Build a Polyhedra LP with the remaining coords — optional sanity check)
