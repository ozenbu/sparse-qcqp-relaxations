import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum

def build_and_solve(data, variant='EXACT', verbose=False, build_only=False):
    """
    Solve either the exact MIQCQP ('EXACT') or one of the
    first-level RLT relaxations ('E1','E2','I1','I2') via Gurobi.
    If build_only=True, returns the Model without solving.
    """
    # -- unpack & defaults --
    n, rho = data['n'], data['rho']
    e      = np.ones((n,1))
    Q0, q0 = data['Q0'], data['q0']
    Qi, qi, ri = data.get('Qi') or [], data.get('qi') or [], data.get('ri') or []
    Pi, pi, si = data.get('Pi') or [], data.get('pi') or [], data.get('si') or []
    A, b    = data.get('A'), data.get('b')
    H, h    = data.get('H'), data.get('h')
    M_raw   = data['M']
    Mdiag   = M_raw if M_raw.ndim==1 else np.diag(M_raw)
    Mmat    = np.diag(Mdiag)

    if b is not None and b.size:
        b_col = b.reshape((-1,1))

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', int(verbose))
    env.start()
    mdl = gp.Model(env=env)

    is_exact = (variant=='EXACT')
    x = mdl.addMVar(n, lb=-GRB.INFINITY, name="x")
    if is_exact:
        u = mdl.addMVar(n, vtype=GRB.BINARY, name="u")
    else:
        u = mdl.addMVar(n, vtype=GRB.CONTINUOUS,
                        lb=-GRB.INFINITY, ub=GRB.INFINITY,
                        name="u")
    # allow all entries of X, R, U to be free real variables
    X = mdl.addMVar((n,n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="X")
    R = mdl.addMVar((n,n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R")
    U = mdl.addMVar((n,n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="U")
    # enforce symmetry of X and U
    for i in range(n):
        for j in range(i+1, n):
            mdl.addConstr(X[i, j] == X[j, i], name=f"sym_X_{i}_{j}")
            mdl.addConstr(U[i, j] == U[j, i], name=f"sym_U_{i}_{j}")


    # objective
    if is_exact:
        mdl.setObjective(0.5*x@Q0@x + q0@x, GRB.MINIMIZE)
    else:
        lin_quad = 0.5*quicksum(Q0[i,j]*X[i,j] for i in range(n) for j in range(n))
        lin_lin  = quicksum(q0[i]*x[i] for i in range(n))
        mdl.setObjective(lin_quad + lin_lin, GRB.MINIMIZE)

    # linear & Big-M
    if A is not None and A.size:
        mdl.addConstr(A@x <= b, "Ax<=b")
    if H is not None and H.size:
        mdl.addConstr(H@x == h, "Hx=h")
    mdl.addConstr(-Mmat@u <= x, "BigM_low")
    mdl.addConstr( x <= Mmat@u, "BigM_up")

    if is_exact:
        for Q_mat, q_vec, r_term in zip(Qi, qi, ri):
            mdl.addQConstr(0.5*x@Q_mat@x + q_vec@x + r_term <= 0)
        for P_mat, p_vec, s_term in zip(Pi, pi, si):
            mdl.addQConstr(0.5*x@P_mat@x + p_vec@x + s_term == 0)
        mdl.Params.NonConvex = 2
        mdl.addConstr(u.sum() <= rho, "card")
    else:
        # lifted quadratics
        for Q_mat, q_vec, r_term in zip(Qi, qi, ri):
            expr = 0.5*quicksum(Q_mat[i,j]*X[i,j]
                                for i in range(n) for j in range(n))
            expr += quicksum(q_vec[i]*x[i] for i in range(n)) + r_term
            mdl.addConstr(expr <= 0)
        for P_mat, p_vec, s_term in zip(Pi, pi, si):
            expr = 0.5*quicksum(P_mat[i,j]*X[i,j]
                                for i in range(n) for j in range(n))
            expr += quicksum(p_vec[i]*x[i] for i in range(n)) + s_term
            mdl.addConstr(expr == 0)

        # diag(U)=u
        for i in range(n):
            mdl.addConstr(U[i,i] == u[i])
        
        if H is not None and H.size:
            eta = H.shape[0]
            # HX = h xᵀ
            for a in range(eta):
                for b_ in range(n):
                    mdl.addConstr(
                        gp.quicksum(H[a, k] * X[k, b_] for k in range(n))
                        == h[a] * x[b_],
                        name=f"HX[{a},{b_}]")
            # HR = h uᵀ
            for a in range(eta):
                for b_ in range(n):
                    mdl.addConstr(
                        gp.quicksum(H[a, k] * R[k, b_] for k in range(n))
                        == h[a] * u[b_],
                        name=f"HR[{a},{b_}]")

        
        sum_u = quicksum(u[j] for j in range(n))

        # sparsity
        if variant[0]=='E':
            mdl.addConstr(u.sum() == rho)
            # enforce Re= rho x  and Ue= rho u, for j=0,1
            for i in range(n):
                # ∑_j R[i,j] = ρ · x[i]
                mdl.addConstr(
                    quicksum(R[i, j] for j in range(n)) == rho * x[i],
                    name=f"sparsity_R_row{i}"
                )
            
                # ∑_j U[i,j] = ρ · u[i]
                mdl.addConstr(
                    quicksum(U[i, j] for j in range(n)) == rho * u[i],
                    name=f"sparsity_U_row{i}"
                )
        else:
            mdl.addConstr(u.sum() <= rho)

        # common RLT: Big-M × Big-M
        for i in range(n):
            for j in range(n):
                # (Mu - x)(Mu - x)^T ≥ 0  ⇒ UU block
                expr_UU = (
                    Mdiag[i]*U[i,j]*Mdiag[j]
                  - Mdiag[i]*R[j,i]
                  - R[i,j]*Mdiag[j]
                  + X[i,j]
                )
                mdl.addConstr(expr_UU >= 0, name=f"BigM_UU[{i},{j}]")
        
                # (Mu + x)(Mu + x)^T ≥ 0  ⇒ LL block
                expr_LL = (
                    Mdiag[i]*U[i,j]*Mdiag[j]
                  + Mdiag[i]*R[j,i]
                  + R[i,j]*Mdiag[j]
                  + X[i,j]
                )
                mdl.addConstr(expr_LL >= 0, name=f"BigM_LL[{i},{j}]")
        
                # (Mu - x)(Mu + x)^T ≥ 0  ⇒ UL block
                expr_UL = (
                    Mdiag[i]*U[i,j]*Mdiag[j]
                  + Mdiag[i]*R[j,i]
                  - R[i,j]*Mdiag[j]
                  - X[i,j]
                )
                mdl.addConstr(expr_UL >= 0, name=f"BigM_UL[{i},{j}]")

        # RLT from Ax ≤ b
        if A is not None and A.size:
            p = A.shape[0]

            # 1) (b - A x)(b - A x)^T >= 0  ⇒  for each i,j in 0…p-1
            for i in range(p):
                for j in range(p):
                    expr = (
                        quicksum(A[i,k]*X[k,l]*A[j,l] 
                                 for k in range(n) for l in range(n))
                      - quicksum(A[i,k]*x[k]*b[j]     for k in range(n))
                      - quicksum(b[i]*A[j,k]*x[k]     for k in range(n))
                      + b[i]*b[j]
                    )
                    mdl.addConstr(expr >= 0, name=f"AxAx_[{i},{j}]")

            # 2) Cross‐RLT: (b - A x) × (Big-M bounds)
            for i in range(p):
                # build b_i - A_i*x just once
                bi = b[i]
                Ai = A[i, :]
                # for each variable j
                for j in range(n):
                    Mjj = Mdiag[j]
                    # sum_k A[i,k]*X[k,j]
                    sum_AX = quicksum(Ai[k] * X[k, j] for k in range(n))
                    # sum_k A[i,k]*R[k,j]
                    sum_AR = quicksum(Ai[k] * R[k, j] * Mjj for k in range(n))
        
                    # (b_i - A x) * (Mjj u_j - x_j) >= 0
                    expr_low = (
                        + Mjj * bi * u[j]
                        - bi * x[j]
                        - sum_AR
                        + sum_AX  
                    )
                    mdl.addConstr(expr_low >= 0, name=f"AxBigM_low[{i},{j}]")
        
                    # (b_i - A x) * (Mjj u_j + x_j) >= 0
                    expr_up = (
                        + Mjj * bi * u[j]
                        + bi * x[j]
                        - sum_AR
                        - sum_AX
                    )
                    mdl.addConstr(expr_up  >= 0, name=f"AxBigM_up[{i},{j}]")

        # --- I-variant cross-RLT  (only for I1/I2) ---
        if variant[0]=='I':
            # Self-RLT: (ρ - eᵀu)² ≥ 0
            mdl.addConstr(
                rho*rho 
              - 2*rho*u.sum() 
              + quicksum(U[i,j] for i in range(n) for j in range(n))
              >= 0,
                name="rho_self"
            )
        
            # Cross-RLT with Ax ≤ b
            if A is not None and A.size:
                p = A.shape[0]
                for r in range(p):
                    # build (b_r - A[r,:] x)
                    linAx = quicksum(A[r, l] * x[l] for l in range(n))
                    # build (rho - sum_j u[j])
                    # cross-term: (A[r,:] x) * (sum u)  lifts to A R e
                    ARsum = quicksum(A[r,l] * quicksum(R[l,k] for k in range(n))
                                     for l in range(n))
                    expr = (
                        rho * b[r]
                      - b[r] * sum_u
                      - rho * linAx
                      + ARsum
                    )
                    mdl.addConstr(expr >= 0, name=f"rho_A[{r}]")
            
            # BigM x rho RLTs            
            for i in range(n):
                # 1) (Mu - x)_i = Mdiag[i]*u[i] - x[i]
                diff_minus = Mdiag[i]*u[i] - x[i]
                # 2) (Mu + x)_i = Mdiag[i]*u[i] + x[i]
                diff_plus  = Mdiag[i]*u[i] + x[i]
            
                # Lifted form of diff_* * (rho - sum_u):
                # diff_* * rho  => rho*diff_*
                # diff_* * sum_u => 
                #      Mdiag[i]*(∑_j U[i,j])   from u[i]*u[j]
                #    ± ∑_j R[i,j]              from ± x[i]*u[j]
                #  with the same sign as ± in diff_*
            
                # --- minus case: (Mu - x)·(ρ - ∑u) ≥ 0 ---
                lhs1 = rho * diff_minus \
                     - (
                         Mdiag[i]*quicksum(U[i,j] for j in range(n))
                       - quicksum(R[i,j]       for j in range(n))
                       )
                mdl.addConstr(lhs1 >= 0, name=f"rho_BigM_minus[{i}]")
            
                # --- plus case: (Mu + x)·(ρ - ∑u) ≥ 0 ---
                lhs2 = rho * diff_plus  \
                     - (
                         Mdiag[i]*quicksum(U[i,j] for j in range(n))
                       + quicksum(R[i,j]       for j in range(n))
                       )
                mdl.addConstr(lhs2 >= 0, name=f"Icross_BigM_plus[{i}]")

            # constraints spesific to I2
            if variant.endswith('2'):            # I2 only
                for i in range(n):
                    sumUrow = gp.quicksum(U[i, j] for j in range(n)) # ∑_j U[i,j]
                    expr = rho - rho*u[i] - sum_u + sumUrow
                    mdl.addConstr(expr >= 0, name=f"rho_uUp[{i}]")

        # box‐u
        if variant.endswith('2'):
            # u ≤ 1
            for i in range(n):
                mdl.addConstr(u[i] <= 1, name=f"u_ub_{i}")
        
            # Box-u RLT alone: for all i,j
            #   U[i,j] − u[i] − u[j] + 1 ≥ 0    (UU)
            for i in range(n):
                for j in range(n):
                    mdl.addConstr(U[i,j] - u[i] - u[j] + 1 >= 0,
                                  name=f"uBox_UU[{i},{j}]")
           
            # Cross-RLT: box-u × Big-M, for all i,j
            for i in range(n):
                for j in range(n):
                    mdl.addConstr( Mdiag[i]*u[i] - x[i]
                                 - Mdiag[i]*U[i,j] + R[i,j]               >= 0,
                                  name=f"BigM_uUp[{i},{j}]")
                    mdl.addConstr( Mdiag[i]*u[i] + x[i]
                                 - Mdiag[i]*U[i,j] - R[i,j]               >= 0,
                                  name=f"BigM2_uUp[{i},{j}]")
        
            # If A x ≤ b is present, cross-RLT: (b - A x) × box-u
            if A is not None and A.size:
                p = A.shape[0]
                for r in range(p):
                    for c in range(n):

                        # (b[r] - A[r,:] x)·(1-u[c]) ≥ 0
                        expr_up = (
                            b[r]
                          - b[r]*u[c]
                          - quicksum(A[r,k]*x[k] for k in range(n))
                          + quicksum(A[r,k]*R[k,c] for k in range(n))
                        )
                        mdl.addConstr(expr_up >= 0,
                                      name=f"Ax_uUp[{r},{c}]")

    if build_only:
        return mdl

    mdl.optimize()
    
    status = mdl.Status

    if status == GRB.OPTIMAL:
         # grab the solution
         x_val, u_val = x.X, u.X


         if not is_exact:
            # switch to simplex + crossover so duals exist
            mdl.Params.Method    = 0
            mdl.Params.Crossover = 1
            mdl.optimize()
            
            # Write the dualized LP
            dualfile = f"{variant}_primal.dua"
            mdl.write(dualfile)
            # convert the .dua to a human-readable .lp
            dual = gp.read(dualfile)
            dual.write(f"{variant}_dual.lp")

 
         if is_exact:
             return mdl.ObjVal, x_val, u_val
         else:
             return mdl.ObjVal, x_val, u_val, X.X, R.X, U.X

    elif status == GRB.UNBOUNDED:
        return 'UNBOUNDED', None, None, None, None, None

    elif status == GRB.INF_OR_UNBD:
        return 'INF_OR_UNBD', None, None, None, None, None

    elif status == GRB.INFEASIBLE:
        return 'INFEASIBLE', None, None, None, None, None

    else:
        raise RuntimeError(f"Unexpected Gurobi status {status}")

    
    
    
    
    
def inspect_model(data, variant):
    """
    Build (but don’t solve) the given variant, dump its LP,
    and print every variable’s bounds and every constraint.
    """
    # 1) build
    mdl = build_and_solve(data, variant=variant, build_only=True)

    # 2) write out to file
    fname = f"{variant}_model.lp"
    mdl.write(fname)
    print(f"\n{variant}_model.lp written.\n")

    # 3) variable bounds
    print("Variable bounds:")
    for v in mdl.getVars():
        print(f"  {v.varName:10s} LB = {v.LB:>8},  UB = {v.UB:>8}")
    print()

    # 4) constraints
    print("Constraints:")
    for c in mdl.getConstrs():
        row = mdl.getRow(c)
        print(f"  {c.constrName:15s} : {row}  ({c.Sense}{c.RHS})")
    print()


if __name__=='__main__':
    # test rho 1 and 2
    data = {
    "n"  : 4,
    "rho": 3.0,

    # objective  min  –x1
    "Q0" : np.zeros((4, 4)),
    "q0" : np.array([-1.0, 0.0, 0.0, 0.0]),

    # no quadratic rows
    "Qi": None, "qi": None, "ri": None,
    "Pi": None, "pi": None, "si": None,

    # no linear rows
    "A": None, "b": None,
    "H": None, "h": None,

    # Big-M : identity
    "M": np.eye(4)
    }
    
    # data = {
    # "n"   : 2,            # dimension of x
    # "rho" : 2.0,          # sum of binaries
    # "Q0"  : np.zeros((2, 2)),
    # "q0"  : np.array([-1.0, 0.0]),
    # # -------- NEW linear block --------
    # "A"   : np.array([[ 1.0,  0.5 ]]),   # 1 × 2, definitely not identity
    # "b"   : np.array([1.0]),
    # # ----------------------------------
    # "Qi"  : None, "qi": None, "ri": None,
    # "Pi"  : None, "pi": None, "si": None,
    # "H"   : None, "h" : None,
    # "M"   : np.eye(2)
    # }

    # data = {
    #     'n'   : 2,
    #     'rho' : 1.0,
    #     # objective: minimize −x₁  ⇒  Q₀=0, q₀=[−1,0]
    #     'Q0'  : np.zeros((2,2)),
    #     'q0'  : np.array([-1.0, 0.0]),
    #     # no original quadratics
    #     'Qi'  : None, 'qi'  : None, 'ri'  : None,
    #     'Pi'  : None, 'pi'  : None, 'si'  : None,
    #     # two linear rows: x₁+x₂ ≤ 0.6,    0·x₁ − x₂ ≤ −0.1  (i.e. x₂ ≥ 0.1)
    #     'A'   : np.array([[ 1.0,  1.0],
    #                       [ 0.0, -1.0]]),
    #     'b'   : np.array([ 0.6, -0.1]),
    #     # no equalities
    #     'H'   : None, 'h'   : None,
    #     # Big-M = identity
    #     'M'   : np.eye(2)
    # }



    # 2) Then solve all variants
    for var in ['EXACT','E1','E2','I1','I2']:
        res = build_and_solve(data, variant=var)
        status = res[0]
    
        # if the first element is a string, it’s a special status
        if isinstance(status, str):
            print(f"{var:5s} → status = {status}")
            continue
    
        # otherwise it’s optimal, unpack and print
        if var == 'EXACT':
            val, x, u = res
            print(f"{var:5s} → obj {val:.3f}, x={x}, u={u}")
            print()
        else:
            val, x, u, X, R, U = res
            print(f"""{var:5s} → obj {val:.3f}, x={x}, u={u}
      X =\n{X}
      R =\n{R}
      U =\n{U}""")
            print()


    #inspect_model(data, 'E1')
    inspect_model(data, 'E1')



# 1) Read in the dualized model you wrote
dual = gp.read("E1_primal.dua")

# 2) (Optional) silence the log if you don’t want to see the solver chatter
dual.Params.LogToConsole = 0

# 3) Optimize the dual LP
dual.optimize()

# 4) Print objective
print("Dual objective value =", dual.ObjVal)

# 5) Print every dual‐variable’s name and value
print("\nDual variable values:")
for v in dual.getVars():
    print(f"  {v.VarName:12s} = {v.X:.6g}")
