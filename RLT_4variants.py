import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum


def as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple)) else [obj]

# outer product 
def outer(a, b):
    return a[:, None] * b[None, :]

def build_and_solve(data, variant='EXACT', verbose=False, build_only=False):
    """
    Solve either the exact MIQCQP ('EXACT') or one of the
    first-level RLT relaxations ('E1','E2','I1','I2') via Gurobi.
    If build_only=True, returns the Model without solving.
    """
    # -- unpack & defaults --
    n = data['n']
    rho = data['rho']

    Q0, q0 = data['Q0'], data['q0']
    Qi = as_list(data.get('Qi'))
    qi = as_list(data.get('qi'))
    ri = as_list(data.get('ri'))
    Pi = as_list(data.get('Pi'))
    pi = as_list(data.get('pi'))
    si = as_list(data.get('si'))

    A = data.get('A')
    b = data.get('b')
    if A is None:
        A = np.zeros((0, n))
        b = np.zeros(0)
    ell = A.shape[0]

    H = data.get('H')
    h = data.get('h')
    if H is None:
        H = np.zeros((0, n))
        h = np.zeros(0)
    eta = H.shape[0]

    # Big-M
    Mraw = data['M']
    Mdiag = Mraw if Mraw.ndim == 1 else np.diag(Mraw)
    Mmat = np.diag(Mdiag)

    # the numeric ones‐vector
    e = np.ones(n)

    # create model
    mdl = gp.Model("dual_SQCQP_RLT")
    mdl.Params.LogToConsole = int(verbose)

    x = mdl.addMVar(n, lb=-GRB.INFINITY, name="x")

    if variant == 'EXACT':
        u = mdl.addMVar(n, vtype=GRB.BINARY, name="u")
    else:
        u = mdl.addMVar(n, vtype=GRB.CONTINUOUS,
                        lb=-GRB.INFINITY, ub=GRB.INFINITY,
                        name="u")

    # create lifted MVars
    X = mdl.addMVar((n, n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="X")
    R = mdl.addMVar((n, n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="R")
    U = mdl.addMVar((n, n), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="U")

    # 1) enforce symmetry of X and U with one matrix‐constraint each:
    mdl.addConstr(X == X.transpose(), name="sym_X")
    mdl.addConstr(U == U.transpose(), name="sym_U")

    # objective (same as before) …
    if variant == 'EXACT':
        mdl.setObjective(0.5*(x @ Q0 @ x) + q0@x, GRB.MINIMIZE)
        mdl.Params.NonConvex = 2

    else:
        mdl.setObjective(0.5*(quicksum(Q0[i, j]*X[i, j] for i in range(n) for j in range(n)))
                         + quicksum(q0[i]*x[i] for i in range(n)),
                         GRB.MINIMIZE)

    # original constraints

    # linear & Big-M
    if ell:
        mdl.addConstr(A@x <= b, "Ax<=b")
    if eta:
        mdl.addConstr(H@x == h, "Hx=h")
    mdl.addConstr(u.sum() <= rho, "card")
    mdl.addConstr(-Mmat@u <= x, "BigM_low")
    mdl.addConstr(x <= Mmat@u, "BigM_up")

    if variant != 'EXACT':

        for Q_mat, q_vec, r_term in zip(Qi, qi, ri):
            expr = 0.5*quicksum(Q_mat[i, j]*X[i, j]
                                for i in range(n) for j in range(n))
            expr += quicksum(q_vec[i]*x[i] for i in range(n)) + r_term
            mdl.addConstr(expr <= 0)
        for P_mat, p_vec, s_term in zip(Pi, pi, si):
            expr = 0.5*quicksum(P_mat[i, j]*X[i, j]
                                for i in range(n) for j in range(n))
            expr += quicksum(p_vec[i]*x[i] for i in range(n)) + s_term
            mdl.addConstr(expr == 0)

        # diag(U)=u
        for i in range(n):
            mdl.addConstr(U[i,i] == u[i])
            mdl.addConstr(U[i,i] == u[i], name=f"diagU_{i}")
        # H X = h x^T and H R = h u^T
        if eta:
            mdl.addConstr(H @ X == outer(h,x), name="HX_eq_hxT")
            mdl.addConstr(H @ R == outer(h,u), name="HR_eq_huT")


        # 7) “Big‐M McCormick” (Upper–Upper, Lower–Lower, Upper–Lower) in three matrix inequalities:
        #    (M U M  –  M R^T  –  R M  +  X)  ≥ 0  elementwise
        mdl.addConstr(
            Mmat @ U @ Mmat
            - Mmat @ R.T
            - R @ Mmat
            + X >= 0,
            name="BigM_UU"
        )
        mdl.addConstr(
            Mmat @ U @ Mmat
            + Mmat @ R.T
            + R @ Mmat
            + X >= 0,
            name="BigM_LL"
        )
        mdl.addConstr(
            Mmat @ U @ Mmat
            + Mmat @ R.T
            - R @ Mmat
            - X >= 0,
            name="BigM_UL"
        )

        # RLT from A x ≤ b
        if ell:
            # (b – A x)(b – A x)^T ≥ 0  
            mdl.addConstr(
                A @ X @ A.T - A @ outer(x, b) - outer(b,x) @ A.T + outer(b,b) >= 0,
                name="AxAx"
            )
            # (b - A x) (M u - x).T 
            mdl.addConstr(
               A @ X - outer(b,x) - A @ R @ Mmat + outer(b,u) @ Mmat >= 0,
                name="AxBigM_low"
            )
            # (b - A x) (M u + x).T 
            mdl.addConstr(
                - A @ X + outer(b,x) - A @ R @ Mmat + outer(b,u) @ Mmat >= 0,
                name="AxBigM_up"
            )


        # equality case E1/E2:
        if variant[0] == 'E':
            # e^T u = ρ
            mdl.addConstr(u.sum() == rho, name="card_eq")
            # R e = ρ·x and  U e = ρ·u
            mdl.addConstr(R @ e == rho * x, name="Re_eq_rho_x")
            mdl.addConstr(U @ e == rho * u, name="Ue_eq_rho_u")
        
        # inequality case I1/I2:
        else:
            # e^T u ≤ ρ
            mdl.addConstr(u.sum() <= rho, name="card_leq")

            # Self‐RLT (ρ - e^T u)^2 ≥ 0  
            mdl.addConstr(
                rho*rho
                - 2 * rho * u.sum()
                + e.T @ U @ e
                >= 0,
                name="rho_self_RLT"
            )

            # (b-Ax)(ρ - eᵀu)^T  
            if ell:
                mdl.addConstr(
                    rho * b - outer(b,u) @ e
                    - rho*(A @ x)
                    + A @ R @ e >= 0,
                    name="Ax_rho_eu"
                )

            # Big‐M × (ρ - eᵀu) 
            # (Mu-x)(ρ - eᵀu)^T 
            mdl.addConstr(
                rho * (Mmat @ u) - rho* x
                - Mmat @ U @ e + R @ e >= 0,
                name="BigM_rho_eu_low"
            )
            # (Mu + x)(ρ - eᵀu)^T 
            mdl.addConstr(
                rho * (Mmat @ u) + rho* x
                - Mmat @ U @ e - R @ e  >= 0,
                name="BigM_rho_eu_up"
            )
        
        if variant.endswith('2'):
            
            mdl.addConstr(u <= np.ones(n), name="u_bound")
            
            # self-RLT (e-u)(e-u).T
            # U - u e^T - e u^T + e e^T ≥ 0  as n×n:
            eeT = np.ones((n, n))
            mdl.addConstr( 
                U - outer(u,e) - outer(e,u) + eeT >= 0,
                name="u_bound_self"
            )
            
            # (b - Ax) × (e - u)^T 
            if ell:
                mdl.addConstr(
                    + outer(b,e) - outer(b,u) - outer(A @ x, e) + A @ R >= 0,   
                    name="Ax_ubound"
                )

            # BigM × (e - u)^T  
            # (Mu - x)(e - u)^T 
            mdl.addConstr(
                Mmat @ outer(u, e) - outer(x,e) - Mmat @ U + R >= 0,
                name="BigM_low_ubound"
            )
            # (Mu + x)(e - u)^T 
            mdl.addConstr(
                Mmat @ outer(u, e) + outer(x,e) - Mmat @ U - R >= 0,
                name="BigM_up_u_bound"
            )

            # For I2 only
            if variant[0] == 'I' and variant.endswith('2'):
                
                # (ρ - eᵀu)(e - u)^T 
                mdl.addConstr(
                    rho * e  - rho * u - u.sum() * e + U @ e >= 0,
                    name="rho_u_bound"
                )
                
                
    if build_only:
        return mdl

    mdl.optimize()

    status = mdl.Status

    if status == GRB.OPTIMAL:
        # grab the solution
        x_val, u_val = x.X, u.X

        if variant != 'EXACT':
            # switch to simplex + crossover so duals exist
            mdl.Params.Method = 0
            mdl.Params.Crossover = 1
            mdl.optimize()

            # Write the dualized LP
            dualfile = f"{variant}_primal.dua"
            mdl.write(dualfile)
            # convert the .dua to a human-readable .lp
            dual = gp.read(dualfile)
            dual.write(f"{variant}_dual.lp")

        if variant == 'EXACT':
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


if __name__ == '__main__':
    
    
    
    # I1 weaker than all 
    # data = {
    #     'n': 2, 'rho': 2.0,
    #     'Q0': np.zeros((2, 2)), 'q0': np.array([-1.0, 0.0]),
    #     'Qi': None, 'qi': None, 'ri': None,
    #     'Pi': None, 'pi': None, 'si': None,
    #     'A': None, 'b': None, 'H': None, 'h': None,
    #     'M': np.eye(2)
    # }
    
    
    
    # linear rows added
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
    
    
    
    # Exact = E2 = I2 > E1 > I1
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
    # "n"  : 3,
    # "rho": 2.0,                  # ρ ≤ n

    # # Amaç :  min  –x1 – x2     (x1’ i ve x2’ yi olabildiğince büyüt)
    # "Q0" : np.zeros((3,3)),
    # "q0" : np.array([-1.0, -1.0, 0.0]),

    # # Küçük bir “kaynak” limiti:  x2 + x3 ≤ 0.4
    # "A"  : np.array([[0., 1., 1.]]),
    # "b"  : np.array([0.4]),

    # # Başka kuadratik / eşitlik yok
    # "Qi": None, "qi": None, "ri": None,
    # "Pi": None, "pi": None, "si": None,
    # "H" : None, "h" : None,

    # # Gevşek Big-M  (5)
    # "M" : 5 * np.eye(3)
    # }

    q0_opt = [-1, 0, 0, 0]
    data = {
    "n"  : 4,
    "rho": 3.0,

    # objective  min  –x1
    "Q0" : np.zeros((4, 4)),
    "q0" : np.array(q0_opt),

    # no quadratic rows
    "Qi": None, "qi": None, "ri": None,
    "Pi": None, "pi": None, "si": None,

    # no linear rows
    "A": None, "b": None,
    "H": None, "h": None,

    # Big-M : identity
    "M": np.eye(4)
    }
    

    n   = 4
    rho = 3.0
    
    ell = np.array([-1.0, -0.5, -2.0, -0.3])   # lower bounds
    u    = np.array([ 1.0,  0.8,  0.7,  2.5])  # upper bounds
    
    # Big-M vector and diagonal matrix
    Mvec = np.maximum(np.abs(ell), np.abs(u))
    M    = np.diag(Mvec)
    
    # Ax ≤ b for the box:  [ I; -I ] x ≤ [u; -ell]
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([u, -ell])
    
    data3 = {
        "n":   n,
        "rho": rho,
        "Q0":  np.zeros((n, n)),
        "q0":  np.array([-100.0, 0.0, 100.0, -100.0]),
        "Qi":  None, "qi": None, "ri": None,
        "Pi":  None, "pi": None, "si": None,
        "A":   A, "b": b,
        "H":   None, "h": None,
        "M":   M
    }

   
    for var in ['EXACT', 'E1', 'E2', 'I1', 'I2']:
        res = build_and_solve(data3, variant=var)
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

    # inspect_model(data, 'E1')
    #inspect_model(data, 'I1')
