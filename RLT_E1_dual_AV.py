import numpy as np
import gurobipy as gp
from gurobipy import GRB

def as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple)) else [obj]

# outer product 
def outer(a, b):
    return a[:, None] * b[None, :]

def build_dual(data, variant="E1", solve=False, verbose=False):
    n   = data['n']
    rho = data['rho']

    Q0, q0 = data['Q0'], data['q0']
    Qi = as_list(data.get('Qi'));   qi = as_list(data.get('qi'));   ri = as_list(data.get('ri'))
    Pi = as_list(data.get('Pi'));   pi = as_list(data.get('pi'));   si = as_list(data.get('si'))

    A = data.get('A');  b = data.get('b')
    if A is None:
        A = np.zeros((0, n));  b = np.zeros(0)
    ell = A.shape[0]

    H = data.get('H');  h = data.get('h')
    if H is None:
        H = np.zeros((0, n));  h = np.zeros(0)
    eta = H.shape[0]

    # Big-M
    Mraw = data['M']
    Mdiag = Mraw if Mraw.ndim == 1 else np.diag(Mraw)
    Mmat  = np.diag(Mdiag)

    # the numeric ones‐vector
    e = np.ones(n)

    # create model
    mdl = gp.Model("dual_SQCQP_RLT")
    mdl.Params.LogToConsole = int(verbose)

    # -------------------------------------------------------------------------
    # dual variables
    # -------------------------------------------------------------------------
    m, k = len(Qi), len(Pi)

    alpha  = mdl.addMVar( m,          lb=0.0,           name="alpha")
    beta   = mdl.addMVar( k,          lb=-GRB.INFINITY, name="beta")
    mu     = mdl.addMVar( ell,        lb=0.0,           name="mu")    
    lam    = mdl.addMVar( eta,        lb=-GRB.INFINITY, name="lam")  
    gamma  = mdl.addMVar( n,          lb=0.0,           name="gamma")
    delta  = mdl.addMVar( n,          lb=0.0,           name="delta")
    sigma = mdl.addMVar(1, lb=-GRB.INFINITY, ub= GRB.INFINITY, name="sigma")[0]
    tau    = mdl.addMVar( n,          lb=-GRB.INFINITY, name="tau")
    varphi = mdl.addMVar( n,          lb=-GRB.INFINITY, name="varphi")
    psi    = mdl.addMVar( n,          lb=-GRB.INFINITY, name="psi")
    
    Theta = mdl.addMVar((ell, ell),   lb=0.0,           name="Theta") 
    Phi   = mdl.addMVar((eta,  n),    lb=-GRB.INFINITY, name="Phi")    
    Xi    = mdl.addMVar((eta,  n),    lb=-GRB.INFINITY, name="Xi")     
    GAU   = mdl.addMVar((ell,  n),    lb=0.0,           name="Gamma_AU")
    GAL   = mdl.addMVar((ell,  n),    lb=0.0,           name="Gamma_AL")
    GUU = mdl.addMVar((n, n), lb=0.0, name="Gamma_UU")
    GLL = mdl.addMVar((n, n), lb=0.0, name="Gamma_LL")
    GUL = mdl.addMVar((n, n), lb=0.0, name="Gamma_UL")
    

    # symmetry
    mdl.addConstr(Theta == Theta.T, name="sym_Theta")
    mdl.addConstr(GUU == GUU.T,   name="sym_GUU")
    mdl.addConstr(GLL == GLL.T,   name="sym_GLL")



    if variant == "E1":

        # C_x == 0
        Cx = -gp.quicksum(alpha[i] * qi[i] for i in range(m)) - gp.quicksum(beta [j] * pi[j] for j in range(k)) - gamma + delta + rho * varphi  
        if ell:
            Cx +=  -A.T @ mu - (A.T @ Theta @ b) - (GAU.T - GAL.T) @ b
        if eta:
            Cx += - H.T @ lam + Phi.T @ h
        mdl.addConstr(Cx == q0, name="Cx")
    
        # C_X == 0
        CX = - gp.quicksum(alpha[i] * Qi[i] for i in range(m)) - gp.quicksum(beta [j] * Pi[j] for j in range(k)) + GUU + GLL - GUL - GUL.T
        if ell:
            CX +=  A.T @ Theta @ A + A.T @ (GAU - GAL) + (GAU.T - GAL.T) @ A
        if eta:
            CX += - H.T @ Phi - Phi.T @ H
        mdl.addConstr(CX == Q0, name="CX")
        
        # C_u == 0
        Cu = - Mmat @ (gamma + delta) + sigma * e - tau - rho * psi
        if ell:
            Cu += - Mmat @ ((GAU.T + GAL.T) @ b)
        if eta:
            Cu += - Xi.T @ h
        mdl.addConstr(Cu == 0, name="Cu")
    
        # C_R == 0
        CR = (GUU - GLL + GUL - GUL.T) @ Mmat
        if ell:
            CR += A.T @ (GAU + GAL) @ Mmat
        if eta:
            CR += H.T @ Xi
        CR += outer(varphi, e)
        mdl.addConstr(CR == 0, name="CR")
    
        # C_U == 0
        CU = - 0.5 * Mmat @ (GUU + GLL + GUL + GUL.T) @ Mmat 
        CU += 0.5*(outer(psi, e)+ outer(e,psi))
        # add Diag(tau)
        for i in range(n):
            CU[i, i] += tau[i]
        mdl.addConstr(CU == 0, name="CU")
    
        # -------------------------------------------------------------------------
        # dual objective
        # -------------------------------------------------------------------------
        

        obj = (
            gp.quicksum(alpha[i] * ri[i] for i in range(m))
            + gp.quicksum(beta[j] * si[j] for j in range(k))
            - mu.T @ b                   
            - lam.T @ h                 
            - rho * sigma
            - 0.5* (b.T @ Theta @ b)
        )
        mdl.setObjective(obj, GRB.MAXIMIZE)

    if solve:
        mdl.optimize()

    return mdl



def inspect_model(data, variant):
    """
    Build (but don’t solve) the given variant, dump its LP,
    and print every variable’s bounds and every constraint.
    """
    # build
    mdl = build_dual(data, variant="E1", solve=True, verbose=True)

    # write out to file
    fname = f"{variant}_my_dual.lp"
    mdl.write(fname)
    print(f"\n{variant}_my_dual.lp written.\n")

    # variable bounds
    print("Variable bounds:")
    for v in mdl.getVars():
        print(f"  {v.varName:10s} LB = {v.LB:>8},  UB = {v.UB:>8}")
    print()
    
    # print the dual objective (expression and value)
    print("Dual objective expression:")
    print(" ", mdl.getObjective())          # symbolic objective
    print("Optimal objective value:", mdl.ObjVal)
    print()

    # constraints
    print("Constraints:")
    for c in mdl.getConstrs():
        row = mdl.getRow(c)                 # MLinExpr
        print(f"  {c.ConstrName:15s} : {row}  ({c.Sense}{c.RHS})")
  
    for qc in mdl.getQConstrs():
        qexpr  = mdl.getQCRow(qc)          # QuadExpr (includes the linear part)
        sense  = qc.QCSense.replace('<', '≤').replace('>', '≥')
        rhs    = qc.QCRHS
        print(f"  {qc.QCName:15s} : {qexpr}  ({sense}{rhs})")




# ---------------------------------------------------------------------------
# toy test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = {
        'n': 2, 'rho': 2.0,
        'Q0': np.zeros((2, 2)),  'q0': np.array([-1.0, 0.0]),
        'Qi': None, 'qi': None, 'ri': None,
        'Pi': None, 'pi': None, 'si': None,
        'A': None,  'b': None,
        'H': None,  'h': None,
        'M': np.eye(2)
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

    mdl = build_dual(data, variant="E1", solve=True, verbose=True)
                                                                                                                                                                             print("Dual objective =", mdl.ObjVal)
    for v in mdl.getVars():
        print(f"{v.VarName} = {v.X}")
        
        
    inspect_model(data, variant="E1")
    
    



