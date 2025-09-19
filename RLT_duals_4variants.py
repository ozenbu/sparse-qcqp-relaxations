#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified dual builder for SQCQP-RLT variants

  variant = "E1" :  eᵀu = ρ
            "E2" :  eᵀu = ρ  &   u ≤ e
            "I1" :  eᵀu ≤ ρ
            "I2" :  eᵀu ≤ ρ  &   u ≤ e
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ------------------------------------------------------------------ helpers
def as_list(obj):
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple)) else [obj]


def outer(a, b):               # rank-1 matrix  a bᵀ
    return a[:, None] * b[None, :]





# ------------------------------------------------------------------ builder
def build_dual(data, variant="E1", solve=False, verbose=False):
    if variant not in {"E1", "E2", "I1", "I2"}:
        raise ValueError("variant must be one of {'E1','E2','I1','I2'}")

    # ------------------------- problem data
    n, rho = data["n"], data["rho"]
    Q0, q0 = data["Q0"], data["q0"]
    Qi,  qi,  ri = map(as_list, (data.get("Qi"), data.get("qi"),
                                 data.get("ri")))
    Pi,  pi,  si = map(as_list, (data.get("Pi"), data.get("pi"),
                                 data.get("si")))

    A, b = data.get("A"), data.get("b")
    if A is None:
        A = np.zeros((0, n));  b = np.zeros(0)
    ell = A.shape[0]

    H, h = data.get("H"), data.get("h")
    if H is None:
        H = np.zeros((0, n));  h = np.zeros(0)
    eta = H.shape[0]

    Mraw = data["M"]                       # Big-M
    Mdiag = Mraw if Mraw.ndim == 1 else np.diag(Mraw)
    Mmat  = np.diag(Mdiag)
    e     = np.ones(n)

    # ------------------------------------------------- model + core multipliers
    mdl = gp.Model(f"dual_{variant}")
    mdl.Params.LogToConsole = int(verbose)

    m, k = len(Qi), len(Pi)

    alpha  = mdl.addMVar(m,   lb=0.0,           name="alpha")
    beta   = mdl.addMVar(k,   lb=-GRB.INFINITY, name="beta")
    mu     = mdl.addMVar(ell, lb=0.0,           name="mu")
    lam    = mdl.addMVar(eta, lb=-GRB.INFINITY, name="lam")
    gamma  = mdl.addMVar(n,   lb=0.0,           name="gamma")
    delt   = mdl.addMVar(n,   lb=0.0,           name="delta")
    sigma = mdl.addMVar(1, lb=-GRB.INFINITY, ub= GRB.INFINITY, name="sigma")[0]
    tau    = mdl.addMVar(n,   lb=-GRB.INFINITY, name="tau")

    # φ, ψ only when eᵀu = ρ is enforced (E-variants)
    if variant in {"E1", "E2"}:
        varphi = mdl.addMVar(n, lb=-GRB.INFINITY, name="varphi")
        psi    = mdl.addMVar(n, lb=-GRB.INFINITY, name="psi")

    Theta  = mdl.addMVar((ell, ell), lb=0.0, name="Theta")
    Phi    = mdl.addMVar((eta, n),   lb=-GRB.INFINITY, name="Phi")
    Xi     = mdl.addMVar((eta, n),   lb=-GRB.INFINITY, name="Xi")
    GAU    = mdl.addMVar((ell, n),   lb=0.0, name="Gamma_AU")
    GAL    = mdl.addMVar((ell, n),   lb=0.0, name="Gamma_AL")
    GUU    = mdl.addMVar((n, n), lb=0.0, name="Gamma_UU")
    GLL    = mdl.addMVar((n, n), lb=0.0, name="Gamma_LL")
    GUL    = mdl.addMVar((n, n), lb=0.0, name="Gamma_UL")

    mdl.addConstr(Theta == Theta.T, name="sym_Theta")
    mdl.addConstr(GUU   == GUU.T,   name="sym_GUU")
    mdl.addConstr(GLL   == GLL.T,   name="sym_GLL")

    # --------------- E2 / I2 : box-u multipliers
    if variant in {"E2", "I2"}:
        kappa   = mdl.addMVar(n,     lb=0.0,           name="kappa")
        Lambda  = mdl.addMVar((n,n), lb=0.0,           name="Lambda")
        Omega   = mdl.addMVar((ell,n), lb=-0.0, name="Omega")
        Piplus  = mdl.addMVar((n,n), lb=-0.0, name="Pi_plus")
        Piminus = mdl.addMVar((n,n), lb=-0.0, name="Pi_minus")
        mdl.addConstr(Lambda == Lambda.T, name="sym_Lambda")

    # --------------- I1 / I2 : ρ-SUM multipliers
    if variant in {"I1", "I2"}:
        upsilon = mdl.addMVar(ell, lb=0.0, name="upsilon")
        chi_p   = mdl.addMVar(n,   lb=0.0, name="chi_plus")
        chi_m   = mdl.addMVar(n,   lb=0.0, name="chi_minus")
        zeta    = mdl.addMVar(1, lb=0.0, ub= GRB.INFINITY, name="zeta")[0]

        sigma.lb = 0.0                       # σ ≥ 0 in I-variants

    # --------------- I2 only : (ρ-SUM)×(box-u)
    if variant == "I2":
        iota = mdl.addMVar(n, lb=0.0, name="iota")



    # ================================================= core stationarity rows
    # -- Cx -----------------------------------------------------------
    Cx = gp.quicksum(alpha[i]*qi[i] for i in range(m)) \
         + gp.quicksum(beta[j]*pi[j]  for j in range(k))  \
         + gamma - delt + q0
    if ell: Cx += A.T@mu + A.T@Theta@b + (GAU.T-GAL.T)@b
    if eta: Cx += H.T@lam - Phi.T@h
    
    if variant in {"E1", "E2"}:
        Cx += - rho * varphi
   
    if variant in {"E2", "I2"}:
       Cx += -A.T@Omega@e + (Piplus- Piminus)@e
       
    if variant in {"I1", "I2"}:
        Cx +=  rho*A.T@upsilon + rho*(chi_p-chi_m)

    mdl.addConstr(Cx == 0, name="Cx")



    # -- CX -----------------------------------------------------------
    CX = -gp.quicksum(alpha[i]*Qi[i] for i in range(m)) \
         -gp.quicksum(beta[j]*Pi[j]  for j in range(k)) \
         + GUU + GLL - GUL - GUL.T
    if ell:
        CX += A.T@Theta@A + A.T@(GAU-GAL) + (GAU.T-GAL.T)@A
    if eta:
        CX += H.T@Phi + Phi.T@H * (-1)
    mdl.addConstr(CX == Q0, name="CX")



    # -- Cu -----------------------------------------------------------
    Cu = -Mmat@(gamma+delt) + sigma*e - tau
    if ell: 
        Cu += -Mmat@((GAU.T+GAL.T)@b)
    if eta: 
        Cu += -Xi.T@h
    
    if variant in {"E1","E2"}: 
        Cu += -rho*psi
    
    if variant in {"E2", "I2"}:
       Cu += kappa + Omega.T@b - Mmat@(Piplus+Piminus)@e + Lambda@e
       
    if variant in {"I1", "I2"}:
       Cu += -rho*Mmat@(chi_p+chi_m) + 2*rho*zeta*e
       if ell:
           Cu +=  outer(e,upsilon)@b
       
    if variant == "I2":
        Cu += rho*iota + e*(e.T@iota)
    
    mdl.addConstr(Cu == 0, name="Cu")



    # -- CR -----------------------------------------------------------
    CR = (GUU - GLL + GUL - GUL.T)@Mmat
    if ell: CR += A.T@(GAU+GAL)@Mmat
    if eta: CR += H.T@Xi
    if variant in {"E1","E2"}: 
        CR += outer(varphi, e)
    
    if variant in {"E2", "I2"}:
        CR += -A.T@Omega - Piplus + Piminus  

    if variant in {"I1", "I2"}:
        CR += -A.T@upsilon@e.T - outer(chi_p-chi_m, e)
    mdl.addConstr(CR == 0, name="CR")



    # -- CU -----------------------------------------------------------
    CU = -0.5*Mmat@(GUU + GLL + GUL + GUL.T)@Mmat
    if variant in {"E1","E2"}:
        CU += 0.5*(outer(psi,e)+outer(e,psi))
    for i in range(n):
        CU[i,i] += tau[i]
        
    if variant in {"E2", "I2"}:
        CU += -0.5*Lambda +0.5*( Mmat@(Piplus+Piminus) + (Piplus.T+Piminus.T)@Mmat ) 
        
    if variant in {"I1", "I2"}:
        CU += -zeta * outer(e, e) \
              + 0.5 * ( Mmat @ outer(chi_p + chi_m, e)    # v eᵀ   with v = M(χ₊+χ₋)
                      + outer(e, chi_p + chi_m) @ Mmat )  # e vᵀ   (no extra M on RHS)
  
    if variant == "I2":
        CU += -0.5*(outer(iota,e)+outer(e,iota)) 
    
    mdl.addConstr(CU == 0, name="CU")


    # ================================================= dual objective
    obj = gp.quicksum(alpha[i]*ri[i] for i in range(m)) \
        + gp.quicksum(beta[j]*si[j]  for j in range(k)) \
        - mu.T@b  - lam.T@h  - 0.5*(b.T@Theta@b) - rho*sigma

    if variant in {"I1","I2"}:
        obj +=- rho*upsilon.T@b - (rho**2)*zeta

    # box-u pieces
    if variant in {"E2","I2"}:
        obj += -kappa.T@e - e.T@Omega.T@b - 0.5*e.T@Lambda@e

    # (ρ-SUM)×(box-u) piece
    if variant == "I2":
        obj += -rho*iota.T@e

    mdl.setObjective(obj, GRB.MAXIMIZE)

    if solve:
        mdl.Params.InfUnbdInfo = 1   # ask Gurobi to distinguish
        mdl.optimize()

        st = mdl.Status
        if st == GRB.OPTIMAL or st == GRB.SUBOPTIMAL:
            print(f" obj({var}) =", mdl.ObjVal)
        elif st == GRB.INFEASIBLE:
            print(f"{var}: model is INFEASIBLE")
        elif st == GRB.UNBOUNDED:
            print(f"{var}: model is UNBOUNDED")
        else:
            print(f"{var}: solver status {st}")

    return mdl


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    
    # data = {
    #     "n": 2, "rho": 2.0,
    #     "Q0": np.zeros((2,2)), "q0": np.array([-1.,0]),
    #     "Qi": None, "qi": None, "ri": None,
    #     "Pi": None, "pi": None, "si": None,
    #     "A": None, "b": None,
    #     "H": None, "h": None,
    #     "M": np.eye(2)
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

    q0_opt = [-8, -6, 130, -17]
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

    for var in ("E1","E2","I1","I2"):
        print(f"\n=== building {var} ===")
        mdl = build_dual(data, variant=var, solve=True, verbose=False)
        # for v in mdl.getVars():
        #     print(f"{v.VarName} = {v.X}")
