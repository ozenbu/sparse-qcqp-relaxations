\ Model dual_SQCQP_RLT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 2 sigma[0]
Subject To
 sym_GUU[0,0]: = 0
 sym_GUU[0,1]: Gamma_UU[0,1] - Gamma_UU[1,0] = 0
 sym_GUU[1,0]: - Gamma_UU[0,1] + Gamma_UU[1,0] = 0
 sym_GUU[1,1]: = 0
 sym_GLL[0,0]: = 0
 sym_GLL[0,1]: Gamma_LL[0,1] - Gamma_LL[1,0] = 0
 sym_GLL[1,0]: - Gamma_LL[0,1] + Gamma_LL[1,0] = 0
 sym_GLL[1,1]: = 0
 Cx[0]: - gamma[0] + delta[0] + 2 varphi[0] = -1
 Cx[1]: - gamma[1] + delta[1] + 2 varphi[1] = 0
 CX[0,0]: Gamma_UU[0,0] + Gamma_LL[0,0] - 2 Gamma_UL[0,0] = 0
 CX[0,1]: Gamma_UU[0,1] + Gamma_LL[0,1] - Gamma_UL[0,1] - Gamma_UL[1,0]
   = 0
 CX[1,0]: Gamma_UU[1,0] + Gamma_LL[1,0] - Gamma_UL[0,1] - Gamma_UL[1,0]
   = 0
 CX[1,1]: Gamma_UU[1,1] + Gamma_LL[1,1] - 2 Gamma_UL[1,1] = 0
 Cu[0]: - gamma[0] - delta[0] + sigma[0] - tau[0] - 2 psi[0] = 0
 Cu[1]: - gamma[1] - delta[1] + sigma[0] - tau[1] - 2 psi[1] = 0
 CR[0,0]: varphi[0] + Gamma_UU[0,0] - Gamma_LL[0,0] = 0
 CR[0,1]: varphi[0] + Gamma_UU[0,1] - Gamma_LL[0,1] + Gamma_UL[0,1]
   - Gamma_UL[1,0] = 0
 CR[1,0]: varphi[1] + Gamma_UU[1,0] - Gamma_LL[1,0] - Gamma_UL[0,1]
   + Gamma_UL[1,0] = 0
 CR[1,1]: varphi[1] + Gamma_UU[1,1] - Gamma_LL[1,1] = 0
 CU[0,0]: tau[0] + psi[0] - 0.5 Gamma_UU[0,0] - 0.5 Gamma_LL[0,0]
   - Gamma_UL[0,0] = 0
 CU[0,1]: 0.5 psi[0] + 0.5 psi[1] - 0.5 Gamma_UU[0,1] - 0.5 Gamma_LL[0,1]
   - 0.5 Gamma_UL[0,1] - 0.5 Gamma_UL[1,0] = 0
 CU[1,0]: 0.5 psi[0] + 0.5 psi[1] - 0.5 Gamma_UU[1,0] - 0.5 Gamma_LL[1,0]
   - 0.5 Gamma_UL[0,1] - 0.5 Gamma_UL[1,0] = 0
 CU[1,1]: tau[1] + psi[1] - 0.5 Gamma_UU[1,1] - 0.5 Gamma_LL[1,1]
   - Gamma_UL[1,1] = 0
Bounds
 sigma[0] free
 tau[0] free
 tau[1] free
 varphi[0] free
 varphi[1] free
 psi[0] free
 psi[1] free
End
