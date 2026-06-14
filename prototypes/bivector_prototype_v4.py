"""Rung 3: curvature Σ_i = dσ_i + (σ∧σ terms), expand in Z^j/Z̃^j -> Weyl C_ij.
MH eq34: Σ1=dσ1+2σ1∧σ2, Σ2=dσ2+σ1∧σ3, Σ3=dσ3+2σ2∧σ3.
MH eq33: Σ_i = (C_ij + (R/6)γ_ij) Z^j + E_ij Z̃^j.
For vacuum: Σ_i = C_ij Z^j only (R=0, E=0). Read Ψ2 = C_22 (the central entry).
"""
import sys; sys.path.insert(0,'/home/claude/handoff')
import sympy as sp
from pick.karlhede import frame_vectors_from_coframe, null_coframe_from_diagonal_metric
from pick.metrics import METRICS

m=METRICS['schwar'](); g,coords=m['g'],m['coords']; th=coords[2]
def simp(e):
    e=sp.cancel(e)
    if e.has(sp.sin,sp.cos,sp.Abs,sp.sign,sp.DiracDelta,sp.sqrt):
        e=sp.refine(sp.simplify(e),sp.Q.positive(sp.sin(th)))
    return e
n=4; gi=g.inv()
l,nv,mv,mb=null_coframe_from_diagonal_metric(g,'lorentzian',simp_fn=sp.cancel)
cf=[list(l),list(nv),list(mv),list(mb)]; legs=frame_vectors_from_coframe(cf,gi,n)

def wedge1(A,B): return {(mu,nu):simp(A[mu]*B[nu]-A[nu]*B[mu]) for mu in range(n) for nu in range(n) if mu<nu}
def neg(d): return {k:-v for k,v in d.items()}
def addd(*ds):
    out={}
    for d in ds:
        for k,v in d.items(): out[k]=simp(out.get(k,sp.S.Zero)+v)
    return out
Z={1:neg(wedge1(cf[1],cf[3])),2:addd(neg(wedge1(cf[0],cf[1])),wedge1(cf[2],cf[3])),3:wedge1(cf[0],cf[2])}

# reuse the connection solve from v3
exec(open('/tmp/biv3.py').read().split("print(\"\\n=== spin coefficients")[0])
# now sig[i] = list of coordinate components of σ_i (from the solved system)
sigma={i:[simp(so.get(s[i][mu],s[i][mu])) for mu in range(n)] for i in [1,2,3]}

def dform1(si):  # d of 1-form -> 2-form dict, (dσ)_{μν}=∂_μ σ_ν - ∂_ν σ_μ
    return {(mu,nu):simp(sp.diff(si[nu],coords[mu])-sp.diff(si[mu],coords[nu])) for mu in range(n) for nu in range(n) if mu<nu}
def w11(A,B):  # σ∧σ' for two 1-forms (coordinate component lists) -> 2-form
    return {(mu,nu):simp(A[mu]*B[nu]-A[nu]*B[mu]) for mu in range(n) for nu in range(n) if mu<nu}
def scale(d,k): return {key:simp(k*v) for key,v in d.items()}

Sig={}
Sig[1]=addd(dform1(sigma[1]), scale(w11(sigma[1],sigma[2]),2))
Sig[2]=addd(dform1(sigma[2]), w11(sigma[1],sigma[3]))
Sig[3]=addd(dform1(sigma[3]), scale(w11(sigma[2],sigma[3]),2))

# Expand each Σ_i in the Z^j basis. Z^j are self-dual 2-forms; for vacuum
# Σ_i = C_ij Z^j. Solve for C_ij by matching components.
# Build a basis-matching: pick independent components of the Z^j.
def comps(d): return d  # 2-form as dict over (μ<ν)
keys=[(a,b) for a in range(n) for b in range(n) if a<b]  # 6 independent 2-form comps
import sympy as sp2
def vec(d): return sp.Matrix([d.get(k,sp.S.Zero) for k in keys])
ZM=sp.Matrix.hstack(*[vec(Z[j]) for j in [1,2,3]])  # 6x3
print("Solving Σ_i = C_ij Z^j for each i (least squares over 6 comps, 3 unknowns):")
Cmat=sp.zeros(3,3)
for i in [1,2,3]:
    b=vec(Sig[i])
    # solve ZM * c = b  (overdetermined but consistent if Σ_i is self-dual)
    c=ZM.solve_least_squares(b)
    for j in range(3): Cmat[i-1,j]=simp(c[j])
print("C matrix (rows i=1..3, cols j=1..3):")
for i in range(3): print("  ", [Cmat[i,j] for j in range(3)])
# MH eq29: C=[[Ψ0,Ψ1,Ψ2],[Ψ1,Ψ2,Ψ3],[Ψ2,Ψ3,Ψ4]]. Ψ2 is C[0,2]=C[1,1]=C[2,0].
print("\nΨ2 candidates: C[0,2]=",simp(Cmat[0,2])," C[1,1]=",simp(Cmat[1,1])," C[2,0]=",simp(Cmat[2,0]))
print("Expected Ψ2 = -M/r³")
