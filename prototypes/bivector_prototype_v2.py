"""Test: self-dual bivectors via FRAME Hodge star with eps_{lnmmb}=±i
(the complex-null-basis Jacobian factor), not coordinate sqrt(-g)."""
import sys; sys.path.insert(0,'/home/claude/handoff')
import sympy as sp
from pick.karlhede import frame_vectors_from_coframe, null_coframe_from_diagonal_metric
from pick.metrics import METRICS

m = METRICS['schwar']()
g, coords = m['g'], m['coords']; th=coords[2]
def simp(e):
    e=sp.cancel(e)
    if e.has(sp.sin,sp.cos,sp.Abs,sp.sign,sp.DiracDelta,sp.sqrt):
        e=sp.refine(sp.simplify(e),sp.Q.positive(sp.sin(th)))
    return e
n=4; gi=g.inv()
l,nv,mv,mb=null_coframe_from_diagonal_metric(g,'lorentzian',simp_fn=sp.cancel)
cf=[list(l),list(nv),list(mv),list(mb)]
legs=frame_vectors_from_coframe(cf,gi,n)   # e_a^mu

# Frame metric H_ab (constant): l·n=-1, m·mb=+1
H=sp.Matrix(n,n,lambda a,b: simp(sum(cf[a][mu]*legs[b][mu] for mu in range(n))))
print("H =", H.tolist())
Hinv=H.inv()

# Express a 2-form in FRAME components: F_ab = legs[a]^mu legs[b]^nu F_munu
def to_frame(Fco):
    return sp.Matrix(n,n,lambda a,b: simp(sum(legs[a][mu]*legs[b][nu]*Fco[mu,nu]
        for mu in range(n) for nu in range(n))))
def wedge_co(A,B):
    return sp.Matrix(n,n,lambda mu,nu: simp(A[mu]*B[nu]-A[nu]*B[mu]))

# MH self-dual basis translated to PICK legs (0=l,1=n,2=m,3=mb):
#   Z1=-n∧mb,  Z2=-l∧n+m∧mb,  Z3=l∧m
Z1=to_frame(wedge_co([-x for x in cf[1]],cf[3]))
Z2=sp.Matrix(n,n,lambda mu,nu: 0)  # build in frame directly below
# easier: build all in coordinates then push to frame
Z1co=wedge_co(cf[1],cf[3]); Z1co=-Z1co
Z2co=sp.Matrix(n,n,lambda mu,nu: simp(-wedge_co(cf[0],cf[1])[mu,nu]+wedge_co(cf[2],cf[3])[mu,nu]))
Z3co=wedge_co(cf[0],cf[2])
Zco={1:Z1co,2:Z2co,3:Z3co}

# FRAME Levi-Civita with eps_{0123}=eps_val (test ±i and ±1)
from itertools import permutations
def perm_sign(p):
    s=1; p=list(p)
    for a in range(n):
        for b in range(a+1,n):
            if p[a]>p[b]: s=-s
    return s
def frame_hodge(Ff, eps_val):
    # (*F)_ab = ½ eps_ab^{cd} F_cd, indices raised with Hinv, eps_abcd=eps_val·perm_sign
    out=sp.zeros(n,n)
    for a in range(n):
        for b in range(n):
            tot=sp.S.Zero
            for c in range(n):
                for d in range(n):
                    if len({a,b,c,d})<4: continue
                    eps=eps_val*perm_sign((a,b,c,d))
                    for e in range(n):
                        for f in range(n):
                            tot+=eps*Hinv[c,e]*Hinv[d,f]*Ff[e,f]
            out[a,b]=simp(sp.Rational(1,2)*tot)
    return out

for eps_val,lab in [(sp.I,'i'),(-sp.I,'-i'),(sp.Integer(1),'1')]:
    print(f"\n--- eps_0123 = {lab} ---")
    for j in [1,2,3]:
        Zf=to_frame(Zco[j])
        hZ=frame_hodge(Zf,eps_val)
        is_pi=all(simp(hZ[a,b]-sp.I*Zf[a,b])==0 for a in range(n) for b in range(n))
        is_mi=all(simp(hZ[a,b]+sp.I*Zf[a,b])==0 for a in range(n) for b in range(n))
        print(f"  Z{j}: *Z=+iZ? {is_pi}   *Z=-iZ? {is_mi}")
