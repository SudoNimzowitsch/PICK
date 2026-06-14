"""Rung 2: dZ^i -> solve for connection 1-forms sigma_i -> spin coefficients.
MH eq24:  dZ1 = -2 σ2∧Z1 - σ3∧Z2
          dZ2 =  2 σ1∧Z1 - 2 σ3∧Z3
          dZ3 =    σ1∧Z2 + 2 σ2∧Z3
Each σ_i = σ_i,μ dx^μ (4 complex unknowns each, 12 total). dZ^i are 3-forms.
Solve the linear system component-wise."""
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
cf=[list(l),list(nv),list(mv),list(mb)]

def wedge1(A,B):  # 1∧1 -> 2-form components
    return {(mu,nu): simp(A[mu]*B[nu]-A[nu]*B[mu]) for mu in range(n) for nu in range(n) if mu<nu}
def d2(Fdict):  # exterior derivative of 2-form -> 3-form, components (a<b<c)
    out={}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if not(a<b<c): continue
                # (dF)_{abc} = ∂_a F_{bc} - ∂_b F_{ac} + ∂_c F_{ab}
                def F(i,j):
                    if i<j: return Fdict.get((i,j),sp.S.Zero)
                    elif i>j: return -Fdict.get((j,i),sp.S.Zero)
                    return sp.S.Zero
                out[(a,b,c)]=simp(sp.diff(F(b,c),coords[a])-sp.diff(F(a,c),coords[b])+sp.diff(F(a,b),coords[c]))
    return out

# Z^i as 2-forms (coordinate components): Z1=-n∧mb, Z2=-l∧n+m∧mb, Z3=l∧m
def neg(d): return {k:-v for k,v in d.items()}
def addd(d1,d2):
    out=dict(d1)
    for k,v in d2.items(): out[k]=simp(out.get(k,sp.S.Zero)+v)
    return out
Z={}
Z[1]=neg(wedge1(cf[1],cf[3]))
Z[2]=addd(neg(wedge1(cf[0],cf[1])), wedge1(cf[2],cf[3]))
Z[3]=wedge1(cf[0],cf[2])
dZ={i:d2(Z[i]) for i in [1,2,3]}

# unknown sigma_i components: s[i][mu]
s={i:[sp.Symbol(f's{i}_{mu}') for mu in range(n)] for i in [1,2,3]}
def sig_wedge_Z(si, Zj):  # (σ_i ∧ Z_j) as 3-form: σ a 1-form (list), Z_j a 2-form dict
    out={}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if not(a<b<c): continue
                def Zc(i,j):
                    if i<j: return Zj.get((i,j),sp.S.Zero)
                    elif i>j: return -Zj.get((j,i),sp.S.Zero)
                    return sp.S.Zero
                # (σ∧Z)_{abc} = σ_a Z_{bc} - σ_b Z_{ac} + σ_c Z_{ab}
                out[(a,b,c)]=simp(si[a]*Zc(b,c)-si[b]*Zc(a,c)+si[c]*Zc(a,b))
    return out
def scale(d,k): return {key:simp(k*v) for key,v in d.items()}
def add3(*ds):
    out={}
    for d in ds:
        for k,v in d.items(): out[k]=simp(out.get(k,sp.S.Zero)+v)
    return out

# RHS of eq24
rhs={}
rhs[1]=add3(scale(sig_wedge_Z(s[2],Z[1]),-2), scale(sig_wedge_Z(s[3],Z[2]),-1))
rhs[2]=add3(scale(sig_wedge_Z(s[1],Z[1]), 2), scale(sig_wedge_Z(s[3],Z[3]),-2))
rhs[3]=add3(sig_wedge_Z(s[2],Z[3]) and scale(sig_wedge_Z(s[2],Z[3]),2), sig_wedge_Z(s[1],Z[2]))
# fix rhs[3]
rhs[3]=add3(sig_wedge_Z(s[1],Z[2]), scale(sig_wedge_Z(s[2],Z[3]),2))

# Build equations dZ^i - rhs^i = 0 over all 3-form components
eqs=[]
allkeys=set()
for i in [1,2,3]:
    for k in set(list(dZ[i].keys())+list(rhs[i].keys())):
        allkeys.add((i,k))
        eqs.append(simp(dZ[i].get(k,sp.S.Zero)-rhs[i].get(k,sp.S.Zero)))
unknowns=[s[i][mu] for i in [1,2,3] for mu in range(n)]
sol=sp.solve(eqs, unknowns, dict=True)
print("num solutions:", len(sol))
if sol:
    so=sol[0]
    for i in [1,2,3]:
        comps=[simp(so.get(s[i][mu], s[i][mu])) for mu in range(n)]
        print(f"σ{i} =", comps)

print("\n=== spin coefficients via MH eq(20) ===")
# MH eq(20), σ_i expanded in θ^a basis (their labels θ1=n,θ2=l,θ3=-m̃,θ4=-m):
#   σ1 = κθ1+τθ2+σ θ3+ρθ4   = m^a ∇_b l_a dx^b
#   σ2 = εθ1+γθ2+βθ3+αθ4
#   σ3 = πθ1+νθ2+μθ3+λθ4   = n^a ∇_b m̃_a dx^b
# To extract components, express σ_i (a 1-form) back in the θ basis. We have
# σ_i in coordinate components; θ^a are the coframe (cf). Need dual basis: the
# frame VECTORS legs[a]^μ satisfy θ^b(legs[a])=δ. So component of σ_i along θ^a
# is σ_i(legs[a]) = σ_{i,μ} legs[a]^μ. But careful: that gives the coefficient
# of θ^a in σ_i = Σ_a c_a θ^a iff legs are dual to cf. They are (frame vectors).
legs=frame_vectors_from_coframe(cf,gi,n)
so=sol[0]
sig={i:[simp(so.get(s[i][mu],s[i][mu])) for mu in range(n)] for i in [1,2,3]}
def comp_along(si, a):  # coefficient of θ^a in 1-form si
    return simp(sum(si[mu]*legs[a][mu] for mu in range(n)))
# PICK leg order 0=l,1=n,2=m,3=mb. MH θ1=n(idx1),θ2=l(idx0),θ3=-mb(idx3),θ4=-m(idx2)
# So coefficient "along θ1"=along n=comp_along(...,1); θ2=along l=idx0;
# θ3 = along -mb -> -comp_along(...,3); θ4 = along -m -> -comp_along(...,2)
def decomp(i):
    c_th1=comp_along(sig[i],1)   # n
    c_th2=comp_along(sig[i],0)   # l
    c_th3=-comp_along(sig[i],3)  # -mb
    c_th4=-comp_along(sig[i],2)  # -m
    return c_th1,c_th2,c_th3,c_th4
k_,tau,sig_,rho = decomp(1)   # σ1 -> κ,τ,σ,ρ
eps,gam,beta,alf = decomp(2)  # σ2 -> ε,γ,β,α
pi_,nu_,mu_,lam = decomp(3)   # σ3 -> π,ν,μ,λ
names=[('κ',k_),('σ',sig_),('λ',lam),('ν',nu_),('ρ',rho),('μ',mu_),
       ('τ',tau),('π',pi_),('ε',eps),('γ',gam),('β',beta),('α',alf)]
for nm,v in names:
    print(f"  {nm} = {v}")
print("\nType-D check (expect κ=σ=λ=ν=0):",
      all(simp(x)==0 for x in [k_,sig_,lam,nu_]))

print("\n=== retry with θ3/θ4 (m/mb) assignment swapped ===")
# MH θ3=-m̃=-mb? vs θ4=-m. Try θ3 along -m (idx2), θ4 along -mb (idx3).
def decomp2(i):
    c_th1=comp_along(sig[i],1)   # n
    c_th2=comp_along(sig[i],0)   # l
    c_th3=-comp_along(sig[i],2)  # -m  (idx2)  <-- swapped
    c_th4=-comp_along(sig[i],3)  # -mb (idx3)  <-- swapped
    return c_th1,c_th2,c_th3,c_th4
k_,tau,sig_,rho = decomp2(1)
eps,gam,beta,alf = decomp2(2)
pi_,nu_,mu_,lam = decomp2(3)
names=[('κ',k_),('σ',sig_),('λ',lam),('ν',nu_),('ρ',rho),('μ',mu_),
       ('τ',tau),('π',pi_),('ε',eps),('γ',gam),('β',beta),('α',alf)]
for nm,v in names: print(f"  {nm} = {simp(v)}")
print("Type-D check (κ=σ=λ=ν=0):", all(simp(x)==0 for x in [k_,sig_,lam,nu_]))

print("\n=== simplify key spin coefficients (assume r>2M) ===")
M_,r_=sp.symbols('M r',positive=True)
def clean(e):
    # substitute sqrt(-1/(2M-r)) = 1/sqrt(r-2M) for r>2M
    e=e.subs(sp.sqrt(-1/(2*M-r)), 1/sp.sqrt(r-2*M))
    return sp.simplify(e)
rho_c=clean(rho); mu_c=clean(mu_); gam_c=clean(gam); eps_c=clean(eps)
print(f"  ρ = {rho_c}")
print(f"  μ = {mu_c}")
print(f"  ε = {eps_c}")
print(f"  γ = {gam_c}")
# Known Schwarzschild (standard symmetric tetrad, A=1-2M/r):
# ρ = -1/r·√(A/2)... tetrad-normalization dependent. The robust check is Ψ2.
# Ψ2 from spin coeffs (NP): for type D vacuum, Ψ2 = ... but cleanest is to
# proceed to curvature. Print √(A/2)/r and √(A/2)·M/r² for comparison:
A=1-2*M/r
print(f"\n  compare ρ to -√(A/2)/r·(?): √(A/2)/r = {sp.simplify(sp.sqrt(A/2)/r)}")
print(f"  ρ·(-1) numeric at M=1,r=10: ρ={float(rho_c.subs({M:1,r:10})):.5f}  √(A/2)/r={float((sp.sqrt(A/2)/r).subs({M:1,r:10})):.5f}")

print("\n=== simplify (fixed symbol scope) ===")
Msym, rsym = sp.symbols('M r', positive=True)
def clean2(e):
    e=e.subs(sp.sqrt(-1/(2*Msym-rsym)), 1/sp.sqrt(rsym-2*Msym))
    return sp.simplify(e)
rho_c=clean2(rho); mu_c=clean2(mu_); eps_c=clean2(eps); gam_c=clean2(gam)
print(f"  ρ = {rho_c}")
print(f"  μ = {mu_c}")
print(f"  ε = {eps_c}")
print(f"  γ = {gam_c}")
A=1-2*Msym/rsym
print(f"  √(A/2)/r = {sp.simplify(sp.sqrt(A/2)/rsym)}")
print(f"  numeric M=1,r=10:  ρ={float(rho_c.subs({Msym:1,rsym:10})):.6f}  √(A/2)/r={float((sp.sqrt(A/2)/rsym).subs({Msym:1,rsym:10})):.6f}")
print(f"  ρ==√(A/2)/r ? {sp.simplify(rho_c - sp.sqrt(A/2)/rsym)==0}")
