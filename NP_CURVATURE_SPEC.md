# Frame-curvature + spinor-derivative rewrite — implementation spec (v2)

Supersedes v1. v1 mislabeled this as "solve the NP equations." Correct framing,
confirmed against CLASSI source + MacCallum & Åman 1986 (Class. Quantum Grav. 3
1133, PDF in uploads):

  ORDER 0:  Cartan structure equations in a NULL frame (Ricci rotation
            coefficients), then project Weyl frame components to Ψ_k — the
            projection PICK already has and which is CORRECT.
  ORDER ≥1: spinor covariant derivative acting on the symmetric curvature
            spinors, organized as MacCallum-Åman's minimal set V^nR.

## Why (the verdict that forced this)

Overnight 2026-06-13: ki (Kinnersley NUT) consumed ~28 GB (12 RAM + 16 swap)
over 8.5 h without clearing stage 3 (Ricci & Weyl), then OOM-killed. Correct
(reported vacuum) but infeasible. The blowup is the dense COORDINATE Riemann
tensor: entries are coordinate derivatives of Christoffels, degree-8 rationals
in 5 vars on ki, which `cancel` chokes on. memlog confirmed RAM+swap pegged.

## What CLASSI actually does (source-confirmed, /tmp/sheep/.../clasrc/)

Ψ_k are PROJECTED from the frame Weyl tensor — NOT solved from NP equations.
`psiphi.shp` BUPSINUL, null tetrad (PICK leg order 0=l,1=n,2=m,3=mb):
    Ψ0=WEYL(2,0,0,2)  Ψ1=WEYL(1,0,0,2)  Ψ2=WEYL(0,2,1,3)
    Ψ3=WEYL(3,1,2,3)  Ψ4=WEYL(3,1,1,3)
(SHMINIF on NEGRIESIGN/GETSPACELIKE = a sign tied to metric signature
convention — pin against PICK's existing weyl_spinor_from_coframe, which is
already correct for all 14 metrics, so the NEW path must reproduce it.)

The frame Weyl tensor (weyltf.shp WEYLF) = RIEF + (C1,C2 trace terms with
RICF,RSCLF). The win is RIEF (frame Riemann), built NOT from coordinate
Christoffels but from the FRAME CONNECTION:
    frame.shp line 180:  RIEF = Riesign*(GAM·Z - GAM·Z + H(GAM·GAM - GAM·GAM - GAM·C))
    frame.shp line 342:  GAMF (Ricci rotation coeffs) from LIE (tetrad commutators)
i.e. Cartan: differentiate the 4 legs, antisymmetrize into commutators C^a_{bc},
lower to rotation coefficients, build frame Riemann algebraically. All objects
carry FRAME indices (small), never the dense coordinate rank-4 array.

The spinor machinery (SPCURV/NEWPEN in inewpn.shp, spcurv.shp) is a FALLBACK
("if RIEF not made and WEYLTF not loaded") and for NP worked-examples — NOT the
classification path for order 0. Do not port it for order 0.

## Order ≥1: this is where spinors are mandatory, not optional

MacCallum-Åman 1986 §2 defines the minimal set V^nR (what makes 27,962,020
components collapse to 8,690 at n=10):
  (i)   totally symmetrised spinor nth derivs of Weyl spinor Ψ_HKLM
  (ii)  totally symmetrised spinor nth derivs of Ricci spinor Φ
  (iii) totally symmetrised spinor nth derivs of scalar curvature Λ
  (iv)  n≥1: symmetrised (n-1)th deriv of the Bianchi curl Ξ_DEF W' (eq 1.2 LHS)
  (v)   n≥2: d'Alembertian □ of all quantities in V^{n-2}R

The reductions that make this minimal are the Ricci identity (eq 2.1) and its
contraction-corollary (2.2), plus the Bianchi identities (1.2, 1.3). These are
identities of the SPINOR covariant derivative ∇_{AA'} on symmetric spinors;
they are what let you DISCARD all the non-symmetric/contracted derivative
components instead of computing them. The tensorial ∇C path (PICK's current
compute_psid_direct: ∂C - Γ-contractions in coordinate indices, then project)
can get the same NUMBERS but must rediscover these reductions; the spinor path
has them in the index structure. This is the real reason CLASSI is spinorial
end to end.

CLASSI's families (equspi.shp lines 17-23) map 1:1 onto V^nR:
    DPSI = symmetrised ∇Ψ  (type i)     XI  = Bianchi curl Ξ      (type iv)
    DPHI = symmetrised ∇Φ  (type ii)    APSI/APHI = □Ψ, □Φ        (type v)
    DLAMBDA = ∇Λ           (type iii)
  prefix D = symmetrised spinor deriv,  prefix A = d'Alembertian (box).
Vacuum: only type (i) survives → (n+1)(n+5) quantities at order n (Penrose 1960).
Conformally flat: types (i),(iv) vanish, (iii) discardable for n>1.

L-projection notation (p.1138): ∇Ψ_{20'} = ∇^{X'}_{(A}Ψ_{BCDE)} o^A o^B ι^C ι^D ō_{X'},
labelled by (# of ι among unprimed = 2, # of ῑ/ō among primed = 0). PICK's
PSID component indexing must map to this (k, d) labeling.

## Conventions to PIN WITH UNIT TESTS before wiring in (each = a silent Ψ flip)

ORDER 0 (frame curvature):
 - tetrad commutator sign: C^c_{ab} from [e_a,e_b]=C^c_{ab}e_c, and the leg-
   raising metric (PICK: l·n=-1, m·mb=+1, g=-(l⊗n+n⊗l-m⊗mb-mb⊗m)).
 - rotation-coeff index order in GAMF (which slots are antisymmetric).
 - the Riesign/NEGRIESIGN signature sign in WEYLF and in the Ψ projection.
 - Ψ_k leg map above must reproduce existing weyl_spinor_from_coframe exactly.

ORDER ≥1 (spinor deriv):
 - ε_{AB} convention (ε_{01}=+1 vs -1) and raising order ψ^A=ε^{AB}ψ_B.
 - spinor covariant derivative ∇_{AA'} acting on a symmetric spinor: the spin-
   coefficient connection terms (κ,σ,ρ,τ,… and primed) and their signs. Pin
   against Schwarzschild ∇Ψ (we have PSID for it from the current path).
 - symmetrisation + L-projection (k,d) labeling vs PICK's PSID component order.

## Test gate (in order; each must pass before the next)

1. Schwarzschild Ψ2 = -M/r^3 only (type D). Cleanest order-0 pin.
2. Reissner-Nordström: Ψ2 AND Φ11 — pins Φ/Λ branch.
3. Type-N (wavez constructed): Ψ4-only raw frame — catches leg-order flips.
4. Order-1: Schwarzschild PSID must match current compute_psid_direct output
   exactly (we have it). This pins the spinor-derivative conventions.
5. FULL 14-metric battery (minkow schwar desitt renord einuni friedc try2
   szek1 godel berob3 vaidya wavez wavezk + sage suite) — every t-seq, H-seq,
   r EXACT before the new path replaces the old.

## Architecture / scoping decision (decide FIRST, it sizes everything)

frame_curvature_from_coframe(coframe, coords, simp) -> (PSI, PHI, LAMBD),
drop-in for weyl_ten + weyl_spinor_from_coframe + ricci_spinor_from_coframe.
Keep OLD path behind a flag; run BOTH on the battery, assert equal during
bring-up (cross-validation is free correctness insurance).

ORDER ≥1 — two options:
  (a) FULL: port the spinor covariant derivative + V^nR minimal set. Maximal
      correctness/coherence with CLASSI; larger; the "right" end state.
  (b) BRIDGE: frame curvature for order 0 only; keep existing tensorial
      compute_psid_direct for ∇C, fed by the frame-derived PSI/PHI. LOWER RISK
      for first landing; STILL removes the order-0 dense blowup that kills ki.
RECOMMENDATION: (b) first — get ki/kiva/petrme passing on the strength of the
order-0 fix alone (their blowup is order 0), land it, THEN consider (a) as a
separate correctness/coherence improvement with its own test gate. Do NOT
bundle (a) into the first landing.

## State at spec v2 (2026-06-13)

- 20 commits on main (HEAD c24678f), 14 metrics passing, tree clean.
- ki/kiva need NO SUL (their .nud NEWSULs are CLASSI-internal repr tricks);
  only blocker is the order-0 dense blowup this fixes. Coframes captured in
  run_ki.py / run_kiva.py (outputs); ki confirmed vacuum, infeasible on old path.
- Reference: MacCallum & Åman 1986 CQG 3 1133 (uploads) = the order-≥1 spec.

## PINNED CONVENTIONS (validated on Schwarzschild, 2026-06-13)

Prototype /tmp/fr2.py established the ground truth. Confirmed pieces:

- Frame metric H_ab from coframe·legs: l·n=-1, m·mb=+1, constant. ✓
- Commutators C^c_{ab} = θ^c_μ [e_a,e_b]^μ where
  [e_a,e_b]^μ = e_a^ν ∂_ν e_b^μ - e_b^ν ∂_ν e_a^μ. Antisymmetric in (a,b). ✓
- Rotation coeffs GAM_{abc} = ½(C_{cab}+C_{bac}-C_{abc}), C_{abc}=H_{ad}C^d_{bc};
  antisymmetric in (a,b). ✓ (matches frame.shp line 148)
- **GROUND TRUTH projection** (use this to validate any frame-Riemann formula):
    R_lowered(a,b,c,d) = g_{ae} R^e_{bcd}     [PICK riemann() returns R^r_{smv}, MIXED]
    frameR(a,b,c,d) = legs[a]^μ legs[b]^ν legs[c]^ρ legs[d]^σ R_lowered(μνρσ)
      where legs = frame_vectors_from_coframe (RAISED, e_a^μ).
    Schwarzschild slots (_PSI_LEGS): Ψ2=frameR(0,2,3,1)=-M/r³, all others 0. ✓ EXACT.
- **Simplifier MUST be trig-aware with sin(θ)>0**: plain cancel leaves
  Abs(sin)/sign(sin)/DiracDelta(sin) spuria that fake a nonzero frame Ricci.
  Use refine(simplify(e), Q.positive(sin(theta))) on any expr with trig/Abs/sqrt.

## CRITICAL ARCHITECTURE DECISION (resolved)

Do NOT re-derive the Ψ projection. The existing weyl_ten + weyl_spinor_from_coframe
is correct for all 14 metrics and encodes the index convention (raise all of C,
contract with COVARIANT coframe forms). The new frame path should produce the
SAME fully-covariant R_{abcd} that weyl_ten consumes, then reuse weyl_ten +
weyl_spinor_from_coframe UNCHANGED. New code = "tetrad commutators → frame
connection → frame Riemann (fully covariant)" validated against frameR_truth
above. The from-scratch Cartan frame-Riemann formula (frame.shp line 180) had a
sign/index error in the prototype (produced nonzero frame Ricci on vacuum
Schwarzschild) — must be debugged component-by-component against frameR_truth
BEFORE wiring in. The commutator/rotation-coeff stages are already correct.

NB the whole point is to AVOID the coordinate Riemann (the dense object that
OOMs ki). frameR_truth above USES the coordinate Riemann — it is the validation
oracle ONLY, not the implementation. The implementation must build frame Riemann
from GAM (rotation coeffs) + their frame-directional derivatives, which stay
small. Validate the GAM-based formula against frameR_truth on schwar/renord, THEN
confirm it stays small (no coordinate Riemann) on ki.

## Convention cross-check: McIntosh & Hickman 1985 (GRG 17 111, uploads)

A clean published source for the null-tetrad Cartan + NP conventions. Use its
STRUCTURE, not its signs verbatim — its signature is OPPOSITE to PICK's:
  M&H eq(2,6): θ1·θ2=+1, θ3·θ4=-1, signature (+,-,-,-), l·n=+1, m·mb=-1.
  PICK:        l·n=-1, m·mb=+1, signature (-,+,+,+).
So any explicit Ψ/Φ projection sign in M&H flips relative to PICK. This is the
same NEGRIESIGN/GETSPACELIKE signature-sign that BUPSINUL carries in CLASSI.

CONFIRMED CONSISTENT WITH PICK:
- Riemann→Weyl decomposition, M&H eq(28):
    R_abcd = C_abcd - ½(g_ac R_bd - g_ad R_bc + g_bd R_ac - g_bc R_ad)
             - (R/6)(g_ad g_bc - g_ac g_bd)
  matches PICK weyl_ten exactly (f1=1/2, f2=R/6, same +(g_ac g_bd - g_ad g_bc)
  sign). => keep reusing weyl_ten UNCHANGED, as already decided.

USEFUL FOR THE FRAME-RIEMANN DEBUG (the current blocker):
- M&H eq(27): Cartan 2nd structure eq in tetrad components,
    Θ^a_b = -½ R^a_bcd θ^cd = dΓ^a_b + Γ^a_c ∧ Γ^c_b,   Γ_(ab)=0.
  This is the form to validate my frame Riemann against (it produced spurious
  nonzero frame Ricci on vacuum Schwarzschild — sign/index error in the
  GAM·GAM / GAM·C terms). Note Γ_(ab)=0 (connection antisymmetric in ab) is a
  hard check on the rotation coeffs — PICK's GAM already satisfies it.
- M&H eq(40): spin coefficients = named components of spinor connection Γ_XYZŻ:
    κ=Γ_0000', τ=Γ_0010', σ=Γ_0001', ρ=Γ_0001'... (see paper for full map)
    [primed/unprimed dyad-index labelling; maps to PICK PSID (k,d) indices]

INDEPENDENT CONFIRMATION OF CONVENTIONS WE PINNED EMPIRICALLY THIS SESSION:
- M&H eq(55): null rotation about l (param K):
    Ψ0'=Ψ0, Ψ1'=Ψ1+KΨ0, Ψ2'=Ψ2+2KΨ1+K²Ψ0,
    Ψ3'=Ψ3+3KΨ2+3K²Ψ1+K³Ψ0, Ψ4'=Ψ4+4KΨ3+6K²Ψ2+4K³Ψ1+K⁴Ψ0.
  Exactly PICK's nl(E) root-action algebra. Confirms the DYTAUT standardisers.
- M&H eq(60): boost-spin (param M): Ψ_a' = M^{2(2-a)} Ψ_a, so Ψ0'=M⁴Ψ0. With
  the boost A := M², Ψ0 → A²Ψ0 — INDEPENDENT confirmation of the B2 type-N
  normalisation (boost absorbs |Ψ0| as A²) pinned empirically on wavez.
- M&H §11 eq(81,82): vacuum type D ⟺ Ψ0=Ψ1=Ψ3=Ψ4=0 with l,n both repeated
  PNDs (κ=σ=ν=λ=0). This is the ki/kiva target structure (00D). The frame
  path must reproduce it.

KEY EFFICIENCY POINT (M&H p.117, after eq 25): spin coefficients can be read
directly off the dZ^j (self-dual bivector) equations — 12 complex coefficients
from 3 bivector eqs — instead of the 24-real-coefficient commutator route. Not
required for our frame-Riemann path (we get curvature from GAM directly), but
relevant if we later want spin coefficients explicitly for the order-≥1 spinor
derivative (option (a)).

## Frame-Riemann debug state (2026-06-13, prototype frame_curv_prototype.py)

Validated correct: frame metric H, commutators C^c_ab, rotation coeffs GAM_abc
(antisymmetric in ab per M&H Γ_(ab)=0). Oracle (frame_oracle_prototype.py) gives
Schwarzschild Ψ2=-M/r³ exact.

BUG: my frame-Riemann "variantA" disagrees with oracle by a clean ±(4M-r)/2r³
on every tested slot:
  R[0202]: oracle 0,     variantA (-4M+r)/2r³
  R[0231]: oracle -M/r³, variantA (2M-r)/2r³
  R[1313]: oracle 0,     variantA (-4M+r)/2r³
The r-term (2-sphere curvature) and the factor-of-2 on M localize the error to
the directional-derivative term t1 (GAM_abd;c - GAM_abc;d) and/or its
interaction with the connection-commutator term t3 (-GAM_abe C^e_cd). Next step:
rebuild the frame Riemann strictly from M&H eq(27) Θ^a_b=dΓ^a_b+Γ^a_c∧Γ^c_b in
tetrad components and re-validate against oracle slot-by-slot. Do NOT guess more
variants — derive t1,t3 from eq(27) directly. variantA's t2 (GAM·GAM) is likely
correct; the dGAM piece is where the frame (non-coordinate) directional
derivative needs the extra commutator term that eq(27)'s exterior derivative
supplies.

## PICK metric/tetrad convention — STATED EXPLICITLY (verified empirically 2026-06-13)

Empirical check (minkow, schwar, desitt): l·n=-1, m·mb=+1, all legs null,
metric reconstructs exactly from g = -(l⊗n+n⊗l - m⊗mb - mb⊗m). Confirmed.

The overall MINUS sign matters and is easy to drop mentally. PICK is:

    g = -(l⊗n + n⊗l - m⊗mb - mb⊗m)      ds² = -2 ln + 2 m·mb   (symmetric)
    l·n = -1,  m·mb = +1,  signature (-,+,+,+)   [MTW/Wald]

Standard reality assignment (what all PICK metrics use, e.g. Schwarzschild IZUD):
    l, n   = REAL null one-forms          (the timelike-containing 2-plane)
    m, mb  = COMPLEX-CONJUGATE pair        (the transverse spatial 2-plane)

Minkowski worked example IN PICK's convention (do not drop the leading minus):
    l = dt+dz, n = dt-dz   (real)    -> -2ln = -dt²+dz²
    m = dx+i dy, mb = dx-i dy (conj) -> +2 m·mb = dx²+dy²
    ds² = -dt²+dx²+dy²+dz²   ✓ standard (-,+,+,+), standard l,n real / m,mb conj.

PITFALL (raised by G, 2026-06-13): if one writes the convention as ds² = +ln - mm̄
(the (+,-,-,-) form, WITHOUT PICK's overall minus), then to recover standard
Minkowski one is forced into the BACKWARDS assignment l,n = conjugate pair
(dx±i dy), m,mb = real pair (dt±dz). That apparent paradox is an artifact of
the dropped overall sign; PICK's -(...) undoes it and the standard assignment
holds. The l,n-real / m,mb-conjugate naming is NOT swapped in PICK.

DECISION: keep PICK in (-,+,+,+). Do NOT flip to M&H (+,-,-,-). Rationale:
(1) verified self-consistent across the 14-metric battery; (2) classification
outputs (Petrov type, isotropy, t-seq) are signature-independent, so a flip
changes only internal bookkeeping at real risk; (3) M&H structure is translated
at the boundary (signature sign on the Ψ projection), already documented above.
Revisit ONLY at the option-(a) full-spinor-derivative stage IF published
∇-formulae in (+,-,-,-) make boundary translation a persistent error source —
and then only as an isolated, battery-gated commit asserting identical outputs.

## Frame-Riemann debug — progress 2026-06-13 (session 2), NOT yet resolved

Ruled OUT as the bug:
- Curvature formula arrangement: three independent forms (directional-deriv;
  exterior-deriv with -ω([e_c,e_d]); Wald frame form) ALL give the identical
  discrepancy ±(4M-r)/(2r³) on Schwarzschild. So the bug is NOT in the
  curvature assembly.
- Connection metric-compatibility & torsion: commutator-based GAM_{abc} is
  antisymmetric in (ab) [0 violations] AND torsion-free [reproduces C^a_{bc},
  0 violations]. So GAM looks like a valid connection.

KEY FINDING (the lead to chase): I built an independent "Christoffel-based"
frame connection Gu_indep(a,b,c) = θ^a_μ (e_c^ν ∂_ν e_b^μ + Γ^μ_{ρν} e_b^ρ e_c^ν)
as a cross-check. Its all-down form Γ_true_{abc}=H_{ae}Gu_indep(e,b,c) is NOT
antisymmetric in (a,b) (28 violations, e.g. Γ_true_{022}+Γ_true_{202}≠0). A
metric connection in a CONSTANT frame metric MUST be antisym in (ab). Therefore
Gu_indep is itself mis-indexed — its (b,c) slot assignment does not match the
true ∇_{e_c} e_b. Also: Γ_true_{023}≠0 while ALL six commutator orderings of
(0,2,3) vanish, which is impossible for a constant-frame connection built from
commutators — further proof Gu_indep is wrong, not GAM.

So the discrepancy is a MISMATCH between (a) the commutator-based GAM index
convention and (b) the index wiring of the curvature formula / oracle leg
contraction — NOT a missing physical term.

NEXT MOVE (do NOT test more curvature variants):
1. Build Γ^a_{bc} := component of ∇_{e_c} e_b along e^a, computed cleanly:
   ∇_{e_c} e_b has coordinate components (e_c^ν ∂_ν e_b^μ + Γ^μ_{ρν} e_b^ρ e_c^ν);
   then Γ^a_{bc} = θ^a_μ (that). CHECK it is antisym-in-(ab) all-down (it must
   be). The earlier Gu_indep failed this — find the index slot error (likely
   θ^a_μ vs which leg, or b/c swap) until antisymmetry holds.
2. With that verified Γ, assemble R^a_{bcd}=e_c Γ^a_{bd}-e_d Γ^a_{bc}
   +Γ^a_{ec}Γ^e_{bd}-Γ^a_{ed}Γ^e_{bc}-Γ^a_{be}C^e_{cd}; validate vs oracle
   (frame_oracle_prototype.py, Ψ2=-M/r³ exact) slot-by-slot.
3. Then confirm GAM (commutator) and the verified Γ agree once indices align —
   that tells us the exact GAM→Γ index map to use in the production function.
Prototype scratch: frame_curv_prototype.py (variants), /tmp/fr8.py (the index
diagnostic that found the antisymmetry failure).

## RECOMMENDED APPROACH (revised 2026-06-13 sess.2): self-dual bivector route

G's insight: instead of rotation coefficients (whose [ab]-antisymmetric pair
bookkeeping is the source of the order-0 frame-Riemann index bug), use the
McIntosh-Hickman self-dual BIVECTOR formalism. It packages the [ab] pair into a
single index i∈{1,2,3}, removing exactly that ambiguity, AND yields the 12
complex NP spin coefficients directly (needed for order ≥1 anyway). MH p.117
explicitly: read spin coeffs off dZ^i (12 complex) vs commutators (24 real,
"unnecessary work").

Chain (MH §4-5, eqs 11,17,20,21,24,33,34):
  1. Self-dual 2-forms Z^i (eq 11, MH labels θ1=n,θ2=l,θ3=-m̃,θ4=-m):
       Z1=θ13, Z2=θ12-θ34, Z3=θ42.  Translate to PICK legs (0=l,1=n,2=m,3=mb).
  2. dZ^i = σ^i_j ∧ Z^j (eq 17/24): solve for connection bivector 1-forms σ_i.
       eq24: dZ1=-2σ2∧Z1-σ3∧Z2; dZ2=2σ1∧Z1-2σ3∧Z3; dZ3=σ1∧Z2+2σ2∧Z3.
  3. spin coeffs = components of σ_i in θ basis (eq 20-21).
  4. curvature Σ_i=dσ_i+(σ∧σ) (eq 34): Σ1=dσ1+2σ1∧σ2, Σ2=dσ2+σ1∧σ3,
       Σ3=dσ3+2σ2∧σ3.
  5. Σ_i=(C_ij+(R/6)γ_ij)Z^j + E_ij Z̃^j (eq 33): read Ψ matrix C_ij (eq 29:
       C=[[Ψ0,Ψ1,Ψ2],[Ψ1,Ψ2,Ψ3],[Ψ2,Ψ3,Ψ4]]) and Ricci E_ij. γ_ij = eq 19.

TRANSLATION LAYERS (the crux, per G): MH is (+,-,-,-) AND relabeled tetrad
(θ1=n etc). PICK is (-,+,+,+), legs 0=l,1=n,2=m,3=mb. Every eq needs signature
sign + index relabel. Validation gate UNCHANGED: reproduce PICK's existing Ψ on
the 14-metric battery; Schwarzschild Ψ2=-M/r³ first pin.

BLOCKER HIT (prototype /tmp/biv.py): the self-dual basis must be PINNED FIRST by
verifying *Z=iZ (Hodge star eigenform) BEFORE computing dZ. Current Hodge-star
impl gives none of l∧m, n∧mb, l∧n-m∧mb as clean ±i eigenforms — the sqrt(-g)
orientation/normalization is inconsistent with the leg sqrt(-2M+r) factors
(e.g. *(l∧n) came out 0, *(l∧m)/(l∧m) didn't reduce to ±i). FIX THE HODGE STAR
FIRST: build ε_{μνρσ}=sqrt(-g)·sign(perm), *F_{μν}=½ε_{μν}^{ρσ}F_{ρσ}, and
confirm *Z=iZ on the three basis self-dual 2-forms for Schwarzschild. Only then
proceed to dZ→σ→Σ→Ψ. Discipline ladder: self-duality ✓ → dZ/σ ✓ → spin coeffs
sanity (Schwarzschild ρ,μ,γ; κ=σ=λ=ν=0 type D) → Ψ2=-M/r³ → battery.

This bivector route SUPERSEDES the rotation-coefficient route for the build. The
rotation-coeff findings above remain useful cross-checks (GAM is metric-compat &
torsion-free; the oracle frameR_truth gives Ψ2=-M/r³ exact and is the validation
target for either route).

## RESOLVED: self-dual basis pinned + left/right (conjugation) design (sess.2)

### Hodge bug fixed (prototype bivector_prototype_v2.py = /tmp/biv2.py)
The coordinate Hodge star with real sqrt(-g) was WRONG: it used a real
orientation and lost the factor of i intrinsic to a complex null tetrad. The
complex transformation orthonormal→(l,n,m,mb) has Jacobian with (m,mb)-block
det = -i, so the FRAME Levi-Civita component eps_{l n m mb} = ±i, NOT 1. This
is the same "dual acts in the flat frame η, not curved g" principle PICK found
in the Kleinian Weyl work.

PINNED (Schwarzschild, exact): with the FRAME Hodge star
  (*F)_ab = ½ eps_ab^{cd} F_cd,  indices raised with H^{ab} (frame metric),
  eps_{0123}=eps_{l,n,m,mb}= i,
the MH self-dual basis (translated to PICK legs 0=l,1=n,2=m,3=mb)
  Z1 = -n∧mb,   Z2 = -l∧n + m∧mb,   Z3 = l∧m
are ALL exact eigenforms *Z = +iZ.  (eps=-i gives the anti-self-dual half
*Z=-iZ; eps=+1 gives nothing — that was the original bug.)
Frame metric H = [[0,-1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,1,0]] (l·n=-1,m·mb=+1).

Next rung: dZ^i (exterior deriv) → solve dZ^i = σ^i_j∧Z^j (MH eq24) for the
connection σ_i → spin coeffs (eq20-21) → curvature Σ_i=dσ_i+σ∧σ (eq34) →
C_ij (Ψ) via eq33. Validate Ψ2=-M/r³.

### LEFT/RIGHT design (G's point): when to conjugate vs compute both halves
The right-half (anti-self-dual, tilded) quantities σ̃_i, Ψ̃ relate to the left
half by COMPLEX CONJUGATION only in real Lorentzian signature. Reality structure:
  - Real Lorentzian: l,n real, mb=conj(m). σ̃=conj(σ), Ψ̃=conj(Ψ). LEFT HALF ONLY.
  - Kleinian/split (2,2): real null tetrad, m,mb REAL & independent.
    SL(2,R)×SL(2,R); halves independent; Ψ,Ψ̃ may have DIFFERENT Petrov types
    (MH "type II⊗N"). COMPUTE BOTH HALVES.
  - Euclidean (4,0): SU(2)×SU(2); independent. COMPUTE BOTH.
  - Complex: fully independent. COMPUTE BOTH.

RUNTIME CONDITIONAL (keys on the coframe itself, real coords; no signature flag
needed — reads reality structure off the legs):
  take conjugation fast-path  IFF  cf[3]==conj(cf[2]) componentwise AND
  cf[0],cf[1] real.  Else build the anti-self-dual basis Z̃^i (eps=-i half)
  and solve dZ̃^i=σ̃^i_j∧Z̃^j independently.
Conservative in the safe direction: only conjugates when provably valid;
otherwise computes both (correct, slower). This makes the implementation correct
for PICK's Kleinian (2,2) metrics (es1111, the kii* family) and any complexified
input, while staying fast on the Lorentzian battery.

Implementation sketch:
  Z = selfdual_basis(cf, legs, H)            # eps=+i
  sigma = solve_connection(Z, dZ)            # left
  C, Eleft = weyl_ricci_from_curvature(dsigma+sigma_wedge, Z)
  if cf[3]==conj(cf[2]) and real(cf[0],cf[1]):
      C_tilde = conj(C)                       # FAST PATH (Lorentzian)
  else:
      Ztil = antiselfdual_basis(cf,legs,H)   # eps=-i
      sigma_til = solve_connection(Ztil, dZtil)
      C_tilde, Eright = weyl_ricci_from_curvature(dsigma_til+..., Ztil)
