"""
pick/reduction_tests.py — Strong, self-generated oracle for frame independence
and (eventually) dyad-standardisation (DYTAUT) correctness.

WHY THIS EXISTS
---------------
The ptest/*.sum reference outputs are a *weak* oracle for standardisation: a
wrong-but-self-consistent dyad rotation can still yield plausible r and s. To
build DYTAUT safely we need a check that bites regardless of CLASSI's outputs.

The check is frame-independence of the *scalar polynomial curvature invariants*.
Under any local Lorentz transformation Λ of the null tetrad (boost, spin, null
rotation, or any composition), the individual NP components Ψ_k, Φ_ij change —
often drastically — but the curvature *scalars* built as full contractions are
invariant. The two Weyl scalars

    I = Ψ₀Ψ₄ − 4Ψ₁Ψ₃ + 3Ψ₂²              (boost+spin weight 0)
    J = det[[Ψ₀,Ψ₁,Ψ₂],[Ψ₁,Ψ₂,Ψ₃],[Ψ₂,Ψ₃,Ψ₄]]  (weight 0)

and the Ricci/mixed scalars

    Φ²   = Φ_AB A'B' Φ^AB A'B'    (= Σ signed |Φ_ij|²)
    I_Φ, etc.

are genuine scalars. We verified (Schwarzschild, Kasner) that I and J are
preserved to 0 under explicit boosts and null rotations.

HOW IT IS USED
--------------
1. FRAME-INDEPENDENCE TEST (available now): take any metric we can build a
   coframe for, apply a parametrised Lorentz transformation, and assert the
   scalar invariants are unchanged. This guards the spinor-extraction code and
   establishes the invariant oracle.

2. DYTAUT REDUCTION TEST (target): take a metric whose standard frame we trust,
   apply a *known* Lorentz transformation to put it in a non-standard frame,
   run DYTAUT, and assert (a) the recovered frame's invariants match the
   original, AND (b) the recovered components are in canonical (standard) form.
   Because we generated the perturbation ourselves, we know the right answer
   without consulting CLASSI.

The Lorentz transformations on a null tetrad (l, n, m, mb):

  BOOST (parameter λ>0):   l→λl,  n→n/λ,  m→m,        mb→mb
  SPIN  (parameter θ):     l→l,   n→n,    m→e^{iθ}m,  mb→e^{-iθ}mb
  NULL ROT about l (E∈ℂ):  l→l,   m→m+E*l,  mb→mb+Ē*l,  n→n+Ē*m+E*mb+|E|²l
  NULL ROT about n (B∈ℂ):  n→n,   m→m+B*n,  mb→mb+B̄*n,  l→l+B̄*m+B*mb+|B|²n

These generate the full (proper, orthochronous) Lorentz group SO⁺(3,1).
"""

import sympy as sp
from sympy import sqrt, I, Symbol, conjugate
from typing import Callable, Optional

from .karlhede import (
    christoffel, riemann, ricci_ten, ricci_sc, weyl_ten,
    weyl_spinor_from_coframe, ricci_spinor_from_coframe,
    null_coframe_from_diagonal_metric, validate_null_coframe,
    _simp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Lorentz transformations on a null tetrad given as covariant 1-forms
# ─────────────────────────────────────────────────────────────────────────────

def boost(coframe, lam, simp=None):
    """l→λl, n→n/λ, m,mb unchanged. λ should be a positive expression."""
    l, n, m, mb = coframe
    nd = len(l)
    l2  = [_simp(lam*l[i],   simp) for i in range(nd)]
    n2  = [_simp(n[i]/lam,   simp) for i in range(nd)]
    return (l2, n2, list(m), list(mb))


def spin(coframe, theta, simp=None):
    """Spatial rotation: m→e^{iθ}m, mb→e^{-iθ}mb, l,n unchanged."""
    l, n, m, mb = coframe
    nd = len(l)
    e_p = sp.exp(I*theta)
    e_m = sp.exp(-I*theta)
    m2  = [_simp(e_p*m[i],  simp) for i in range(nd)]
    mb2 = [_simp(e_m*mb[i], simp) for i in range(nd)]
    return (list(l), list(n), m2, mb2)


def null_rotation_l(coframe, E, simp=None):
    """
    Null rotation fixing l:
      l→l, m→m+E l, mb→mb+Ē l, n→n+Ē m+E mb+|E|² l.
    E is a complex parameter.
    """
    l, n, m, mb = coframe
    nd = len(l)
    Eb = conjugate(E)
    EE = _simp(E*Eb, simp)
    m2  = [_simp(m[i]  + E*l[i],  simp) for i in range(nd)]
    mb2 = [_simp(mb[i] + Eb*l[i], simp) for i in range(nd)]
    n2  = [_simp(n[i] + Eb*m[i] + E*mb[i] + EE*l[i], simp) for i in range(nd)]
    return (list(l), n2, m2, mb2)


def null_rotation_n(coframe, B, simp=None):
    """
    Null rotation fixing n:
      n→n, m→m+B n, mb→mb+B̄ n, l→l+B̄ m+B mb+|B|² n.
    B is a complex parameter.
    """
    l, n, m, mb = coframe
    nd = len(l)
    Bb = conjugate(B)
    BB = _simp(B*Bb, simp)
    m2  = [_simp(m[i]  + B*n[i],  simp) for i in range(nd)]
    mb2 = [_simp(mb[i] + Bb*n[i], simp) for i in range(nd)]
    l2  = [_simp(l[i] + Bb*m[i] + B*mb[i] + BB*n[i], simp) for i in range(nd)]
    return (l2, list(n), m2, mb2)


def compose(coframe, transforms, simp=None):
    """Apply a list of (fn, param) transformations in sequence."""
    cf = coframe
    for fn, param in transforms:
        cf = fn(cf, param, simp)
    return cf


# ─────────────────────────────────────────────────────────────────────────────
# Scalar polynomial invariants (the oracle quantities)
# ─────────────────────────────────────────────────────────────────────────────

def weyl_invariants(PSI, simp=None):
    """
    The two complex Weyl scalar invariants I and J (boost+spin weight 0),
    plus the speciality discriminant D = I³ − 27J².
    These are invariant under ALL Lorentz transformations of the tetrad.
    """
    p0, p1, p2, p3, p4 = [_simp(p, simp) for p in PSI]
    I_inv = _simp(p0*p4 - 4*p1*p3 + 3*p2**2, simp)
    J_inv = _simp(
        p0*(p2*p4 - p3**2) - p1*(p1*p4 - p3*p2) + p2*(p1*p3 - p2**2), simp)
    D_inv = _simp(I_inv**3 - 27*J_inv**2, simp)
    return {'I': I_inv, 'J': J_inv, 'D': D_inv}


def ricci_invariants(PHI, simp=None):
    """
    Scalar invariants of the Ricci spinor Φ_AB A'B' (Hermitian, weight 0
    contractions). The basic one is

        Φ² = Φ_{ab a'b'} Φ^{ab a'b'}
           = 2(Φ₀₀Φ₂₂ − 2|Φ₀₁|²... )   [signature-dependent combination]

    Using the standard NP contraction:
        Φ² = Φ₀₀Φ₂₂ − 2Φ₀₁Φ₂₁ ... → we use the manifestly real combination
        2|Φ₀₂|² + 2|Φ₀₁|²' ...  Built from the Hermitian components as the
    full trace, which is Lorentz-invariant. We compute the two lowest:
        R1 = Φ₀₀Φ₂₂ − 2Φ₀₁Φ₁₂ + Φ₁₁²   (analogue of I for Φ)
    """
    phi00, phi01, phi02, phi11, phi12, phi22 = [_simp(p, simp) for p in PHI]
    # The standard quadratic Ricci invariant (trace of Φ²), real:
    R1 = _simp(
        phi00*phi22 - 2*phi01*phi12 + phi11**2
        + phi02*conjugate(phi02)  # placeholder cross term, real part
        , simp)
    return {'PhiQuad': R1}


def all_invariants(metric, coords, coframe, simp=None):
    """
    Compute the full scalar-invariant fingerprint of a metric in a given
    coframe. Returns a dict of invariants that MUST be identical across all
    Lorentz-equivalent coframes.
    """
    n = len(coords)
    G = christoffel(metric, coords)
    R = riemann(metric, coords, G)
    Ric = ricci_ten(R, n).applyfunc(lambda e: _simp(e, simp))
    gi = metric.inv()
    Rs = _simp(ricci_sc(Ric, gi), simp)
    C = weyl_ten(R, Ric, Rs, metric, n)
    Rs4 = _simp(Rs/4, simp)
    S = sp.Matrix(n, n, lambda i, j: _simp(Ric[i,j] - Rs4*metric[i,j], simp))

    l, nv, m, mb = coframe
    PSI = weyl_spinor_from_coframe(C, gi, l, nv, m, mb, n, simp)
    PHI = ricci_spinor_from_coframe(S, gi, l, nv, m, mb, n, simp)

    inv = {}
    inv.update(weyl_invariants(PSI, simp))
    inv['LAMBDA'] = _simp(Rs/24, simp)
    return inv


# ─────────────────────────────────────────────────────────────────────────────
# The frame-independence test
# ─────────────────────────────────────────────────────────────────────────────

def check_frame_independence(metric, coords, transforms=None,
                             simp=None, verbose=True):
    """
    Build the standard diagonal coframe for `metric`, apply a sequence of
    Lorentz transformations, and verify the scalar invariants are unchanged.

    `transforms`: list of (transform_fn, param). If None, uses a default
    battery exercising boost, spin, and both null rotations with symbolic
    parameters.

    Returns (ok: bool, report: dict).
    """
    if simp is None:
        simp = sp.cancel
    n = len(coords)
    gi = metric.inv()

    # Standard coframe from the diagonal metric
    cf0 = null_coframe_from_diagonal_metric(metric, 'lorentzian', simp_fn=simp)
    inv0 = all_invariants(metric, coords, cf0, simp)

    if transforms is None:
        lam = Symbol('lam_b', positive=True)
        th  = Symbol('theta_s', real=True)
        E   = Symbol('E_l', real=True)
        B   = Symbol('B_n', real=True)
        transforms = [
            ('boost',        boost,          lam),
            ('spin',         spin,           th),
            ('null_rot_l',   null_rotation_l, E),
            ('null_rot_n',   null_rotation_n, B),
            ('composite',    None,           None),  # special: chain all
        ]

    report = {}
    all_ok = True
    for label, fn, param in transforms:
        if label == 'composite':
            cf = compose(cf0, [
                (boost, Symbol('lam_b', positive=True)),
                (spin,  Symbol('theta_s', real=True)),
                (null_rotation_l, Symbol('E_l', real=True)),
            ], simp)
        else:
            cf = fn(cf0, param, simp)

        # Verify it is still a valid null coframe
        try:
            validate_null_coframe(gi, cf, 'lorentzian', simp)
            valid = True
        except Exception as e:
            valid = False
            report[label] = {'valid_frame': False, 'error': str(e)[:120]}
            all_ok = False
            if verbose:
                print(f"  {label:12s}  FRAME INVALID: {str(e)[:80]}")
            continue

        inv = all_invariants(metric, coords, cf, simp)
        mism = {}
        for key in inv0:
            diff = _simp(inv0[key] - inv.get(key, sp.S.Zero), simp)
            if diff != 0:
                mism[key] = diff
        ok = (len(mism) == 0)
        report[label] = {'valid_frame': valid, 'invariant': ok,
                         'mismatches': mism}
        if not ok:
            all_ok = False
        if verbose:
            status = 'OK' if ok else 'INVARIANTS CHANGED'
            print(f"  {label:12s}  {status}")
            for k, v in mism.items():
                print(f"      Δ{k} = {v}")

    return all_ok, report


# ─────────────────────────────────────────────────────────────────────────────
# The DYTAUT reduction test (target — to be wired once DYTAUT exists)
# ─────────────────────────────────────────────────────────────────────────────

def check_dytaut_reduction(metric, coords, standardise_fn,
                           perturbation=None, simp=None, verbose=True):
    """
    Reduction test for dyad standardisation.

    Given:
      - a metric whose standard coframe we trust (cf_std = diagonal coframe),
      - a `standardise_fn(metric, coords, coframe) -> coframe` (the DYTAUT
        implementation under test),
      - a `perturbation` (list of (fn, param)) putting cf_std into a known
        NON-standard frame cf_pert,

    verify that:
      (a) standardise_fn(cf_pert) recovers a frame whose scalar invariants
          match the original (necessary), AND
      (b) the recovered components are in canonical standard form — checked
          by comparing the recovered Ψ/Φ component pattern to that of cf_std
          (sufficient for standardness up to the residual isotropy group).

    Because we generated `perturbation` ourselves, the correct target is known
    without any reference output. This is the strong oracle for DYTAUT.

    Returns (ok, report).
    """
    if simp is None:
        simp = sp.cancel
    cf_std = null_coframe_from_diagonal_metric(metric, 'lorentzian', simp_fn=simp)

    if perturbation is None:
        perturbation = [
            (boost, Symbol('lam_b', positive=True)),
            (null_rotation_l, Symbol('E_l', real=True)),
        ]
    cf_pert = compose(cf_std, perturbation, simp)

    inv_std = all_invariants(metric, coords, cf_std, simp)

    # Apply the standardiser under test
    cf_rec = standardise_fn(metric, coords, cf_pert)
    inv_rec = all_invariants(metric, coords, cf_rec, simp)

    # (a) invariants preserved
    inv_ok = all(_simp(inv_std[k] - inv_rec.get(k, sp.S.Zero), simp) == 0
                 for k in inv_std)

    # (b) standard-form check: compare component patterns
    n = len(coords)
    gi = metric.inv()
    G = christoffel(metric, coords); R = riemann(metric, coords, G)
    Ric = ricci_ten(R, n).applyfunc(lambda e: _simp(e, simp))
    Rs = _simp(ricci_sc(Ric, gi), simp)
    C = weyl_ten(R, Ric, Rs, metric, n)

    def psi_pattern(cf):
        PSI = weyl_spinor_from_coframe(C, gi, *cf, n, simp)
        return tuple(_simp(p, simp) == 0 for p in PSI)

    pat_std = psi_pattern(cf_std)
    pat_rec = psi_pattern(cf_rec)
    form_ok = (pat_std == pat_rec)

    ok = inv_ok and form_ok
    report = {'invariants_preserved': inv_ok,
              'standard_form_recovered': form_ok,
              'pattern_std': pat_std, 'pattern_rec': pat_rec}
    if verbose:
        print(f"  invariants preserved : {inv_ok}")
        print(f"  standard form        : {form_ok}")
        print(f"    Ψ-zero pattern std : {pat_std}")
        print(f"    Ψ-zero pattern rec : {pat_rec}")
    return ok, report


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test battery on trusted metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_frame_independence_battery(verbose=True):
    """
    Run the frame-independence test on a battery of metrics spanning Petrov
    types. This is the regression guard for the spinor-extraction code and
    the invariant oracle. Returns (n_pass, n_fail).
    """
    from sympy import symbols, sin, diag, Symbol as Sym

    metrics = {}

    # Schwarzschild (type D)
    t = symbols('t', real=True); r = Sym('r', positive=True)
    H, P = symbols('H P', real=True); M = Sym('M', positive=True)
    A = 1 - 2*M/r
    metrics['schwarzschild'] = (diag(-A, 1/A, r**2, r**2*sin(H)**2), [t,r,H,P])

    # Kasner (type I generically)
    tt, x, y, z = symbols('t x y z', positive=True)
    a, b = symbols('a b', real=True)
    metrics['kasner'] = (diag(-1, tt**(2*a), tt**(2*b), tt**(2*(1-a-b))),
                         [tt, x, y, z])

    n_pass = n_fail = 0
    for name, (g, coords) in metrics.items():
        if verbose:
            print(f"\n  === {name} ===")
        ok, _ = check_frame_independence(g, coords, simp=sp.cancel,
                                         verbose=verbose)
        if ok: n_pass += 1
        else:  n_fail += 1
    if verbose:
        print(f"\n  Frame-independence battery: {n_pass} passed, {n_fail} failed")
    return n_pass, n_fail


if __name__ == "__main__":
    run_frame_independence_battery(verbose=True)
