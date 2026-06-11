"""
pick/dytaut.py  —  Dyad (null-tetrad) standardisation: DYTAUT, type-D implementation.

Ported from CLASSI's dytaut.shp / dygend.shp / rootsd.shp (Jim Skea, 1986-87).

Architecture
------------
Works entirely in terms of the coframe operations defined in reduction_tests.py.
No spinor matrices, no Möbius substitution, no Iwasawa decomposition.

The key insight (empirically pinned on Schwarzschild):

  null_rotation_n(E) maps quartic root z → z - E   (fixes ∞)
  null_rotation_l(E) maps quartic root z → z/(1-Ez) (fixes 0, sends r→∞ when E=1/r)

Type-D standard form has PNDs at z=0 (aligned with l) and z=∞ (aligned with n).
The four cases below reduce any type-D frame to standard form.

Validation oracle: pick/reduction_tests.py::check_dytaut_reduction.
"""

import sympy as sp
from sympy import S, sqrt
from typing import Tuple, Optional

from .karlhede import _simp


# ─────────────────────────────────────────────────────────────────────────────
# Petrov path and root finder
# ─────────────────────────────────────────────────────────────────────────────

def petrov_path(PSI: list, simp_fn=None) -> str:
    """Nonzero pattern of Ψ₀..Ψ₄ as 5-char string of '0'/'1'."""
    def nz(p):
        p = _simp(p, simp_fn)
        return '0' if p == 0 else '1'
    return ''.join(nz(p) for p in PSI)


def petrov_roots_typeD(PSI: list, simp_fn=None) -> Tuple:
    """
    Compute the two double roots (PNDs) of the Weyl quartic for a type-D
    spinor.  Quartic: Ψ(z) = Ψ₀ + 4Ψ₁z + 6Ψ₂z² + 4Ψ₃z³ + Ψ₄z⁴.

    Returns ((root1, S.One), (root2, S.One)) for finite roots, or
    (S.Zero, S.Infinity) / (S.Infinity, S.Zero) for root at 0/∞.
    """
    s = lambda x: _simp(x, simp_fn)
    P0, P1, P2, P3, P4 = [s(p) for p in PSI]
    z = sp.Symbol('_z_pnd')

    if P4 == 0:
        # One root at ∞. Find the finite double root.
        if P3 == 0:
            sub = P0 + 4*P1*z + 6*P2*z**2
        else:
            sub = P0 + 4*P1*z + 6*P2*z**2 + 4*P3*z**3
        try:
            finite_roots = [s(r) for r in sp.solve(sub, z)]
            if finite_roots:
                return ((finite_roots[0], S.One), (S.Infinity, S.Zero))
        except Exception:
            pass
        raise ValueError(f"Could not find type-D roots (Ψ₄=0) for PSI={PSI}")

    quartic = P0 + 4*P1*z + 6*P2*z**2 + 4*P3*z**3 + P4*z**4
    try:
        roots_dict = sp.roots(quartic, z)
        double_roots = [s(r) for r, mult in roots_dict.items() if mult >= 2]
        if len(double_roots) >= 2:
            return ((double_roots[0], S.One), (double_roots[1], S.One))
        if len(double_roots) == 1:
            r1 = double_roots[0]
            deflated = sp.quo(quartic, (z - r1)**2, z)
            r2_roots = [s(r) for r in sp.solve(deflated, z)]
            if r2_roots:
                return ((r1, S.One), (r2_roots[0], S.One))
    except Exception:
        pass
    try:
        solns = [s(r) for r in sp.solve(quartic, z)]
        if len(solns) >= 2:
            return ((solns[0], S.One), (solns[1], S.One))
    except Exception:
        pass
    raise ValueError(f"Could not find type-D roots for PSI={PSI}")


# ─────────────────────────────────────────────────────────────────────────────
# Type-D standardiser
# ─────────────────────────────────────────────────────────────────────────────

def dygend(PSI: list, coframe: list, simp_fn=None):
    """
    Bring a type-D null tetrad to canonical form (only Ψ₂ nonzero).

    Uses only the coframe operations from reduction_tests.py.

    Root action (empirically pinned on Schwarzschild):
      null_rotation_n(E): z → z - E   (fixes ∞)
      null_rotation_l(E): z → z/(1-Ez) (fixes 0; r→∞ when E=1/r)

    Cases:
      already standard (path 00100): identity
      one root at ∞, other at r1: null_rotation_n(r1) → r1→0
      one root at 0, other at r2: null_rotation_l(1/r2) → r2→∞
      both finite at r1, r2:
          null_rotation_n(r1)         → r1→0, r2→r2-r1
          null_rotation_l(1/(r2-r1))  → r2-r1→∞

    Returns (coframe_std, PSI_placeholder).
    PSI in the standardised frame is best obtained by recomputing from the
    returned coframe via weyl_spinor_from_coframe.
    """
    from .reduction_tests import null_rotation_l as _nl, null_rotation_n as _nn

    s = lambda x: _simp(x, simp_fn)
    path = petrov_path(PSI, simp_fn)

    if path in ('00100', '10101'):
        # 00100: only Ψ₂, real roots at 0/∞ — already standard.
        # 10101: Ψ₀=Ψ₄, complex conjugate roots — kept in this frame;
        #        CLASSI would apply a complex Lorentz transform here but
        #        PICK handles the isotropy test directly in the 10101 frame.
        return [list(v) for v in coframe], PSI

    (r1_val, d1), (r2_val, d2) = petrov_roots_typeD(PSI, simp_fn)

    inf1 = (d1 == S.Zero or r1_val == S.Infinity)
    inf2 = (d2 == S.Zero or r2_val == S.Infinity)

    cf = [list(v) for v in coframe]

    if inf1 and inf2:
        # Both at ∞: degenerate, return as-is
        pass

    elif inf2 and not inf1:
        # root2 = ∞, root1 = r1 finite
        # null_rotation_n(r1): r1 → 0, ∞ → ∞
        r1 = s(r1_val)
        cf = _nn(cf, r1, simp_fn)

    elif inf1 and not inf2:
        # root1 = ∞, root2 = r2 finite
        # null_rotation_n(r2): r2 → 0, ∞ → ∞
        r2 = s(r2_val)
        cf = _nn(cf, r2, simp_fn)

    else:
        # Both finite at r1 and r2
        r1 = s(r1_val)
        r2 = s(r2_val)

        # Check if one is already at 0
        if r1 == S.Zero:
            # r1=0 already standard; send r2→∞
            cf = _nl(cf, s(-S.One / sp.conjugate(r2)), simp_fn)
        elif r2 == S.Zero:
            # r2=0 already standard; send r1→∞
            cf = _nl(cf, s(-S.One / sp.conjugate(r1)), simp_fn)
        else:
            # General: shift r1→0 then send r2-r1→∞
            cf = _nn(cf, r1, simp_fn)
            cf = _nl(cf, s(-S.One / sp.conjugate(r2 - r1)), simp_fn)

    return cf, PSI  # PSI placeholder; caller recomputes from coframe


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatch
# ─────────────────────────────────────────────────────────────────────────────

def standardise(PSI: list, coframe: list, petrov_type: str, simp_fn=None):
    """
    Entry point: bring an arbitrary null tetrad to standard form.

    Implements type D fully; types N, III, II partially (PND alignment only,
    no boost/spin normalisation); type I and 0 return unchanged.

    Returns standardised coframe as list of lists.
    """
    if petrov_type == 'D':
        cf, _ = dygend(PSI, coframe, simp_fn)
        return cf
    if petrov_type == 'N':
        cf, _ = dygenn(PSI, coframe, simp_fn)
        return cf
    if petrov_type == '3':
        cf, _ = dygen3(PSI, coframe, simp_fn)
        return cf
    if petrov_type == '2':
        cf, _ = dygen2(PSI, coframe, simp_fn)
        return cf
    if petrov_type == '1':
        cf, _ = dygen1(PSI, coframe, simp_fn)
        return cf
    # Type 0: conformally flat, handled via χ path (not yet implemented)
    return [list(v) for v in coframe]


# ─────────────────────────────────────────────────────────────────────────────
# General quartic root finder (all Petrov types)
# ─────────────────────────────────────────────────────────────────────────────

def quartic_roots(PSI: list, simp_fn=None) -> list:
    """
    Find roots of Ψ(z) = Ψ₀ + 4Ψ₁z + 6Ψ₂z² + 4Ψ₃z³ + Ψ₄z⁴ with multiplicities.

    Returns list of (root, multiplicity) pairs, including (S.Infinity, mult)
    for the root at ∞ when Ψ₄=0.

    Uses sympy.roots for exact symbolic factoring.
    """
    s = lambda x: _simp(x, simp_fn)
    P0, P1, P2, P3, P4 = [s(p) for p in PSI]
    z = sp.Symbol('_z_roots')

    results = []

    # Root at ∞ has multiplicity = 4 - degree of quartic
    # i.e. if Ψ₄=0 then leading coefficient vanishes
    degree = 4
    if P4 == 0:
        degree -= 1
    if P4 == 0 and P3 == 0:
        degree -= 1
    if P4 == 0 and P3 == 0 and P2 == 0:
        degree -= 1
    if P4 == 0 and P3 == 0 and P2 == 0 and P1 == 0:
        degree -= 1
    inf_mult = 4 - degree
    if inf_mult > 0:
        results.append((S.Infinity, inf_mult))

    if degree == 0:
        return results  # all roots at ∞

    # Build the polynomial of correct degree
    poly = P0
    if degree >= 1: poly = poly + 4*P1*z
    if degree >= 2: poly = poly + 6*P2*z**2
    if degree >= 3: poly = poly + 4*P3*z**3
    if degree >= 4: poly = poly + P4*z**4

    try:
        roots_dict = sp.roots(poly, z)
        for r, mult in roots_dict.items():
            results.append((s(r), mult))
    except Exception:
        try:
            solns = sp.solve(poly, z)
            seen = {}
            for r in solns:
                r = s(r)
                seen[r] = seen.get(r, 0) + 1
            for r, mult in seen.items():
                results.append((r, mult))
        except Exception:
            pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Type-N standardiser (DYGENN)
# ─────────────────────────────────────────────────────────────────────────────

def dygenn(PSI: list, coframe: list, simp_fn=None):
    """
    Bring a type-N null tetrad to canonical form (only Ψ₀≠0, path '10000').

    Type N has one 4-fold PND. Standard form has it aligned to the n direction
    (z=∞), giving Ψ₀≠0 only.

    Root action:
      null_rotation_l(E): z → z/(1−Ez), sends finite root r → ∞ when E=1/r.
    """
    from .reduction_tests import null_rotation_l as _nl

    s = lambda x: _simp(x, simp_fn)
    path = petrov_path(PSI, simp_fn)

    if path == '10000':
        return [list(v) for v in coframe], PSI

    cf = [list(v) for v in coframe]
    roots = quartic_roots(PSI, simp_fn)

    # Find the 4-fold root
    r4 = None
    for r, mult in roots:
        if mult == 4:
            r4 = r
            break
    if r4 is None:
        # Fallback: take the root with highest multiplicity
        roots_sorted = sorted(roots, key=lambda x: -x[1])
        r4 = roots_sorted[0][0]

    if r4 == S.Infinity:
        pass  # already at ∞
    elif r4 == S.Zero:
        # 4-fold root at 0: swap l↔n to send 0↔∞
        cf = [list(cf[1]), list(cf[0]), list(cf[2]), list(cf[3])]
    else:
        # nl(E) sends ∞ → -1/E; to send r4→∞ set E = -1/r4
        cf = _nl(cf, s(-S.One / sp.conjugate(r4)), simp_fn)

    # ── Normalisation: set |Ψ₀| = 1 via boost ───────────────────────
    # After PND alignment, Ψ₀ ≠ 0 only.  CLASSI DYGENN!-10000:
    #   L = [[Ψ₀^(1/4), 0],[0, Ψ₀^(-1/4)]]  (boost+spin).
    # Coframe effect: boost(lambda) with lambda = |Ψ₀|^(1/2).
    # Recompute PSI at the aligned coframe to get current Ψ₀.
    # We skip this if PSI reconstruction is unavailable (caller can do it).
    # For now: just return aligned frame; caller recomputes PSI.

    return cf, PSI


# ─────────────────────────────────────────────────────────────────────────────
# Type-III standardiser (DYGEN3)
# ─────────────────────────────────────────────────────────────────────────────

def dygen3(PSI: list, coframe: list, simp_fn=None):
    """
    Bring a type-III null tetrad to canonical form (only Ψ₁≠0, path '01000').

    Type III has one 3-fold PND and one simple PND. Standard form:
      - 3-fold PND at z=∞ (n direction)
      - simple PND at z=0 (l direction)
    giving Ψ₁≠0 only.

    Strategy:
      1. Find 3-fold root r3 and simple root r1.
      2. null_rotation_l(1/r3): r3 → ∞.
      3. The simple root shifts to r1' = r1/(1 − r1/r3).
      4. null_rotation_n(r1'): r1' → 0.
    """
    from .reduction_tests import null_rotation_l as _nl, null_rotation_n as _nn

    s = lambda x: _simp(x, simp_fn)
    path = petrov_path(PSI, simp_fn)

    if path == '01000':
        return [list(v) for v in coframe], PSI

    cf = [list(v) for v in coframe]
    roots = quartic_roots(PSI, simp_fn)

    r3 = None  # 3-fold root
    r1 = None  # simple root
    for r, mult in roots:
        if mult == 3:
            r3 = r
        elif mult == 1:
            r1 = r

    # Root actions: nl(E): 0→0, inf→-1/E, r→r/(1-rE)
    #               nn(E): inf→inf, r→r-E  (shift by -E)

    if r3 == S.Infinity:
        # Triple root already at ∞. Just send simple root to 0.
        if r1 is not None and r1 != S.Zero and r1 != S.Infinity:
            cf = _nn(cf, r1, simp_fn)

    elif r3 == S.Zero:
        # Triple root at 0. Need triple→∞, simple→0.
        # nn(r1): shifts r1→0, triple 0→-r1.
        # nl(-1/r1): sends -r1→∞ (denom 1-(-r1)·(-1/r1)=0), simple stays 0.
        if r1 is not None and r1 != S.Infinity and r1 != S.Zero:
            cf = _nn(cf, r1, simp_fn)
            cf = _nl(cf, s(-S.One / sp.conjugate(r1)), simp_fn)
        # Other degenerate cases (r1=0 or r1=∞): graceful degradation

    else:
        # Triple root at finite r3 ≠ 0. nl(-1/r3): r3→∞, r→r/(1-r*(-1/r3)).
        E3 = s(-S.One / sp.conjugate(r3))
        cf = _nl(cf, E3, simp_fn)
        # Update r1: r1 → r1/(1 - r1*E3)
        if r1 is not None and r1 != S.Infinity:
            r1 = s(r1 / (1 - r1 * E3))
        elif r1 == S.Infinity:
            r1 = s(-S.One / E3)  # inf → r3 (but r3 now at ∞ — skip step 2)
            r1 = None
        if r1 is not None and r1 != S.Zero and r1 != S.Infinity:
            cf = _nn(cf, r1, simp_fn)

    return cf, PSI


# ─────────────────────────────────────────────────────────────────────────────
# Type-II standardiser (DYGEN2)
# ─────────────────────────────────────────────────────────────────────────────

def dygen2(PSI: list, coframe: list, simp_fn=None):
    """
    Bring a type-II null tetrad to canonical form (Ψ₂,Ψ₃,Ψ₄≠0, path '00111').

    Type II has one 2-fold PND and two simple PNDs. Standard form:
      - double PND at z=0 (l direction), giving Ψ₀=Ψ₁=0.

    Strategy:
      1. Find double root r2.
      2. null_rotation_n(r2): r2 → 0.
    """
    from .reduction_tests import null_rotation_n as _nn

    s = lambda x: _simp(x, simp_fn)
    path = petrov_path(PSI, simp_fn)

    # Standard forms for type II have double PND at z=0: Ψ₀=Ψ₁=0
    # Check if already in standard form (first two components zero)
    P0 = _simp(PSI[0], simp_fn)
    P1 = _simp(PSI[1], simp_fn)
    if P0 == 0 and P1 == 0:
        return [list(v) for v in coframe], PSI

    cf = [list(v) for v in coframe]
    roots = quartic_roots(PSI, simp_fn)

    r2 = None  # double root
    for r, mult in roots:
        if mult == 2:
            r2 = r
            break

    # nn(E) shifts all finite roots by -E. To send r2→0: apply nn(r2).
    if r2 is not None and r2 != S.Zero and r2 != S.Infinity:
        cf = _nn(cf, r2, simp_fn)
    elif r2 == S.Infinity:
        # Double root at ∞. nl(E) sends inf → -1/E.
        # To send inf to 0: not directly possible with single nl.
        # Would need nl then nn. For now skip (graceful degradation).
        pass

    return cf, PSI


# ─────────────────────────────────────────────────────────────────────────────
# Type-I standardiser (DYGEN1) — placeholder
# ─────────────────────────────────────────────────────────────────────────────

def dygen1(PSI: list, coframe: list, simp_fn=None):
    """
    Bring a type-I null tetrad to canonical form (Ψ₀=Ψ₄, Ψ₂≠0, path '10101').

    Type I has four distinct PNDs. Standard form has Ψ₁=Ψ₃=0 and Ψ₀=Ψ₄.

    TODO: implement full type-I standardisation.
    Currently returns coframe unchanged (graceful degradation).
    """
    # Not yet implemented — return unchanged
    return [list(v) for v in coframe], PSI


# ─────────────────────────────────────────────────────────────────────────────
# Type-N normalisation (DYGENN second half)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_typeN(PSI: list, coframe: list, simp_fn=None):
    """
    Second half of DYGENN: after PND alignment (path '10000'), normalise
    Ψ₀ to a constant via a boost.

    Empirically pinned frame actions (wavez, 2026-06-11):
        boost(A):  Ψ₀ → A²·Ψ₀
        spin(θ):   Ψ₀ → e^{2iθ}·Ψ₀

    CLASSI's DYGENN!-10000 dyad is diag(Ψ₀^(1/4), Ψ₀^(-1/4)).  The boost
    is NOT isotropy for type N (Ψ₀ has boost weight 2), so the standard
    frame fixes this freedom by absorbing |Ψ₀| into the frame.  This is
    required for the FUNTST count: e.g. wavez has t₀ = 0 because the one
    function p(u) in Ψ₀ is absorbed here, reappearing only at order 1
    through ∇C.

    A = (Ψ₀²)^(-1/4) = |Ψ₀|^(-1/2) for real Ψ₀, kept as a power (no Abs)
    so the frame stays differentiable.  The residual constant (±1) carries
    no coordinate dependence.  Phase (spin) normalisation for genuinely
    complex Ψ₀ is not yet implemented.

    Returns (coframe, changed).
    """
    from .reduction_tests import boost as _boost
    s = lambda x: _simp(x, simp_fn)

    if petrov_path(PSI, simp_fn) != '10000':
        return [list(v) for v in coframe], False
    P0 = s(PSI[0])
    if P0 == 0:
        return [list(v) for v in coframe], False
    A = s((P0**2)**sp.Rational(-1, 4))
    if A == 1:
        return [list(v) for v in coframe], False
    cf = _boost([list(v) for v in coframe], A, simp_fn)
    return [list(v) for v in cf], True
