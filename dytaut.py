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

    if path == '00100':
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
            cf = _nl(cf, s(S.One / r2), simp_fn)
        elif r2 == S.Zero:
            # r2=0 already standard; send r1→∞
            cf = _nl(cf, s(S.One / r1), simp_fn)
        else:
            # General: shift r1→0 then send r2-r1→∞
            cf = _nn(cf, r1, simp_fn)
            cf = _nl(cf, s(S.One / (r2 - r1)), simp_fn)

    return cf, PSI  # PSI placeholder; caller recomputes from coframe


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatch
# ─────────────────────────────────────────────────────────────────────────────

def standardise(PSI: list, coframe: list, petrov_type: str, simp_fn=None):
    """
    Entry point: bring an arbitrary null tetrad to standard form.
    Currently implements type D.  Other types raise NotImplementedError.
    """
    if petrov_type == 'D':
        return dygend(PSI, coframe, simp_fn)
    raise NotImplementedError(
        f"Standardisation for Petrov type {petrov_type} not yet implemented.")
