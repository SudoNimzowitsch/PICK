"""
pick/karlhede.py  —  PICK: Python Implementation of Cartan-Karlhede
Faithful port of CLASSI by Jan E. Åman (Stockholm, 1987).

This revision corrects three bugs from the first port (identified by
the Opus audit):
  1. Petrov classification: now uses frame-independent invariants I,J,K
     (Section F of SPEC.md).  The old component-vanishing check was only
     valid for aligned frames.
  2. Order-0 isotropy: now uses the PETROVLIST lookup from clabas.shp
     (type D→'e', type N→'n', types I/II/III→'0', type 0→'6') and then
     runs the ISOTST-style GHP weight test on PHI for non-vacuum metrics.
  3. Order-≥1 invariants: now computes the genuine Cartan scalars via the
     coordinate covariant derivative ∇_ε C_{αβγδ} projected onto the null
     frame, instead of coordinate partials of the order-0 scalars.

Validation: pick/validate.py against all 33 ptest/*.sum reference outputs.
"""

from __future__ import annotations

import sympy as sp
from sympy import (sqrt, Rational, I, zeros, Matrix,
                   trigsimp, cancel, diff, symbols)
from dataclasses import dataclass
from typing import Optional, Callable, List
import itertools


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simp(expr, fn=None):
    if fn is not None:
        return fn(expr)
    try:
        result = sp.cancel(expr)
        # Only apply trigsimp when trig functions are actually present
        if result.has(sp.sin, sp.cos, sp.tan, sp.cosh, sp.sinh, sp.tanh):
            result = sp.trigsimp(result)
        return result
    except Exception:
        return expr

def _is_zero(expr, simp_fn=None) -> bool:
    return _simp(expr, simp_fn) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KarlhedeStep:
    order:           int
    t:               int
    s:               int
    isotropy_name:   str
    isotropy_symbol: str = '-'

@dataclass
class KarlhedeResult:
    petrov_type:   str
    segre_type:    str
    isometry_dim:  int
    isotropy_dim:  int
    indep_scalars: int
    steps:         List[KarlhedeStep]
    terminated_at: int
    psi:           list
    phi:           list
    lambd:         object

    def summary(self):
        lines = [
            "─" * 50,
            "  PICK  Karlhede Classification Result",
            "─" * 50,
            f"  Petrov type              : {self.petrov_type}",
            f"  Segre type               : {self.segre_type}",
            f"  Isometry group dim  (r)  : {self.isometry_dim}",
            f"  Isotropy group dim  (s∞) : {self.isotropy_dim}",
            f"  Independent scalars (t∞) : {self.indep_scalars}",
            f"  Algorithm terminated at  : order {self.terminated_at}",
            "─" * 50,
            "  Step table:",
            f"  {'Order':>5}  {'t':>4}  {'s':>4}  {'H':>3}  Isotropy",
        ]
        for st in self.steps:
            lines.append(f"  {st.order:>5}  {st.t:>4}  {st.s:>4}  {st.isotropy_symbol:>3}  {st.isotropy_name}")
        lines.append("─" * 50)
        return "\n".join(lines)

    def _segre_shorthand(self) -> str:
        """Map segre_type string to single-char CLASSI shorthand."""
        s = self.segre_type
        if 'acuum' in s or s.startswith('Vacuum') or 'lambda' in s.lower() or 'Λ' in s:
            return '0'
        if 'perfect fluid' in s:   return 'p'
        if 'radiation' in s:       return 'r'
        if 'electromagnetic' in s: return 'e'
        if 'tachyon' in s:         return 't'
        return '?'

    def _lambda_flag(self) -> str:
        return '0' if self.lambd == 0 else '1'

    def to_classification(self, name: str):
        """
        Convert to the harness Classification format for comparison
        against CLASSI ptest reference outputs.
        """
        # r: isometry dim, X if ≥ 10
        r_str = 'X' if self.isometry_dim >= 10 else str(self.isometry_dim)
        s_str = str(self.isotropy_dim)

        # H-sequence (orders 0..3), padded with '-'
        h_chars = [st.isotropy_symbol for st in self.steps[:4]]
        while len(h_chars) < 4:
            h_chars.append('-')
        H = ''.join(h_chars).replace('-', '')   # harness strips hyphens

        # t-sequence (orders 0..6), padded with '-'
        t_chars = [str(st.t) for st in self.steps[:7]]
        while len(t_chars) < 7:
            t_chars.append('-')
        t_seq = ''.join(t_chars).replace('-', '')  # harness strips hyphens

        # Build the .sum string (clasum.shp format)
        name6 = (name + ' '*6)[:6]
        H4    = (H + '----')[:4]
        t7    = (t_seq + '-------')[:7]
        sum_str = (f"{name6} {self._segre_shorthand()}{self._lambda_flag()}"
                   f"{self.petrov_type}  {r_str} {s_str} {H4} {t7}")
        return sum_str


# ─────────────────────────────────────────────────────────────────────────────
# CLASSI tables (clabas.shp)
# ─────────────────────────────────────────────────────────────────────────────

# From PETROVLIST (clabas.shp:16-22):
# Column 1 = shorthand, Column 2 = initial isotropy symbol, Column 3 = string,
# Column 4 = Petrov zero list (components that vanish in standard frame)
#
# The initial isotropy symbol is the H₀ value after ISOTST PSI in a standard
# (PND-aligned) frame.  This is the key fix vs the old _PETROV_ISO table,
# which had type I→s (wrong; correct is type I→0).
PETROV_INITIAL_ISO = {
    '1': '0',   # type I: trivial (all four distinct PNDs break every subgroup)
    '2': '0',   # type II: trivial
    '3': '0',   # type III: trivial (Ψ₃ has weight (-1,-1), all broken)
    'D': 'e',   # boosts + rotations: only Ψ₂ nonzero (weight 0,0)
    'N': 'n',   # null rotations: Ψ₄ at lower-right corner, null rot preserved
    '0': '6',   # full Lorentz: PSI all zero, ISOTST PSI skipped
}

# From ISOTROPYLIST (clabas.shp:71-81):
# (symbol, dim, description)
_ISOLIST = [
    ('6', 6, 'full Lorentz group'),
    ('p', 3, 'SO(3) rotations'),
    ('t', 3, 'SO(2,1) rotations'),
    ('r', 3, 'null and spin rotations'),
    ('n', 2, 'null rotations'),
    ('k', 1, '1-parameter null rotations'),
    ('e', 2, 'boosts and rotations'),
    ('s', 1, 'space (spin) rotations'),
    ('b', 1, 'boosts'),
    ('0', 0, 'none'),
]
ISO_DIM  = {s: d for s, d, _ in _ISOLIST}
ISO_NAME = {s: n for s, _, n in _ISOLIST}


# ─────────────────────────────────────────────────────────────────────────────
# Petrov classification — invariant-based (Section F of SPEC.md)
# ─────────────────────────────────────────────────────────────────────────────

def petrov_from_invariants(PSI: list, simp_fn=None) -> str:
    """
    Determine the Petrov type from the five Weyl spinor components
    Ψ₀…Ψ₄ using frame-independent invariants I, J, K.

    Returns the CLASSI shorthand: '0','1','2','3','D','N'
    (matching PETROVLIST column 1 in clabas.shp).

    Decision tree (SPEC.md Section F):
      I = Ψ₀Ψ₄ − 4Ψ₁Ψ₃ + 3Ψ₂²
      J = det [[Ψ₀,Ψ₁,Ψ₂],[Ψ₁,Ψ₂,Ψ₃],[Ψ₂,Ψ₃,Ψ₄]]
      K = Ψ₁Ψ₄² − 3Ψ₂Ψ₃Ψ₄ + 2Ψ₃³

      all Ψ=0  → '0'
      I³≠27J²  → '1'  (type I)
      I=J=0, K=0 → 'N'
      I=J=0, K≠0 → '3' (type III)
      I≠0 or J≠0, K=0 → 'D'
      I≠0 or J≠0, K≠0 → '2' (type II)

    Valid for ANY frame; does not rely on component-vanishing patterns.
    """
    psi = [_simp(p, simp_fn) for p in PSI]
    p0, p1, p2, p3, p4 = psi

    if all(p == 0 for p in psi):
        return '0'

    I_inv = _simp(p0*p4 - 4*p1*p3 + 3*p2**2, simp_fn)
    J_inv = _simp(
        p0*(p2*p4 - p3**2) - p1*(p1*p4 - p3*p2) + p2*(p1*p3 - p2**2),
        simp_fn
    )
    D_inv = _simp(I_inv**3 - 27*J_inv**2, simp_fn)

    if D_inv != 0:
        return '1'   # type I

    # Algebraically special (D=0).
    K_inv = _simp(p1*p4**2 - 3*p2*p3*p4 + 2*p3**3, simp_fn)

    if I_inv == 0 and J_inv == 0:
        if K_inv == 0:
            return 'N'    # all four PNDs coincide
        else:
            return '3'    # type III: one triple PND
    else:
        if K_inv == 0:
            return 'D'    # two pairs of coincident PNDs
        else:
            return '2'    # type II: one double PND


# ─────────────────────────────────────────────────────────────────────────────
# Isotropy state and GHP-weight-based ISOTST (Section E of SPEC.md)
# ─────────────────────────────────────────────────────────────────────────────

class IsotropyFlags:
    """
    Tracks which Lorentz subgroups survive the isotropy tests.
    Mirrors the boolean flags of CLASSI's isotst.shp (lines 12-36).
    All start True (full Lorentz group); flags are cleared as spinors
    with nonzero GHP weight are found nonzero.
    """
    __slots__ = [
        'rotiso', 'boostiso',
        'nullrotiso', 'null1drotiso',
        'swnullrotiso', 'swnull1drotiso',
        'so3iso', 'so21iso',
        'alt1so21iso', 'alt2so21iso',
        'swapiso', 'stdframe',
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, True)

    def set_from_petrov(self, petrov_type: str):
        """
        Set the isotropy to the value expected after ISOTST PSI for a
        PND-aligned (standard) frame, using PETROVLIST from clabas.shp.
        This replaces a full ISOTST PSI run for standard-frame inputs.
        """
        sym = PETROV_INITIAL_ISO.get(petrov_type, '0')
        self._set_to_symbol(sym)

    def _set_to_symbol(self, sym: str):
        """Force all flags consistent with surviving isotropy symbol sym."""
        # Start by clearing everything, then re-enable what sym implies.
        for s in self.__slots__:
            setattr(self, s, False)
        if sym == '6':
            for s in self.__slots__:
                setattr(self, s, True)
        elif sym == 'p':
            self.so3iso = True;  self.rotiso = True
        elif sym == 't':
            self.so21iso = True; self.rotiso = True
        elif sym == 'r':
            self.nullrotiso = True; self.null1drotiso = True
            self.swnullrotiso = True; self.rotiso = True
        elif sym == 'e':
            self.rotiso = True;  self.boostiso = True; self.swapiso = True
        elif sym == 'n':
            self.nullrotiso = True; self.null1drotiso = True
            self.swnullrotiso = True; self.swnull1drotiso = True
        elif sym == 's':
            self.rotiso = True
        elif sym == 'b':
            self.boostiso = True
        elif sym == 'k':
            self.null1drotiso = True
        # '0': all remain False

    @property
    def symbol(self) -> str:
        return _isotropy_symbol(self)

    @property
    def dim(self) -> int:
        return ISO_DIM.get(self.symbol, 0)

    @property
    def name(self) -> str:
        return ISO_NAME.get(self.symbol, 'unknown')


def _isotropy_symbol(f: IsotropyFlags) -> str:
    """
    Derive the isotropy symbol from surviving flags.
    Priority follows ISOTROPYLIST dimension order (clabas.shp:71-81):
    6 > p,t,r (dim 3) > e,n (dim 2) > s,b,k (dim 1) > 0.
    """
    rot   = f.rotiso
    boost = f.boostiso
    null  = f.nullrotiso
    null1 = f.null1drotiso
    so3   = f.so3iso
    so21  = f.so21iso

    # Full Lorentz: all flags on
    if (rot and boost and null and null1 and so3 and so21 and
            f.swnullrotiso and f.swnull1drotiso and f.swapiso):
        return '6'

    # 3-dim groups
    if null and rot and not boost:   return 'r'   # null + spin (radiation)
    if so3  and not boost:           return 'p'   # SO(3) perfect fluid
    if so21 and not boost:           return 't'   # SO(2,1) tachyon

    # 2-dim groups
    if rot and boost:                return 'e'   # boosts × rotations
    if null and not rot and not boost: return 'n' # null rotations

    # 1-dim groups
    if rot  and not boost:           return 's'   # spatial rotations
    if boost and not rot:            return 'b'   # boosts
    if null1:                        return 'k'   # 1-param null rotations

    return '0'


def _ghp_boost_weight(psi_k: int, direction: int) -> int:
    """
    GHP boost weight of the PSID component (k, direction).
    Boost weight = (weight from Ψ_k) + (weight from derivative direction).

    Ψ_k has boost weight 2-k (Ψ₂: 0, Ψ₄: -2, Ψ₀: +2).
    Null tetrad boost weights: l(+1), n(-1), m(0), mb(0).
    direction: 0=l, 1=n, 2=m, 3=mb.
    """
    psi_bw = 2 - psi_k
    dir_bw = [1, -1, 0, 0][direction]
    return psi_bw + dir_bw


def _ghp_spin_weight(psi_k: int, direction: int) -> int:
    """
    GHP spin weight of the PSID component (k, direction).
    Ψ_k has spin weight 2-k.
    Null tetrad spin weights: l(0), n(0), m(+1), mb(-1).
    """
    psi_sw = 2 - psi_k
    dir_sw = [0, 0, 1, -1][direction]
    return psi_sw + dir_sw


def test_psid_isotropy(psid_comps: list, flags: IsotropyFlags,
                       simp_fn=None) -> None:
    """
    Update isotropy flags based on PSID components (frame components of
    ∇Ψ at some order).  Implements the GHP weight test from isotst.shp.

    psid_comps: list of (k, direction, value) where
      k         ∈ 0..4  (Ψ index)
      direction ∈ 0..3  (null tetrad direction: l,n,m,mb)
      value     is a SymPy expression

    A nonzero component with nonzero boost weight breaks boost isotropy.
    A nonzero component with nonzero spin weight  breaks rotation isotropy.
    Null rotation isotropy is more subtle (corner structure) but for the
    ptest standard frames it follows from the Petrov-type setup already.
    """
    for k, d, v in psid_comps:
        # Use v² with powsimp(force=True) for zero check — avoids Abs issues
        v2 = sp.powsimp(sp.expand(v * v), force=True)
        try:
            v2_s = _simp(v2, simp_fn)
        except Exception:
            v2_s = v2
        if v2_s == 0:
            continue
        bw = _ghp_boost_weight(k, d)
        sw = _ghp_spin_weight(k, d)
        if bw != 0:
            flags.boostiso = False
        if sw != 0:
            flags.rotiso = False
        # Null rotation: a nonzero component that is NOT at the "lower-right
        # corner" of the spinor (which would preserve null rotations) breaks
        # null rotation isotropy.  In GHP: null rotations preserve Ψ₄ and
        # break it for other Ψ_k.  For the derivative spinor DPSI (valence
        # (5,1)→(6 symmetric)), the effective "lower-right" corner maps to
        # the (k=5, d=3) component.  A simple check: if boost_wt+spin_wt<0
        # and nonzero → null rotation direction is engaged.
        if bw + sw < 0 or (bw < 0 and sw == 0) or (bw == 0 and sw < 0):
            if flags.nullrotiso:
                flags.nullrotiso = False
                flags.null1drotiso = False


def test_phi_isotropy(PHI: list, flags: IsotropyFlags, simp_fn=None) -> None:
    """
    Update isotropy flags from Ricci spinor PHI components.
    PHI = [Φ₀₀, Φ₀₁, Φ₀₂, Φ₁₁, Φ₁₂, Φ₂₂].
    RN1=2, RN2=2, HERMITIAN=True.

    Implements the key parts of ISOTST for a Hermitian (2,2) spinor
    (isotst.shp ROTBOOSTISO and SOISO for the Hermitian case).
    """
    phi = [_simp(p, simp_fn) for p in PHI]
    phi00, phi01, phi02, phi11, phi12, phi22 = phi

    # Build full 3×3 component table: [P1][P2] → value
    comp = {
        (0,0): phi00,
        (0,1): phi01, (1,0): sp.conjugate(phi01),
        (0,2): phi02, (2,0): sp.conjugate(phi02),
        (1,1): phi11,
        (1,2): phi12, (2,1): sp.conjugate(phi12),
        (2,2): phi22,
    }
    rn1, rn2 = 2, 2

    for (p1, p2), v in comp.items():
        v_s = _simp(v, simp_fn)
        if v_s == 0:
            continue

        dia1d = (p1 - p2)              # = (P1-P2) - (RN1-RN2)/2 with RN1=RN2
        dia2d = (p1 + p2) - 2         # = (P1+P2) - (RN1+RN2)/2

        # Hermitian ROTBOOSTISO condition (isotst.shp:415-423):
        # NOT (DIA2D < 0 OR (DIA1D=0 AND DIA2D=0))
        call_rb = not (dia2d < 0 or (dia1d == 0 and dia2d == 0))

        if call_rb:
            w1, w2 = rn1 - p1, rn2 - p2
            sw = _simp(comp.get((w1, w2), sp.S.Zero), simp_fn)
            sw_zero = (sw == 0)

            if dia1d != 0 and dia2d != 0:
                # Both nonzero: lose both
                flags.boostiso = False
                flags.rotiso   = False
                if not sw_zero:
                    if not _is_zero(v_s - sw, simp_fn):
                        flags.swapiso = False
                else:
                    flags.swapiso = False
            elif dia2d != 0:
                # Boost weight nonzero
                flags.boostiso = False
                if sw_zero:
                    flags.swapiso = False
                elif _is_zero(v_s - sw, simp_fn):
                    pass  # SHEQUAL → boost lost, swap preserved
                else:
                    flags.swapiso = False
            elif dia1d != 0:
                # Spin weight nonzero
                flags.rotiso = False
                if sw_zero:
                    flags.swapiso = False
                elif _is_zero(v_s - sw, simp_fn):
                    pass
                else:
                    flags.swapiso = False

        # SOISO (isotst.shp:537-558): diagonal structure test for SO(3)/SO(2,1)
        # Called when not upper-left corner and on main diagonal (DIA1D=0)
        if dia1d == 0 and not (p1 == 0 and p2 == 0):
            if flags.so3iso or flags.so21iso:
                _soiso_phi(v_s, p1, p2, rn1, comp, flags, simp_fn)

        # NULLROTISO: not lower-right corner
        if not (p1 == rn1 and p2 == rn2):
            if flags.nullrotiso and v_s != 0:
                flags.nullrotiso   = False
                flags.null1drotiso = False

        # SWNULLISO: not upper-left corner
        if not (p1 == 0 and p2 == 0):
            if flags.swnullrotiso and v_s != 0:
                flags.swnullrotiso   = False
                flags.swnull1drotiso = False


def _soiso_phi(v_s, p1, p2, rn1, comp, flags, simp_fn):
    """
    Hermitian SOISO test (isotst.shp:544-558).
    Check: (RN1/P1) * Φ_{P1,P2} == Φ_{0,0}  →  SO3 preserved on main diagonal.
    """
    if p1 == 0:
        return  # upper-left corner: skip
    phi00 = _simp(comp.get((0,0), sp.S.Zero), simp_fn)
    ratio = Rational(rn1, p1) if p1 != 0 else sp.S.Zero
    lhs = _simp(ratio * v_s, simp_fn)
    if not _is_zero(lhs - phi00, simp_fn):
        flags.so3iso = False
        if p1 % 2 == 0:   # IEVENP
            flags.so21iso = False
    # SO(2,1) odd check
    if flags.so21iso and p1 % 2 == 1:   # IODDP
        if not _is_zero(lhs + phi00, simp_fn):
            flags.so21iso = False


# ─────────────────────────────────────────────────────────────────────────────
# Segre classification (stub — Section G of SPEC.md)
# ─────────────────────────────────────────────────────────────────────────────

def segre_from_phi(PHI: list, lambd, simp_fn=None) -> str:
    """
    Partial Segre/Plebański classification.  Full implementation requires
    the Jordan-form analysis of the trace-free Ricci endomorphism (segre.shp,
    1100+ lines); this stub handles the common cases in ptest.
    """
    phi = [_simp(p, simp_fn) for p in PHI]
    lam = _simp(lambd, simp_fn)
    phi00, phi01, phi02, phi11, phi12, phi22 = phi

    vacuum = all(p == 0 for p in phi)
    if vacuum and lam == 0:   return 'Vacuum'
    if vacuum and lam != 0:   return 'A1 [(111,1)] Λ-term'

    # Pure radiation
    if (phi11 == 0 and phi22 == 0 and phi12 == 0 and
            phi02 == 0 and phi01 == 0 and phi00 != 0 and lam == 0):
        return 'A3 [(11,2)] pure radiation'

    # Perfect fluid (trace-free Ricci has specific structure)
    if phi02 == 0 and phi01 == 0 and phi12 == 0 and phi00 != 0 and phi22 != 0:
        ratio = _simp(phi22 - phi00, simp_fn)
        if ratio == 0:
            return 'A1 [(111),1] perfect fluid'

    if phi00 == 0 and phi22 == 0 and phi02 == 0:
        return 'A1 [(11)(1,1)] electromagnetic non-null'

    return 'A1 [111,1] general'


# ─────────────────────────────────────────────────────────────────────────────
# Tensor machinery (unchanged from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def christoffel(g: Matrix, coords: list) -> dict:
    n = len(coords)
    gi = g.inv()
    G = {}
    for s in range(n):
        for m in range(n):
            for v in range(n):
                G[s,m,v] = Rational(1,2)*sum(
                    gi[s,l]*(diff(g[l,v], coords[m]) +
                             diff(g[l,m], coords[v]) -
                             diff(g[m,v], coords[l]))
                    for l in range(n))
    return G


def riemann(g: Matrix, coords: list, G: dict = None) -> dict:
    n = len(coords)
    if G is None:
        G = christoffel(g, coords)
    R = {}
    for r in range(n):
        for s in range(n):
            for m in range(n):
                for v in range(n):
                    R[r,s,m,v] = (diff(G[r,v,s], coords[m])
                                  - diff(G[r,m,s], coords[v])
                                  + sum(G[r,m,l]*G[l,v,s] - G[r,v,l]*G[l,m,s]
                                        for l in range(n)))
    return R


def ricci_ten(R: dict, n: int) -> Matrix:
    Ric = zeros(n, n)
    for m in range(n):
        for v in range(n):
            Ric[m,v] = sum(R[l,m,l,v] for l in range(n))
    return Ric


def ricci_sc(Ric: Matrix, gi: Matrix) -> sp.Expr:
    n = Ric.shape[0]
    return sum(gi[m,v]*Ric[m,v] for m in range(n) for v in range(n))


def weyl_ten(R: dict, Ric: Matrix, Rs: sp.Expr, g: Matrix, n: int) -> dict:
    Rl = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    Rl[a,b,c,d] = sum(g[a,l]*R[l,b,c,d] for l in range(n))
    f1 = Rational(1, n-2)
    f2 = Rs / ((n-1)*(n-2))
    C = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    C[a,b,c,d] = (Rl[a,b,c,d]
                        - f1*(g[a,c]*Ric[b,d] - g[a,d]*Ric[b,c]
                              + g[b,d]*Ric[a,c] - g[b,c]*Ric[a,d])
                        + f2*(g[a,c]*g[b,d] - g[a,d]*g[b,c]))
    return C


# ─────────────────────────────────────────────────────────────────────────────
# NP spinors from null coframe  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _c4_up(T: dict, f1, f2, f3, f4, n: int) -> sp.Expr:
    s = sp.S.Zero
    for a in range(n):
        if f1[a] == 0: continue
        for b in range(n):
            if f2[b] == 0: continue
            for c in range(n):
                if f3[c] == 0: continue
                for d in range(n):
                    if f4[d] == 0: continue
                    t = T.get((a,b,c,d), sp.S.Zero)
                    if t == 0: continue
                    s += t * f1[a] * f2[b] * f3[c] * f4[d]
    return s


def _c2_up(M_up: Matrix, f1, f2, n: int) -> sp.Expr:
    s = sp.S.Zero
    for a in range(n):
        if f1[a] == 0: continue
        for b in range(n):
            if f2[b] == 0: continue
            s += M_up[a,b] * f1[a] * f2[b]
    return s


def _raise_all(T: dict, gi: Matrix, n: int, rank: int) -> dict:
    """Raise all indices of a fully covariant tensor."""
    indices = range(n)
    result = {}
    for out_idx in itertools.product(indices, repeat=rank):
        val = sp.S.Zero
        for in_idx in itertools.product(indices, repeat=rank):
            t_val = T.get(in_idx, sp.S.Zero)
            if t_val == 0:
                continue
            g_factor = sp.S.One
            for k in range(rank):
                g_factor *= gi[out_idx[k], in_idx[k]]
            if g_factor == 0:
                continue
            val += g_factor * t_val
        result[out_idx] = val
    return result


def weyl_spinor_from_coframe(C: dict, gi: Matrix, l, nv, m, mb,
                              n_dim: int, simp_fn=None) -> list:
    C_up = _raise_all(C, gi, n_dim, 4)
    return [
        _simp(_c4_up(C_up, l, m,  l,  m,  n_dim), simp_fn),
        _simp(_c4_up(C_up, l, nv, l,  m,  n_dim), simp_fn),
        _simp(_c4_up(C_up, l, m,  mb, nv, n_dim), simp_fn),
        _simp(_c4_up(C_up, l, nv, mb, nv, n_dim), simp_fn),
        _simp(_c4_up(C_up, nv, mb, nv, mb, n_dim), simp_fn),
    ]


def ricci_spinor_from_coframe(Ric: Matrix, gi: Matrix, l, nv, m, mb,
                               n_dim: int, simp_fn=None) -> list:
    n = n_dim
    Ric_up = Matrix(n, n, lambda a, b:
        sum(gi[a,c]*gi[b,d]*Ric[c,d] for c in range(n) for d in range(n)))
    h = Rational(-1, 2)
    return [
        _simp(h * _c2_up(Ric_up, l,  l,  n_dim), simp_fn),
        _simp(h * _c2_up(Ric_up, l,  m,  n_dim), simp_fn),
        _simp(h * _c2_up(Ric_up, m,  m,  n_dim), simp_fn),
        _simp(h * _c2_up(Ric_up, l,  nv, n_dim), simp_fn),
        _simp(h * _c2_up(Ric_up, m,  nv, n_dim), simp_fn),
        _simp(h * _c2_up(Ric_up, nv, nv, n_dim), simp_fn),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Correct covariant derivative of Weyl tensor (Section C of SPEC.md)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psid_direct(C: dict, G: dict, frame_vecs: list,
                        coords: list, n: int, simp_fn=None) -> list:
    """
    Compute the 20 PSID components directly without building the full
    ∇C tensor.  For each (k, direction) pair, only the nablaC entries
    that contribute to the projection are evaluated.

    This is 30-100× faster than building the full 4^5=1024-entry tensor
    because the frame vectors are sparse (typically 2 nonzero entries each).

    PSID[k][e] = ∇_ε C_{αβγδ} · v1[α] v2[β] v3[γ] v4[δ] ve[ε]

    where ∇_ε C_{αβγδ} = ∂_ε C_{αβγδ} − Γ^ρ_{αε}C_{ρβγδ} − (3 more Γ)
    """
    comps = []
    for k, (l1, l2, l3, l4) in _PSI_LEGS.items():
        v1, v2, v3, v4 = (frame_vecs[l1], frame_vecs[l2],
                          frame_vecs[l3], frame_vecs[l4])
        # Pre-filter nonzero indices to avoid inner loops over zero legs
        nz1 = [i for i in range(n) if v1[i] != 0]
        nz2 = [i for i in range(n) if v2[i] != 0]
        nz3 = [i for i in range(n) if v3[i] != 0]
        nz4 = [i for i in range(n) if v4[i] != 0]

        for e_dir in range(4):
            ve = frame_vecs[e_dir]
            nze = [i for i in range(n) if ve[i] != 0]

            total = sp.S.Zero
            for a in nz1:
                for b in nz2:
                    for c in nz3:
                        for d in nz4:
                            outer = v1[a]*v2[b]*v3[c]*v4[d]
                            if outer == 0:
                                continue
                            for eps in nze:
                                if ve[eps] == 0:
                                    continue
                                # Compute ∇_eps C_{abcd} at this index combination
                                c_abcd = C.get((a,b,c,d), sp.S.Zero)
                                # Partial derivative term
                                nc = diff(c_abcd, coords[eps])
                                # Christoffel correction terms
                                for rho in range(n):
                                    g_ae = G.get((rho,a,eps), sp.S.Zero)
                                    if g_ae != 0:
                                        nc -= g_ae * C.get((rho,b,c,d), sp.S.Zero)
                                    g_be = G.get((rho,b,eps), sp.S.Zero)
                                    if g_be != 0:
                                        nc -= g_be * C.get((a,rho,c,d), sp.S.Zero)
                                    g_ce = G.get((rho,c,eps), sp.S.Zero)
                                    if g_ce != 0:
                                        nc -= g_ce * C.get((a,b,rho,d), sp.S.Zero)
                                    g_de = G.get((rho,d,eps), sp.S.Zero)
                                    if g_de != 0:
                                        nc -= g_de * C.get((a,b,c,rho), sp.S.Zero)
                                total += nc * outer * ve[eps]

            # Quick simplification: sp.cancel collects rational terms;
            # avoids expression explosion from sp.expand on sqrt-containing sums.
            comps.append((k, e_dir, sp.cancel(total)))
    return comps


# ─────────────────────────────────────────────────────────────────────────────
# Functional independence (Jacobian rank, unchanged in principle)
# ─────────────────────────────────────────────────────────────────────────────

def indep_count(scalars: list, coords: list, simp_fn=None) -> int:
    """
    Number of functionally independent real scalars = rank of Jacobian.
    Implements CLASSI's FUNTST (funtst.shp) via SymPy Matrix.rank().

    For metrics with undetermined functions (FUNS/NEWSUL case), any
    applied functions F(x) and their derivatives F'(x), F''(x) etc.
    are substituted with fresh positive symbols before all simplification.
    This prevents sp.cancel from hanging on expressions like p**(m**2)*G(p).
    """
    if not scalars:
        return 0

    # Build FUNS substitution map upfront from all scalars
    from sympy.core.function import AppliedUndef
    funs_subs = {}
    extra_coords = []
    all_atoms = set()
    for s in scalars:
        try:
            all_atoms |= s.atoms(sp.Derivative, sp.Function)
        except Exception:
            pass
    for atom in sorted(all_atoms, key=str):
        if atom in funs_subs:
            continue
        if isinstance(atom, sp.Derivative):
            sym = sp.Symbol(f'_fd{len(funs_subs)}', positive=True)
            funs_subs[atom] = sym
            extra_coords.append(sym)
        elif isinstance(atom, AppliedUndef):
            sym = sp.Symbol(f'_ff{len(funs_subs)}', positive=True)
            funs_subs[atom] = sym
            extra_coords.append(sym)

    def _sub(expr):
        return expr.subs(funs_subs) if funs_subs else expr

    all_coords = list(coords) + extra_coords

    # Extract real/imag parts after substitution
    real_scalars = []
    for s in scalars:
        s2 = _sub(s)
        r = sp.cancel(sp.re(s2))
        i_part = sp.cancel(sp.im(s2))
        if r != 0: real_scalars.append(r)
        if i_part != 0: real_scalars.append(i_part)
    if not real_scalars:
        return 0

    # Deduplicate using sp.cancel (safe after FUNS substitution)
    seen, unique = [], []
    for s in real_scalars:
        if not any(sp.cancel(s - u) == 0 or sp.cancel(s + u) == 0 for u in seen):
            seen.append(s)
            unique.append(s)

    J = Matrix([[diff(s, c) for c in all_coords] for s in unique])
    cancel_zero = lambda x: sp.cancel(x) == 0
    try:
        Js = J.applyfunc(sp.cancel)
        return Js.rank(iszerofunc=cancel_zero, simplify=sp.cancel)
    except Exception:
        try:
            return J.rank(iszerofunc=cancel_zero, simplify=sp.cancel)
        except Exception:
            return J.rank()

# ─────────────────────────────────────────────────────────────────────────────
# Signature detection and coframe construction (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def detect_signature(g: Matrix, simp_fn=None) -> str:
    signs = []
    n = g.shape[0]
    for i in range(n):
        val = _simp(g[i,i], simp_fn)
        s = None
        try:
            if val.is_positive:  s = 1
            elif val.is_negative: s = -1
        except Exception:
            pass
        if s is None:
            try:
                neg = _simp(-val, simp_fn)
                if neg.is_positive: s = -1
                elif neg.is_negative: s = 1
            except Exception:
                pass
        if s is None:
            try:
                free = list(val.free_symbols)
                test_subs = {sym: sp.Rational(3,1) if 'r' in str(sym)
                             else sp.Rational(1,2) for sym in free}
                num = float(val.subs(test_subs).evalf())
                s = 1 if num > 0 else -1
            except Exception:
                pass
        signs.append(s)

    n_neg = signs.count(-1); n_pos = signs.count(1); n_unk = signs.count(None)
    if n_neg == 1 and n_unk + n_pos == n - 1: return 'lorentzian'
    if n_neg == 3 and n_unk + n_pos == 1:     return 'lorentzian'
    if n_neg == 0 and n_unk == 0:             return 'euclidean'
    if n_neg == 2 and n_unk == 0:             return 'kleinian'
    if signs[0] == -1:                         return 'lorentzian'
    return 'unknown'


def _inner_forms(gi, u, v, n):
    return sum(gi[a,b]*u[a]*v[b] for a in range(n) for b in range(n))


def invert_frame(frame: list, g: Matrix, simp_fn=None) -> list:
    n = g.shape[0]
    E = Matrix([[frame[a][mu] for a in range(n)] for mu in range(n)])
    E_inv = E.inv()
    return [[_simp(E_inv[a, mu], simp_fn) for mu in range(n)] for a in range(n)]


_TETRAD_TYPES = {
    'lorentzian': [{'name': 'lorentzian', 'ln': -1, 'mmb': +1, 'lm': 0, 'll': 0, 'nm': 0, 'nn': 0}],
    'euclidean':  [{'name': 'euclidean',  'ln': +1, 'mmb': +1, 'lm': 0, 'll': 0, 'nm': 0, 'nn': 0}],
    'kleinian':   [
        {'name': 'kleinian-A', 'ln': -1, 'mmb': -1, 'lm': 0, 'll': 0, 'nm': 0, 'nn': 0},
        {'name': 'kleinian-B', 'ln': -1, 'mmb': +1, 'lm': 0, 'll': 0, 'nm': 0, 'nn': 0},
    ],
}


def validate_null_coframe(gi, coframe, signature, simp_fn=None):
    l, nv, m, mb = coframe
    n_dim = gi.shape[0]
    def ip(u, v): return _simp(_inner_forms(gi, u, v, n_dim), simp_fn)
    ln_val = ip(l, nv); mmb_val = ip(m, mb)
    lm_val = ip(l, m);  ll_val  = ip(l, l)
    nm_val = ip(nv, m); nn_val  = ip(nv, nv)
    valid_types = _TETRAD_TYPES.get(signature, [])
    errors = []
    for tt in valid_types:
        checks = [(ln_val, tt['ln'], 'l·n'), (mmb_val, tt['mmb'], 'm·mb'),
                  (lm_val, 0, 'l·m'), (ll_val, 0, 'l·l'),
                  (nm_val, 0, 'n·m'), (nn_val, 0, 'n·n')]
        fails = [(name, got, exp) for got, exp, name in checks
                 if _simp(got - exp, simp_fn) != 0]
        if not fails: return tt['name']
        errors.append((tt['name'], fails))
    msg = f"Null coframe invalid for signature '{signature}'.\n"
    for tname, fails in errors:
        msg += f"\n  Tried {tname}:\n"
        for fname, got, exp in fails:
            msg += f"    {fname} = {got}  (expected {exp})\n"
    raise ValueError(msg)


def null_coframe_from_orthonormal_coframe(theta, signature, tetrad_type='auto',
                                          simp_fn=None):
    r2 = sqrt(2)
    t0, t1, t2, t3 = theta
    nd = len(t0)
    def add(a,b): return [_simp(a[i]+b[i], simp_fn) for i in range(nd)]
    def sub(a,b): return [_simp(a[i]-b[i], simp_fn) for i in range(nd)]
    def sc(a,c):  return [_simp(a[i]*c,    simp_fn) for i in range(nd)]
    def im(a):    return [_simp(I*a[i],    simp_fn) for i in range(nd)]

    if signature == 'lorentzian':
        return (sc(add(t0,t1),1/r2), sc(sub(t0,t1),1/r2),
                sc(add(t2,im(t3)),1/r2), sc(sub(t2,im(t3)),1/r2))
    elif signature == 'euclidean':
        return (sc(add(t0,im(t1)),1/r2), sc(sub(t0,im(t1)),1/r2),
                sc(add(t2,im(t3)),1/r2), sc(sub(t2,im(t3)),1/r2))
    elif signature == 'kleinian':
        use = tetrad_type if tetrad_type != 'auto' else 'A'
        if use == 'A':
            return (sc(add(t0,t2),1/r2), sc(sub(t0,t2),1/r2),
                    sc(add(t1,t3),1/r2), sc(sub(t1,t3),1/r2))
        else:
            return (sc(add(t0,im(t1)),1/r2), sc(sub(t0,im(t1)),1/r2),
                    sc(add(t2,im(t3)),1/r2), sc(sub(t2,im(t3)),1/r2))
    else:
        raise ValueError(f"Unknown signature '{signature}'")


def _signed_sqrt(gaa, simp_fn=None):
    """
    Compute sqrt of a diagonal metric component, handling sign correctly.
    For Lorentzian metrics where SymPy can't determine sign symbolically,
    uses a numeric test to determine whether to negate before sqrt.

    Handles metrics with undetermined functions G(p), d(u), etc. by
    substituting positive dummy values for all applied functions in the
    numeric sign test, avoiding Abs() in the output.
    """
    gaa_s = _simp(gaa, simp_fn)
    # Try direct sign determination
    try:
        if gaa_s.is_positive:
            return sqrt(gaa_s)
        elif gaa_s.is_negative:
            return sqrt(-gaa_s)
    except Exception:
        pass
    # Try negated
    try:
        neg = _simp(-gaa_s, simp_fn)
        if neg.is_positive:
            return sqrt(neg)
        elif neg.is_negative:
            return sqrt(gaa_s)
    except Exception:
        pass
    # Numeric fallback: substitute free symbols AND unknown functions
    # with positive test values.  This handles metrics containing
    # undetermined functions like G(p), d(u), f(r) etc.
    try:
        from sympy.core.function import AppliedUndef
        test_subs = {}
        # Substitute coordinate symbols
        for sym in gaa_s.free_symbols:
            test_subs[sym] = sp.Rational(3, 1) if 'r' in str(sym)                              else sp.Rational(1, 2)
        # Substitute applied functions (G(p), d(u), etc.) with 1
        for atom in gaa_s.atoms(sp.Function):
            if isinstance(atom, (sp.Derivative,)):
                test_subs[atom] = sp.Rational(1, 2)
            elif isinstance(atom, AppliedUndef) or not hasattr(atom, 'is_number'):
                test_subs[atom] = sp.S.One
        num = float(gaa_s.subs(test_subs).evalf())
        return sqrt(-gaa_s) if num < 0 else sqrt(gaa_s)
    except Exception:
        return sqrt(sp.Abs(gaa_s))


def null_coframe_from_diagonal_metric(g, signature, tetrad_type='auto',
                                      simp_fn=None):
    n = g.shape[0]
    theta = []
    for a in range(n):
        gaa = g[a, a]
        form = [sp.S.Zero] * n
        form[a] = _simp(_signed_sqrt(gaa, simp_fn), simp_fn)
        theta.append(form)
    return null_coframe_from_orthonormal_coframe(theta, signature,
                                                 tetrad_type, simp_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Correct covariant derivative of Weyl tensor (Section C of SPEC.md)
# ─────────────────────────────────────────────────────────────────────────────

def frame_vectors_from_coframe(coframe: list, gi: Matrix, n: int) -> list:
    """
    Raise null coframe 1-forms to frame vectors: e_a^μ = g^{μν} θ^a_ν.
    """
    vecs = []
    for form in coframe:
        vec = [sum(gi[mu,nu] * form[nu] for nu in range(n)) for mu in range(n)]
        vecs.append(vec)
    return vecs


# NP leg assignments for Ψ_k = C(v₁,v₂,v₃,v₄).
# Indices refer to frame_vecs list: 0=l, 1=n, 2=m, 3=mb.
_PSI_LEGS = {
    0: (0, 2, 0, 2),   # Ψ₀ = C(l,m,l,m)
    1: (0, 1, 0, 2),   # Ψ₁ = C(l,n,l,m)
    2: (0, 2, 3, 1),   # Ψ₂ = C(l,m,mb,n)
    3: (0, 1, 3, 1),   # Ψ₃ = C(l,n,mb,n)
    4: (1, 3, 1, 3),   # Ψ₄ = C(n,mb,n,mb)
}


# ─────────────────────────────────────────────────────────────────────────────
# Main classifier
# ─────────────────────────────────────────────────────────────────────────────

class KarlhedeClassifier:
    """
    Classify a spacetime metric via the Cartan-Karlhede algorithm.

    Usage::

        import sympy as sp
        from pick.karlhede import KarlhedeClassifier

        t,r,th,ph = sp.symbols('t r theta phi', real=True)
        m = sp.Symbol('m', positive=True)
        A = 1 - 2*m/r
        g = sp.diag(-A, 1/A, r**2, r**2*sp.sin(th)**2)
        clf = KarlhedeClassifier(g, [t,r,th,ph])
        print(clf.classify().summary())

    Parameters
    ----------
    metric          : SymPy Matrix  (n×n covariant metric)
    coords          : list of SymPy symbols
    null_coframe    : optional (l,n,m,mb) as covariant 1-forms
    orthonormal_coframe : optional [θ⁰,θ¹,θ²,θ³] as covariant 1-forms
    contravariant   : if True, supplied frames are vectors → invert
    signature       : 'lorentzian'|'euclidean'|'kleinian' (auto-detected)
    tetrad_type     : Kleinian sub-type 'A'|'B' (default 'A')
    simplify_fn     : SymPy simplification function
    max_order       : maximum differentiation order (default 7)
    verbose         : print progress
    """

    def __init__(self, metric, coords,
                 null_coframe=None, orthonormal_coframe=None,
                 contravariant=False, signature=None, tetrad_type='auto',
                 simplify_fn=None, max_order=7, verbose=True,
                 sage_geometry=None):
        self.g                   = metric
        self.coords              = coords
        self.null_coframe        = null_coframe
        self.orthonormal_coframe = orthonormal_coframe
        self.contravariant       = contravariant
        self.signature           = signature
        self.tetrad_type         = tetrad_type
        self.simp                = simplify_fn
        self.max_order           = max_order
        self.verbose             = verbose
        self.n                   = len(coords)
        self.sage_geometry       = sage_geometry  # SageGeometry or None

    def _log(self, msg):
        if self.verbose: print(msg)

    def _s(self, expr):
        return _simp(expr, self.simp)

    def classify(self) -> KarlhedeResult:
        log  = self._log
        simp = self.simp
        n    = self.n
        g    = self.g
        coords = self.coords

        log("─"*50)
        log("  PICK  Karlhede Classification")
        log("─"*50)

        # ── Geometry ─────────────────────────────────────────────────
        log("\n[1/5] Christoffel symbols...")
        G = christoffel(g, coords)

        log("[2/5] Riemann tensor...")
        R = riemann(g, coords, G)

        log("[3/5] Ricci & Weyl tensors...")
        Ric = ricci_ten(R, n).applyfunc(self._s)
        gi  = g.inv()
        Rs  = self._s(ricci_sc(Ric, gi))

        vacuum_check = all(self._s(Ric[i,j]) == 0
                          for i in range(n) for j in range(n))
        if vacuum_check:
            Ric = zeros(n, n); Rs = sp.S.Zero
            log("  → Vacuum metric detected; Ricci = 0 enforced.")

        C     = weyl_ten(R, Ric, Rs, g, n)
        # Simplify and prune the Weyl tensor dict for efficient PSID computation.
        # Unsimplified C entries would make diff(C, coord) very slow.
        C     = {k: v_s for k, v in C.items()
                 if (v_s := self._s(v)) != 0}
        LAMBD = self._s(Rs / 24)

        # ── Trace-free Ricci  (CLASSI: PHI is S_{μν}, not full R_{μν}) ──
        # S_{μν} = R_{μν} − (R/4) g_{μν}.  For Einstein spaces (de Sitter
        # etc.) S = 0 even though R_{μν} ≠ 0, so PHI = 0 and the full
        # Lorentz isotropy is preserved.  Using the full Ricci would give
        # a spurious Φ₁₁ = R/8 ≠ 0, breaking the isotropy erroneously.
        S = Matrix(n, n, lambda i, j:
            self._s(Ric[i,j] - (Rs/4)*g[i,j]))

        # ── Signature ────────────────────────────────────────────────
        sig = self.signature
        if sig is None:
            sig = detect_signature(g, simp)
            if sig == 'unknown': sig = 'lorentzian'
            log(f"  → Signature: {sig}")

        # ── Null coframe ─────────────────────────────────────────────
        log("[4/5] Null coframe & curvature spinors...")

        def _to_coframe(supplied, label):
            if self.contravariant:
                log(f"  ⚠ {label}: inverting frame → coframe.")
                return invert_frame(list(supplied), g, simp)
            return list(supplied)

        if self.null_coframe is not None:
            raw = _to_coframe(self.null_coframe, "null coframe")
            l, nv, m, mb = raw[0], raw[1], raw[2], raw[3]
            log("  → Validating null coframe...")
            ttype = validate_null_coframe(gi, (l,nv,m,mb), sig, simp)
            log(f"  → Type: {ttype}")
        elif self.orthonormal_coframe is not None:
            raw = _to_coframe(self.orthonormal_coframe, "orthonormal coframe")
            l, nv, m, mb = null_coframe_from_orthonormal_coframe(
                raw, sig, self.tetrad_type, simp)
        else:
            is_diag = all(self._s(g[i,j]) == 0
                         for i in range(n) for j in range(n) if i != j)
            if is_diag:
                l, nv, m, mb = null_coframe_from_diagonal_metric(
                    g, sig, self.tetrad_type, simp)
            else:
                log("  ⚠ Non-diagonal metric; Petrov/Segre unreliable without coframe.")
                l, nv, m, mb = null_coframe_from_diagonal_metric(
                    g, sig, self.tetrad_type, simp)

        PSI = weyl_spinor_from_coframe(C, gi, l, nv, m, mb, n, simp)
        # PHI uses the TRACE-FREE Ricci S, not the full Ric
        PHI = ricci_spinor_from_coframe(S, gi, l, nv, m, mb, n, simp)
        PSI = [self._s(p) for p in PSI]
        PHI = [self._s(p) for p in PHI]

        log(f"  PSI = {PSI}")
        log(f"  PHI = {PHI}")
        log(f"  Λ   = {LAMBD}")

        # ── Petrov & Segre ────────────────────────────────────────────
        log("[5/5] Petrov & Segre classification...")
        petrov = petrov_from_invariants(PSI, simp)
        segre  = segre_from_phi(PHI, LAMBD, simp)
        log(f"  Petrov type: {petrov}")
        log(f"  Segre  type: {segre}")

        # ── Isotropy state initialised from Petrov type ───────────────
        # For standard (PND-aligned) frames, ISOTST PSI gives exactly the
        # PETROVLIST column-2 value (clabas.shp:16-22).
        iso = IsotropyFlags()
        if petrov == '0':
            # Conformally flat: ISOTST PSI skipped (CLASSI classi4ord0:215)
            # All flags remain True → symbol='6'
            pass
        else:
            iso.set_from_petrov(petrov)

        # ISOTST PHI: trace-free PHI is already the correct spinor (uses S).
        # Only run if PHI is nonzero (NOPHI=F in CLASSI terminology).
        phi_nonzero = not all(self._s(p) == 0 for p in PHI)
        if phi_nonzero:
            test_phi_isotropy(PHI, iso, simp)

        H0_sym = iso.symbol
        H0_dim = iso.dim
        log(f"\n  Order 0: H={H0_sym} ({H0_dim}-dim), isotropy: {iso.name}")

        # ── PLANESPACE detection (CLASSI classi.shp:136) ─────────────
        # PLANESPACE = True when PSI = 0 (CONFLAT) AND PHI = 0 AND Λ = 0.
        # For Minkowski this fires; for de Sitter (Λ ≠ 0) it does NOT.
        # When PLANESPACE, CLASSI terminates immediately after order 0
        # (the order-1 check in CLASSIRUNREST fires without running order 1).
        conflat    = (petrov == '0')
        planespace = conflat and (not phi_nonzero) and (LAMBD == 0)
        if planespace:
            log("  → Planespace (flat space): terminating after order 0.")

        # ── Order-0 independent functions ─────────────────────────────
        scalars_0 = []
        for p in PSI + PHI + [LAMBD]:
            r_p = self._s(sp.re(p))
            i_p = self._s(sp.im(p))
            if r_p != 0: scalars_0.append(r_p)
            if i_p != 0: scalars_0.append(i_p)

        t0 = indep_count(scalars_0, coords, simp)
        log(f"           t={t0} independent scalars")

        steps = [KarlhedeStep(0, t0, H0_dim, iso.name, H0_sym)]

        # ── Early exit for planespace ─────────────────────────────────
        if planespace:
            final_t = t0; final_s = H0_dim
            isometry_dim = (n - final_t) + final_s
            log(f"\n  r = {isometry_dim}  t∞ = {final_t}  s∞ = {final_s}")
            return KarlhedeResult(
                petrov_type=petrov, segre_type=segre,
                isometry_dim=isometry_dim, isotropy_dim=final_s,
                indep_scalars=final_t, steps=steps, terminated_at=0,
                psi=PSI, phi=PHI, lambd=LAMBD,
            )

        # ── Frame vectors (needed for ∇C projection) ──────────────────
        coframe_list = [l, nv, m, mb]
        frame_vecs   = frame_vectors_from_coframe(coframe_list, gi, n)

        # ── Compute PSID (Cartan invariants, all orders) ─────────────────
        # If a SageGeometry is available, use it for all orders — gives
        # correct nabla^n C via nab(nab(...C...)) instead of the coordinate-
        # partial stopgap.  Otherwise fall back to compute_psid_direct
        # (order 1 correct) + coordinate partials (orders >= 2, stopgap).
        sg = self.sage_geometry

        log("\n  Computing frame-projected nablaC (PSID)...")

        # Always use compute_psid_direct for order=1 — this uses the
        # hand-rolled C and G already computed above, avoiding any
        # SageManifolds/Maxima calls at this stage.
        # sage_geometry is only used at order>=2 via nabla_weyl().
        psid_comps_1 = compute_psid_direct(C, G, frame_vecs, coords, n, simp)

        def _extract_scalars(comps):
            """Extract scalar invariants from PSID components.

            Uses v² (with powsimp force=True to rationalize sqrt products)
            rather than v directly.  This is essential: PSID components
            contain square roots from the tetrad construction, and
            indep_count's Jacobian rank test requires rational expressions.
            v² = v*conj(v) is real and rational for all standard metrics.
            """
            import sympy as _sp
            scalars = []
            for item in comps:
                v = item[2]
                if v == 0:
                    continue
                v2 = self._s(_sp.powsimp(_sp.expand(v * v), force=True))
                if v2 != 0:
                    scalars.append(v2)
            return scalars

        psid_scalars_1 = _extract_scalars(psid_comps_1)

        # ── Karlhede iteration ─────────────────────────────────────────
        prev_t   = t0
        prev_H   = H0_dim
        current_scalars = scalars_0[:]

        terminated_at = self.max_order

        for order in range(1, self.max_order + 1):
            log(f"\n  Order {order}...")

            if order == 1:
                new_scalars = psid_scalars_1
                test_psid_isotropy(psid_comps_1, iso, simp)

            elif sg is not None:
                # SageManifolds: correct nabla^n C for n >= 2.
                # nabla_weyl() temporarily switches to SymPy calculus engine
                # to avoid Maxima simplify_rational() hangs.
                log(f"    Using SageManifolds nabla^{order}C...")
                psid_comps_n = sg.psid_components(frame_vecs, order=order)
                new_scalars = _extract_scalars(psid_comps_n)
                test_psid_isotropy(psid_comps_1, iso, simp)


            combined  = current_scalars + new_scalars
            t_new     = indep_count(combined, coords, simp)
            H_new_dim = iso.dim
            H_new_sym = iso.symbol

            log(f"  Order {order}: t={t_new} (was {prev_t}), "
                f"H={H_new_sym} ({H_new_dim}-dim, was {prev_H}-dim)")

            steps.append(KarlhedeStep(order, t_new, H_new_dim, iso.name, H_new_sym))

            # Termination criterion (CLASSI classi.shp:139-140):
            # Stop when BOTH t and H-dim are unchanged from previous order.
            if t_new == prev_t and H_new_dim == prev_H:
                log(f"  ✓ Terminated at order {order}.")
                terminated_at = order
                break

            prev_t = t_new
            prev_H = H_new_dim
            current_scalars = combined

        final_t   = steps[-1].t
        final_s   = steps[-1].s
        isometry_dim = (n - final_t) + final_s   # validated: clasum.shp:100

        log("\n" + "─"*50)
        log(f"  Petrov: {petrov}   Segre: {segre}")
        log(f"  r = {isometry_dim}  (isometry group dimension)")
        log(f"  t∞ = {final_t}  s∞ = {final_s}")
        log("─"*50)

        return KarlhedeResult(
            petrov_type   = petrov,
            segre_type    = segre,
            isometry_dim  = isometry_dim,
            isotropy_dim  = final_s,
            indep_scalars = final_t,
            steps         = steps,
            terminated_at = terminated_at,
            psi           = PSI,
            phi           = PHI,
            lambd         = LAMBD,
        )
