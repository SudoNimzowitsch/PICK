"""
pick/sage_backend.py  —  SageManifolds backend for PICK (Phase 1)

Converts a SageMath metric definition into the SymPy dict/Matrix objects
that the existing karlhede.py algorithm layer already consumes.  This is a
pure adapter: no algorithm code lives here, nothing in karlhede.py changes.

Architecture
------------
SageManifolds provides:
  - g.riemann()          → Riemann tensor (any signature)
  - g.weyl()             → Weyl tensor
  - g.ricci()            → Ricci tensor
  - nab = g.connection() → Levi-Civita connection
  - nab(T)               → covariant derivative of any tensor field T
  - nab(nab(T))          → ∇²T, etc.

We use set_calculus_method('sympy') so every scalar expression is a native
SymPy Expr.  Components are extracted via T[i,j,...].expr(), giving the
exact same type as the hand-rolled functions in karlhede.py.

The adapter produces:
  SageGeometry.g_matrix   : sympy.Matrix  (covariant metric)
  SageGeometry.gi_matrix  : sympy.Matrix  (inverse metric)
  SageGeometry.G_dict     : dict {(s,m,v): expr}  (Christoffel)
  SageGeometry.R_dict     : dict {(r,s,m,v): expr} (Riemann, lower-first)
  SageGeometry.C_dict     : dict {(a,b,c,d): expr} (Weyl, pruned)
  SageGeometry.Ric_matrix : sympy.Matrix  (Ricci)
  SageGeometry.Rs_expr    : sympy.Expr    (Ricci scalar)
  SageGeometry.coords     : list of sympy.Symbol

  SageGeometry.nabla_weyl(order) → list of (k, e_dir, expr) in the same
                                    format as compute_psid_direct() output

Usage
-----
    from sage.manifolds.manifold import Manifold
    from pick.sage_backend import SageGeometry, sage_metric_from_matrix

    # Option A: build from a SymPy metric Matrix (for drop-in with metrics.py)
    import sympy as sp
    from pick.metrics import METRICS
    m = METRICS['schwar']()
    sg = sage_metric_from_matrix(m['g'], m['coords'], signature='lorentzian',
                                 simp_fn=m.get('simp'))

    # Option B: define natively in SageMath
    M = Manifold(4, 'M', structure='Lorentzian')
    X.<t,r,th,ph> = M.chart(r't r:(0,+oo) th:(0,pi):\\theta ph:(0,2*pi):\\phi')
    g = M.metric('g')
    g[0,0] = -(1 - 2*M_/r); ...
    sg = SageGeometry(g, X)

    # Either way, feed into the existing classifier:
    from pick.karlhede import KarlhedeClassifier
    clf = KarlhedeClassifier(sg.g_matrix, sg.coords)

Compatibility guarantee
-----------------------
If SageMath is not installed, importing this module raises ImportError with a
clear message.  All other pick/ modules are unaffected.
"""

from __future__ import annotations

import sympy as sp
from sympy import Matrix, Rational, diff, zeros
from typing import Optional, Callable, List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# SageMath import guard
# ─────────────────────────────────────────────────────────────────────────────

try:
    from sage.all import Manifold, var, SR
    from sage.manifolds.manifold import TopologicalManifold
    _SAGE_AVAILABLE = True
except ImportError:
    _SAGE_AVAILABLE = False
    _SAGE_IMPORT_ERROR = (
        "SageMath is not installed.  Install via conda:\n"
        "  conda install -c conda-forge sage\n"
        "or download from https://www.sagemath.org/\n"
        "The SymPy backend (karlhede.py) works without SageMath."
    )


def _require_sage():
    if not _SAGE_AVAILABLE:
        raise ImportError(_SAGE_IMPORT_ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# Signature map: PICK string → SageManifolds integer (n_+ − n_−)
# ─────────────────────────────────────────────────────────────────────────────

_SIGNATURE_MAP = {
    'lorentzian':  2,   # (−,+,+,+) → 4−2×1 = 2
    'kleinian':    0,   # (−,−,+,+) → 4−2×2 = 0
    'euclidean':   4,   # (+,+,+,+) → 4
}

# Inverse: SageManifolds integer → PICK string (for 4-dim)
_SIG_TO_PICK = {2: 'lorentzian', 0: 'kleinian', 4: 'euclidean'}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract a SymPy expression from a SageManifolds scalar
# ─────────────────────────────────────────────────────────────────────────────

def _to_sympy(sage_expr, simp_fn=None, sym_map=None):
    # NEVER use sage_expr == 0 to test for zero — on SR engine expressions
    # this triggers Maxima's trigsimp/simplify_full which hangs.
    # Instead use the string representation or type name.
    try:
        type_name = type(sage_expr).__name__
        # Fast path: known zero sentinel types
        if type_name in ('Zero', 'ZeroElement'):
            return sp.S.Zero
        # Check string representation without triggering Maxima
        s_check = str(sage_expr)
        if s_check == '0':
            return sp.S.Zero
    except Exception:
        return sp.S.Zero
    try:
        s = str(sage_expr.expr()).replace(chr(94), chr(42)+chr(42)); import re as _re; s = _re.sub(r"(?<![a-zA-Z0-9_])e\*\*\(([^)]+)\)", r"exp(\1)", s); s = _re.sub(r"(?<![a-zA-Z0-9_])e\*\*([0-9a-zA-Z_]+)", r"exp(\1)", s)
        local = {k: v for k, v in sym_map.items()} if sym_map else {}
        e = sp.sympify(s, locals=local)
    except Exception:
        try:
            s = str(sage_expr).replace(chr(94), chr(42)+chr(42)); s = _re.sub(r"(?<![a-zA-Z0-9_])e\*\*\(([^)]+)\)", r"exp(\1)", s); s = _re.sub(r"(?<![a-zA-Z0-9_])e\*\*([0-9a-zA-Z_]+)", r"exp(\1)", s)
            local = {k: v for k, v in sym_map.items()} if sym_map else {}
            e = sp.sympify(s, locals=local)
        except Exception:
            return sp.S.Zero
    if sym_map:
        e = e.subs([(sp.Symbol(k), v) for k, v in sym_map.items()])
    if simp_fn is not None:
        e = simp_fn(e)
    return e


def _to_sympy_zero_if_small(sage_expr, simp_fn=None) -> sp.Expr:
    """Like _to_sympy but returns sp.S.Zero for expressions that simplify
    to zero.  Used when building sparse dicts."""
    e = _to_sympy(sage_expr, simp_fn)
    if e == 0:
        return sp.S.Zero
    return e


# ─────────────────────────────────────────────────────────────────────────────
# SageGeometry: the adapter object
# ─────────────────────────────────────────────────────────────────────────────

class SageGeometry:
    """
    Wraps a SageManifolds pseudo-Riemannian metric and exposes the tensors
    that KarlhedeClassifier needs as SymPy objects.

    Parameters
    ----------
    sage_metric :
        A SageManifolds PseudoRiemannianMetric object (g).
    chart :
        The coordinate chart on which to extract components.
    simp_fn : callable, optional
        Applied to every extracted SymPy expression.  Defaults to
        sp.cancel (fast rational simplification).
    """

    def __init__(self, sage_metric, chart, simp_fn=None, manifold=None, sym_map=None):
        _require_sage()

        self._g  = sage_metric
        self._ch = chart
        self._simp = simp_fn if simp_fn is not None else sp.cancel

        # The manifold must be passed explicitly (SageMath 10.x metric objects
        # do not expose a back-reference to their manifold)
        if manifold is None:
            raise ValueError(
                "SageGeometry requires the manifold to be passed explicitly "
                "as manifold=M.  The metric object does not carry a back-ref "                "in SageMath 10.x.")
        self._n = manifold.dim()
        try:
            self._sig_int = manifold.signature()
        except Exception:
            self._sig_int = 2  # default Lorentzian


        # Extract coordinate symbols from the chart
        self.coords: List[sp.Symbol] = [
            sp.Symbol(str(c)) for c in chart[:]
        ]

        # Lazily-computed tensors (computed on first access)
        self._g_matrix:   Optional[Matrix]        = None
        self._gi_matrix:  Optional[Matrix]        = None
        self._G_dict:     Optional[Dict]          = None
        self._Ric_matrix: Optional[Matrix]        = None
        self._Rs_expr:    Optional[sp.Expr]       = None
        self._C_dict:     Optional[Dict]          = None
        self._nabC:       Optional[object]        = None   # SageManifolds tensor
        self._sympy_sym_map = sym_map or {}
        self._nabC_cache: Dict[int, List]         = {}     # order → component list

    # ── metric ────────────────────────────────────────────────────────────────

    @property
    def g_matrix(self):
        if self._g_matrix is None:
            n = self._n
            comps = self._g.comp()
            def _gc(i, j):
                try:
                    return self._simp(_to_sympy(comps[i, j], sym_map=self._sympy_sym_map))
                except Exception:
                    return sp.S.Zero
            self._g_matrix = Matrix(n, n, _gc)
        return self._g_matrix

    @property
    def gi_matrix(self) -> Matrix:
        if self._gi_matrix is None:
            self._gi_matrix = self.g_matrix.inv()
        return self._gi_matrix

    # ── Christoffel (Γ^σ_{μν}) ───────────────────────────────────────────────

    @property
    def G_dict(self):
        if self._G_dict is None:
            n = self._n
            G = {}
            comps = self._g.connection().coef()
            for s in range(n):
                for mu in range(n):
                    for nu in range(n):
                        try:
                            v = self._simp(_to_sympy(comps[s, mu, nu], sym_map=self._sympy_sym_map))
                        except Exception:
                            v = sp.S.Zero
                        if v != 0:
                            G[s, mu, nu] = v
            self._G_dict = G
        return self._G_dict

    # ── Riemann (R^r_{smv}) ──────────────────────────────────────────────────

    @property
    def R_dict(self) -> Dict:
        """
        Returns R_{rsmv} (all lower indices) in PICK's convention.
        PICK uses R[r,s,m,v] = R^r_{smv} contracted with g to lower
        the first index.  Actually, PICK's riemann() returns the
        (1,3) tensor R^r_{smv} directly.  We match that.
        """
        if not hasattr(self, '_R_dict') or self._R_dict is None:
            n = self._n
            sage_Riem = self._nab.riemann()  # (1,3) tensor R^a_{bcd}
            R = {}
            for r in range(n):
                for s in range(n):
                    for m in range(n):
                        for v in range(n):
                            val = self._simp(
                                _to_sympy(sage_Riem[r, s, m, v]))
                            if val != 0:
                                R[r, s, m, v] = val
            self._R_dict = R
        return self._R_dict

    # ── Ricci ─────────────────────────────────────────────────────────────────

    @property
    def Ric_matrix(self):
        if self._Ric_matrix is None:
            n = self._n
            comps = self._g.ricci().comp()
            def _rc(i, j):
                try:
                    return self._simp(_to_sympy(comps[i, j], sym_map=self._sympy_sym_map))
                except Exception:
                    return sp.S.Zero
            self._Ric_matrix = Matrix(n, n, _rc)
        return self._Ric_matrix

    @property
    def Rs_expr(self):
        if self._Rs_expr is None:
            self._Rs_expr = self._simp(_to_sympy(self._g.ricci_scalar(), sym_map=self._sympy_sym_map))
        return self._Rs_expr

    # ── Weyl ──────────────────────────────────────────────────────────────────

    @property
    def C_dict(self):
        if self._C_dict is None:
            n = self._n
            comps = self._g.weyl().comp()
            gm = self.g_matrix
            raw = {}
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        for d in range(n):
                            try:
                                val = _to_sympy(comps[a,b,c,d], sym_map=self._sympy_sym_map)
                                if val != 0:
                                    raw[a,b,c,d] = self._simp(val)
                            except Exception:
                                pass
            import sympy as _sp
            C = {}
            for e in range(n):
                for b in range(n):
                    for c in range(n):
                        for d in range(n):
                            val = _sp.S.Zero
                            for a in range(n):
                                gae = gm[a,e]
                                if gae != 0 and (a,b,c,d) in raw:
                                    val += gae * raw[a,b,c,d]
                            if val != 0:
                                val = self._simp(_sp.cancel(val))
                            if val != 0:
                                C[e,b,c,d] = val
            self._C_dict = C
        return self._C_dict

    # ── Covariant derivatives of Weyl: ∇^order C ─────────────────────────────

    def nabla_weyl(self, order: int = 1) -> dict:
        """
        Return raw components of ∇^order C as a sparse dict.

        Keys are (4+order)-tuples of coordinate indices (a,b,c,d,e1,...,en).
        Values are SymPy expressions.

        Uses a fresh SageMath manifold with SymPy calculus engine to compute
        nab(nab(...C...)) correctly for any order without Maxima hangs.
        The original manifold (used for C_dict, G_dict etc.) is untouched.
        Cached after first call.
        """
        if order in self._nabC_cache:
            return self._nabC_cache[order]

        from sage.manifolds.manifold import Manifold as _SM
        import sympy as _sp
        import itertools

        n = self._n
        sig_int = self._sig_int

        # Build a fresh manifold with SymPy engine set BEFORE chart creation.
        # This avoids the Maxima simplify_rational() hang that occurs with
        # the default SR engine during nab(T).
        M2 = _SM(n, 'M_nab', structure='pseudo-Riemannian', signature=sig_int)
        M2.set_calculus_method('sympy')
        coord_names = ' '.join(str(c) for c in self.coords)
        X2 = M2.chart(coord_names)
        sage_coords2 = [_sp.Symbol(str(c)) for c in X2[:]]
        subs_map = {orig: sc for orig, sc in zip(self.coords, sage_coords2)}

        # Fill metric from already-extracted SymPy g_matrix
        g2 = M2.metric('g')
        for i in range(n):
            for j in range(i, n):
                val = self.g_matrix[i, j]
                if val != 0:
                    g2[i, j] = X2.function(val.subs(subs_map))

        # Compute ∇^order C
        nab2 = g2.connection()
        T = g2.weyl()
        for _ in range(order):
            T = nab2(T)

        # Extract nonzero components via .comp() (avoids Zero sentinel bug)
        comps_obj = T.comp()
        result = {}
        for idx in itertools.product(range(n), repeat=4 + order):
            try:
                v = _to_sympy(comps_obj[idx], sym_map=self._sympy_sym_map)
                if v != 0:
                    v = self._simp(v)
                if v != 0:
                    result[idx] = v
            except Exception:
                pass

        self._nabC_cache[order] = result
        return result

    def psid_components(self, frame_vecs: list, order: int = 1) -> list:
        """
        Project ∇^order C onto the null frame.

        Returns (k, dir_label, expr) triples in the same format as
        compute_psid_direct() in karlhede.py.

        At order=1: uses compute_psid_direct() with SageManifolds C_dict
        and G_dict — fast and correct.

        At order>1: uses nabla_weyl() on a fresh SymPy-engine manifold —
        slow but mathematically correct for any metric.

        dir_label is an integer (0-3) at order=1 for backward compatibility
        with test_psid_isotropy(), and a tuple at order>1.
        """
        from .karlhede import _PSI_LEGS, _simp as _k_simp, compute_psid_direct
        import itertools as _it

        n = self._n

        if order == 1:
            # Fast path: compute_psid_direct with SageManifolds tensors.
            # C_dict and G_dict use the correct SageManifolds Christoffel
            # symbols and Weyl tensor (with first index lowered).
            return compute_psid_direct(
                self.C_dict, self.G_dict, frame_vecs, self.coords, n,
                self._simp)

        # Slow but correct path for order >= 2
        nabC = self.nabla_weyl(order)

        comps = []
        for k, (l1, l2, l3, l4) in _PSI_LEGS.items():
            v1 = frame_vecs[l1]
            v2 = frame_vecs[l2]
            v3 = frame_vecs[l3]
            v4 = frame_vecs[l4]
            nz1 = [i for i in range(n) if v1[i] != 0]
            nz2 = [i for i in range(n) if v2[i] != 0]
            nz3 = [i for i in range(n) if v3[i] != 0]
            nz4 = [i for i in range(n) if v4[i] != 0]

            for e_dirs in _it.product(range(4), repeat=order):
                e_vecs = [frame_vecs[ed] for ed in e_dirs]
                nze_list = [[i for i in range(n) if ev[i] != 0]
                            for ev in e_vecs]

                total = sp.S.Zero
                for a in nz1:
                    for b in nz2:
                        for c in nz3:
                            for d in nz4:
                                outer = v1[a]*v2[b]*v3[c]*v4[d]
                                if outer == 0:
                                    continue
                                for eps_combo in _it.product(*nze_list):
                                    ev_factors = sp.S.One
                                    for ev, eps in zip(e_vecs, eps_combo):
                                        ev_factors *= ev[eps]
                                    if ev_factors == 0:
                                        continue
                                    key = (a, b, c, d) + eps_combo
                                    val = nabC.get(key, sp.S.Zero)
                                    if val != 0:
                                        total += val * outer * ev_factors

                total = _k_simp(sp.cancel(total), self._simp)
                comps.append((k, e_dirs, total))

        return comps


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructor: build SageGeometry from a SymPy metric Matrix
# (drop-in for the existing metrics.py definitions)

def sage_metric_from_matrix(g_sym, coords_sym, signature='lorentzian', simp_fn=None):
    _require_sage()
    n = len(coords_sym)
    sig_int = _SIGNATURE_MAP.get(signature)
    if sig_int is None:
        raise ValueError("Unknown signature: " + str(signature))
    from sage.manifolds.manifold import Manifold as SageManifold
    import sympy as _sp
    M = SageManifold(n, 'M', structure='pseudo-Riemannian', signature=sig_int)
    coord_names = ' '.join(str(c) for c in coords_sym)
    X = M.chart(coord_names)
    sage_coords = X[:]
    sage_coord_as_sympy = [_sp.Symbol(str(c)) for c in sage_coords]
    subs_map = {orig: new for orig, new in zip(coords_sym, sage_coord_as_sympy)}
    g = M.metric('g')
    for i in range(n):
        for j in range(i, n):
            val = g_sym[i, j]
            if val != 0:
                g[i, j] = X.function(val.subs(subs_map))
    all_syms = set()
    for i in range(n):
        for j in range(n):
            try:
                all_syms |= g_sym[i,j].free_symbols
            except Exception:
                pass
    return SageGeometry(g, X, simp_fn=simp_fn, manifold=M, sym_map={str(s): s for s in all_syms})
