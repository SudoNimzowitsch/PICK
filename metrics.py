"""
pick/metrics.py — Python metric definitions decoded from CLASSI ptest/*.dia files.

Each entry is a dict with:
  'g': SymPy Matrix (covariant metric)
  'coords': list of symbols
  'simp': simplification function (optional, defaults to trigsimp+cancel)
  'ptest_name': name in ptest/*.sum (for harness lookup)
  'desc': human-readable description
"""

import sympy as sp
from sympy import (symbols, Function, sin, cos, cosh, sinh, tan,
                   exp, sqrt, diag, Rational, pi, Symbol, Abs)

# ─────────────────────────────────────────────────────────────────────────────
# Simplification helpers
# ─────────────────────────────────────────────────────────────────────────────

def simp_basic(x):
    """Rational simplification — fast for polynomial/rational expressions."""
    return sp.cancel(x)

def simp_trig(x):
    """Full simplification including trig — for metrics with sin/cos."""
    result = sp.cancel(x)
    if result.has(sp.sin, sp.cos, sp.tan, sp.cosh, sp.sinh, sp.sqrt):
        result = sp.trigsimp(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build diagonal metric from CLASSI GD tetrad leg list
# In DIAINP, GD = (θ⁰, θ¹, θ², θ³) where for Lorentzian (−,+,+,+):
#   g_{00} = −(θ⁰)²,  g_{ii} = +(θⁱ)²
# ─────────────────────────────────────────────────────────────────────────────

def metric_from_tetrad_legs(legs):
    """Build diagonal metric from CLASSI GD tetrad legs."""
    assert len(legs) == 4
    return sp.diag(-legs[0]**2, legs[1]**2, legs[2]**2, legs[3]**2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Minkowski — minkow.dia
# GD = (1, 1, R, R*sin(H))
# ─────────────────────────────────────────────────────────────────────────────

def minkowski():
    t, r, H, P = symbols('t r H P', real=True)
    r = symbols('r', positive=True)   # r > 0 avoids Abs(r) in frame vectors
    g = metric_from_tetrad_legs([1, 1, r, r*sin(H)])
    return {'g': g, 'coords': [t,r,H,P], 'simp': simp_basic,
            'ptest_name': 'minkow',
            'desc': 'Minkowski (spherical)'}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schwarzschild — schwar.dia
# A = 1 − 2M/R;  GD = (A^(1/2), A^(-1/2), R, R*sin(H))
# ─────────────────────────────────────────────────────────────────────────────

def schwarzschild():
    t = symbols('t', real=True)
    r = symbols('r', positive=True)   # r > 0 avoids Abs(r) in frame vectors
    H, P = symbols('H P', real=True)
    M = Symbol('M', positive=True)
    A = 1 - 2*M/r
    g = metric_from_tetrad_legs([sqrt(A), 1/sqrt(A), r, r*sin(H)])
    return {'g': g, 'coords': [t,r,H,P], 'simp': simp_basic,
            'ptest_name': 'schwar',
            'desc': 'Schwarzschild'}


# ─────────────────────────────────────────────────────────────────────────────
# 3. de Sitter — desitt.dia
# GD = (1, A*cosh(T/A), A*cosh(T/A)*sin(X), A*cosh(T/A)*sin(X)*sin(H))
# A is the de Sitter radius (related to Λ = 3/A²)
# ─────────────────────────────────────────────────────────────────────────────

def desitter():
    T, X, H, P = symbols('T X H P', real=True)
    A = Symbol('A', positive=True)
    r = A * sp.cosh(T/A)
    g = metric_from_tetrad_legs([1, r, r*sp.sin(X), r*sp.sin(X)*sp.sin(H)])
    return {'g': g, 'coords': [T,X,H,P], 'simp': simp_trig,
            'ptest_name': 'desitt',
            'desc': 'de Sitter (FLRW form)'}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reissner-Nordström — renord.dia
# A = 1 − 2M/R + Q²/R²;  GD = (A^(1/2), A^(-1/2), R, R*sin(H))
# ─────────────────────────────────────────────────────────────────────────────

def reissner_nordstrom():
    t = symbols('t', real=True)
    r = symbols('r', positive=True)
    H, P = symbols('H P', real=True)
    M, Q = symbols('M Q', positive=True)
    A = 1 - 2*M/r + Q**2/r**2
    g = metric_from_tetrad_legs([sqrt(A), 1/sqrt(A), r, r*sin(H)])
    return {'g': g, 'coords': [t,r,H,P], 'simp': simp_basic,
            'ptest_name': 'renord',
            'desc': 'Reissner-Nordstrom'}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Einstein static universe — einuni.dia
# GD = (1, A, A*sin(X), A*sin(X)*sin(H));  A = constant radius
# ─────────────────────────────────────────────────────────────────────────────

def einstein_static():
    T, X, H, PH = symbols('T X H PH', real=True)
    A = Symbol('A', positive=True)
    g = metric_from_tetrad_legs([1, A, A*sin(X), A*sin(X)*sin(H)])
    return {'g': g, 'coords': [T,X,H,PH], 'simp': simp_trig,
            'ptest_name': 'einuni',
            'desc': 'Einstein static universe'}


# ─────────────────────────────────────────────────────────────────────────────
# 6. FRW closed dust — friedc.dia
# A = A0*(1 − cos(N));  GD = (A, A, A*sin(X), A*sin(X)*sin(H))
# ─────────────────────────────────────────────────────────────────────────────

def frw_closed():
    N, X, H, P = symbols('N X H P', real=True)
    A0 = Symbol('A0', positive=True)
    A = A0 * (1 - cos(N))
    g = metric_from_tetrad_legs([A, A, A*sin(X), A*sin(X)*sin(H)])
    return {'g': g, 'coords': [N,X,H,P], 'simp': simp_trig,
            'ptest_name': 'friedc',
            'desc': 'FRW closed dust (conformal time)'}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Szekeres — szek1.dia
# GD = (X³, X²*cos(2T), X/sqrt(cos(2T)), X/sqrt(cos(2T)))
# ─────────────────────────────────────────────────────────────────────────────

def szekeres():
    T, X, Y, Z = symbols('T X Y Z', real=True)
    g = metric_from_tetrad_legs([X**3, X**2*cos(2*T),
                                  X/sqrt(cos(2*T)), X/sqrt(cos(2*T))])
    return {'g': g, 'coords': [T,X,Y,Z], 'simp': simp_trig,
            'ptest_name': 'szek1',
            'desc': 'Szekeres solution'}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Try2 test metric — try2.dia
# F = X*T + T^C;  GD = (e^F, 1, 1, 1);  C is a constant
# ─────────────────────────────────────────────────────────────────────────────

def try2():
    T, X, Y, Z = symbols('T X Y Z', positive=True)  # T>0 needed for T^C to be real
    C = Symbol('C', positive=True)
    F = X*T + T**C
    g = metric_from_tetrad_legs([exp(F), 1, 1, 1])
    return {'g': g, 'coords': [T,X,Y,Z], 'simp': simp_basic,
            'ptest_name': 'try2',
            'desc': 'Try2 test metric'}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Bertotti-Robinson — berob3.dia
# F = 1 + L*Z², G = 1 − L*Y²;  GD = (F^(1/2), G^(1/2), G^(-1/2), F^(-1/2))
# ─────────────────────────────────────────────────────────────────────────────

def bertotti_robinson():
    T, X, Y, Z = symbols('T X Y Z', real=True)
    L = Symbol('L', positive=True)
    F = 1 + L*Z**2
    G = 1 - L*Y**2
    g = metric_from_tetrad_legs([sqrt(F), sqrt(G), 1/sqrt(G), 1/sqrt(F)])
    return {'g': g, 'coords': [T,X,Y,Z], 'simp': simp_basic,
            'ptest_name': 'berob3',
            'desc': 'Bertotti-Robinson homogeneous EM'}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Cylema — cylema.dia
# Static cylindrically symmetric Einstein-Maxwell (angular magnetic field)
# Exact solutions (20.9a)
# GD = (G*p^{m²}, G*p, G*p^{m²}, 1/G), G = B*cosh(log(A*p^m))
# Use auxiliary symbol G to avoid differentiating cosh(log(...)).
# dG/dp = B*m/G/p * sinh(log(A*p^m))^2... use CLASSI substitution:
# sinh^2 = cosh^2 - 1 → dG/dp = B*m*(G^2/B^2 - 1)*B/(G*p*B) = m*(G^2-B^2)/(G*B*p)
# Simpler: from G = B*cosh(log(A*p^m)), dG/dp = B*sinh(log(A*p^m))*m/p
# With H = dG/dp (declared as a function of p), G and H satisfy: H^2 = m^2*(G^2-B^2)/p^2
# For the metric, just use G and H as independent functions of p with that relation.
# ─────────────────────────────────────────────────────────────────────────────

def cylema():
    """
    Cylema: static cylindrical EM field (angular magnetic field).
    GD = diag(-G*p^{m²}, G*p, G*p^{m²}, 1/G)
    CLASSI: G = B*cosh(log(A*p^m)), treated as a function of p.
    We use G as an auxiliary symbol with G>0 to keep computation tractable.
    """
    t, ph, p, z = symbols('t ph p z', real=True)
    m, B = symbols('m B', positive=True)
    G = sp.Function('G')(p)   # G(p) = B*cosh(log(A*p^m)) > 0
    # Diagonal metric: ds^2 = -G p^{m^2} dt^2 + G p dphi^2 + G p^{m^2} dp^2 + 1/G dz^2
    g = sp.diag(
        -G * p**m**2,
         G * p,
         G * p**m**2,
         1/G
    )
    return {'g': g, 'coords': [t, ph, p, z], 'simp': sp.cancel,
            'ptest_name': 'cylema',
            'desc': 'Cylema: static cylindrical EM (angular field)'}


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cylemb — cylemb.dia
# Static cylindrically symmetric Einstein-Maxwell (longitudinal magnetic field)
# Exact solutions (20.9b)
# GD = (G*p^{m²}, 1/G, G*p^{m²}, G*p), G = B*cosh(log(A*p^m))
# ─────────────────────────────────────────────────────────────────────────────

def cylemb():
    """
    Cylemb: static cylindrical EM field (longitudinal magnetic field).
    GD = diag(-G*p^{m²}, 1/G, G*p^{m²}, G*p)
    Same G as cylema but different metric component ordering.
    """
    t, ph, p, z = symbols('t ph p z', real=True)
    m, B = symbols('m B', positive=True)
    G = sp.Function('G')(p)
    g = sp.diag(
        -G * p**m**2,
         1/G,
         G * p**m**2,
         G * p
    )
    return {'g': g, 'coords': [t, ph, p, z], 'simp': sp.cancel,
            'ptest_name': 'cylemb',
            'desc': 'Cylemb: static cylindrical EM (longitudinal field)'}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

METRICS = {
    'minkow':  minkowski,
    'schwar':  schwarzschild,
    'desitt':  desitter,
    'renord':  reissner_nordstrom,
    'einuni':  einstein_static,
    'friedc':  frw_closed,
    'szek1':   szekeres,
    'try2':    try2,
    'berob3':  bertotti_robinson,
    'cylema':  cylema,
    'cylemb':  cylemb,
}
