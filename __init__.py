"""
PICK — Python Implementation of Cartan-Karlhede

A modern Python reimplementation of the CLASSI system by Jan E. Åman
(University of Stockholm, 1987), which itself implemented the algorithm
of A. Karlhede (Gen. Rel. Grav. 12, 1980) for the invariant classification
of spacetimes in General Relativity.

PICK classifies a spacetime metric by:
  - Computing the Newman-Penrose curvature spinors (Ψ, Φ, Λ)
  - Determining the Petrov and Segre types
  - Running the Karlhede iteration to find the isometry group dimension,
    the number of functionally independent scalar invariants, and the
    residual isotropy group at each order of differentiation

Typical usage::

    import sympy as sp
    from pick import KarlhedeClassifier

    t, r, th, ph = sp.symbols('t r theta phi', real=True)
    m = sp.Symbol('m', positive=True)
    A = 1 - 2*m/r
    g = sp.diag(-A, 1/A, r**2, r**2*sp.sin(th)**2)

    clf = KarlhedeClassifier(g, [t, r, th, ph])
    result = clf.classify()
    print(result.summary())

References
----------
[1] A. Karlhede, Gen. Rel. Grav. 12 (1980) 693.
[2] A. Karlhede, Gen. Rel. Grav. 12 (1981) 963.
[3] J.E. Åman & A. Karlhede, Proc. SYMSAC '81, ACM (1981) 79.
[4] M.A.H. MacCallum & J.E. Åman, Class. Quantum Grav. 3 (1986) 1133.
[5] J.E. Åman, CLASSI/SHEEP source code (1987), github.com/grmath/sheep

Original CLASSI system copyright © Jan E. Åman.
PICK is released under the GPL-3.0 license, in keeping with the
license of the original SHEEP repository.
"""

__version__ = "0.1.0"
__author__  = "PICK contributors"
__license__ = "GPL-3.0"

from .karlhede import KarlhedeClassifier, KarlhedeResult, KarlhedeStep

__all__ = ["KarlhedeClassifier", "KarlhedeResult", "KarlhedeStep"]
