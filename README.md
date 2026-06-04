# PICK — Python Implementation of Cartan-Karlhede

PICK is a Python reimplementation of [CLASSI](https://github.com/janeaman/sheep),
the symbolic spacetime classifier written by Jan Åman and colleagues in the 1980s
using the SHEEP/Reduce computer algebra system. PICK modernises this tool for the
general relativity community, making it pip-installable and runnable on contemporary
Python infrastructure.

## What it does

Given a metric tensor g_{μν} as a SymPy matrix, PICK computes the complete
Cartan-Karlhede (CK) classification of the spacetime — the local invariant that
distinguishes spacetimes up to diffeomorphism. The output includes:

- **Petrov type** (0, I, II, III, D, N) — algebraic classification of the Weyl tensor
- **Segre type** — algebraic classification of the Ricci tensor (matter content)
- **Isotropy group H** — the residual frame freedom at each order
- **t-sequence** — number of functionally independent scalar invariants found at each order of covariant differentiation
- **Isometry group dimension r** — dimension of the local isometry group

These quantities together constitute the CK invariant, which is a complete local
invariant of the spacetime geometry.

## Example

```python
from pick.karlhede import KarlhedeClassifier
from pick.metrics import METRICS

# Classify the Schwarzschild metric
m = METRICS['schwar']()
clf = KarlhedeClassifier(m['g'], m['coords'], simplify_fn=m.get('simp'))
result = clf.classify()

print(result)
# Petrov: D   Segre: Vacuum
# r = 4  (isometry group dimension)
# t∞ = 1  s∞ = 1
# H-sequence: e → s
```

## Architecture

PICK has two backends:

**SymPy backend** (`karlhede.py`) — pure SymPy, no additional dependencies beyond
`pip install sympy`. Computes geometry (Christoffel, Riemann, Weyl) analytically.
Correct for all metrics at order 1; uses coordinate partial derivatives as a
stopgap for orders ≥ 2 (correct for highly symmetric metrics).

**SageManifolds backend** (`sage_backend.py`) — requires SageMath. Provides
mathematically correct ∇ⁿC for all orders via `nab(nab(...C...))`, which is
essential for accurate higher-order classification. The two backends are
interface-compatible; `sage_geometry=sg` is passed as an optional parameter
to `KarlhedeClassifier`.

## Installation

**SymPy backend only:**
```bash
pip install sympy
```

**With SageManifolds backend (recommended):**
```bash
# WSL2 / Linux
conda install -c conda-forge sage
```

## Validation

PICK is validated against the 33 reference metrics in CLASSI's `ptest/` suite.
Current status: 7/33 metrics fully validated (56/56 checks), covering vacuum,
electrovac, cosmological, and pp-wave spacetimes. The remaining metrics require
implementation of DYTAUT (dyad standardization) and NEWSUL (field equation
constraints for metrics with undetermined functions).

Run the validation suite:
```bash
sage -c "
import sys, importlib.util
sys.path.insert(0, '/path/to/pick/parent')
spec = importlib.util.spec_from_file_location('test_sb', 'pick/test_sage_backend.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.run_checks()
"
```

### Validated metrics

| Metric | Description | Petrov | r | t-sequence |
|--------|-------------|--------|---|------------|
| minkow | Minkowski | 0 | ∞ | 0 |
| schwar | Schwarzschild | D | 4 | 111 |
| desitt | de Sitter | 0 | ∞ | 00 |
| renord | Reissner-Nordström | D | 4 | 111 |
| einuni | Einstein static universe | 0 | 7 | 00 |
| friedc | FRW (closed) | 0 | 6 | 11 |
| try2 | pp-wave variant | D | 3 | 122 |

## Reference

PICK is a software implementation of the algorithm described in:

- Karlhede, A. (1980). A review of the geometrical equivalence of metrics in general relativity. *General Relativity and Gravitation*, 12(9), 693–707.
- Åman, J.E. et al. (1984). CLASSI: A computer algebra system for the classification of spacetimes. *University of Stockholm preprint*.
- MacCallum, M.A.H. & Åman, J.E. (1986). Algebraically independent nth derivatives of the Riemann curvature spinor in a general spacetime. *Classical and Quantum Gravity*, 3(6), 1133.

## Status

PICK is under active development. A software announcement paper targeting
*Computer Physics Communications* is in preparation.

## License

To be determined prior to public release.
