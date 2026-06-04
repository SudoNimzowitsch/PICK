"""
pick/test_sage_backend.py  —  Phase 1 validation harness

Verifies that the SageManifolds adapter (sage_backend.py) produces
tensor components that exactly match the hand-rolled SymPy geometry
layer in karlhede.py.

Checks, for each of the 7 fast diagonal metrics:
  1. g_matrix    — metric matches
  2. gi_matrix   — inverse metric matches
  3. G_dict      — Christoffel symbols match (spot-check nonzero entries)
  4. Ric_matrix  — Ricci tensor matches
  5. Rs_expr     — Ricci scalar matches
  6. C_dict      — Weyl tensor matches (spot-check nonzero entries)
  7. PSI          — Weyl spinor Ψ₀…Ψ₄ matches (via existing projection code)
  8. Classification — final r, t-sequence, H-sequence match CLASSI reference

Run with:
    sage --python test_sage_backend.py
  or inside a sage session:
    exec(open('pick/test_sage_backend.py').read())

If SageMath is not available, the script exits cleanly with a skip message.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

# ── Try importing SageMath ────────────────────────────────────────────────────
try:
    from pick.sage_backend import sage_metric_from_matrix, _SAGE_AVAILABLE
    if not _SAGE_AVAILABLE:
        print("SKIP: SageMath not available.")
        sys.exit(0)
except ImportError as e:
    print(f"SKIP: {e}")
    sys.exit(0)

from pick.metrics import METRICS
from pick.karlhede import (
    christoffel, riemann, ricci_ten, ricci_sc, weyl_ten,
    weyl_spinor_from_coframe, null_coframe_from_diagonal_metric,
    frame_vectors_from_coframe, detect_signature, KarlhedeClassifier,
    _simp,
)
from pick.validate import REFERENCES, parse_sum, compare

# ── Test configuration ────────────────────────────────────────────────────────

FAST_METRICS = ['minkow', 'schwar', 'desitt', 'renord', 'einuni', 'friedc', 'try2']

# Tolerance: expressions are "equal" if their difference simplifies to 0
def exprs_equal(a, b, simp_fn=None):
    diff = _simp(sp.cancel(a - b), simp_fn)
    return diff == 0


def run_checks():
    passed = 0
    failed = 0
    errors = []

    for name in FAST_METRICS:
        print(f"\n{'='*60}")
        print(f"  Testing: {name}")
        print(f"{'='*60}")

        m = METRICS[name]()
        g_sym   = m['g']
        coords  = m['coords']
        simp_fn = m.get('simp')
        n       = len(coords)

        # ── Reference: hand-rolled SymPy geometry ────────────────────────────
        sig = detect_signature(g_sym, simp_fn)
        gi  = g_sym.inv()
        G_ref = christoffel(g_sym, coords)
        R_ref = riemann(g_sym, coords, G_ref)
        Ric_ref = ricci_ten(R_ref, n).applyfunc(
            lambda x: _simp(x, simp_fn))
        Rs_ref = _simp(ricci_sc(Ric_ref, gi), simp_fn)

        # Weyl (pruned)
        C_ref_full = weyl_ten(R_ref, Ric_ref, Rs_ref, g_sym, n)
        C_ref = {k: v for k, v in C_ref_full.items()
                 if _simp(v, simp_fn) != 0}

        l, nv, m_leg, mb = null_coframe_from_diagonal_metric(g_sym, sig,
                                                              simp_fn=simp_fn)
        PSI_ref = weyl_spinor_from_coframe(C_ref, gi, l, nv, m_leg, mb, n, simp_fn)
        PSI_ref = [_simp(p, simp_fn) for p in PSI_ref]

        # ── SageManifolds adapter ─────────────────────────────────────────────
        try:
            sg = sage_metric_from_matrix(g_sym, coords,
                                         signature=sig, simp_fn=simp_fn)
        except Exception as e:
            print(f"  FAIL: sage_metric_from_matrix raised: {e}")
            failed += 1
            errors.append((name, 'sage_construct', str(e)))
            continue

        # ── Check 1: metric ───────────────────────────────────────────────────
        metric_ok = True
        for i in range(n):
            for j in range(n):
                if not exprs_equal(sg.g_matrix[i, j], g_sym[i, j], simp_fn):
                    metric_ok = False
                    errors.append((name, f'g[{i},{j}]',
                        f"sage={sg.g_matrix[i,j]}  ref={g_sym[i,j]}"))
        _report('metric', metric_ok)
        if metric_ok: passed += 1
        else: failed += 1

        # ── Check 2: inverse metric ───────────────────────────────────────────
        gi_ok = True
        for i in range(n):
            for j in range(n):
                if not exprs_equal(sg.gi_matrix[i, j], gi[i, j], simp_fn):
                    gi_ok = False
                    errors.append((name, f'gi[{i},{j}]',
                        f"sage={sg.gi_matrix[i,j]}  ref={gi[i,j]}"))
        _report('inverse metric', gi_ok)
        if gi_ok: passed += 1
        else: failed += 1

        # ── Check 3: Christoffel (nonzero entries only) ───────────────────────
        G_ok = True
        all_keys = set(G_ref.keys()) | set(sg.G_dict.keys())
        for key in all_keys:
            v_ref  = G_ref.get(key, sp.S.Zero)
            v_sage = sg.G_dict.get(key, sp.S.Zero)
            if not exprs_equal(v_sage, v_ref, simp_fn):
                G_ok = False
                errors.append((name, f'G{key}',
                    f"sage={v_sage}  ref={v_ref}"))
                if len([e for e in errors if e[0]==name and 'G' in e[1]]) > 3:
                    errors.append((name, 'G', '(further mismatches suppressed)'))
                    break
        _report(f'Christoffel ({len(G_ref)} nonzero ref)', G_ok)
        if G_ok: passed += 1
        else: failed += 1

        # ── Check 4: Ricci tensor ─────────────────────────────────────────────
        Ric_ok = True
        for i in range(n):
            for j in range(n):
                if not exprs_equal(sg.Ric_matrix[i, j], Ric_ref[i, j], simp_fn):
                    Ric_ok = False
                    errors.append((name, f'Ric[{i},{j}]',
                        f"sage={sg.Ric_matrix[i,j]}  ref={Ric_ref[i,j]}"))
        _report('Ricci tensor', Ric_ok)
        if Ric_ok: passed += 1
        else: failed += 1

        # ── Check 5: Ricci scalar ─────────────────────────────────────────────
        Rs_ok = exprs_equal(sg.Rs_expr, Rs_ref, simp_fn)
        _report('Ricci scalar', Rs_ok)
        if Rs_ok: passed += 1
        else:
            failed += 1
            errors.append((name, 'Rs',
                f"sage={sg.Rs_expr}  ref={Rs_ref}"))

        # ── Check 6: Weyl tensor (nonzero entries) ────────────────────────────
        C_ok = True
        all_C_keys = set(C_ref.keys()) | set(sg.C_dict.keys())
        mismatch_count = 0
        for key in all_C_keys:
            v_ref  = C_ref.get(key, sp.S.Zero)
            v_sage = sg.C_dict.get(key, sp.S.Zero)
            if not exprs_equal(v_sage, v_ref, simp_fn):
                C_ok = False
                mismatch_count += 1
                if mismatch_count <= 3:
                    errors.append((name, f'C{key}',
                        f"sage={v_sage}  ref={v_ref}"))
                elif mismatch_count == 4:
                    errors.append((name, 'C', '(further mismatches suppressed)'))
        _report(f'Weyl tensor ({len(C_ref)} nonzero ref)', C_ok)
        if C_ok: passed += 1
        else: failed += 1

        # ── Check 7: Weyl spinors (Ψ₀…Ψ₄) ───────────────────────────────────
        # We use the existing spinor projection code on sg.C_dict + sg.gi_matrix
        PSI_sage = weyl_spinor_from_coframe(
            sg.C_dict, sg.gi_matrix, l, nv, m_leg, mb, n, simp_fn)
        PSI_sage = [_simp(p, simp_fn) for p in PSI_sage]

        psi_ok = all(exprs_equal(PSI_sage[k], PSI_ref[k], simp_fn)
                     for k in range(5))
        _report('Weyl spinors Ψ₀…Ψ₄', psi_ok)
        if psi_ok: passed += 1
        else:
            failed += 1
            for k in range(5):
                if not exprs_equal(PSI_sage[k], PSI_ref[k], simp_fn):
                    errors.append((name, f'PSI[{k}]',
                        f"sage={PSI_sage[k]}  ref={PSI_ref[k]}"))

        # ── Check 8: End-to-end classification ───────────────────────────────
        # Run the existing KarlhedeClassifier with the sg-derived matrices
        # (same code path as current PICK, but geometry comes from SageMath)
        ref = REFERENCES.get(name)
        try:
            clf = KarlhedeClassifier(
                sg.g_matrix, coords,
                simplify_fn=simp_fn,
                verbose=False, max_order=4,
            )
            result = clf.classify()
            got = parse_sum(result.to_classification(name))
            cls_ok, mismatches = compare(got, ref, ignore=('segre',))
            _report(f'classification (r={ref.r}, t={ref.t}, H={ref.H})', cls_ok)
            if cls_ok: passed += 1
            else:
                failed += 1
                for mm in mismatches:
                    errors.append((name, 'classification', mm))
        except Exception as e:
            print(f"  FAIL (classification raised): {e}")
            failed += 1
            errors.append((name, 'classification', str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  RESULT: {passed}/{total} checks passed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed checks:")
        for metric, field, msg in errors:
            print(f"  [{metric}] {field}: {msg}")

    return failed == 0


def _report(label: str, ok: bool):
    status = '✓' if ok else '✗'
    print(f"  {status}  {label}")


if __name__ == '__main__':
    success = run_checks()
    sys.exit(0 if success else 1)
