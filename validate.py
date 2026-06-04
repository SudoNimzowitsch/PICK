"""
pick/validate.py  —  Regression harness against CLASSI's own ptest/ outputs.

This is the acceptance gate for the rebuild. The 33 reference strings below are
copied verbatim from sheep/062/clasrc/ptest/*.sum — the classification summaries
the original CLASSI system produced for its own test metrics. An implementation
is correct to the extent it reproduces these.

Format (decoded from clasum.shp:33-56):

    [name:6] ' ' [Segre][Lambda][Petrov] '  ' [r] ' ' [s_inf] ' ' [H0H1H2H3] ' ' [t0..t6]

  name    : 6 chars, lowercase, blank padded
  Segre   : Segre/Plebanski letter ('?' if undetermined)
  Lambda  : '0' if R==0 else '1'
  Petrov  : one of 0 1 2 3 D N  ('?' if undetermined)
  r       : isometry-group dimension, single digit or 'X' (== 10)
  s_inf   : final isotropy-group dimension
  H0..H3  : isotropy symbol per differentiation order; '-' if not reached
            (6=full Lorentz, p/t/r=3-dim, e/n=2-dim, s/b/k=1-dim, 0=trivial)
  t0..t6  : cumulative count of functionally independent invariants per order;
            '-' if not reached

Usage:

    from pick.validate import REFERENCES, parse_sum, compare, run_suite

    # Once the classifier exists:
    def classify_to_sum(name, metric, coords, **kw) -> str: ...
    run_suite(classify_to_sum, ignore=('segre',))   # ignore Segre until implemented
"""

from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# The 33 reference outputs, verbatim from sheep/062/clasrc/ptest/*.sum
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_SUMS = [
    "allnut p13  1 0 00-- 33-----",
    "berob3 e00  6 2 ee-- 00-----",
    "cylema e01  3 0 00-- 11-----",
    "cylemb e01  3 0 00-- 11-----",
    "desitt 010  X 6 66-- 00-----",
    "diie   00D  2 0 e00- 222----",
    "e1035a r00  7 3 rr-- 00-----",
    "einuni p10  7 3 pp-- 00-----",
    "elwave r0N  6 2 nn-- 00-----",
    "es107  r0N  5 2 nn-- 11-----",
    "es107a r00  6 3 rrr- 011----",
    "es1111 s1D  4 1 ss-- 11-----",
    "friedc p10  6 3 pp-- 11-----",
    "godel  p1D  5 1 ss-- 00-----",
    "goty   111  3 0 00-- 11-----",
    "gotysc t10  7 3 tt-- 00-----",
    "ki     00D  4 1 ess- 111----",
    "kiiemb 00D  2 0 e00- 222----",
    "kiiia  00D  2 0 e00- 122----",
    "kiva   00D  4 1 ebb- 111----",
    "melnic p1D  4 1 ss-- 11-----",
    "minkow 000  X 6 6--- 0------",
    "ozsiii p1D  4 0 s00- 000----",
    "petrme 003  2 0 000- 022----",
    "renord e0D  4 1 ess- 111----",
    "schwar 00D  4 1 ess- 111----",
    "szek1  p1D  3 1 sss- 122----",
    "try2   e1D  3 1 ess- 122----",
    "vaidya h1D  3 1 ss-- 22-----",
    "vaidyl r1D  3 1 sss- 122----",
    "wavez  00N  5 2 nnn- 011----",
    "wavezk 00N  1 0 n000 0233---",
    "wilson e01  3 0 00-- 11-----",
]

# Human-readable identification of the test metrics (for triage/reporting).
METRIC_NOTES = {
    "minkow": "Minkowski (flat)",
    "desitt": "de Sitter (Lambda)",
    "schwar": "Schwarzschild",
    "ki":     "Kerr",
    "kiva":   "Kerr (variant frame)",
    "diie":   "type D, II-electric variant",
    "godel":  "Godel rotating dust",
    "friedc": "closed FRW perfect fluid",
    "einuni": "Einstein static universe",
    "wavez":  "plane wave (type N)",
    "wavezk": "special plane wave (slow termination, order 4)",
    "elwave": "EM + gravitational wave (type N radiation)",
    "goty":   "type I example",
    "petrme": "type III example (Petrov metric)",
    "vaidya": "Vaidya null radiation",
    "renord": "Reissner-Nordstrom",
}


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Classification:
    name:    str
    segre:   str          # single char
    lam:     str          # '0' or '1'
    petrov:  str          # 0 1 2 3 D N or ?
    r:       str          # single char: digit or 'X'
    s_inf:   str          # single char digit
    H:       str          # isotropy sequence, hyphens stripped (e.g. 'ess')
    t:       str          # independent-function sequence, hyphens stripped (e.g. '111')

    def r_int(self) -> int:
        return 10 if self.r == 'X' else int(self.r)


def parse_sum(line: str) -> Classification:
    """
    Parse a .sum line into its fields. Robust to the fixed-column layout:
    name occupies cols 0-5, then a space, then the 3-char Segre/Lambda/Petrov
    code, then the rest is whitespace-separated.
    """
    raw = line.rstrip("\r\n")
    name = raw[:6].strip()
    rest = raw[6:]
    # rest begins with ' ' then 'SLP' (3 chars) then '  ' then tokens
    rest = rest.lstrip(" ")
    code = rest[:3]
    segre, lam, petrov = code[0], code[1], code[2]
    tokens = rest[3:].split()
    # tokens: [r, s_inf, Hseq, tseq]
    r, s_inf, Hseq, tseq = tokens[0], tokens[1], tokens[2], tokens[3]
    return Classification(
        name=name, segre=segre, lam=lam, petrov=petrov,
        r=r, s_inf=s_inf,
        H=Hseq.replace("-", ""),
        t=tseq.replace("-", ""),
    )


REFERENCES = {c.name: c for c in (parse_sum(s) for s in REFERENCE_SUMS)}


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

# Field weights / groupings so partial implementations can be scored.
FIELD_GROUPS = {
    "petrov": ["petrov"],
    "lambda": ["lam"],
    "isometry": ["r"],
    "isotropy_dim": ["s_inf"],
    "isotropy_seq": ["H"],
    "indep_seq": ["t"],
    "segre": ["segre"],
}


def compare(got: Classification, ref: Classification, ignore=()):
    """
    Compare a produced Classification against the reference.
    `ignore` is a set of FIELD_GROUPS keys to skip (e.g. ('segre',)).
    Returns (ok: bool, mismatches: list[str]).
    """
    mismatches = []
    for group, fields in FIELD_GROUPS.items():
        if group in ignore:
            continue
        for f in fields:
            g = getattr(got, f)
            r = getattr(ref, f)
            if g != r:
                mismatches.append(f"{group}.{f}: got {g!r}, expected {r!r}")
    return (len(mismatches) == 0, mismatches)


def format_as_sum(c: Classification) -> str:
    """Re-serialise a Classification to the .sum layout (for eyeballing)."""
    def pad_seq(seq, n):
        return (seq + "-" * n)[:n]
    name = (c.name + " " * 6)[:6]
    H = pad_seq(c.H, 4)
    t = pad_seq(c.t, 7)
    return f"{name} {c.segre}{c.lam}{c.petrov}  {c.r} {c.s_inf} {H} {t}"


# ─────────────────────────────────────────────────────────────────────────────
# Suite runner — plug the classifier in here
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(classify_fn, metrics: dict = None, ignore=(), verbose=True):
    """
    classify_fn(name) -> Classification   (or a .sum string, which we parse)

    `metrics` optionally maps reference-name -> whatever your classifier needs;
    if you wire metric definitions in, classify_fn can look them up by name.
    Cases not produced by classify_fn (KeyError / NotImplemented / None) are
    reported as SKIPPED, not failed, so you can gate milestone by milestone.

    Returns (n_pass, n_fail, n_skip).
    """
    n_pass = n_fail = n_skip = 0
    for name, ref in REFERENCES.items():
        try:
            out = classify_fn(name)
        except (NotImplementedError, KeyError):
            out = None
        except Exception as e:
            if verbose:
                print(f"  ERROR  {name:8s}  {type(e).__name__}: {e}")
            n_fail += 1
            continue
        if out is None:
            n_skip += 1
            continue
        got = parse_sum(out) if isinstance(out, str) else out
        ok, mism = compare(got, ref, ignore=ignore)
        if ok:
            n_pass += 1
            if verbose:
                print(f"  PASS   {name:8s}  {format_as_sum(ref)}")
        else:
            n_fail += 1
            if verbose:
                note = METRIC_NOTES.get(name, "")
                print(f"  FAIL   {name:8s}  {note}")
                print(f"           ref: {format_as_sum(ref)}")
                print(f"           got: {format_as_sum(got)}")
                for m in mism:
                    print(f"             - {m}")
    if verbose:
        print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped "
              f"(of {len(REFERENCES)})")
    return n_pass, n_fail, n_skip


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: confirms the parser round-trips all references
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Self-test: parse + re-serialise all 33 references\n")
    bad = 0
    for s in REFERENCE_SUMS:
        c = parse_sum(s)
        # Normalise spacing for comparison (the references have a 2-space gap
        # after the code and single spaces elsewhere).
        rebuilt = format_as_sum(c)
        # Compare field-wise via re-parse rather than string (spacing differs).
        c2 = parse_sum(rebuilt)
        ok, mism = compare(c, c2)
        flag = "ok " if ok else "DIFF"
        if not ok:
            bad += 1
        print(f"  [{flag}] {c.name:8s} P={c.petrov} r={c.r} s={c.s_inf} "
              f"H={c.H:4s} t={c.t}")
    print(f"\n  {len(REFERENCE_SUMS)} references, {bad} round-trip problems.")
    print("  (H/t shown with hyphens stripped; sequences are the live orders.)")
