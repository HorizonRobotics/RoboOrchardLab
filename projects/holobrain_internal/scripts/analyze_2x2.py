#!/usr/bin/env python3
"""The 2x2 factorial, analysed as a factorial rather than as four numbers.

The pre-registered criterion was "the diagonal (matched) beats the
anti-diagonal (mismatched)". That claim is the INTERACTION term, not a
comparison of any two cells, and it has to be tested as one -- the two rows
have very different baselines, so pooling "matched" against "mismatched"
across them would be comparing a stride1 cell with a stride32 cell.

Matched means the opposite numbering in each row: on stride-1 weights the OLD
numbering puts adjacent bank entries 1 apart, as training did; on stride-32
weights the FIXED numbering puts them 32 apart, as training did.

Woolf's homogeneity test with the Haldane-Anscombe 0.5 correction, because
one cell has a single success and an uncorrected log odds ratio would be
unstable there. The correction also makes the test conservative, which is the
right direction for a claim I am invested in.
"""
from math import comb, erfc, exp, log, sqrt


def fisher(a, b, c, d):
    n = a + b + c + d
    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    obs = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9))


def chi2_p(x):          # 1 degree of freedom
    return erfc(sqrt(x / 2))


N = 100
cell = {("stride1", "old"): 23, ("stride1", "fixed"): 13,
        ("stride32", "old"): 1, ("stride32", "fixed"): 5}

print("cover_blocks, two seeds pooled, 100 episodes per cell")
print(f"{'':<10}{'old':>12}{'fixed':>12}")
for tr in ("stride1", "stride32"):
    print(f"{tr:<10}{str(cell[(tr,'old')]) + '/100':>12}"
          f"{str(cell[(tr,'fixed')]) + '/100':>12}")
print()

print("within row (matched vs mismatched):")
print(f"  stride1   old 23 vs fixed 13   p={fisher(23,77,13,87):.3f}"
      "   (old is the matched one)")
print(f"  stride32  fixed 5 vs old 1     p={fisher(5,95,1,99):.3f}"
      "   (fixed is the matched one)")
print()

print("main effects:")
print(f"  training  stride1 36/200 vs stride32 6/200   p={fisher(36,164,6,194):.2e}")
print(f"  numbering old 24/200 vs fixed 18/200         p={fisher(24,176,18,182):.3f}")
print()


def logodds(a, n):
    a2 = a + 0.5
    b2 = n - a + 0.5
    return log(a2 / b2), 1.0 / a2 + 1.0 / b2


l1o, v1o = logodds(23, N)
l1f, v1f = logodds(13, N)
l2o, v2o = logodds(1, N)
l2f, v2f = logodds(5, N)
L1, V1 = l1o - l1f, v1o + v1f      # stride1  log OR(old / fixed)
L2, V2 = l2o - l2f, v2o + v2f      # stride32 log OR(old / fixed)
W = (L1 - L2) ** 2 / (V1 + V2)

print("interaction -- the term the 2x2 was built to test (Woolf, 1 df):")
print(f"  stride1   OR(old/fixed) = {exp(L1):.2f}   (>1: old better)")
print(f"  stride32  OR(old/fixed) = {exp(L2):.2f}   (<1: fixed better)")
print(f"  chi2 = {W:.2f}   p = {chi2_p(W):.3f}")
print()
print("Both rows favour their own matched cell, so the interaction is in the")
print("predicted direction. Neither row reaches 0.05 on its own, and the")
print("stride32 row sits at 1 and 5 successes where any test has almost no")
print("power -- the interaction estimate rests on those two cells.")
