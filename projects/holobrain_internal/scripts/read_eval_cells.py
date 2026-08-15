#!/usr/bin/env python3
"""Read an eval run's cells from the authoritative files, and test them.

    read_eval_cells.py <run_dir> [--vs NAME=WINS/N ...]

Why this exists rather than a shell one-liner retyped each time:

1. The top-level summary_seed_*.json and benchmark_summary_seed_*.json are
   OVERWRITTEN by each later stage that shares a seed. On E1, a0_mem32
   clobbered s1_fix_s0's, and a1_fifo clobbered a0's. Reading them gives the
   last stage's numbers under the run's name. The per-cell files under
   RoboDojo/<task>/.../<run_tag>_<task>/_result.json carry the stage tag and
   are not overwritten -- those are the source of truth.

2. Partial credit matters. Two runs have now differed significantly in
   "scored anything at all" while their success counts were identical
   (a1_fifo 30/50 vs 39/50, E4 ensemble 23/50 vs 39/50, p=0.0018), and a
   table of successes alone hides that.

3. Fisher by hand invites arithmetic slips in exactly the direction one
   hopes for.

Measured noise floor, for reading the output: one layout has been observed at
0.05 on one run and 1.00 on another under an identical configuration, so no
per-layout comparison at small N means anything. Whole cells of 50 have a
floor around +/-2 successes.
"""
import argparse
import collections
import json
import pathlib
import re
from math import comb


def fisher(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)

    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    obs = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument(
        "--vs",
        action="append",
        default=[],
        metavar="NAME=WINS/N",
        help="baseline to test every cover_blocks cell against, e.g. "
        "old=9/50. Repeatable.",
    )
    args = ap.parse_args()

    baselines = []
    for spec in args.vs:
        name, _, frac = spec.partition("=")
        wins, _, total = frac.partition("/")
        baselines.append((name, int(wins), int(total)))

    root = pathlib.Path(args.run_dir)
    # aidi_seed_<seed>_worker_<w>_<stage>_<task>. The task is matched against
    # the known set rather than by splitting on an underscore: stage names
    # contain them too, so `a1_fifo_cover_blocks` otherwise parses as stage
    # `a1` with task `fifo_cover_blocks`, and `swap_T` is dropped by any
    # lowercase-only pattern.
    TASKS = (
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    )
    head = re.compile(r"aidi_seed_(\d+)_worker_\d+_(.+)$")
    cells: dict = collections.defaultdict(dict)
    for f in sorted(root.rglob("_result.json")):
        m = head.match(f.parent.name)
        task = next((t for t in TASKS if m and m.group(2).endswith("_" + t)),
                    None)
        if not m or task is None:
            print(f"  ! unparsed cell dir: {f.parent.name}")
            continue
        seed = m.group(1)
        stage = m.group(2)[: -len(task) - 1]
        det = json.load(open(f)).get("details") or {}
        wins = sum(1 for v in det.values() if v.get("score") == 1.0)
        part = sum(1 for v in det.values() if 0 < (v.get("score") or 0) < 1.0)
        cells[(stage, seed)][task] = (wins, part, len(det))

    if not cells:
        print("no cells found -- has any stage finished?")
        return

    tasks = sorted({t for v in cells.values() for t in v})
    w = max(len(s) for s, _ in cells)
    print(f"{'stage':<{w}}  seed  " + "  ".join(f"{t[:24]:>24}" for t in tasks))
    for (stage, seed), per in sorted(cells.items()):
        row = []
        for t in tasks:
            if t in per:
                a, p, n = per[t]
                row.append(f"{a}/{n} (+{p} partial)".rjust(24))
            else:
                row.append("-".rjust(24))
        print(f"{stage:<{w}}  {seed:>4}  " + "  ".join(row))

    if not baselines:
        return
    print()
    print("Fisher exact vs baselines, cover_blocks successes:")
    for (stage, seed), per in sorted(cells.items()):
        cb = per.get("cover_blocks")
        if not cb:
            continue
        a, _, n = cb
        for name, bw, bn in baselines:
            print(f"  {stage:<{w}} seed{seed}  {a}/{n} vs {name} {bw}/{bn}"
                  f"   p={fisher(a, n - a, bw, bn - bw):.3f}")
    print()
    print("Reading note: the per-cell floor is about +/-2 successes in 50, and")
    print("one layout has been measured at 0.05 and at 1.00 under an identical")
    print("configuration -- no small-N per-layout comparison is meaningful.")


if __name__ == "__main__":
    main()
