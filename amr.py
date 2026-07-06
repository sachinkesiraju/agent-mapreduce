#!/usr/bin/env python3
"""amr.py -- deterministic bookkeeping for the agentic map-reduce loop (see program.md).

The agent reasons; this file does arithmetic. State lives entirely in
results.tsv (append-only) + git. Stdlib only, no config, no dependencies.

  python3 amr.py log <gen> <commit> <parent> <score|-> <ran|crash|note> <description...>
  python3 amr.py reduce --gen N --margin M [--beam B] [--maximize]
  python3 amr.py tree [--maximize]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TSV = Path("results.tsv")
HEADER = "gen\tcommit\tparent\tscore\tstatus\tdescription"
STATUSES = ("ran", "crash", "note")


def read_rows() -> list[dict]:
    if not TSV.exists():
        return []
    lines = TSV.read_text().splitlines()
    if lines and lines[0] != HEADER:
        sys.exit(f"results.tsv: bad header, expected: {HEADER}")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            sys.exit(f"results.tsv:{n}: expected 6 tab-separated columns, got {len(parts)}")
        gen, commit, parent, score, status, desc = parts
        rows.append({
            "gen": int(gen), "commit": commit, "parent": parent,
            "score": None if score == "-" else float(score),
            "status": status, "desc": desc,
        })
    return rows


def node_scores(rows: list[dict], minimize: bool) -> dict[str, float]:
    """Best observed score per commit (repeat runs collapse to the best)."""
    runs: dict[str, list[float]] = {}
    for r in rows:
        if r["status"] == "ran" and r["score"] is not None:
            runs.setdefault(r["commit"], []).append(r["score"])
    pick = min if minimize else max
    return {c: pick(v) for c, v in runs.items()}


def cmd_log(args) -> None:
    if args.status not in STATUSES:
        sys.exit(f"status must be one of {STATUSES}")
    if args.score != "-":
        try:
            float(args.score)
        except ValueError:
            sys.exit(f"score must be a float or '-', got {args.score!r}")
    elif args.status == "ran":
        sys.exit("status 'ran' requires a numeric score")
    desc = " ".join(args.description).replace("\t", " ").replace("\n", " ").strip()
    if not TSV.exists():
        TSV.write_text(HEADER + "\n")
    with TSV.open("a") as f:
        f.write(f"{args.gen}\t{args.commit}\t{args.parent}\t{args.score}\t{args.status}\t{desc}\n")
    print(f"logged: gen {args.gen} {args.commit} {args.status} {desc}")


def cmd_reduce(args) -> None:
    minimize = not args.maximize
    rows = read_rows()
    scores = node_scores(rows, minimize)
    cands: dict[str, dict] = {}  # unique candidate nodes of this gen (first row wins)
    for r in rows:
        if r["gen"] == args.gen and r["status"] == "ran" and r["commit"] not in cands:
            cands[r["commit"]] = r
    if not cands:
        sys.exit(f"no 'ran' rows for gen {args.gen}")
    survivors, parents = [], {}
    for c, r in cands.items():
        p = r["parent"]
        if p == "-":
            continue  # baseline rows: nothing to beat
        if p not in scores:
            print(f"warn: {c}: parent {p} has no logged score; skipping", file=sys.stderr)
            continue
        parents[p] = scores[p]
        delta = scores[c] - scores[p]
        if (-delta if minimize else delta) > args.margin:
            survivors.append((c, scores[c], p, delta, r["desc"]))
    survivors.sort(key=lambda t: t[1], reverse=not minimize)
    kept = survivors[:args.beam]
    print(f"gen {args.gen}: {len(cands)} candidates, {len(survivors)} beat their parent "
          f"(margin {args.margin:g}), keeping {len(kept)}")
    for c, s, p, d, desc in kept:
        print(f"KEEP\t{c}\t{s:.6f}\t(parent {p}, delta {d:+.6f})\t{desc}")
    if kept:
        print("FRONTIER: " + " ".join(c for c, *_ in kept))
    else:
        print("STALL: no candidate beat its parent by more than the margin. Frontier unchanged.")
        print("FRONTIER: " + " ".join(sorted(parents, key=parents.get, reverse=not minimize)))


def cmd_tree(args) -> None:
    rows = read_rows()
    if not rows:
        sys.exit("no results.tsv yet")
    scores = node_scores(rows, minimize=not args.maximize)
    nodes: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for r in rows:
        if r["status"] == "note" or r["commit"] in nodes:
            continue
        nodes[r["commit"]] = r
        children.setdefault(r["parent"], []).append(r["commit"])
    roots = [c for c, r in nodes.items() if r["parent"] == "-" or r["parent"] not in nodes]

    def render(c: str, prefix: str, last: bool, root: bool) -> None:
        s = f"{scores[c]:.6f}" if c in scores else "crash"
        connector = "" if root else ("└── " if last else "├── ")
        print(f"{prefix}{connector}{c}  {s}  {nodes[c]['desc']}")
        kids = children.get(c, [])
        ext = "" if root else ("    " if last else "│   ")
        for i, k in enumerate(kids):
            render(k, prefix + ext, i == len(kids) - 1, False)

    for root in roots:
        render(root, "", True, True)
    notes = [r for r in rows if r["status"] == "note"]
    if notes:
        print("\nnotes/laws:")
        for r in notes:
            print(f"  gen {r['gen']}: {r['desc']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("log", help="append one row to results.tsv")
    p.add_argument("gen", type=int)
    p.add_argument("commit")
    p.add_argument("parent")
    p.add_argument("score", help="metric value, or '-' for crash/note rows")
    p.add_argument("status", help="ran | crash | note")
    p.add_argument("description", nargs="+")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("reduce", help="pick the surviving frontier for a generation")
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--beam", type=int, default=2)
    p.add_argument("--maximize", action="store_true")
    p.set_defaults(func=cmd_reduce)

    p = sub.add_parser("tree", help="print the search tree")
    p.add_argument("--maximize", action="store_true")
    p.set_defaults(func=cmd_tree)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
