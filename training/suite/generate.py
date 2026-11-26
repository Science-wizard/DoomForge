"""Generate a split of the 14-category benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.suite.spec import CATEGORIES, SPLIT_SEEDS, build_map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("train", "val", "test", "ood"))
    ap.add_argument("--categories", default="all")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="training/data/suite")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cats = list(CATEGORIES) if args.categories == "all" else args.categories.split(",")
    base = SPLIT_SEEDS[args.split]
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cid in cats:
        n = args.n
        if cid == "C12_freedoom":
            n = min(n, {"train": 16, "val": 6}.get(args.split, 10))
        rows = []
        for i in range(n):
            row = build_map(cid, base + i, args.split, args.out, force=args.force)
            row["category"] = cid
            rows.append(row)
            all_rows.append(row)
        (root / args.split / cid).mkdir(parents=True, exist_ok=True)
        (root / args.split / cid / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"{cid}: {len(rows)} maps", flush=True)
    (root / args.split / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows) + "\n")
    print(f"wrote {root / args.split / 'manifest.jsonl'} ({len(all_rows)} maps)")


if __name__ == "__main__":
    main()
