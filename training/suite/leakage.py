"""Leakage checks and the test-touch log for the benchmark suite."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def hashes(manifest: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["spec_hash"]] = r["map_id"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="training/data/suite/train/manifest.jsonl")
    ap.add_argument("--test", default="training/data/suite/test/manifest.jsonl")
    ap.add_argument("--touch", default="", help="note to append to test_touch.log")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    if args.touch:
        p = Path("test_touch.log")
        with p.open("a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} ckpt={args.ckpt} note={args.touch}\n")
        print("logged", p)
        return
    a = hashes(args.train)
    b = hashes(args.test)
    inter = sorted(set(a) & set(b))
    print(f"train maps={len(a)} test maps={len(b)} hash intersections={len(inter)}")
    for h in inter[:10]:
        print("  overlap", h, a[h], b[h])
    raise SystemExit(1 if inter else 0)


if __name__ == "__main__":
    main()
