"""Score every new PPO snapshot on the full val split while training runs.

    python tools/auto_eval.py --run training/runs/ppo_ego3 --baseline training/runs/suite/valfull_ppo1_u50_t1

For each ``ckpt_uXXXX.pt`` it writes a sampling-config copy, runs the suite
through the real DoomAgent, compares against the baseline run and appends one
JSON line to ``<run>/auto_eval.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


def summary(run: Path) -> dict:
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    cats: dict[str, list] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: round(sum(int(r["exit"]) for r in v) / len(v), 2) for c, v in sorted(cats.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-category", type=int, default=10)
    args = ap.parse_args()
    run = Path(args.run)
    log = run / "auto_eval.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(l)["ckpt"] for l in log.read_text().splitlines() if l.strip()}
    while True:
        cks = sorted(run.glob("ckpt_u*.pt"))
        todo = [c for c in cks if c.name not in done]
        if not todo:
            time.sleep(60)
            continue
        ck = todo[0]
        time.sleep(5)  # let the trainer finish writing
        c = torch.load(ck, map_location="cpu", weights_only=False)
        c["config"].update(temperature=1.0, route_combat=False)
        tmp = run / f"eval_{ck.name}"
        torch.save(c, tmp)
        out = Path("training/runs/suite") / f"auto_{run.name}_{ck.stem}"
        subprocess.run(["rm", "-rf", str(out)])
        subprocess.run([sys.executable, "-m", "training.suite.run", "--agent", "--primary", str(tmp),
                        "--combat", "none", "--split", "val", "--per-category", str(args.per_category),
                        "--workers", str(args.workers), "--out", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmp = subprocess.run([sys.executable, "-m", "training.suite.compare", "--baseline", args.baseline,
                              "--candidate", str(out)], capture_output=True, text=True).stdout
        cand = re.search(r"candidate weighted proxy: ([+-][\d.]+)", cmp)
        delta = re.search(r"paired delta: \*\*([+-][\d.]+)\*\* \(95% CI ([+-][\d.]+) … ([+-][\d.]+)\)", cmp)
        dec = re.search(r"decision: \*\*(\w+)\*\*", cmp)
        row = {"ckpt": ck.name, "proxy": float(cand.group(1)) if cand else None,
               "delta": float(delta.group(1)) if delta else None,
               "ci": [float(delta.group(2)), float(delta.group(3))] if delta else None,
               "decision": dec.group(1) if dec else None, "exit": summary(out)}
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        tmp.unlink(missing_ok=True)
        done.add(ck.name)


if __name__ == "__main__":
    main()
