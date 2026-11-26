"""Broad held-out benchmark (fixed seeds, never used for training).

Suites:
  t1  arena combat          12 maps, 2100 ticks
  t2  multi-room exploration 12 maps, 3150 ticks
  t3  key -> locked door     12 maps, 4200 ticks
  t4  full gauntlet          8 maps, 4200 ticks
  drills  door/key isolation 12 maps, 2100 ticks
  freedoom MAP23-32 raw      10 maps, 4200 ticks

Usage:
    python -m training.bench --ckpt training/runs/bc_map2/ckpt_e0.pt --out training/runs/bench_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.evaluate import load_net, run_agent_episode
from training.maps.drills import generate_drill
from training.maps.gen_wad import generate_map
from training.splits import FREEDOOM_EVAL

BENCH_SEEDS = {
    "t1": (30_000, 12, 1, 2100),
    "t2": (30_100, 12, 2, 3150),
    "t3": (30_200, 12, 3, 4200),
    "t4": (30_300, 8, 4, 4200),
}


def build_suite(n_maps: int) -> list[tuple[str, str, str, int]]:
    out = []
    tmp = "training/data/bench_maps"
    for label, (seed0, n, tier, ticks) in BENCH_SEEDS.items():
        for i in range(min(n, n_maps)):
            spec = generate_map(seed0 + i, tier, tmp)
            out.append((label, str(spec.path), spec.map_name, ticks))
    for i in range(min(12, n_maps)):
        kw = [(1, False), (1, True), (2, False)][i % 3]
        spec = generate_drill(30_400 + i, tmp, n_doors=kw[0], key=kw[1])
        out.append(("drills", str(spec.path), spec.map_name, 2100))
    for name in FREEDOOM_EVAL:
        p = Path(f"maps/freedoom/{name}.wad")
        if p.exists():
            out.append(("freedoom", str(p), name, 4200))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="training/runs/bench")
    ap.add_argument("--maps", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-freedoom", action="store_true")
    ap.add_argument("--agent", action="store_true")
    args = ap.parse_args()

    if not args.agent and not args.ckpt:
        raise SystemExit("--ckpt is required unless --agent is used")
    agent = None
    if args.agent:
        from types import SimpleNamespace
        from submission.agent import DoomAgent
        agent = DoomAgent(None, SimpleNamespace(shape=(9,)), args.device)
        net = None
    else:
        net = load_net(args.ckpt, args.device)
    suite = build_suite(args.maps)
    if args.no_freedoom:
        suite = [s for s in suite if s[0] != "freedoom"]
    rows = []
    for label, wad, map_name, ticks in suite:
        r = run_agent_episode(net, wad, map_name, device=args.device, seed=0, ticks=ticks,
                              agent=agent)
        r["label"] = label
        rows.append(r)
        print(f"{label:8s} {Path(wad).stem:12s} exit={int(r['exit'])} steps={r['steps']:4d} "
              f"kills={r['kills']:2d} keys={r['keys']} doors={r['doors']} "
              f"cov={r['coverage']:5.1%} proxy={r['proxy']:6.2f}", flush=True)

    summary = {}
    for r in rows:
        s = summary.setdefault(r["label"], {"n": 0, "exit": 0, "proxy": 0.0, "kills": 0,
                                            "keys": 0, "doors": 0, "cov": 0.0, "p99": 0.0})
        s["n"] += 1
        s["exit"] += int(r["exit"])
        s["proxy"] += r["proxy"]
        s["kills"] += r["kills"]
        s["keys"] += r["keys"]
        s["doors"] += r["doors"]
        s["cov"] += r["coverage"]
        s["p99"] += r["p99_ms"]
    print("\n| suite | n | exit | proxy | kills | keys | doors | coverage | p99 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in summary.items():
        n = max(1, s["n"])
        print(f"| {k} | {s['n']} | {s['exit']}/{s['n']} | {s['proxy']/n:+.2f} | {s['kills']} | "
              f"{s['keys']} | {s['doors']} | {s['cov']/n:.1%} | {s['p99']/n:.1f} ms |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
