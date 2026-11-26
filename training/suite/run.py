"""Run a checkpoint on a suite split and write one JSONL row per episode.

Episodes are sharded across CPU workers; each worker loads the model once.  The
test split is protected: it needs ``--allow-test`` and appends to
``test_touch.log`` so the number of test looks stays honest.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training.evaluate import load_net, run_agent_episode

_WORKER_NET = None
_WORKER_AGENT = None
_WORKER_DEVICE = "cpu"


def shortest_path(wad: str, map_name: str, start: tuple[float, float], cache: dict) -> float | None:
    key = (wad, map_name, start)
    if key in cache:
        return cache[key]
    try:
        from training.teacher.teacher import make_teacher
        teacher = make_teacher(wad, map_name)
        teacher.reset(start)
        best = None
        for gx, gy, kind, radius in teacher.grid.exit_goals:
            p = teacher.grid.path(start, (gx, gy), keys=np.ones(6, np.float32))
            if p is None:
                continue
            length = sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
                         for i in range(len(p) - 1))
            best = length if best is None else min(best, length)
        cache[key] = best
        return best
    except Exception:
        cache[key] = None
        return None


def _init_worker(ckpt: str, device: str, use_agent: bool = False, overrides: dict | None = None) -> None:
    global _WORKER_NET, _WORKER_AGENT, _WORKER_DEVICE
    import torch
    torch.set_num_threads(1)
    if use_agent:
        from types import SimpleNamespace
        import submission.agent as agent_mod
        from submission.agent import DoomAgent
        for name, val in (overrides or {}).items():
            if val:
                setattr(agent_mod, name, Path(val) if val != "none" else Path("/nonexistent"))
        _WORKER_AGENT = DoomAgent(None, SimpleNamespace(shape=(9,)), device)
        _WORKER_NET = None
    else:
        _WORKER_NET = load_net(ckpt, device)
    _WORKER_DEVICE = device


def _run_shard(tasks: list[dict]) -> tuple[int, list[dict]]:
    rows = []
    cache = {}
    for t in tasks:
        r = run_agent_episode(_WORKER_NET, t["wad"], t["map_name"], device=_WORKER_DEVICE,
                              ticks=t["ticks"], seed=t["engine_seed"], record_traj=True,
                              track_sectors=True, agent=_WORKER_AGENT)
        r["category"] = t["category"]
        r["tier"] = t["tier"]
        r["map_id"] = t["map_id"]
        r["engine_seed"] = t["engine_seed"]
        r["spec_hash"] = t["spec_hash"]
        start = tuple(r.get("start", (0.0, 0.0)))
        s = shortest_path(t["wad"], t["map_name"], start, cache)
        r["spl"] = (s / max(r.get("path_len", s), s)) if (r["exit"] and s) else 0.0
        rows.append({k: v for k, v in r.items() if k != "frames"})
    return len(rows), rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit-maps", type=int, default=0)
    ap.add_argument("--categories", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="first 5 maps per category, 1 seed")
    ap.add_argument("--per-category", type=int, default=0)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--agent", action="store_true", help="run submission/agent.py end-to-end")
    ap.add_argument("--no-spl", action="store_true")
    ap.add_argument("--primary", default="", help="--agent: override weights/doomforge.pt")
    ap.add_argument("--combat", default="", help="--agent: override weights/combat.pt ('none' to disable)")
    ap.add_argument("--specialist", default="", help="--agent: override weights/specialist.pt ('none')")
    args = ap.parse_args()
    overrides = {"PRIMARY": args.primary, "COMBAT": args.combat, "SPECIALIST": args.specialist}

    if not args.agent and not args.ckpt:
        raise SystemExit("--ckpt is required unless --agent is used")
    if args.split in ("test", "ood") and not args.allow_test:
        raise SystemExit("refusing to touch the test split without --allow-test "
                         "(every look is logged in test_touch.log)")

    root = Path(args.out or f"training/runs/suite/{args.split}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("training/data/suite") / args.split / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if args.categories:
        cats = set(args.categories.split(","))
        rows = [r for r in rows if r["category"] in cats]
    if args.fast or args.per_category:
        n_cat = 5 if args.fast else args.per_category
        per_cat: dict[str, list[dict]] = {}
        for r in rows:
            per_cat.setdefault(r["category"], []).append(r)
        rows = [r for cat in sorted(per_cat) for r in per_cat[cat][:n_cat]]
    if args.limit_maps:
        rows = rows[: args.limit_maps]

    out = root / "episodes.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["map_id"], int(r["engine_seed"])))
            except Exception:
                pass
    tasks = []
    for row in rows:
        for seed in range(args.seeds):
            if (row["map_id"], seed) in done:
                continue
            tasks.append({**row, "engine_seed": seed})
    if args.no_spl:
        for t in tasks:
            t["no_spl"] = True

    if args.split in ("test", "ood"):
        stamp = datetime.now(timezone.utc).isoformat()
        with Path("test_touch.log").open("a") as f:
            f.write(f"{stamp} split={args.split} ckpt={Path(args.ckpt).name} "
                    f"episodes={len(tasks)} workers={args.workers} fast={args.fast} "
                    f"note=suite_run\n")

    t0 = time.time()
    n_done = 0
    if args.workers <= 1 or len(tasks) <= args.workers:
        _init_worker(args.ckpt, args.device, args.agent, overrides)
        n, shard = _run_shard(tasks)
        with out.open("a") as f:
            for r in shard:
                f.write(json.dumps(r) + "\n")
        n_done = n
        print(f"{n_done}/{len(tasks)} episodes", flush=True)
    else:
        k = max(1, math.ceil(len(tasks) / args.workers))
        chunks = [tasks[i:i + k] for i in range(0, len(tasks), k)]
        with mp.get_context("spawn").Pool(len(chunks), initializer=_init_worker,
                                          initargs=(args.ckpt, args.device, args.agent, overrides)) as pool:
            results = pool.map(_run_shard, chunks)
        with out.open("a") as f:
            for n, shard in results:
                for r in shard:
                    f.write(json.dumps(r) + "\n")
                n_done += n
        print(f"{n_done}/{len(tasks)} episodes in {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out} ({n_done} new)")


if __name__ == "__main__":
    main()
