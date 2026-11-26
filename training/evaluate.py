"""Run a trained DoomNet in the official environment and compute proxy metrics.

Usage:
    python -m training.evaluate --ckpt training/runs/bc_nav/ckpt.pt --suite gen_val
    python -m training.evaluate --ckpt ... --suite freedoom_eval --episodes 10
    python -m training.evaluate --ckpt ... --wad maps/freedoom/MAP23.wad --record out.gif
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from submission.model import DoomNet
from submission.model_v1 import DoomNet as DoomNetV1
from training.coverage import SectorTracker
from training.maps.gen_wad import generate_map
from training.maps.drills import generate_drill
from training.obs import flatten_obs, stack_obs, step_prev
from submission.mapmem import MapBuilder
from training.splits import FREEDOOM_EVAL

DEFAULT_WEIGHTS = {
    "exit": 10.0,
    "key": 3.0,
    "lock": 3.0,
    "door": 0.5,
    "sector": 0.3,
    "sector_cap": 10.0,
    "kill": 0.6,
    "damage": 0.01,
    "death_base": 1.0,
    "death_early": 3.0,
}


class Runner:
    def __init__(self, net: DoomNet, device: str = "cpu"):
        self.net = net
        self.device = device
        self.h = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map = MapBuilder()

    def reset(self) -> None:
        self.h = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map.reset()

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        self.map.update(obs, self.prev)
        flat = flatten_obs(obs, self.prev, self.turn_held, self.use_last,
                           map_crop=self.map.crop(), pose=self.map.pose())
        batch = stack_obs([flat], self.device)
        logits, value, aux, self.h = self.net.forward_step(batch, self.h)
        heads = self.net.flat_action(logits, sample=False)
        a = self.net.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)
        self.prev, self.turn_held, self.use_last = step_prev(a, self.prev, self.turn_held, self.use_last)
        return a


def _door_summary(env, ceilings, floors):
    opened = locks = 0
    for sec, lock in env.mapinfo.door_sectors.items():
        if sec < len(ceilings) and (ceilings[sec] - floors[sec]) >= 56:
            opened += 1
            if lock > 0:
                locks += 1
    return opened, locks


def run_agent_episode(net: DoomNet, wad: str, map_name: str = "MAP01", device: str = "cpu",
                      ticks: int = 4200, seed: int = 0, record: bool = False,
                      sample: bool = False, throttle_ms: float = 0.0,
                      weights: dict | None = None, record_traj: bool = False,
                      track_sectors: bool = True, obs_transform=None,
                      memory_wipe_at_key: bool = False, agent=None) -> dict:
    from doom_env import make_env

    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    runner = Runner(net, device) if agent is None else None
    tracker = SectorTracker(env) if track_sectors else None
    if tracker is not None:
        tracker.update(p0["x"], p0["y"])
    frames = []
    lat = []
    steps = 0
    if agent is not None:
        agent.reset()
    elif net is not None:
        net.eval()
    prev_pos = (p0["x"], p0["y"])
    prev_angle = p0["angle"]
    prev_keys = int((np.asarray(p0["keys"]) > 0.5).sum())
    path_len = 0.0
    spin_steps = noop_steps = 0
    unique_cells: set[tuple[int, int]] = set()
    actions_hash = __import__("hashlib").sha1()
    wiped = False
    while True:
        if throttle_ms:
            time.sleep(throttle_ms / 1000.0)
        if obs_transform is not None:
            obs = obs_transform(obs, steps)
        t0 = time.perf_counter()
        action = runner.act(obs) if runner is not None else agent.act(obs)
        lat.append((time.perf_counter() - t0) * 1000)
        actions_hash.update(np.asarray(action, np.int8).tobytes())
        if record:
            frames.append(np.asarray(obs["rgb"]).copy())
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        p = env.privileged_state()
        dx, dy = p["x"] - prev_pos[0], p["y"] - prev_pos[1]
        disp = float(np.hypot(dx, dy))
        path_len += disp
        if disp < 2.0 and abs((p["angle"] - prev_angle + 180) % 360 - 180) > 1.0:
            spin_steps += 1
        if int(np.asarray(action).sum()) == 0:
            noop_steps += 1
        unique_cells.add((int(p["x"] // 64), int(p["y"] // 64)))
        prev_pos, prev_angle = (p["x"], p["y"]), p["angle"]
        if tracker is not None:
            tracker.update(p["x"], p["y"])
        keys_now = int((np.asarray(p["keys"]) > 0.5).sum())
        if memory_wipe_at_key and not wiped and keys_now > prev_keys and runner is not None:
            runner.reset()
            wiped = True
        prev_keys = keys_now
        if term or trunc:
            break
    p = env.privileged_state()
    doors, locks = _door_summary(env, np.asarray(p["sector_ceiling"]), np.asarray(p["sector_floor"]))
    keys = int((np.asarray(p["keys"]) > 0.5).sum())
    total_sectors = max(1, env.mapinfo.num_sectors)
    if tracker is not None:
        coverage = len(tracker.visited) / total_sectors
        sectors = len(tracker.visited)
    else:
        coverage = 0.0
        sectors = len(unique_cells)
    comp = {
        "exit": weights["exit"] * float(p["exit_reached"]),
        "keys": weights["key"] * keys,
        "locks": weights["lock"] * locks,
        "doors": weights["door"] * max(0, doors - locks),
        "sectors": min(weights["sector"] * sectors, weights["sector_cap"]),
        "kills": weights["kill"] * max(0, p["kills"] - p0["kills"]),
        "damage": -weights["damage"] * float(p["damage_taken"]),
        "death": 0.0,
    }
    if p["dead"]:
        frac = steps / max(1, ticks // 4)
        comp["death"] = -(weights["death_early"] * (1.0 - frac) + weights["death_base"])
    out = {
        "wad": str(wad),
        "map": map_name,
        "seed": seed,
        "exit": bool(p["exit_reached"]),
        "dead": bool(p["dead"]),
        "steps": steps,
        "ticks": p["tic"],
        "kills": int(p["kills"] - p0["kills"]),
        "health": float(p["health"]),
        "damage": float(p["damage_taken"]),
        "keys": keys,
        "doors": int(doors),
        "locks": int(locks),
        "sectors": sectors,
        "coverage": coverage,
        "proxy": float(sum(comp.values())),
        "components": comp,
        "p50_ms": float(np.median(lat)) if lat else 0.0,
        "p99_ms": float(np.percentile(lat, 99)) if lat else 0.0,
        "frames": frames,
        "path_len": path_len,
        "spin_frac": spin_steps / max(1, steps),
        "noop_frac": noop_steps / max(1, steps),
        "unique_cells": len(unique_cells),
        "action_hash": actions_hash.hexdigest(),
        "memory_wiped": bool(wiped),
        "start": [float(p0["x"]), float(p0["y"])],
    }
    env.close()
    return out


def load_net(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cls = DoomNet if "goal_head.net.0.weight" in sd else DoomNetV1
    net = cls(hidden=ckpt.get("config", {}).get("hidden", 384) if isinstance(ckpt, dict) else 384)
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


def suite_maps(suite: str, n: int) -> list[tuple[str, str, str]]:
    """(wad, map, label) tuples. Generated maps are written under training/data/eval_maps."""
    from pathlib import Path as _P
    out = []
    tmp = "training/data/eval_maps"
    if suite in ("gen_val", "all"):
        for tier in (1, 2, 3, 4):
            for k in range(n):
                spec = generate_map(10_000 + k + tier * 100, tier, tmp)
                out.append((str(spec.path), "MAP01", f"gen_t{tier}"))
    if suite in ("drills", "all"):
        tmpd = "training/data/eval_maps"
        specs = [("drill_1", dict(n_doors=1, key=False)),
                 ("drill_key", dict(n_doors=1, key=True)),
                 ("drill_2", dict(n_doors=2, key=False))]
        for k in range(n):
            for i, (label, kw) in enumerate(specs):
                spec = generate_drill(9000 + k * 10 + i, tmpd, **kw)
                out.append((str(spec.path), spec.map_name, label))
    if suite in ("freedoom_eval", "all"):
        for name in FREEDOOM_EVAL:
            p = _P(f"maps/freedoom/{name}.wad")
            if p.exists():
                out.append((str(p), name, "freedoom_eval"))
    if suite in ("ood", "all"):
        for k in range(n):
            spec = generate_map(20_000 + k, 4, tmp)
            out.append((str(spec.path), "MAP01", "ood_t4"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="training/runs/bc_nav/ckpt.pt")
    ap.add_argument("--suite", default="gen_val", choices=("gen_val", "freedoom_eval", "ood", "drills", "all"))
    ap.add_argument("--episodes", type=int, default=4, help="maps per tier for generated suites")
    ap.add_argument("--seeds", type=int, default=1, help="engine seeds per map")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="training/runs/eval")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_net(args.ckpt, dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for wad, map_name, label in suite_maps(args.suite, args.episodes):
        if not wad or not Path(wad).exists():
            continue
        for s in range(args.seeds):
            rec = args.record and s == 0
            r = run_agent_episode(net, wad, map_name, device=dev, seed=s, record=rec)
            r["label"] = label
            rows.append(r)
            print(f"{label:14s} {Path(wad).stem:10s} exit={int(r['exit'])} "
                  f"steps={r['steps']:4d} kills={r['kills']} keys={r['keys']} "
                  f"sectors={r['sectors']:3d} proxy={r['proxy']:6.2f} p99={r['p99_ms']:.1f}ms",
                  flush=True)

    summary = {}
    for r in rows:
        s = summary.setdefault(r["label"], {"n": 0, "exit": 0, "proxy": 0.0, "kills": 0,
                                            "coverage": 0.0, "p99": 0.0, "keys": 0})
        s["n"] += 1
        s["exit"] += int(r["exit"])
        s["proxy"] += r["proxy"]
        s["kills"] += r["kills"]
        s["keys"] += r["keys"]
        s["coverage"] += r["coverage"]
        s["p99"] += r["p99_ms"]
    for k, s in summary.items():
        n = max(1, s["n"])
        print(f"SUMMARY {k:14s} n={s['n']} exit={s['exit']}/{s['n']} "
              f"proxy={s['proxy']/n:.2f} kills={s['kills']} keys={s['keys']} "
              f"coverage={s['coverage']/n:.2%} p99={s['p99']/n:.1f}ms")

    clean = [{k: v for k, v in r.items() if k != "frames"} for r in rows]
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": clean}, indent=1))
    if args.record:
        frames = [r["frames"] for r in rows if r.get("frames")]
        (out / "frames.json").write_text(json.dumps({"count": len(frames)}))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
