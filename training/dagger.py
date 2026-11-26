"""DAgger: run the student, relabel every visited state with the privileged expert.

Bootstraps recovery from the BC policy's covariate shift.  Shards use the same
format as `training.collect`, so they can be mixed into BC training directly.

Usage:
    python -m training.dagger --ckpt training/runs/bc_a/ckpt.pt --episodes 120 \
        --source gen --no-doors --no-monsters --out training/data/demos_dagger
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.collect import _blank, _exit_targets, _labels, run_episode
from training.evaluate import Runner, load_net
from training.maps.gen_wad import generate_map
from training.obs import flatten_obs, step_prev
from training.splits import FREEDOOM_TRAIN, GEN_TRAIN
from training.teacher.teacher import make_teacher
from submission.mapmem import MapBuilder


def dagger_episode(net, device: str, wad: str, map_name: str, seed: int,
                   max_steps: int = 1200, beta: float = 0.0,
                   episode_ticks: int = 4200, rng=None) -> tuple[dict, dict]:
    from doom_env import make_env

    rng = rng or np.random.default_rng(0)
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    priv = env.privileged_state()
    teacher = make_teacher(wad, map_name)
    teacher.reset((priv["x"], priv["y"]))
    targets = _exit_targets(teacher.md, teacher.grid, (priv["x"], priv["y"]))
    runner = Runner(net, device)
    buf = _blank(max_steps)
    buf["exec"] = np.zeros((max_steps, 9), np.uint8)
    steps = 0
    kills0 = priv["kills"]
    prev_action, turn_held, use_last = None, 0, 0
    builder = MapBuilder()
    health = []
    while steps < max_steps:
        label = teacher.act(obs, priv)
        student = runner.act(obs)
        action = label if rng.random() < beta else student
        builder.update(obs, prev_action)
        flat = flatten_obs(obs, prev_action, turn_held, use_last,
                           map_crop=builder.crop(), pose=builder.pose())
        vis, bearing, dbin, kcols = _labels(teacher.md, teacher.grid, (priv["x"], priv["y"]),
                                           float(priv["angle"]), targets)
        buf["dist"][steps] = flat["rays_dist"]
        buf["hit"][steps] = flat["rays_hit"]
        buf["efeat"][steps] = flat["ent_feat"]
        buf["etype"][steps] = flat["ent_type"]
        buf["emask"][steps] = flat["ent_mask"]
        buf["vitals"][steps] = flat["vitals"]
        buf["rgb"][steps] = flat["rgb"]
        buf["prev"][steps] = flat["prev_action"]
        buf["map"][steps] = flat["map"]
        buf["pose"][steps] = flat["pose"]
        buf["action"][steps] = label          # <-- teacher label, not the executed action
        buf["exec"][steps] = action
        buf["keys"][steps] = np.asarray(priv["keys"], np.uint8)
        buf["priv"][steps] = (priv["x"], priv["y"], priv["angle"],
                              priv["health"], priv["kills"], priv["damage_taken"])
        buf["exit_visible"][steps] = vis
        buf["exit_bearing"][steps] = bearing
        buf["exit_dist"][steps] = dbin
        buf["key_colours"][steps] = kcols
        health.append(float(priv["health"]))
        prev_action, turn_held, use_last = step_prev(action, prev_action, turn_held, use_last)
        obs, reward, term, trunc, info = env.step(action)
        steps += 1
        priv = env.privileged_state()
        if term or trunc:
            break
    while len(health) < steps:
        health.append(float(priv["health"]))
    hp = np.asarray(health, np.float32)
    dmg = np.maximum(0.0, hp - np.concatenate([hp[8:], np.full(min(8, steps), hp[-1])]))[:steps]
    buf["damage8"][:steps] = np.clip(dmg / 100.0, 0, 1).astype(np.float16)
    buf["done"][steps - 1] = True
    meta = {"steps": steps, "exit": bool(priv["exit_reached"]), "dead": bool(priv["dead"]),
            "kills": int(priv["kills"] - kills0), "health": float(priv["health"]),
            "doors": int(teacher.stats.doors_used), "ticks": int(priv["tic"]),
            "exit_visible_frac": float(buf["exit_visible"][:steps].mean())}
    env.close()
    return {k: v[:steps] for k, v in buf.items()}, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="training/runs/bc_a/ckpt.pt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="training/data/demos_dagger")
    ap.add_argument("--source", default="gen", choices=("gen", "freedoom_nav", "mix"))
    ap.add_argument("--tier", type=int, default=0)
    ap.add_argument("--no-doors", action="store_true")
    ap.add_argument("--no-monsters", action="store_true")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    net = load_net(args.ckpt, args.device)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = (out / "manifest.jsonl").open("a")
    t0 = time.time()
    exits = 0
    for i in range(args.episodes):
        src = args.source
        if src == "mix":
            src = "gen" if rng.random() < 0.5 else "freedoom_nav"
        if src == "gen":
            tier = args.tier or int(rng.integers(2, 5))
            seed = int(rng.integers(*GEN_TRAIN))
            spec = generate_map(seed, tier, "training/data/maps",
                                no_doors=args.no_doors, no_monsters=args.no_monsters)
            wad, map_name, tag = str(spec.path), spec.map_name, f"gen_t{tier}"
        else:
            name = FREEDOOM_TRAIN[i % len(FREEDOOM_TRAIN)]
            wad, map_name, tag = f"maps/freedoom_nav/{name}.wad", name, "freedoom_nav"
        data, meta = dagger_episode(net, args.device, wad, map_name, seed=i,
                                    max_steps=args.max_steps, beta=args.beta, rng=rng)
        np.savez_compressed(out / f"ep_{args.seed:04d}_{i:06d}.npz", **data)
        manifest.write(json.dumps({"file": f"ep_{args.seed:04d}_{i:06d}.npz", "wad": wad,
                                   "map": map_name, "tag": tag, "split": "train",
                                   "seed": args.seed, **meta}) + "\n")
        manifest.flush()
        exits += int(meta["exit"])
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{args.episodes} exits={exits} {time.time() - t0:.0f}s", flush=True)
    manifest.close()
    print(f"done: {args.episodes} episodes, {exits} exits, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
