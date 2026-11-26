"""Collect privileged-expert demonstrations for behaviour cloning.

Every shard stores the multimodal observation (rays, entities, vitals, 30x40 RGB),
the previous-action vector, the teacher's 9-button action, and auxiliary labels:

  exit_visible / exit_bearing(13 classes) / exit_distance(6 bins)
  key_colours(6)          - colour visible in the current view
  damage_next8            - health lost over the next 8 decisions

Labels are computed from privileged map geometry and pose (training only).

Usage:
    python -m training.collect --episodes 200 --source gen --no-doors --no-monsters --split train
    python -m training.collect --episodes 200 --source freedoom_nav --split train
    python -m training.collect --episodes 200 --source drill --split train
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from doom_env import make_env
from training.maps.gen_wad import generate_map
from training.obs import flatten_obs, step_prev
from training.splits import FREEDOOM_TRAIN, GEN_TRAIN, GEN_VAL
from training.teacher.mapdata import SWITCH_EXITS
from training.teacher.teacher import make_teacher
from submission.mapmem import MapBuilder

LOCK_TO_COLOUR = {2: 0, 3: 1, 1: 2}  # blue / yellow / red card channels
EXIT_BINS = 12
DIST_EDGES = (256.0, 512.0, 768.0, 1024.0, 1400.0)


def _blank(T: int) -> dict:
    return {
        "dist": np.zeros((T, 64), np.float16),
        "hit": np.zeros((T, 64), np.uint8),
        "efeat": np.zeros((T, 32, 5), np.float16),
        "etype": np.zeros((T, 32), np.uint8),
        "emask": np.zeros((T, 32), np.uint8),
        "vitals": np.zeros((T, 17), np.float32),
        "rgb": np.zeros((T, 3, 30, 40), np.uint8),
        "prev": np.zeros((T, 12), np.float16),
        "action": np.zeros((T, 9), np.uint8),
        "reward": np.zeros(T, np.float32),
        "done": np.zeros(T, bool),
        "keys": np.zeros((T, 6), np.uint8),
        "priv": np.zeros((T, 6), np.float32),
        "exit_visible": np.zeros(T, np.uint8),
        "exit_bearing": np.zeros(T, np.uint8),
        "exit_dist": np.zeros(T, np.uint8),
        "key_colours": np.zeros((T, 6), np.uint8),
        "damage8": np.zeros(T, np.float16),
        "map": np.zeros((T, 5, 32, 32), np.uint8),
        "pose": np.zeros((T, 6), np.float16),
    }


def _exit_targets(md, grid, pos):
    out = []
    for x1, y1, x2, y2, sp in md.exit_lines():
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = dy / ln, -dx / ln
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if sp in SWITCH_EXITS:
            for side in (1.0, -1.0):
                tx, ty = mx + nx * 56 * side, my + ny * 56 * side
                ix, iy = grid.world_to_cell(tx, ty)
                if 0 <= ix < grid.w and 0 <= iy < grid.h and grid.free is not None and grid.free[iy, ix]:
                    out.append((tx, ty))
                    break
        else:
            out.append((mx, my))
    return out


def _labels(md, grid, pos, angle, targets):
    """Nearest visible exit + visible key colours from privileged geometry."""
    visible, bearing, dist_bin = 0, EXIT_BINS, 5
    best = None
    for tx, ty in targets:
        dx, dy = tx - pos[0], ty - pos[1]
        d = math.hypot(dx, dy)
        if d > 1400 or d < 1:
            continue
        rel = (math.degrees(math.atan2(dy, dx)) - angle + 180.0) % 360.0 - 180.0
        if abs(rel) > 55.0:
            continue
        if not grid.visible(pos, (tx, ty)):
            continue
        if best is None or d < best[0]:
            best = (d, rel)
    if best is not None:
        visible = 1
        bearing = int(((best[1] + 180.0) / 360.0) * EXIT_BINS) % EXIT_BINS
        dist_bin = int(sum(best[0] > e for e in DIST_EDGES))
    key_cols = np.zeros(6, np.float32)
    for kx, ky, lock in md.key_positions():
        dx, dy = kx - pos[0], ky - pos[1]
        d = math.hypot(dx, dy)
        if d > 1400 or d < 1:
            continue
        rel = (math.degrees(math.atan2(dy, dx)) - angle + 180.0) % 360.0 - 180.0
        if abs(rel) > 60.0 or not grid.visible(pos, (kx, ky)):
            continue
        col = LOCK_TO_COLOUR.get(lock)
        if col is not None:
            key_cols[col] = 1.0
    return visible, bearing, dist_bin, key_cols


def run_episode(wad: str, map_name: str, seed: int, max_steps: int = 1200,
                episode_ticks: int = 4200) -> tuple[dict, dict]:
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    priv = env.privileged_state()
    teacher = make_teacher(wad, map_name)
    teacher.reset((priv["x"], priv["y"]))
    targets = _exit_targets(teacher.md, teacher.grid, (priv["x"], priv["y"]))
    buf = _blank(max_steps)
    steps = 0
    kills0 = priv["kills"]
    prev_action, turn_held, use_last = None, 0, 0
    builder = MapBuilder()
    health = []
    while steps < max_steps:
        action = teacher.act(obs, priv)
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
        buf["action"][steps] = action
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
        buf["reward"][steps] = reward
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
    meta = {
        "steps": steps,
        "exit": bool(priv["exit_reached"]),
        "dead": bool(priv["dead"]),
        "kills": int(priv["kills"] - kills0),
        "health": float(priv["health"]),
        "doors": int(teacher.stats.doors_used),
        "ticks": int(priv["tic"]),
        "exit_visible_frac": float(buf["exit_visible"][:steps].mean()),
    }
    env.close()
    return {k: v[:steps] for k, v in buf.items()}, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="training/data/demos")
    ap.add_argument("--source", default="gen", choices=("gen", "freedoom", "freedoom_nav", "drill", "roomdoor", "mix"))
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--tier", type=int, default=0, help="0 = randomise 2..4")
    ap.add_argument("--no-doors", action="store_true")
    ap.add_argument("--no-monsters", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--drill-doors", type=int, default=2)
    ap.add_argument("--drill-key", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = (out / "manifest.jsonl").open("a")
    freedoom = [Path(f"maps/freedoom_nav/{m}.wad") for m in FREEDOOM_TRAIN]
    freedoom_mon = [Path(f"maps/freedoom_open/{m}.wad") for m in FREEDOOM_TRAIN]
    rng = np.random.default_rng(args.seed)
    seed0, seed1 = GEN_TRAIN if args.split == "train" else GEN_VAL

    t0 = time.time()
    n_exit = 0
    for i in range(args.episodes):
        src = args.source
        if src == "mix":
            src = "gen" if rng.random() < 0.5 else "freedoom_nav"
        if src == "roomdoor":
            from training.maps.doorroom import generate_room_door
            spec = generate_room_door(args.seed * 100000 + i, "training/data/maps",
                                      key=False, two_doors=bool(i % 2))
            wad, map_name, tag = str(spec.path), spec.map_name, "roomdoor"
        elif src == "drill":
            from training.maps.drills import generate_drill
            spec = generate_drill(args.seed * 100000 + i, "training/data/maps",
                                  n_doors=args.drill_doors, key=args.drill_key)
            wad, map_name, tag = str(spec.path), spec.map_name, "drill"
        elif src == "gen":
            tier = args.tier or int(rng.integers(2, 5))
            seed = int(rng.integers(seed0, seed1 + 1))
            spec = generate_map(seed, tier, "training/data/maps",
                                no_doors=args.no_doors, no_monsters=args.no_monsters)
            wad, map_name, tag = str(spec.path), spec.map_name, f"gen_t{tier}"
        elif src in ("freedoom_nav", "freedoom"):
            pool = freedoom if src == "freedoom_nav" else freedoom_mon
            if not pool:
                raise SystemExit("run python -m training.maps.prepare_freedoom first")
            path = pool[i % len(pool)]
            wad, map_name, tag = str(path), path.stem, src
        else:
            raise SystemExit(f"unknown source {src}")

        data, meta = run_episode(wad, map_name, seed=i, max_steps=args.max_steps)
        name = f"ep_{args.seed:04d}_{i:06d}.npz"
        np.savez_compressed(out / name, **data)
        n_exit += int(meta["exit"])
        manifest.write(json.dumps({"file": name, "wad": wad, "map": map_name, "tag": tag,
                                   "split": args.split, "seed": args.seed, **meta}) + "\n")
        manifest.flush()
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{args.episodes}  exits={n_exit}  {time.time() - t0:.0f}s", flush=True)
    manifest.close()
    print(f"done: {args.episodes} episodes, {n_exit} exits, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
