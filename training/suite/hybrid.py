"""Perception-routed hybrid: PPO-25 for visible combat, Atlas-G otherwise.

This is an experiment harness.  It runs both policies on every observation (so
both map memories stay current), routes the executed action, and reports the
category outcome.  No privileged state is used by the router.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.evaluate import DEFAULT_WEIGHTS, Runner, SectorTracker, _door_summary, load_net
from doom_env import make_env

ENEMY_TYPES = (1, 2, 3)


def enemies_visible(obs: dict) -> bool:
    mask = np.asarray(obs["visible_entities"]["mask"]) > 0
    types = np.asarray(obs["visible_entities"]["entity_type"])
    return bool(np.any(mask & np.isin(types, ENEMY_TYPES)))


def run_hybrid(combat_net, nav_net, wad: str, map_name: str, ticks: int, seed: int = 0,
               hysteresis: int = 16):
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    combat = Runner(combat_net, "cpu")
    nav = Runner(nav_net, "cpu")
    tracker = SectorTracker(env)
    tracker.update(p0["x"], p0["y"])
    last_route = None
    routes = 0
    steps = 0
    combat_hold = 0
    while True:
        a_combat = combat.act(obs)
        a_nav = nav.act(obs)
        combat_hold = hysteresis if enemies_visible(obs) else max(0, combat_hold - 1)
        route = "combat" if combat_hold > 0 else "nav"
        if route != last_route:
            routes += 1
            last_route = route
        action = a_combat if route == "combat" else a_nav
        obs, _, term, trunc, _ = env.step(action)
        steps += 1
        p = env.privileged_state()
        tracker.update(p["x"], p["y"])
        if term or trunc:
            break
    p = env.privileged_state()
    doors, locks = _door_summary(env, np.asarray(p["sector_ceiling"]), np.asarray(p["sector_floor"]))
    keys = int((np.asarray(p["keys"]) > 0.5).sum())
    total = max(1, env.mapinfo.num_sectors)
    comp = {
        "exit": DEFAULT_WEIGHTS["exit"] * float(p["exit_reached"]),
        "keys": DEFAULT_WEIGHTS["key"] * keys,
        "locks": DEFAULT_WEIGHTS["lock"] * locks,
        "doors": DEFAULT_WEIGHTS["door"] * max(0, doors - locks),
        "sectors": min(DEFAULT_WEIGHTS["sector"] * len(tracker.visited), DEFAULT_WEIGHTS["sector_cap"]),
        "kills": DEFAULT_WEIGHTS["kill"] * max(0, p["kills"] - p0["kills"]),
        "damage": -DEFAULT_WEIGHTS["damage"] * float(p["damage_taken"]),
    }
    if p["dead"]:
        frac = steps / max(1, ticks // 4)
        comp["death"] = -(DEFAULT_WEIGHTS["death_early"] * (1.0 - frac) + DEFAULT_WEIGHTS["death_base"])
    out = {"exit": bool(p["exit_reached"]), "steps": steps, "keys": keys, "locks": int(locks),
           "kills": int(p["kills"] - p0["kills"]), "damage": float(p["damage_taken"]),
           "coverage": len(tracker.visited) / total, "proxy": float(sum(comp.values())),
           "routes": routes}
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combat", default="training/runs/ppo_explore/ckpt.pt")
    ap.add_argument("--nav", default="training/runs/bc_v5/ckpt.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--categories", default="C0_sanity,C1_duel,C2_arena,C4_room_door,C5_key_lock,C6_two_key,C8_backtrack")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--out", default="training/runs/suite/hybrid_probe")
    ap.add_argument("--hysteresis", type=int, default=16)
    args = ap.parse_args()

    combat = load_net(args.combat)
    nav = load_net(args.nav)
    rows = [json.loads(l) for l in
            (Path("training/data/suite") / args.split / "manifest.jsonl").read_text().splitlines() if l.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for cat in args.categories.split(","):
        cat_rows = [r for r in rows if r["category"] == cat][: args.per_category]
        vals = []
        for row in cat_rows:
            r = run_hybrid(combat, nav, row["wad"], row["map_name"], row["ticks"], 0,
                       hysteresis=args.hysteresis)
            r["category"] = cat
            r["map_id"] = row["map_id"]
            vals.append(r)
            results.append(r)
        print(f"{cat}: exits={sum(int(v['exit']) for v in vals)}/{len(vals)} "
              f"proxy={np.mean([v['proxy'] for v in vals]):+.2f} "
              f"keys={np.mean([v['keys'] for v in vals]):.2f} "
              f"kills={np.mean([v['kills'] for v in vals]):.1f} "
              f"routes={np.mean([v['routes'] for v in vals]):.1f}", flush=True)
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print("wrote", out / "episodes.jsonl")


if __name__ == "__main__":
    main()
