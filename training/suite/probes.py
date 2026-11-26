"""Diagnostic probes: door angles, exit sign, colour, corruption, memory, determinism."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

from training.evaluate import load_net, run_agent_episode
from training.maps.gen_wad import generate_map, TILE, _build_lumps, _write_wad
from training.maps.doorroom import generate_room_door
from training.suite.spec import rebuild, set_start


def _run(net, wad, ticks=2100, seed=0, **kw):
    return run_agent_episode(net, wad, "MAP01", device="cpu", ticks=ticks, seed=seed,
                             record_traj=True, **kw)


def door_angle_maps(out: Path, n_angles: int = 8, n_seeds: int = 3) -> list[tuple[str, int]]:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in range(n_seeds):
        for a in range(n_angles):
            angle = int(a * 360 / n_angles)
            spec = generate_room_door(9000 + s * 10 + a, str(out), key=False, two_doors=False)
            set_start(spec, angle=angle)
            rebuild(spec)
            rows.append((str(spec.path), angle))
    return rows


def c5_maps(out: Path, n: int, sign: str | None = None, door_tex: str | None = None,
            no_monsters: bool = True) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        spec = generate_map(7000 + i, 3, str(out), no_monsters=no_monsters)
        spec.exit_sign_texture = sign
        spec.door_texture_override = door_tex
        rebuild(spec)
        paths.append(str(spec.path))
    return paths


def corrupt_obs(obs: dict, step: int) -> dict:
    o = {k: (copy.deepcopy(v) if isinstance(v, dict) else v.copy()) for k, v in obs.items()}
    d = o["rays"]["distance"]
    d[::10] = 1.0
    o["rays"]["hit_type"][::10] = 0
    o["rgb"] = (o["rgb"].astype(np.float32) * 0.75).astype(np.uint8)
    o["rgb"][::2, ::2] = 0
    o["visible_entities"]["features"][:, 0] *= 0.5
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="training/runs/probes")
    ap.add_argument("--probes", default="door_angles,exit_sign,colour_swap,corruption,determinism,memory_wipe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(args.ckpt)
    want = set(args.probes.split(","))
    report = {}

    if "door_angles" in want:
        rows = door_angle_maps(out / "door_angles")
        ok = 0
        per_angle = {}
        for wad, angle in rows:
            r = _run(net, wad)
            ok += int(r["exit"])
            per_angle.setdefault(angle, []).append(int(r["exit"]))
        report["door_angles"] = {"n": len(rows), "success": ok / max(1, len(rows)),
                                 "per_angle": {k: sum(v) / len(v) for k, v in per_angle.items()}}
        print("door_angles", report["door_angles"], flush=True)

    if "exit_sign" in want:
        normal = c5_maps(out / "sign_normal", 6, sign=None)
        hidden = c5_maps(out / "sign_hidden", 6, sign="STONE2")
        a = np.mean([int(_run(net, w, ticks=3150)["exit"]) for w in normal])
        b = np.mean([int(_run(net, w, ticks=3150)["exit"]) for w in hidden])
        report["exit_sign"] = {"normal": float(a), "ablated": float(b), "drop": float(a - b)}
        print("exit_sign", report["exit_sign"], flush=True)

    if "colour_swap" in want:
        normal = c5_maps(out / "colour_normal", 6, door_tex=None)
        swapped = c5_maps(out / "colour_swap", 6, door_tex="DOORBLU")
        a = np.mean([_run(net, w, ticks=3150)["locks"] for w in normal])
        b = np.mean([_run(net, w, ticks=3150)["locks"] for w in swapped])
        report["colour_swap"] = {"normal_locks": float(a), "swapped_locks": float(b)}
        print("colour_swap", report["colour_swap"], flush=True)

    if "corruption" in want:
        maps = c5_maps(out / "corruption", 4, no_monsters=False)
        clean = [_run(net, w, ticks=3150)["proxy"] for w in maps]
        corr = [_run(net, w, ticks=3150, obs_transform=corrupt_obs)["proxy"] for w in maps]
        report["corruption"] = {"clean_proxy": float(np.mean(clean)),
                                "corrupt_proxy": float(np.mean(corr))}
        print("corruption", report["corruption"], flush=True)

    if "determinism" in want:
        wad = c5_maps(out / "determinism", 1, no_monsters=False)[0]
        a = _run(net, wad, ticks=3150, seed=0)
        b = _run(net, wad, ticks=3150, seed=0)
        report["determinism"] = {"same_actions": a["action_hash"] == b["action_hash"],
                                 "exit_a": a["exit"], "exit_b": b["exit"],
                                 "steps_a": a["steps"], "steps_b": b["steps"]}
        print("determinism", report["determinism"], flush=True)

    if "memory_wipe" in want:
        maps = c5_maps(out / "memory", 4, no_monsters=True)
        normal = [int(_run(net, w, ticks=3150)["exit"]) for w in maps]
        wiped = [int(_run(net, w, ticks=3150, memory_wipe_at_key=True)["exit"]) for w in maps]
        report["memory_wipe"] = {"normal": float(np.mean(normal)), "wiped": float(np.mean(wiped)),
                                 "drop": float(np.mean(normal) - np.mean(wiped))}
        print("memory_wipe", report["memory_wipe"], flush=True)

    (out / "probes.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "probes.json")


if __name__ == "__main__":
    main()
