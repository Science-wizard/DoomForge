"""Learned mixture-of-experts gate.

Two frozen specialists:
  arena : tier-1 arena fine-tune (5/8 held-out arena exits)
  door  : drill/room-door specialist (6/8 door drills)

A tiny gate maps observation-only door/ray/map features to a per-step routing weight.
The final logits are `g · door + (1−g) · arena`.  The gate is trained by distillation
on the mixed demonstration pool; the experts are never updated, so neither skill is
forgotten and no rule tree decides actions.

Usage:
    python -m training.moe --arena training/runs/bc_arena/ckpt.pt \
        --door training/runs/bc_door_expert/ckpt.pt --out training/runs/moe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from submission.mapmem import MapBuilder
from submission.model import HEAD_SIZES, DoomNet, Gate, gate_features
from training.bc import collate, load_episodes, window_episodes
from training.evaluate import load_net
from training.obs import flatten_obs, stack_obs, step_prev

OBS_KEYS = ("rays_dist", "rays_hit", "ent_feat", "ent_type", "ent_mask",
            "vitals", "rgb", "prev_action", "map", "pose")


def _ce_masked(logits, heads, mask):
    denom = mask.sum().clamp(min=1.0)
    total = 0.0
    for j, part in enumerate(torch.split(logits, HEAD_SIZES, dim=-1)):
        c = F.cross_entropy(part.reshape(-1, part.shape[-1]),
                            heads[..., j].reshape(-1), reduction="none")
        total = total + (c * mask.reshape(-1)).sum() / denom
    return total


def train_gate(arena: DoomNet, door: DoomNet, data: str, out: Path, epochs: int,
               batch_size: int, lr: float, device: str) -> Gate:
    raw = load_episodes(data)
    eps = window_episodes(raw, 128, 64)
    gate = Gate().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = rng.permutation(len(eps))
        tot = 0.0
        n = 0
        for b in range(0, len(eps), batch_size):
            batch = collate([eps[i] for i in order[b:b + batch_size]], device)
            obs = {k: batch[k] for k in OBS_KEYS}
            with torch.no_grad():
                la = arena.forward_seq(arena.encode(obs))[0]
                ld = door.forward_seq(door.encode(obs))[0]
            g = gate(gate_features(obs))
            logits = g * ld + (1.0 - g) * la
            door_any = ((batch["rays_hit"] == 2) | (batch["rays_hit"] == 3)).any(-1).float()
            w = batch["mask"] * (1.0 + 5.0 * door_any)
            loss = _ce_masked(logits, batch["heads"], w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
            if n % 10 == 0:
                print(f"  gate epoch {epoch} {n}/{len(eps)//batch_size} loss={tot/n:.3f}", flush=True)
        print(f"gate epoch {epoch} done loss={tot/max(1,n):.3f}", flush=True)
        torch.save({"gate": gate.state_dict()}, out / f"gate_e{epoch}.pt")
        torch.save({"gate": gate.state_dict()}, out / "gate.pt")
    return gate


# --------------------------------------------------------------------------- eval
class MoERunner:
    def __init__(self, arena: DoomNet, door: DoomNet, gate: Gate, device: str = "cpu"):
        self.arena, self.door, self.gate, self.device = arena, door, gate, device
        self.h_a = self.h_d = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map = MapBuilder()

    def reset(self):
        self.h_a = self.h_d = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map.reset()

    @torch.no_grad()
    def act(self, obs):
        self.map.update(obs, self.prev)
        flat = flatten_obs(obs, self.prev, self.turn_held, self.use_last,
                           map_crop=self.map.crop(), pose=self.map.pose())
        batch = stack_obs([flat], self.device)
        la, _, _, self.h_a = self.arena.forward_step(batch, self.h_a)
        ld, _, _, self.h_d = self.door.forward_step(batch, self.h_d)
        g = self.gate(gate_features(batch))
        logits = g * ld + (1.0 - g) * la
        heads = self.arena.flat_action(logits, sample=False)
        a = self.arena.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)
        self.prev, self.turn_held, self.use_last = step_prev(a, self.prev, self.turn_held, self.use_last)
        return a


def run_moe_episode(arena, door, gate, wad, map_name="MAP01", device="cpu",
                    ticks=4200, seed=0) -> dict:
    from doom_env import make_env
    from training.coverage import SectorTracker

    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    runner = MoERunner(arena, door, gate, device)
    tracker = SectorTracker(env)
    tracker.update(p0["x"], p0["y"])
    steps = 0
    lat = []
    while True:
        t0 = __import__("time").perf_counter()
        a = runner.act(obs)
        lat.append((__import__("time").perf_counter() - t0) * 1000)
        obs, rew, term, trunc, info = env.step(a)
        steps += 1
        p = env.privileged_state()
        tracker.update(p["x"], p["y"])
        if term or trunc:
            break
    p = env.privileged_state()
    opened = 0
    locks = 0
    for sec, lock in env.mapinfo.door_sectors.items():
        if sec < len(p["sector_ceiling"]) and (p["sector_ceiling"][sec] - p["sector_floor"][sec]) >= 56:
            opened += 1
            if lock > 0:
                locks += 1
    keys = int((np.asarray(p["keys"]) > 0.5).sum())
    comp = {
        "exit": 10.0 * float(p["exit_reached"]),
        "keys": 3.0 * keys,
        "locks": 3.0 * locks,
        "doors": 0.5 * max(0, opened - locks),
        "sectors": min(0.3 * len(tracker.visited), 10.0),
        "kills": 0.6 * max(0, p["kills"] - p0["kills"]),
        "damage": -0.01 * float(p["damage_taken"]),
        "death": (-(3.0 * (1.0 - steps / max(1, ticks // 4)) + 1.0)) if p["dead"] else 0.0,
    }
    out = {"wad": str(wad), "map": map_name, "seed": seed,
           "exit": bool(p["exit_reached"]), "dead": bool(p["dead"]), "steps": steps,
           "kills": int(p["kills"] - p0["kills"]), "keys": keys, "doors": opened,
           "locks": locks, "sectors": len(tracker.visited),
           "coverage": len(tracker.visited) / max(1, env.mapinfo.num_sectors),
           "proxy": float(sum(comp.values())), "components": comp,
           "p50_ms": float(np.median(lat)) if lat else 0.0,
           "p99_ms": float(np.percentile(lat, 99)) if lat else 0.0}
    env.close()
    return out


def load_moe(arena_ckpt: str, door_ckpt: str, gate_ckpt: str, device="cpu"):
    arena = load_net(arena_ckpt, device)
    door = load_net(door_ckpt, device)
    ck = torch.load(gate_ckpt, map_location=device)
    gate = Gate()
    gate.load_state_dict(ck["gate"])
    gate.to(device).eval()
    return arena, door, gate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arena", default="training/runs/bc_arena/ckpt.pt")
    ap.add_argument("--door", default="training/runs/bc_door_expert/ckpt.pt")
    ap.add_argument("--data", default="training/data/demos_all,training/data/demos_drills,training/data/demos_roomdoor")
    ap.add_argument("--out", default="training/runs/moe")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arena = load_net(args.arena, args.device)
    door = load_net(args.door, args.device)
    arena.eval()
    door.eval()
    for p in arena.parameters():
        p.requires_grad_(False)
    for p in door.parameters():
        p.requires_grad_(False)
    gate = train_gate(arena, door, args.data, out, args.epochs, args.batch, args.lr, args.device)
    torch.save({"gate": gate.state_dict()}, out / "gate.pt")
    print("saved", out / "gate.pt")


if __name__ == "__main__":
    main()
