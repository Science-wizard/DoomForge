"""PPO fine-tuning for Baseline A/B+ (map-aware).

A compact recurrent PPO:
  * 8 sequential Doom envs, map-aware observations
  * privileged reward shaping (exit, keys, locks, doors, sectors/cells, combat)
  * stored GRU state at each step; the update recomputes the encoder and runs the
    GRU over 32-step windows with the stored state detached
  * curriculum by training progress

This is intentionally single-process: it is easy to debug and leaves CPU headroom
for the environment.  Scale to worker processes only after profiling.

Usage:
    python -m training.ppo --init training/runs/bc_map/ckpt.pt --updates 200
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from submission.mapmem import MapBuilder
from submission.model import HEAD_SIZES, DoomNet
from training.evaluate import load_net
from training.maps.drills import generate_drill
from training.maps.gen_wad import generate_map
from training.obs import flatten_obs, stack_obs, step_prev
from training.splits import FREEDOOM_TRAIN, GEN_TRAIN

N_ENVS = 8
T_ROLLOUT = 128
GAMMA = 0.997
LAM = 0.95
CLIP = 0.15
LR = 2.5e-4
EPOCHS = 4
MINIBATCH = 4
WINDOW = 32


# --------------------------------------------------------------------------- maps
class Sampler:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.out = Path("training/data/ppo_maps")
        self.freedoom = [Path(f"maps/freedoom/{m}.wad") for m in FREEDOOM_TRAIN]

    def sample(self, progress: float) -> tuple[str, str, int]:
        if progress < 0.20:
            if self.rng.random() < 0.5:
                s = generate_drill(int(self.rng.integers(0, 10**6)), self.out, n_doors=2,
                                   key=bool(self.rng.integers(0, 2)))
                return str(s.path), s.map_name, 2100
            s = generate_map(int(self.rng.integers(*GEN_TRAIN)), 2, self.out,
                             no_doors=True, no_monsters=True)
            return str(s.path), s.map_name, 2100
        if progress < 0.50:
            s = generate_map(int(self.rng.integers(*GEN_TRAIN)), 2, self.out, no_monsters=True)
            return str(s.path), s.map_name, 3150
        if progress < 0.75:
            s = generate_map(int(self.rng.integers(*GEN_TRAIN)), int(self.rng.integers(2, 4)), self.out)
            return str(s.path), s.map_name, 3150
        if self.rng.random() < 0.2 and self.freedoom:
            p = self.freedoom[int(self.rng.integers(0, len(self.freedoom)))]
            return str(p), p.stem, 4200
        s = generate_map(int(self.rng.integers(*GEN_TRAIN)), int(self.rng.integers(2, 5)), self.out)
        return str(s.path), s.map_name, 4200


# --------------------------------------------------------------------------- reward
class Shaper:
    def __init__(self, env, ticks: int):
        self.ticks = ticks
        self.steps = 0
        p = env.privileged_state()
        self.kills = p["kills"]
        self.damage = p["damage_taken"]
        self.keys = int((np.asarray(p["keys"]) > 0.5).sum())
        self.opened: set[int] = set()
        self.visited: set[tuple[int, int]] = set()
        self.doors = dict(env.mapinfo.door_sectors)

    def reset_map(self):
        self.opened.clear()
        self.visited.clear()

    def step(self, env, p, builder) -> tuple[float, dict]:
        r = 0.0
        comp = {}
        if p["exit_reached"]:
            r += 10.0
            comp["exit"] = 10.0
        if p["dead"]:
            frac = self.steps / max(1.0, self.ticks / 4)
            d = -(3.0 * (1.0 - frac) + 1.0)
            r += d
            comp["death"] = d
        dk = p["kills"] - self.kills
        if dk:
            r += 0.6 * dk
            comp["kills"] = 0.6 * dk
        self.kills = p["kills"]
        dd = p["damage_taken"] - self.damage
        if dd:
            r -= 0.01 * dd
            comp["damage"] = -0.01 * dd
        self.damage = p["damage_taken"]
        nk = int((np.asarray(p["keys"]) > 0.5).sum())
        if nk > self.keys:
            r += 3.0 * (nk - self.keys)
            comp["keys"] = 3.0 * (nk - self.keys)
        self.keys = nk
        ceil = np.asarray(p["sector_ceiling"])
        floor = np.asarray(p["sector_floor"])
        for sec, lock in self.doors.items():
            if sec in self.opened or sec >= len(ceil):
                continue
            if ceil[sec] - floor[sec] >= 56:
                self.opened.add(sec)
                r += 3.0 if lock > 0 else 0.5
                comp["doors"] = comp.get("doors", 0) + (3.0 if lock > 0 else 0.5)
        cell = (int(builder.px // 64), int(builder.py // 64))
        if cell not in self.visited:
            self.visited.add(cell)
            r += 0.01
            comp["cells"] = 0.01
        r -= 0.001
        self.steps += 1
        return r, comp


def _slot_obs(env, builder, prev, turn_held, use_last):
    obs, _ = env.reset(seed=0) if False else (None, None)
    raise RuntimeError


# --------------------------------------------------------------------------- vector env
class Slot:
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.env = None
        self.builder = None
        self.shaper = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.ticks = 2100

    def reset(self, progress: float):
        from doom_env import make_env
        if self.env is not None:
            self.env.close()
        wad, name, ticks = self.sampler.sample(progress)
        self.ticks = ticks
        self.env = make_env(wad, name, episode_ticks=ticks, render_buffers=True)
        obs, _ = self.env.reset(seed=int(np.random.default_rng().integers(0, 2**31 - 1)))
        self.builder = MapBuilder()
        self.prev, self.turn_held, self.use_last = None, 0, 0
        self.builder.update(obs, None)
        self.shaper = Shaper(self.env, ticks)
        return obs


def flat_obs(obs, slot: Slot):
    return flatten_obs(obs, slot.prev, slot.turn_held, slot.use_last,
                       map_crop=slot.builder.crop(), pose=slot.builder.pose())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--out", default="training/runs/ppo_a")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-envs", type=int, default=N_ENVS)
    ap.add_argument("--rollout", type=int, default=T_ROLLOUT)
    args = ap.parse_args()

    torch.set_num_threads(max(1, (torch.get_num_threads() // 2)))
    net = DoomNet().to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device)["model"])
        print("initialised from", args.init)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sampler = Sampler(0)
    slots = [Slot(sampler) for _ in range(args.n_envs)]
    obs = [s.reset(0.0) for s in slots]
    h = [None for _ in slots]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train.jsonl").open("a")
    t0 = time.time()
    ep_returns = [0.0 for _ in slots]
    ep_done = [0 for _ in slots]
    best = -1e9

    for update in range(args.updates):
        progress = update / max(1, args.updates)
        buf = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": [], "h": []}
        for t in range(args.rollout):
            batch = stack_obs([flat_obs(o, s) for o, s in zip(obs, slots)], args.device)
            with torch.no_grad():
                h_in = torch.cat([hh if hh is not None else torch.zeros(1, 1, net.hidden, device=args.device)
                                  for hh in h], dim=1)
                logits, value, aux, h_new = net.forward_step(batch, h_in)
                heads = net.flat_action(logits, sample=True)
                lp, _ = net.action_logprob_entropy(logits, heads)
                actions = net.actions_to_binary(heads).cpu().numpy().astype(np.int8)
                logp = lp.cpu().numpy()
                vals = value[:, 0].cpu().numpy()
            buf["obs"].append({k: v.detach().cpu() for k, v in batch.items()})
            buf["act"].append(heads[:, 0].cpu())
            buf["logp"].append(torch.as_tensor(logp))
            buf["val"].append(torch.as_tensor(vals))
            buf["h"].append(torch.stack([hh[0] if hh is not None else torch.zeros(net.hidden) for hh in h]))
            rew, dones = [], []
            for i, s in enumerate(slots):
                a = actions
                o, r, term, trunc, info = s.env.step(a[i])
                p = s.env.privileged_state()
                shaped, _ = s.shaper.step(s.env, p, s.builder)
                ep_returns[i] += shaped
                done = term or trunc
                dones.append(done)
                if done:
                    ep_done[i] += 1
                    o = s.reset(progress)
                    h[i] = None
                else:
                    s.builder.update(o, s.prev)
                    s.prev, s.turn_held, s.use_last = step_prev(a[i], s.prev, s.turn_held, s.use_last)
                obs[i] = o
                rew.append(shaped)
            buf["rew"].append(torch.as_tensor(rew, dtype=torch.float32))
            buf["done"].append(torch.as_tensor(dones, dtype=torch.float32))
            h = [h_new[:, i:i + 1].detach() for i in range(args.n_envs)]
            for i, d in enumerate(dones):
                if d:
                    h[i] = None

        # ---- GAE per slot
        adv = torch.zeros(args.rollout, args.n_envs)
        last = torch.zeros(args.n_envs)
        for t in reversed(range(args.rollout)):
            nextv = buf["val"][t + 1] if t + 1 < args.rollout else torch.zeros(args.n_envs)
            nonterm = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + GAMMA * nextv * nonterm - buf["val"][t]
            last = delta + GAMMA * LAM * nonterm * last
            adv[t] = last
        ret = adv + torch.stack(buf["val"])

        # ---- update on windows
        for epoch in range(EPOCHS):
            starts = [(i, t0) for i in range(args.n_envs) for t0 in range(0, args.rollout - WINDOW + 1, WINDOW)]
            np.random.shuffle(starts)
            for b in range(0, len(starts), MINIBATCH):
                chunk = starts[b:b + MINIBATCH]
                losses = []
                for i, t0 in chunk:
                    obs_win = {k: torch.stack([buf["obs"][t][k][i] for t in range(t0, t0 + WINDOW)])
                               for k in buf["obs"][0]}
                    # add batch dimension and run through the encoder/GRU
                    obs_win = {k: v.unsqueeze(0) for k, v in obs_win.items()}
                    hh = buf["h"][t0][i].view(1, 1, -1)
                    emb = net.encode({k: v for k, v in obs_win.items()})
                    logits, value, aux, _ = net.forward_seq(emb, hh)
                    act = torch.stack([buf["act"][t][i] for t in range(t0, t0 + WINDOW)]).unsqueeze(0)
                    oldlp = torch.stack([buf["logp"][t][i] for t in range(t0, t0 + WINDOW)]).unsqueeze(0).detach()
                    adv_w = adv[t0:t0 + WINDOW, i].unsqueeze(0)
                    ret_w = ret[t0:t0 + WINDOW, i].unsqueeze(0)
                    lp, ent = net.action_logprob_entropy(logits, act)
                    ratio = (lp - oldlp).exp()
                    s1 = ratio * adv_w
                    s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_w
                    pi_loss = -torch.min(s1, s2).mean()
                    v_loss = F.mse_loss(value[..., 0], ret_w)
                    loss = pi_loss + 0.5 * v_loss - 0.01 * ent.mean()
                    losses.append(loss)
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if (update + 1) % 5 == 0:
            row = {"update": update, "steps": (update + 1) * args.rollout * args.n_envs,
                   "episodes": int(sum(ep_done)), "mean_return": float(np.mean(ep_returns)),
                   "seconds": time.time() - t0}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            torch.save({"model": net.state_dict(), "config": {"hidden": net.hidden},
                        "update": update}, out / "ckpt.pt")
        # decay entropy/lr
        for g in opt.param_groups:
            g["lr"] = LR * (1.0 - 0.8 * progress)
    torch.save({"model": net.state_dict(), "config": {"hidden": net.hidden},
                "update": args.updates}, out / "ckpt.pt")
    print("saved", out / "ckpt.pt")


if __name__ == "__main__":
    main()
