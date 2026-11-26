"""Atlas-E PPO: parallel recurrent PPO with an asymmetric critic.

Replaces ``training/ppo.py`` (kept for reference).  What changed and why:

* **Parallel environments.**  ``--workers`` processes each step ``--envs-per-worker``
  ViZDoom instances; the learner batches inference for all of them.  The old
  trainer stepped 8 envs sequentially in one process (~50 steps/s).
* **Correct returns.**  GAE bootstraps from the value of the observation after
  the rollout (the old code assumed every rollout ended in a terminal state),
  advantages are normalised per update, and episode returns are logged per
  episode (the old ``mean_return`` was a cumulative sum that never reset).
* **Asymmetric critic.**  The value baseline sees 16 privileged progress
  features (time left, geodesic distance to the current objective, keys,
  coverage, ...).  It is a training-time baseline only: acting never reads it,
  and the submitted agent does not load it.  Time-left makes the tick cap part
  of the state, so truncation is a terminal transition without bias.
* **Episode-aware recurrent windows.**  The GRU state is reset inside a window
  where an episode boundary falls, instead of leaking across episodes.
* **Geodesic progress shaping (training only).**  The teacher's nav grid gives
  a distance field to the current objective (exit if reachable with the keys
  held, otherwise the nearest needed key); moving closer is rewarded.  Plus the
  proxy components: exit, keys, locks, doors, new sectors, new cells, kills,
  damage and early death.
* **Learning-progress curriculum** over the suite train split plus freshly
  generated maps, weighted towards categories that are not yet solved.
* **KL anchor** to the behaviour-cloned initialisation (stored per step during
  the rollout, so it is exact), decayed over training.

Usage:
    python -m training.ppo2 --init training/runs/bc_ego1/ckpt.pt --out training/runs/ppo_ego1
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

PRIV = 16
IDLE_PENALTY = 0.01
STAG_WINDOW = 16
STAG_PENALTY = 0.02
KEY_REWARD = 5.0
# Survival: at 3/1 and 0.01/HP the policy traded its life for kills (u100 died
# in 8/10 val arenas), so dying and damage cost more.
DEATH_EARLY = 6.0
DEATH_BASE = 4.0
DAMAGE_PER_HP = 0.02
USE_FAR_PENALTY = 0.02
SEEN_REWARD = 0.004
SUITE_TRAIN = "training/data/suite/train/manifest.jsonl"
ENEMY_TYPES = (1, 2, 3)
_LOCK_TO_KEY = {1: 2, 2: 0, 3: 1}


# =========================================================================== worker
def _distance_field(passable: np.ndarray, goals: list[tuple[int, int]]) -> np.ndarray:
    """8-connected wavefront distance (in cells) from goal cells over ``passable``."""
    H, W = passable.shape
    dist = np.full((H, W), np.inf, np.float32)
    front = np.zeros((H, W), bool)
    for gx, gy in goals:
        if 0 <= gx < W and 0 <= gy < H:
            front[gy, gx] = True
    front &= passable
    if not front.any():
        return dist
    d = 0
    seen = front.copy()
    while front.any():
        dist[front] = d
        nb = np.zeros_like(front)
        nb[1:, :] |= front[:-1, :]
        nb[:-1, :] |= front[1:, :]
        nb[:, 1:] |= front[:, :-1]
        nb[:, :-1] |= front[:, 1:]
        nb[1:, 1:] |= front[:-1, :-1]
        nb[:-1, :-1] |= front[1:, 1:]
        nb[1:, :-1] |= front[:-1, 1:]
        nb[:-1, 1:] |= front[1:, :-1]
        front = nb & passable & ~seen
        seen |= front
        d += 1
        if d > 4000:
            break
    return dist


class Objective:
    """Privileged geodesic distance to the current objective (training only)."""

    def __init__(self, wad: str, map_name: str, start: tuple[float, float]):
        from training.teacher.teacher import make_teacher
        self.ok = False
        try:
            t = make_teacher(wad, map_name)
            t.reset(start)
            self.grid = t.grid
            self.keys_xy = list(t.md.key_positions())
            g = self.grid
            self.exit_cells = []
            for gx, gy, _kind, _r in g.exit_goals:
                ix, iy = g.world_to_cell(gx, gy)
                if 0 <= ix < g.w and 0 <= iy < g.h and not g.free[iy, ix]:
                    ix, iy = g._nearest_free(g.free, ix, iy)
                self.exit_cells.append((ix, iy))
            self.cache: dict = {}
            self.ok = g.free is not None and len(self.exit_cells) > 0
        except Exception:
            self.ok = False

    def _passable(self, keys: tuple) -> np.ndarray:
        g = self.grid
        p = g.free.copy()
        lock = g.door_lock
        for lk, idx in _LOCK_TO_KEY.items():
            if not (keys[idx] or keys[idx + 3]):
                p &= lock != lk
        return p

    def _field(self, kind: str, keys: tuple, target=None) -> np.ndarray:
        key = (kind, keys, target)
        if key not in self.cache:
            pas = self._passable(keys)
            if kind == "exit":
                goals = self.exit_cells
            else:
                g = self.grid
                ix, iy = g.world_to_cell(*target)
                if 0 <= ix < g.w and 0 <= iy < g.h and not g.free[iy, ix]:
                    ix, iy = g._nearest_free(g.free, ix, iy)
                goals = [(ix, iy)]
            self.cache[key] = _distance_field(pas, goals)
        return self.cache[key]

    def query(self, x: float, y: float, keys_arr) -> tuple[str, float, tuple[float, float] | None]:
        """(objective kind, distance in map units or inf, objective xy)."""
        if not self.ok:
            return "none", math.inf, None
        g = self.grid
        keys = tuple(int(k > 0.5) for k in np.asarray(keys_arr)[:6])
        ix, iy = g.world_to_cell(x, y)
        if not (0 <= ix < g.w and 0 <= iy < g.h):
            return "none", math.inf, None
        f = self._field("exit", keys)
        d = float(f[iy, ix])
        if math.isfinite(d):
            gx, gy = self.exit_cells[0]
            return "exit", d * g.cell, g.cell_to_world(gx, gy)
        best = ("none", math.inf, None)
        for kx, ky, lock in self.keys_xy:
            idx = _LOCK_TO_KEY.get(lock)
            if idx is None or keys[idx] or keys[idx + 3]:
                continue
            fk = self._field("key", keys, (kx, ky))
            dk = float(fk[iy, ix])
            if math.isfinite(dk) and dk * g.cell < best[1]:
                best = ("key", dk * g.cell, (kx, ky))
        return best


class Slot:
    """One environment + the agent-side state the policy needs + privileged shaping."""

    def __init__(self, rng: random.Random, rows: dict[str, list[dict]], gen_prob: float,
                 out_dir: str):
        self.rng = rng
        self.rows = rows
        self.gen_prob = gen_prob
        self.out_dir = out_dir
        self.env = None
        self.obj_cache: dict = {}

    # -------------------------------------------------------------- episode setup
    def _pick(self, weights: dict[str, float]):
        from training.maps.gen_wad import generate_map
        from training.splits import GEN_TRAIN
        if self.rng.random() < self.gen_prob:
            tier = self.rng.choice((1, 2, 2, 3, 4))
            seed = self.rng.randrange(GEN_TRAIN[0], GEN_TRAIN[1])
            nm = self.rng.random() < 0.4
            spec = generate_map(seed, tier, self.out_dir, no_monsters=nm)
            return str(spec.path), spec.map_name, 2100 if tier == 1 else 4200, f"gen_t{tier}"
        cats = [c for c in self.rows if weights.get(c, 1.0) > 0]
        w = [weights.get(c, 1.0) for c in cats]
        cat = self.rng.choices(cats, weights=w, k=1)[0]
        row = self.rng.choice(self.rows[cat])
        return row["wad"], row["map_name"], int(row["ticks"]), cat

    def reset(self, weights: dict[str, float]):
        from doom_env import make_env
        from submission.mapmem import EgoMap
        from training.coverage import SectorTracker
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        for _ in range(5):
            wad, name, ticks, cat = self._pick(weights)
            try:
                self.env = make_env(wad, name, episode_ticks=ticks, render_buffers=True)
                obs, _ = self.env.reset(seed=self.rng.randrange(0, 2**31 - 1))
                break
            except Exception:
                self.env = None
        self.wad, self.cat, self.ticks = wad, cat, ticks
        self.T = ticks // 4
        p = self.env.privileged_state()
        okey = (wad, name, round(p["x"]), round(p["y"]))
        if okey not in self.obj_cache:
            if len(self.obj_cache) > 48:
                self.obj_cache.pop(next(iter(self.obj_cache)))
            self.obj_cache[okey] = Objective(wad, name, (p["x"], p["y"]))
        self.obj = self.obj_cache[okey]
        self.map = EgoMap()
        self.map.update(obs, None)
        self.tracker = SectorTracker(self.env)
        self.tracker.update(p["x"], p["y"])
        self.n_sectors = max(1, self.env.mapinfo.num_sectors)
        self.doors = dict(self.env.mapinfo.door_sectors)
        self.opened: set[int] = set()
        self.cells: set[tuple[int, int]] = {(int(p["x"] // 64), int(p["y"] // 64))}
        self.prev, self.turn_held, self.use_last = None, 0, 0
        self.steps = 0
        self.kills, self.damage = p["kills"], p["damage_taken"]
        self.keys = int((np.asarray(p["keys"]) > 0.5).sum())
        self.obj_kind, self.obj_d, self.obj_xy = self.obj.query(p["x"], p["y"], p["keys"])
        self.last_progress = 0
        self.idle = 0
        self.pos_hist = deque([(p["x"], p["y"])], maxlen=STAG_WINDOW)
        self.seen = int(self.map.ego[0].sum())
        self.ep = {"return": 0.0, "exit": 0, "keys": 0, "locks": 0, "doors": 0, "kills": 0,
                   "dead": 0, "steps": 0, "cat": cat}
        self.obs = obs
        self.p = p
        return obs

    # -------------------------------------------------------------- features
    def priv_features(self) -> np.ndarray:
        p = self.p
        f = np.zeros(PRIV, np.float32)
        f[0] = self.steps / max(1, self.T)
        d = self.obj_d if math.isfinite(self.obj_d) else 4000.0
        f[1] = min(d, 6000.0) / 2000.0
        f[2] = self.obj_kind == "exit"
        f[3] = self.obj_kind == "key"
        f[4] = self.obj_kind == "none"
        f[5] = self.keys / 2.0
        f[6] = float(p["health"]) / 100.0
        v = np.asarray(self.obs["vitals"], np.float32)
        f[7] = min(float(v[3]), 100.0) / 50.0
        f[8] = len(self.tracker.visited) / self.n_sectors
        f[9] = min(self.ep["kills"], 20) / 10.0
        f[10] = min(len(self.opened), 10) / 5.0
        if self.obj_xy is not None:
            rel = math.atan2(self.obj_xy[1] - p["y"], self.obj_xy[0] - p["x"]) - math.radians(p["angle"])
            f[11], f[12] = math.sin(rel), math.cos(rel)
        f[13] = min(len(self.cells), 400) / 200.0
        f[14] = min(float(p["damage_taken"]), 200.0) / 100.0
        f[15] = min(self.steps - self.last_progress, 400) / 200.0
        return f

    def payload(self) -> dict:
        from training.egomaps import pack
        from training.obs import flatten_obs
        flat = flatten_obs(self.obs, self.prev, self.turn_held, self.use_last,
                           map_crop=np.zeros((1, 1, 1), np.uint8), pose=self.map.pose())
        flat["map"] = pack(self.map.ego_crop())
        flat["priv"] = self.priv_features()
        return flat

    # -------------------------------------------------------------- step
    def step(self, action: np.ndarray, weights: dict[str, float]):
        from training.obs import step_prev
        obs, _r, term, trunc, _info = self.env.step(action)
        p = self.env.privileged_state()
        self.steps += 1
        frac = self.steps / max(1, self.T)
        r = -0.001
        # Stagnation: moving < 48 units over the last 16 decisions costs
        # 0.02/step whatever buttons are pressed.  The forgetting mode found in
        # Phase 5 was parking at a locked door and pressing USE every ~25 steps
        # (a pure no-op counter is dodged by that).
        self.pos_hist.append((p["x"], p["y"]))
        if len(self.pos_hist) == STAG_WINDOW:
            x0, y0 = self.pos_hist[0]
            if math.hypot(p["x"] - x0, p["y"] - y0) < 48.0:
                r -= STAG_PENALTY
        # Wasted USE: on a locked door without any key, or with no door within
        # reach (~72 units) straight ahead.  The C6 failure was holding both
        # keys and pressing USE 130 units short of the second lock forever.
        a_np = np.asarray(action)
        if int(a_np[7]):
            hit = np.asarray(self.obs["rays"]["hit_type"])
            dist = np.asarray(self.obs["rays"]["distance"])
            front = [0, 1, 2, 62, 63]
            door_close = (hit[front] == 2) | (hit[front] == 3)
            door_close &= dist[front] < 0.07
            if not door_close.any():
                r -= USE_FAR_PENALTY
            elif float(np.asarray(p["keys"]).sum()) < 0.5 and np.any((hit[front] == 3) & (dist[front] < 0.12)):
                r -= 0.02
        if p["exit_reached"]:
            r += 10.0 + 2.0 * max(0.0, 1.0 - frac)
            self.ep["exit"] = 1
        if p["dead"]:
            r -= DEATH_EARLY * max(0.0, 1.0 - frac) + DEATH_BASE
            self.ep["dead"] = 1
        dk = p["kills"] - self.kills
        if dk > 0:
            r += 0.6 * dk
            self.ep["kills"] += dk
        self.kills = p["kills"]
        dd = p["damage_taken"] - self.damage
        if dd > 0:
            r -= DAMAGE_PER_HP * dd
        self.damage = p["damage_taken"]
        nk = int((np.asarray(p["keys"]) > 0.5).sum())
        if nk > self.keys:
            r += KEY_REWARD * (nk - self.keys)
            self.ep["keys"] += nk - self.keys
            self.last_progress = self.steps
        self.keys = nk
        ceil, floor = np.asarray(p["sector_ceiling"]), np.asarray(p["sector_floor"])
        for sec, lock in self.doors.items():
            if sec in self.opened or sec >= len(ceil):
                continue
            if ceil[sec] - floor[sec] >= 56:
                self.opened.add(sec)
                r += 3.0 if lock > 0 else 0.5
                self.ep["locks" if lock > 0 else "doors"] += 1
                self.last_progress = self.steps
        n_before = len(self.tracker.visited)
        self.tracker.update(p["x"], p["y"])
        if len(self.tracker.visited) > n_before:
            r += 0.3
            self.last_progress = self.steps
        cell = (int(p["x"] // 64), int(p["y"] // 64))
        if cell not in self.cells:
            self.cells.add(cell)
            r += 0.01
        # geodesic progress towards the current objective
        kind, d, xy = self.obj.query(p["x"], p["y"], p["keys"])
        if kind == self.obj_kind and kind != "none" and math.isfinite(d) and math.isfinite(self.obj_d):
            r += float(np.clip(0.003 * (self.obj_d - d), -0.1, 0.1))
        self.obj_kind, self.obj_d, self.obj_xy = kind, d, xy
        self.ep["return"] += r
        self.ep["steps"] = self.steps
        done = bool(term or trunc)
        info = None
        if done:
            info = dict(self.ep)
            info["coverage"] = len(self.tracker.visited) / self.n_sectors
            obs = self.reset(weights)
        else:
            self.map.update(obs, action)
            # Dense exploration signal from the agent's own map: newly seen
            # free floor (64-unit cells).  The policy can observe this, so it
            # is learnable everywhere, unlike the exit location.
            seen = int(self.map.ego[0].sum())
            if seen > self.seen:
                bonus = min(0.1, SEEN_REWARD * (seen - self.seen))
                r += bonus
                self.ep["return"] += bonus
            self.seen = seen
            self.prev, self.turn_held, self.use_last = step_prev(action, self.prev,
                                                                 self.turn_held, self.use_last)
            self.obs, self.p = obs, p
        return r, done, info


def worker_main(conn, wid: int, n_envs: int, seed: int, gen_prob: float, categories: list[str]):
    import torch
    torch.set_num_threads(1)
    rows: dict[str, list[dict]] = {}
    for line in Path(SUITE_TRAIN).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not categories or r["category"] in categories:
                rows.setdefault(r["category"], []).append(r)
    out_dir = f"training/data/ppo_maps/w{wid}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed * 7919 + wid)
    weights: dict[str, float] = {}
    slots = [Slot(random.Random(rng.random()), rows, gen_prob, out_dir) for _ in range(n_envs)]
    for s in slots:
        s.reset(weights)
    conn.send([s.payload() for s in slots])
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            actions, new_w = data
            if new_w is not None:
                weights = new_w
            out = []
            for s, a in zip(slots, actions):
                r, done, info = s.step(np.asarray(a, np.int8), weights)
                out.append((s.payload(), r, done, info))
            conn.send(out)
        elif cmd == "close":
            for s in slots:
                try:
                    s.env.close()
                except Exception:
                    pass
            conn.close()
            return


# =========================================================================== learner
OBS_KEYS = ("rays_dist", "rays_hit", "ent_feat", "ent_type", "ent_mask", "vitals", "rgb",
            "prev_action", "map", "pose")
STORE = {"rays_dist": ((64,), np.float16), "rays_hit": ((64,), np.uint8),
         "ent_feat": ((32, 5), np.float16), "ent_type": ((32,), np.uint8),
         "ent_mask": ((32,), np.uint8), "vitals": ((17,), np.float32),
         "rgb": ((3, 30, 40), np.uint8), "prev_action": ((12,), np.float32),
         "map": ((32, 32), np.uint16), "pose": ((6,), np.float32), "priv": ((PRIV,), np.float32)}


def to_device(arrs: dict, device: str):
    import torch
    from training.egomaps import unpack_torch
    out = {}
    for k in OBS_KEYS:
        a = arrs[k]
        if k == "map":
            out[k] = unpack_torch(torch.from_numpy(a.astype(np.int32)).to(device))
        elif k == "rgb":
            out[k] = torch.from_numpy(a).to(device).float() / 255.0
        elif k in ("rays_hit", "ent_type"):
            out[k] = torch.from_numpy(a.astype(np.int64)).to(device)
        else:
            out[k] = torch.from_numpy(a.astype(np.float32)).to(device)
    out["priv"] = torch.from_numpy(arrs["priv"].astype(np.float32)).to(device)
    return out


def stack_payloads(payloads: list[dict]) -> dict:
    return {k: np.stack([np.asarray(p[k]) for p in payloads]).astype(dt) for k, (_, dt) in STORE.items()}


def run_gru_masked(net, emb, h0, starts):
    """GRU over (B, L, D) with the state zeroed where ``starts`` (B, L) is 1."""
    import torch
    outs = []
    h = h0
    for t in range(emb.shape[1]):
        h = h * (1.0 - starts[:, t]).view(1, -1, 1)
        y, h = net.gru(emb[:, t:t + 1], h)
        outs.append(y)
    y = torch.cat(outs, dim=1)
    z = net.trunk(y)
    return net.actor(z), z


def bc_loss(net, windows: list, batch: int, device: str):
    """Teacher cross-entropy on random demo windows (DAPG-style regulariser)."""
    import torch
    import torch.nn.functional as F
    from submission.model import HEAD_SIZES
    from training.egomaps import unpack_torch
    sel = [windows[i] for i in np.random.randint(0, len(windows), size=batch)]
    T = max(len(w["action"]) for w in sel)
    B = len(sel)
    def pad(key, dtype, shape):
        out = np.zeros((B, T) + shape, dtype)
        for i, w in enumerate(sel):
            out[i, :len(w["action"])] = w[key]
        return out
    mask = np.zeros((B, T), np.float32)
    for i, w in enumerate(sel):
        mask[i, :len(w["action"])] = 1.0
    obs = {
        "rays_dist": torch.from_numpy(pad("dist", np.float32, (64,))),
        "rays_hit": torch.from_numpy(pad("hit", np.int64, (64,))),
        "ent_feat": torch.from_numpy(pad("efeat", np.float32, (32, 5))),
        "ent_type": torch.from_numpy(pad("etype", np.int64, (32,))),
        "ent_mask": torch.from_numpy(pad("emask", np.float32, (32,))),
        "vitals": torch.from_numpy(pad("vitals", np.float32, (17,))),
        "rgb": torch.from_numpy(pad("rgb", np.float32, (3, 30, 40))) / 255.0,
        "prev_action": torch.from_numpy(pad("prev", np.float32, (12,))),
        "pose": torch.from_numpy(pad("pose", np.float32, (6,))),
    }
    obs = {k: v.to(device) for k, v in obs.items()}
    obs["map"] = unpack_torch(torch.from_numpy(pad("map", np.int32, (32, 32))).to(device))
    heads = torch.from_numpy(pad("heads", np.int64, (6,))).to(device)
    mk = torch.from_numpy(mask).to(device)
    emb = net.encode(obs)
    logits, _, _, _ = net.forward_seq(emb)
    loss = 0.0
    for j, part in enumerate(torch.split(logits, HEAD_SIZES, dim=-1)):
        ce = F.cross_entropy(part.reshape(-1, part.shape[-1]), heads[..., j].reshape(-1), reduction="none")
        loss = loss + (ce * mk.reshape(-1)).sum() / mk.sum().clamp(min=1.0)
    return loss


def main() -> None:
    import torch
    import torch.nn.functional as F
    from submission.model import HEAD_SIZES, DoomNet, build_from_state

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="training/runs/ppo_ego")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--envs-per-worker", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--minibatch-windows", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--kl-beta", type=float, default=0.3)
    ap.add_argument("--kl-beta-final", type=float, default=0.02)
    ap.add_argument("--critic-warmup", type=int, default=15)
    ap.add_argument("--gen-prob", type=float, default=0.15)
    ap.add_argument("--categories", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--anchor", default="", help="KL anchor checkpoint (default: --init)")
    ap.add_argument("--mastered-weight", type=float, default=0.35)
    ap.add_argument("--bc-data", default="", help="demo dirs (with ego/ maps) for an imitation loss")
    ap.add_argument("--bc-weight", type=float, default=0.0)
    ap.add_argument("--bc-weight-final", type=float, default=0.0)
    ap.add_argument("--bc-batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--ema", type=float, default=0.95, help="per-update weight EMA decay (0 = off)")
    args = ap.parse_args()

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    net = build_from_state(ck["model"], ck.get("config", {})).to(device)
    assert net.arch == "ego", "ppo2 trains the Atlas-E architecture"
    ak = torch.load(args.anchor, map_location="cpu", weights_only=False) if args.anchor else ck
    anchor = build_from_state(ak["model"], ak.get("config", {})).to(device).eval()
    for p_ in anchor.parameters():
        p_.requires_grad_(False)
    start_update = 0
    if args.resume and (out / "ckpt.pt").exists():
        rk = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(rk["model"])
        start_update = int(rk.get("update", 0)) + 1
        print("resumed at update", start_update)
    # The goal head imitated the teacher; PPO leaves it alone (its output is still
    # an input feature, so it is kept fixed rather than drifting).
    for p_ in net.goal_head.parameters():
        p_.requires_grad_(False)
    policy_params = [p_ for n_, p_ in net.named_parameters()
                     if p_.requires_grad and not n_.startswith(("critic", "critic_priv"))]
    critic_params = list(net.critic.parameters()) + list(net.critic_priv.parameters())
    opt = torch.optim.Adam([{"params": policy_params, "lr": args.lr},
                            {"params": critic_params, "lr": 3e-4}], eps=1e-5)

    bc_windows: list = []
    if args.bc_data and args.bc_weight > 0:
        from training.bc import load_episodes, window_episodes
        eps = load_episodes(args.bc_data, arch="ego")
        keep = ("dist", "hit", "efeat", "etype", "emask", "vitals", "rgb", "prev", "map", "pose",
                "heads", "action")
        eps = [{**{k: e[k] for k in keep}, "meta": e["meta"]} for e in eps]
        bc_windows = window_episodes(eps, 64, 64)
        print(f"imitation loss: {len(bc_windows)} demo windows", flush=True)
    cats = [c for c in args.categories.split(",") if c]
    W, K = args.workers, args.envs_per_worker
    E = W * K
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for w in range(W):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=worker_main, args=(b, w, K, args.seed + start_update, args.gen_prob, cats),
                         daemon=True)
        pr.start()
        conns.append(a)
        procs.append(pr)
    payloads = []
    for c in conns:
        payloads.extend(c.recv())
    cur = stack_payloads(payloads)
    h = torch.zeros(1, E, net.hidden, device=device)
    starts_next = np.ones(E, np.float32)

    T, L = args.rollout, args.window
    buf = {k: np.zeros((T, E) + shp, dt) for k, (shp, dt) in STORE.items()}
    b_heads = np.zeros((T, E, 6), np.int64)
    b_logp = np.zeros((T, E), np.float32)
    b_val = np.zeros((T, E), np.float32)
    b_rew = np.zeros((T, E), np.float32)
    b_done = np.zeros((T, E), np.float32)
    b_start = np.zeros((T, E), np.float32)
    b_h = np.zeros((T, E, net.hidden), np.float32)
    b_ha = np.zeros((T, E, anchor.hidden), np.float32)
    h_anchor = torch.zeros(1, E, anchor.hidden, device=device)

    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if args.ema > 0 else None
    if ema is not None and args.resume and (out / "ema.pt").exists():
        ema = torch.load(out / "ema.pt", map_location=device, weights_only=False)
    log = (out / "train.jsonl").open("a")
    recent: deque = deque(maxlen=400)
    cat_stats: dict[str, deque] = {}
    weights: dict[str, float] | None = None
    t_start = time.time()
    total_steps = 0

    for update in range(start_update, args.updates):
        t0 = time.time()
        progress = update / max(1, args.updates)
        kl_beta = args.kl_beta + (args.kl_beta_final - args.kl_beta) * min(1.0, progress / 0.5)
        bc_w = args.bc_weight + (args.bc_weight_final - args.bc_weight) * progress
        net.eval()
        t_inf = t_wait = 0.0
        # Double-buffered rollout: the workers of one half step their envs while
        # the learner runs inference for the other half.
        groups = [list(range(0, W // 2)), list(range(W // 2, W))] if W >= 2 else [list(range(W))]
        gidx = [np.arange(g[0] * K, (g[-1] + 1) * K) for g in groups]

        def infer(g: int, t: int) -> None:
            idx = gidx[g]
            sl = slice(int(idx[0]), int(idx[-1]) + 1)
            for k in STORE:
                buf[k][t, sl] = cur[k][sl]
            b_start[t, sl] = starts_next[sl]
            b_h[t, sl] = hs[0, sl].cpu().numpy()
            b_ha[t, sl] = has[0, sl].cpu().numpy()
            with torch.no_grad():
                obs_t = to_device({k: v[sl] for k, v in cur.items()}, device)
                st = torch.from_numpy(starts_next[sl]).to(device).view(1, -1, 1)
                hg = hs[:, sl] * (1.0 - st)
                has[:, sl] = has[:, sl] * (1.0 - st)
                emb = net.encode({k: obs_t[k] for k in OBS_KEYS})
                logits, _v, _aux, z, hg = net.forward_seq_z(emb.unsqueeze(1), hg)
                hs[:, sl] = hg
                logits, z = logits[:, 0], z[:, 0]
                v = net.value_priv(z, obs_t["priv"])[:, 0]
                heads = net.flat_action(logits, sample=True)
                lp, _ = net.action_logprob_entropy(logits, heads)
                acts = net.actions_to_binary(heads).cpu().numpy().astype(np.int8)
            b_heads[t, sl] = heads.cpu().numpy()
            b_logp[t, sl] = lp.cpu().numpy()
            b_val[t, sl] = v.cpu().numpy()
            for j, wi in enumerate(groups[g]):
                conns[wi].send(("step", (acts[j * K:(j + 1) * K], weights)))

        def receive(g: int, t: int) -> None:
            for wi in groups[g]:
                res = conns[wi].recv()
                for j, (pl, r, d, info) in enumerate(res):
                    e = wi * K + j
                    for k in STORE:
                        cur[k][e] = pl[k]
                    b_rew[t, e] = r
                    b_done[t, e] = d
                    starts_next[e] = float(d)
                    if info is not None:
                        recent.append(info)
                        cat_stats.setdefault(info["cat"], deque(maxlen=60)).append(info["exit"])

        hs = h
        has = h_anchor
        ta = time.time()
        infer(0, 0)
        weights = weights if len(groups) > 1 else None
        for t in range(T):
            tb = time.time()
            if len(groups) > 1:
                infer(1, t)
                weights = None
            t_inf += time.time() - tb
            tb = time.time()
            receive(0, t)
            t_wait += time.time() - tb
            if t + 1 < T:
                tb = time.time()
                infer(0, t + 1)
                t_inf += time.time() - tb
            if len(groups) > 1:
                tb = time.time()
                receive(1, t)
                t_wait += time.time() - tb
        h, h_anchor = hs, has
        total_steps += T * E
        t_roll = time.time() - t0

        # bootstrap value for the observation after the rollout
        with torch.no_grad():
            obs_t = to_device(cur, device)
            st = torch.from_numpy(starts_next).to(device).view(1, -1, 1)
            emb = net.encode({k: obs_t[k] for k in OBS_KEYS})
            _l, _v, _a, z, _h = net.forward_seq_z(emb.unsqueeze(1), h * (1.0 - st))
            last_v = net.value_priv(z[:, 0], obs_t["priv"])[:, 0].cpu().numpy()
        adv = np.zeros((T, E), np.float32)
        lastgaelam = np.zeros(E, np.float32)
        for t in reversed(range(T)):
            nextv = last_v if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + args.gamma * nextv * nonterm - b_val[t]
            lastgaelam = delta + args.gamma * args.lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + b_val

        # ---------------------------------------------------------------- update
        net.train()
        warm = update - start_update < args.critic_warmup and not args.resume
        wins = [(e, s) for e in range(E) for s in range(0, T - L + 1, L)]
        stats = {"pg": [], "vf": [], "ent": [], "kl_anchor": [], "approx_kl": [], "clipfrac": []}
        adv_all = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _epoch in range(1 if warm else args.epochs):
            random.shuffle(wins)
            for mb in range(0, len(wins), args.minibatch_windows):
                chunk = wins[mb:mb + args.minibatch_windows]
                es = np.array([c[0] for c in chunk])
                ss = np.array([c[1] for c in chunk])
                tidx = ss[:, None] + np.arange(L)[None, :]  # (B, L)
                eidx = es[:, None].repeat(L, 1)
                arrs = {k: buf[k][tidx, eidx].reshape((-1,) + buf[k].shape[2:]) for k in STORE}
                obs_b = to_device(arrs, device)
                B = len(chunk)
                obs_seq = {k: v.reshape((B, L) + v.shape[1:]) for k, v in obs_b.items()}
                emb = net.encode({k: obs_seq[k] for k in OBS_KEYS})
                h0 = torch.from_numpy(b_h[ss, es]).to(device).unsqueeze(0)
                # b_h holds the state *before* the episode-start reset, so the
                # mask applies at t=0 as well.
                starts = torch.from_numpy(b_start[tidx, eidx]).to(device)
                logits, z = run_gru_masked(net, emb, h0, starts)
                # Critics read detached features: only the policy loss (and the
                # anchor) shape the shared encoder/GRU.
                vp = net.value_priv(z.detach(), obs_seq["priv"])[..., 0]
                ret_t = torch.from_numpy(ret[tidx, eidx]).to(device)
                vf = F.smooth_l1_loss(vp, ret_t, beta=2.0)
                vf_plain = F.smooth_l1_loss(net.critic(z.detach())[..., 0], ret_t, beta=2.0)
                if warm:
                    loss = vf + 0.1 * vf_plain
                    pg = ent = kla = akl = cf = torch.zeros((), device=device)
                else:
                    acts = torch.from_numpy(b_heads[tidx, eidx]).to(device)
                    lp, entropy = net.action_logprob_entropy(logits, acts)
                    oldlp = torch.from_numpy(b_logp[tidx, eidx]).to(device)
                    a_t = torch.from_numpy(adv_all[tidx, eidx]).to(device)
                    ratio = (lp - oldlp).exp()
                    pg = -torch.min(ratio * a_t, ratio.clamp(1 - args.clip, 1 + args.clip) * a_t).mean()
                    ent = entropy.mean()
                    kla = torch.zeros((), device=device)
                    if kl_beta > 0:
                        with torch.no_grad():
                            aemb = anchor.encode({k: obs_seq[k] for k in OBS_KEYS})
                            ha0 = torch.from_numpy(b_ha[ss, es]).to(device).unsqueeze(0)
                            alog, _az = run_gru_masked(anchor, aemb, ha0, starts)
                        for cur_l, ref_l in zip(torch.split(logits, HEAD_SIZES, -1),
                                                torch.split(alog, HEAD_SIZES, -1)):
                            kla = kla + (F.softmax(ref_l, -1) * (F.log_softmax(ref_l, -1)
                                                                - F.log_softmax(cur_l, -1))).sum(-1).mean()
                    loss = pg - args.ent * ent + kl_beta * kla + vf + 0.1 * vf_plain
                    if bc_windows and bc_w > 0:
                        loss = loss + bc_w * bc_loss(net, bc_windows, args.bc_batch, device)
                    with torch.no_grad():
                        akl = ((ratio - 1) - (lp - oldlp)).mean()
                        cf = ((ratio - 1).abs() > args.clip).float().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                for k_, v_ in (("pg", pg), ("vf", vf), ("ent", ent), ("kl_anchor", kla),
                               ("approx_kl", akl), ("clipfrac", cf)):
                    stats[k_].append(float(v_.detach()))

        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v.detach(), alpha=1.0 - args.ema)
                    else:
                        ema[k].copy_(v)
        # curriculum weights: favour categories that are not yet solved
        if cat_stats:
            weights = {}
            for c, dq in cat_stats.items():
                pr = float(np.mean(dq)) if dq else 0.0
                weights[c] = args.mastered_weight + (1.0 - pr) * (0.5 + 0.5 * (len(dq) >= 5))
        if recent:
            rr = list(recent)[-200:]
            row = {"update": update, "steps": total_steps, "sps": round(T * E / (time.time() - t0)),
                   "t_roll": round(t_roll, 1), "t_inf": round(t_inf, 1), "t_wait": round(t_wait, 1), "t_upd": round(time.time() - t0 - t_roll, 1),
                   "ep_return": round(float(np.mean([r["return"] for r in rr])), 3),
                   "exit": round(float(np.mean([r["exit"] for r in rr])), 3),
                   "coverage": round(float(np.mean([r["coverage"] for r in rr])), 3),
                   "warm": warm, "kl_beta": round(kl_beta, 3),
                   **{k: round(float(np.mean(v)), 4) for k, v in stats.items() if v},
                   "cat_exit": {c: round(float(np.mean(d)), 2) for c, d in sorted(cat_stats.items())},
                   "minutes": round((time.time() - t_start) / 60, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if (update + 1) % 10 == 0 or update == args.updates - 1:
            state = {"model": net.state_dict(), "config": {"hidden": net.hidden, "arch": "ego"},
                     "update": update, "steps": total_steps}
            torch.save(state, out / "ckpt.pt")
            if ema is not None:
                torch.save(ema, out / "ema.pt")
            if (update + 1) % args.save_every == 0:
                torch.save(state, out / f"ckpt_u{update + 1:04d}.pt")
                if ema is not None:
                    torch.save({**state, "model": {k: v.cpu() for k, v in ema.items()}},
                               out / f"ckpt_u{update + 1:04d}_ema.pt")

    for c in conns:
        c.send(("close", None))
    for pr in procs:
        pr.join(timeout=5)


if __name__ == "__main__":
    main()
