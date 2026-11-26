"""DoomForge policy network — Baseline A.

Shared by behaviour cloning, PPO and the submitted agent.  Compact enough for one
CPU thread and the 40 ms/decision budget: ~1.8 M parameters, 2–4 ms per decision.

Inputs (all ego-centric, no global coordinates):
  rays      64 distances + 6 hit types            -> circular 1-D CNN      -> 128
  entities  <=32 x (5 features + type)            -> attention + pool      -> 128
  vitals    17                                    -> MLP                   ->  64
  rgb       120x160x3, pooled to 30x40x3          -> small CNN             -> 128
  prev      9 action bits + USE-last + turn-held  -> MLP                   ->  32
                                         concat 480 -> GRU(384) -> 384

Heads:
  action  move(3) strafe(3) turn(3) attack(2) use(2) next(2) = 15 logits
  value   1
  aux     keys(6) enemy(1) door(1) exit_visible(1) exit_bearing(13)
          exit_distance(6) key_colours(6) damage_next8(1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_RAYS = 64
NUM_RAY_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
VITALS = 17
RGB_H, RGB_W = 30, 40
PREV = 12
MAP_CH, MAP_H, MAP_W = 6, 32, 32
POSE = 6
GATE_IN = 31
GOAL = 8
GOAL_TYPES = 4
HIDDEN = 384

HEADS = (("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("next", 2))
HEAD_SIZES = [n for _, n in HEADS]

# Auxiliary head layout, used by training losses and by the network definition.
AUX_KEYS = slice(0, 6)
AUX_ENEMY = 6
AUX_DOOR = 7
AUX_EXIT_VIS = 8
AUX_EXIT_BEARING = slice(9, 22)     # 13 classes: 12 bins + "not visible"
AUX_EXIT_DIST = slice(22, 28)       # 6 bins: 5 ranges + "not visible"
AUX_KEY_COLOURS = slice(28, 34)     # blue/yellow/red card + three skulls
AUX_DAMAGE = 34
AUX_SIZE = 35

EXIT_BEARING_CLASSES = 13
EXIT_DIST_CLASSES = 6


def actions_to_vector(move: int, strafe: int, turn: int, attack: int, use: int, nxt: int) -> list[int]:
    a = [0] * 9
    if move == 1:
        a[0] = 1
    elif move == 2:
        a[1] = 1
    if strafe == 1:
        a[2] = 1
    elif strafe == 2:
        a[3] = 1
    if turn == 1:
        a[4] = 1
    elif turn == 2:
        a[5] = 1
    a[6] = int(attack)
    a[7] = int(use)
    a[8] = int(nxt)
    return a


def vector_to_actions(vec) -> tuple[int, ...]:
    move = 1 if vec[0] else (2 if vec[1] else 0)
    strafe = 1 if vec[2] else (2 if vec[3] else 0)
    turn = 1 if vec[4] else (2 if vec[5] else 0)
    return move, strafe, turn, int(vec[6]), int(vec[7]), int(vec[8])


class RayEncoder(nn.Module):
    def __init__(self, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 32, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1, padding_mode="circular"), nn.GELU(),
        )
        self.out = nn.Linear(128, out)

    def forward(self, dist: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        b = dist.shape[0]
        ang = torch.arange(NUM_RAYS, device=dist.device, dtype=dist.dtype) * (2 * torch.pi / NUM_RAYS)
        hit1 = F.one_hot(hit.long(), NUM_RAY_TYPES).to(dist.dtype).permute(0, 2, 1)
        x = torch.cat([dist.unsqueeze(1), hit1,
                       torch.sin(ang)[None, None].expand(b, 1, -1),
                       torch.cos(ang)[None, None].expand(b, 1, -1)], dim=1)
        y = self.net(x)
        return self.out(torch.cat([y.mean(dim=2), y.max(dim=2).values], dim=1))


class EntityEncoder(nn.Module):
    def __init__(self, out: int = 128, d: int = 64, heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.inp = nn.Sequential(nn.Linear(5 + 16, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.out = nn.Linear(2 * d, out)

    def forward(self, feat: torch.Tensor, etype: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask > 0.5
        x = self.inp(torch.cat([feat, self.type_emb(etype.long())], dim=-1))
        safe = m.clone()
        safe[:, 0] = True  # all-masked rows must still be valid for attention
        attn = self.attn(x, x, x, key_padding_mask=~safe, need_weights=False)[0]
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ff(x))
        x = x * m.unsqueeze(-1).to(x.dtype)
        denom = m.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        mean = x.sum(dim=1) / denom
        mx = x.masked_fill(~m.unsqueeze(-1), -1e9).max(dim=1).values
        mx = torch.where(m.any(dim=1, keepdim=True), mx, torch.zeros_like(mx))
        return self.out(torch.cat([mean, mx], dim=-1))


class VitalsEncoder(nn.Module):
    def __init__(self, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(VITALS, 64), nn.GELU(), nn.Linear(64, out), nn.GELU())

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v * 0.01)


class RGBEncoder(nn.Module):
    """30x40x3 colour image -> 128. Coarse but enough for exit signs and door colours."""

    def __init__(self, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 5, out), nn.GELU())

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(rgb))


class MapEncoder(nn.Module):
    """5x32x32 allocentric crop -> 128. Deterministic map, learned readout."""

    def __init__(self, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(MAP_CH, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, out), nn.GELU())

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(m))


class EgoMapEncoder(nn.Module):
    """16x32x32 heading-up two-scale crop -> 192 (Atlas-E).

    The first layer keeps full resolution so one-cell features (a door in a wall,
    a frontier cell) survive; three stride-2 layers then reach a 4x4 summary.
    """

    def __init__(self, in_ch: int = 16, out: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, out), nn.GELU())

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(m))


PRIV = 16  # privileged critic features (training only, never used to act)


class PoseEncoder(nn.Module):
    def __init__(self, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(POSE, 32), nn.GELU(), nn.Linear(32, out), nn.GELU())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class PrevEncoder(nn.Module):
    def __init__(self, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PREV, 32), nn.GELU(), nn.Linear(32, out), nn.GELU())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


def gate_features(batch: dict) -> torch.Tensor:
    """Compact, observation-only routing features for the mixture-of-experts gate."""
    d = batch["rays_dist"]
    h = batch["rays_hit"]
    v = batch["vitals"]
    m = batch["map"]
    p = batch["prev_action"]
    door = ((h == 2) | (h == 3)).to(d.dtype)
    locked = (h == 3).to(d.dtype)
    front = d[..., :5]
    min_front = front.min(dim=-1).values
    door_any = door.max(dim=-1).values
    door_front = door[..., :5].max(dim=-1).values
    locked_any = locked.max(dim=-1).values
    min_door = torch.where(door > 0.5, d, torch.ones_like(d)).min(dim=-1).values
    keys = v[..., 8:14]
    stats = v[..., 0:5]
    map_means = m.mean(dim=(-1, -2))
    prev = torch.stack([p[..., 7], p[..., 10], v[..., 14], v[..., 15]], dim=-1)
    return torch.cat([front, min_front[..., None], door_any[..., None], door_front[..., None],
                      locked_any[..., None], min_door[..., None], keys, stats, map_means, prev], dim=-1)


class Gate(nn.Module):
    """Learned per-step routing weight: 1 = door expert, 0 = arena expert."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GATE_IN, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(f))


class GoalHead(nn.Module):
    """Predicts the ego-relative goal vector (8) and goal type (4) from the fused obs."""

    def __init__(self, inp: int = 640, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.vec = nn.Linear(hidden, GOAL)
        self.type_head = nn.Linear(hidden, GOAL_TYPES)

    def forward(self, base: torch.Tensor):
        h = self.net(base)
        return torch.tanh(self.vec(h)), self.type_head(h)


class GoalEncoder(nn.Module):
    def __init__(self, out: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GOAL, 32), nn.GELU(), nn.Linear(32, out), nn.GELU())

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class DoomNet(nn.Module):
    """Atlas-G (``arch="atlas"``, 6x32x32 world-aligned map) or Atlas-E
    (``arch="ego"``, 16x32x32 heading-up two-scale map + privileged critic)."""

    def __init__(self, hidden: int = HIDDEN, arch: str = "atlas"):
        super().__init__()
        self.hidden = hidden
        self.arch = arch
        self.map_ch = 16 if arch == "ego" else MAP_CH
        self.ray = RayEncoder(128)
        self.ent = EntityEncoder(128)
        self.vit = VitalsEncoder(64)
        self.rgb = RGBEncoder(128)
        self.prev = PrevEncoder(32)
        if arch == "ego":
            self.map = EgoMapEncoder(self.map_ch, 192)
            base = 128 + 128 + 64 + 128 + 32 + 192 + 32
        else:
            self.map = MapEncoder(128)
            base = 640
        self.pose = PoseEncoder(32)
        self.goal_head = GoalHead(base, 64)
        self.goal = GoalEncoder(32)
        self.fuse = nn.Sequential(nn.Linear(base + 32, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.actor = nn.Linear(hidden, sum(HEAD_SIZES))
        self.critic = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, AUX_SIZE)
        if arch == "ego":
            # Asymmetric critic: sees privileged progress features during PPO.
            # It is a training baseline only; acting never reads it.
            self.critic_priv = nn.Sequential(nn.Linear(hidden + PRIV, 256), nn.GELU(),
                                             nn.Linear(256, 1))

    def value_priv(self, z: torch.Tensor, priv: torch.Tensor) -> torch.Tensor:
        return self.critic_priv(torch.cat([z, priv], dim=-1))

    def forward_seq_z(self, emb: torch.Tensor, h: torch.Tensor | None = None):
        """Like forward_seq but also returns the trunk features (for the critic)."""
        y, h = self.gru(emb, h)
        z = self.trunk(y)
        return self.actor(z), self.critic(z), self.aux(z), z, h

    # ------------------------------------------------------------------ encoding
    def encode_with_goal(self, obs: dict):
        lead = obs["rays_dist"].shape[:-1]
        n = math.prod(lead) if lead else 1
        r = self.ray(obs["rays_dist"].reshape(n, 64), obs["rays_hit"].reshape(n, 64))
        e = self.ent(obs["ent_feat"].reshape(n, 32, 5), obs["ent_type"].reshape(n, 32),
                     obs["ent_mask"].reshape(n, 32))
        v = self.vit(obs["vitals"].reshape(n, 17))
        g = self.rgb(obs["rgb"].reshape(n, 3, RGB_H, RGB_W))
        p = self.prev(obs["prev_action"].reshape(n, PREV))
        m = self.map(obs["map"].reshape(n, self.map_ch, MAP_H, MAP_W))
        q = self.pose(obs["pose"].reshape(n, POSE))
        base = torch.cat([r, e, v, g, p, m, q], dim=-1)
        gv, gt = self.goal_head(base)
        ge = self.goal(gv)
        z = self.fuse(torch.cat([base, ge], dim=-1))
        if lead:
            z = z.reshape(*lead, -1)
            gv = gv.reshape(*lead, GOAL)
            gt = gt.reshape(*lead, GOAL_TYPES)
        return z, gv, gt

    def encode(self, obs: dict) -> torch.Tensor:
        z, _, _ = self.encode_with_goal(obs)
        return z

    def forward_seq(self, emb: torch.Tensor, h: torch.Tensor | None = None):
        y, h = self.gru(emb, h)
        z = self.trunk(y)
        return self.actor(z), self.critic(z), self.aux(z), h

    def forward_step_full(self, obs: dict, h: torch.Tensor | None = None):
        emb, gv, gt = self.encode_with_goal(obs)
        logits, value, aux, h = self.forward_seq(emb.unsqueeze(1), h)
        return logits[:, 0], value[:, 0], aux[:, 0], gv, gt, h

    def forward_step(self, obs: dict, h: torch.Tensor | None = None):
        logits, value, aux, gv, gt, h = self.forward_step_full(obs, h)
        return logits, value, aux, h

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def split_logits(logits: torch.Tensor) -> list[torch.Tensor]:
        return torch.split(logits, HEAD_SIZES, dim=-1)

    @staticmethod
    def flat_action(logits: torch.Tensor, sample: bool = True, generator=None) -> torch.Tensor:
        out = []
        for part in DoomNet.split_logits(logits):
            if sample:
                out.append(torch.distributions.Categorical(logits=part).sample())
            else:
                out.append(part.argmax(dim=-1))
        return torch.stack(out, dim=-1)

    @staticmethod
    def action_logprob_entropy(logits: torch.Tensor, actions: torch.Tensor):
        lp, ent = 0.0, 0.0
        for part, a in zip(DoomNet.split_logits(logits), actions.unbind(dim=-1)):
            d = torch.distributions.Categorical(logits=part)
            lp = lp + d.log_prob(a)
            ent = ent + d.entropy()
        return lp, ent

    @staticmethod
    def actions_to_binary(actions: torch.Tensor) -> torch.Tensor:
        m, s, t, atk, use, nxt = actions.unbind(dim=-1)
        return torch.stack([
            (m == 1).float(), (m == 2).float(),
            (s == 1).float(), (s == 2).float(),
            (t == 1).float(), (t == 2).float(),
            atk.float(), use.float(), nxt.float(),
        ], dim=-1)


def build_from_state(sd: dict, config: dict | None = None) -> "DoomNet":
    """Instantiate the right architecture for a saved state dict."""
    config = config or {}
    arch = config.get("arch") or ("ego" if sd["map.net.0.weight"].shape[1] == 16 else "atlas")
    net = DoomNet(hidden=config.get("hidden", HIDDEN), arch=arch)
    net.load_state_dict(sd)
    return net


@dataclass
class ModelConfig:
    hidden: int = HIDDEN
