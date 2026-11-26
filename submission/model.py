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
MAP_CH, MAP_H, MAP_W = 5, 32, 32
POSE = 6
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


class DoomNet(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.ray = RayEncoder(128)
        self.ent = EntityEncoder(128)
        self.vit = VitalsEncoder(64)
        self.rgb = RGBEncoder(128)
        self.prev = PrevEncoder(32)
        self.map = MapEncoder(128)
        self.pose = PoseEncoder(32)
        self.fuse = nn.Sequential(nn.Linear(640, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.actor = nn.Linear(hidden, sum(HEAD_SIZES))
        self.critic = nn.Linear(hidden, 1)
        self.aux = nn.Linear(hidden, AUX_SIZE)

    # ------------------------------------------------------------------ encoding
    def encode(self, obs: dict) -> torch.Tensor:
        lead = obs["rays_dist"].shape[:-1]
        n = math.prod(lead) if lead else 1
        r = self.ray(obs["rays_dist"].reshape(n, 64), obs["rays_hit"].reshape(n, 64))
        e = self.ent(obs["ent_feat"].reshape(n, 32, 5), obs["ent_type"].reshape(n, 32),
                     obs["ent_mask"].reshape(n, 32))
        v = self.vit(obs["vitals"].reshape(n, 17))
        g = self.rgb(obs["rgb"].reshape(n, 3, RGB_H, RGB_W))
        p = self.prev(obs["prev_action"].reshape(n, PREV))
        m = self.map(obs["map"].reshape(n, MAP_CH, MAP_H, MAP_W))
        q = self.pose(obs["pose"].reshape(n, POSE))
        z = self.fuse(torch.cat([r, e, v, g, p, m, q], dim=-1))
        return z.reshape(*lead, -1) if lead else z

    def forward_seq(self, emb: torch.Tensor, h: torch.Tensor | None = None):
        y, h = self.gru(emb, h)
        z = self.trunk(y)
        return self.actor(z), self.critic(z), self.aux(z), h

    def forward_step(self, obs: dict, h: torch.Tensor | None = None):
        emb = self.encode(obs).unsqueeze(1)
        logits, value, aux, h = self.forward_seq(emb, h)
        return logits[:, 0], value[:, 0], aux[:, 0], h

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


@dataclass
class ModelConfig:
    hidden: int = HIDDEN
