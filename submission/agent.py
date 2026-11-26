"""Evaluation entry point for DoomForge.

Contract (docs/CONTRACT.md §6):
    DoomAgent(observation_space, action_space, device)
    reset()
    act(observation) -> np.ndarray of shape (9,), values in {0,1}

Every action is an argmax over the factorised heads of a trained network.  The
networks read the observation plus a deterministic ego-motion map that is built
from observations and the agent's own executed actions (perception, no search).

Configurations (chosen by which weights are present):
  * ``doomforge.pt`` alone                  -> single policy
  * ``doomforge.pt`` + ``combat.pt``        -> perception-routed hybrid: the combat
    expert acts while an enemy is visible (16-decision hysteresis), the primary
    policy otherwise; both see every observation so their memories stay current
  * legacy Atlas-G primary + ``specialist.pt`` -> the Phase-4 corridor-door gate
The architecture of each checkpoint (Atlas-G world map / Atlas-E ego map / v1)
is read from the state dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
PRIMARY = WEIGHTS_DIR / "doomforge.pt"
COMBAT = WEIGHTS_DIR / "combat.pt"
SPECIALIST = WEIGHTS_DIR / "specialist.pt"
ENEMY_TYPES = (1, 2, 3)
COMBAT_HOLD = 16

try:  # normal case: the scorer puts submission/ on sys.path
    from model import HEAD_SIZES, DoomNet, build_from_state
    from model_v1 import DoomNet as DoomNetV1
    from mapmem import EgoMap, MapBuilder
except ImportError:  # package-style import
    from submission.model import HEAD_SIZES, DoomNet, build_from_state
    from submission.model_v1 import DoomNet as DoomNetV1
    from submission.mapmem import EgoMap, MapBuilder

RGB_H, RGB_W = 30, 40


def _load(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if "goal_head.net.0.weight" in sd:
        net = build_from_state(sd, cfg)
    else:
        net = DoomNetV1(hidden=cfg.get("hidden", 384))
        net.load_state_dict(sd)
    return net.to(device).eval(), cfg


def _is_ego(net) -> bool:
    return getattr(net, "arch", "atlas") == "ego"


class _Expert:
    """One network plus its own recurrent state and previous-action bookkeeping."""

    def __init__(self, net, device: str, temperature: float = 0.0):
        self.net = net
        self.device = device
        self.ego = _is_ego(net)
        # temperature > 0: sample from the policy (seeded per map, so runs are
        # reproducible); 0: argmax.
        self.temperature = temperature
        self.gen = torch.Generator(device="cpu")
        self.reset()

    def reset(self) -> None:
        self.gen.manual_seed(12345)
        self.h = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0

    def act(self, obs: dict, crop: np.ndarray, pose: np.ndarray) -> np.ndarray:
        batch = _batch(obs, self.prev, self.turn_held, self.use_last, crop, pose, self.device)
        logits, _, _, self.h = self.net.forward_step(batch, self.h)
        if self.temperature > 0:
            parts = torch.split(logits[0].float().cpu() / self.temperature, list(HEAD_SIZES))
            heads = torch.stack([torch.multinomial(torch.softmax(p, -1), 1, generator=self.gen)[0]
                                 for p in parts])[None]
        else:
            heads = DoomNet.flat_action(logits, sample=False)
        a = DoomNet.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)
        tl, tr = int(a[4]), int(a[5])
        if tl and not tr:
            turn = self.turn_held + 1 if self.prev is not None and int(self.prev[4]) else 1
        elif tr and not tl:
            turn = self.turn_held + 1 if self.prev is not None and int(self.prev[5]) else 1
        else:
            turn = 0
        self.prev, self.turn_held, self.use_last = a, min(turn, 4), int(a[7])
        return a


def _batch(obs: dict, prev, turn_held: int, use_last: int, crop, pose, device: str) -> dict:
    rays = obs["rays"]
    ent = obs["visible_entities"]
    rgb = np.asarray(obs["rgb"], dtype=np.uint8)
    rgb = np.ascontiguousarray(np.transpose(rgb[::4, ::4][:RGB_H, :RGB_W], (2, 0, 1)))
    p = np.zeros(12, np.float32)
    if prev is not None:
        p[:9] = np.asarray(prev, np.float32)[:9]
    p[9] = float(use_last)
    p[10] = min(float(turn_held), 4.0) / 4.0
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)[None]  # noqa: E731
    return {
        "rays_dist": t(np.nan_to_num(np.asarray(rays["distance"], np.float32), nan=1.0, posinf=1.0, neginf=0.0)),
        "rays_hit": t(np.clip(np.asarray(rays["hit_type"], np.int64), 0, 5), torch.long),
        "ent_feat": t(np.nan_to_num(np.asarray(ent["features"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)),
        "ent_type": t(np.clip(np.asarray(ent["entity_type"], np.int64), 0, 11), torch.long),
        "ent_mask": t(np.asarray(ent["mask"], np.float32)),
        "vitals": t(np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=1e4, neginf=-1e4)),
        "rgb": t(rgb) / 255.0,
        "prev_action": t(p),
        "map": t(np.asarray(crop, np.float32)),
        "pose": t(np.asarray(pose, np.float32)),
    }


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device in ("cpu", "cuda") else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        net, cfg = _load(PRIMARY, self.device)
        self.primary = _Expert(net, self.device, float(cfg.get("temperature", 0.0)))
        self.combat = None
        self.specialist = None
        use_combat = cfg.get("route_combat", True)
        if use_combat and COMBAT.exists() and hasattr(net, "goal_head"):
            try:
                self.combat = _Expert(_load(COMBAT, self.device)[0], self.device)
            except Exception as exc:
                print(f"[DoomForge] combat checkpoint unusable ({exc}); single model", file=sys.stderr)
        if self.combat is not None and not self.primary.ego and SPECIALIST.exists():
            try:
                spec = _load(SPECIALIST, self.device)[0]
                if hasattr(spec, "goal_head") and not _is_ego(spec):
                    self.specialist = _Expert(spec, self.device)
            except Exception as exc:
                print(f"[DoomForge] specialist unusable ({exc})", file=sys.stderr)
        experts = [e for e in (self.primary, self.combat, self.specialist) if e is not None]
        self.need_ego = any(e.ego for e in experts)
        self.need_legacy = any(not e.ego for e in experts)
        # The reflex is a fallback for the Phase-3/4 hybrid, whose policies could
        # freeze against walls.  Atlas-E checkpoints are trained without it.
        self.use_reflex = cfg.get("reflex", not self.primary.ego)
        mode = ("hybrid" if self.combat is not None else "single") + \
               (" + door specialist" if self.specialist is not None else "")
        print(f"[DoomForge] {getattr(net, 'arch', 'v1')} primary, {mode}", file=sys.stderr)
        self._warned = False
        self.reset()
        self._warmup()

    # ------------------------------------------------------------------ contract
    def reset(self) -> None:
        for e in (self.primary, self.combat, self.specialist):
            if e is not None:
                e.reset()
        self.ego_map = EgoMap() if self.need_ego else None
        self.legacy_map = MapBuilder(legacy=True) if self.need_legacy else None
        self.prev_exec = None
        self.combat_hold = 0
        self.route_mode = None
        self.wall_streak = 0
        self.idle_streak = 0
        self.escape_left = 0
        self.escape_dir = 1

    @staticmethod
    def _sanitize(obs: dict) -> dict:
        """Replace NaN/inf so a corrupt observation cannot poison the map."""
        rays, ent = obs["rays"], obs["visible_entities"]
        return {
            "rays": {"distance": np.clip(np.nan_to_num(np.asarray(rays["distance"], np.float32),
                                                       nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
                     "hit_type": np.clip(np.asarray(rays["hit_type"], np.int64), 0, 5)},
            "visible_entities": {
                "features": np.nan_to_num(np.asarray(ent["features"], np.float32),
                                          nan=0.0, posinf=0.0, neginf=0.0),
                "entity_type": np.clip(np.asarray(ent["entity_type"], np.int64), 0, 11),
                "mask": np.asarray(ent["mask"])},
            "vitals": np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=1e4, neginf=-1e4),
            "rgb": obs["rgb"],
        }

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            observation = self._sanitize(observation)
            ego_crop = ego_pose = leg_crop = leg_pose = None
            if self.ego_map is not None:
                self.ego_map.update(observation, self.prev_exec)
                ego_crop, ego_pose = self.ego_map.ego_crop(), self.ego_map.pose()
            if self.legacy_map is not None:
                self.legacy_map.update(observation, self.prev_exec)
                leg_crop, leg_pose = self.legacy_map.crop(), self.legacy_map.pose()

            def run(e: _Expert) -> np.ndarray:
                return e.act(observation, ego_crop if e.ego else leg_crop,
                             ego_pose if e.ego else leg_pose)

            a = run(self.primary)
            ve = observation["visible_entities"]
            mask = np.asarray(ve["mask"]) > 0
            types = np.asarray(ve["entity_type"])
            enemy = bool(np.any(mask & np.isin(types, ENEMY_TYPES)))
            if self.combat is not None:
                combat_a = run(self.combat)
                self.combat_hold = COMBAT_HOLD if enemy else max(0, self.combat_hold - 1)
                if self.combat_hold > 0:
                    a = combat_a
                elif self.specialist is not None:
                    a = self._door_gate(observation, mask, types, a, run)
            if self.use_reflex:
                a = self._reflex(a, observation, enemy)
            self.prev_exec = a
            return a
        except Exception as exc:  # never let one bad observation zero the whole map
            if not self._warned:
                print(f"[DoomForge] act() fallback: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._warned = True
            self.reset()
            return np.zeros(self.action_space.shape, dtype=np.int8)

    # ------------------------------------------------------------------ legacy gate
    def _door_gate(self, observation, mask, types, a, run):
        """Phase-4 corridor-door specialist gate (legacy Atlas-G primary only)."""
        lm = self.legacy_map
        keys_held = float(np.asarray(observation["vitals"])[8:14].sum())
        gate = (not lm.grid[3].any() and not bool(np.any(mask & (types == 9)))
                and keys_held == 0 and lm.grid[2].any())
        if gate and self.route_mode is None:
            d = np.asarray(observation["rays"]["distance"])
            side = np.concatenate([d[:8], d[-8:]])
            self.route_mode = "spec" if float(np.median(side)) < 0.23 else "nav"
        if not gate and (lm.grid[3].any() or keys_held > 0):
            self.route_mode = None
        if gate and self.route_mode == "spec":
            return run(self.specialist)
        return a

    def _reflex(self, a: np.ndarray, observation: dict, enemy: bool) -> np.ndarray:
        """Observation-only anti-freeze fallback for the legacy hybrid.

        8 forward decisions without forward velocity -> turn for 4 decisions;
        8 pure no-ops -> forward + USE.  Disabled while an enemy is visible.
        """
        if enemy:
            self.wall_streak = self.idle_streak = self.escape_left = 0
            return a
        if self.escape_left > 0:
            out = np.zeros_like(a)
            out[4 if self.escape_dir > 0 else 5] = 1
            self.escape_left -= 1
            return out
        vf = float(np.asarray(observation["vitals"])[14])
        self.wall_streak = self.wall_streak + 1 if (int(a[0]) and abs(vf) < 0.5) else 0
        self.idle_streak = self.idle_streak + 1 if int(a.sum()) == 0 else 0
        if self.wall_streak >= 8:
            self.wall_streak = 0
            self.escape_left = 4
            self.escape_dir *= -1
        elif self.idle_streak >= 8:
            self.idle_streak = 0
            a = a.copy()
            a[0] = 1
            a[7] = 1
        return a

    # ------------------------------------------------------------------ helpers
    def _warmup(self) -> None:
        z = {"rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
             "visible_entities": {"features": np.zeros((32, 5), np.float32),
                                  "entity_type": np.zeros(32, np.int64), "mask": np.zeros(32, np.int8)},
             "vitals": np.zeros(17, np.float32), "rgb": np.zeros((120, 160, 3), np.uint8)}
        for _ in range(3):
            self.act(z)
        self.reset()
