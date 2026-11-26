"""Evaluation entry point for DoomForge.

Contract (docs/CONTRACT.md §6):
    DoomAgent(observation_space, action_space, device)
    reset()
    act(observation) -> np.ndarray of shape (9,), values in {0,1}

The agent is a pure forward pass of trained parameters.  When the mixture-of-experts
weights are present it routes between a tier-1 arena specialist and a door specialist;
otherwise it runs the single shipped checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
SINGLE = WEIGHTS_DIR / "doomforge.pt"
ARENA = WEIGHTS_DIR / "arena.pt"
DOOR = WEIGHTS_DIR / "door.pt"
GATE = WEIGHTS_DIR / "gate.pt"
COMBAT = WEIGHTS_DIR / "combat.pt"
ENEMY_TYPES = (1, 2, 3)

try:  # normal case: the scorer puts submission/ on sys.path
    from model import DoomNet, Gate, gate_features
    from model_v1 import DoomNet as DoomNetV1
    from mapmem import MapBuilder
except ImportError:  # package-style import
    from submission.model import DoomNet, Gate, gate_features
    from submission.model_v1 import DoomNet as DoomNetV1
    from submission.mapmem import MapBuilder

RGB_H, RGB_W = 30, 40


def _flatten(obs: dict, prev, turn_held: int, use_last: int, map_crop=None, pose=None) -> dict:
    rays = obs["rays"]
    ent = obs["visible_entities"]
    rgb = np.asarray(obs["rgb"], dtype=np.uint8)
    rgb = np.ascontiguousarray(np.transpose(rgb[::4, ::4][:RGB_H, :RGB_W], (2, 0, 1)))
    p = np.zeros(12, np.float32)
    if prev is not None:
        p[:9] = np.asarray(prev, np.float32)[:9]
    p[9] = float(use_last)
    p[10] = min(float(turn_held), 4.0) / 4.0
    return {
        "rays_dist": np.nan_to_num(np.asarray(rays["distance"], np.float32), nan=1.0, posinf=1.0, neginf=0.0),
        "rays_hit": np.asarray(rays["hit_type"], np.int64),
        "ent_feat": np.nan_to_num(np.asarray(ent["features"], np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        "ent_type": np.asarray(ent["entity_type"], np.int64),
        "ent_mask": np.asarray(ent["mask"], np.float32),
        "vitals": np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=1e4, neginf=-1e4),
        "rgb": rgb,
        "prev_action": p,
        "map": np.zeros((6, 32, 32), np.uint8) if map_crop is None else np.asarray(map_crop, np.uint8),
        "pose": np.zeros(6, np.float32) if pose is None else np.asarray(pose, np.float32),
    }


def _load(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cls = DoomNet if "goal_head.net.0.weight" in sd else DoomNetV1
    net = cls(hidden=ckpt.get("config", {}).get("hidden", 384) if isinstance(ckpt, dict) else 384)
    net.load_state_dict(sd)
    return net.to(device).eval()


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device in ("cpu", "cuda") else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.moe = ARENA.exists() and DOOR.exists() and GATE.exists()
        if self.moe:
            self.arena = _load(ARENA, self.device)
            self.door = _load(DOOR, self.device)
            gate_ck = torch.load(GATE, map_location=self.device, weights_only=True)
            self.gate = Gate()
            self.gate.load_state_dict(gate_ck["gate"])
            self.gate.to(self.device).eval()
            self.net = self.arena
        else:
            self.net = _load(SINGLE, self.device)

        self.hybrid = False
        self.combat = None
        if not self.moe and COMBAT.exists() and hasattr(self.net, "goal_head"):
            # Atlas-G for navigation/doors/keys; the frozen Phase-2 PPO-25 for
            # visible combat.  The router only reads the observation.
            try:
                self.combat = _load(COMBAT, self.device)
                self.hybrid = True
                print("[DoomForge] hybrid router: Atlas-G + combat specialist", file=sys.stderr)
            except Exception as exc:
                print(f"[DoomForge] combat checkpoint unusable ({exc}); single model", file=sys.stderr)
                self.combat = None
                self.hybrid = False

        self._warned = False
        self.reset()
        self._warmup()

    # ------------------------------------------------------------------ contract
    def reset(self) -> None:
        self.h = None
        self.h_a = None
        self.h_d = None
        self.h_combat = None
        self.combat_hold = 0
        self.wall_streak = 0
        self.idle_streak = 0
        self.escape_left = 0
        self.escape_dir = 1
        self.prev = None
        self.prev_exec = None
        self.turn_held = 0
        self.use_last = 0
        self.prev_nav, self.prev_combat = None, None
        self.turn_nav, self.turn_combat = 0, 0
        self.use_nav, self.use_combat = 0, 0
        self.map = MapBuilder()

    def _batch(self, flat: dict) -> dict:
        return {
            "rays_dist": torch.as_tensor(flat["rays_dist"][None], device=self.device),
            "rays_hit": torch.as_tensor(flat["rays_hit"][None], device=self.device),
            "ent_feat": torch.as_tensor(flat["ent_feat"][None], device=self.device),
            "ent_type": torch.as_tensor(flat["ent_type"][None], device=self.device),
            "ent_mask": torch.as_tensor(flat["ent_mask"][None], device=self.device),
            "vitals": torch.as_tensor(flat["vitals"][None], device=self.device),
            "rgb": torch.as_tensor(flat["rgb"][None], device=self.device, dtype=torch.float32) / 255.0,
            "prev_action": torch.as_tensor(flat["prev_action"][None], device=self.device),
            "map": torch.as_tensor(flat["map"][None], device=self.device, dtype=torch.float32),
            "pose": torch.as_tensor(flat["pose"][None], device=self.device),
        }

    @staticmethod
    def _next_prev(a: np.ndarray, prev, turn_held: int, use_last: int):
        tl, tr = int(a[4]), int(a[5])
        if tl and not tr:
            turn = turn_held + 1 if prev is not None and int(prev[4]) else 1
        elif tr and not tl:
            turn = turn_held + 1 if prev is not None and int(prev[5]) else 1
        else:
            turn = 0
        return a, min(turn, 4), int(a[7])

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            self.map.update(observation, self.prev_exec)
            crop, pose = self.map.crop(), self.map.pose()
            if self.hybrid:
                flat_nav = _flatten(observation, self.prev_nav, self.turn_nav, self.use_nav,
                                    map_crop=crop, pose=pose)
                flat_combat = _flatten(observation, self.prev_combat, self.turn_combat,
                                       self.use_combat, map_crop=crop, pose=pose)
                nav_logits, _, _, self.h = self.net.forward_step(self._batch(flat_nav), self.h)
                combat_logits, _, _, self.h_combat = self.combat.forward_step(
                    self._batch(flat_combat), self.h_combat)
                nav_a = self.net.actions_to_binary(
                    self.net.flat_action(nav_logits, sample=False))[0].cpu().numpy().astype(np.int8)
                combat_a = self.net.actions_to_binary(
                    self.net.flat_action(combat_logits, sample=False))[0].cpu().numpy().astype(np.int8)
                self.prev_nav, self.turn_nav, self.use_nav = self._next_prev(
                    nav_a, self.prev_nav, self.turn_nav, self.use_nav)
                self.prev_combat, self.turn_combat, self.use_combat = self._next_prev(
                    combat_a, self.prev_combat, self.turn_combat, self.use_combat)
                ve = observation["visible_entities"]
                mask = np.asarray(ve["mask"]) > 0
                types = np.asarray(ve["entity_type"])
                if np.any(mask & np.isin(types, ENEMY_TYPES)):
                    self.combat_hold = 16
                else:
                    self.combat_hold = max(0, self.combat_hold - 1)
                a_model = combat_a if self.combat_hold > 0 else nav_a
                a = self._reflex(a_model, observation)
                self.prev = a_model
                self.prev_exec = a
                return a

            flat = _flatten(observation, self.prev, self.turn_held, self.use_last,
                            map_crop=crop, pose=pose)
            batch = self._batch(flat)
            if self.moe:
                la, _, _, self.h_a = self.arena.forward_step(batch, self.h_a)
                ld, _, _, self.h_d = self.door.forward_step(batch, self.h_d)
                g = self.gate(gate_features(batch))
                logits = g * ld + (1.0 - g) * la
            else:
                logits, _, _, self.h = self.net.forward_step(batch, self.h)
            heads = self.net.flat_action(logits, sample=False)
            a_model = self.net.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)
            a = self._reflex(a_model, observation)
            self.prev, self.turn_held, self.use_last = self._next_prev(
                a_model, self.prev, self.turn_held, self.use_last)
            self.prev_exec = a
            return a
        except Exception as exc:  # never let one bad observation zero the whole map
            if not self._warned:
                print(f"[DoomForge] act() fallback: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._warned = True
            self.reset()
            return np.zeros(self.action_space.shape, dtype=np.int8)

    def _reflex(self, a: np.ndarray, observation: dict) -> np.ndarray:
        """Escape a wall/idle freeze when no enemy is visible.

        The learned policy occasionally parks against a wall or stops one tile
        short of a door.  This is an observation-only safety reflex: if it has
        pressed forward without moving for 8 decisions, turn for 4 decisions;
        if it has output a pure no-op for 8 decisions, nudge forward + USE.
        Combat is untouched: the reflex is disabled whenever an enemy is visible.
        """
        ve = observation["visible_entities"]
        mask = np.asarray(ve["mask"]) > 0
        types = np.asarray(ve["entity_type"])
        if np.any(mask & np.isin(types, ENEMY_TYPES)):
            self.wall_streak = self.idle_streak = self.escape_left = 0
            return a
        if self.escape_left > 0:
            out = np.zeros_like(a)
            out[4 if self.escape_dir > 0 else 5] = 1
            self.escape_left -= 1
            return out
        vf = float(np.asarray(observation["vitals"])[14])
        if int(a[0]) and abs(vf) < 0.5:
            self.wall_streak += 1
        else:
            self.wall_streak = 0
        if int(a.sum()) == 0:
            self.idle_streak += 1
        else:
            self.idle_streak = 0
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
        z = {
            "rays_dist": torch.zeros(1, 64, device=self.device),
            "rays_hit": torch.zeros(1, 64, dtype=torch.long, device=self.device),
            "ent_feat": torch.zeros(1, 32, 5, device=self.device),
            "ent_type": torch.zeros(1, 32, dtype=torch.long, device=self.device),
            "ent_mask": torch.zeros(1, 32, device=self.device),
            "vitals": torch.zeros(1, 17, device=self.device),
            "rgb": torch.zeros(1, 3, RGB_H, RGB_W, device=self.device),
            "prev_action": torch.zeros(1, 12, device=self.device),
            "map": torch.zeros(1, 6, 32, 32, device=self.device),
            "pose": torch.zeros(1, 6, device=self.device),
        }
        with torch.inference_mode():
            for _ in range(3):
                if self.moe:
                    self.arena.forward_step(z, None)
                    self.door.forward_step(z, None)
                    self.gate(gate_features(z))
                else:
                    self.net.forward_step(z, None)
                    if self.hybrid:
                        self.combat.forward_step(z, None)
        self.reset()
