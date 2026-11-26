"""Evaluation entry point for DoomForge.

Contract (docs/CONTRACT.md §6):
    DoomAgent(observation_space, action_space, device)
    reset()
    act(observation) -> np.ndarray of shape (9,), values in {0,1}

The agent is a pure forward pass of trained parameters.  It never imports the
training package, vizdoom, or anything that touches the engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
WEIGHTS = WEIGHTS_DIR / "doomforge.pt"

try:  # normal case: the scorer puts submission/ on sys.path
    from model import DoomNet
    from mapmem import MapBuilder
except ImportError:  # package-style import
    from submission.model import DoomNet
    from submission.mapmem import MapBuilder

RGB_H, RGB_W = 30, 40


def _flatten(obs: dict, prev: np.ndarray | None, turn_held: int, use_last: int,
             map_crop=None, pose=None) -> dict:
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
        "map": np.zeros((5, 32, 32), np.uint8) if map_crop is None else np.asarray(map_crop, np.uint8),
        "pose": np.zeros(6, np.float32) if pose is None else np.asarray(pose, np.float32),
    }


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device in ("cpu", "cuda") else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        self.net = DoomNet(hidden=ckpt.get("config", {}).get("hidden", 384))
        self.net.load_state_dict(ckpt["model"])
        self.net.to(self.device).eval()

        self._warned = False
        self.reset()
        self._warmup()

    # ------------------------------------------------------------------ contract
    def reset(self) -> None:
        self.h = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map = MapBuilder()

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            self.map.update(observation, self.prev)
            flat = _flatten(observation, self.prev, self.turn_held, self.use_last,
                            map_crop=self.map.crop(), pose=self.map.pose())
            batch = {
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
            logits, value, aux, self.h = self.net.forward_step(batch, self.h)
            heads = self.net.flat_action(logits, sample=False)
            a = self.net.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)

            tl, tr = int(a[4]), int(a[5])
            if tl and not tr:
                self.turn_held = self.turn_held + 1 if self.prev is not None and int(self.prev[4]) else 1
            elif tr and not tl:
                self.turn_held = self.turn_held + 1 if self.prev is not None and int(self.prev[5]) else 1
            else:
                self.turn_held = 0
            self.turn_held = min(self.turn_held, 4)
            self.use_last = int(a[7])
            self.prev = a
            return a
        except Exception as exc:  # never let one bad observation zero the whole map
            if not self._warned:
                print(f"[DoomForge] act() fallback: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._warned = True
            self.reset()
            return np.zeros(self.action_space.shape, dtype=np.int8)

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
            "map": torch.zeros(1, 5, 32, 32, device=self.device),
            "pose": torch.zeros(1, 6, device=self.device),
        }
        with torch.inference_mode():
            for _ in range(3):
                self.net.forward_step(z, None)
        self.reset()
