"""Evaluation entry point for DoomForge.

Contract (docs/CONTRACT.md §6):
    DoomAgent(observation_space, action_space, device)
    reset()
    act(observation) -> np.ndarray of shape (9,), values in {0,1}

Every action is sampled from the trained Atlas-E network.  Per decision:

    sanitise observation -> EgoMap update (dead reckoning + projection)
    -> one forward pass -> sample the six factorised heads -> 9 buttons

The map is deterministic perception built from observations and the agent's
own executed actions; there is no search, planning or rule-based override.
Sampling uses a generator re-seeded on every reset(), so runs are reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
PRIMARY = SUBMISSION_DIR / "weights" / "doomforge.pt"
SEED = 12345

try:  # normal case: the scorer puts submission/ on sys.path
    from model import HEAD_SIZES, DoomNet, build_from_state
    from mapmem import EgoMap
except ImportError:  # package-style import
    from submission.model import HEAD_SIZES, DoomNet, build_from_state
    from submission.mapmem import EgoMap

RGB_H, RGB_W = 30, 40


def _sanitize(obs: dict) -> dict:
    """Replace NaN/inf and clip type ids so a corrupt frame cannot poison the map."""
    rays, ent = obs["rays"], obs["visible_entities"]
    return {
        "rays": {"distance": np.clip(np.nan_to_num(np.asarray(rays["distance"], np.float32),
                                                   nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
                 "hit_type": np.clip(np.asarray(rays["hit_type"], np.int64), 0, 5)},
        "visible_entities": {
            "features": np.nan_to_num(np.asarray(ent["features"], np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            "entity_type": np.clip(np.asarray(ent["entity_type"], np.int64), 0, 11),
            "mask": np.asarray(ent["mask"])},
        "vitals": np.nan_to_num(np.asarray(obs["vitals"], np.float32), nan=0.0, posinf=1e4, neginf=-1e4),
        "rgb": obs["rgb"],
    }


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device if device in ("cpu", "cuda") else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        ckpt = torch.load(PRIMARY, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {})
        self.net = build_from_state(ckpt["model"], cfg).to(self.device).eval()
        self.temperature = float(cfg.get("temperature", 1.0))
        self.gen = torch.Generator(device="cpu")
        print(f"[DoomForge] Atlas-E loaded ({sum(p.numel() for p in self.net.parameters()):,} params)",
              file=sys.stderr)
        self._warned = False
        self.reset()
        self._warmup()

    # ------------------------------------------------------------------ contract
    def reset(self) -> None:
        self.gen.manual_seed(SEED)
        self.h = None
        self.prev = None
        self.turn_held = 0
        self.use_last = 0
        self.map = EgoMap()

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        try:
            obs = _sanitize(observation)
            self.map.update(obs, self.prev)          # integrate last executed action, project obs
            logits, _, _, self.h = self.net.forward_step(self._batch(obs), self.h)
            heads = self._sample(logits)
            a = DoomNet.actions_to_binary(heads)[0].cpu().numpy().astype(np.int8)
            self._remember(a)
            return a
        except Exception as exc:  # never let one bad observation zero the whole map
            if not self._warned:
                print(f"[DoomForge] act() fallback: {type(exc).__name__}: {exc}", file=sys.stderr)
                self._warned = True
            self.reset()
            return np.zeros(self.action_space.shape, dtype=np.int8)

    # ------------------------------------------------------------------ internals
    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return DoomNet.flat_action(logits, sample=False)
        parts = torch.split(logits[0].float().cpu() / self.temperature, list(HEAD_SIZES))
        return torch.stack([torch.multinomial(torch.softmax(p, -1), 1, generator=self.gen)[0]
                            for p in parts])[None]

    def _remember(self, a: np.ndarray) -> None:
        """Previous-action input: buttons, USE-last and how long a turn has been held."""
        tl, tr = int(a[4]), int(a[5])
        if tl and not tr:
            turn = self.turn_held + 1 if self.prev is not None and int(self.prev[4]) else 1
        elif tr and not tl:
            turn = self.turn_held + 1 if self.prev is not None and int(self.prev[5]) else 1
        else:
            turn = 0
        self.prev, self.turn_held, self.use_last = a, min(turn, 4), int(a[7])

    def _batch(self, obs: dict) -> dict:
        rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        rgb = np.ascontiguousarray(np.transpose(rgb[::4, ::4][:RGB_H, :RGB_W], (2, 0, 1)))
        p = np.zeros(12, np.float32)
        if self.prev is not None:
            p[:9] = self.prev[:9]
        p[9] = float(self.use_last)
        p[10] = min(float(self.turn_held), 4.0) / 4.0
        dev = self.device

        def t(x, dt=torch.float32):
            return torch.as_tensor(x, dtype=dt, device=dev)[None]

        ent = obs["visible_entities"]
        return {
            "rays_dist": t(obs["rays"]["distance"]),
            "rays_hit": t(obs["rays"]["hit_type"], torch.long),
            "ent_feat": t(ent["features"]),
            "ent_type": t(ent["entity_type"], torch.long),
            "ent_mask": t(np.asarray(ent["mask"], np.float32)),
            "vitals": t(obs["vitals"]),
            "rgb": t(rgb) / 255.0,
            "prev_action": t(p),
            "map": t(self.map.ego_crop()),
            "pose": t(self.map.pose()),
        }

    def _warmup(self) -> None:
        z = {"rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
             "visible_entities": {"features": np.zeros((32, 5), np.float32),
                                  "entity_type": np.zeros(32, np.int64), "mask": np.zeros(32, np.int8)},
             "vitals": np.zeros(17, np.float32), "rgb": np.zeros((120, 160, 3), np.uint8)}
        for _ in range(3):
            self.act(z)
        self.reset()
