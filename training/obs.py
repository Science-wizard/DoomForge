"""Observation dict -> flat arrays / batched tensors (shared by training and eval)."""

from __future__ import annotations

import numpy as np

RGB_H, RGB_W = 30, 40   # 120x160 pooled 4x for the colour encoder
PREV = 12
MAP_CH, MAP_H, MAP_W = 6, 32, 32
POSE = 6


def downsample_rgb(rgb) -> np.ndarray:
    """120x160x3 uint8 -> 3x30x40 uint8 (nearest 4x4). Keeps colour, costs 3.6 KB."""
    a = np.asarray(rgb, dtype=np.uint8)
    a = a[::4, ::4][:RGB_H, :RGB_W]
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)))


def prev_vector(prev_action=None, turn_held: int = 0, use_last: int = 0) -> np.ndarray:
    p = np.zeros(PREV, np.float32)
    if prev_action is not None:
        p[:9] = np.asarray(prev_action, np.float32).ravel()[:9]
    p[9] = float(use_last)
    p[10] = min(float(turn_held), 4.0) / 4.0
    return p


def flatten_obs(obs: dict, prev_action=None, turn_held: int = 0, use_last: int = 0,
                map_crop=None, pose=None) -> dict:
    rays = obs["rays"]
    ent = obs["visible_entities"]
    out = {
        "rays_dist": np.asarray(rays["distance"], dtype=np.float32),
        "rays_hit": np.asarray(rays["hit_type"], dtype=np.int64),
        "ent_feat": np.asarray(ent["features"], dtype=np.float32),
        "ent_type": np.asarray(ent["entity_type"], dtype=np.int64),
        "ent_mask": np.asarray(ent["mask"], dtype=np.float32),
        "vitals": np.asarray(obs["vitals"], dtype=np.float32),
        "rgb": downsample_rgb(obs["rgb"]),
        "prev_action": prev_vector(prev_action, turn_held, use_last),
        "map": np.zeros((MAP_CH, MAP_H, MAP_W), np.uint8) if map_crop is None else np.asarray(map_crop, np.uint8),
        "pose": np.zeros(POSE, np.float32) if pose is None else np.asarray(pose, np.float32),
    }
    return out


def stack_obs(obs_list: list[dict], device="cpu") -> dict:
    import torch

    out = {}
    for key in ("rays_dist", "rays_hit", "ent_feat", "ent_type", "ent_mask", "vitals", "prev_action", "pose"):
        out[key] = torch.as_tensor(np.stack([np.asarray(o[key]) for o in obs_list]), device=device)
    rgb = np.stack([np.asarray(o["rgb"], np.uint8) for o in obs_list])
    out["rgb"] = torch.as_tensor(rgb, device=device, dtype=torch.float32) / 255.0
    mp = np.stack([np.asarray(o["map"], np.uint8) for o in obs_list])
    out["map"] = torch.as_tensor(mp, device=device, dtype=torch.float32)
    return out


def step_prev(action, prev_action, turn_held: int, use_last: int):
    """Update the previous-action bookkeeping for the next observation."""
    a = np.asarray(action).astype(np.int8).ravel()
    tl, tr = int(a[4]), int(a[5])
    if tl and not tr:
        turn_held = turn_held + 1 if prev_action is not None and int(prev_action[4]) else 1
    elif tr and not tl:
        turn_held = turn_held + 1 if prev_action is not None and int(prev_action[5]) else 1
    else:
        turn_held = 0
    return a, min(turn_held, 4), int(a[7])
