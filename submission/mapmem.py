"""Deterministic ego-motion map (B+ minimal) — perception, not planning.

The map is built only from observations and actions:
  * heading from the measured turn table and the action history
  * translation from ego-frame velocity in `vitals[14:16]`
  * rays projected into a fixed allocentric grid (96 units/cell, origin at spawn)
  * item/key sightings from the visible-entity list

No engine state, no global coordinates, no search.  The policy reads a 32×32 crop
around the agent plus a 6-value pose vector.  This module is used by the collector,
the trainer and the submitted agent; it has no torch dependency.
"""

from __future__ import annotations

import math

import numpy as np

MAP_SIZE = 64
MAP_CELL = 96.0
CROP = 32
CHANNELS = 5  # visited, wall, door, locked door, item/key
# Measured per-decision turn deltas (degrees), keyed by warm-up step and turn run.
# Table from PLAN.md Appendix A / host measurement:
#   warm 0 (first action after reset): 0.000, 3.516, 14.063, 14.063 ...
#   warm 1 (second action):            1.758, 12.305, 14.063 ...
#   warm 2+ (steady state):            7.031, 12.305, 14.063 ...
TURN_TABLE = (
    (0.0, 3.516, 14.063),
    (1.758, 12.305, 14.063),
    (7.031, 12.305, 14.063),
)
FRAME_SKIP = 4


class MapBuilder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid = np.zeros((CHANNELS, MAP_SIZE, MAP_SIZE), np.uint8)
        self.px = 0.0
        self.py = 0.0
        self.heading = 0.0  # degrees, 0 = east, CCW positive
        self.vf = 0.0
        self.vr = 0.0
        self.step = 0
        self.turn_run = 0
        self.last_turn = 0

    # ------------------------------------------------------------------ geometry
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int(math.floor(x / MAP_CELL)) + MAP_SIZE // 2
        cy = int(math.floor(y / MAP_CELL)) + MAP_SIZE // 2
        return cx, cy

    def _line(self, x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # ------------------------------------------------------------------ update
    def update(self, obs: dict, prev_action=None) -> None:
        """Integrate the previous action, then project the current observation."""
        if prev_action is not None:
            a = np.asarray(prev_action).ravel()
            turn = int(a[4]) - int(a[5])
            warm = min(self.step, 2)
            if turn == 0:
                delta = 0.0
                self.turn_run = 0
                self.last_turn = 0
            else:
                run = self.turn_run if turn == self.last_turn else 0
                delta = TURN_TABLE[warm][min(run, 2)] * turn
                self.turn_run = run + 1
                self.last_turn = turn
            old_h = self.heading
            self.heading = (self.heading + delta) % 360.0
            v = np.asarray(obs["vitals"], np.float32)
            vf, vr = float(v[14]), float(v[15])
            # displacement over the step is best approximated by the mean velocity,
            # at the mid-step heading
            avg_vf, avg_vr = 0.5 * (self.vf + vf), 0.5 * (self.vr + vr)
            h = math.radians(old_h + delta * 0.5)
            dx = avg_vf * math.cos(h) + avg_vr * math.sin(h)
            dy = avg_vf * math.sin(h) - avg_vr * math.cos(h)
            self.px += dx * FRAME_SKIP
            self.py += dy * FRAME_SKIP
            self.vf, self.vr = vf, vr
            self.step += 1

        rays = obs["rays"]
        dist = np.asarray(rays["distance"], np.float32)
        hit = np.asarray(rays["hit_type"], np.int64)
        ax, ay = self.world_to_cell(self.px, self.py)
        for i in range(64):
            ang = math.radians(self.heading) + i * (2 * math.pi / 64)
            d = float(dist[i]) * 1024.0
            ex = self.px + d * math.cos(ang)
            ey = self.py + d * math.sin(ang)
            ex, ey = self.world_to_cell(ex, ey)
            for cx, cy in self._line(ax, ay, ex, ey):
                if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
                    self.grid[0, cy, cx] = 1
            if 0 <= ex < MAP_SIZE and 0 <= ey < MAP_SIZE:
                t = int(hit[i])
                if t == 1:
                    self.grid[1, ey, ex] = 1
                elif t == 2:
                    self.grid[2, ey, ex] = 1
                elif t == 3:
                    self.grid[3, ey, ex] = 1
                elif t == 5:
                    self.grid[4, ey, ex] = 1

        ent = obs["visible_entities"]
        feat = np.asarray(ent["features"], np.float32)
        etype = np.asarray(ent["entity_type"], np.int64)
        mask = np.asarray(ent["mask"]).astype(bool)
        for j in np.flatnonzero(mask):
            if 5 <= int(etype[j]) <= 10:  # health, armor, ammo, weapon, key, other
                ang = math.radians(self.heading) + math.atan2(float(feat[j, 1]), float(feat[j, 2]))
                d = float(feat[j, 0]) * 1024.0
                cx, cy = self.world_to_cell(self.px + d * math.cos(ang), self.py + d * math.sin(ang))
                if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
                    self.grid[4, cy, cx] = 1

    # ------------------------------------------------------------------ readout
    def crop(self) -> np.ndarray:
        cx, cy = self.world_to_cell(self.px, self.py)
        out = np.zeros((CHANNELS, CROP, CROP), np.uint8)
        x0, y0 = cx - CROP // 2, cy - CROP // 2
        for c in range(CHANNELS):
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(MAP_SIZE, x0 + CROP), min(MAP_SIZE, y0 + CROP)
            if sx1 > sx0 and sy1 > sy0:
                out[c, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.grid[c, sy0:sy1, sx0:sx1]
        return out

    def pose(self) -> np.ndarray:
        h = math.radians(self.heading)
        return np.array([
            np.clip(self.px / 2048.0, -1, 1),
            np.clip(self.py / 2048.0, -1, 1),
            math.sin(h),
            math.cos(h),
            np.clip(self.vf / 16.0, -1, 1),
            np.clip(self.vr / 16.0, -1, 1),
        ], np.float32)
