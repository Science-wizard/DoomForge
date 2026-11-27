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
CHANNELS = 6  # visited, wall, door, locked door, item/key, frontier
# Measured per-decision turn deltas (degrees), keyed by warm-up step and turn run.
# Measured on this engine build (per decision = 4 tics):
#   warm 0 (first action after reset): 0.000, 3.516, 14.063, 14.063 ...
#   warm 1 (second action):            1.758, 12.305, 14.063 ...
#   warm 2+ (steady state):            7.031, 12.305, 14.063 ...
TURN_TABLE = (
    (0.0, 3.516, 14.063),
    (1.758, 12.305, 14.063),
    (7.031, 12.305, 14.063),
)
FRAME_SKIP = 4
FRICTION = 0.90625  # Doom ORIG_FRICTION (0xE800 / 0x10000)
FRICTION4 = FRICTION ** 4
FRICTION_SUM = FRICTION + FRICTION ** 2 + FRICTION ** 3 + FRICTION ** 4


class MapBuilder:
    """``legacy=True`` reproduces the Phase-1..4 odometry that the Atlas-G and
    PPO-25 checkpoints were trained with (turn run reset on reversal, mean-
    velocity translation, no wall clip).  New models use the corrected odometry:
    heading error 0.02 deg and position error ~30 units after 500 steps, versus
    ~25 deg and ~430 units for the legacy integration."""

    def __init__(self, legacy: bool = False) -> None:
        self.legacy = legacy
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
        self.last_rays = None

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
        self.integrate(obs, prev_action)
        self._project_legacy(obs)

    def integrate(self, obs: dict, prev_action=None) -> None:
        """Dead-reckon heading and position from the executed action and vitals."""
        if prev_action is not None:
            a = np.asarray(prev_action).ravel()
            turn = int(a[4]) - int(a[5])
            warm = min(self.step, 2)
            if turn == 0:
                delta = 0.0
                self.turn_run = 0
                self.last_turn = 0
            else:
                # Doom's turnheld counts consecutive turning tics in either
                # direction, so a reversal keeps the accelerated rate.
                if self.legacy:
                    run = self.turn_run if turn == self.last_turn else 0
                else:
                    run = self.turn_run if self.last_turn != 0 else 0
                delta = TURN_TABLE[warm][min(run, 2)] * turn
                self.turn_run = run + 1
                self.last_turn = turn
            old_h = self.heading
            self.heading = (self.heading + delta) % 360.0
            v = np.asarray(obs["vitals"], np.float32)
            vf, vr = float(v[14]), float(v[15])
            # Doom per tic: momentum += thrust; position += momentum; momentum *=
            # friction.  Recover the (constant) thrust from the world-frame
            # momentum at the start and end of the step, then integrate the four
            # tics.  Measured 0.83 units/step mean error vs 2.25 for the mean-
            # velocity approximation.
            if self.legacy:
                avg_vf, avg_vr = 0.5 * (self.vf + vf), 0.5 * (self.vr + vr)
                h = math.radians(old_h + delta * 0.5)
                self.px += (avg_vf * math.cos(h) + avg_vr * math.sin(h)) * FRAME_SKIP
                self.py += (avg_vf * math.sin(h) - avg_vr * math.cos(h)) * FRAME_SKIP
                self.vf, self.vr = vf, vr
                self.step += 1
                self.last_rays = None
                return
            h0, h1 = math.radians(old_h), math.radians(self.heading)
            m0x = self.vf * math.cos(h0) + self.vr * math.sin(h0)
            m0y = self.vf * math.sin(h0) - self.vr * math.cos(h0)
            m4x = vf * math.cos(h1) + vr * math.sin(h1)
            m4y = vf * math.sin(h1) - vr * math.cos(h1)
            tx = (m4x - FRICTION4 * m0x) / FRICTION_SUM
            ty = (m4y - FRICTION4 * m0y) / FRICTION_SUM
            mx, my, dx, dy = m0x, m0y, 0.0, 0.0
            for _ in range(FRAME_SKIP):
                mx += tx
                my += ty
                dx += mx
                dy += my
                mx *= FRICTION
                my *= FRICTION
            # Momentum keeps building while the player is pinned against a
            # wall, so clip the step by the free space the previous rays saw in
            # the direction of travel (player radius 16 units).
            step_len = math.hypot(dx, dy)
            if self.last_rays is not None and step_len > 1e-6:
                rel = (math.atan2(dy, dx) - h0) % (2 * math.pi)
                k = int(round(rel / (2 * math.pi / 64))) % 64
                near = [self.last_rays[(k + j) % 64] for j in (-1, 0, 1)]
                free = max(0.0, min(near) * 1024.0 - 17.0)
                if step_len > free:
                    dx *= free / step_len
                    dy *= free / step_len
            self.px += dx
            self.py += dy
            self.vf, self.vr = vf, vr
            self.step += 1
        self.last_rays = [float(x) for x in np.asarray(obs["rays"]["distance"], np.float32)]

    def _project_legacy(self, obs: dict) -> None:
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

        # frontier: visited free cells adjacent to an unknown cell
        visited = self.grid[0] > 0
        known = (self.grid[0] > 0) | (self.grid[1] > 0) | (self.grid[2] > 0) | \
                (self.grid[3] > 0) | (self.grid[4] > 0)
        front = np.zeros_like(visited, dtype=np.uint8)
        front[1:-1, 1:-1] = (visited[1:-1, 1:-1] & (
            (~known[:-2, 1:-1]) | (~known[2:, 1:-1]) |
            (~known[1:-1, :-2]) | (~known[1:-1, 2:]))).astype(np.uint8)
        self.grid[5] = front

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


# ---------------------------------------------------------------------------
# Egocentric two-scale map (Atlas-E).
#
# Same odometry as MapBuilder, but a finer allocentric grid (64 units/cell,
# 160x160 = 10,240 units) with eight channels, read out as two heading-up crops:
#   fine   32x32 at  64 u/cell (2,048 units, the ray range)
#   coarse 32x32 at 256 u/cell (8,192 units, max-pooled)
# Heading-up crops make "frontier ahead-left" the same pixel pattern regardless of
# the absolute heading, which is what a small CNN can actually learn.  This is
# deterministic perception (projection of rays/entities at the estimated pose);
# there is no search or planning.
EGO_SIZE = 160
EGO_CELL = 64.0
EGO_CROP = 32
EGO_COARSE = 4
EGO_CHANNELS = 8  # free, wall, door, locked door, item, key, trail, frontier
EGO_MAP_CH = 2 * EGO_CHANNELS
_RAY_ANG = np.arange(64, dtype=np.float64) * (2 * np.pi / 64)
_RAY_T = np.linspace(0.0, 1.0, 33)[:-1]  # 32 samples along each ray (start .. hit)
_cr = (EGO_CROP / 2.0) - (np.arange(EGO_CROP) + 0.5)
EGO_FWD, EGO_LEFT = np.meshgrid(_cr, _cr, indexing="ij")  # row 0 = ahead, col 0 = left


class EgoMap(MapBuilder):
    def __init__(self) -> None:
        super().__init__(legacy=False)

    def reset(self) -> None:  # noqa: D401 - extends MapBuilder.reset
        super().reset()
        self.ego = np.zeros((EGO_CHANNELS, EGO_SIZE, EGO_SIZE), np.uint8)

    def ego_cell(self, x, y):
        cx = np.floor(np.asarray(x) / EGO_CELL).astype(np.int64) + EGO_SIZE // 2
        cy = np.floor(np.asarray(y) / EGO_CELL).astype(np.int64) + EGO_SIZE // 2
        return cx, cy

    def update(self, obs: dict, prev_action=None) -> None:
        self.integrate(obs, prev_action)
        g = self.ego
        rays = obs["rays"]
        dist = np.clip(np.nan_to_num(np.asarray(rays["distance"], np.float64), nan=1.0), 0.0, 1.0) * 1024.0
        hit = np.asarray(rays["hit_type"], np.int64)
        ang = math.radians(self.heading) + _RAY_ANG
        ca, sa = np.cos(ang), np.sin(ang)
        # free space: points strictly before the hit
        r = dist[:, None] * _RAY_T[None, :]
        fx, fy = self.ego_cell(self.px + r * ca[:, None], self.py + r * sa[:, None])
        ok = (fx >= 0) & (fx < EGO_SIZE) & (fy >= 0) & (fy < EGO_SIZE)
        g[0, fy[ok], fx[ok]] = 1
        # endpoints (only rays that actually hit something)
        ex, ey = self.ego_cell(self.px + dist * ca, self.py + dist * sa)
        ok = (ex >= 0) & (ex < EGO_SIZE) & (ey >= 0) & (ey < EGO_SIZE) & (dist < 1023.0)
        for t, ch in ((1, 1), (2, 2), (3, 3), (5, 4)):
            m = ok & (hit == t)
            g[ch, ey[m], ex[m]] = 1
        # doors that are now open let rays pass: clear stale door marks on free cells
        opened = (g[0] > 0) & ((g[2] > 0) | (g[3] > 0))
        if opened.any():
            ox, oy = self.ego_cell(self.px + dist * ca, self.py + dist * sa)
            hit_cells = np.zeros_like(opened)
            m2 = (ox >= 0) & (ox < EGO_SIZE) & (oy >= 0) & (oy < EGO_SIZE) & ((hit == 2) | (hit == 3))
            hit_cells[oy[m2], ox[m2]] = True
            stale = opened & ~hit_cells
            g[2][stale] = 0
            g[3][stale] = 0
        # entity sightings: items (5-8, 10) and keys (9)
        ent = obs["visible_entities"]
        mask = np.asarray(ent["mask"]).astype(bool)
        if mask.any():
            feat = np.asarray(ent["features"], np.float64)[mask]
            et = np.asarray(ent["entity_type"], np.int64)[mask]
            sel = (et >= 5) & (et <= 10)
            if sel.any():
                feat, et = feat[sel], et[sel]
                a2 = math.radians(self.heading) + np.arctan2(feat[:, 1], feat[:, 2])
                d2 = feat[:, 0] * 1024.0
                kx, ky = self.ego_cell(self.px + d2 * np.cos(a2), self.py + d2 * np.sin(a2))
                okk = (kx >= 0) & (kx < EGO_SIZE) & (ky >= 0) & (ky < EGO_SIZE)
                key = okk & (et == 9)
                item = okk & (et != 9)
                g[5, ky[key], kx[key]] = 1
                g[4, ky[item], kx[item]] = 1
        # trail: cells the agent has physically occupied
        ax, ay = self.ego_cell(self.px, self.py)
        if 0 <= ax < EGO_SIZE and 0 <= ay < EGO_SIZE:
            g[6, ay, ax] = 1
            g[0, ay, ax] = 1
        # frontier: seen-free cells next to unknown cells
        known = (g[0] | g[1] | g[2] | g[3]) > 0
        free = g[0] > 0
        fr = np.zeros((EGO_SIZE, EGO_SIZE), bool)
        fr[1:-1, 1:-1] = free[1:-1, 1:-1] & (~known[:-2, 1:-1] | ~known[2:, 1:-1]
                                              | ~known[1:-1, :-2] | ~known[1:-1, 2:])
        g[7] = fr

    def _sample(self, grid: np.ndarray, cell: float) -> np.ndarray:
        h = math.radians(self.heading)
        ch, cs = math.cos(h), math.sin(h)
        dx = (EGO_FWD * ch - EGO_LEFT * cs) * cell
        dy = (EGO_FWD * cs + EGO_LEFT * ch) * cell
        size = grid.shape[-1]
        ix = np.floor((self.px + dx) / cell).astype(np.int64) + size // 2
        iy = np.floor((self.py + dy) / cell).astype(np.int64) + size // 2
        ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        out = np.zeros((grid.shape[0], EGO_CROP, EGO_CROP), np.uint8)
        out[:, ok] = grid[:, iy[ok], ix[ok]]
        return out

    def ego_crop(self) -> np.ndarray:
        """16x32x32 uint8: fine (64 u/cell) then coarse (256 u/cell), heading-up."""
        k = EGO_COARSE
        coarse = self.ego.reshape(EGO_CHANNELS, EGO_SIZE // k, k, EGO_SIZE // k, k).max(axis=(2, 4))
        return np.concatenate([self._sample(self.ego, EGO_CELL),
                               self._sample(coarse, EGO_CELL * k)], axis=0)

    def crop(self) -> np.ndarray:
        return self.ego_crop()
