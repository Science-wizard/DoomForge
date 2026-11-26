"""Sector lookup / coverage for the proxy scorer (evaluation only).

The official scorer tracks unique sectors; the wrapper exposes `sector_polygons()`
but not the sector the player is standing in.  We precompute each sector's bounding
box and segment list, then test only the few candidate sectors around the player.
"""

from __future__ import annotations

import numpy as np


def _point_in_sector(x: float, y: float, segs: np.ndarray) -> bool:
    if segs is None or not len(segs):
        return False
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    cond = (y1 > y) != (y2 > y)
    if not cond.any():
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1[cond] + (y - y1[cond]) * (x2[cond] - x1[cond]) / (y2[cond] - y1[cond])
    return bool(np.count_nonzero(xint > x) % 2)


class SectorTracker:
    def __init__(self, env):
        polys = env.sector_polygons()
        self.ids = [int(sid) for sid, _ in polys]
        self.segs = [np.asarray(segs, np.float64).reshape(-1, 4) for _, segs in polys]
        self.lo = np.array([[s[:, 0].min(), s[:, 1].min()] for s in self.segs]) if self.segs else np.zeros((0, 2))
        self.hi = np.array([[s[:, 0].max(), s[:, 1].max()] for s in self.segs]) if self.segs else np.zeros((0, 2))
        self.visited: set[int] = set()

    def update(self, x: float, y: float) -> int:
        for k in range(len(self.ids)):
            if not (self.lo[k, 0] - 1 <= x <= self.hi[k, 0] + 1 and
                    self.lo[k, 1] - 1 <= y <= self.hi[k, 1] + 1):
                continue
            if _point_in_sector(x, y, self.segs[k]):
                sid = self.ids[k]
                self.visited.add(sid)
                return sid
        return -1

    @property
    def coverage(self) -> int:
        return len(self.visited)
