"""16-unit navigation grid built from static WAD geometry (training only)."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .mapdata import MapData, SWITCH_EXITS

CELL = 16.0
AGENT_RADIUS = 20.0
DIAG = math.sqrt(2.0)


@dataclass
class NavGrid:
    md: MapData
    cell: float = CELL
    radius: float = AGENT_RADIUS
    x0: float = 0.0
    y0: float = 0.0
    w: int = 0
    h: int = 0
    free: np.ndarray | None = None        # (H, W) bool, reachable from the start region
    door_lock: np.ndarray | None = None   # (H, W) int8, -1 none, 0 door, >0 lock
    door_id: np.ndarray | None = None     # (H, W) int32, door sector id or -1
    door_closed: np.ndarray | None = None # (H, W) bool, closed door cells are opaque to LOS
    door_centres: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    door_pass: dict[int, tuple[float, float]] = field(default_factory=dict)
    exit_goals: list[tuple[float, float, str, float]] = field(default_factory=list)
    start_xy: tuple[float, float] = (0.0, 0.0)

    def build(self, start_xy: tuple[float, float], open_doors: set[int] | None = None,
              force: bool = False) -> None:
        if self.free is not None and not force:
            return
        md = self.md
        open_doors = open_doors or set()
        x0, y0, x1, y1 = md.bounds
        pad = 64.0
        self.x0, self.y0 = x0 - pad, y0 - pad
        self.w = int(math.ceil((x1 - x0 + 2 * pad) / self.cell))
        self.h = int(math.ceil((y1 - y0 + 2 * pad) / self.cell))
        blocked = np.zeros((self.h, self.w), bool)
        p1, p2 = md.line_points()
        door_lines = np.zeros(len(p1), bool)
        if len(p1):
            door_secs = np.array(sorted(md.info.door_sectors), dtype=np.int64)
            if len(door_secs):
                # Only the faces between a door and a real neighbour are passable.
                # The door's side/track walls (back == -1) stay blocking; treating
                # them as open carved false shortcuts through walls.
                door_lines = ((np.isin(md.front, door_secs) & (md.back >= 0)) |
                              (np.isin(md.back, door_secs) & (md.front >= 0)))
        walls = ((md.back < 0) | md.blocking) & ~door_lines
        for i in np.flatnonzero(walls):
            self._raster_segment(blocked, p1[i], p2[i])
        # Door cells: free cells inside the door sector's bounding box (shrunk a little).
        self.door_lock = np.full((self.h, self.w), -1, np.int8)
        self.door_id = np.full((self.h, self.w), -1, np.int32)
        self.door_closed = np.zeros((self.h, self.w), bool)
        self.door_centres = md.door_centres()
        for sec, (cx, cy, lock) in self.door_centres.items():
            mask = (md.front == sec) | (md.back == sec)
            if not mask.any():
                continue
            pts = np.concatenate([p1[mask], p2[mask]], axis=0)
            bx0, by0 = pts.min(axis=0)
            bx1, by1 = pts.max(axis=0)
            ix0, iy0 = self.world_to_cell(bx0, by0)
            ix1, iy1 = self.world_to_cell(bx1, by1)
            for iy in range(max(0, iy0), min(self.h, iy1 + 1)):
                for ix in range(max(0, ix0), min(self.w, ix1 + 1)):
                    if not blocked[iy, ix]:
                        self.door_lock[iy, ix] = lock
                        self.door_id[iy, ix] = sec
            # Passage direction: perpendicular to the door's longest face line.  Face lines
            # are the ones that connect the door sector to another sector (not the solid tracks).
            face = ((md.front == sec) & (md.back >= 0) & (md.back != sec)) | \
                   ((md.back == sec) & (md.front >= 0) & (md.front != sec))
            axis = None
            if face.any():
                seg = p2[face] - p1[face]
                ln = np.linalg.norm(seg, axis=1)
                k = int(np.argmax(ln))
                if ln[k] > 1e-6:
                    d = seg[k] / ln[k]
                    axis = (-d[1], d[0])
            if axis is None:
                # Fallback: from one neighbouring room centre to the other.
                sec_lines = md.info.sector_lines()
                adj = sorted({int(s) for s in np.concatenate([md.front[mask], md.back[mask]])
                              if int(s) >= 0 and int(s) != sec})
                if len(adj) >= 2:
                    def centre(s):
                        segs = sec_lines.get(s)
                        if segs is None or not len(segs):
                            return None
                        return segs.reshape(-1, 2).mean(axis=0)
                    ca, cb = centre(adj[0]), centre(adj[1])
                    if ca is not None and cb is not None:
                        v = ca - cb
                        n = float(np.linalg.norm(v))
                        if n > 1e-6:
                            axis = (v[0] / n, v[1] / n)
            if axis is None:
                axis = (1.0, 0.0) if (bx1 - bx0) >= (by1 - by0) else (0.0, 1.0)
            self.door_pass[sec] = axis
        # Flood fill from the start; outside-the-map cells are unreachable anyway.
        si, sj = self.world_to_cell(*start_xy)
        free = ~blocked
        if not (0 <= si < self.w and 0 <= sj < self.h) or not free[sj, si]:
            si, sj = self._nearest_free(free, si, sj)
        reach = np.zeros_like(free)
        if 0 <= si < self.w and 0 <= sj < self.h:
            stack = [(si, sj)]
            reach[sj, si] = True
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h and free[ny, nx] and not reach[ny, nx]:
                        reach[ny, nx] = True
                        stack.append((nx, ny))
        self.free = reach
        self.start_xy = start_xy
        self.exit_goals = self._build_exit_goals()

    # ------------------------------------------------------------------ geometry
    def _raster_segment(self, blocked: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        ix0, iy0 = self.world_to_cell(min(a[0], b[0]) - self.radius, min(a[1], b[1]) - self.radius)
        ix1, iy1 = self.world_to_cell(max(a[0], b[0]) + self.radius, max(a[1], b[1]) + self.radius)
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(self.w - 1, ix1), min(self.h - 1, iy1)
        if ix1 < ix0 or iy1 < iy0:
            return
        xs = self.x0 + (np.arange(ix0, ix1 + 1) + 0.5) * self.cell
        ys = self.y0 + (np.arange(iy0, iy1 + 1) + 0.5) * self.cell
        gx, gy = np.meshgrid(xs, ys)
        ab = b - a
        ll = float(ab @ ab)
        if ll < 1e-9:
            d = np.hypot(gx - a[0], gy - a[1])
        else:
            t = ((gx - a[0]) * ab[0] + (gy - a[1]) * ab[1]) / ll
            t = np.clip(t, 0.0, 1.0)
            d = np.hypot(gx - (a[0] + t * ab[0]), gy - (a[1] + t * ab[1]))
        blocked[iy0:iy1 + 1, ix0:ix1 + 1] |= d <= self.radius

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) / self.cell), int((y - self.y0) / self.cell)

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell

    def _nearest_free(self, free: np.ndarray, ix: int, iy: int, max_r: int = 24) -> tuple[int, int]:
        for r in range(1, max_r):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = ix + dx, iy + dy
                    if 0 <= x < self.w and 0 <= y < self.h and free[y, x]:
                        return x, y
        return ix, iy

    # ------------------------------------------------------------------ exits
    def _build_exit_goals(self) -> list[tuple[float, float, str, float]]:
        md = self.md
        p1, p2 = md.line_points()
        out = []
        for x1, y1, x2, y2, sp in md.exit_lines():
            dx, dy = x2 - x1, y2 - y1
            n = math.hypot(dx, dy) or 1.0
            nx, ny = dy / n, -dx / n  # front side (right of direction)
            kind = "switch" if sp in SWITCH_EXITS else "walk"
            if kind == "walk":
                # Walk-over line: target the midpoint itself so the path crosses it.
                out.append(((x1 + x2) / 2, (y1 + y2) / 2, kind, 0.0))
                continue
            # Switch: stand in front of the line, inside a reachable sector.
            for sign in (1.0, -1.0):
                tx, ty = (x1 + x2) / 2 + nx * 56 * sign, (y1 + y2) / 2 + ny * 56 * sign
                ix, iy = self.world_to_cell(tx, ty)
                if 0 <= ix < self.w and 0 <= iy < self.h and self.free is not None and self.free[iy, ix]:
                    out.append((tx, ty, kind, 40.0))
                    break
            else:
                ix, iy = self.world_to_cell((x1 + x2) / 2, (y1 + y2) / 2)
                sx, sy = self._nearest_free(self.free if self.free is not None else np.ones((self.h, self.w), bool), ix, iy)
                out.append((*self.cell_to_world(sx, sy), kind, 48.0))
        return out

    # ------------------------------------------------------------------ search
    def path(self, start: tuple[float, float], goal: tuple[float, float],
             keys: np.ndarray | None = None, door_penalty: float = 4.0,
             max_expand: int = 120_000) -> list[tuple[float, float]] | None:
        """A* over free cells. Locked doors are impassable without the key."""
        if self.free is None:
            return None
        keys = keys if keys is not None else np.zeros(6, np.float32)
        si, sj = self.world_to_cell(*start)
        gi, gj = self.world_to_cell(*goal)
        free = self.free
        if not (0 <= gi < self.w and 0 <= gj < self.h):
            return None
        if not free[gj, gi]:
            gi, gj = self._nearest_free(free, gi, gj)
        if not free[sj, si]:
            si, sj = self._nearest_free(free, si, sj)
        if (si, sj) == (gi, gj):
            return [start, goal]

        def passable(x: int, y: int) -> float:
            if not free[y, x]:
                return -1.0
            lock = self.door_lock[y, x]
            if lock is None or lock < 0:
                return 1.0
            if lock > 0:
                idx = {1: 2, 2: 0, 3: 1}[lock]
                if not (keys[idx] or keys[idx + 3]):
                    return -1.0
            return door_penalty

        W, H = self.w, self.h
        g = np.full((H, W), np.inf, np.float32)
        g[sj, si] = 0.0
        parent = np.full((H, W), -1, np.int32)
        open_heap = [(0.0, si, sj)]
        expanded = 0
        while open_heap and expanded < max_expand:
            f, x, y = heapq.heappop(open_heap)
            if x == gi and y == gj:
                break
            if f - self._h(x, y, gi, gj) > g[y, x] + 1e-4:
                continue
            expanded += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                c = passable(nx, ny)
                if c < 0:
                    continue
                if dx and dy:
                    if passable(x + dx, y) < 0 or passable(x, y + dy) < 0:
                        continue
                    c *= DIAG
                ng = g[y, x] + c
                if ng < g[ny, nx] - 1e-5:
                    g[ny, nx] = ng
                    parent[ny, nx] = y * W + x
                    heapq.heappush(open_heap, (ng + self._h(nx, ny, gi, gj), nx, ny))
        if not np.isfinite(g[gj, gi]):
            return None
        cells = []
        cx, cy = gi, gj
        while True:
            cells.append((cx, cy))
            p = parent[cy, cx]
            if p < 0:
                break
            cx, cy = int(p % W), int(p // W)
        cells.reverse()
        pts = [self.cell_to_world(x, y) for x, y in cells]
        pts[0] = start
        pts[-1] = goal
        return pts

    def _h(self, x: int, y: int, gx: int, gy: int) -> float:
        dx, dy = abs(x - gx), abs(y - gy)
        return (dx + dy) + (DIAG - 2.0) * min(dx, dy)

    def update_doors(self, floor: np.ndarray, ceil: np.ndarray) -> None:
        if self.door_id is None:
            return
        for sec in self.door_centres:
            if sec < len(floor) and sec < len(ceil):
                closed = (ceil[sec] - floor[sec]) < 56.0
                self.door_closed[self.door_id == sec] = closed

    def visible(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Grid line-of-sight. Closed door cells are opaque."""
        if self.free is None:
            return False
        x0, y0 = self.world_to_cell(*a)
        x1, y1 = self.world_to_cell(*b)
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(1, n + 1):
            t = i / n
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            if not (0 <= x < self.w and 0 <= y < self.h) or not self.free[y, x]:
                return False
            if self.door_closed is not None and self.door_closed[y, x]:
                return False
        return True

    def carrot(self, path: list[tuple[float, float]], start: tuple[float, float],
               path_i: int = 0, distance: float = 110.0) -> tuple[tuple[float, float], int]:
        """Point ~``distance`` map units ahead along the path (no corner cutting)."""
        if not path:
            return start, path_i
        lo = max(0, path_i - 4)
        hi = min(len(path), path_i + 14)
        best_i, best_d = lo, float("inf")
        for i in range(lo, hi):
            d = (path[i][0] - start[0]) ** 2 + (path[i][1] - start[1]) ** 2
            if d < best_d:
                best_i, best_d = i, d
        acc = 0.0
        prev = start
        i = best_i
        while i < len(path) - 1:
            i += 1
            acc += math.hypot(path[i][0] - prev[0], path[i][1] - prev[1])
            prev = path[i]
            if acc >= distance:
                break
        # If the straight line to the carrot is blocked, back off to the last visible point.
        j = i
        while j > best_i and not self.visible(start, path[j]):
            j -= 1
        return path[j], j

