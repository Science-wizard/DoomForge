"""Frontier-exploration teacher.

The A* teacher beelines to the exit, which is not observable to the student.  This
teacher instead targets the nearest *unvisited navigable cell*, a goal that the
student can see in its own map (visited channel + walls).  It beelines to the exit
only when the exit is visible, and to a key only when the key is visible.  Doors and
combat are inherited from the base teacher.

This is training-time scaffolding only; the submitted agent has no planner.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .mapdata import load_mapdata
from .teacher import Teacher


class FrontierTeacher(Teacher):
    def __post_init__(self):
        super().__post_init__()
        self.visited: set[tuple[int, int]] = set()

    def reset(self, start_xy=None, angle=None) -> None:
        super().reset(start_xy, angle)
        self.visited = set()
        if start_xy is not None:
            self.visited.add(self.grid.world_to_cell(*start_xy))

    def act(self, obs, priv):
        if self.grid is not None:
            self.visited.add(self.grid.world_to_cell(priv["x"], priv["y"]))
        return super().act(obs, priv)

    def _choose_goal(self, priv) -> bool:
        g = self.grid
        pos = (priv["x"], priv["y"])
        keys = np.asarray(priv["keys"], np.float32)
        if g is None or g.free is None:
            return False

        # 1. Visible exit within 700 units: approach it (teaches the RGB cue).
        for gx, gy, kind, _ in g.exit_goals:
            if np.hypot(gx - pos[0], gy - pos[1]) < 700 and g.visible(pos, (gx, gy)):
                p = g.path(pos, (gx, gy), keys)
                if p is not None:
                    self.path, self.goal, self.goal_kind = p, (gx, gy), "exit_" + kind
                    return True

        # 2. Visible key while a locked door is still unresolved.
        for sec, (cx, cy, lock) in g.door_centres.items():
            if lock <= 0 or self._key_held(keys, lock):
                continue
            for kx, ky, klock in self.md.key_positions():
                if klock == lock and np.hypot(kx - pos[0], ky - pos[1]) < 800 and g.visible(pos, (kx, ky)):
                    p = g.path(pos, (kx, ky), keys)
                    if p is not None:
                        self.path, self.goal, self.goal_kind = p, (kx, ky), "key"
                        return True

        # 3. Nearest unvisited navigable cell (frontier).
        sx, sy = g.world_to_cell(*pos)
        free = g.free
        if not (0 <= sx < g.w and 0 <= sy < g.h) or not free[sy, sx]:
            sx, sy = g._nearest_free(free, sx, sy)
        seen = {(sx, sy)}
        q = deque([(sx, sy)])
        target = None
        while q:
            x, y = q.popleft()
            if (x, y) not in self.visited:
                target = (x, y)
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < g.w and 0 <= ny < g.h and free[ny, nx] and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        if target is not None:
            tx, ty = g.cell_to_world(*target)
            p = g.path(pos, (tx, ty), keys)
            if p is not None:
                self.path, self.goal, self.goal_kind = p, (tx, ty), "frontier"
                return True

        # 4. Everything reachable has been visited: go to the exit.
        for gx, gy, kind, _ in g.exit_goals:
            p = g.path(pos, (gx, gy), keys)
            if p is not None:
                self.path, self.goal, self.goal_kind = p, (gx, gy), "exit_" + kind
                return True
        return False


def make_frontier_teacher(wad: str, map_name: str = "MAP01") -> FrontierTeacher:
    return FrontierTeacher(load_mapdata(wad, map_name))
