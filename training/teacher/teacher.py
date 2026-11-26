"""Privileged scripted expert used to bootstrap the neural agent (training only).

The expert reads the ground-truth state that the scorer keeps private (position,
heading, key inventory, sector heights) and the static WAD geometry.  It is a
*teacher*: its actions are imitation targets and its behaviour is distilled into
the network.  The submitted agent contains none of this logic.

Skills demonstrated:
  * multi-room navigation with string-pulled A* waypoints
  * door use (press once, wait for the door to open)
  * keycard retrieval and locked-door progression
  * exit switches and walk-over exits
  * target acquisition from the ego-centric entity list
  * projectile dodging and weapon switching
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .mapdata import MapData, load_mapdata
from .navgrid import NavGrid

LOCK_TO_KEY = {1: 2, 2: 0, 3: 1}  # mapinfo lock -> vitals key index
ENEMY_TYPES = (1, 2, 3)
PROJECTILE_TYPE = 4


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class TeacherStats:
    steps: int = 0
    replans: int = 0
    doors_used: int = 0
    shots: int = 0
    dist: float = 0.0


@dataclass
class Teacher:
    md: MapData
    grid: NavGrid | None = None
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    replan_every: int = 24
    aggressive: bool = True
    # runtime
    path: list[tuple[float, float]] = field(default_factory=list)
    path_i: int = 0
    goal: tuple[float, float] | None = None
    goal_kind: str = ""
    door_task: dict | None = None
    use_cool: int = 0
    last_use: int = 0
    stuck: int = 0
    last_pos: tuple[float, float] = (0.0, 0.0)
    progress_pos: tuple[float, float] = (0.0, 0.0)
    progress_t: int = 0
    step: int = 0
    stuck_count: int = 0
    weapon_switch_cool: int = 0
    escape: int = 0
    stuck_moves: int = 0
    last_move_pos: tuple[float, float] = (0.0, 0.0)
    stats: TeacherStats = field(default_factory=TeacherStats)

    def __post_init__(self):
        if self.grid is None:
            self.grid = NavGrid(self.md)

    def reset(self, start_xy: tuple[float, float] | None = None, angle: float | None = None) -> None:
        if start_xy is not None and self.grid is not None:
            self.grid.build(start_xy)
        self.path = []
        self.path_i = 0
        self.goal = None
        self.goal_kind = ""
        self.door_task = None
        self.use_cool = 0
        self.last_use = 0
        self.stuck = 0
        self.stuck_count = 0
        self.progress_t = 0
        self.step = 0
        self.weapon_switch_cool = 0
        self.escape = 0
        self.stuck_moves = 0
        self.last_move_pos = start_xy if start_xy is not None else (0.0, 0.0)
        self.stats = TeacherStats()

    # ------------------------------------------------------------------ planning
    def _key_held(self, keys: np.ndarray, lock: int) -> bool:
        idx = LOCK_TO_KEY[lock]
        return bool(keys[idx] > 0.5 or keys[idx + 3] > 0.5)

    def _choose_goal(self, priv: dict) -> bool:
        assert self.grid is not None
        pos = (priv["x"], priv["y"])
        keys = np.asarray(priv["keys"], np.float32)

        # 1. Exit, if reachable with the keys we hold.
        best = None
        for gx, gy, kind, _ in self.grid.exit_goals:
            p = self.grid.path(pos, (gx, gy), keys)
            if p is not None and (best is None or len(p) < len(best[0])):
                best = (p, (gx, gy), "exit_" + kind)
        if best is not None:
            self.path, self.goal, self.goal_kind = best
            return True

        # 2. A locked door blocks the exit: fetch its key.
        options = []
        for sec, (cx, cy, lock) in self.grid.door_centres.items():
            if lock <= 0 or self._key_held(keys, lock):
                continue
            door_path = self.grid.path(pos, (cx, cy), keys, door_penalty=1.0)
            if door_path is None:
                continue
            for kx, ky, klock in self.md.key_positions():
                if klock != lock:
                    continue
                kp = self.grid.path(pos, (kx, ky), keys)
                if kp is not None:
                    options.append((len(kp), kp, (kx, ky), "key"))
        if options:
            options.sort(key=lambda o: o[0])
            _, self.path, self.goal, self.goal_kind = options[0]
            return True

        # 3. No exit found: explore the farthest reachable cell.
        free_ix, free_iy = np.nonzero(self.grid.free)
        if len(free_ix):
            cx, cy = self.grid.world_to_cell(*pos)
            d = (free_ix - cx) ** 2 + (free_iy - cy) ** 2
            for k in np.argsort(d)[::-1][:40]:
                gx, gy = self.grid.cell_to_world(int(free_ix[k]), int(free_iy[k]))
                p = self.grid.path(pos, (gx, gy), keys)
                if p is not None:
                    self.path, self.goal, self.goal_kind = p, (gx, gy), "explore"
                    return True
        return False

    def _replan(self, priv: dict) -> None:
        assert self.grid is not None
        self.stats.replans += 1
        self.progress_pos = (priv["x"], priv["y"])
        self.progress_t = self.step
        if not self._choose_goal(priv):
            self.path = []
            self.goal = None
        self.path_i = 1

    def _find_door_task(self, pos: tuple[float, float], priv: dict) -> dict | None:
        """First closed door cell on the path ahead, within reach."""
        if self.grid is None or not self.path:
            return None
        floor = np.asarray(priv["sector_floor"])
        ceil = np.asarray(priv["sector_ceiling"])
        for k in range(self.path_i, min(len(self.path), self.path_i + 18)):
            p = self.path[k]
            if math.hypot(p[0] - pos[0], p[1] - pos[1]) > 300:
                continue
            ix, iy = self.grid.world_to_cell(*p)
            if not (0 <= ix < self.grid.w and 0 <= iy < self.grid.h):
                continue
            sec = int(self.grid.door_id[iy, ix])
            if sec >= 0 and sec < len(floor) and (ceil[sec] - floor[sec]) < 56.0:
                return {"sec": sec, "cell": p, "idx": k, "stage": 1, "wait": 0}
        return None

    # ------------------------------------------------------------------ actions
    def act(self, obs: dict, priv: dict) -> np.ndarray:
        assert self.grid is not None
        a = np.zeros(9, np.int8)
        pos = (priv["x"], priv["y"])
        angle = float(priv["angle"])
        keys = np.asarray(priv["keys"], np.float32)
        self.step += 1
        self.stats.steps += 1
        self.use_cool = max(0, self.use_cool - 1)
        self.grid.update_doors(np.asarray(priv["sector_floor"]), np.asarray(priv["sector_ceiling"]))

        # Local door reflex: a ray hit on a door means the door is closed (open doors
        # let rays pass).  Press USE once when close, then wait; never bounce it.
        rays = obs.get("rays", {})
        if rays:
            d0 = np.asarray(rays["distance"])
            h0 = np.asarray(rays["hit_type"])
            ht0 = int(h0[0])
            keys_held = np.asarray(priv["keys"], np.float32)
            if ht0 == 2 and float(d0[0]) < 0.10:
                if self.use_cool == 0:
                    a[7] = 1
                    self.use_cool = 30
                    self.stats.doors_used += 1
                self.last_use = int(a[7])
                return a
            if ht0 == 3 and float(d0[0]) < 0.10 and float(keys_held.sum()) > 0:
                if self.use_cool == 0:
                    a[7] = 1
                    self.use_cool = 30
                    self.stats.doors_used += 1
                self.last_use = int(a[7])
                return a

        if self.step - self.progress_t > 30:
            moved = math.hypot(pos[0] - self.progress_pos[0], pos[1] - self.progress_pos[1])
            if moved < 12:
                self.stuck_count += 1
            self.progress_pos = pos
            self.progress_t = self.step

        if (not self.path or self.path_i >= len(self.path) or
                self.step % self.replan_every == 0 or self.stuck_count >= 2):
            if self.stuck_count >= 2:
                self.stuck_count = 0
            self._replan(priv)
            if not self.path:
                return a
        if self.door_task is None:
            self.door_task = self._find_door_task(pos, priv)

        # ---- door task: approach the exact path cell, use, wait, cross
        if self.door_task is not None:
            task = self.door_task
            sec = task["sec"]
            if sec >= len(priv["sector_floor"]) or not self.path:
                self.door_task = None
            else:
                dx, dy = task["cell"]
                closed = (priv["sector_ceiling"][sec] - priv["sector_floor"][sec]) < 56.0
                dist = math.hypot(dx - pos[0], dy - pos[1])
                lock = self.grid.door_centres.get(sec, (dx, dy, 0))[2]
                if dist > 280:
                    self.door_task = None
                elif lock > 0 and not self._key_held(keys, lock):
                    self.door_task = None
                    self._replan(priv)
                    return a
                elif task["stage"] == 1:
                    if not closed:
                        task["stage"] = 3
                    elif dist > 52:
                        want = math.degrees(math.atan2(dy - pos[1], dx - pos[0]))
                        err = wrap180(want - angle)
                        if abs(err) > 40:
                            a[4 if err > 0 else 5] = 1
                        else:
                            a[0] = 1
                            if abs(err) > 8:
                                a[4 if err > 0 else 5] = 1
                        self.last_move_pos = pos
                        self.last_use = 0
                        return a
                    else:
                        want = math.degrees(math.atan2(dy - pos[1], dx - pos[0]))
                        err = wrap180(want - angle)
                        if abs(err) > 14:
                            a[4 if err > 0 else 5] = 1
                        elif self.use_cool == 0:
                            a[7] = 1
                            self.use_cool = 5
                            task["stage"] = 2
                            task["wait"] = 0
                            self.stats.doors_used += 1
                        self.last_use = int(a[7])
                        return a
                elif task["stage"] == 2:
                    if not closed:
                        task["stage"] = 3
                    else:
                        task["wait"] += 1
                        if task["wait"] > 26:
                            task["stage"] = 1
                        self.last_use = 0
                        return a
                if task["stage"] == 3:
                    k = min(task["idx"] + 3, len(self.path) - 1)
                    target = self.path[k]
                    if dist > 90:
                        self.door_task = None
                    else:
                        want = math.degrees(math.atan2(target[1] - pos[1], target[0] - pos[0]))
                        err = wrap180(want - angle)
                        if abs(err) > 24:
                            a[4 if err > 0 else 5] = 1
                        else:
                            a[0] = 1
                            moved = math.hypot(pos[0] - self.last_move_pos[0], pos[1] - self.last_move_pos[1])
                            if moved < 2.0:
                                # Slide along the doorway until aligned with its centre line.
                                into = math.radians(want)
                                lat = (-math.sin(into), math.cos(into))
                                off = (pos[0] - dx) * lat[0] + (pos[1] - dy) * lat[1]
                                wl = (-math.sin(math.radians(angle)), math.cos(math.radians(angle)))
                                world = (-lat[0] * (1 if off > 0 else -1), -lat[1] * (1 if off > 0 else -1))
                                a[2 if (world[0] * wl[0] + world[1] * wl[1]) > 0 else 3] = 1
                        self.last_move_pos = pos
                        self.last_use = 0
                        return a

        # ---- combat override
        ent = obs.get("visible_entities", {})
        mask = np.asarray(ent.get("mask", np.zeros(0))).astype(bool)
        if mask.any() and self.aggressive:
            etype = np.asarray(ent["entity_type"])
            feat = np.asarray(ent["features"])
            enemy = mask & np.isin(etype, ENEMY_TYPES)
            proj = mask & (etype == PROJECTILE_TYPE)
            if proj.any():
                pf = feat[proj]
                near = pf[(pf[:, 0] < 0.35) & (pf[:, 3] < 0.5)]
                if len(near):
                    # Strafe away from the closest incoming projectile.
                    side = 2 if near[0, 1] > 0 else 3
                    a[side] = 1
                    if len(near) > 1:
                        a[0] = 1
                    self.last_use = 0
                    return a
            if enemy.any():
                ef = feat[enemy]
                dist = ef[:, 0] * 1024.0
                j = int(np.argmin(dist))
                d = float(dist[j])
                sin, cos = float(ef[j, 1]), float(ef[j, 2])
                rel = math.degrees(math.atan2(sin, cos))
                ammo = float(obs["vitals"][3]) + float(obs["vitals"][4])
                hp = float(priv["health"])
                engage = d < 640 and (cos > -0.25 or d < 240) and ammo > 0
                # Low health against a close enemy: disengage towards the exit.
                if hp < 34 and d < 260 and self.goal is not None:
                    engage = False
                if engage:
                    # Circle-strafe while firing; backpedal from melee range.
                    phase = (self.step // 7) % 2
                    if d < 130 and hp < 80:
                        a[1] = 1
                        a[2 if phase == 0 else 3] = 1
                    elif d < 380:
                        a[2 if phase == 0 else 3] = 1
                    else:
                        a[0] = 1
                        a[2 if phase == 0 else 3] = 1
                    if abs(rel) > 12.0:
                        a[4 if rel > 0 else 5] = 1
                        a[0] = 0
                        if abs(rel) < 45.0 and d > 200:
                            a[0] = 1
                    else:
                        a[6] = 1
                        self.stats.shots += 1
                    # Weapon switch to a better owned weapon when available.
                    if self.weapon_switch_cool == 0 and float(obs["vitals"][5]) > 0 and int(obs["vitals"][2]) == 2:
                        a[8] = 1
                        self.weapon_switch_cool = 40
                    self.weapon_switch_cool = max(0, self.weapon_switch_cool - 1)
                    self.last_use = 0
                    return a

        # ---- navigation
        wp, wi = self.grid.carrot(self.path, pos, self.path_i, distance=120.0)
        self.path_i = max(self.path_i, wi)
        dx, dy = wp[0] - pos[0], wp[1] - pos[1]
        want = math.degrees(math.atan2(dy, dx))
        err = wrap180(want - angle)
        # Lateral escape when a door frame or wall corner blocks forward motion.
        if self.escape > 0:
            self.escape -= 1
            lat = wrap180(want - angle)
            a[2 if lat > 0 else 3] = 1
            if abs(err) < 70:
                a[0] = 1
            if abs(err) > 20:
                a[4 if err > 0 else 5] = 1
        elif abs(err) > 42:
            a[4 if err > 0 else 5] = 1
        else:
            a[0] = 1
            if abs(err) > 8:
                a[4 if err > 0 else 5] = 1
            moved = math.hypot(pos[0] - self.last_move_pos[0], pos[1] - self.last_move_pos[1])
            if moved < 3.0:
                self.escape = 10
                self.stuck_moves += 1
            else:
                self.stuck_moves = 0
            self.last_move_pos = pos
        # Exit switch: press USE when close.
        if self.goal_kind == "exit_switch" and self.goal is not None:
            gd = math.hypot(self.goal[0] - pos[0], self.goal[1] - pos[1])
            if gd < 72 and self.use_cool == 0 and self.last_use == 0:
                a[:] = 0
                want = math.degrees(math.atan2(self.goal[1] - pos[1], self.goal[0] - pos[0]))
                err = wrap180(want - angle)
                if abs(err) > 18:
                    a[4 if err > 0 else 5] = 1
                else:
                    a[7] = 1
                    self.use_cool = 6
        self.last_use = int(a[7])
        self.stats.dist += math.hypot(pos[0] - self.last_pos[0], pos[1] - self.last_pos[1])
        self.last_pos = pos
        return a


def make_teacher(wad: str, map_name: str = "MAP01") -> Teacher:
    return Teacher(load_mapdata(wad, map_name))
