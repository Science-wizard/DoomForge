"""Room-to-door drills: the door is on a wall and the start is random.

Unlike the corridor drills, the agent must turn, cross the room, align with the door
and use it.  The privileged teacher solves these; they bridge the gap between straight
door drills and generated multi-room maps.
"""

from __future__ import annotations

import random
from pathlib import Path

from .gen_wad import (LOCK_BLUE, MapSpec, Rect, T_BLUECARD, T_CLIP, T_PLAYER1, T_STIMPACK,
                      TILE, Door, _build_lumps, _door_tile, _write_wad)


def generate_room_door(seed: int, out_dir: str | Path | None = None, key: bool = False,
                       two_doors: bool = False) -> MapSpec:
    rng = random.Random(seed * 7919 + 31)
    H, W = 13, 30
    cy = 6
    grid = [[-1] * W for _ in range(H)]
    # start room 9x9
    for y in range(2, 11):
        for x in range(1, 10):
            grid[y][x] = 0
    # corridor and exit room, sector 1
    for x in range(11, W - 6):
        grid[cy][x] = 1
    for y in range(3, 10):
        for x in range(W - 6, W - 1):
            grid[y][x] = 1
    rooms = [Rect(1, 2, 10, 11), Rect(W - 6, 3, W - 1, 10)]
    if key:
        key_sec = len(rooms)
        for y in range(2, 5):
            for x in range(2, 5):
                grid[y][x] = key_sec
        rooms.append(Rect(2, 2, 5, 5))

    # door(s): one in the room's east wall, optionally a second before the exit room
    doors: list[Door] = []
    door_x = 10
    idx = len(doors)
    grid[cy][door_x] = _door_tile(idx)
    doors.append(Door(door_x, cy, "v", a=0, b=1, lock=0, sector=len(rooms) + idx, index=idx))
    if two_doors:
        dx2 = W - 6
        grid[cy][dx2] = _door_tile(len(doors))
        doors.append(Door(dx2, cy, "v", a=1, b=1, lock=0, sector=len(rooms) + len(doors), index=len(doors)))
    if key:
        (doors[-1] if len(doors) >= 2 else doors[0]).lock = LOCK_BLUE
    spec = MapSpec(seed=seed, tier=1, kind="room_door", width=W * TILE, height=H * TILE,
                   grid=grid, rooms=rooms, doors=doors,
                   start=(0.0, 0.0), start_angle=0,
                   exit_line=((W - 4) * TILE + 32, 3 * TILE + 24, (W - 4) * TILE + 32, 9 * TILE + 40),
                   exit_room=1)
    sx, sy = rng.randint(2, 8), rng.randint(3, 10)
    while grid[sy][sx] < 0:
        sx, sy = rng.randint(2, 8), rng.randint(3, 10)
    spec.start = (sx * TILE + 32.0, sy * TILE + 32.0)
    spec.start_angle = rng.choice((0, 90, 180, 270))
    spec.things.append((T_PLAYER1, *spec.start, spec.start_angle))
    if key:
        spec.things.append((T_BLUECARD, 3 * TILE + 32.0, 3 * TILE + 32.0, 0))
    for i in range(2):
        spec.things.append((rng.choice((T_STIMPACK, T_CLIP)), (W - 10 + i * 2) * TILE + 32.0,
                            (cy - 1) * TILE + 32.0, 0))
    spec.lumps = _build_lumps(spec)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"roomdoor_s{seed:05d}.wad"
        _write_wad(path, spec.lumps, spec.map_name)
        spec.path = path
    return spec
