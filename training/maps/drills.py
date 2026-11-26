"""Simple door / key drills — reliable demonstrations of one skill at a time.

Layout: start room -> straight 1-tile corridor with isolated doors -> exit room.
Optional side room holding a keycard and a locked door further down the corridor.
No monsters and no branching beyond the key alcove, so the expert's path is easy.
"""

from __future__ import annotations

import random
from pathlib import Path

from .gen_wad import (LOCK_BLUE, MapSpec, Rect, T_BLUECARD, T_CLIP, T_PLAYER1, T_STIMPACK,
                      TILE, Door, _build_lumps, _door_tile, _write_wad)


def generate_drill(seed: int, out_dir: str | Path | None = None, n_doors: int = 2,
                   key: bool = False, width: int = 34) -> MapSpec:
    rng = random.Random(seed * 7919 + 17)
    H, W = 9, width
    cy = 4
    grid = [[-1] * W for _ in range(H)]
    for y in range(2, 7):
        for x in range(1, 6):
            grid[y][x] = 0
        for x in range(W - 6, W - 1):
            grid[y][x] = 1
    for x in range(5, W - 5):
        grid[cy][x] = 2

    rooms = [Rect(1, 2, 6, 7), Rect(W - 6, 2, W - 1, 7), Rect(5, cy, W - 5, cy + 1)]
    doors: list[Door] = []
    span = max(1, W - 18)
    door_xs = sorted({8 + (i * span) // max(1, n_doors) for i in range(n_doors)})
    for x in door_xs:
        if not (7 <= x < W - 7):
            continue
        index = len(doors)
        grid[cy][x] = _door_tile(index)
        doors.append(Door(x, cy, "v", a=2, b=2, lock=0,
                          sector=len(rooms) + index, index=index))

    key_room = None
    if key and len(doors) >= 2:
        kx0 = door_xs[0] + 2
        kx1 = min(kx0 + 4, W - 7)
        key_sec = len(rooms)
        for y in range(cy + 2, H - 1):
            for x in range(kx0, kx1):
                grid[y][x] = key_sec
        grid[cy + 1][kx0] = key_sec
        rooms.append(Rect(kx0, cy + 2, kx1, H - 1))
        key_room = len(rooms) - 1
        doors[1].lock = LOCK_BLUE

    spec = MapSpec(seed=seed, tier=1, kind="drill", width=W * TILE, height=H * TILE,
                   grid=grid, rooms=rooms, doors=doors,
                   start=(3 * TILE + 32.0, 4 * TILE + 32.0), start_angle=0,
                   exit_line=((W - 4) * TILE + 32, 2 * TILE + 24, (W - 4) * TILE + 32, 6 * TILE + 40),
                   exit_room=1)
    spec.things.append((T_PLAYER1, *spec.start, 0))
    if key_room is not None:
        cx, cyy = rooms[key_room].center()
        spec.things.append((T_BLUECARD, cx * TILE + 32.0, cyy * TILE + 32.0, 0))
    for i in range(2):
        spec.things.append((rng.choice((T_STIMPACK, T_CLIP)), (10 + i * 6) * TILE + 32.0,
                            (cy - 2) * TILE + 32.0, 0))
    spec.lumps = _build_lumps(spec)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"drill_s{seed:05d}.wad"
        _write_wad(path, spec.lumps, spec.map_name)
        spec.path = path
    return spec
