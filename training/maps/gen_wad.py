"""Procedural UDMF map generator for the DevForge DOOM curriculum.

Maps are authored as a tile grid (64 map units per tile) and emitted as UDMF
``TEXTMAP`` inside a PWAD.  ZDoom (and therefore ViZDoom) builds BSP nodes at load
time, so no external node builder is required.

Difficulty tiers mirror the hidden evaluation suite:
    tier 1  arena        combat, cover, exit at the far end
    tier 2  rooms        multi-room exploration, unlocked doors, loops
    tier 3  progression  keycard -> locked door -> exit
    tier 4  gauntlet     two keys, two locks, loops, scarce resources

Every map is a pure function of (seed, tier): deterministic geometry, things and
textures.  This module is training-only and is never imported by the agent.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from dataclasses import dataclass, field
from pathlib import Path

TILE = 64  # map units per tile

# Textures verified against the bundled Freedoom 2 IWAD.
WALLS = ("STONE2", "STONE3", "BRICK1", "BRICK10", "BROWN1", "METAL1", "COMPUTE1",
         "SUPPORT2", "TEKWALL1", "GRAY1", "WOOD1", "STARTAN2", "BIGBRIK1", "BLAKWAL1")
FLOORS = ("FLOOR0_1", "FLOOR0_2", "FLOOR1_1", "FLOOR4_8", "FLOOR5_1", "FLOOR7_2", "FLAT1")
CEILINGS = ("CEIL1_1", "CEIL3_5", "CEIL5_1", "CEIL5_2")
DOOR_TEXTURE = "BIGDOOR2"
DOOR_TRACK = "DOORTRAK"
DOOR_TEXTURES = ("BIGDOOR2", "DOOR1", "BIGDOOR4")
LOCK_TEXTURE = {1: "DOORRED", 2: "DOORBLU", 3: "DOORYEL"}
EXIT_TEXTURES = ("EXITSIGN", "SW1EXIT", "SW1BRN1")

# Doom thing numbers (identical in UDMF).
T_PLAYER1 = 1
T_BLUECARD, T_YELLOWCARD, T_REDCARD = 5, 6, 13
T_STIMPACK, T_MEDIKIT, T_HEALTH_BONUS = 2011, 2012, 2014
T_GREEN_ARMOR, T_BLUE_ARMOR = 2018, 2019
T_CLIP, T_SHELLS, T_CELL = 2007, 2008, 2047
T_SHOTGUN, T_CHAINGUN, T_ROCKETLAUNCHER, T_PLASMA = 2001, 2002, 2003, 2004
T_ZOMBIE, T_SHOTGUNGUY, T_IMP, T_DEMON, T_CACODEMON = 3004, 9, 3001, 3002, 3005

# Doom linedef specials.
S_DOOR_OPEN = 1
S_DOOR_BLUE = 26
S_DOOR_YELLOW = 27
S_DOOR_RED = 28
S_EXIT_WALK = 52

# Lock numbers in doom_env.mapinfo: 0 none, 1 red, 2 blue, 3 yellow.
LOCK_RED, LOCK_BLUE, LOCK_YELLOW = 1, 2, 3
KEY_THING = {LOCK_RED: T_REDCARD, LOCK_BLUE: T_BLUECARD, LOCK_YELLOW: T_YELLOWCARD}
DOOR_SPECIAL = {LOCK_RED: S_DOOR_RED, LOCK_BLUE: S_DOOR_BLUE, LOCK_YELLOW: S_DOOR_YELLOW}


@dataclass
class Rect:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def center(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2


@dataclass
class Door:
    x: int
    y: int
    axis: str            # 'v': passage east-west (door sits in a vertical wall)
    a: int = -1          # room on one side
    b: int = -1          # room on the other side
    lock: int = 0
    sector: int = -1
    index: int = -1

    @property
    def special(self) -> int:
        return DOOR_SPECIAL.get(self.lock, S_DOOR_OPEN)


@dataclass
class MapSpec:
    seed: int
    tier: int
    kind: str
    width: int
    height: int
    grid: list[list[int]]                 # -1 solid, >=0 room id, -2-index door
    rooms: list[Rect]
    doors: list[Door]
    start: tuple[float, float]
    start_angle: int
    exit_line: tuple[float, float, float, float]
    exit_room: int
    keys: list[tuple[int, int]] = field(default_factory=list)   # (room id, lock)
    things: list[tuple[int, float, float, int]] = field(default_factory=list)
    adj: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    lumps: dict[str, bytes] = field(default_factory=dict)
    path: Path | None = None
    exit_sign_texture: str | None = None
    door_texture_override: str | None = None

    @property
    def map_name(self) -> str:
        return "MAP01"

    @property
    def door_rooms(self) -> dict[int, Door]:
        return {d.sector: d for d in self.doors}


def _door_tile(index: int) -> int:
    return -2 - index


def _is_door(v: int) -> bool:
    return v <= -2


# --------------------------------------------------------------------------- layout
def _split_layout(rng: random.Random, size: int, max_depth: int, min_room: int = 4):
    grid = [[-1] * size for _ in range(size)]
    rooms: list[Rect] = []
    splits: list[tuple[str, int, int, int]] = []  # axis, gap coordinate, range lo, range hi

    def carve(rect: Rect, depth: int) -> None:
        can_x = rect.w >= 2 * min_room + 1 and depth < max_depth
        can_y = rect.h >= 2 * min_room + 1 and depth < max_depth
        if not (can_x or can_y):
            rid = len(rooms)
            for y in range(rect.y0, rect.y1):
                for x in range(rect.x0, rect.x1):
                    if grid[y][x] == -1:
                        grid[y][x] = rid
            rooms.append(rect)
            return
        axis = "x" if (can_x and (not can_y or rect.w >= rect.h)) else "y"
        if axis == "x":
            cut = rng.randint(min_room, rect.w - min_room - 1)
            gap = rect.x0 + cut
            splits.append(("x", gap, rect.y0, rect.y1))
            carve(Rect(rect.x0, rect.y0, gap, rect.y1), depth + 1)
            carve(Rect(gap + 1, rect.y0, rect.x1, rect.y1), depth + 1)
        else:
            cut = rng.randint(min_room, rect.h - min_room - 1)
            gap = rect.y0 + cut
            splits.append(("y", gap, rect.x0, rect.x1))
            carve(Rect(rect.x0, rect.y0, rect.x1, gap), depth + 1)
            carve(Rect(rect.x0, gap + 1, rect.x1, rect.y1), depth + 1)

    carve(Rect(1, 1, size - 1, size - 1), 0)

    # Doors are placed after every room exists, so a deeper wall band can never
    # swallow a door opening.  If a band has no valid position the map is retried.
    doors: list[Door] = []
    rng.shuffle(splits)
    for axis, gap, lo, hi in splits:
        if axis == "x":
            cand = [y for y in range(lo + 1, hi - 1)
                    if grid[y][gap] == -1 and grid[y][gap - 1] >= 0 and grid[y][gap + 1] >= 0
                    and not _near_door(grid, gap, y)]
            if not cand:
                return None
            y = rng.choice(cand)
            grid[y][gap] = _door_tile(len(doors))
            doors.append(Door(gap, y, "v", index=len(doors)))
        else:
            cand = [x for x in range(lo + 1, hi - 1)
                    if grid[gap][x] == -1 and grid[gap - 1][x] >= 0 and grid[gap + 1][x] >= 0
                    and not _near_door(grid, x, gap)]
            if not cand:
                return None
            x = rng.choice(cand)
            grid[gap][x] = _door_tile(len(doors))
            doors.append(Door(x, gap, "h", index=len(doors)))
    return grid, rooms, doors


def _near_door(grid: list[list[int]], x: int, y: int) -> bool:
    h, w = len(grid), len(grid[0])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and _is_door(grid[ny][nx]):
            return True
    return False


def _resolve_door_sides(grid: list[list[int]], doors: list[Door]) -> None:
    h, w = len(grid), len(grid[0])

    def walk(x: int, y: int, dx: int, dy: int) -> int:
        while 0 <= x < w and 0 <= y < h and _is_door(grid[y][x]):
            x += dx
            y += dy
        return grid[y][x] if 0 <= x < w and 0 <= y < h else -1

    for d in doors:
        if d.axis == "v":
            a = walk(d.x - 1, d.y, -1, 0)
            b = walk(d.x + 1, d.y, 1, 0)
        else:
            a = walk(d.x, d.y - 1, 0, -1)
            b = walk(d.x, d.y + 1, 0, 1)
        if not (a >= 0 and b >= 0):
            raise ValueError(f"door {d.index} at ({d.x},{d.y}) does not connect two rooms ({a},{b})")
        d.a, d.b = a, b


def _adjacency(n_rooms: int, doors: list[Door]) -> dict[int, set[int]]:
    adj = {i: set() for i in range(n_rooms)}
    for d in doors:
        adj[d.a].add(d.b)
        adj[d.b].add(d.a)
    return adj


def _bfs(adj, start: int) -> dict[int, int]:
    dist = {start: 0}
    q = [start]
    for u in q:
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _path(adj, s: int, t: int) -> list[int]:
    prev = {s: None}
    q = [s]
    for u in q:
        if u == t:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    out, u = [], t
    while u is not None:
        out.append(u)
        u = prev.get(u)
    return out[::-1]


def _reachable(adj, start: int, blocked_edges: set[frozenset] | None = None) -> set[int]:
    blocked_edges = blocked_edges or set()
    seen = {start}
    q = [start]
    for u in q:
        for v in adj[u]:
            if frozenset((u, v)) in blocked_edges or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return seen


def _add_loops(grid, doors: list[Door], rng: random.Random, n_loops: int) -> None:
    h, w = len(grid), len(grid[0])
    cands = []
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if grid[y][x] != -1:
                continue
            if grid[y][x - 1] >= 0 and grid[y][x + 1] >= 0 and grid[y - 1][x] == -1 and grid[y + 1][x] == -1:
                cands.append((x, y, "v"))
            elif grid[y - 1][x] >= 0 and grid[y + 1][x] >= 0 and grid[y][x - 1] == -1 and grid[y][x + 1] == -1:
                cands.append((x, y, "h"))
    rng.shuffle(cands)
    added = 0
    for x, y, axis in cands:
        if added >= n_loops:
            break
        a = grid[y][x - 1] if axis == "v" else grid[y - 1][x]
        b = grid[y][x + 1] if axis == "v" else grid[y + 1][x]
        if a == b or _near_door(grid, x, y):
            continue
        grid[y][x] = _door_tile(len(doors))
        doors.append(Door(x, y, axis, a, b, index=len(doors)))
        added += 1


def _clear_line(grid, rooms, rid: int, horizontal: bool, rng: random.Random):
    """A clear walk-over line spanning room ``rid`` (no solid tiles on the line)."""
    r = rooms[rid]
    if horizontal:
        rows = list(range(r.y0, r.y1))
        rng.shuffle(rows)
        for y in rows:
            if all(grid[y][x] >= 0 for x in range(r.x0, r.x1)):
                return (r.x0 * TILE + TILE / 2, y * TILE + TILE / 2,
                        r.x1 * TILE - TILE / 2, y * TILE + TILE / 2)
    else:
        cols = list(range(r.x0, r.x1))
        rng.shuffle(cols)
        for x in cols:
            if all(grid[y][x] >= 0 for y in range(r.y0, r.y1)):
                return (x * TILE + TILE / 2, r.y0 * TILE + TILE / 2,
                        x * TILE + TILE / 2, r.y1 * TILE - TILE / 2)
    c = r.center()
    return (c[0] * TILE + TILE / 2, r.y0 * TILE + TILE / 2,
            c[0] * TILE + TILE / 2, r.y1 * TILE - TILE / 2)


# --------------------------------------------------------------------------- tiers
def generate_map(seed: int, tier: int = 2, out_dir: str | Path | None = None,
                 no_doors: bool = False, no_monsters: bool = False,
                 size_override: int | None = None, depth_override: int | None = None) -> MapSpec:
    rng = random.Random(seed * 1000003 + tier * 7919)
    if tier == 1:
        spec = _arena(seed, rng)
    else:
        spec = _rooms(seed, tier, rng, out_dir, no_doors=no_doors, no_monsters=no_monsters,
                      size_override=size_override, depth_override=depth_override)
    spec.lumps = _build_lumps(spec)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"t{tier}_s{seed:05d}.wad"
        _write_wad(path, spec.lumps, spec.map_name)
        spec.path = path
    return spec


def _arena(seed: int, rng: random.Random) -> MapSpec:
    size = rng.randint(20, 26)
    grid = [[-1] * size for _ in range(size)]
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            grid[y][x] = 0
    for _ in range(rng.randint(5, 10)):
        pw, ph = rng.randint(1, 2), rng.randint(1, 2)
        px, py = rng.randint(4, size - 5 - pw), rng.randint(4, size - 5 - ph)
        for y in range(py, py + ph):
            for x in range(px, px + pw):
                grid[y][x] = -1
    # Start in a corner; exit line across the opposite end.
    sx, sy = rng.choice(((2, 2), (size - 3, 2), (2, size - 3), (size - 3, size - 3)))
    horizontal = abs(sy - size / 2) < abs(sx - size / 2)
    exit_y = size - 3 if sy < size / 2 else 2
    exit_x = size - 3 if sx < size / 2 else 2
    if horizontal:
        line = (1.5 * TILE, exit_y * TILE + 32, (size - 1.5) * TILE, exit_y * TILE + 32)
    else:
        line = (exit_x * TILE + 32, 1.5 * TILE, exit_x * TILE + 32, (size - 1.5) * TILE)
    spec = MapSpec(seed=seed, tier=1, kind="arena", width=size * TILE, height=size * TILE,
                   grid=grid, rooms=[Rect(1, 1, size - 1, size - 1)], doors=[],
                   start=(sx * TILE + 32.0, sy * TILE + 32.0), start_angle=0,
                   exit_line=line, exit_room=0)
    free = [(x, y) for y in range(2, size - 2) for x in range(2, size - 2)
            if grid[y][x] == 0 and math.hypot(x * TILE + 32 - spec.start[0], y * TILE + 32 - spec.start[1]) > 200]
    rng.shuffle(free)
    spec.things.append((T_PLAYER1, *spec.start, 0))
    monsters = [T_ZOMBIE, T_ZOMBIE, T_SHOTGUNGUY, T_IMP, T_DEMON]
    for _ in range(min(rng.randint(4, 8), len(free))):
        x, y = free.pop()
        spec.things.append((rng.choice(monsters), x * TILE + 32.0, y * TILE + 32.0,
                            rng.choice((0, 90, 180, 270))))
    for _ in range(min(rng.randint(3, 6), len(free))):
        x, y = free.pop()
        spec.things.append((rng.choice((T_STIMPACK, T_CLIP, T_SHELLS, T_GREEN_ARMOR, T_SHOTGUN)),
                            x * TILE + 32.0, y * TILE + 32.0, 0))
    return spec


def _rooms(seed: int, tier: int, rng: random.Random, out_dir=None, no_doors: bool = False,
           no_monsters: bool = False, size_override: int | None = None,
           depth_override: int | None = None) -> MapSpec:
    size = size_override or {2: 30, 3: 34, 4: 40}[tier]
    depth = depth_override or {2: 3, 3: 4, 4: 5}[tier]
    layout = None
    for attempt in range(24):
        layout = _split_layout(rng if attempt == 0 else random.Random(rng.random()), size, depth, min_room=4)
        if layout is not None:
            break
    if layout is None:
        raise RuntimeError(f"could not build a connected layout (seed={seed}, tier={tier})")
    grid, rooms, doors = layout
    if len(rooms) < 4:
        return generate_map(seed + 1, tier, out_dir, no_doors=no_doors, no_monsters=no_monsters,
                            size_override=size_override, depth_override=depth_override)
    _resolve_door_sides(grid, doors)
    if no_doors:
        for d in doors:
            grid[d.y][d.x] = d.a
        doors = []
    adj = _adjacency(len(rooms), doors)

    start_room = rng.randrange(len(rooms))
    if no_doors:
        c0 = rooms[start_room].center()
        exit_room = max(range(len(rooms)), key=lambda r: (rooms[r].center()[0] - c0[0]) ** 2 + (rooms[r].center()[1] - c0[1]) ** 2)
        dist = {}
    else:
        dist = _bfs(adj, start_room)
        exit_room = max(range(len(rooms)), key=lambda r: (dist.get(r, -1), rng.random()))

    keys: list[tuple[int, int]] = []
    if tier >= 3 and not no_doors:
        path = _path(adj, start_room, exit_room)
        candidates = [d for d in doors if {d.a, d.b} <= set(path)]
        rng.shuffle(candidates)
        for d in candidates:
            reach = _reachable(adj, start_room, {frozenset((d.a, d.b))})
            options = [r for r in reach if r != start_room]
            if options:
                d.lock = LOCK_BLUE
                key_room = max(options, key=lambda r: (dist.get(r, 0), rng.random()))
                keys.append((key_room, LOCK_BLUE))
                break
        if tier == 4 and keys:
            # A red lock on a door behind the blue one, with the red key on the blue side.
            blue = next(d for d in doors if d.lock == LOCK_BLUE)
            reach = _reachable(adj, start_room, {frozenset((blue.a, blue.b))})
            later = [d for d in doors if d is not blue and d.lock == 0 and {d.a, d.b} <= reach]
            if later:
                d2 = rng.choice(later)
                d2.lock = LOCK_RED
                reach2 = _reachable(adj, start_room, {frozenset((d2.a, d2.b))})
                opts = [r for r in reach2 if r not in (start_room,) and r not in [k[0] for k in keys]]
                if opts:
                    keys.append((max(opts, key=lambda r: (dist.get(r, 0), rng.random())), LOCK_RED))
                else:
                    d2.lock = 0
    if tier in (2, 4) and not no_doors:
        _add_loops(grid, doors, rng, 2 if tier == 2 else 3)
        adj = _adjacency(len(rooms), doors)

    for i, d in enumerate(doors):
        d.sector = len(rooms) + i

    horizontal = rng.random() < 0.5
    line = _clear_line(grid, rooms, exit_room, horizontal, rng)
    spec = MapSpec(seed=seed, tier=tier, kind="rooms", width=size * TILE, height=size * TILE,
                   grid=grid, rooms=rooms, doors=doors,
                   start=(rooms[start_room].center()[0] * TILE + 32.0,
                          rooms[start_room].center()[1] * TILE + 32.0),
                   start_angle=rng.choice((0, 90, 180, 270)),
                   exit_line=line, exit_room=exit_room, keys=keys)
    spec.adj = {i: sorted({(d.b if d.a == i else d.a, d.sector) for d in doors if i in (d.a, d.b)})
                for i in range(len(rooms))}
    _populate(spec, rng, start_room, exit_room, no_monsters=no_monsters)
    return spec


def _populate(spec: MapSpec, rng: random.Random, start_room: int, exit_room: int, no_monsters: bool = False) -> None:
    spec.things.append((T_PLAYER1, *spec.start, spec.start_angle))
    key_rooms = {r: lock for r, lock in spec.keys}
    monster_budget = 0 if no_monsters else {1: 8, 2: 6, 3: 10, 4: 16}[spec.tier]
    for rid, rect in enumerate(spec.rooms):
        cells = [(x, y) for y in range(rect.y0, rect.y1) for x in range(rect.x0, rect.x1)
                 if spec.grid[y][x] == rid]
        if not cells:
            continue

        def clearance(x, y):
            return all(0 <= y + dy < len(spec.grid) and 0 <= x + dx < len(spec.grid[0])
                       and spec.grid[y + dy][x + dx] == rid
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1))

        spots = [(x, y) for x, y in cells if clearance(x, y)]
        if not spots:
            spots = list(cells)
        rng.shuffle(spots)

        def place(t, angle=0, jitter=10.0):
            x, y = spots.pop()
            spec.things.append((t, x * TILE + 32.0 + rng.uniform(-jitter, jitter),
                                y * TILE + 32.0 + rng.uniform(-jitter, jitter), angle))

        if rid in key_rooms and spots:
            place(KEY_THING[key_rooms[rid]])
        if rid == exit_room:
            continue
        if rid == start_room:
            n_mon = {1: 0, 2: 1, 3: 1, 4: 2}[spec.tier]
        else:
            n_mon = {1: 1, 2: 2, 3: 2, 4: 3}[spec.tier]
        monsters = {1: [T_IMP], 2: [T_ZOMBIE, T_SHOTGUNGUY, T_IMP, T_DEMON],
                    3: [T_ZOMBIE, T_IMP, T_DEMON, T_CACODEMON],
                    4: [T_IMP, T_DEMON, T_CACODEMON, T_SHOTGUNGUY]}[spec.tier]
        for _ in range(min(n_mon, len(spots), monster_budget)):
            place(rng.choice(monsters), rng.choice((0, 90, 180, 270)))
            monster_budget -= 1
        for _ in range(min(rng.randint(1, 3), len(spots))):
            place(rng.choice((T_STIMPACK, T_CLIP, T_SHELLS, T_GREEN_ARMOR, T_SHOTGUN)), 0)


# --------------------------------------------------------------------------- binary writer
def _build_lumps(spec: MapSpec) -> dict[str, bytes]:
    """Emit a node-less Doom binary map (ZDoom builds BSP nodes at load time)."""
    grid = spec.grid
    h, w = len(grid), len(grid[0])
    verts: list[tuple[int, int]] = []
    vid: dict[tuple[int, int], int] = {}

    def V(x: int, y: int) -> int:
        key = (x, y)
        if key not in vid:
            vid[key] = len(verts)
            verts.append((int(x * TILE), int(y * TILE)))
        return vid[key]

    def Vraw(x: float, y: float) -> int:
        verts.append((int(round(x)), int(round(y))))
        return len(verts) - 1

    sectors: list[tuple[int, int, str, str, int]] = []
    room_wall: dict[int, str] = {}
    for i, r in enumerate(spec.rooms):
        rr = random.Random(spec.seed * 31 + i)
        room_wall[i] = rr.choice(WALLS)
        sectors.append((0, rr.choice((160, 176, 192, 224)), rr.choice(FLOORS),
                        rr.choice(CEILINGS), rr.choice((128, 160, 192, 224))))
    for d in spec.doors:
        sectors.append((0, 0, "FLOOR0_1", "CEIL1_1", 160))

    sides: list[tuple[int, str, str]] = []
    lines: list[tuple[int, int, int, int, int, int, int]] = []
    door_by_tile = {_door_tile(d.index): d for d in spec.doors}

    def add_side(sec: int, middle: str, upper: str = "-") -> int:
        sides.append((sec, middle, upper))
        return len(sides) - 1

    def add_line(a: int, b: int, front: int, back: int | None, special: int = 0,
                 blocking: bool = False, front_mid: str = "-", back_mid: str = "-",
                 front_up: str = "-", back_up: str = "-") -> None:
        if back is None:
            f = add_side(front, front_mid, front_up)
            lines.append((a, b, 1 if blocking else 0, special, 0, f, 0xFFFF))
        else:
            f = add_side(front, front_mid, front_up)
            bk = add_side(back, back_mid, back_up)
            lines.append((a, b, 1 if blocking else 0, special, 0, f, bk))

    def at(x: int, y: int) -> int:
        return grid[y][x] if 0 <= x < w and 0 <= y < h else -1

    def sector_for(v: int) -> int:
        return v if v >= 0 else door_by_tile[v].sector

    def wall_tex(v: int) -> str:
        return room_wall.get(v, DOOR_TRACK) if v >= 0 else DOOR_TRACK

    door_mid = DOOR_TEXTURES[spec.seed % len(DOOR_TEXTURES)]

    def tex_for(front: int, back: int) -> tuple[str, str]:
        for t in (front, back):
            if _is_door(t):
                lock = door_by_tile[t].lock
                tex = spec.door_texture_override or LOCK_TEXTURE.get(lock, door_mid)
                return tex, tex
        return "-", "-"

    for x in range(w + 1):
        for y in range(h):
            L, R = at(x - 1, y), at(x, y)
            if L == R:
                continue
            if _is_door(L) and _is_door(R):
                continue
            if _is_door(R):
                a, b, front, back = (V(x, y + 1), V(x, y), L, R) if L >= 0 else (V(x, y), V(x, y + 1), R, -1)
            elif _is_door(L):
                a, b, front, back = (V(x, y), V(x, y + 1), R, L) if R >= 0 else (V(x, y + 1), V(x, y), L, -1)
            elif R >= 0 and L >= 0:
                a, b, front, back = V(x, y), V(x, y + 1), R, L
            elif R >= 0:
                a, b, front, back = V(x, y), V(x, y + 1), R, -1
            else:
                a, b, front, back = V(x, y + 1), V(x, y), L, -1
            special = door_by_tile[back].special if front >= 0 and _is_door(back) else 0
            mid_f, mid_b = tex_for(front, back)
            up_f = wall_tex(front) if back != -1 and not _is_door(front) and not _is_door(back) else "-"
            up_b = up_f
            add_line(a, b, sector_for(front), sector_for(back) if back != -1 else None,
                     special=special, blocking=back == -1,
                     front_mid=wall_tex(front) if back == -1 else mid_f,
                     back_mid=mid_b if back != -1 else "-",
                     front_up=up_f, back_up=up_b)

    for y in range(h + 1):
        for x in range(w):
            U, D = at(x, y - 1), at(x, y)
            if U == D:
                continue
            if _is_door(U) and _is_door(D):
                continue
            if _is_door(D):
                a, b, front, back = (V(x, y), V(x + 1, y), U, D) if U >= 0 else (V(x + 1, y), V(x, y), D, -1)
            elif _is_door(U):
                a, b, front, back = (V(x + 1, y), V(x, y), D, U) if D >= 0 else (V(x, y), V(x + 1, y), U, -1)
            elif D >= 0 and U >= 0:
                a, b, front, back = V(x + 1, y), V(x, y), D, U
            elif D >= 0:
                a, b, front, back = V(x + 1, y), V(x, y), D, -1
            else:
                a, b, front, back = V(x, y), V(x + 1, y), U, -1
            special = door_by_tile[back].special if front >= 0 and _is_door(back) else 0
            mid_f, mid_b = tex_for(front, back)
            up_f = wall_tex(front) if back != -1 and not _is_door(front) and not _is_door(back) else "-"
            add_line(a, b, sector_for(front), sector_for(back) if back != -1 else None,
                     special=special, blocking=back == -1,
                     front_mid=wall_tex(front) if back == -1 else mid_f,
                     back_mid=mid_b if back != -1 else "-",
                     front_up=up_f, back_up=up_f)

    # Walk-over exit line: same sector on both sides, invisible.
    x1, y1, x2, y2 = spec.exit_line
    a = (int(round(x1 / TILE)), int(round(y1 / TILE)))
    b = (int(round(x2 / TILE)), int(round(y2 / TILE)))
    add_line(V(*a), V(*b), spec.exit_room, spec.exit_room, special=S_EXIT_WALK, blocking=False)

    # A visible exit sign on a small freestanding wall just behind the walk-over
    # line.  The RGB encoder learns "this bright sign means cross the line"; the
    # structured observations stay unchanged.
    dx, dy = x2 - x1, y2 - y1
    ln = math.hypot(dx, dy) or 1.0
    ux, uy = dx / ln, dy / ln
    nx, ny = dy / ln, -dx / ln
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    sign_tex = spec.exit_sign_texture or EXIT_TEXTURES[spec.seed % len(EXIT_TEXTURES)]
    for side in (1.0, -1.0):
        px, py = mx + nx * 56 * side, my + ny * 56 * side
        ix, iy = int(px // TILE), int(py // TILE)
        if at(ix, iy) == spec.exit_room:
            va = Vraw(px - ux * 32, py - uy * 32)
            vb = Vraw(px + ux * 32, py + uy * 32)
            add_line(va, vb, spec.exit_room, None, special=0, blocking=True,
                     front_mid=sign_tex)
            break

    lumps: dict[str, bytes] = {}
    lumps["THINGS"] = b"".join(struct.pack("<hhhhh", int(x), int(y), int(ang) % 360, int(t), 7)
                               for t, x, y, ang in spec.things)
    lumps["LINEDEFS"] = b"".join(struct.pack("<HHHHhHH", *ln) for ln in lines)
    lumps["SIDEDEFS"] = b"".join(struct.pack("<hh8s8s8sh", 0, 0,
                                             u.encode("ascii")[:8].ljust(8, b"\0"),
                                             b"-".ljust(8, b"\0"),
                                             m.encode("ascii")[:8].ljust(8, b"\0"), s)
                                for s, m, u in sides)
    lumps["VERTEXES"] = b"".join(struct.pack("<hh", x, y) for x, y in verts)
    lumps["SECTORS"] = b"".join(struct.pack("<hh8s8shhh", f, c,
                                            ft.encode("ascii")[:8].ljust(8, b"\0"),
                                            ct.encode("ascii")[:8].ljust(8, b"\0"),
                                            light, 0, 0)
                               for f, c, ft, ct, light in sectors)
    return lumps


def _write_wad(path: Path, lumps: dict[str, bytes], map_name: str) -> None:
    order = ["THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SECTORS"]
    all_lumps = [(map_name, b"")] + [(k, lumps[k]) for k in order if k in lumps]
    body = b""
    directory = b""
    offset = 12
    for name, data in all_lumps:
        directory += struct.pack("<ii8s", offset, len(data), name.upper().encode("ascii").ljust(8, b"\0"))
        body += data
        offset += len(data)
    path.write_bytes(b"PWAD" + struct.pack("<ii", len(all_lumps), offset) + body + directory)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", type=int, default=2, choices=(1, 2, 3, 4))
    ap.add_argument("--out", default="maps/generated")
    ap.add_argument("--count", type=int, default=1)
    args = ap.parse_args()
    for i in range(args.count):
        spec = generate_map(args.seed + i, args.tier, args.out)
        print(f"{spec.path}  rooms={len(spec.rooms)} doors={len(spec.doors)} "
              f"exit_room={spec.exit_room} keys={spec.keys}")


if __name__ == "__main__":
    main()
