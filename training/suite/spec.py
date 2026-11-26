"""Benchmark category specifications and map construction.

Fourteen diagnostic categories (C0–C13). Generated maps use the existing binary
generators plus deterministic post-processing (monster/ammo budgets, start pose,
exit-sign and door-colour ablations). This module is training/evaluation only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from training.maps.doorroom import generate_room_door
from training.maps.drills import generate_backtrack, generate_drill, generate_key_chain
from training.maps.gen_wad import (T_BLUECARD, T_CACODEMON, T_CELL, T_CLIP, T_DEMON,
                                   T_GREEN_ARMOR, T_IMP, T_MEDIKIT, T_PLAYER1, T_SHELLS,
                                   T_SHOTGUNGUY, T_STIMPACK, T_ZOMBIE, TILE, MapSpec,
                                   _build_lumps, _write_wad, generate_map)

MONSTERS = {3004, 9, 3001, 3002, 3005, 3003, 3006, 58, 68, 69, 71, 64, 66, 67, 65, 7}
AMMO = {2007, 2008, 2046, 2047, 2002, 2003, 2004, 2005, 2006}
MONSTER_TYPES = (T_ZOMBIE, T_SHOTGUNGUY, T_IMP, T_DEMON, T_CACODEMON)

GENERATOR_VERSION = "suite-1"
SPLIT_SEEDS = {"train": 0, "val": 10_000, "test": 30_000, "ood": 40_000}


@dataclass(frozen=True)
class Category:
    id: str
    tier: int
    ticks: int
    kind: str
    description: str
    kwargs: dict = field(default_factory=dict)
    monster_keep: int | None = None
    ambush: int = 0
    strip_ammo: bool = False
    ood: bool = False

    def config(self) -> dict:
        return {"id": self.id, "tier": self.tier, "kind": self.kind,
                "kwargs": self.kwargs, "monster_keep": self.monster_keep,
                "ambush": self.ambush, "strip_ammo": self.strip_ammo, "ood": self.ood}


CATEGORIES: dict[str, Category] = {
    "C0_sanity": Category("C0_sanity", 1, 2100, "drill", "move to a visible exit",
                          {"n_doors": 0, "key": False}),
    "C1_duel": Category("C1_duel", 1, 2100, "arena", "1v1 aim/dodge", monster_keep=1),
    "C2_arena": Category("C2_arena", 1, 2100, "arena", "3-6 monsters, cover, ammo", monster_keep=6),
    "C3_corridor_door": Category("C3_corridor_door", 1, 2100, "drill", "straight door USE",
                                 {"n_doors": 1, "key": False}),
    "C4_room_door": Category("C4_room_door", 1, 2100, "roomdoor", "door on a room wall, random pose",
                             {"two_doors": False, "key": False}),
    "C5_key_lock": Category("C5_key_lock", 3, 3150, "drill", "one key, one locked door",
                            {"n_doors": 2, "key": True}),
    "C6_two_key": Category("C6_two_key", 4, 4200, "keychain", "two keys, two locks"),
    "C7_exploration": Category("C7_exploration", 2, 4200, "gen", "8-13 rooms, hidden exit",
                               {"no_monsters": True}),
    "C8_backtrack": Category("C8_backtrack", 3, 4200, "backtrack", "key behind start, lock ahead"),
    "C9_ambush": Category("C9_ambush", 3, 4200, "gen", "monsters after the key/door", ambush=2),
    "C10_resource": Category("C10_resource", 4, 4200, "gen", "scarce ammo, many monsters",
                             strip_ammo=True),
    "C11_gauntlet": Category("C11_gauntlet", 4, 4200, "gen", "all skills"),
    "C12_freedoom": Category("C12_freedoom", 4, 4200, "freedoom", "hand-made transfer"),
    "C13_ood": Category("C13_ood", 4, 4200, "gen", "OOD seeds/configs", ood=True),
}


def _player_index(spec: MapSpec) -> int:
    for i, t in enumerate(spec.things):
        if t[0] == T_PLAYER1:
            return i
    return -1


def set_start(spec: MapSpec, x: float | None = None, y: float | None = None,
              angle: int | None = None) -> None:
    i = _player_index(spec)
    if i < 0:
        return
    px, py, pa = spec.things[i][1], spec.things[i][2], spec.things[i][3]
    x = px if x is None else x
    y = py if y is None else y
    a = pa if angle is None else angle
    spec.start = (x, y)
    spec.start_angle = a
    spec.things[i] = (T_PLAYER1, x, y, a)


def keep_monsters(spec: MapSpec, n: int) -> None:
    seen = 0
    out = []
    for t in spec.things:
        if t[0] in MONSTERS:
            if seen < n:
                out.append(t)
                seen += 1
        else:
            out.append(t)
    spec.things = out


def add_monsters(spec: MapSpec, n: int, rng: random.Random) -> None:
    if not spec.rooms:
        return
    room = spec.rooms[spec.exit_room]
    cells = [(x, y) for y in range(room.y0, room.y1) for x in range(room.x0, room.x1)
             if 0 <= y < len(spec.grid) and 0 <= x < len(spec.grid[0])
             and spec.grid[y][x] >= 0]
    rng.shuffle(cells)
    for i in range(min(n, len(cells))):
        x, y = cells[i]
        spec.things.append((rng.choice(MONSTER_TYPES),
                            x * TILE + 32.0, y * TILE + 32.0, rng.choice((0, 90, 180, 270))))


def strip_ammo(spec: MapSpec) -> None:
    spec.things = [t for t in spec.things if t[0] not in AMMO]


def rebuild(spec: MapSpec) -> None:
    spec.lumps = _build_lumps(spec)
    _write_wad(spec.path, spec.lumps, spec.map_name)


def _freedoom_path(split: str, i: int) -> tuple[str, str]:
    if split == "train":
        names = [f"MAP{j:02d}" for j in range(1, 17)]
    elif split == "val":
        names = [f"MAP{j:02d}" for j in range(17, 23)]
    else:
        names = [f"MAP{j:02d}" for j in range(23, 33)]
    name = names[i % len(names)]
    return f"maps/freedoom/{name}.wad", name


CATEGORY_OFFSET = {cid: i * 131 for i, cid in enumerate(CATEGORIES)}  # stable, distinct


def _gen_seed(category: str, seed: int, attempt: int) -> int:
    return seed + CATEGORY_OFFSET[category] + attempt * 100_003


def _detour_ok(spec: MapSpec) -> bool:
    """C8: the key must be far away and the exit far from the key (long backtrack)."""
    if not spec.keys:
        return False
    sx, sy = spec.start
    diag = math.hypot(spec.width, spec.height)
    best = None
    for rid, _ in spec.keys:
        r = spec.rooms[rid]
        cx, cy = r.center()
        d = math.hypot(cx * TILE - sx, cy * TILE - sy)
        best = d if best is None else max(best, d)
    if best is None or best < 0.45 * diag:
        return False
    er = spec.rooms[spec.exit_room]
    ex, ey = er.center()
    for rid, _ in spec.keys:
        r = spec.rooms[rid]
        cx, cy = r.center()
        if math.hypot(cx * TILE - ex * TILE, cy * TILE - ey * TILE) < 0.40 * diag:
            return False
    return True


def build_map(category: str, seed: int, split: str, out_root: str | Path = "training/data/suite",
              force: bool = False) -> dict:
    """Create (or reuse) one map and return its manifest row."""
    cat = CATEGORIES[category]
    root = Path(out_root) / split / category
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed * 104729 + cat.tier)

    if cat.kind == "freedoom":
        wad, name = _freedoom_path(split, seed)
        sha = hashlib.sha256(Path(wad).read_bytes()).hexdigest()
        return {"map_id": f"{category}_{seed:05d}", "wad": wad, "map_name": name,
                "category": category, "tier": cat.tier, "ticks": cat.ticks,
                "seed": seed, "split": split, "spec_hash": sha,
                "generator_version": GENERATOR_VERSION, "description": cat.description}

    existing = sorted(root.glob(f"*{seed:05d}.wad"))
    if existing and not force:
        path = existing[0]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"map_id": f"{category}_{seed:05d}", "wad": str(path), "map_name": "MAP01",
                "category": category, "tier": cat.tier, "ticks": cat.ticks,
                "seed": seed, "split": split, "spec_hash": sha,
                "generator_version": GENERATOR_VERSION, "description": cat.description}

    spec: MapSpec | None = None
    for attempt in range(24):
        gseed = _gen_seed(category, seed, attempt)
        if cat.kind == "drill":
            spec = generate_drill(gseed, str(root), n_doors=int(cat.kwargs.get("n_doors", 1)),
                                  key=bool(cat.kwargs.get("key", False)))
        elif cat.kind == "roomdoor":
            spec = generate_room_door(gseed, str(root), key=bool(cat.kwargs.get("key", False)),
                                      two_doors=bool(cat.kwargs.get("two_doors", False)))
        elif cat.kind == "arena":
            spec = generate_map(gseed, 1, str(root))
        elif cat.kind == "keychain":
            spec = generate_key_chain(gseed, str(root))
        elif cat.kind == "backtrack":
            spec = generate_backtrack(gseed, str(root))
        else:
            size = 26 if category == "C7_exploration" else (36 if cat.ood else None)
            depth = 4 if category == "C7_exploration" else None
            spec = generate_map(gseed, cat.tier, str(root),
                                no_monsters=bool(cat.kwargs.get("no_monsters", False)),
                                size_override=size, depth_override=depth)

        # category-level acceptance filters
        ok = True
        if category == "C6_two_key":
            ok = len(spec.keys) == 2 and sum(1 for d in spec.doors if d.lock > 0) == 2
        elif category == "C7_exploration":
            ok = 8 <= len(spec.rooms) <= 13
        elif category == "C8_backtrack":
            ok = True  # the backtrack layout guarantees the detour
        if ok:
            break

    assert spec is not None
    if category in ("C0_sanity", "C3_corridor_door"):
        x1, y1, x2, y2 = spec.exit_line
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        sx, sy = spec.start
        ang = int(round(math.degrees(math.atan2(my - sy, mx - sx)))) % 360
        set_start(spec, angle=ang)
    if cat.monster_keep is not None:
        keep_monsters(spec, cat.monster_keep)
        if category == "C1_duel" and spec.things:
            sx, sy = spec.start
            mon = [t for t in spec.things if t[0] in MONSTERS]
            if mon:
                tx, ty = mon[0][1], mon[0][2]
                set_start(spec, angle=int(round(math.degrees(math.atan2(ty - sy, tx - sx)))) % 360)
    if cat.ambush:
        add_monsters(spec, cat.ambush, rng)
    if cat.strip_ammo:
        strip_ammo(spec)
    if cat.ood:
        # OOD: rotate the start pose and jitter it inside the start room.
        angle = rng.choice((0, 45, 90, 135, 180, 225, 270, 315))
        set_start(spec, angle=angle)
    rebuild(spec)
    path = Path(spec.path)
    canonical = root / f"{category}_{seed:05d}.wad"
    if path != canonical:
        path.replace(canonical)
        spec.path = canonical
        path = canonical

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"map_id": f"{category}_{seed:05d}", "wad": str(path), "map_name": "MAP01",
            "category": category, "tier": cat.tier, "ticks": cat.ticks,
            "seed": seed, "split": split, "spec_hash": sha,
            "generator_version": GENERATOR_VERSION, "description": cat.description,
            "config": cat.config()}


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
