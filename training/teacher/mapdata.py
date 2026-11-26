"""Static map data used by the training-time expert.

The evaluation agent never imports this module: it only sees the official
observation dictionary.  The expert is allowed privileged access during
training (see docs/RULES.md: "any training-time scaffolding").

Parses Doom binary maps and UDMF ``TEXTMAP`` maps into a common form:
linedefs with front/back sectors and specials, sectors, things and exit lines.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from doom_env.mapinfo import MapInfo, load_mapinfo, parse_udmf
from doom_env.wad import map_lumps, read_wad

# Exit line specials in the Doom/Boom namespace.
EXIT_SPECIALS = {11, 51, 52, 124, 84}
SWITCH_EXITS = {11, 51}
WALK_EXITS = {52, 124, 84}

# Key thing types -> lock number (mapinfo convention).
KEY_THINGS = {5: 2, 6: 3, 13: 1, 40: 2, 39: 3, 38: 1}
MONSTER_THINGS = {3004, 9, 3001, 3002, 3005, 3003, 3006, 58, 68, 69, 71, 64, 66, 67, 65, 7}


@dataclass
class Things:
    x: np.ndarray
    y: np.ndarray
    type: np.ndarray
    angle: np.ndarray


@dataclass
class MapData:
    name: str
    vertexes: np.ndarray            # (V,2)
    l1: np.ndarray                  # (L,) vertex indices
    l2: np.ndarray
    front: np.ndarray               # (L,) sector index or -1
    back: np.ndarray
    special: np.ndarray
    blocking: np.ndarray            # (L,) bool
    floor: np.ndarray               # (S,)
    ceil: np.ndarray
    things: Things
    info: MapInfo
    bounds: tuple[float, float, float, float]

    # ---- derived helpers
    def line_points(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertexes[self.l1], self.vertexes[self.l2]

    def door_centres(self) -> dict[int, tuple[float, float, int]]:
        """sector id -> (x, y, lock)."""
        out = {}
        p1, p2 = self.line_points()
        for sec, lock in self.info.door_sectors.items():
            mask = (self.front == sec) | (self.back == sec)
            if not mask.any():
                continue
            pts = np.concatenate([p1[mask], p2[mask]], axis=0)
            out[sec] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()), int(lock))
        return out

    def exit_lines(self) -> list[tuple[float, float, float, float, int]]:
        """(x1, y1, x2, y2, special) for every exit line."""
        p1, p2 = self.line_points()
        out = []
        for i in np.flatnonzero(np.isin(self.special, list(EXIT_SPECIALS))):
            out.append((float(p1[i, 0]), float(p1[i, 1]), float(p2[i, 0]), float(p2[i, 1]), int(self.special[i])))
        return out

    def key_positions(self) -> list[tuple[float, float, int]]:
        out = []
        for x, y, t in zip(self.things.x, self.things.y, self.things.type):
            lock = KEY_THINGS.get(int(t))
            if lock:
                out.append((float(x), float(y), lock))
        return out

    def monster_positions(self) -> np.ndarray:
        m = np.isin(self.things.type, list(MONSTER_THINGS))
        return np.stack([self.things.x[m], self.things.y[m]], axis=1) if m.any() else np.zeros((0, 2))


def load_mapdata(wad: str | Path, map_name: str = "MAP01") -> MapData:
    lumps = map_lumps(read_wad(wad), map_name)
    info = load_mapinfo(wad, map_name)
    if "TEXTMAP" in lumps:
        md = _from_udmf(lumps["TEXTMAP"].decode("utf-8", "replace"), map_name, info)
    else:
        md = _from_doom(lumps, map_name, info)
    p1, p2 = md.line_points()
    allp = np.concatenate([p1, p2], axis=0) if len(p1) else np.zeros((1, 2))
    md.bounds = (float(allp[:, 0].min()), float(allp[:, 1].min()),
                 float(allp[:, 0].max()), float(allp[:, 1].max()))
    return md


def _from_doom(lumps: dict[str, bytes], name: str, info: MapInfo) -> MapData:
    verts = np.frombuffer(lumps.get("VERTEXES", b""), dtype="<i2").reshape(-1, 2).astype(np.float64)
    raw = lumps.get("LINEDEFS", b"")
    n = len(raw) // 14
    l1 = np.zeros(n, np.int64)
    l2 = np.zeros(n, np.int64)
    special = np.zeros(n, np.int64)
    blocking = np.zeros(n, bool)
    front = np.full(n, -1, np.int64)
    back = np.full(n, -1, np.int64)
    side_sector = np.frombuffer(lumps.get("SIDEDEFS", b""), dtype=np.uint8)
    sd = np.frombuffer(lumps.get("SIDEDEFS", b""), dtype=np.int16).reshape(-1, 15)
    for i in range(n):
        v1, v2, flags, sp, tag, f, b = struct.unpack_from("<HHHHhHH", raw, i * 14)
        l1[i], l2[i], special[i] = v1, v2, sp
        blocking[i] = bool(flags & 1)
        front[i] = int(sd[f, 14]) if f != 0xFFFF and f < len(sd) else -1
        back[i] = int(sd[b, 14]) if b != 0xFFFF and b < len(sd) else -1
    sec = np.frombuffer(lumps.get("SECTORS", b""), dtype=np.int16).reshape(-1, 13)
    floor = sec[:, 0].astype(np.float64) if len(sec) else np.zeros(0)
    ceil = sec[:, 1].astype(np.float64) if len(sec) else np.zeros(0)
    th = np.frombuffer(lumps.get("THINGS", b""), dtype=np.int16).reshape(-1, 5)
    things = Things(th[:, 0].astype(np.float64), th[:, 1].astype(np.float64),
                    th[:, 3].astype(np.int64), th[:, 2].astype(np.int64))
    return MapData(name, verts, l1, l2, front, back, special, blocking, floor, ceil, things, info, (0, 0, 0, 0))


def _from_udmf(text: str, name: str, info: MapInfo) -> MapData:
    b = parse_udmf(text)
    verts = np.array([[float(v.get("x", 0.0)), float(v.get("y", 0.0))] for v in b.get("vertex", [])],
                     dtype=np.float64).reshape(-1, 2)
    sides = b.get("sidedef", [])
    lines = b.get("linedef", [])
    n = len(lines)
    l1 = np.zeros(n, np.int64)
    l2 = np.zeros(n, np.int64)
    special = np.zeros(n, np.int64)
    blocking = np.zeros(n, bool)
    front = np.full(n, -1, np.int64)
    back = np.full(n, -1, np.int64)

    def sec_of(idx) -> int:
        if isinstance(idx, int) and 0 <= idx < len(sides):
            return int(sides[idx].get("sector", -1))
        return -1

    for i, ln in enumerate(lines):
        l1[i] = int(ln.get("v1", 0))
        l2[i] = int(ln.get("v2", 0))
        special[i] = int(ln.get("special", 0) or 0)
        blocking[i] = bool(ln.get("blocking", False))
        front[i] = sec_of(ln.get("sidefront", -1))
        back[i] = sec_of(ln.get("sideback", -1))
    sectors = b.get("sector", [])
    floor = np.array([float(s.get("heightfloor", 0.0)) for s in sectors], dtype=np.float64)
    ceil = np.array([float(s.get("heightceiling", 0.0)) for s in sectors], dtype=np.float64)
    th = b.get("thing", [])
    things = Things(np.array([float(t.get("x", 0.0)) for t in th]),
                    np.array([float(t.get("y", 0.0)) for t in th]),
                    np.array([int(t.get("type", 0)) for t in th], dtype=np.int64),
                    np.array([int(t.get("angle", 0)) for t in th], dtype=np.int64))
    return MapData(name, verts, l1, l2, front, back, special, blocking, floor, ceil, things, info, (0, 0, 0, 0))
