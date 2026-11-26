"""Prepare real Freedoom 2 maps as training maps.

Two variants per map:
  maps/freedoom/<MAP>.wad       original geometry, all things (fine-tuning / eval)
  maps/freedoom_open/<MAP>.wad  door ceilings raised (navigation cloning)
  maps/freedoom_nav/<MAP>.wad   doors raised + monsters removed (pure navigation)

Doors are raised by editing the SECTORS lump of each door sector (identified by
the official mapinfo parser), which is allowed training-time map surgery.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import vizdoom

from doom_env.mapinfo import load_mapinfo
from doom_env.wad import Lump, list_maps, map_lumps, read_wad, write_wad

MONSTER_TYPES = {3004, 9, 3001, 3002, 3005, 3003, 3006, 58, 68, 69, 71, 64, 66, 67, 65, 7}
OPEN_HEIGHT = 128


def _open_doors(sectors: bytes, door_ids: set[int]) -> bytes:
    buf = bytearray(sectors)
    n = len(buf) // 26
    for sid in door_ids:
        if 0 <= sid < n:
            off = sid * 26
            floor = struct.unpack_from("<h", buf, off)[0]
            struct.pack_into("<h", buf, off + 2, max(OPEN_HEIGHT, floor + OPEN_HEIGHT))
    return bytes(buf)


def _strip_monsters(things: bytes) -> bytes:
    out = bytearray()
    for i in range(len(things) // 10):
        rec = things[i * 10:(i + 1) * 10]
        t = struct.unpack_from("<h", rec, 6)[0]
        if t not in MONSTER_TYPES:
            out += rec
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="maps")
    ap.add_argument("--max-sectors", type=int, default=2048)
    args = ap.parse_args()

    iwad = os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")
    lumps = read_wad(iwad)
    root = Path(args.out)
    for sub in ("freedoom", "freedoom_open", "freedoom_nav"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    for name in list_maps(lumps):
        if not name.startswith("MAP"):
            continue
        ml = map_lumps(lumps, name)
        sec = struct.unpack_from("<h", ml["SECTORS"], 2)[0] if ml.get("SECTORS") else 0
        n_sectors = len(ml.get("SECTORS", b"")) // 26
        if n_sectors > args.max_sectors:
            continue
        info = load_mapinfo(iwad, name)
        door_ids = set(info.door_sectors)
        base = [Lump(name, b"")] + [Lump(k, v) for k, v in ml.items()]
        write_wad(root / "freedoom" / f"{name}.wad", base)

        ml_open = dict(ml)
        ml_open["SECTORS"] = _open_doors(ml["SECTORS"], door_ids)
        write_wad(root / "freedoom_open" / f"{name}.wad",
                  [Lump(name, b"")] + [Lump(k, v) for k, v in ml_open.items()])

        ml_nav = dict(ml_open)
        ml_nav["THINGS"] = _strip_monsters(ml_open["THINGS"])
        write_wad(root / "freedoom_nav" / f"{name}.wad",
                  [Lump(name, b"")] + [Lump(k, v) for k, v in ml_nav.items()])
        manifest.append({"map": name, "sectors": n_sectors, "doors": len(door_ids)})

    (root / "freedoom_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} maps to {root}/freedoom*")


if __name__ == "__main__":
    main()
