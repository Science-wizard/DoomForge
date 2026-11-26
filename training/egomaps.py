"""Atlas-E map utilities: bit-packing and offline regeneration for demo shards.

The ego map is 16 binary channels x 32 x 32.  Stored and batched as one uint16
per cell (channel c -> bit c), 2 KB/step instead of 16 KB, and unpacked on the
training device.

Regenerate ego maps for existing demonstration shards (replays the stored
observations and executed actions through ``submission.mapmem.EgoMap``):

    python -m training.egomaps --data training/data/demos_v2_nav,training/data/demos_v3_key
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from submission.mapmem import EGO_MAP_CH, EgoMap

_BITS = (1 << np.arange(EGO_MAP_CH, dtype=np.uint32)).astype(np.uint32)


def pack(crop: np.ndarray) -> np.ndarray:
    """(..., 16, 32, 32) binary -> (..., 32, 32) uint16."""
    c = (np.asarray(crop) > 0).astype(np.uint32)
    return np.tensordot(c, _BITS, axes=([-3], [0])).astype(np.uint16) if c.ndim == 3 else \
        np.einsum("...chw,c->...hw", c, _BITS).astype(np.uint16)


def unpack_np(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed).astype(np.uint32)[..., None, :, :]
    return ((p >> np.arange(EGO_MAP_CH, dtype=np.uint32)[:, None, None]) & 1).astype(np.uint8)


def unpack_torch(packed):
    """uint16-valued int tensor (..., 32, 32) -> float (..., 16, 32, 32)."""
    import torch
    p = packed.to(torch.int32).unsqueeze(-3)
    shifts = torch.arange(EGO_MAP_CH, device=packed.device, dtype=torch.int32).view(-1, 1, 1)
    return ((p >> shifts) & 1).to(torch.float32)


def replay(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(ep["action"])
    m = EgoMap()
    maps = np.zeros((n, 32, 32), np.uint16)
    poses = np.zeros((n, 6), np.float16)
    execs = ep["exec"] if "exec" in ep else ep["action"]
    prev = None
    for t in range(n):
        obs = {"rays": {"distance": ep["dist"][t].astype(np.float32), "hit_type": ep["hit"][t]},
               "visible_entities": {"features": ep["efeat"][t].astype(np.float32),
                                    "entity_type": ep["etype"][t], "mask": ep["emask"][t]},
               "vitals": ep["vitals"][t]}
        m.update(obs, prev)
        maps[t] = pack(m.ego_crop())
        poses[t] = m.pose()
        prev = execs[t]
    return maps, poses


def _one(args: tuple[str, str]) -> int:
    d, f = args
    out = Path(d) / "ego" / f
    if out.exists():
        return 0
    with np.load(Path(d) / f) as z:
        ep = {k: z[k] for k in ("action", "dist", "hit", "efeat", "etype", "emask", "vitals")}
        if "exec" in z.files:
            ep["exec"] = z["exec"]
    maps, poses = replay(ep)
    np.savez_compressed(out, egomap=maps, pose=poses)
    return len(maps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    jobs = []
    for d in [p for p in args.data.split(",") if p]:
        (Path(d) / "ego").mkdir(exist_ok=True)
        for line in (Path(d) / "manifest.jsonl").read_text().splitlines():
            f = json.loads(line)["file"]
            if (Path(d) / f).exists():
                jobs.append((d, f))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        steps = sum(pool.imap_unordered(_one, jobs, chunksize=4))
    print(f"{len(jobs)} shards, {steps} new steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
