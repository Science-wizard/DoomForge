"""Record a short replay and write a GIF for the README.

Usage:
    python -m tools.make_gif --ckpt training/runs/bc_map/ckpt.pt --wad maps/generated/t2_s10000.wad \
        --out docs/assets/doomforge.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from training.evaluate import load_net, run_agent_episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="training/runs/bc_map/ckpt.pt")
    ap.add_argument("--wad", default="maps/two_rooms.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--out", default="docs/assets/doomforge.gif")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=180)
    args = ap.parse_args()

    net = load_net(args.ckpt)
    r = run_agent_episode(net, args.wad, args.map, device="cpu", record=True, seed=0)
    frames = r.get("frames", [])
    if not frames:
        raise SystemExit("no frames recorded")
    frames = frames[::args.stride][: args.max_frames]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(np.asarray(f, np.uint8)).resize(
        (160 * args.scale, 120 * args.scale), Image.NEAREST) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=40, loop=0, optimize=True)
    print(f"wrote {out} ({len(imgs)} frames, exit={r['exit']}, steps={r['steps']})")


if __name__ == "__main__":
    main()
