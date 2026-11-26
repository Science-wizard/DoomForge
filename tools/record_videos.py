"""Record showcase clips with the shipped agent.

Writes MP4s and GIF previews to docs/assets/videos/, plus a metadata JSON with
the category, map, outcome and step count.  Uses the official DoomAgent path so
the clips show exactly what the scorer runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from submission.agent import DoomAgent
from training.evaluate import run_agent_episode

CLIPS = [
    ("C0_sanity", 0, "C0 sanity: visible exit"),
    ("C5_key_lock", 0, "C5 key -> locked door"),
    ("C6_two_key", 0, "C6 two keys, two locks"),
    ("C2_arena", None, "C2 arena combat"),
    ("C4_room_door", None, "C4 room door"),
    ("C8_backtrack", None, "C8 backtrack"),
]

FONT_PATHS = ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf")


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def overlay(frame: np.ndarray, text: str, size: tuple[int, int]) -> Image.Image:
    im = Image.fromarray(frame).resize(size, Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    f = font(max(12, size[1] // 16))
    d.rectangle([0, 0, size[0], max(22, size[1] // 11)], fill=(8, 12, 16, 190))
    d.text((8, 4), text, fill=(240, 245, 250), font=f)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="docs/assets/videos")
    ap.add_argument("--fps", type=float, default=9.0)
    ap.add_argument("--max-per-category", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in
            (Path("training/data/suite") / args.split / "manifest.jsonl").read_text().splitlines()
            if l.strip()]
    agent = DoomAgent(None, SimpleNamespace(shape=(9,)), "cpu")
    meta = []
    for cat, idx, label in CLIPS:
        cat_rows = [r for r in rows if r["category"] == cat]
        chosen, result = None, None
        if idx is not None:
            chosen = cat_rows[idx]
        else:
            for row in cat_rows[: args.max_per_category]:
                r = run_agent_episode(None, row["wad"], row["map_name"], ticks=row["ticks"],
                                      seed=0, record=True, track_sectors=True, agent=agent)
                if r["exit"]:
                    chosen, result = row, r
                    break
            if chosen is None:
                chosen = cat_rows[0]
        if result is None:
            result = run_agent_episode(None, chosen["wad"], chosen["map_name"],
                                       ticks=chosen["ticks"], seed=0, record=True,
                                       track_sectors=True, agent=agent)
        frames = result["frames"]
        if not frames:
            print("no frames for", cat)
            continue
        tag = f"{cat}_{chosen['map_id'].split('_')[-1]}"
        text = (f"{label} | {result['steps']} steps | "
                f"{'exit' if result['exit'] else 'no exit'}")
        mp4 = out / f"{tag}.mp4"
        gif = out / f"{tag}.gif"
        big = [np.asarray(overlay(f, text, (480, 360))) for f in frames]
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(mp4, big, fps=args.fps, codec="libx264", quality=8,
                             pixelformat="yuv420p", macro_block_size=1)
        except Exception as exc:
            print("mp4 failed", exc)
        small = [overlay(f, text, (320, 240)).convert("P", palette=Image.ADAPTIVE, colors=128)
                 for f in frames[::2]]
        small[0].save(gif, save_all=True, append_images=small[1:],
                      duration=int(1000 / max(1.0, args.fps / 2)), loop=0, optimize=True)
        meta.append({"category": cat, "map_id": chosen["map_id"], "mp4": str(mp4),
                     "gif": str(gif), "steps": result["steps"], "exit": bool(result["exit"]),
                     "kills": result["kills"], "keys": result["keys"],
                     "proxy": result["proxy"]})
        print(f"{cat}: {chosen['map_id']} {result['steps']} steps exit={result['exit']} "
              f"-> {mp4.name}, {gif.name}", flush=True)
        del result
    (out / "clips.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", out / "clips.json")


if __name__ == "__main__":
    main()
