"""Record full-length gameplay footage for the whole benchmark suite.

Unlike ``record_videos.py`` (six short showcase clips for the README), this
script records one complete, uncut episode per category into
``docs/assets/videos/full/``: every decision from spawn to exit, death or
timeout, at 4x the native 160x120 frame with a small step counter.

Usage:
    python -m tools.record_gallery                     # all 14 categories
    python -m tools.record_gallery --categories C9_ambush,C12_freedoom
    python -m tools.record_gallery --force             # re-record existing files
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
from training.suite.spec import CATEGORIES

FONT_PATHS = ("/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf")

BAR_BG = (8, 12, 16, 205)
INK = (240, 245, 250)
ACCENT = (255, 170, 60)
GOOD = (120, 220, 140)


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """4x nearest-neighbour upscale keeps the pixel-art look crisp."""
    return Image.fromarray(frame).resize(size, Image.NEAREST)


def decorate(frame: np.ndarray, size: tuple[int, int], title: str,
             step: int, total: int) -> Image.Image:
    im = _frame(frame, size)
    d = ImageDraw.Draw(im, "RGBA")
    bar = max(30, size[1] // 14)
    d.rectangle([0, 0, size[0], bar], fill=BAR_BG)
    f = font(max(13, size[1] // 30))
    d.text((10, bar // 2), title, fill=INK, font=f, anchor="lm")
    counter = f"{step}/{total}"
    d.text((size[0] - 10, bar // 2), counter, fill=ACCENT, font=f, anchor="rm")
    return im


def outcome_card(frame: np.ndarray, size: tuple[int, int], title: str,
                 outcome: str, detail: str) -> Image.Image:
    im = _frame(frame, size).convert("RGB")
    shade = Image.new("RGB", size, (6, 9, 12))
    im = Image.blend(im, shade, 0.72)
    d = ImageDraw.Draw(im)
    big = font(max(22, size[1] // 13))
    small = font(max(12, size[1] // 28))
    d.text((size[0] // 2, size[1] // 2 - size[1] // 18), title,
           fill=INK, font=small, anchor="mm")
    d.text((size[0] // 2, size[1] // 2 + size[1] // 30), outcome,
           fill=GOOD if outcome.startswith("EXIT") else ACCENT, font=big, anchor="mm")
    d.text((size[0] // 2, size[1] // 2 + size[1] // 8), detail,
           fill=INK, font=small, anchor="mm")
    return im


def render_mp4(frames: list[np.ndarray], path: Path, size: tuple[int, int],
               title: str, outcome: str, detail: str, fps: float) -> None:
    import imageio.v2 as imageio

    total = len(frames)
    images = [np.asarray(decorate(f, size, title, i + 1, total)) for i, f in enumerate(frames)]
    images += [np.asarray(outcome_card(frames[-1], size, title, outcome, detail))] * max(1, int(fps))
    imageio.mimwrite(path, images, fps=fps, codec="libx264", quality=6,
                     pixelformat="yuv420p", macro_block_size=1)


def poster(frames: list[np.ndarray], path: Path) -> None:
    frame = frames[len(frames) // 2]
    _frame(frame, (320, 240)).save(path, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="docs/assets/videos/full")
    ap.add_argument("--fps", type=float, default=9.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--attempts", type=int, default=4,
                    help="episodes to try per category when hunting for an exit")
    ap.add_argument("--categories", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    size = (args.width, args.height)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "posters").mkdir(exist_ok=True)

    manifest = Path("training/data/suite") / args.split / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    cats = list(CATEGORIES) if args.categories == "all" else args.categories.split(",")
    agent = DoomAgent(None, SimpleNamespace(shape=(9,)), "cpu")
    meta = []
    for cat in cats:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            print(f"{cat}: no rows, skipping")
            continue
        # Pass 1: cheap, no frames. Pick the first exit, else the best proxy.
        best, best_score = None, None
        for row in cat_rows[: args.attempts]:
            r = run_agent_episode(None, row["wad"], row["map_name"], ticks=row["ticks"],
                                  seed=args.seed, record=False, track_sectors=True,
                                  agent=agent)
            score = (1 if r["exit"] else 0, r["proxy"])
            if best is None or score > best_score:
                best, best_score = row, score
            if r["exit"]:
                break
        if best is None:
            print(f"{cat}: no episode")
            continue
        mp4 = out / f"{best['map_id']}.mp4"
        jpg = out / "posters" / f"{best['map_id']}.jpg"
        if mp4.exists() and not args.force:
            print(f"{cat}: {mp4.name} exists, skipping")
            continue
        # Pass 2: replay the chosen episode with frames. Deterministic seed.
        r = run_agent_episode(None, best["wad"], best["map_name"], ticks=best["ticks"],
                              seed=args.seed, record=True, track_sectors=True, agent=agent)
        frames = r["frames"]
        if not frames:
            print(f"{cat}: no frames")
            continue
        title = f"{CATEGORIES[cat].description}  ·  {best['map_id']}"
        if r["exit"]:
            outcome = "EXIT"
            detail = f"{r['steps']} decisions · {r['kills']} kills · {r['keys']} keys"
        elif r["dead"]:
            outcome = "DEATH"
            detail = f"{r['steps']} decisions · {r['kills']} kills · {r['coverage']:.0%} mapped"
        else:
            outcome = "TIMEOUT"
            detail = (f"{r['steps']} decisions · {r['kills']} kills · "
                      f"{r['coverage']:.0%} mapped")
        render_mp4(frames, mp4, size, title, outcome, detail, args.fps)
        poster(frames, jpg)
        meta.append({
            "category": cat,
            "map_id": best["map_id"],
            "wad": best["wad"],
            "map_name": best["map_name"],
            "mp4": str(mp4),
            "poster": str(jpg),
            "steps": r["steps"],
            "exit": bool(r["exit"]),
            "dead": bool(r["dead"]),
            "kills": r["kills"],
            "keys": r["keys"],
            "coverage": round(r["coverage"], 3),
            "proxy": round(r["proxy"], 2),
            "outcome": outcome,
        })
        print(f"{cat}: {best['map_id']} {r['steps']} steps {outcome.lower()} "
              f"({r['kills']} kills, {r['coverage']:.0%} mapped) -> {mp4.name}", flush=True)
        del frames, r

    index = out / "clips.json"
    if index.exists():
        old = json.loads(index.read_text())
        seen = {m["category"] for m in meta}
        meta = [m for m in old if m["category"] not in seen] + meta
    # Drop superseded files from earlier picks in the same category.
    keep = {Path(m["mp4"]).name for m in meta}
    keep |= {Path(m["poster"]).name for m in meta}
    for cat in cats:
        for stale in out.glob(f"{cat}_*.mp4"):
            if stale.name not in keep:
                stale.unlink()
        for stale in (out / "posters").glob(f"{cat}_*.jpg"):
            if stale.name not in keep:
                stale.unlink()
    meta.sort(key=lambda m: list(CATEGORIES).index(m["category"]))
    index.write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", index)


if __name__ == "__main__":
    main()
