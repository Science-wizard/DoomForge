"""Render the figures in docs/brief/assets/ from the shipped agent and the logs.

    PYTHONPATH=. uv run python docs/brief/make_figures.py

Every figure comes from real runs: the shipped `submission/agent.py` on
validation maps, the odometry measured against the engine's true pose, PPO
training logs, and the suite results in training/runs/suite/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/brief/assets"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#0f172a", "#64748b", "#e2e8f0"
RED, BLUE, GREEN, ORANGE, PURPLE, GREY = "#b91c1c", "#2563eb", "#16a34a", "#ea580c", "#9333ea", "#cbd5e1"
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "svg.fonttype": "none",
})


def val_row(map_id: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / "training/data/suite/val/manifest.jsonl").read_text().splitlines() if l]
    return next(r for r in rows if r["map_id"] == map_id)


def play(map_id: str, extra=None):
    """Run the shipped agent; return path, events and the agent."""
    from doom_env import make_env
    from submission.agent import DoomAgent
    r = val_row(map_id)
    agent = DoomAgent(None, SimpleNamespace(shape=(9,)), "cpu")
    env = make_env(r["wad"], r["map_name"], episode_ticks=r["ticks"], render_buffers=True)
    obs, _ = env.reset(seed=0)
    agent.reset()
    p = env.privileged_state()
    path, keys, kills = [(p["x"], p["y"])], [], []
    nk, nkill = int(np.sum(p["keys"])), p["kills"]
    while True:
        a = agent.act(obs)
        if extra:
            extra(obs, a, env)
        obs, _, te, tr, _ = env.step(a)
        p = env.privileged_state()
        path.append((p["x"], p["y"]))
        if int(np.sum(p["keys"])) > nk:
            keys.append((p["x"], p["y"]))
            nk = int(np.sum(p["keys"]))
        if p["kills"] > nkill:
            kills.append((p["x"], p["y"]))
            nkill = p["kills"]
        if te or tr:
            break
    out = {"path": np.array(path), "keys": keys, "kills": kills, "exit": bool(p["exit_reached"]),
           "steps": len(path) - 1, "row": r}
    env.close()
    return out, agent


def draw_map(ax, row, res, lw=0.8, scale=1.0):
    from training.teacher.mapdata import load_mapdata
    md = load_mapdata(row["wad"], row["map_name"])
    p1, p2 = md.line_points()
    doors = set(md.info.door_sectors)
    locks = {s for s, l in md.info.door_sectors.items() if l > 0}
    for i in range(len(p1)):
        sec = {md.front[i], md.back[i]}
        col, w = (RED, 1.8 * scale) if sec & locks else ((ORANGE, 1.5 * scale) if sec & doors else (INK, lw * scale))
        ax.plot([p1[i][0], p2[i][0]], [p1[i][1], p2[i][1]], color=col, lw=w, solid_capstyle="round")
    for x1, y1, x2, y2, _ in md.exit_lines():
        ax.plot([x1, x2], [y1, y2], color=GREEN, lw=4 * scale, solid_capstyle="butt")
    xs = res["path"]
    ax.plot(xs[:, 0], xs[:, 1], color=BLUE, lw=1.3 * scale, alpha=0.95, zorder=3)
    ax.plot(*xs[0], "o", color=BLUE, ms=5 * scale, zorder=4)
    if res["exit"]:
        ax.plot(*xs[-1], "*", color=GREEN, ms=11 * scale, zorder=5, markeredgecolor="white", markeredgewidth=0.6)
    for k in res["keys"]:
        ax.plot(*k, "D", color=PURPLE, ms=5.5 * scale, zorder=5, markeredgecolor="white", markeredgewidth=0.6)
    for k in res["kills"]:
        ax.plot(*k, "X", color=RED, ms=6 * scale, zorder=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_skills() -> None:
    """Hero replay + four skill panels."""
    for map_id, name in (("C7_exploration_10000", "hero"), ("C6_two_key_10001", "skill_twokey"),
                         ("C4_room_door_10000", "skill_roomdoor"), ("C8_backtrack_10000", "skill_backtrack"),
                         ("C2_arena_10001", "skill_arena")):
        res, agent = play(map_id)
        hero = name == "hero"
        fig, ax = plt.subplots(figsize=(4, 4) if hero else (8, 8), dpi=220)
        draw_map(ax, res["row"], res, lw=0.9 if hero else 1.2, scale=1.0 if hero else 2.2)
        fig.tight_layout(pad=0.05)
        fig.savefig(OUT / f"{name}.png", transparent=True)
        plt.close(fig)
        im = Image.open(OUT / f"{name}.png")
        im.crop(im.getbbox()).save(OUT / f"{name}.png")  # trim transparent margins
        print(name, map_id, "steps", res["steps"], "exit", res["exit"], "keys", len(res["keys"]),
              "kills", len(res["kills"]))
        if name == "hero":
            agent_map(agent, res)


def agent_map(agent, res) -> None:
    """The ego map the agent built itself during the hero episode."""
    g = agent.map.ego.copy()
    img = np.full(g.shape[1:] + (3,), 255, np.uint8)
    for ch, col in ((0, (226, 232, 240)), (7, (250, 204, 21)), (6, (37, 99, 235)), (1, (15, 23, 42)),
                    (4, (34, 197, 94)), (5, (147, 51, 234)), (2, (234, 88, 12)), (3, (185, 28, 28))):
        img[g[ch] > 0] = col
    img = img[::-1]
    ys, xs = np.nonzero((g.sum(0) > 0)[::-1])
    img = img[ys.min() - 2: ys.max() + 3, xs.min() - 2: xs.max() + 3]
    im = Image.fromarray(img).resize((img.shape[1] * 12, img.shape[0] * 12), Image.NEAREST)
    from doom_env import make_env  # spawn heading -> world frame
    r = res["row"]
    env = make_env(r["wad"], r["map_name"], episode_ticks=r["ticks"], render_buffers=False)
    env.reset(seed=0)
    a0 = env.privileged_state()["angle"]
    env.close()
    im = im.rotate(a0, expand=True, fillcolor=(255, 255, 255), resample=Image.NEAREST)
    im = im.crop(Image.eval(im.convert("L"), lambda v: 255 - v).getbbox())
    im.save(OUT / "agent_map.png")


def fig_odometry() -> None:
    """Position error of the corrected vs naive dead reckoning on the agent's own episodes
    (mean and interquartile band over every validation episode that lasts 500 decisions)."""
    from submission.mapmem import MapBuilder
    errs = {"new": [], "old": []}
    maps = [f"C7_exploration_{10000 + i}" for i in range(6)] + [f"C11_gauntlet_{10000 + i}" for i in range(3)] \
        + [f"C13_ood_{10000 + i}" for i in range(3)]
    for map_id in maps:
        state = {"new": MapBuilder(legacy=False), "old": MapBuilder(legacy=True), "prev": None, "t0": None,
                 "e": {"new": [], "old": []}}

        def hook(obs, a, env, s=state):
            p = env.privileged_state()
            if s["t0"] is None:
                s["t0"] = (p["x"], p["y"], p["angle"])
            for k in ("new", "old"):
                s[k].integrate(obs, s["prev"])
            x0, y0, h0 = s["t0"]
            c, sn = np.cos(np.radians(-h0)), np.sin(np.radians(-h0))
            dx, dy = p["x"] - x0, p["y"] - y0
            tx, ty = dx * c - dy * sn, dx * sn + dy * c
            for k in ("new", "old"):
                s["e"][k].append(float(np.hypot(tx - s[k].px, ty - s[k].py)))
            s["prev"] = a
        play(map_id, extra=hook)
        for k in ("new", "old"):
            e = state["e"][k][:500]
            if len(e) >= 500:
                errs[k].append(e)
    n = min(len(errs["new"]), len(errs["old"]))
    new, old = np.array(errs["new"][:n]), np.array(errs["old"][:n])
    t = np.arange(new.shape[1])
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=220)
    ax.fill_between(t, np.percentile(old, 25, 0), np.percentile(old, 75, 0), color=GREY, alpha=0.55, lw=0)
    ax.plot(t, old.mean(0), color=MUTED, lw=1.4, label="naive integration")
    ax.fill_between(t, np.percentile(new, 25, 0), np.percentile(new, 75, 0), color=RED, alpha=0.18, lw=0)
    ax.plot(t, new.mean(0), color=RED, lw=1.8, label="DoomForge odometry")
    ax.set_xlabel("decisions")
    ax.set_ylabel("position error (map units)")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "odometry.svg")
    plt.close(fig)
    print("odometry @500: naive %.0f  ours %.0f units (n=%d episodes)" % (old[:, 499].mean(), new[:, 499].mean(), n))


def fig_learning() -> None:
    """Training-map success rates across the three PPO stages."""
    stages = [("training/runs/ppo_ego1/train.jsonl", 50), ("training/runs/ppo_ego5/train.jsonl", 100),
              ("training/runs/ppo_ego6/train.jsonl", 50)]
    xs, series, bounds, steps = [], {}, [], 0
    cats = {"C4_room_door": ("Door on a room wall", BLUE), "C8_backtrack": ("Key behind spawn", PURPLE),
            "C1_duel": ("Duel", RED), "C7_exploration": ("Multi-room exploration", GREEN)}
    for path, last in stages:
        seen = set()
        rows = []
        for l in (ROOT / path).read_text().splitlines():
            r = json.loads(l)
            if r["update"] < last and r["update"] not in seen:
                seen.add(r["update"])
                rows.append(r)
        prev = rows[0]["steps"] - (rows[1]["steps"] - rows[0]["steps"])
        for r in rows:
            steps += r["steps"] - prev
            prev = r["steps"]
            xs.append(steps / 1e6)
            for c in cats:
                series.setdefault(c, []).append(r["cat_exit"].get(c, np.nan))
        bounds.append(steps / 1e6)
    fig, ax = plt.subplots(figsize=(3.4, 2.15), dpi=220)
    for c, (label, col) in cats.items():
        y = np.array(series[c], float)
        y = np.convolve(np.nan_to_num(y), np.ones(5) / 5, mode="same")
        ax.plot(xs[2:-2], 100 * y[2:-2], color=col, lw=1.5, label=label)
    for b, s in zip(bounds[:-1], ("B", "C")):
        ax.axvline(b, color=RULE, lw=0.8, zorder=0)
        ax.text(b + 0.02, 4, f"stage {s}", fontsize=6.5, color=MUTED)
    ax.text(0.02, 4, "stage A", fontsize=6.5, color=MUTED)
    ax.set_ylim(0, 112)
    ax.set_xlim(0, xs[-1])
    ax.set_xlabel("PPO environment steps (millions)")
    ax.set_ylabel("exit rate on training maps (%)")
    ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "learning.svg")
    plt.close(fig)


def suite_summary(run: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / "training/runs/suite" / run / "episodes.jsonl").read_text().splitlines() if l]
    out = {}
    for r in rows:
        d = out.setdefault(r["category"], {"proxy": [], "cov": [], "exit": []})
        d["proxy"].append(r["proxy"])
        d["cov"].append(r["coverage"])
        d["exit"].append(r["exit"])
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in out.items()}


NAMES = {"C1_duel": "Duel", "C4_room_door": "Door on a room wall", "C8_backtrack": "Key behind spawn",
         "C7_exploration": "Multi-room exploration", "C10_resource": "Scarce ammo", "C9_ambush": "Ambush",
         "C13_ood": "Out-of-distribution maps", "C11_gauntlet": "Gauntlet", "C0_sanity": "Sanity",
         "C3_corridor_door": "Corridor door", "C6_two_key": "Two keys, two locks", "C2_arena": "Arena",
         "C5_key_lock": "Key → lock", "C12_freedoom": "Freedoom"}


def fig_gains() -> None:
    """Largest per-category score gains on the held-out test split, and hard-map coverage."""
    ours, ref = suite_summary("test_ego6_u50"), suite_summary("test_hybrid_p4")
    gains = sorted(((ours[c]["proxy"] - ref[c]["proxy"], c) for c in ours), reverse=True)[:6]
    fig, ax = plt.subplots(figsize=(3.4, 1.95), dpi=220)
    y = np.arange(len(gains))[::-1]
    ax.barh(y, [g for g, _ in gains], color=RED, height=0.62)
    for yi, (g, c) in zip(y, gains):
        ax.text(g + 0.15, yi, f"+{g:.1f}", va="center", fontsize=7, color=RED, fontweight="bold")
    ax.set_yticks(y, [NAMES[c] for _, c in gains], fontsize=7.2, color=INK)
    ax.set_xlabel("score gain per map vs reference (test split)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(g for g, _ in gains) * 1.18)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "gains.svg")
    plt.close(fig)

    hard = ["C9_ambush", "C10_resource", "C11_gauntlet", "C13_ood"]
    fig, ax = plt.subplots(figsize=(3.4, 1.7), dpi=220)
    x = np.arange(len(hard))
    ax.bar(x - 0.19, [100 * ref[c]["cov"] for c in hard], width=0.36, color=GREY, label="reference")
    ax.bar(x + 0.19, [100 * ours[c]["cov"] for c in hard], width=0.36, color=RED, label="DoomForge")
    for xi, c in zip(x, hard):
        mult = ours[c]["cov"] / max(ref[c]["cov"], 1e-6)
        ax.text(xi + 0.19, 100 * ours[c]["cov"] + 1, f"{mult:.1f}×", ha="center", fontsize=7, color=RED, fontweight="bold")
    ax.set_xticks(x, [NAMES[c].replace(" maps", "") for c in hard], fontsize=7, color=INK)
    ax.set_ylabel("map explored (%)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax.set_ylim(0, max(100 * ours[c]["cov"] for c in hard) * 1.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "coverage.svg")
    plt.close(fig)
    print("gains:", [(c, round(g, 2)) for g, c in gains])


if __name__ == "__main__":
    which = sys.argv[1:] or ["skills", "odometry", "learning", "gains"]
    for w in which:
        {"skills": fig_skills, "odometry": fig_odometry, "learning": fig_learning, "gains": fig_gains}[w]()
