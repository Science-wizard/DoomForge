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
    ours, ref = suite_summary("test_ego8_u10"), suite_summary("test_hybrid_p4")
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




# ----------------------------------------------------------------------------- perception
def record(map_id: str):
    """Play the shipped agent and keep, per decision: frame, observation, ego crop, the
    network's auxiliary predictions and its six action-head distributions."""
    import torch
    from submission.model import AUX_ENEMY, AUX_EXIT_BEARING, AUX_EXIT_VIS, AUX_KEY_COLOURS, HEAD_SIZES
    steps = []
    cap = {}

    def hook(obs, a, env):
        p = env.privileged_state()
        steps.append({"rgb": np.asarray(obs["rgb"]).copy(), "obs": obs, "a": a.copy(),
                      "keys": int(np.sum(p["keys"])), **cap})
    res, agent = None, None
    from submission.agent import DoomAgent
    orig = DoomAgent._batch

    def spy(self, o):
        b = orig(self, o)
        cap["crop"] = b["map"][0].cpu().numpy().astype(np.uint8)
        return b
    DoomAgent._batch = spy
    import submission.model as M
    fwd = M.DoomNet.forward_step

    def fwd_spy(self, batch, h=None):
        logits, value, aux, h2 = fwd(self, batch, h)
        sig = torch.sigmoid(aux[0])
        cap["exit_vis"] = float(sig[AUX_EXIT_VIS])
        cap["enemy"] = float(sig[AUX_ENEMY])
        cap["key_cols"] = sig[AUX_KEY_COLOURS].numpy().copy()
        cap["bearing"] = int(torch.argmax(aux[0, AUX_EXIT_BEARING]))
        cap["heads"] = [torch.softmax(x, -1).numpy() for x in torch.split(logits[0], HEAD_SIZES)]
        return logits, value, aux, h2
    M.DoomNet.forward_step = fwd_spy
    try:
        res, agent = play(map_id, extra=hook)
    finally:
        DoomAgent._batch = orig
        M.DoomNet.forward_step = fwd
    return steps, res


def save_shot(step, name: str) -> None:
    Image.fromarray(step["rgb"]).resize((480, 360), Image.LANCZOS).save(OUT / f"{name}.jpg", quality=92)


def fig_perception() -> None:
    """Gameplay shots with the network's own predictions, the sensor view, the 16 map
    channels and the action-head distributions at one real decision.  Shot decisions were
    picked by eye from candidate frames (same seed, so they are reproducible)."""
    out = {}
    c4, _ = record("C4_room_door_10000")
    s = 188
    save_shot(c4[s], "shot_exit")
    out["exit"] = {"map": "C4_room_door_10000", "step": s, "p": c4[s]["exit_vis"], "bearing_bin": c4[s]["bearing"]}

    c6, _ = record("C6_two_key_10001")
    cols = ["blue", "yellow", "red", "blue skull", "yellow skull", "red skull"]
    s = 141
    save_shot(c6[s], "shot_key")
    out["key"] = {"map": "C6_two_key_10001", "step": s, "colour": cols[int(np.argmax(c6[s]["key_cols"]))],
                  "p": float(c6[s]["key_cols"].max())}
    s = 129
    save_shot(c6[s], "shot_lock")
    out["lock"] = {"map": "C6_two_key_10001", "step": s, "keys_held": c6[s]["keys"],
                   "red_key_p": float(c6[s]["key_cols"][2]), "move_fwd": float(c6[s]["heads"][0][1])}

    c2, _ = record("C2_arena_10001")
    s = 74
    ve = c2[s]["obs"]["visible_entities"]
    n_en = int(np.sum((np.asarray(ve["mask"]) > 0) & np.isin(np.asarray(ve["entity_type"]), (1, 2, 3))))
    save_shot(c2[s], "shot_combat")
    out["combat"] = {"map": "C2_arena_10001", "step": s, "enemies": n_en, "p": c2[s]["enemy"],
                     "attack": float(c2[s]["heads"][3][1])}
    sensor_view(c2[s], "sensors")
    heads_view(c2[s], "heads")
    Image.fromarray(c2[s]["rgb"][::4, ::4][:30, :40]).resize((160, 120), Image.NEAREST).save(OUT / "pooled.png")
    Image.fromarray(c2[s]["rgb"]).resize((320, 240), Image.LANCZOS).save(OUT / "full.png")

    c7, _ = record("C7_exploration_10000")
    channels_view(c7[min(450, len(c7) - 1)]["crop"], "channels")
    (OUT / "perception.json").write_text(json.dumps(out, indent=1))
    print("perception:", out)


def sensor_view(step, name: str) -> None:
    obs = step["obs"]
    d = np.asarray(obs["rays"]["distance"], float)
    h = np.asarray(obs["rays"]["hit_type"])
    ang = np.arange(64) * 2 * np.pi / 64 + np.pi / 2     # ray 0 straight ahead = up
    col = {0: "#cbd5e1", 1: INK, 2: ORANGE, 3: RED, 4: "#dc2626", 5: GREEN}
    fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=220)
    for i in range(64):
        x, y = d[i] * np.cos(ang[i]), d[i] * np.sin(ang[i])
        ax.plot([0, x], [0, y], color="#e2e8f0", lw=0.6, zorder=1)
        ax.plot(x, y, "o", ms=2.6, color=col[int(h[i])], zorder=3)
    ve = obs["visible_entities"]
    for f, t, m in zip(np.asarray(ve["features"]), np.asarray(ve["entity_type"]), np.asarray(ve["mask"])):
        if m > 0:
            r = f[0]
            a = np.arctan2(f[1], f[2]) + np.pi / 2
            ax.plot(r * np.cos(a), r * np.sin(a), "^" if t in (1, 2, 3) else "s", ms=5,
                    color=RED if t in (1, 2, 3) else PURPLE, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    fov = np.linspace(np.pi / 2 - np.pi / 4, np.pi / 2 + np.pi / 4, 30)
    ax.fill(np.r_[0, 1.05 * np.cos(fov), 0], np.r_[0, 1.05 * np.sin(fov), 0], color=BLUE, alpha=0.06, lw=0)
    ax.plot(0, 0, marker=(3, 0, 0), ms=8, color=BLUE)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def heads_view(step, name: str) -> None:
    labels = [("move", ["—", "fwd", "back"]), ("strafe", ["—", "left", "right"]), ("turn", ["—", "left", "right"]),
              ("attack", ["no", "fire"]), ("use", ["no", "use"]), ("weapon", ["keep", "next"])]
    fig, axes = plt.subplots(1, 6, figsize=(3.6, 1.05), dpi=220, sharey=True)
    for ax, (lab, opts), p in zip(axes, labels, step["heads"]):
        x = np.arange(len(p))
        ax.bar(x, p, color=[RED if i == int(np.argmax(p)) else GREY for i in x], width=0.75)
        ax.set_xticks(x, opts, fontsize=5.2, rotation=0)
        ax.set_title(lab, fontsize=6.4, color=INK, pad=2)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=5.2, length=2)
        ax.tick_params(axis="x", length=0)
        for sp in ("left",):
            ax.spines[sp].set_visible(ax is axes[0])
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=False)
    fig.tight_layout(pad=0.15, w_pad=0.3)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def channels_view(crop, name: str) -> None:
    names = ["free", "wall", "door", "locked", "item", "key", "trail", "frontier"]
    cmap_cols = ["#94a3b8", INK, ORANGE, RED, GREEN, PURPLE, BLUE, "#eab308"]
    fig, axes = plt.subplots(2, 8, figsize=(4.6, 1.35), dpi=220)
    for r, scale in enumerate(("64 u/cell", "256 u/cell")):
        for c in range(8):
            ax = axes[r, c]
            img = np.ones((32, 32, 3))
            m = crop[r * 8 + c] > 0
            rgb = matplotlib.colors.to_rgb(cmap_cols[c])
            img[m] = rgb
            ax.imshow(img, interpolation="nearest")
            ax.plot(15.5, 15.5, marker=(3, 0, 0), ms=3, color=BLUE if c != 6 else RED)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(RULE)
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(names[c], fontsize=6, color=INK, pad=1.5)
        axes[r, 0].set_ylabel(scale, fontsize=5.6, color=MUTED)
    fig.tight_layout(pad=0.1, w_pad=0.15, h_pad=0.2)
    fig.savefig(OUT / f"{name}.png", dpi=260)
    plt.close(fig)


def fig_model() -> None:
    """Parameters per module, and the reward terms as a diverging chart."""
    from submission.model import DoomNet
    net = DoomNet(arch="ego")
    groups = {"GRU 384": ["gru"], "Map CNN": ["map"], "Fusion": ["fuse"], "Colour CNN": ["rgb"], "Trunk": ["trunk"],
              "Goal head + enc.": ["goal_head", "goal"], "Entity attention": ["ent"], "Ray conv": ["ray"],
              "Heads (act/aux/V)": ["actor", "aux", "critic"], "Vitals/prev/pose": ["vit", "prev", "pose"]}
    vals = {k: sum(p.numel() for n in v for p in getattr(net, n).parameters()) / 1e3 for k, v in groups.items()}
    fig, ax = plt.subplots(figsize=(2.9, 2.0), dpi=220)
    ks = list(vals)[::-1]
    ax.barh(ks, [vals[k] for k in ks], color=[RED if k == "Map CNN" else ("#334155" if k == "GRU 384" else GREY) for k in ks], height=0.66)
    for i, k in enumerate(ks):
        ax.text(vals[k] + 12, i, f"{vals[k]:.0f}k", va="center", fontsize=6.4, color=INK)
    ax.tick_params(axis="y", labelsize=6.6, length=0, colors=INK)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("parameters (thousands) · 2.00 M acting total", fontsize=6.6)
    ax.set_xlim(0, max(vals.values()) * 1.18)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "params.svg")
    plt.close(fig)

    terms = [("exit (+2 × time left)", 10, "event"), ("key picked up", 5, "event"), ("locked door opened", 3, "event"),
             ("kill", 0.6, "event"), ("door opened", 0.5, "event"), ("new sector", 0.3, "event"),
             ("progress to objective", 0.09, "≤ per step"), ("newly seen floor", 0.1, "≤ per step"),
             ("new 64-unit cell", 0.01, "event"), ("time", -0.001, "per step"),
             ("stagnation", -0.02, "per step"), ("USE on lock, no key", -0.02, "per press"),
             ("damage", -0.02, "per HP"), ("death", -7, "6×t+4")]
    fig, ax = plt.subplots(figsize=(3.2, 2.45), dpi=220)
    y = np.arange(len(terms))[::-1]
    v = np.array([t[1] for t in terms])
    ax.barh(y, np.sign(v) * np.log10(1 + 100 * np.abs(v)), color=[GREEN if x > 0 else RED for x in v], height=0.64)
    for yi, (lab, val, unit) in zip(y, terms):
        x = np.sign(val) * np.log10(1 + 100 * abs(val))
        txt = f"{val:+g}" if lab != "death" else "−4…−10"
        ax.text(x + 0.08 if val > 0 else 0.08, yi, f"{txt}  {unit}" if unit != "event" else txt,
                va="center", ha="left", fontsize=5.6, color=MUTED)
    ax.set_yticks(y, [t[0] for t in terms], fontsize=6.3, color=INK)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-3.0, 3.9)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "rewards.svg")
    plt.close(fig)


FIGS = {"skills": fig_skills, "odometry": fig_odometry, "learning": fig_learning, "gains": fig_gains,
        "perception": fig_perception, "model": fig_model}


if __name__ == "__main__":
    for w in sys.argv[1:] or list(FIGS):
        FIGS[w]()
