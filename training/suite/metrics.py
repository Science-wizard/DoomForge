"""Metrics, bootstrap confidence intervals and Markdown reports for the suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TIER_WEIGHTS = {1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0}
METRICS = ("exit", "proxy", "coverage", "keys", "locks", "doors", "kills",
           "damage", "steps", "spl", "spin_frac", "noop_frac", "dead")


def load_rows(run: str | Path) -> list[dict]:
    p = Path(run)
    if p.is_dir():
        p = p / "episodes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def map_level(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        cat = r["category"]
        mid = r["map_id"]
        slot = out.setdefault(cat, {}).setdefault(mid, {"n": 0, **{m: 0.0 for m in METRICS},
                                                       "tier": r["tier"], "exit_any": 0})
        slot["n"] += 1
        slot["exit_any"] = max(slot["exit_any"], int(r["exit"]))
        for m in METRICS:
            slot[m] += float(r.get(m, 0.0))
    for cat, maps in out.items():
        for mid, s in maps.items():
            for m in METRICS:
                s[m] /= max(1, s["n"])
    return out


def iqm(values: list[float]) -> float:
    x = sorted(values)
    if not x:
        return 0.0
    k = max(1, len(x) // 4)
    return float(np.mean(x[k:len(x) - k])) if len(x) > 2 * k else float(np.mean(x))


def bootstrap_ci(values: list[float], n: int = 4000) -> tuple[float, float]:
    x = np.asarray(values, np.float64)
    if len(x) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict]) -> dict:
    maps = map_level(rows)
    cats = {}
    for cat, m in maps.items():
        vals = {k: [s[k] for s in m.values()] for k in METRICS}
        n = len(m)
        cats[cat] = {
            "n_maps": n,
            "tier": next(iter(m.values()))["tier"],
            "exit": float(np.mean(vals["exit"])),
            "exit_ci": bootstrap_ci(vals["exit"]),
            "success_at_k": float(np.mean([s["exit_any"] for s in m.values()])),
            "proxy": float(np.mean(vals["proxy"])),
            "proxy_ci": bootstrap_ci(vals["proxy"]),
            "proxy_iqm": iqm(vals["proxy"]),
            "coverage": float(np.mean(vals["coverage"])),
            "keys": float(np.mean(vals["keys"])),
            "locks": float(np.mean(vals["locks"])),
            "doors": float(np.mean(vals["doors"])),
            "kills": float(np.mean(vals["kills"])),
            "damage": float(np.mean(vals["damage"])),
            "steps": float(np.mean(vals["steps"])),
            "spl": float(np.mean(vals["spl"])),
            "spin": float(np.mean(vals["spin_frac"])),
            "noop": float(np.mean(vals["noop_frac"])),
            "dead": float(np.mean(vals["dead"])),
            "steps": float(np.mean(vals["steps"])),
        }
    # tier-weighted proxy over all maps
    per_map = []
    for cat, m in maps.items():
        w = TIER_WEIGHTS.get(next(iter(m.values()))["tier"], 1.0)
        for s in m.values():
            per_map.append((s["proxy"], w, cat, s["exit"]))
    weighted = sum(p * w for p, w, _, _ in per_map) / max(1e-9, sum(w for _, w, _, _ in per_map))
    # stratified bootstrap over maps
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(3000):
        vals = []
        for cat, m in maps.items():
            arr = list(m.values())
            pick = [arr[i] for i in rng.integers(0, len(arr), len(arr))]
            w = TIER_WEIGHTS.get(arr[0]["tier"], 1.0)
            vals += [(s["proxy"], w) for s in pick]
        boot.append(sum(p * w for p, w in vals) / sum(w for _, w in vals))
    return {
        "n_episodes": len(rows),
        "n_maps": len(per_map),
        "weighted_proxy": float(weighted),
        "weighted_proxy_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "categories": cats,
    }


def markdown(summary: dict, baseline: dict | None = None) -> str:
    lines = ["# Suite report", "",
             f"- episodes: {summary['n_episodes']}",
             f"- maps: {summary['n_maps']}",
             f"- tier-weighted proxy: **{summary['weighted_proxy']:+.3f}** "
             f"(95% CI {summary['weighted_proxy_ci'][0]:+.3f} … {summary['weighted_proxy_ci'][1]:+.3f})",
             ""]
    if baseline:
        d = summary["weighted_proxy"] - baseline["weighted_proxy"]
        lines.append(f"- delta vs baseline: **{d:+.3f}**")
        lines.append("")
    lines += ["| category | maps | exit | success@k | proxy | coverage | keys | locks | kills | dead | SPL | spin | steps |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cat in sorted(summary["categories"]):
        s = summary["categories"][cat]
        lines.append(f"| {cat} | {s['n_maps']} | {s['exit']:.1%} | {s['success_at_k']:.1%} | "
                     f"{s['proxy']:+.2f} | {s['coverage']:.1%} | {s['keys']:.2f} | "
                     f"{s['locks']:.2f} | {s['kills']:.1f} | {s['dead']:.0%} | {s['spl']:.2f} | "
                     f"{s['spin']:.1%} | {s['steps']:.0f} |")
    lines += ["", "Proxy components are assumptions; the official weights are hidden. "
                  "Every category is reported, no cherry-picking."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="")
    args = ap.parse_args()
    rows = load_rows(args.run)
    summary = summarize(rows)
    base = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    out = Path(args.out or Path(args.run) / "summary")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text(markdown(summary, base))
    print(markdown(summary, base))


if __name__ == "__main__":
    main()
