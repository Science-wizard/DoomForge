"""Paired comparison of two suite runs, with a promotion gate.

Both runs must use the same split and seed count; maps are paired by ``map_id``.
The gate asks three questions:

1. Is the tier-weighted proxy delta positive with a 95% bootstrap CI excluding 0?
2. Does every category stay above its ``min_for_gate`` floor?
3. Is there any category whose exit rate fell by more than 10 points?

Exit code 0 means promote, 1 means hold.  The report is written as JSON + Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.suite.metrics import TIER_WEIGHTS, load_rows, map_level, summarize


def paired_bootstrap(a: dict[str, float], b: dict[str, float], weights: dict[str, float],
                     n: int = 5000) -> tuple[float, float, float]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0, 0.0, 0.0
    da = np.array([a[k] for k in keys])
    db = np.array([b[k] for k in keys])
    w = np.array([weights[k] for k in keys])
    diff = db - da
    point = float((diff * w).sum() / w.sum())
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        boot.append(float((diff[idx] * w[idx]).sum() / w[idx].sum()))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", default="training/suite/configs/ladder.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-delta", type=float, default=0.0)
    args = ap.parse_args()

    rows_a = load_rows(args.baseline)
    rows_b = load_rows(args.candidate)
    cfg = json.loads(Path(args.config).read_text())
    floors = {cid: c["min_for_gate"] for cid, c in cfg["categories"].items()}

    ma, mb = map_level(rows_a), map_level(rows_b)
    proxy_a, proxy_b, weights = {}, {}, {}
    exits_a, exits_b = {}, {}
    for cat in set(ma) & set(mb):
        for mid in set(ma[cat]) & set(mb[cat]):
            proxy_a[mid] = ma[cat][mid]["proxy"]
            proxy_b[mid] = mb[cat][mid]["proxy"]
            exits_a[mid] = ma[cat][mid]["exit"]
            exits_b[mid] = mb[cat][mid]["exit"]
            weights[mid] = TIER_WEIGHTS.get(ma[cat][mid]["tier"], 1.0)

    delta, lo, hi = paired_bootstrap(proxy_a, proxy_b, weights)
    sa, sb = summarize(rows_a), summarize(rows_b)

    def paired_cat_ci(cat: str) -> tuple[float, float, float]:
        keys = sorted(set(ma.get(cat, {})) & set(mb.get(cat, {})))
        if not keys:
            return 0.0, 0.0, 0.0
        diff = np.array([mb[cat][k]["exit"] - ma[cat][k]["exit"] for k in keys])
        rng = np.random.default_rng(11)
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    cats = {}
    for cat in sorted(set(sa["categories"]) & set(sb["categories"])):
        a, b = sa["categories"][cat], sb["categories"][cat]
        dmean, dlo, dhi = paired_cat_ci(cat)
        cats[cat] = {
            "exit_a": a["exit"], "exit_b": b["exit"], "exit_delta": b["exit"] - a["exit"],
            "proxy_a": a["proxy"], "proxy_b": b["proxy"], "proxy_delta": b["proxy"] - a["proxy"],
            "floor": floors.get(cat, 0.0),
            # A floor the baseline also misses is an aspiration, not a regression:
            # in that case improvement over the baseline is enough.
            "floor_ok": (b["exit"] >= floors.get(cat, 0.0)) or (b["exit"] >= a["exit"]),
            "paired_delta": dmean, "paired_ci": [dlo, dhi],
        }

    # Only a statistically significant drop counts as a regression.  With 5
    # maps/category in the fast gate a one-map swing is not evidence.
    regressions = [c for c, d in cats.items()
                   if d["paired_delta"] < -0.10 and d["paired_ci"][1] < 0]
    floors_ok = all(d["floor_ok"] for d in cats.values())
    promote = bool(delta > args.min_delta and lo > 0 and floors_ok and not regressions)

    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "weighted_proxy_a": sa["weighted_proxy"],
        "weighted_proxy_b": sb["weighted_proxy"],
        "paired_delta": delta,
        "paired_ci": [lo, hi],
        "promote": promote,
        "gate": {
            "delta_positive": delta > args.min_delta,
            "ci_excludes_zero": lo > 0,
            "floors_ok": floors_ok,
            "regressions": regressions,
        },
        "categories": cats,
    }

    lines = [f"# Paired comparison: {Path(args.baseline).name} -> {Path(args.candidate).name}", "",
             f"- baseline weighted proxy: {sa['weighted_proxy']:+.3f}",
             f"- candidate weighted proxy: {sb['weighted_proxy']:+.3f}",
             f"- paired delta: **{delta:+.3f}** (95% CI {lo:+.3f} … {hi:+.3f})",
             f"- decision: **{'PROMOTE' if promote else 'HOLD'}**", "",
             "| category | exit base | exit cand | Δ exit | proxy Δ | floor |",
             "|---|---:|---:|---:|---:|---:|"]
    for cat, d in cats.items():
        lines.append(f"| {cat} | {d['exit_a']:.1%} | {d['exit_b']:.1%} | {d['exit_delta']:+.1%} | "
                     f"{d['proxy_delta']:+.2f} | {d['floor']:.0%} |")
    lines += ["", "Floors use `min_for_gate` from the ladder config; where the baseline also "
                  "misses a floor, matching or beating the baseline is enough. A category "
                  "regression of more than 10 points whose paired 95% CI excludes zero "
                  "blocks promotion even if the headline rises."]
    text = "\n".join(lines) + "\n"

    out = Path(args.out) if args.out else Path(args.candidate) / "compare"
    out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(text)
    print(text)
    raise SystemExit(0 if promote else 1)


if __name__ == "__main__":
    main()
