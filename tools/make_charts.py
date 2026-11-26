"""Generate README charts from training/evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def training_chart(run: Path, out: Path) -> None:
    rows = []
    p = run / "metrics.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), dpi=160)
    ax[0].plot(ep, [r["loss"] for r in rows], marker="o", label="total")
    ax[0].plot(ep, [r["ce"] for r in rows], marker="o", label="action CE")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("Behaviour cloning loss")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25)
    ax[1].plot(ep, [r.get("exit_rate", 0) for r in rows], marker="o", color="#b3261e")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("epoch")
    ax[1].set_title("Held-out exit rate (door-free)")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def results_chart(results: Path, out: Path) -> None:
    if not results.exists():
        return
    data = json.loads(results.read_text())
    summary = data.get("summary", {})
    if not summary:
        return
    labels = list(summary)
    exits = [summary[k]["exit"] / max(1, summary[k]["n"]) for k in labels]
    proxy = [summary[k]["proxy"] / max(1, summary[k]["n"]) for k in labels]
    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=160)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, exits, width=0.4, label="exit rate", color="#b3261e")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, proxy, width=0.4, label="proxy score", color="#0969da")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("exit rate")
    ax2.set_ylabel("proxy score")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="training/runs/bc_map2")
    ap.add_argument("--results", default="training/runs/eval/results.json")
    ap.add_argument("--out", default="docs/assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    training_chart(Path(args.run), out / "training.png")
    results_chart(Path(args.results), out / "results.png")


if __name__ == "__main__":
    main()
