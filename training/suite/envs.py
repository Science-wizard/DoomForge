"""Curriculum sampling from the suite train split.

The suite is not just a scoreboard: PPO/DAgger draw training maps from it with a
stage-aware mix (mastered / current / hardest, from the approved ladder).  The
test split is never read here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class SuiteSampler:
    def __init__(self, manifest: str | Path = "training/data/suite/train/manifest.jsonl",
                 config: str | Path = "training/suite/configs/ladder.json",
                 seed: int = 0, suite_prob: float = 0.6,
                 categories: list[str] | None = None):
        self.rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
        if categories:
            self.rows = [r for r in self.rows if r["category"] in categories]
        self.by_cat: dict[str, list[dict]] = {}
        for r in self.rows:
            self.by_cat.setdefault(r["category"], []).append(r)
        cfg = json.loads(Path(config).read_text())
        cur = cfg["curriculum"]
        self.stages = {k: [c for c in cur[k] if c in self.by_cat] for k in ("early", "mid", "late")}
        self.mix = cur["mix"]
        self.suite_prob = suite_prob
        self.rng = random.Random(seed)

    def _stage(self, progress: float) -> str:
        if progress < 0.25:
            return "early"
        if progress < 0.6:
            return "mid"
        return "late"

    def _pick(self, cats: list[str]) -> dict:
        # Weak categories first: weight by tier so later, harder maps are not starved.
        weights = [1.0 + 0.35 * (self.by_cat[c][0]["tier"] - 1) for c in cats]
        cat = self.rng.choices(cats, weights=weights, k=1)[0]
        return self.rng.choice(self.by_cat[cat])

    def sample(self, progress: float) -> tuple[str, str, int, str]:
        if self.rng.random() > self.suite_prob:
            return "", "", 0, ""
        stage = self._stage(progress)
        r = self.rng.random()
        if r < self.mix["mastered"]:
            cats = self.stages["early"]
        elif r < self.mix["mastered"] + self.mix["current"]:
            cats = self.stages[stage] or self.stages["early"]
        else:
            cats = self.stages["late"] or self.stages[stage]
        row = self._pick(cats)
        return row["wad"], row["map_name"], int(row["ticks"]), row["category"]
