# Results

The official scorer is hidden. These numbers come from our 14-category
diagnostic suite ([`../training/suite/README.md`](../training/suite/README.md)) using
the proxy score below. They are not leaderboard claims.

```
proxy = 10·exit + 3·keys + 3·locks + 0.5·doors + min(0.3·sectors, 10) + 0.6·kills
        − 0.01·damage − (3·time-left + 1)·died
headline = tier-weighted mean over categories (tier 1/2/3/4 → 1.0/1.2/1.5/2.0),
           stratified bootstrap 95% CI over maps
```

The reference agent is the previous DoomForge design (three networks plus a
router and reflex rules), run on identical maps for paired comparisons.

## Held-out test split

140 episodes (10 maps per category, engine seed 0), one logged look
(`test_touch.log`). The test maps are never used for training or selection.

| Category | Tier | Reference | **Atlas-E** | Atlas-E coverage | Atlas-E deaths |
|---|---:|---:|---:|---:|---:|
| C0 sanity | 1 | 100% | **100%** | 100% | 0% |
| C1 duel | 1 | 20% | **90%** | 100% | 0% |
| C2 arena | 1 | 60% | 50% | 100% | 50% |
| C3 corridor door | 1 | 100% | **100%** | 100% | 0% |
| C4 room door | 1 | 30% | **100%** | 73% | 0% |
| C5 key → lock | 3 | 100% | 80% | 77% | 0% |
| C6 two keys, two locks | 4 | 100% | **100%** | 100% | 0% |
| C7 exploration (8–13 rooms) | 2 | 0% | **10%** | 40% | 0% |
| C8 backtrack | 3 | 30% | **100%** | 100% | 0% |
| C9 ambush | 3 | 0% | 0% | 28% | 40% |
| C10 scarce ammo | 4 | 0% | 0% | 16% | 10% |
| C11 gauntlet | 4 | 0% | 0% | 11% | 20% |
| C12 Freedoom | 4 | 0% | 0% | 3% | 70% |
| C13 OOD | 4 | 0% | 0% | 17% | 30% |
| **Tier-weighted proxy** | | +6.11 | **+8.48** | | |

Paired difference: **+2.37 (95% CI +1.47…+3.24)**. Coverage on the hard
categories (C9/C10/C13) is 3–4× the reference's (7% / 5% / 6%).

## Validation

Full validation split, two engine seeds (272 episodes):

| | Reference | Atlas-E |
|---|---:|---:|
| Tier-weighted proxy | +6.11 | **+8.28** |
| Paired difference | | **+2.17 (95% CI +1.23…+3.08)** |

Per category (exit rate): C0 100, C1 95, C2 50, C3 100, C4 100, C5 100, C6 70,
C7 20, C8 100, C9–C13 0. On C6, 3 of the 10 validation layouts fail (on both
seeds): the agent holds both keys but stops ~130 units short of the second lock.
On test, C6 is 10/10.

## Effect of individual design choices

Measured during development on the 70-episode fast validation split unless noted.

| Change | Effect |
|---|---|
| Corrected dead reckoning (turn-run + friction + wall clip) | map drift after 500 decisions: 431 → 31 units, heading 25° → 0.02° |
| Heading-up two-scale ego map vs world-aligned map (same BC data) | BC alone solves C3 (80%) and C6 (100%) without any routing |
| Seeded sampling vs argmax (same weights) | +6.38 → +7.80; removes freeze-at-tie episodes |
| Single PPO network vs routing to a separate combat expert | routing lowers C2 (80% → 20%) |
| Seen-floor and stagnation shaping in PPO | C7 training exits 3% → 12–15%; first validation exits on C7 |
| Heavier death/damage cost | arena deaths 8/10 → 5/10 on validation |

## Runtime

Official validator (`validate.py`): **PASS**, init 0.4 s, decision latency
p50 1.6 ms / p99 1.9 ms / max 3 ms on one CPU thread (budget 40 ms average).
The contract stress test (`tools/test_agent_contract.py`: NaN/inf/mask
corruption, determinism) passes. On the validator's single dry-run map
(`two_rooms.wad`, one monster), the seeded episode ends with the agent killed at
step 301. The validator still passes, because it checks the contract and timing, not the outcome.

## Known limitations

- **No exits on C9–C13** (ambush, scarce ammo, gauntlet, Freedoom, OOD). The
  agent explores, fights and opens doors there for partial credit, but these maps
  combine long exploration with dense combat. PPO ran for ~1.6M steps on a laptop;
  this skill class needs far more samples.
- **Arena survival.** 50% of C2 test arenas end in death.
- **Small per-category samples.** 10 maps per category, and the drill categories
  share a layout family, so single categories can swing 20–30 points between
  checkpoints. The headline CIs account for this; individual category numbers
  should be read with that in mind.
- **Proxy weights are assumptions.** Components are reported separately so they
  can be re-weighted.

## Reproduce

```bash
uv run python -m training.suite.run --agent --split val --per-category 10 --seeds 2 \
    --workers 8 --out training/runs/suite/val_check
uv run python -m training.suite.metrics --run training/runs/suite/val_check
uv run python -m training.suite.compare --baseline <reference run> --candidate training/runs/suite/val_check
```
