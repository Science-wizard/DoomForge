# Evaluation and results

The official scorer is hidden, so DoomForge is measured on its own 14-category
diagnostic suite (`training/suite/`). It is harder and more diagnostic than the
validator, and it is also the PPO curriculum. Numbers are proxy scores, not
leaderboard claims.

## How we measure

### Categories

| ID | Tier | Ticks | What it isolates |
|---|---:|---:|---|
| C0 sanity | 1 | 2100 | walk a clear corridor to a visible exit |
| C1 duel | 1 | 2100 | 1v1 aim, strafe, dodge |
| C2 arena | 1 | 2100 | 3–6 monsters, cover, ammo |
| C3 corridor door | 1 | 2100 | straight-line door USE |
| C4 room door | 1 | 2100 | door on a room wall, random start pose |
| C5 key → lock | 3 | 3150 | one key, one locked door on the only path |
| C6 two keys | 4 | 4200 | key → red door → key → blue door |
| C7 exploration | 2 | 4200 | 8–13 rooms, hidden exit |
| C8 backtrack | 3 | 4200 | key behind the spawn, lock ahead |
| C9 ambush | 3 | 4200 | monsters added in the exit room |
| C10 scarce ammo | 4 | 4200 | ammo stripped, many monsters |
| C11 gauntlet | 4 | 4200 | all skills combined |
| C12 Freedoom | 4 | 4200 | hand-made maps (transfer) |
| C13 OOD | 4 | 4200 | larger maps, rotated start pose |

C5, C6 and C8 use purpose-built drill layouts, so the lock is on the only path.
C0/C3 start facing the exit and C1 starts facing its monster.

### Splits and leakage

| Split | Seeds | Freedoom | Use |
|---|---|---|---|
| train | 0+ (150 maps/category, 1,966 total) | MAP01–16 | PPO maps, demos |
| val | 10 000+ (10 maps/category, 136 total) | MAP17–22 | selection |
| test | 30 000+ (30 maps/category) | MAP23–32 | one logged look per shipping decision |
| ood | 40 000+ | | distribution-shift probe |

Every WAD is hashed; train∩val and train∩test are empty
(`python -m training.suite.leakage`). Test/OOD runs require `--allow-test` and
append to `test_touch.log`.

### Score

```
proxy    = 10·exit + 3·keys + 3·locks + 0.5·doors + min(0.3·sectors, 10) + 0.6·kills
           − 0.01·damage − (3·time-left + 1)·died
headline = tier-weighted mean over categories (tiers 1/2/3/4 → 1.0/1.2/1.5/2.0),
           stratified bootstrap 95% CI over maps
```

The weights are our assumption about the hidden scorer. Every component (exit,
coverage, keys, locks, kills, damage, deaths, steps, SPL, spin/no-op fraction)
is reported per category, so the score can be re-weighted.

### Promotion gate

1. The **paired** bootstrap CI of the headline difference against the current
   agent (same maps, per-map differences) must lie above zero.
2. No category may regress by more than 10 points with a CI excluding zero.
3. Finalists are re-scored on two engine seeds. The winner gets one logged
   look at the test split.

Every PPO snapshot is scored automatically on validation by `tools/auto_eval.py`.
`training/suite/probes.py` adds diagnostics a single score hides: door-angle sweep,
exit-sign ablation, door-colour swap, observation corruption, determinism, and
memory wipe at key pickup.

## Results

The reference is our earlier design (three networks plus a router and reflex
rules), run on identical maps for paired comparisons.

### Held-out test split

140 episodes (10 maps/category, engine seed 0), never used for training or
selection.

| Category | Reference | **DoomForge** | DoomForge coverage | DoomForge deaths |
|---|---:|---:|---:|---:|
| C0 sanity | 100% | **100%** | 100% | 0% |
| C1 duel | 20% | **90%** | 100% | 0% |
| C2 arena | 60% | 50% | 100% | 50% |
| C3 corridor door | 100% | **100%** | 100% | 0% |
| C4 room door | 30% | **100%** | 73% | 0% |
| C5 key → lock | 100% | 80% | 77% | 0% |
| C6 two keys | 100% | **100%** | 100% | 0% |
| C7 exploration | 0% | **10%** | 40% | 0% |
| C8 backtrack | 30% | **100%** | 100% | 0% |
| C9 ambush | 0% | 0% | 28% | 40% |
| C10 scarce ammo | 0% | 0% | 16% | 10% |
| C11 gauntlet | 0% | 0% | 11% | 20% |
| C12 Freedoom | 0% | 0% | 3% | 70% |
| C13 OOD | 0% | 0% | 17% | 30% |
| **Tier-weighted proxy** | +6.11 | **+8.48** | | |

Paired difference **+2.37 (95% CI +1.47…+3.24)**. On C9/C10/C13 DoomForge covers
3–4× more of the map than the reference (7% / 5% / 6%).

### Validation (two engine seeds, 272 episodes)

**+8.28** vs +6.11, paired **+2.17 (95% CI +1.23…+3.08)**. Exits: C0 100,
C1 95, C2 50, C3 100, C4 100, C5 100, C6 70, C7 20, C8 100, C9–C13 0. On 3 of
the 10 C6 validation layouts the agent holds both keys but stops ~130 units
short of the second lock. On test, C6 is 10/10.

### What each design choice contributed

| Change | Measured effect |
|---|---|
| Corrected dead reckoning (turn-run, per-tic friction, wall clip) | drift after 500 decisions on the agent's own episodes 225 → 99 units (random-action test drive: 431 → 31); heading 25° → 0.02° |
| Heading-up two-scale ego map | BC alone solves C3 (80%) and C6 (100%) with no routing |
| Seeded sampling instead of argmax (same weights) | +6.38 → +7.80 (fast val); no freeze-at-tie episodes |
| One PPO network instead of routing to a combat expert | routing cut C2 from 80% to 20% |
| Seen-floor + stagnation shaping | C7 training exits 3% → 12–15%; first validation exits on C7 |
| Heavier death/damage cost | arena deaths 8/10 → 5/10 on validation |

### Runtime

Official validator: **PASS**. Init 0.4 s; p50 1.6 ms, p99 2.0 ms per decision on
one CPU thread (budget: 40 ms average). The contract stress test (NaN/inf/mask
corruption, determinism) passes. On the validator's single dry-run map
(`two_rooms.wad`), the seeded episode ends with the agent killed at step 301.
The validator checks the contract and timing, not the outcome.

### Limitations

Full-length footage of every category, including the failures below, is in
[`GAMEPLAY.md`](GAMEPLAY.md).

- **No exits on C9–C13.** The agent earns partial credit there (coverage,
  kills, doors, keys), but these maps combine long exploration with dense combat
  and need far more than the ~1.6M PPO steps a laptop allowed.
- **Arena survival.** Half of the C2 test arenas end in death.
- **Small per-category samples.** 10 maps per category, and drill categories
  share a layout family, so single categories swing between checkpoints. The
  headline CIs account for this.

### Stage D (2026-11-27): footage-driven shaping, first gauntlet exits

From replay review: PPO fine-tune (30 updates, ~140k steps) from the shipped
net with three new training-time costs: re-entering a well-trodden 64-unit cell
(circling), any non-door ray under ~24 units (wall hugging), and extra damage
cost per additional visible enemy (fighting in the open instead of isolating
one enemy at a time).

| | Previous | **Stage D (u30 EMA)** |
|---|---:|---:|
| Val, 2 seeds (272 ep) | +8.28 | **+9.05** (paired +0.77, CI −0.09…+1.60) |
| Test (140 ep) | +8.48 | **+8.61** (paired +0.13, CI −0.76…+0.99) |
| C11 gauntlet exits, val / test | 0% / 0% | **10% / 10%** (C11 proxy +4.0 / +3.6) |
| C10 scarce-ammo exits, val | 0% | 10% |
| C2 arena exits, val / test | 50% / 50% | 85% / 60% |
| C5 key→lock exits, val / test | 100% / 80% | 70% / 70% |

Later snapshots (u40, u50) and a 50/50 soup with the previous net scored lower.
The headline CIs include zero; the gauntlet gain is consistent across splits.

Teacher audit: with full map access the scripted teacher itself only exited
C11 1/10 and C9 0/10 on val (4/10 and 7/10 in god mode). Two deadlocks were
fixed: the door reflex idled through its USE cooldown (35 presses, no
movement), and the explore fallback picked only cells behind locked doors.
The next demo collection should use the fixed teacher.

## Commands

```bash
uv run python -m training.suite.generate --split val --n 10        # build a split (idempotent)
uv run python -m training.suite.run --agent --split val --per-category 10 --seeds 2 \
    --workers 8 --out training/runs/suite/val_check                # shipped agent
uv run python -m training.suite.metrics --run training/runs/suite/val_check
uv run python -m training.suite.compare --baseline <run> --candidate training/runs/suite/val_check
uv run python -m training.suite.probes --ckpt submission/weights/doomforge.pt
```
