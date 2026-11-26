# DoomForge diagnostic suite (suite-1)

The official validator answers one question: does the agent boot, move, and stay
inside the time budget? It does not measure whether the agent can *play*. This
suite is DoomForge's development target. It is deliberately harder than the
validator, covers fourteen distinct capabilities, and serves both as the
scoreboard and as the PPO curriculum.

## Splits

| split | seeds | use |
|---|---|---|
| train | 0+ | PPO/demo map source, 150 maps per category (never scored) |
| val | 10 000+ | model selection, fast iteration |
| test | 30 000+ | final comparison only, every look logged |
| ood | 40 000+ | distribution-shift probe |

Test and OOD runs require `--allow-test` and append a line to `test_touch.log`.
Leakage is checked by hashing every WAD (`training/suite/leakage.py`).

## Categories

| id | tier | ticks | what it isolates |
|---|---|---|---|
| C0_sanity | 1 | 2100 | walk a clear corridor to a visible exit |
| C1_duel | 1 | 2100 | 1v1 aim, strafe, dodge |
| C2_arena | 1 | 2100 | 3–6 monsters, cover, ammo |
| C3_corridor_door | 1 | 2100 | straight-line door USE |
| C4_room_door | 1 | 2100 | door on a room wall, random pose |
| C5_key_lock | 3 | 3150 | one key, one locked door |
| C6_two_key | 4 | 4200 | two keys, two locks, ordering |
| C7_exploration | 2 | 4200 | 8–13 rooms, hidden exit |
| C8_backtrack | 3 | 4200 | key far from start, long return |
| C9_ambush | 3 | 4200 | monsters added in the exit room |
| C10_resource | 4 | 4200 | scarce ammo, many monsters |
| C11_gauntlet | 4 | 4200 | all skills combined |
| C12_freedoom | 4 | 4200 | hand-made Freedoom transfer |
| C13_ood | 4 | 4200 | 36-cell maps, rotated start pose |

Maps are generated deterministically (generator version `suite-1`) and filtered
per category: C7 accepts only 8–13 rooms; C0/C3 orient the start toward the
exit; C1 orients toward its single monster. C12 uses disjoint Freedoom maps per
split (train MAP01–16, val MAP17–22, test MAP23–32).

C5, C6 and C8 use purpose-built drill layouts rather than the room generator,
because the room generator's locked doors are often optional side doors. The
drill layouts guarantee the lock is on the only path: C5 = key alcove → locked
door → exit; C6 = key → red door → key → blue door; C8 = key alcove behind the
spawn → locked door ahead.

## Metrics

Per episode: exit, proxy score, coverage, keys, locks, doors, kills, damage,
steps, SPL, spin fraction, no-op fraction, path length.

Per category: exit rate, success@k (≥1 exit on a map across engine seeds),
proxy mean with bootstrap CI, IQM, plus the raw components. The headline is the
**tier-weighted proxy** (weights 1.0/1.2/1.5/2.0) with a stratified bootstrap CI.
No category is dropped or cherry-picked.

The proxy weights are our assumption because the organizers' weights are hidden;
the components are reported separately so a reader can re-weight them.

## Gates

- **Fast**: 5 maps/category, 1 seed, 70 episodes (quick reads only; single
  categories swing a lot).
- **Selection**: full validation split, 10 maps/category (136 episodes); every
  PPO snapshot is scored automatically by `tools/auto_eval.py`.
- **Promotion**: the paired bootstrap CI of the tier-weighted proxy difference vs
  the current agent must exclude zero, and no category may regress by more than
  10 points with a CI excluding zero (`compare.py`, floors in
  `configs/ladder.json`). Finalists are re-scored on two engine seeds.
- **Test**: one logged look for the promoted candidate.

## Probes

`training/suite/probes.py` runs diagnostics that a single score hides:

- door-angle sweep (8 angles × 3 maps): is USE direction-dependent?
- exit-sign ablation: is the exit found by the sign or by layout?
- door-colour swap: does the policy use lock colour or geometry?
- observation corruption (ray holes, dim RGB, half-res): graceful degradation?
- determinism: same checkpoint + seed ⇒ identical action hash.
- memory wipe at key pickup: does success depend on the recurrent state?

## Commands

```bash
# build splits (idempotent)
python -m training.suite.generate --split train --n 20
python -m training.suite.generate --split val --n 10
python -m training.suite.generate --split test --n 30

# evaluate a checkpoint (test needs --allow-test)
python -m training.suite.run --agent --split val --per-category 10 --workers 8 \
    --out training/runs/suite/val_check
python -m training.suite.metrics --run training/runs/suite/val_check

# leakage check
python -m training.suite.leakage \
    --train training/data/suite/train/manifest.jsonl \
    --test training/data/suite/test/manifest.jsonl

# probes
python -m training.suite.probes --ckpt submission/weights/doomforge.pt
```

## How the suite drives training

`training/ppo2.py` draws 85% of its maps from the train split, weighting each
category by 0.6 + its recent failure rate, and 15% from freshly generated
maps. Demonstrations for door/key skills come from the same layouts
(`collect.py --source suite`). Validation decides which snapshot advances; the
test split is touched once per shipping decision.

## Current agent (test split, 140 episodes)

| category | exit | category | exit |
|---|---:|---|---:|
| C0 sanity | 100% | C7 exploration | 10% |
| C1 duel | 90% | C8 backtrack | 100% |
| C2 arena | 50% | C9 ambush | 0% |
| C3 corridor door | 100% | C10 scarce ammo | 0% |
| C4 room door | 100% | C11 gauntlet | 0% |
| C5 key → lock | 80% | C12 Freedoom | 0% |
| C6 two keys | 100% | C13 OOD | 0% |

Tier-weighted proxy **+8.48** (reference design +6.11, paired +2.37, CI
+1.47…+3.24). Details: [`../../docs/RESULTS.md`](../../docs/RESULTS.md).
