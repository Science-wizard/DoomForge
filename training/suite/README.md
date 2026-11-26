# DoomForge diagnostic suite (suite-1)

The official validator answers one question: does the agent boot, move, and stay
inside the time budget? It does not measure whether the agent can *play*. This
suite is the development target for Atlas-G. It is deliberately harder than the
validator, covers fourteen distinct capabilities, and is used both as a scoreboard
and as a curriculum source for training.

## Splits

| split | seeds | use |
|---|---|---|
| train | 0+ | BC/DAgger/PPO map source (never scored) |
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

C5, C6 and C8 use purpose-built drill layouts rather than the room generator.
The room generator's locked doors are often optional side doors (measured: only
5/30 tier-4 maps actually required a key, 0/30 required two), which makes a key
category meaningless. The drill layouts guarantee the lock is a cut edge:
C5 = key alcove → locked door → exit; C6 = key → red door → key → blue door;
C8 = key alcove behind the spawn → locked door ahead. The privileged teacher
solves all three 3/3; the old room-generated key maps were 0/3.

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

- **Fast**: 5 maps/category, 1 seed, 70 episodes. Used after every training run.
- **Full**: 30 maps/category × 5 seeds. Used to promote a checkpoint.
- Promotion requires the paired CI on the tier-weighted proxy to exclude zero
  versus the shipped checkpoint, plus no category regression below its
  `min_for_gate` in `configs/ladder.json`.

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
python -m training.suite.run --ckpt training/runs/bc_v2/ckpt.pt \
    --split val --workers 6 --fast
python -m training.suite.metrics --run training/runs/suite/val_bc_v2

# leakage check
python -m training.suite.leakage \
    --train training/data/suite/train/manifest.jsonl \
    --test training/data/suite/test/manifest.jsonl

# probes
python -m training.suite.probes --ckpt training/runs/bc_v2/ckpt.pt
```

## How the suite drives training

`training/suite/envs.py` samples PPO maps from the train split with the approved
30/50/20 mastered/current/hardest mix (`ppo.py --suite`). DAgger runs
`--source suite` so labels are collected on the same category distribution,
including the failure states the suite exposes. The fast val run after each
training stage decides whether the checkpoint advances; the test split is only
touched for a promotion decision.

## Baseline snapshot (val fast, 70 episodes)

| category | PPO-25 | Atlas-G | Hybrid + reflex |
|---|---:|---:|---:|
| C0 sanity | 0% | 100% | 100% |
| C1 duel | 10% | 0% | 20% |
| C2 arena | 80% | 60% | 80% |
| C3 corridor door | 0% | 0% | 0% |
| C4 room door | 0% | 20% | 20% |
| C5 key → lock | 0% | 100% | 100% |
| C6 two keys | 0% | 0% | 100% |
| C7 exploration | 0% | 0% | 0% |
| C8 backtrack | 0% | 20% | 40% |
| weighted proxy | +1.18 | +3.07 | **+5.70** |

Paired delta vs PPO-25: **+4.52** (95% CI +2.59…+6.73), gate **PROMOTE**; test +5.79.
These are validation numbers; test is scored once and logged in `test_touch.log`.
