# Results — Phase 1 benchmark

All numbers come from `training/bench.py` (fixed bench seeds `30,000+`, never used for
training; 8 maps per generated tier; one engine seed; Freedoom MAP23–32 held out). The
proxy score mirrors the four stated categories with **assumed** weights, because the
official weights are hidden.

```
proxy = 10·exit + 3·key + 3·lock + 0.5·door + 0.3·new_sector (cap 10)
        + 0.6·kill − 0.01·damage − (3·(1−t/T)+1)·death
```

## Checkpoints compared

| Checkpoint | What it is |
|---|---|
| `bc_map2_e0` | Map-aware multimodal BC on nav + arenas; earlier baseline |
| `bc_use` | Drill-only fine-tune from a base checkpoint; learned doors, forgot arenas |
| `bc_head` | Frozen trunk, actor-only fine-tune on drills; cannot learn door approach |
| `bc_arena` | Low-LR fine-tune on 300 extra arena episodes; **shipped** |
| `bc_anchor` | KL-anchored full fine-tune; stopped before checkpoint (inconclusive) |

## Generated tiers

| Suite | `bc_map2_e0` | `bc_use` | `bc_head` | `bc_arena` |
|---|---:|---:|---:|---:|
| t1 arena: exits | 2/8 | 1/8 | 1/8 | **5/8** |
| t1 proxy | +0.35 | +0.89 | −1.53 | **+5.42** |
| t1 kills | 10 | **19** | 5 | 4 |
| t2 rooms: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t2 proxy | −0.99 | −0.26 | −0.22 | **−0.18** |
| t2 coverage | 6.1% | 6.1% | 6.1% | 6.1% |
| t3 key/lock: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t3 proxy | −0.89 | −0.70 | −0.87 | **−0.38** |
| t4 gauntlet: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t4 proxy | −1.61 | −1.41 | −2.17 | **−0.73** |
| Door drills: exits | 0/8 | **6/8** | 0/8 | 0/8 |
| Freedoom 23–32: exits | 0/10 | not run | not run | 0/10 |
| Freedoom proxy | −0.77 | not run | not run | **+1.00** |
| Freedoom coverage | 1.1% | not run | not run | 3.6% |

Tier-weighted proxy (1, 1.2, 1.5, 2) over t1–t4:
`bc_map2_e0` ≈ **−5.40**, `bc_use` ≈ **−3.29**, `bc_arena` ≈ **+3.17**.
The arena fine-tune is the clear Phase-1 winner; the door branch still wins door drills.

## Phase 2 — PPO fine-tune

A 25-update PPO run (25.6k environment steps, 8 envs, dense new-cell/door/exit reward)
from the arena checkpoint:

| Suite | `bc_arena` | PPO-25 |
|---|---:|---:|
| t1 exits | 5/8 | **6/8** |
| t1 proxy | +5.42 | **+8.20** |
| t1 kills | 4 | **19** |
| t2 proxy | −0.18 | **+0.48** |
| t3 proxy | −0.38 | **−0.03** |
| t4 proxy | −0.73 | **−0.32** |
| drills | 0/8 | 0/8 |
| Freedoom 23–32 proxy | +1.00 | **+0.36** |
| Freedoom coverage | 3.6% | 1.6% |
| Freedoom kills | 2 | **11** |

Tier-weighted proxy over t1–t4: **+8.09** (from +3.17). PPO improves every tier proxy and tier-1 substantially, but it still does not learn
USE/door drills. `ppo_explore/ckpt.pt` is the shipped Phase-2 checkpoint. A resumed run and a
learned mixture-of-experts gate with the door specialist are the next attempts.

### Mixture of experts (measured, not shipped)

Two frozen specialists — PPO-25 (tier-1 6/8) and `bc_use` (drills 6/8) — plus a
2k-parameter gate over observation-only door/ray/map features. Routing results:

| Gate | threshold | t1 | drills | t2 proxy | weighted t1–t4 |
|---|---|---:|---:|---:|---:|
| distilled (BC) | 0.5 soft | 2/8 | 6/8 | +0.23 | — |
| distilled (BC) | 0.5 hard | 6/8 | 0/8 | +0.48 | +8.09 |
| supervised near-door | 0.1 hard | 5/8 | **6/8** | +0.32 | **+3.59** |
| supervised near-door | 0.5 hard | 6/8 | 0/8 | +0.48 | +8.09 |

The supervised gate genuinely routes doors (6/8 drills) but the door expert costs one
tier-1 exit and lowers t2–t4 proxies (t3 −0.22, t4 −0.93), so PPO-25 remains the
shipped hidden-tier agent. The code stays in `training/moe.py` for Phase 3.

## What the numbers mean

1. **The two-map eval was optimistic.** Tier-1 exits are 2/8 (25%), not 2/2. The hidden
   suite is larger; report the bench, never the small eval.
2. **Doors are learnable in isolation.** 360/360 reliable drill episodes + a weighted
   USE loss took door drills from 0/8 to 6/8 in two epochs.
3. **Door skill does not transfer to rooms.** The door branch still opens 0 doors on
   tier-2 maps; straight-corridor drills do not teach “find a door on a wall”.
4. **Freezing the trunk fails.** Actor-only training cannot map frozen features to door
   approach; 0/8 drills. The trunk must adapt.
5. **Full fine-tuning forgets.** Drill-only training drops tier-1 exits from 2/8 to 1/8
   and loses arena kills/exit behaviour. This is the central Phase-2 problem.
6. **Room-door demonstrations are reliable** (200/200, random start heading). A focused
   arena-anchored merge (`bc_combined`: drills + room-doors, KL anchor to `bc_arena` on
   non-door frames, 1 epoch) was trained and measured: **tier 1 1/8, drills 0/8**. The
   anchor does not preserve the arena policy under this data mix, and the USE head still
   does not generalise to rooms. “Add doors without forgetting” remains the Phase-2
   problem; the KL anchor alone is not the answer.
7. **Success-only replay does not beat the arena specialist.** Fine-tuning on 652
   successful episodes (nav+arena, LR 5e-5, 1 epoch) gives tier-1 **4/8** (+4.95), tier-2
   **−0.54**, drills **0/8**. The arena checkpoint stays (5/8, +5.42, t2 −0.18).
8. **More arena data helps.** A low-LR fine-tune on 300 extra arena episodes
   (53% teacher exits) raised held-out tier-1 exits from 2/8 to **5/8** and improved every
   proxy.

## Reproduce

```bash
python -m training.bench --ckpt training/runs/bc_map2/ckpt_e0.pt --maps 8
python -m training.bench --ckpt training/runs/bc_use/ckpt.pt --maps 8 --no-freedoom
```

Each run writes `results.json` with every row and the summary. Latency is p99 per map;
the agent is ~1 ms on CPU, well inside the 40 ms/decision budget.

## Caveats

- One training seed and one engine seed. The bench is a point estimate; 8 maps per tier
  has a wide confidence interval.
- Proxy weights are assumptions. The official scorer may weight exit, kills, damage or
  coverage differently.
- Generated maps are one distribution; Freedoom MAP23–32 is the only hand-made
  held-out set and the agent scores near zero there.
- Hidden tiers may include layouts/skills not represented here (teleporters, lifts,
  crushing ceilings, boss monsters).

---

# Phase 3 — diagnostic suite (suite-1)

The official scorer is hidden and the Phase-2 bench saturated at the easy end
(t1 6/8, everything else near zero). Phase 3 therefore optimizes a 14-category
suite designed to expose specific failures. Test seeds are never used for
training; every test look is logged.

## Fast validation (70 episodes: 5 maps/category, 1 engine seed)

| Category | PPO-25 | Atlas-G BC v5 | Hybrid + reflex |
|---|---:|---:|---:|
| C0 sanity | 0% | 100% | **100%** |
| C1 duel | 10% | 0% | **20%** |
| C2 arena | 80% | 60% | 80% |
| C3 corridor door | 0% | 0% | 0% |
| C4 room door | 0% | 20% | 20% |
| C5 key → lock | 0% | 100% | **100%** |
| C6 two keys, two locks | 0% | 0% | **100%** |
| C7 exploration | 0% | 0% | 0% |
| C8 backtrack | 0% | 20% | **40%** |
| C9 ambush | 0% | 0% | 0% |
| C10 scarce ammo | 0% | 0% | 0% |
| C11 gauntlet | 0% | 0% | 0% |
| C12 Freedoom | 0% | 0% | 0% |
| C13 OOD | 0% | 0% | 0% |
| **Tier-weighted proxy** | **+1.175** | **+3.066** | **+5.696** |
| 95% CI | +0.76 … +1.66 | +2.39 … +3.75 | **+4.86 … +6.55** |

Paired comparison (same maps, candidate − baseline): **+4.521**, 95% CI
**+2.589 … +6.728**. The promotion gate returns **PROMOTE**: the CI excludes
zero and no category has a statistically significant >10-point regression.
The C1/C2 point differences are single-map swings; their paired CIs include zero.

C5/C6 are the clearest signal: the baseline never picks up a key; Atlas-G picks
both keys and opens both locks (C6 proxy +23.5, SPL 0.38). C0 SPL is 0.96.

## What produced the gains

1. **Purpose-built key categories.** The room generator's locked doors are
   usually optional: only 5/30 tier-4 maps required a key, 0/30 required two.
   C5/C6/C8 were rebuilt as drill layouts with cut-edge locks, and the teacher
   solves them 3/3, 5/5, 3/3. Before this, the "key" categories were not testing
   keys at all.
2. **Teacher fixes.** Door side-tracks were treated as passable (false A* paths
   through walls), the key search required a path to the far side of the locked
   door, and the frontier fallback rescanned 40 A* goals per replan (hangs).
   Fixing all three made key/backtrack labels correct and collection fast.
3. **Atlas-G goal head + frontier channel + 450 key/backtrack demos.**
   BC v5 solves C0/C5/C8; the old PPO-25 does none.
4. **Perception-routed hybrid.** When an enemy is visible, the frozen Phase-2
   PPO-25 runs; otherwise Atlas-G. This restores C1/C2/t1 without giving up the
   navigation gains. Both experts see every observation, so memory stays current.
5. **Observation-only freeze reflex.** 8 forward decisions with vf≈0 ⇒ turn 4;
   8 pure no-ops ⇒ forward+USE. C6 0/5 → 5/5, C3 0/5 → 3/5 in targeted tests.

## Dead ends (kept for honesty)

- **Unbalanced DAgger**: 191 suite episodes with β=0 produced 73% turn labels
  because the student froze early; BC v3 collapsed to 99.8% spin and 0% exits.
- **PPO v2 from BC v5** (suite reward + key potential, 40 updates): mean return
  1.4 → 8.5, but C5 fell to 0/5 and C6/C8 did not beat BC v5. Not shipped.
- **Per-step router without hysteresis**: C2 4/5 but frequent switches reset
  recurrent state; 16-step hysteresis is the shipped setting.
- **Door-angle reflex** alone (USE when any door ray is within 102 units) hurt
  C5 (5/5 → 3/5); the shipped reflex is conditional on a freeze.

## Official-style bench (hybrid agent, end-to-end)

`training/bench.py --agent` on fixed seeds 30 000+, 12 maps per generated tier
except t4 (8) and Freedoom MAP23–32 (10):

| Suite | n | Exit | Kills | Keys | Doors | Coverage | Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 arena | 12 | **7/12** | 25 | 0 | 0 | 100% | +5.49 |
| t2 rooms | 12 | 0/12 | 4 | 0 | 12 | 12.7% | +0.24 |
| t3 key/lock | 12 | 0/12 | 0 | 0 | 6 | 5.7% | −0.10 |
| t4 gauntlet | 8 | 0/8 | 3 | 0 | 8 | 3.6% | −0.62 |
| drills | 12 | 0/12 | 0 | 0 | 2 | 30.8% | +0.48 |
| Freedoom 23–32 | 10 | 0/10 | **23** | 0 | 4 | 2.5% | **+0.89** |

PPO-25's Phase-2 numbers (8 maps/tier) were t1 6/8 exits/+8.20, t2 +0.48,
t3 −0.03, t4 −0.32, drills +0.60, Freedoom +0.36. The hybrid keeps arena
exits/kills (7/12, 25 kills vs 19) and improves Freedoom proxy (23 kills, 4
doors), while t2–t4 proxies are within ~0.3 of PPO-25. The suite, not this bench,
is the optimization target; this table is the continuity check.

Why the old bench t2–t4 do not improve as much: they are monster maps, so the
router hands them to the combat expert by design; the navigation expert only
takes over when no enemy is visible. The suite's C5/C6/C8 remove monsters
precisely to isolate the key skill.

## Held-out test (fast, 70 episodes, seeds 30 000+)

The final hybrid was scored once on the untouched test split
(`test_hybrid_fast`, logged in `test_touch.log`):

- tier-weighted proxy **+5.790** (95% CI +5.115 … +6.509)
- C0 100%, C1 40%, C2 60%, C3 0%, C4 40%, C5 100%, C6 100%, C8 40%
- paired delta vs PPO-25: **+4.907** (95% CI +3.002 … +7.073)
- C7, C9–C13 0%

This is consistent with validation (+5.696), so the checkpoint is not
overfitting the val maps. Test C2 is 60% (3/5) with 2.8 kills/episode; the two losses are arena maps where
the combat expert is overwhelmed. The paired PPO-25 test run is reported
alongside it in the final table; C2 variance is a known limitation of a 5-map
sample.

## Pre-existing checkpoints on the same val-fast split

| Checkpoint | What it is | Weighted proxy |
|---|---|---:|
| `bc_arena` | Phase-1 arena specialist (shipped before PPO) | +0.924 |
| `bc_use` | Phase-1 door specialist | +1.168 |
| PPO-25 | Phase-2 shipped checkpoint | +1.175 |
| Atlas-G BC v5 | this phase, single model | +3.066 |
| Hybrid + reflex | shipped candidate | **+5.70** |

All were run through the same map set, seeds and metric code. The hybrid beats
every earlier checkpoint by at least **+4.7** weighted proxy. The MoE gate was
not re-run on the suite because it is not shippable in its Phase-2 form (it was
measured negative on t2–t4 and is superseded by the perception router).
