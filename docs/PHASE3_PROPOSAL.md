# Phase 3 Proposal — Goal-Conditioned Memory Agent + Comprehensive Suite

**Status:** proposal for approval. No code has been changed for this phase.
**Companion:** [`../research.md`](../research.md) (19.7k-word literature/engineering review),
[`../docs/RESULTS.md`](RESULTS.md) (measured baselines), [`../PLAN.md`](../PLAN.md).

---

## 1. What the evidence says

The current PPO-25 agent is a strong tier-1 specialist but not a general agent. The
measured failures are consistent across checkpoints:

| Failure | Measurement |
|---|---|
| The teacher’s global path is not observable | BC turn accuracy 43.2% (chance 33.3); exact six-head 20.8% |
| No goal means no exploration | tier-2 coverage 6.1%; tiers 2–4 exits 0/8 |
| Doors do not transfer | 6/8 straight drills, **0/8 room doors** |
| Exit is invisible to structured obs | 0/10 Freedoom exits; aux head only fires when the sign is in view |
| Dead reckoning drifts | 16° heading / 361 units over 250 steps after the turn-table fix |
| Specialist fine-tuning forgets | drill-only: tier-1 2/2 → 0/2; KL/success-only/MoE all failed to combine |

The encoder and GRU are not the problem. PPO already learned combat and arena exits
(6/8, 19 kills). The missing pieces are a **learned goal**, a **usable map**, and a
**data distribution that teaches local goal-following** rather than a global route.

---

## 2. Options considered

| # | Option | What changes | Expected gain | Risk | Verdict |
|---|---|---|---|---|---|
| A | More PPO + curriculum only | Longer PPO, tier-balanced sampler | +0–1 tier-2 exits | Low | Not enough |
| B | B+ lite v2 (fixed frontier selector) | 64 u/cell map, scan match, tokens, goal vector from a fixed nearest-frontier selector, asymmetric critic, BC/DAgger/PPO | +2–4 tier-2 exits; coverage 15–25% | Medium | Good, but the selector is a heuristic and the goal is still not learned |
| **C** | **Atlas-G: learned goal manager + B+ core** | B+ core + a **learned manager** over event tokens/frontiers/exit cue that outputs a goal vector; BC from teacher goal labels, then PPO; asymmetric critic; BC/DAgger/PPO | **+3–5 tier-2/3 exits; coverage 20–30%** | Medium-high | **Recommended** |
| D | Full options/feudal hierarchy | Manager + worker trained with option-critic/HIRO | +2–4 if it trains | High | Too fragile for 12 h |
| E | Model-based Dreamer / transformer / MoE | World model, long-context transformer, router | Uncertain / measured negative | High | Reject (research §4/§6) |

The MoE is already measured: it routes drills (6/8) but costs tier-1 and lowers t2–t4.
More PPO alone is cheap but has a ceiling; the agent still has no goal.

**Choice: C — Atlas-G.** It keeps the proven B+ improvements (map, scan match, tokens,
asymmetric critic, balanced data) and adds the one missing architectural idea: a *learned
goal*. The manager is a small, supervised-then-PPO scorer over observable candidates; it
is not a fixed rule and not a full option-critic stack.

---

## 3. Atlas-G architecture

### 3.1 Observation and memory

```
rays 64 ─► circular Conv1d ─────────────► 128 ┐
entities ≤32 ─► type emb + MHA + pool ──► 128 ┤
vitals 17 ─► MLP ───────────────────────►  64 ┤
RGB 30×40×3 ─► Conv2d ×3 ───────────────► 128 ┤
prev action 12 ─► MLP ──────────────────►  32 ┤
local map 8×32×32 ─► CNN ───────────────► 128 ┤
global map 8×32×32 ─► CNN ──────────────►  96 ┤
pose 8 ─► MLP ──────────────────────────►  32 ┤
goal 8 ─► MLP ──────────────────────────►  32 ┤
token readout ─► attention query ───────► 128 ┤
                                              ├─ concat 768 → 384
                                              ▼
                                         GRU(384)
                                              │
        ┌────────────┬──────────────┬─────────┼──────────────┐
        ▼            ▼              ▼         ▼              ▼
    actor 15     critic 1        aux 45   token q       manager (every 8)
```

**Map.** 96×96 cells at 64 units/cell, 8 channels: visited/free, wall, door, locked
door, item/key, frontier, exit likelihood, enemy-decay. Two heading-aligned crops:
local 32×32 at 64 units/cell (2,048-unit span) and global 32×32 at 128 units/cell
(4,096-unit span). 64 u/cell makes a 64-unit door one cell; the current 96 u/cell map
cannot resolve it.

**Pose.** Turn table keyed by `turn_held` and warm-up state; ego-velocity integration;
translation-only scan match over ±8 cells (289 candidates) with a sharp-peak acceptance
test. Pose vector 8: x, y, sin/cos, vf, vr, scan confidence, drift flag. Target drift
< 0.5 cell / 250 steps.

**Event tokens (≤32).** Locked doors, doors, keys, exit cues, item caches, frontier
clusters. Twenty features: relative x/y/distance/sin/cos, type one-hot, colour, age,
visits, resolved, confidence, sector novelty, door-open flag. Merge within 64 units;
evict resolved first. Attention readout with a query from the GRU; the attention weights
are the live-demo visual for “the agent remembers the door when it picks up the key.”

### 3.2 Learned goal manager

Every K=8 decisions, score candidates:

- the ≤32 event tokens,
- eight frontier sectors (coarse directions to unvisited regions),
- the visible exit cue,
- “continue current goal.”

Input per candidate: token features + GRU state + global map summary + vitals. A 2-layer
MLP scores each; softmax over candidates. The selected candidate yields a goal vector:

```
goal = [dx/512, dy/512, dist/1024, sin(rel), cos(rel),
        type/6, colour/3, confidence]
```

The worker receives this vector every step. The manager is trained in two stages:

1. **Supervised goal labels (BC).** During demonstration collection, the teacher knows
   its current objective (`goal_kind`, world goal). Label the candidate nearest the
   teacher’s goal; if the objective is not tokenized, label the nearest frontier
   direction to the teacher’s next waypoint. This teaches “which remembered thing to go
   to” from privileged training state.
2. **PPO at option boundaries.** +0.5 when the selected token is resolved (door opened,
   key picked, exit reached) or within 96 units; +0.1 for a new sector during the option;
   −0.05 on option timeout (64 steps); entropy 0.01. The manager is frozen for the first
   PPO run; unfreeze only if the supervised manager clears the tier-2 gate.

If the manager fails, the worker still has a valid fallback: the goal vector can be
zeroed and the existing PPO-25 checkpoint is the shippable fallback. No hardcoded
“nearest frontier” rule is used at inference.

### 3.3 Worker and critic

- Worker: existing multimodal encoders + map + tokens + goal → GRU(384) → factorized
  heads. Goal conditioning by concatenation; FiLM only if the ablation is positive.
- **Asymmetric critic**: 384-d GRU state + 32 privileged features (true pose, keys,
  locks, geodesic distance to objective, monster count, time, door states, exit-visible).
  Training only; the actor sees no privileged key. Geodesic potential shaping
  `Φ = −d/1000`, potential-based and annealed.
- Aux heads: the existing 35 + frontier bearing 8 + scan confidence 1 + goal completion
  1 = 45.

### 3.4 Training phases and gates

| Phase | Time | Data / objective | Gate |
|---|---:|---|---|
| 0. Suite + baseline | 1 h | Build suite; run PPO-25 on fast suite; freeze fallback | Baseline JSON + CIs |
| 1. Map/pose v2 | 2 h | Scan match, 8-channel map, crops, unit tests | Drift <0.5 cell/250; p99 <2 ms |
| 2. Balanced BC + goal labels | 3 h | 150–250k: nav 30%, frontier 20%, doors 20%, arena 20%, keys 10%; 3× USE/key/exit frames; teacher goal labels | Turn >65%; doors ≥6/8; t1 ≥5/8; t2 coverage ≥15% |
| 3. DAgger | 2 h | 30–50k student states; frontier/A*/door teachers; manager in the loop | t2 exits ≥2/8; no t1 regression |
| 4. PPO + asymmetric critic | 3 h | 8 envs, 128 rollouts, 4 epochs; 30% rehearsal; KL to BC early | t2 ≥3/8; t3 ≥1/8; weighted proxy > +8.09 |
| 5. Manager PPO (stretch) | 1 h | Option-level PPO; entropy; completion reward | t3 ≥1/8 or return-to-door >50% |
| 6. Full suite + freeze | 2 h | 5 seeds × 30 maps; probes; brief | Ship best held-out |

### 3.5 Inference and rules

- 2.0–2.6 M parameters; target 3–6 ms p99 on one CPU thread; >6× budget margin.
- No engine, no privileged state, no A*/BFS/Dijkstra at inference. The map writer,
  scan matcher and token bank are deterministic perception; the manager is a learned
  policy. Ask the organisers for a written yes on the map/frontier interpretation.
- Deterministic argmax; explicit state; fallback to PPO-25 if anything regresses.

### 3.6 What to cut

| Time left | Cut | Keep |
|---|---|---|
| <2 h | Everything except eval | Ship PPO-25; run suite; write brief |
| 2–4 h | Manager, tokens, PLR | Scan match + balanced BC + asymmetric critic |
| 4–6 h | Manager, learned writes | Map + scan match + tokens + BC/DAgger |
| 6–10 h | PLR, multi-seed, tier-4 specialisation | Full Atlas-G + DAgger + PPO |
| 10 h+ | Only ablations | Full recommendation |

Never cut: held-out split, full suite, validator, deterministic inference, fallback.

---

## 4. The benchmark suite

This is a deliverable and the engine of improvement, not a scoreboard. It follows the
research review’s 14-category design and adds the integration hooks the current bench
lacks.

### 4.1 Categories

| ID | Category | Skill | Ticks |
|---|---|---|---:|
| C0 | Sanity | Walk to a visible exit; one door | 2100 |
| C1 | Duel | 1v1 aim/dodge at 128/256/512 u | 2100 |
| C2 | Arena | 3–6 monsters, cover, limited ammo | 2100 |
| C3 | Corridor door | USE edge, alignment | 2100 |
| C4 | Room door | 16 start poses, arbitrary approach | 2100 |
| C5 | Key–lock | Colour, one key/one lock | 3150 |
| C6 | Two-key chain | Ordering, loops, long horizon | 4200 |
| C7 | Exploration | 8–12 rooms, hidden exit, coverage | 4200 |
| C8 | Backtrack | Key behind start, lock ahead | 4200 |
| C9 | Ambush | Monsters after key/door, re-orientation | 4200 |
| C10 | Resource | Scarce ammo, many monsters | 4200 |
| C11 | Gauntlet | All skills, 2 keys, 16 monsters | 4200 |
| C12 | Freedoom | MAP23–32 hand-made transfer | 4200 |
| C13 | OOD | Larger/deeper, new textures/configs | 4200 |

### 4.2 Generator

Room graph (4–20 rooms, degree 2–4), 0–8 doors at 1 tile, 0–3 keys/locks with a
solvable ordering, 0–4 loops, 0–3 dead ends, 1–2 tile cover, monster mix, ammo budget,
random start room/position/heading, visible exit sign or switch. Every map asserts
solvability (exit reachable with obtainable keys; keys reachable before locks; start not
in a wall/monster; no unopenable door). `(seed, spec)` → exact WAD and `spec_hash`.

### 4.3 Splits and leakage

| Split | Seeds | Use |
|---|---|---|
| Train | 0–9,999 generated; Freedoom MAP01–22 | BC/DAgger/PPO |
| Val-ID | 10,000–10,999 same configs | Gates/checkpoint selection |
| Test-ID | 30,000–30,999 same configs | Final report, one touch |
| Test-OOD | 40,000+ new sizes/depths/textures/mixes | Generalisation |
| Test-Freedoom | MAP23–32 | Hand-made transfer |

No test map in any training manifest; `spec_hash` intersection must be empty; a
`test_touch.log` records every evaluation. Texture and monster holdouts are separate
axes. Seed holdout tests memorisation; config holdout tests grammar generalisation.

### 4.4 Metrics

Exit, proxy, coverage, keys/locks/doors, kills/damage/deaths, SPL, success@k (k=5),
sample efficiency (steps to 50% final exit), latency p50/p99 and per-map budget, RAM/VRAM,
determinism hash, exit/colour/door calibration, degeneracy (spin, no-op, action entropy,
revisit ratio).

Proxy:

```
proxy_map = 10·exit + 3·keys + 3·locks + 0.5·doors
          + min(0.3·sectors, 10) + 0.6·kills
          − 0.01·damage − (3·(1−steps/T)+1)·death
```

Tier weights [1.0, 1.2, 1.5, 2.0]; report every component and a normalised score.
Official weights are hidden; state that.

### 4.5 Diagnostic probes

| Probe | Measures | Pass |
|---|---|---|
| Memory wipe | Zero map/tokens at key pickup | >30% relative exit drop |
| Locked-door recall | Key far from door; return | >50% return |
| Door angles | 8 headings × 8 offsets | >70% overall; >50% worst |
| Frontier coverage | C7, 300 steps | >20% coverage |
| Combat duels | 1v1 at 3 ranges | >70% at 256 u; damage <30 |
| No-op/spin | 200 steps | spin <10%, no-op <20% |
| Colour swap | Swap textures | accuracy >70%; drop <20% |
| Exit-sign ablation | Sign → wall | >50% exit drop |
| Corruption | 10% ray dropout, blur, shuffle | <20% relative drop |
| Determinism | Two runs | identical hash |

### 4.6 Integration into training

- **Curriculum:** `ladder.yaml`; sample 30% mastered, 50% current, 20% hardest.
- **Prioritized Level Replay:** priority from TD error/learning progress; cap staleness;
  minimum probability per category.
- **Gates:** fast suite (5 maps/category, 1 seed) every 10 PPO updates; full suite at
  phase boundaries; promote only if the tier-weighted proxy improves with a paired
  bootstrap CI excluding 0.
- **Ablations:** map, scan match, tokens, manager, asymmetric critic, exit channel,
  colour head, BC/DAgger, frontier reward; fixed 2 h each; paired deltas/CI.
- **Statistics:** 5 engine seeds; ≥30 maps/category final; stratified bootstrap over
  maps; mean ±95% CI, IQM, performance profiles; pre-register primary metric; never
  “best of n”; keep every failed run.

### 4.7 Files and CLI

```
training/suite/spec.py        categories, map specs, seed blocks, ranges
training/suite/generate.py    WADs + manifest + spec_hash + leakage check
training/suite/run.py         checkpoint -> episode JSONL, resumable
training/suite/metrics.py     proxy, SPL, coverage, bootstrap, IQM
training/suite/probes.py      memory wipe, door angles, colour swap, corruption
training/suite/report.py      Markdown/JSON, baseline diff, CI flags
training/suite/leakage.py     hash intersection, test-touch log
training/suite/configs/ladder.yaml
```

```bash
python -m training.suite.generate --split test --categories all --n 30
python -m training.suite.run --ckpt runs/ppo/ckpt.pt --split test --seeds 5 --full
python -m training.suite.probes --ckpt runs/ppo/ckpt.pt --probes memory_wipe,door_angles
python -m training.suite.report --run runs/suite/test --baseline runs/suite/baseline
```

The fast subset must finish in <10 min on this host; the full suite in <90 min.

---

## 5. Build order (after approval)

1. **Suite first (1 h).** Generator specs/categories, fast + full CLI, baseline PPO-25
   report; commit. This gives every later change a fair measurement.
2. **Map/pose v2 (2 h).** 8-channel 96×96 map, scan match, crops, unit tests, latency.
3. **Data v2 + balanced BC (3 h).** Teacher goal labels, room-door/frontier/arena/nav
   mix, 3× rare frames, per-head accuracy.
4. **DAgger + manager (3 h).** Student states, manager supervised, goal vector,
   asymmetric critic.
5. **PPO (3 h).** 30% rehearsal, KL to BC, curriculum, PLR optional; gates.
6. **Full suite, probes, ablations, brief, README, validator (2 h).**
7. **Freeze.** Best held-out checkpoint; PPO-25 remains the fallback.

---

## 6. What I need from you

1. **Approve Atlas-G (option C)** — learned goal manager + map/tokens + asymmetric
   critic + balanced BC/DAgger/PPO — rather than the simpler fixed-selector B+ or more
   PPO. The manager is the clear architectural step beyond the current agent.
2. **Approve the 14-category suite** as the primary development target, with the
   official validator only as a gate.
3. **Approve the rule interpretation:** the map writer/scan matcher/token bank are
   deterministic perception; the goal manager and worker are learned networks; no
   A*/BFS/Dijkstra at inference. If you want, I will ask the organisers for a written
   confirmation before submission.
4. **Approve the cut line:** if tier-2 <2/8 after DAgger, I ship the best BC/PPO
   fallback and report honestly rather than forcing the manager.

Once approved, I will build in the order above, commit after each phase, and report the
suite numbers at every gate.
