# Architecture Reference — Revision 2

Plan of record: [`../PLAN.md`](../PLAN.md) (24-hour cut). Lab log: [`../progress.md`](../progress.md).

This document describes **Baseline A** (the safety submission) and the deferred
**B+ lite** memory extension. A is implemented; B+ is a roadmap until A clears its
held-out gates.

---

## 1. Contract recap

The scorer runs `submission/agent.py` with no engine reference:

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device: str): ...
    def reset(self) -> None: ...
    def act(self, observation: dict) -> np.ndarray  # int8[9] in {0,1}
```

| Observation | Shape | Notes |
|---|---|---|
| `rays.distance` | 64 float32 | hit distance / 1024 |
| `rays.hit_type` | 64 int | none/wall/door/locked door/enemy/item |
| `visible_entities.features` | 32×5 | dist, sin, cos, relative velocity |
| `visible_entities.entity_type` | 32 int | enemies, items, keys, barrels |
| `visible_entities.mask` | 32 int8 | validity |
| `vitals` | 17 | health, armour, weapon, ammo, six key flags, velocity |
| `rgb` / `depth` | 120×160×3 / 120×160 | pixels |

**Verified gaps (Revision 2):**

- Exit is **not** a ray hit type or entity class. The agent must learn exits from
  RGB; training labels come from privileged exit linedefs.
- Key and lock **colour** is hidden until pickup. The agent learns colour from RGB.
- `USE` fires on button-down and reverses a moving door. Never spam it.
- Turn rate is stateful; the agent receives `turn_held` and `prev_action`.

---

## 2. Baseline A

```
rays 64 ─► circular Conv1d ×3 ─────────────────────────────► 128 ─┐
entities ≤32 ─► type embed + MHA + masked mean/max ─────────► 128 ─┤
vitals 17 ─► MLP ───────────────────────────────────────────►  64 ─┤
RGB 30×40×3 ─► Conv2d ×3 (stride 2) ────────────────────────► 128 ─┤
prev action 12 ─► MLP ─────────────────────────────────────►  32 ─┤
                                                                  ├─ concat 480
                                                                  ▼
                                                           GRU(384) → 384
                                                                  │
                        ┌─────────────────┬───────────────────────┼──────────────────┐
                        ▼                 ▼                       ▼                  ▼
                  action heads         value                 aux heads          (no map)
              move/strafe/turn/…       V(s)        keys, enemy, door, exit visible,
                                                   exit bearing 13, exit dist 6,
                                                   key colours 6, damage-next-8
```

- **1.56 M parameters.** Measured **0.45 ms** per decision on one CPU thread
  (fully loaded agent p50 0.6 ms in the validator). The 40 ms budget is not a risk.
- Factorised heads: `move(3) × strafe(3) × turn(3) × attack(2) × use(2) × next(2)`.
  No contradictory buttons; 15 logits instead of 512.
- RGB is pooled 4× to 30×40 to keep colour while costing 3.6 KB/frame. The CNN sees
  exit signs, blue/red/yellow doors and coloured key pickups.
- Auxiliary heads are trained from privileged labels and are **not used for acting**;
  they shape the shared encoder so the GRU state carries exit/lock/threat information.
- `reset()` clears the GRU state, previous action, `turn_held` and `use_last`.

### Auxiliary labels (training only)

| Label | Source |
|---|---|
| keys held (6) | `vitals[8:14]` |
| enemy visible | entity types 1–3 with mask |
| door ahead | ray 0 hit type 2/3 |
| exit visible | exit linedef in FOV ±55° and grid line-of-sight |
| exit bearing (13 classes) | relative angle bin, class 12 = not visible |
| exit distance (6 bins) | 256/512/768/1024/1400-unit bins |
| key colours (6) | key thing in FOV and line-of-sight, colour from map metadata |
| damage next 8 | `max(0, health[t] − health[t+8]) / 100` |

---

## 3. B+ lite (deferred until A ships)

The GRU is a short-horizon memory. Keys and locked doors require hundreds of steps.
B+ adds **perception memory**, not planning:

- **Pose:** exact heading integration from the measured turn table (`turn_held` +
  warm-up state) and translation from `vitals[14:16]`; optional translation-only scan
  match against the wall channel (±16 units, ~81 candidates, <0.5 ms).
- **Map:** 64 units/cell, 96×96 (6,144 units). Channels: free, wall, door, locked
  door, visited, recency, frontier, item, enemy decay, exit likelihood, opened.
- **Write:** project rays and entity positions at the estimated pose. Deterministic
  perception; no search or planning.
- **Read:** two 32×32 egocentric crops (64 and 128 units/cell), rotated to heading,
  through a small CNN → 256.
- **Event tokens:** ≤64 tokens for locked doors, keys, exit cues and frontier
  clusters. 24 features; relative fields recomputed from pose every step; attention
  query from the GRU state → 128. This is the long-horizon fix and the live-demo
  attention overlay.
- **Optional learned write:** 4 channels projected from the GRU state at the agent's
  cell; keep only if an ablation wins.

**Causal tests:** memory wipe after key pickup; locked-door recall rate; attention
spike on the remembered door when the key flag flips; drift curves.

**Rule check:** deterministic map writing is preprocessing/encoding; the policy still
chooses actions. No A*/BFS/Dijkstra in `submission/`. Ask the organisers for a written
yes before submission.

---

## 4. Training system

```
generated + Freedoom maps
        │
        ├─ privileged teacher (A*, doors, exits, combat)  ──► demonstrations
        │                                                   (rays, entities,
        │                                                    vitals, RGB, prev,
        │                                                    actions, aux labels)
        ▼
  behaviour cloning  ──►  PPO fine-tune  ──►  proxy leaderboard  ──►  weights
```

### Teacher

`training/teacher/`: map parser (binary + UDMF), 16-unit navigation grid, A* with
door costs, door state machine, switch/walk-over exits, circle-strafing combat.
Used only for data; never imported by `submission/`.

### Maps

- `training/maps/gen_wad.py`: BSP rooms, 1-tile isolated doors, coloured locks,
  visible exit sign wall, arena cover, four tiers. Node-less Doom binary (required
  for things to spawn). Deterministic from `(seed, tier)`.
- `training/maps/drills.py`: 1-tile corridors with isolated doors and key alcove.
- `training/maps/prepare_freedoom.py`: 32 maps × raw / doors-open / doors-open+no-
  monsters.

### Splits

`training/splits.py`: Freedoom MAP01–22 train, MAP23–32 held out; generated seeds
0–9,999 train, 10,000–10,999 validation, 20,000+ OOD. The collector and evaluator read
the same file.

### Reward (PPO, proxy)

Exit +10, key +3, lock +3, door +0.5, new sector +0.3, new cell +0.01 capped, kill
+0.6, damage −0.01/HP, death `−(3(1−t/T)+1)`, wasted shot −0.003, useless USE
−0.005, time −0.001, potential-based objective shaping. Exact scorer weights are
hidden; every component is reported separately.

---

## 5. Evaluation

`training/evaluate.py` runs the real environment and reports: exit, keys, locks,
doors, unique sectors (`SectorTracker`), kills, damage, survival, steps, proxy score,
p50/p99 latency. Suites: `gen_val`, `freedoom_eval` (MAP23–32), `ood`, `all`.

`training/coverage.py` maps the player position to a sector via the wrapper's
`sector_polygons()` and counts unique sectors.

---

## 6. Runtime and safety

- Loads one fp32 checkpoint (~6 MB) from `submission/weights/` in `__init__`.
- `torch.set_num_threads(1)`; `torch.inference_mode()`; 3 dummy forwards warm-up.
- Argmax by default; deterministic. No dropout, no batch norm.
- Sanitises NaN/inf; on any failure returns a valid zero action and logs once.
- No network, no engine, no disk writes, no imports from `training/`.

---

## 7. Module map

| Path | Role |
|---|---|
| `submission/agent.py` | evaluation entry point |
| `submission/model.py` | Baseline A network (single source of truth) |
| `submission/weights/doomforge.pt` | Git LFS checkpoint |
| `training/obs.py` | observation flattening, RGB pooling, prev-action vector |
| `training/collect.py` | demonstrations + privileged labels |
| `training/bc.py` | behaviour cloning (windowed episodes) |
| `training/ppo.py` | RL fine-tune (scaffold) |
| `training/evaluate.py` | proxy leaderboard / replays |
| `training/coverage.py` | sector coverage |
| `training/splits.py` | frozen data splits |
| `training/maps/` | generator, drills, Freedoom prep |
| `training/teacher/` | privileged expert (training only) |
| `validate.py` | official validator (untouched) |

---

## 5. Atlas-G v1 (Phase 3)

Atlas-G keeps Baseline A's encoders and GRU and adds a learned goal interface and
a frontier map channel. It is trained on the diagnostic suite, not just generated
tiers.

### 5.1 What changed

| Component | Baseline A | Atlas-G v1 |
|---|---|---|
| Map memory | 5×32×32 @ 64 u/cell | **6×32×32**: visited, wall, door, locked door, item/key, **frontier** |
| Goal signal | none | **GoalHead(640→64) → 8-vector + 4-way type**, encoded to 32 and fused into the 672→384 projection |
| Aux heads | exit visibility/bearing/distance, keys, door, damage | unchanged |
| Action heads | 6 factorised heads (15 logits) | unchanged |
| Parameters | 1.807 M | **1.867 M** |
| Inference | p50 0.86 ms CPU | same budget (goal head adds ~0.01 ms) |

The goal vector is ego-relative: `[dx/512, dy/512, dist/1024, sin(rel), cos(rel),
type/4, mask, 0]`. Type is exit/key/frontier/other. During BC/DAgger the label is
the privileged teacher's current objective; at inference the head predicts it
from the observation, so the shipped agent never sees privileged state.

### 5.2 What is deliberately not in v1

The approved proposal also listed a 96×96 map at 64 u/cell, scan matching,
event tokens, an 8-step goal manager and an asymmetric critic. v1 keeps the
32×32 deterministic map and per-step goal prediction because the measured
bottleneck was not map resolution: it was wall recovery and key-lock sequencing,
which the suite found immediately. The larger map and manager are gated on the
suite showing a spatial-resolution or goal-thrash failure; the probe set
(door-angle, corruption, memory-wipe, determinism) is how that decision is made.

### 5.3 Training stack (measured)

| workload | MLX GPU | PyTorch CPU | PyTorch MPS |
|---|---:|---:|---:|
| 2 M-param MLP, 256×512 | **4.2 ms** | 11.2 ms | 11.3 ms |
| DoomNet 16×128 fwd+bwd | — | 3.50 s | **0.22 s** |

Training runs on MPS; environments, the suite runner and data collection run on
all 10 CPU threads. MLX is not used for the policy because the recurrent graph
loses its advantage; the measured MLP win does not transfer.

### 5.4 Suite coupling

- `ppo.py --suite` samples the train split with a 30/50/20 mastered/current/
  hardest mix, plus dense training-only potential shaping toward uncollected
  keys.
- `dagger.py --source suite` labels student failure states (used once; the
  unbalanced run is documented as a dead end in `progress.md`).
- Every candidate is scored with `training/suite/run.py`; promotion uses
  `training/suite/compare.py` (paired bootstrap + per-category floors).

### 5.5 Atlas-H: perception-routed hybrid (shipped candidate)

Two experts are loaded from `submission/weights/`:

| file | expert | role |
|---|---|---|
| `doomforge.pt` | Atlas-G v1 (BC + suite DAgger/key demos) | navigation, exploration, keys, doors |
| `combat.pt` | Phase-2 PPO-25 (v1, no goal head) | visible combat |
| `specialist.pt` | Atlas-G architecture, BC on C3/C4 suite demos | narrow-corridor doors |

The navigation and combat networks are evaluated on every observation so their
recurrent state and the deterministic map stay current; the door specialist runs
only on gated decisions. The **router reads only the observation and the agent's
own map memory**:

1. any `enemy_melee` / `enemy_hitscan` / `enemy_projectile` visible ⇒ combat
   expert for at least 16 decisions;
2. otherwise, if no locked door is known, no key is visible or held, a door is
   known, and the median side ray at first activation is **< 0.23** (a narrow
   corridor), latch to the door specialist for the episode;
3. otherwise Atlas-G.

No privileged state, category label, or map name is used. The 0.23 threshold is
what keeps the specialist off the two-room validator map and C4 while still
firing on C3.

A deterministic **safety reflex** handles the freeze modes found by the suite:
if no enemy is visible and the policy has pressed forward for 8 decisions with
forward velocity ≈ 0, it turns for 4 decisions; if it has emitted a pure no-op
for 8 decisions, it nudges forward + USE. The reflex is a supervisor: the
policy's own action (not the override) is fed back as ``prev_action`` so its
learned dynamics stay in distribution, while the environment executes the
override. This fixed C6 (0/5 → 5/5), C3
(0/5 → 3/5) and C8 (1/5 → 3/5) on the validation suite and is disabled in
combat, leaving the arena expert untouched.

Cost: two 1.8 M-parameter forward passes normally, a third only on gated
decisions. The official validator measures p50 1.7–2.4 ms, p99 2.0–2.6 ms, max
6 ms per step (single model was p50 0.8 ms, p99 1.0 ms), well inside the 40 ms
average and 5 s worst-case budget.

Why a hybrid rather than one network: the suite showed the failure modes are
anti-correlated. PPO-25 wins arena combat (t1 6/8, C2 80%) but circles in
corridors (C0 0%, C5 0%); Atlas-G solves C0/C3/C5/C8 but is worse in open
combat. A single policy trained on all of it catastrophically forgot one side
(BC v5: C5 100% but t1 0/12; PPO v2: mean return up but C5 0/5). Routing on a
perception cue the agent already possesses is the measured best of both.
