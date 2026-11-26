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
