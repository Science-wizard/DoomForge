# Architecture Reference

This is the technical reference for the submitted agent and the training system.
The plan is in [`../PLAN.md`](../PLAN.md); the lab log is in [`../progress.md`](../progress.md).

---

## 1. Contract recap

The scorer runs `submission/agent.py` in a sandbox with **no engine reference**:

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device: str): ...
    def reset(self) -> None: ...
    def act(self, observation: dict) -> np.ndarray  # int8[9] in {0,1}
```

Observation:

| Key | Shape | Meaning |
|---|---|---|
| `rays.distance` | 64 float32 | hit distance / 1024 (1.0 = nothing) |
| `rays.hit_type` | 64 int | 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item |
| `visible_entities.features` | 32×5 float32 | `[dist/1024, sin(rel), cos(rel), rel_vel_forward, rel_vel_right]` |
| `visible_entities.entity_type` | 32 int | 0 pad, 1 melee, 2 hitscan, 3 projectile enemy, 4 projectile, 5 health, 6 armor, 7 ammo, 8 weapon, 9 key, 10 barrel, 11 other |
| `visible_entities.mask` | 32 int8 | 1 = real entity |
| `vitals` | 17 float32 | health, armor, weapon, ammo, four ammo pools, six key flags, three velocities |
| `rgb`, `depth` | 120×160×3, 120×160 | optional |

Action: 9 binary buttons, held for 4 game tics.

---

## 2. Inference data flow

```
observation dict
   │
   ├─ rays ──────────► RayEncoder (circular Conv1d ×3) ──► 128 ─┐
   ├─ entities ──────► EntityEncoder (embed + MHA + pool) ─► 128 ─┤
   └─ vitals ────────► VitalsEncoder (MLP) ──────────────►  64 ─┤
                                                                ├─ concat 320
                                                                ▼
                                                         GRU(320) → h_t
                                                                │
                                              ┌─────────────────┼──────────────────┐
                                              ▼                 ▼                  ▼
                                        actor heads         value head        aux heads
                                   move/strafe/turn/…        V(s)         keys, enemy, door
                                              │
                                              ▼
                                   argmax each head → 9-bit action
```

The GRU state `h_t` persists across `act()` calls and is zeroed in `reset()`.

### Why not pixels by default?

Pixels cost CPU time on a single thread and the raycast stream already contains
metric free-space information, door/lock types and entity hits. Depth and RGB are
kept as optional ablations; the final model uses whichever wins on the proxy
leaderboard per millisecond.

---

## 3. Action factorisation

`MultiBinary(9)` has 512 combinations; many are contradictory. We factorise:

```python
HEADS = (("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("next", 2))
```

The model emits 15 logits. The vector is reconstructed exactly:

```
move=1 → MOVE_FORWARD     strafe=1 → MOVE_LEFT      turn=1 → TURN_LEFT
move=2 → MOVE_BACKWARD    strafe=2 → MOVE_RIGHT     turn=2 → TURN_RIGHT
attack/use/next are binary.
```

This is action masking by construction, allowed by the rules (“action masking from
observation fields” and learned heads are explicitly permitted).

---

## 4. Model sizes and budgets

| Component | Params |
|---|---:|
| Ray encoder | ~75 k |
| Entity encoder | ~110 k |
| Vitals encoder | ~5 k |
| Fusion + GRU | ~1.0 M |
| Heads | ~5 k |
| **Total** | **~1.2 M** |

Measured targets: 1–3 ms per decision on one CPU thread; 2 GB VRAM is far above the
fp32 checkpoint (~5 MB) and activations. The 40 ms/decision average budget leaves a
>10× margin.

---

## 5. Training system

```
maps/  ──►  Teacher (privileged)  ──►  demonstrations (npz)
                                              │
                                              ▼
                                         Behaviour cloning
                                              │
                                              ▼
                              PPO fine-tune on generated curriculum
                                              │
                                              ▼
                              proxy leaderboard → best checkpoint
                                              │
                                              ▼
                                   submission/weights/*.pt
```

### Teacher (training only)

`training/teacher/`

- **MapData** parses Doom binary and UDMF: linedefs, front/back sectors, specials,
  things, keys, exit lines.
- **NavGrid** rasterises blocking linedefs at 16 units; marks door cells with lock
  metadata and passage direction; A* with door costs; line-of-sight; distance fields.
- **Teacher** sequences objectives (exit → key → locked door), follows a carrot path,
  uses doors with an explicit state machine, fights with circle-strafing and
  backpedalling, and handles switch vs walk-over exits.

The teacher is *not* imported by `submission/`. It uses `env.privileged_state()`,
which the scorer keeps private.

### Maps

- `training/maps/gen_wad.py`: BSP room generator, four tiers, node-less Doom binary,
  1-tile isolated doors, keycard/locked-door progression, arena pillars, monsters,
  ammo/health, walk-over exits.
- `training/maps/drills.py`: 1-tile corridors with isolated doors and an optional key
  alcove — the clean signal for the door/key skill.
- `training/maps/prepare_freedoom.py`: 32 real Freedoom maps in three variants
  (raw / doors-open / doors-open+no-monsters).

### Behaviour cloning

- Episodes padded into `(B, T, …)` sequences; GRU runs over the full episode.
- Loss = cross-entropy on six heads + 0.2 × auxiliary BCE + 0.5 × value MSE.
- Auxiliary labels are self-supervised from the observation: six key flags,
  “enemy visible”, “door ahead” (ray 0 hit type 2/3).
- Episode-level train/val split; the validation metric is *environment* exit rate, not
  just loss.

### PPO

- Clipped surrogate (0.2), GAE(0.95), entropy bonus, advantage normalisation.
- Reward (training only): exit +10, death −5, kill +0.6, damage −0.02/hp, key +2,
  door +0.4, locked door +1, new nav cell +0.01, distance-to-objective penalty
  −0.0002 × BFS cells.
- Curriculum: drills → tier 2 → tier 2–3 with monsters → full tier 2–4 + Freedoom.

---

## 6. Determinism & safety at inference

- `argmax` heads, no sampling, no dropout, no batch norm.
- All state is explicit (`h_t`); `reset()` clears it.
- No file writes, no network, no engine access, no `sys.path` tricks.
- Weights are loaded once in `__init__` from `Path(__file__).parent / "weights"`.
- `torch.set_num_threads(1)` to respect the one-CPU limit.
- Any exception is a zero for that map, so the runtime is deliberately small and
  defensive: shape checks, `np.nan_to_num`, and a fallback zero action only if the
  observation is malformed.

---

## 7. Neural Cartographer (Phase 3)

The GRU is a short-horizon memory. Keys and locked doors require long-horizon
backtracking, so Phase 3 adds an explicit allocentric map:

- Pose from action + velocity, with a learned residual supervised by privileged deltas.
- `M ∈ R^{8×48×48}` at 128 units/cell; warp by `grid_sample`, write the polar ray scan,
  update with a small ConvGRU.
- Read 32 relative probes + global pooling.
- Auxiliary losses: visited mask, next-scan prediction, relative door/key/exit labels.

Ablation plan: GRU only → + visited mask → + learned map. Keep the map only if it
improves tier-3 exit rate per millisecond.

---

## 8. Module map

| Path | Role |
|---|---|
| `submission/agent.py` | evaluation entry point |
| `submission/model.py` | network definition (single source of truth) |
| `submission/weights/` | Git LFS checkpoint |
| `training/model.py` | re-export of the submission model |
| `training/obs.py` | observation flattening/stacking |
| `training/collect.py` | expert demonstration shards |
| `training/bc.py` | behaviour cloning |
| `training/ppo.py` | RL fine-tuning |
| `training/evaluate.py` | proxy leaderboard, replays |
| `training/maps/` | generator, drills, Freedoom prep |
| `training/teacher/` | privileged expert (never imported at inference) |
| `validate.py` | official validator (untouched) |
