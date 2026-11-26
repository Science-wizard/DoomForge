# Architecture: Atlas-E

The shipped agent is one recurrent network that reads the official observation
plus a deterministic ego-motion map. Training: [`TRAINING.md`](TRAINING.md).
Teaching guide: [`../explanation.md`](../explanation.md).

## 1. Contract

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device: str): ...
    def reset(self) -> None: ...
    def act(self, observation: dict) -> np.ndarray  # int8[9] in {0, 1}
```

| Observation | Shape | Used as |
|---|---|---|
| `rays.distance`, `rays.hit_type` | 64, 64 | ray encoder; map projection |
| `visible_entities.features/entity_type/mask` | 32×5, 32, 32 | entity encoder; item/key projection |
| `vitals` | 17 | vitals encoder; odometry (velocity) |
| `rgb` | 120×160×3 | pooled 4× to 30×40 for the colour encoder |
| `depth` | 120×160 | unused (rays already give metric free space) |

## 2. Data flow per decision

```
observation
  │ sanitise (NaN/inf → finite, types clipped)
  ├──────────────► EgoMap.update(observation, last executed action)
  │                  ├─ dead reckoning: heading (turn-run table), position (per-tic friction, wall clip)
  │                  └─ project rays / items / keys / own cell into a 160×160 grid
  │                EgoMap.ego_crop() ─► 16×32×32 heading-up map,  pose() ─► 6
  ▼
DoomNet(arch="ego").forward_step(...)  ─► 15 logits
  ▼
seeded sampling per head (temperature 1) ─► 9 buttons
```

## 3. Ego-motion map (`submission/mapmem.py`)

| Component | Detail |
|---|---|
| Heading | per-decision deltas replayed from the executed action: 7.031°, 12.305°, then 14.063° per decision while a turn is held in either direction (warm-up values for the first two decisions). 0.02° error after 500 decisions. |
| Translation | Doom per tic: momentum += thrust; position += momentum; momentum ×= 0.90625. Thrust recovered from the start/end momentum (`vitals[14:16]` in each heading); four tics integrated. |
| Wall clip | step length limited to the previous ray distance in the travel direction − 17 units (momentum is reported while pinned). ~30 units position error after 500 decisions. |
| Grid | 160×160 cells × 64 units = 10,240 units, origin at spawn |
| Channels | free (ray paths), wall, door, locked door (cleared when rays later pass), item, key, trail (own cells), frontier (free cells next to unknown) |
| Readout | heading-up crops: fine 32×32 at 64 u/cell; coarse 32×32 at 256 u/cell (4×4 max-pool). Concatenated: 16×32×32. |
| Pose | dead-reckoned x/2048, y/2048 (clipped), sin/cos heading, forward/right velocity /16 |
| Cost | ~0.55 ms per decision (NumPy) |

`MapBuilder(legacy=True)` keeps the older world-aligned 6-channel map for older
checkpoint formats; the shipped agent does not use it.

## 4. Network (`submission/model.py`, `DoomNet(arch="ego")`)

| Module | Structure | Out | Params |
|---|---|---:|---:|
| RayEncoder | [distance, one-hot hit (6), sin, cos] × 64 → circular Conv1d 9→32→64→64 (k5,5,3) → mean ⊕ max → Linear | 128 | 40.6k |
| EntityEncoder | [5 features ⊕ type embedding 16] → MLP 64 → 4-head self-attention + FFN (residual + LayerNorm) → masked mean ⊕ max → Linear | 128 | 47.5k |
| VitalsEncoder | vitals × 0.01 → MLP 64 | 64 | 5.3k |
| RGBEncoder | 3×30×40 → Conv2d 32 (5, s2) → 64 (3, s2) → 64 (3, s2) → Linear | 128 | 221.8k |
| PrevEncoder | previous 9 buttons, USE-last, turn-held/4 → MLP | 32 | 1.5k |
| EgoMapEncoder | 16×32×32 → Conv 32 (3, s1) → 64 (s2) → 64 (s2) → 64 (s2) → Linear | 192 | 293.8k |
| PoseEncoder | 6 → MLP | 32 | 1.3k |
| GoalHead | 704 → 64 → 64 → goal vector 8 (tanh) + goal type 4 | 8 + 4 | 50.1k |
| GoalEncoder | 8 → MLP | 32 | 1.3k |
| Fuse | 736 → 384, GELU | 384 | 283.0k |
| GRU | 384 → 384 | 384 | 887.0k |
| Trunk | 384 → 384, GELU | 384 | 147.8k |
| Actor | 384 → 15 (move 3, strafe 3, turn 3, attack 2, use 2, next 2) | 15 | 5.8k |
| Critic | 384 → 1 | 1 | 0.4k |
| Auxiliary | 384 → 35 (keys 6, enemy, door ahead, exit visible, exit bearing 13, exit distance 6, key colours 6, damage-next-8) | 35 | 13.5k |
| **Acting total** | | | **2,000,735** |
| Privileged critic (training only) | 384 ⊕ 16 privileged → 256 → 1 | 1 | 102.9k |

All activations are GELU. There is no batch norm or dropout. Inference is 0.69 ms per
decision on one CPU thread; the full agent (map + network + sampling) measures
p50 1.6 ms / p99 1.9 ms in the official validator.

### Heads and what they are for

- **Action heads.** Factorised categorical distributions. They exclude contradictory
  button pairs (216 joint actions instead of 512) and are converted back to a valid
  `MultiBinary(9)`.
- **Goal head.** Supervised in BC with the teacher's current objective
  (ego-relative dx, dy, distance, bearing, type). Its prediction is re-encoded and
  fused into the policy input. It is frozen during PPO.
- **Auxiliary heads.** Supervised in BC with privileged labels. They shape the
  encoder to represent exit signs, lock colours and threats, and are never read
  when acting.
- **Privileged critic.** PPO baseline. It is stored in the checkpoint and never
  evaluated by the agent.

## 5. Acting (`submission/agent.py`)

- `__init__`: `torch.set_num_threads(1)`, loads `weights/doomforge.pt` on CPU
  (or CUDA if offered), reads the architecture from the state dict and the
  acting config (`temperature`, `route_combat`) from the checkpoint, and runs
  three warm-up decisions.
- `reset()`: clears the GRU state, previous-action bookkeeping and the map, and
  re-seeds the sampling generator (seed 12345).
- `act()`: sanitise → map update → forward → sample six heads → 9 buttons.
  The executed action is fed back to the map's odometry and to the previous-
  action input. Any exception returns a valid zero action and resets.
- No network, engine, disk writes or `training/` imports.

Compatibility paths, inactive in the shipped configuration: older Atlas-G/v1
checkpoints, an optional perception-routed combat expert (`combat.pt`), and a
corridor-door specialist (`specialist.pt`). They activate only if those weight
files are present and the config allows them.

## 6. Module map

| Path | Role |
|---|---|
| `submission/agent.py` | evaluation entry point |
| `submission/model.py` | network definitions (`arch="ego"` shipped) |
| `submission/mapmem.py` | `EgoMap` and the legacy `MapBuilder` |
| `submission/model_v1.py` | oldest checkpoint format (compatibility) |
| `submission/weights/doomforge.pt` | shipped checkpoint (Git LFS, 8.4 MB) |
| `training/` | maps, teacher, demos, BC, PPO, suite (see `training/README.md`) |
