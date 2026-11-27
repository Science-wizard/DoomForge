# Training pipeline

How the shipped `submission/weights/doomforge.pt` is produced, end to end.
Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). Numbers: [`RESULTS.md`](RESULTS.md).
Everything here is training-only; the agent never imports `training/`.

```
1. maps ─► 2. suite splits ─► 3. teacher demos ─► 4. ego maps ─► 5. BC ─► 6. PPO (3 stages) ─► 7. gate ─► ship
```

Hardware used: one Apple-silicon laptop (10 CPU cores, MPS GPU, 16 GB).
Environments, data collection and evaluation run on CPU worker processes;
network training runs on MPS.

---

## 1. Maps

| Source | Module | What it produces |
|---|---|---|
| Procedural rooms | `training/maps/gen_wad.py` | node-less binary WADs, tiers 1–4: arenas with cover and monsters; 7–13 room layouts with doors and loops; key → coloured lock → exit; two keys, two locks, scarce resources. Exit is a walk-over line with a visible exit-sign wall. Textures and lighting are randomised per seed. |
| Drills | `training/maps/drills.py`, `doorroom.py` | corridor doors, a door on a room wall from a random pose, key alcove → lock (C5), key → lock → key → lock (C6), key behind the spawn (C8). Locks sit on the only path. |
| Freedoom | `training/maps/prepare_freedoom.py` | 32 hand-made maps (raw, doors-open and no-monster variants for demos). |

ZDoom rebuilds BSP nodes at load time, so generated maps need no node builder.
Generation is deterministic in `(seed, tier, flags)`.

## 2. Diagnostic suite splits

```bash
uv run python -m training.suite.generate --split train --n 150   # 1,966 maps
uv run python -m training.suite.generate --split val   --n 10    # 136 maps
uv run python -m training.suite.generate --split test  --n 30    # 400 maps
uv run python -m training.suite.leakage --train training/data/suite/train/manifest.jsonl \
                                        --test  training/data/suite/test/manifest.jsonl
```

Seeds: train 0+, val 10 000+, test 30 000+, OOD 40 000+. Freedoom: train
MAP01–16, val MAP17–22, test MAP23–32. The leakage check hashes every WAD;
train∩val and train∩test are empty. Categories, metrics and the promotion gate:
[`RESULTS.md`](RESULTS.md#how-we-measure).

## 3. Teacher demonstrations

```bash
uv run python -m training.collect --episodes 250 --source gen --no-doors --no-monsters --out training/data/demos_v2_nav
uv run python -m training.collect --episodes 120 --source gen --tier 1 --out training/data/demos_v2_arena
uv run python -m training.collect --episodes 400 --source drill --drill-doors 2 --out training/data/demos_v2_doors
uv run python -m training.collect --episodes 150 --source drill --drill-key --out training/data/demos_v3_key
uv run python -m training.collect --episodes 150 --source keychain --out training/data/demos_v3_keychain
uv run python -m training.collect --episodes 150 --source backtrack --out training/data/demos_v3_backtrack
uv run python -m training.collect --episodes 150 --source suite --categories C3_corridor_door --out training/data/demos_v4_c3
uv run python -m training.collect --episodes 150 --source suite --categories C4_room_door --out training/data/demos_v4_c4
uv run python -m training.collect --episodes 400 --source roomdoor --out training/data/demos_v5_c4
# plus arena300, roomdoor, drills, v2_t2 (tier-2 exploration) sets: 13 datasets, 436k steps
```

The teacher (`training/teacher/`) reads privileged pose, keys and sector heights
and the WAD geometry: a 16-unit nav grid, A* with door costs and key gating
(exit → needed key → exploration), a door state machine (approach, align, USE
once, wait, cross), circle-strafing combat with projectile dodging. Each step
stores observation, teacher action, previous action, auxiliary labels (exit
visible / bearing / distance, key colours, damage in the next 8 decisions) and
the teacher's goal as an ego-relative vector.

## 4. Ego maps for the demos

```bash
uv run python -m training.egomaps --data training/data/demos_v2_nav,training/data/demos_v3_key,...
```

Replays each stored episode (observations + executed actions) through
`submission/mapmem.py::EgoMap`, so demo maps are exactly what the agent would
have built. The 16 binary channels are bit-packed into one uint16 per cell
(2 KB/step) and unpacked on the GPU. 436k steps regenerate in under a minute.

## 5. Behaviour cloning

```bash
uv run python -m training.bc --arch ego --out training/runs/bc_ego1 --epochs 5 --batch 16 --lr 3e-4 \
  --drill-repeat 2 --repeat keychain=2,backtrack=2,C4=2 \
  --data training/data/demos_v2_nav,...,training/data/demos_v2_t2   # the 13 datasets
```

- 128-step windows (stride 64), batches of 16 windows, AdamW 3e-4, MPS, ~3.5 min/epoch.
- Loss = six-head cross-entropy (USE weighted ×11 when a door is ahead, ×6 on
  USE labels) + 0.2 × auxiliary losses + 0.5 × value + goal MSE + 0.5 × goal-type CE.
- Result: validation loss 0.63.

## 6. PPO (`training/ppo2.py`)

### Mechanics

| Piece | Setting |
|---|---|
| Parallelism | 9 worker processes × 4–8 ViZDoom envs; batched inference on MPS; double-buffered rollout |
| Rollout / windows | 128 steps per env; 32-step recurrent windows with the GRU reset at episode starts |
| Returns | GAE γ 0.995, λ 0.95, bootstrapped from the next observation's value; per-batch advantage normalisation |
| Policy loss | clipped surrogate (0.1), entropy bonus, gradient clip 0.5, Adam |
| Critic | asymmetric: trunk features (detached) + 16 privileged features → 256 → 1; Smooth-L1; warm-up updates train it alone |
| Privileged features | time used, geodesic distance + bearing to the current objective, objective type, keys, health, ammo, coverage, kills, doors opened, cells visited, damage, steps since progress |
| Regularisers | exact KL to a reference policy (its hidden state tracked along the same trajectories); teacher imitation cross-entropy on random 64-step demo windows |
| Curriculum | 85% suite training maps, weight per category 0.6 + (1 − its recent success rate); 15% freshly generated tier 1–4 maps |
| Goal head | frozen (its prediction stays a stable input feature) |

### Reward (training only)

| Term | Value |
|---|---|
| exit | +10 + 2 × time left |
| key picked up | +5 |
| locked / unlocked door opened | +3 / +0.5 |
| new sector entered / new 64-unit cell | +0.3 / +0.01 |
| newly seen floor in the agent's EgoMap | +0.004 per cell, ≤ 0.1 per step |
| geodesic progress toward the objective | +0.003 per unit, clipped ±0.1 per step |
| kill | +0.6 |
| damage | −0.02 per HP |
| death | −(6 × time left + 4) |
| stagnation (< 48 units moved in 16 decisions) | −0.02 per step |
| USE on a locked door holding no key | −0.02 |
| time | −0.001 per step |

The objective is the exit if it is reachable with the keys held, otherwise the
nearest needed key. Distances come from an 8-connected wavefront distance field
on the teacher's nav grid, cached per map and key state.

### The three stages behind the shipped weights

| Stage | Run | Start from | Envs | Key settings | Snapshot used |
|---|---|---|---:|---|---|
| A | `ppo_ego1` | BC | 36 | lr 1e-4, KL to BC 0.3→0.02, entropy 0.003; reward without seen-floor/stagnation terms, key +3, death −(3·t+1), damage 0.01 | update 50 (230k steps) |
| B | `ppo_ego5` | A u50 | 72 | lr 2e-4, KL 0.03, imitation 0.2→0.05, entropy 0.005, curriculum floor 0.6; + seen-floor, stagnation, locked-USE terms, key +5 | update 100 (0.92M steps) |
| C | `ppo_ego6` | B u100 | 72 | as B, KL anchored to A u50, imitation 0.15→0.05; death −(6·t+4), damage 0.02 | **update 50 (0.46M steps), shipped** |

```bash
D=training/data; DEMOS=$D/demos_v2_nav,$D/demos_v2_arena,...   # same 13 datasets as BC
uv run python -m training.ppo2 --init training/runs/bc_ego1/ckpt.pt --out training/runs/ppo_ego1 \
    --workers 9 --envs-per-worker 4
uv run python -m training.ppo2 --init training/runs/ppo_ego1/ckpt_u0050.pt --out training/runs/ppo_ego5 \
    --workers 9 --envs-per-worker 8 --minibatch-windows 96 --epochs 3 --lr 2e-4 --ent 0.005 \
    --kl-beta 0.03 --kl-beta-final 0.03 --mastered-weight 0.6 --bc-data $DEMOS --bc-weight 0.2 --bc-weight-final 0.05
uv run python -m training.ppo2 --init training/runs/ppo_ego5/ckpt_u0100.pt \
    --anchor training/runs/ppo_ego1/ckpt_u0050.pt --out training/runs/ppo_ego6 \
    --workers 9 --envs-per-worker 8 --minibatch-windows 96 --epochs 3 --lr 2e-4 --ent 0.005 \
    --kl-beta 0.03 --kl-beta-final 0.03 --mastered-weight 0.6 --bc-data $DEMOS --bc-weight 0.15 --bc-weight-final 0.05
```

Stage A's reward is the table above without the seen-floor, stagnation and
locked-USE terms, with key +3, death −(3·t+1) and damage −0.01/HP. The trainer
now also has an out-of-reach USE penalty and a weight EMA (`--ema`), added after
the shipped snapshot was taken.

## 7. Selection and shipping

```bash
uv run python tools/auto_eval.py --run training/runs/ppo_ego6 --baseline <current-agent val run>
uv run python -m training.suite.run --agent --primary <candidate.pt> --combat none --split val \
    --per-category 10 --seeds 2 --workers 8 --out training/runs/suite/val2s_candidate
uv run python -m training.suite.compare --baseline <baseline run> --candidate <candidate run>
uv run python -m training.suite.run --agent --primary <candidate.pt> --combat none --split test \
    --per-category 10 --workers 8 --allow-test --out training/runs/suite/test_candidate   # logged
uv run python validate.py
```

Promotion requires the paired 95% CI of the tier-weighted proxy difference to
exclude zero. A category regression larger than 10 points with a CI excluding
zero blocks it. The shipped checkpoint is saved with
`config = {arch: "ego", temperature: 1.0}`.

## Code map

```
training/
  maps/gen_wad.py          procedural node-less WADs, tiers 1–4
  maps/drills.py           corridor doors, key alcove, key chain, backtrack layouts
  maps/doorroom.py         door on a room wall, random start pose
  maps/prepare_freedoom.py Freedoom variants (raw / doors-open / no monsters)
  teacher/                 privileged expert: mapdata, navgrid (A*), teacher (door FSM, combat)
  collect.py               teacher demonstrations + auxiliary and goal labels
  egomaps.py               replay demos through EgoMap; bit-pack 16-channel maps
  bc.py                    behaviour cloning (--arch ego)
  ppo2.py                  parallel PPO: asymmetric critic, shaping, imitation, KL anchor, EMA
  evaluate.py              episode runner, proxy score; coverage.py sector tracking
  obs.py, splits.py        observation flattening; seed ranges and Freedoom splits
  suite/                   spec, generate, run, metrics, compare, leakage, probes
tools/
  auto_eval.py             score every PPO snapshot on validation
  record_videos.py         gameplay clips through the real agent
  test_agent_contract.py   NaN/inf/mask corruption + determinism stress test
```

Earlier-design tools kept for reproducibility but outside this pipeline:
`ppo.py`, `dagger.py`, `moe.py`, `bench.py`, `suite/hybrid.py`.

Data and checkpoints are gitignored: `training/data/demos_*` (shards, `ego/`
maps), `training/data/suite/<split>` (WADs + manifests), `training/runs/<run>`
(checkpoints, `train.jsonl`, `auto_eval.jsonl`).
