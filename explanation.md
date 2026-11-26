# DoomForge, explained

This is the one document to read before talking to the judges. It explains what
the agent is, how every part works, how it was trained, how it was measured, and
the questions you are likely to be asked. Every number here is reproducible with
commands in [`docs/TRAINING.md`](docs/TRAINING.md) and [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 1. The problem in one page

Track 1 gives an agent **unseen DOOM maps** and scores it on: reaching the exit
(largest), picking up keys and opening doors/locks, exploring unique areas, and
combat, minus self-damage and early death. Harder map tiers weigh more.

The agent never touches the game engine. Every 4 game tics it receives one
observation dictionary and must return 9 buttons:

| Input | What it is |
|---|---|
| `rays` | 64 distances over 360° (index 0 ahead, counter-clockwise) and a hit type: none, wall, door, locked door, enemy, item |
| `visible_entities` | up to 32 things in a 90° field of view: distance, sin/cos of bearing, relative velocity, class (enemy, item, key, …), mask |
| `vitals` | 17 numbers: health, armour, weapon, ammo, six key flags, velocity forward/right/up |
| `rgb` | 120×160 colour image |

| Output | `MultiBinary(9)`: forward, back, strafe left/right, turn left/right, attack, use, next weapon |
|---|---|

**Hard parts.**

1. **No position.** Observations are ego-centric; there are no map coordinates.
   To explore, the agent has to build its own memory of where it has been.
2. **The exit is invisible to structured data.** No ray type or entity marks
   it. It can only be recognised in pixels (exit signs, switch textures).
3. **Key and lock colours are hidden.** Every key is just `key`, every lock just
   `locked door`. Colour is only in pixels.
4. **Doors are fiddly.** `USE` fires on the button-down edge, and pressing it on
   a moving door reverses it. You must be within ~64 units and facing the door.
5. **Budget.** One CPU thread, 40 ms average per decision, 2 GB VRAM, 8 GB RAM.
6. **Rules.** Actions must come from trained parameters. No A*/BFS planning, no
   hand-written rule trees, no `atan2` aimbots in the submitted agent. Anything
   goes at training time.

---

## 2. The agent in one paragraph

**DoomForge is a single 2.0 M-parameter recurrent network (Atlas-E).** Each
decision, a deterministic ego-motion map turns the rays and the agent's own
movement into a heading-up picture of what it has seen and where it has walked.
Seven small encoders read rays, entities, vitals, pooled RGB, the previous action, the
map and the pose; a learned goal head predicts "where should I be going"; a GRU
carries memory across time; six factorised heads output the buttons, sampled
with a seeded random generator. It was trained by **behaviour cloning** on
436k steps of demonstrations from a privileged teacher, then **parallel PPO**
with an asymmetric privileged critic, dense shaping and a teacher imitation
term, on a 1,966-map training split of our 14-category diagnostic suite. It
runs in **1.6 ms per decision** on one CPU thread.

```
observation ──► sanitise ──► EgoMap (dead reckoning + projection) ──► 16×32×32 heading-up map + pose
     │                                                                        │
     ├─ rays ─────────► circular Conv1d ──────────────────────────► 128 ─┐   │
     ├─ entities ─────► type embedding + self-attention + pooling ─► 128 ─┤   │
     ├─ vitals ───────► MLP ───────────────────────────────────────► 64 ─┤   │
     ├─ RGB 30×40 ────► Conv2d ×3 ─────────────────────────────────► 128 ─┤   │
     ├─ prev action ──► MLP ───────────────────────────────────────► 32 ─┤   │
     │                     map ─► Conv2d ×4 (first at full res) ───► 192 ─┤◄──┘
     │                     pose ─► MLP ────────────────────────────► 32 ─┤
     │                                                     704 ──► goal head ─► goal 8 ─► MLP ─► 32
     │                                                     704+32 ─► Linear ─► GRU(384) ─► MLP
     │                                                                          │
     └──────────────────────────────────── six action heads ◄──────────────────┘
                                           move 3 · strafe 3 · turn 3 · attack 2 · use 2 · next 2
                                           (+ value, + auxiliary heads used only in training)
```

---

## 3. Perception: how each input is encoded (`submission/model.py`)

| Module | Input → output | Why it is built this way |
|---|---|---|
| **RayEncoder** | 64 × (distance, one-hot hit type, sin/cos of ray angle) → 3 circular Conv1d → mean+max pool → 128 | Rays wrap around (ray 63 is next to ray 0). Circular padding preserves that topology, so "wall to the left" is the same pattern wherever it appears. |
| **EntityEncoder** | 32 × (5 features + 16-d type embedding) → MLP → one multi-head self-attention block → masked mean+max → 128 | The entity list has a variable length and no natural order. Attention + pooling is permutation-invariant and handles 0 to 32 entities. A flattened vector would not be. |
| **VitalsEncoder** | 17 → MLP → 64 | Health, ammo, keys held, own velocity. |
| **RGBEncoder** | 120×160 RGB subsampled 4× to 30×40 → 3 stride-2 convs → 128 | Colour is the only source of exit signs and red/blue/yellow lock colours. 30×40 keeps colour at 3.6 KB per frame. |
| **PrevEncoder** | last action (9 bits) + "USE was pressed" + turn-held counter → 32 | Doom's turn speed depends on how long a turn has been held, and USE is edge-triggered, so the policy needs to know what it just did. |
| **EgoMapEncoder** | 16×32×32 map → conv 3×3 stride 1 → three stride-2 convs → 4×4×64 → 192 | The first full-resolution layer keeps one-cell features (a door in a wall, a frontier cell). |
| **PoseEncoder** | 6 (dead-reckoned x, y, sin/cos heading, velocities) → 32 | Absolute progress relative to the spawn point. |
| **GoalHead + GoalEncoder** | 704 → 64 → goal 8-vector (dx, dy, distance, sin/cos bearing, type, mask) + 4-way type → re-encoded to 32 | Trained to predict the teacher's current objective (exit, key, frontier). At inference it is predicted from observations only and fed back as an input: a learned "where am I going" signal. |
| **Fuse + GRU(384) + trunk** | 736 → 384 → GRU → 384 | The GRU is the time memory (door timing, remembered threats, "I have the key now"). |
| **Actor** | 384 → 15 logits in six heads | Factorised actions: 3×3×3×2×2×2 = 216 meaningful combinations instead of 512 bit patterns, many of them contradictory (forward + back). Still emitted as a valid `MultiBinary(9)`. |
| **Value + auxiliary heads** | 384 → V, and 35 auxiliary outputs | Training-only signals: exit visible / bearing / distance, key colours visible, keys held, enemy visible, door ahead, damage over the next 8 decisions. They shape the encoder to represent things the observation hides. They are never used to act. |
| **Privileged critic** | 384 + 16 privileged features → 256 → 1 | PPO baseline only (see §6). Stored in the checkpoint, never evaluated by the agent. |

Parameter count: **2,000,735 acting parameters** (GRU 887k, map encoder 294k,
fuse 283k, RGB 222k, trunk 148k, …) plus a 103k training-only critic.
Inference: ~0.7 ms network + ~0.55 ms map per decision on one CPU thread.

---

## 4. Memory: the ego-motion map (`submission/mapmem.py::EgoMap`)

This is the most important piece of engineering in the agent, because it turns
a partially observable problem into one where "where haven't I been?" is
visible in the input.

### 4.1 Dead reckoning (no privileged state)

The agent knows its own actions and velocity, so it can integrate its pose:

- **Heading.** Doom turns at a rate that depends on how many consecutive tics
  a turn button has been held (in either direction). We measured the per-decision
  turn deltas (0°, 3.516°, 7.031°, 12.305°, 14.063° depending on warm-up and run
  length) and replay them from the executed action. Error: **0.02° after 500
  steps**.
- **Translation.** Doom's per-tic physics is: momentum += thrust, position +=
  momentum, momentum ×= 0.90625 (friction). The `vel_forward/vel_right` vitals
  give the momentum at the end of each 4-tic step. We recover the constant
  thrust from the start and end momentum and integrate the four tics exactly.
- **Wall pinning.** When the player pushes into a wall, the engine keeps
  reporting momentum but the player does not move. We clip each step's
  displacement to the free distance the previous rays measured in the direction
  of travel (minus the 16-unit player radius).
- Result: **~30 units of position error after 500 decisions** (a door is 64
  units wide).

### 4.2 The grid

A 160×160 allocentric grid at 64 map units per cell (10,240 units across),
origin at the spawn. Eight binary channels:

| Channel | Written from |
|---|---|
| free | points along every ray before its hit |
| wall | ray endpoints with hit type wall |
| door / locked door | ray endpoints with those hit types; cleared when a later ray passes through (door opened) |
| item / key | item ray hits and visible item/key entities projected by distance and bearing |
| trail | cells the agent physically occupied |
| frontier | free cells next to never-observed cells |

### 4.3 The readout

Two **heading-up** 32×32 crops centred on the agent: fine (64 units/cell,
2,048 units across, the ray range) and coarse (256 units/cell, max-pooled,
8,192 units across). Together 16×32×32.

Heading-up means "an unexplored frontier ahead-left" is the same pixel pattern
whatever direction the agent faces, which is something a small CNN can learn.
A world-aligned map would force the network to learn rotation from a sin/cos
input.

**Why this is legal.** It is deterministic perception: projecting the agent's
own observations at its estimated pose. There is no search, path planning or
decision logic in it; the network decides what to do with the picture.

---

## 5. Acting (`submission/agent.py`)

Per call to `act(observation)`:

1. **Sanitise**: NaN/inf replaced, hit/entity types clipped, so a corrupt
   frame cannot poison the map.
2. **Update the EgoMap** with the observation and the action executed last step.
3. **Forward pass** through Atlas-E with its recurrent state.
4. **Sample** each of the six heads from the network's softmax (temperature 1)
   using a `torch.Generator` that is re-seeded on every `reset()`. The same map
   and engine seed therefore always produce the same actions.
5. Return the 9-button vector.

**Why sample instead of argmax?** PPO trains a stochastic policy. At near-ties
(e.g. "turn left" vs "turn right" at a junction), argmax picks the same answer
forever and the agent freezes; we measured 90%+ no-op episodes under argmax and
+1.4 tier-weighted proxy points from sampling with the same weights. Sampling is
still purely the trained network's output, and seeding keeps it reproducible.

`reset()` clears the GRU state, previous-action bookkeeping and the map. Any
exception inside `act` returns a valid zero action and resets, so the agent
never crashes a map. No network access, no disk writes, no imports from
`training/`.

(The agent can still load older checkpoint formats and an optional routed
combat expert if those weight files are present; the shipped configuration is a
single Atlas-E network, `route_combat=false`.)

---

## 6. How it was made: the pipeline

```
 map generators ─► diagnostic suite (14 categories × train/val/test/ood)
       │                       │
       ▼                       ▼
 privileged teacher ─► demonstrations (436k steps) ─► ego maps regenerated offline
                                                           │
                                                           ▼
                                                 behaviour cloning (Atlas-E)
                                                           │
                                                           ▼
                                   parallel PPO (asymmetric critic, shaping, imitation term)
                                                           │
                                                           ▼
                           auto-evaluation on val ─► paired gate ─► one logged test look ─► ship
```

### 6.1 Maps (`training/maps/`, `training/suite/spec.py`)

- **Procedural generator** (`gen_wad.py`): rooms carved by recursive splitting on
  a 64-unit tile grid, doors on split walls, coloured locks with matching key
  things, a walk-over exit line with a visible exit-sign wall, randomised
  textures and lighting. Four tiers: arenas with monsters, multi-room
  exploration, key→lock progression, and gauntlets (two keys, loops, scarce
  resources). Maps are written as node-less binary WADs; ZDoom builds the BSP at
  load time, so there is no external node builder.
- **Drills** (`drills.py`, `doorroom.py`): corridor doors, a door on a room wall
  from a random pose, key alcove → locked door → exit, two keys → two locks,
  key behind the spawn (backtrack). These guarantee the lock is on the only
  path, so the key skill is actually required.
- **Freedoom**: 32 hand-made maps; MAP01–16 train, 17–22 validation, 23–32 test.

### 6.2 The diagnostic suite (`training/suite/`)

The official scorer is hidden, so we built a harder, more diagnostic proxy.

| Category | Tier | Tests |
|---|---|---|
| C0 sanity | 1 | walk to a visible exit |
| C1 duel / C2 arena | 1 | 1 vs 3–6 monsters, aiming, cover |
| C3 corridor door / C4 room door | 1 | USE on a door straight ahead / on a room wall |
| C5 key → lock | 3 | one key, one cut-edge lock |
| C6 two keys | 4 | two keys, two locks, order |
| C7 exploration | 2 | 8–13 rooms, hidden exit |
| C8 backtrack | 3 | key behind the spawn |
| C9 ambush · C10 scarce ammo · C11 gauntlet | 3–4 | combined skills under pressure |
| C12 Freedoom · C13 OOD | 4 | hand-made maps · bigger maps, rotated starts |

Splits by seed: train 0+ (150 maps per category, 1,966 total), validation
10,000+, test 30,000+, OOD 40,000+; Freedoom by map number. Every WAD is hashed;
train ∩ val and train ∩ test are empty. Test runs need `--allow-test` and are
logged in `test_touch.log`.

**Proxy score** (our assumption about the hidden weights): exit +10, key +3,
lock +3, door +0.5, new sector +0.3 (capped at +10), kill +0.6, damage −0.01/HP,
death −(3·time-left + 1). Headline = tier-weighted mean (1.0/1.2/1.5/2.0) with a
stratified bootstrap 95% CI over maps.

### 6.3 The privileged teacher (`training/teacher/`, training only)

Reads ground-truth pose, keys and door heights, plus the WAD geometry:

- a 16-unit navigation grid from linedefs (door faces passable, locked doors
  only with the key, agent radius inflation);
- A* with door costs and key gating, choosing exit → needed key → exploration;
- a door state machine: approach, align, press USE once, wait, cross;
- circle-strafing combat, projectile dodging, weapon switching;
- labels for every auxiliary head and the goal vector.

It never runs at evaluation; `submission/` never imports it.

### 6.4 Demonstrations and behaviour cloning (`training/collect.py`, `training/bc.py`)

- 13 datasets, 436k steps: door-free navigation, arenas, door drills, key, key
  chain, backtrack, corridor/room doors, multi-room exploration.
- Each step stores the observation, the teacher action, the labels and the goal.
  Ego maps are regenerated **offline** by replaying the stored observations and
  actions through `EgoMap` (`training/egomaps.py`), bit-packed to 2 KB/step.
- Loss: cross-entropy on the six heads (USE up-weighted near doors) + 0.2 ×
  auxiliary losses + 0.5 × value + goal regression/classification. 128-step
  windows, 5 epochs, AdamW 3e-4, on Apple MPS. Validation loss 0.63.

### 6.5 PPO (`training/ppo2.py`)

Behaviour cloning copies the teacher, but the teacher is omniscient and its
choices are not always reproducible from the agent's view. PPO optimises the
task itself.

- **Scale.** 9 worker processes × 8 environments; the learner batches inference
  on MPS and double-buffers (half the workers step while the other half's
  actions are computed). ~500 steps/s on a 10-core laptop.
- **Correct returns.** GAE (γ 0.995, λ 0.95) bootstrapped from the value of the
  next observation; per-batch advantage normalisation; recurrent 32-step windows
  that reset the GRU at episode boundaries inside the window.
- **Asymmetric critic.** The value baseline sees 16 privileged features: time
  left, geodesic distance and bearing to the current objective, keys held,
  coverage, health, ammo, … This removes most of the variance a blind critic
  would have. Time-left makes the tick cap part of the state, so a time-out is a
  proper terminal. Critic features are detached: only the policy loss and the
  regularisers shape the shared network.
- **Rewards (training only).**

  | Term | Value |
  |---|---|
  | exit | +10 + 2 × time left |
  | key / lock / door opened | +5 / +3 / +0.5 |
  | new sector / new 64-unit cell | +0.3 / +0.01 |
  | newly seen floor in the agent's own map | +0.004 per cell (≤0.1/step) |
  | geodesic progress to the current objective | +0.003 per unit closer (±0.1/step) |
  | kill | +0.6 |
  | damage / death | −0.02 per HP / −(6 × time left + 4) |
  | stagnation (moved <48 units in 16 decisions) | −0.02 per step |
  | USE on a locked door without a key | −0.02 |
  | time | −0.001 per step |

  The geodesic distance comes from a wavefront distance field on the teacher's
  nav grid, toward the exit if reachable with the keys held, otherwise toward the
  nearest needed key. The seen-floor and stagnation terms are what made the
  multi-room maps learnable: without them the agent parked in a corner or at a
  locked door.
- **Regularisers.** An exact KL anchor to a reference policy (its hidden state
  is tracked along the same trajectories), plus a teacher **imitation term**
  (cross-entropy on random demo windows) that keeps door/key skills grounded
  while PPO explores.
- **Curriculum.** Maps are drawn from the 14 training categories, weighted
  towards categories that are not yet solved (weight 0.6 + unsolved share), plus
  15% freshly generated maps.

The shipped weights are the result of BC, then three PPO stages, each started
from the previous stage's best checkpoint, about 1.6M PPO steps in total (exact
settings in `docs/TRAINING.md`).

### 6.6 Selection and the promotion gate

`tools/auto_eval.py` scores every snapshot on the full validation split (136
episodes) through the real `DoomAgent`. A candidate is promoted only if:

1. the **paired** bootstrap CI of the tier-weighted proxy difference vs the
   current agent (same maps) is above zero, and
2. no category regresses by >10 points with a CI excluding zero.

Finalists are re-scored on two engine seeds; the winner gets **one** logged look
at the untouched test split.

---

## 7. Results

**Held-out test split** (140 episodes, never trained on or tuned against):

| | Phase-4 hybrid (3 networks + rules) | **DoomForge Atlas-E (shipped)** |
|---|---:|---:|
| Tier-weighted proxy | +6.11 | **+8.48** |
| Paired difference | | **+2.37** (95% CI +1.47…+3.24) |
| C1 duel | 20% | **90%** |
| C4 room door | 30% | **100%** |
| C7 exploration | 0% | **10%** |
| C8 backtrack | 30% | **100%** |
| C0, C3, C6 | 100% | 100% |
| C5 key → lock | 100% | 80% |
| C2 arena | 60% | 50% |
| Coverage on C9 / C10 / C13 | 7% / 5% / 6% | **28% / 16% / 17%** |

Validation (two engine seeds, 272 episodes): **+8.28** vs +6.11 for the hybrid
(paired +2.17, CI +1.23…+3.08).

Official validator: PASS, init 0.4 s, **p50 1.6 ms / p99 1.9 ms** per decision.

**What still fails:** exits on C9–C13 (ambush, scarce ammo, gauntlet, Freedoom,
OOD) are 0%. The agent earns partial credit there (coverage, kills, doors,
keys), but these maps combine long exploration with dangerous combat and need far
more training than the ~1.6M PPO steps a laptop allows.

---

## 8. Rule compliance, point by point

| Rule | How DoomForge complies |
|---|---|
| Decisions from trained parameters | Every button is sampled from the network's output distribution. No rule tree, no override, no reflex in the shipped agent. |
| No A*/BFS/Dijkstra in the agent | Path search and distance fields exist only in `training/` (teacher, reward shaping, SPL metric). `submission/` never imports `training/`. |
| No trigonometric aimbot | The agent never computes a bearing to steer by. `atan2` appears in `mapmem.py` only to *project* sightings into the map. |
| Not off-the-shelf SB3 | Custom architecture, custom BC, custom parallel PPO with an asymmetric critic. No RL library is used. |
| Engine isolation | The agent sees only the observation dict; privileged state is used only for training rewards, labels and the critic. |
| Reproducible | Seeded sampling; identical observations give identical actions. |
| Budget | 1.6 ms of 40 ms; ~8 MB weights; well under RAM/VRAM limits. |

---

## 9. Questions you will likely be asked

**Why not just train PPO from pixels?** Sparse rewards, long horizons and one
CPU thread at inference. A privileged teacher gives a working policy in under an
hour of behaviour cloning; PPO then improves on what the teacher can't do from
the agent's point of view.

**Isn't the map a form of planning?** No. It stores what was observed, at the
pose the agent itself computed: perception and memory. Nothing in it chooses a
target or a route; the network reads the picture and decides.

**How does the agent find the exit if it isn't in the observations?** Three
ways: the RGB encoder is trained (auxiliary heads) to recognise exit signs and
estimate their bearing and distance; the goal head predicts the current
objective; and PPO rewards reaching it, so exploration toward unseen space plus
visual recognition becomes the learned strategy.

**How does it know which key opens which door?** It can see door and key
colours in RGB (trained by the key-colour auxiliary head), remembers keys and
locked doors in the map's key and locked-door channels, and its vitals tell it
which keys it holds.

**What is the privileged critic, and isn't it cheating?** It is the PPO value
baseline, used only to compute advantages during training. It never influences
an action at evaluation and the agent does not even evaluate it. This is
standard asymmetric actor-critic.

**Why sample actions? Isn't that random?** It is the policy the network was
trained to be. Argmax of a stochastic policy is a different, untrained policy
that freezes at ties. The generator is seeded per map, so replays are
identical.

**How do you know you didn't overfit the benchmark?** Separate seeds and WAD
hashes for train/val/test, zero hash overlap, selection on validation only, and
a single logged test look per shipped decision (`test_touch.log`). The test
gain (+2.37) is consistent with validation (+2.17).

**What is the biggest weakness?** The combined-skill maps (ambush, gauntlet,
Freedoom). Exits there need more samples; our PPO ran at ~500 steps/s on a
laptop for ~1.6M steps.

**What's novel?**
- Exact Doom dead reckoning: the turn-run model, the per-tic friction
  integration and the wall-pin clip, measured against ground truth.
- A heading-up two-scale ego map with trail and frontier channels, legal as
  perception.
- A learned goal head fused back into the policy.
- An asymmetric critic with geodesic objective features.
- Seen-floor and stagnation shaping.
- A 14-category diagnostic suite with a paired statistical promotion gate.

---

## 10. File map

```
submission/            what the scorer runs
  agent.py             DoomAgent: sanitise → EgoMap → network → seeded sampling
  model.py             Atlas-E (and the older Atlas-G) network definition
  model_v1.py          oldest checkpoint format (compatibility only)
  mapmem.py            EgoMap: dead reckoning + heading-up two-scale map
  weights/doomforge.pt shipped Atlas-E checkpoint (Git LFS)
training/              never imported by the agent
  maps/                procedural generator, drills, Freedoom preparation
  teacher/             privileged teacher: nav grid, A*, door state machine, combat
  collect.py           teacher demonstrations + labels
  egomaps.py           offline ego-map regeneration, bit packing
  bc.py                behaviour cloning
  ppo2.py              parallel PPO (asymmetric critic, shaping, imitation, EMA)
  evaluate.py          episode runner, proxy score, sector coverage
  suite/               14-category suite: spec, generate, run, metrics, compare, probes
tools/
  auto_eval.py         scores every PPO snapshot on validation
  record_videos.py     gameplay clips through the real agent
  test_agent_contract.py  NaN/inf/mask corruption + determinism stress test
validate.py            official validator (untouched)
```

## 11. Commands you may be asked to run

```bash
uv sync                                   # environment
uv run python validate.py                 # official validator
uv run python -m training.suite.run --agent --split val --per-category 10 \
    --workers 8 --out training/runs/suite/val_check      # benchmark the shipped agent
uv run python -m training.suite.metrics --run training/runs/suite/val_check
uv run python tools/record_videos.py      # regenerate the gameplay clips
```
