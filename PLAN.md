# DoomForge — Track 1 Plan (Revision 2, 24-hour cut)

> **Revision note — 2026-11-26.** This document **replaces** the previous
> `PLAN.md`. The old text is superseded and should not be treated as authoritative.
>
> **Why we are changing it.** The first plan was written as a multi-phase project and
> assumed the structured ray/entity observations were sufficient to find exits. A
> separate strategy review, followed by direct verification against the official
> wrapper and the current model, found three material gaps:
>
> 1. **Exits are invisible to the structured observations.** Rays have no exit hit
>    type, visible entities have no exit class, and the wrapper only learns that the
>    exit was reached when the episode ends. The current model has no RGB input and
>    no exit head, so it cannot recognise an exit switch or exit line.
> 2. **Key colour and locked-door colour are hidden until pickup.** Every key is
>    reported as `key`, and every lock is reported as `locked_door`. The agent cannot
>    know which key opens which door from structured observations alone.
> 3. **Turn odometry was oversimplified.** The old plan used a single
>    `14.06°/decision` figure. The real heading increment depends on how long the
>    turn button has been held and whether the episode warm-up tic has passed.
>
> The review also correctly argued that the automated score decides the top 10, that
> the exact scoring weights are hidden, and that a safety submission must exist before
> the ambitious memory system is finished. This revision keeps the assets and the
> expert-seeded BC → PPO pipeline, but changes the input set, memory design,
> sequencing, evaluation, and schedule.
>
> **What changed at a glance:**
>
> | Area | Old plan | Revision 2 | Why |
> |---|---|---|---|
> | Exit perception | Rays/entities; pixels an ablation | RGB + privileged exit head is P0 | Exit is invisible to structured observations |
> | Colour | Key/lock flags only | RGB key-colour head, or try-all-doors fallback | Colour is hidden until pickup |
> | Memory | Learned 48×48 ConvGRU map | Deterministic coarse map + event tokens; learned write is a stretch goal | Faster to build, easier to debug, easier to defend to judges |
> | Sequencing | BC → PPO → memory | Train A first; ship A by hour 12; promote B+ lite only if it wins | Safety submission before ambition |
> | Turn model | 14.06°/decision | Full measured turn table + `turn_held` + previous action | Measured engine behaviour |
> | Scoring | Specific assumed weights | Proxy scorer with configurable weights; report components | Official weights are hidden |
> | Evaluation | Proxy suite | Explicit train/val/OOD/Freedoom splits + causal memory tests | Prevent leakage and prove memory is used |
> | Schedule | Multi-phase | 24-hour schedule with roles and hard gates | Hackathon deadline |
>
> **What we are keeping from the old plan:** the official template and untouched
> `env/`, the procedural binary map generator, the privileged teacher, the BC/PPO
> approach, the factorised action heads, the validator-first discipline, and the
> do-not-add-CI/CD rule.
>
> **What we are deliberately rejecting from the review:** from-scratch PPO (we already
> have an expert and a BC pipeline), cutting the map generator (already built and
> validated), cutting scan-matching without measuring drift first, switching to UDMF
> (binary is required for things to spawn), and promising 300–500M PPO steps or three
> training seeds on this hardware.

This document is the plan, not the lab log. Measured numbers and dead ends live in
[`progress.md`](progress.md); the architecture reference will live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) once written.

---

## 0. Status at revision time

| Asset | State |
|---|---|
| Official template | Cloned; `env/`, `validate.py`, `submission/agent.py` contract untouched |
| Python environment | Python 3.13, ViZDoom 1.3.3, Gymnasium 1.4; local torch install is currently broken (`ModuleNotFoundError: torch._strobelight`) and must be repaired before training |
| Local validator | **PASSED** with 2 host-only warnings: macOS cannot measure `/proc` RAM or pin a CPU |
| Procedural maps | Binary generator, 4 tiers, 120/120 maps valid and door-metadata exact |
| Real maps | Freedoom 2: 32 maps × 3 variants (raw, doors-open, doors-open + no-monsters) |
| Expert | Privileged A* + door/exit/combat controller: **23/24 exits** on door-free maps; doors remain unreliable in multi-room maps |
| Data pipeline | Episode shard format, collection and BC/evaluation code written; no demos collected yet |
| Model | `submission/model.py`: ray CNN + entity attention + vitals MLP + GRU(320), factorised heads; **no RGB, no exit head, no previous-action input** |
| Memory | Not implemented yet |
| Submission | `submission/agent.py` is still the random baseline; no weights committed |

**Known blockers before training can start:**

1. Repair torch in `.venv` and verify a GRU forward pass on CPU and MPS.
2. Add RGB and exit labels to the observation/collection pipeline.
3. Make generated exits and locked doors visually distinguishable in RGB.
4. Fix `.gitignore`: `training/data/` and `training/runs/` are not currently ignored,
   and `.npz` shards are LFS-tracked. Collecting data without fixing this will bloat
   the repository.
5. The current model cannot see exits; it is a starting point, not a shippable agent.

---

## 1. Thesis

Win with a **small, fast, recurrent policy that can see exits, remembers where things
are, and ships in two stages**:

1. **A baseline (safety):** recurrent multimodal PPO with RGB, entity attention, and
   exit/key-colour auxiliary heads. No explicit map. This is pushed by hour 12.
2. **B+ lite (differentiator):** A plus exact-heading odometry, a deterministic coarse
   map, and event tokens for locked doors, keys, exit cues, and frontiers. Promoted
   only if it beats A on held-out maps.

The memory is **perception**, not planning. It is built from observations and
proprioception; the learned policy reads it. There is no A*, Dijkstra, BFS, or
frontier pathfinding at inference.

**Two rules for the next 24 hours:**

- Score first, story second. The automated score decides the top 10; architecture
  novelty matters for the desk review and final judging.
- Never let the ambitious version delay the safety submission.

---

## 2. What the brief actually rewards

Track 1 asks for an agent that plays **unseen** DOOM maps zero-shot through the
official `doom_env` wrapper.

- **Inputs:** 64 ego-centric raycasts with hit types, a depth buffer, a 17-value
  vitals vector, up to 32 visible entities in relative coordinates, and a 120×160 RGB
  frame. No global coordinates, no map, no god view.
- **Output:** a 9-bit `MultiBinary` action every 4 game tics.
- **Limits:** 2 GB VRAM, 8 GB RAM (including `/tmp`), **1 CPU thread**, 60 s init,
  10 s reset, 10 s first `act`, 5 s any `act`, **40 ms average compute per decision**.
- **Scoring:** tier-weighted, non-binary. Exit is the largest term, then progression
  (keys, doors, locks), exploration (unique sectors), and combat (kills), minus
  damage/death penalties. **Exact weights and tier multipliers are hidden.**
- **Tiers:** 1 combat/survival, 2 multi-room exploration, 3 keycard → locked door →
  exit, 4 full gauntlet.
- **Rules:** decisions must come from trained parameters. No rule trees, no classical
  planning, no `atan2` aimbots. Training-time scaffolding, privileged state, engine
  and map modifications are allowed. The evaluation engine is vanilla and the
  `env/` folder is ignored.

### Strategic implications

1. **Exit perception is the highest-value capability.** A perfect shooter that cannot
   recognise an exit loses the largest term. A mediocre shooter that finishes maps
   wins.
2. **CPU/GPU inference is a hard budget, but not the main risk.** A ~2–3M-parameter
   model with a small RGB encoder fits comfortably. The real risks are crashes,
   warm-up, threading, and NaN handling.
3. **Partial observability + backtracking is the real research problem.** Keys and
   locked doors are explicitly tested. A plain GRU forgets a door seen hundreds of
   steps ago; a deterministic map plus event tokens fixes that.
4. **The validator is a gate, not a score.** Maximise its margins (latency, RAM,
   repo size, determinism), but optimise the hidden score through a proxy suite.

---

## 3. Verified observation gaps and consequences

Everything here was checked against `env/doom_env/sensors.py`, `mapinfo.py`,
`wrapper.py`, `docs/CONTRACT.md`, and the current `submission/model.py`.

| Gap | Evidence | Consequence |
|---|---|---|
| **Exit is invisible** | `RAY_HIT_TYPES` has no exit class; `visible_entities` has no exit type; `mapinfo.py` parses door/lock specials only; `wrapper.py` sets `_exit_reached` only when the episode ends | The agent must learn exits from RGB. In training, parse exit linedef specials (11/51/52/124/84) with the privileged map data to produce labels |
| **Key colour is hidden until pickup** | `classify()` maps all six key classes to `key`; `KeyTracker` sets the colour flag only after pickup | Add an RGB key-colour head, or teach the policy to try remembered locked doors after each new key |
| **Locked-door colour is hidden** | `hit_type == 3` means any lock; the lock number is only in privileged map metadata | Same as above; RGB or try-all-doors |
| **USE reverses moving doors** | `docs/CONTRACT.md`: USE fires on button-down; pressing USE on a moving door reverses it | Add a `use_last` input flag and a useless-USE penalty; never spam USE |
| **Turn acceleration is stateful** | Measured: see Appendix A | Track `turn_held` and `prev_action`; do not use a single 14.06° constant |
| **Scoring weights are hidden** | `docs/EVALUATION.md` | Treat all numeric weights as proxy assumptions; report every component separately |
| **Coverage is sector-based** | `sector_polygons()` docstring says it is used by the scorer | Reward new sectors in training, not only new 64-unit cells |
| **Rays stop at monsters/items** | `sensors.py` ray-vs-circle tests | Rays double as a 360° threat/item radar, including enemies outside the 90° entity list |

---

## 4. Architecture

### 4.1 A baseline — the safety submission

A is a recurrent multimodal policy with no explicit map. It must be trainable and
validatable before memory work begins.

| Stream | Encoder | Output |
|---|---|---|
| 64 rays | circular 1-D CNN over `[distance, 6 hit one-hots, sin θ, cos θ]` | 128 |
| ≤32 entities | type embedding + 1-layer multi-head self-attention + masked mean/max; attention-pooling query from vitals (and `h_{t-1}` if easy) | 128 |
| 17 vitals | 2-layer MLP | 64 |
| RGB 120×160×3 | grayscale/resize to 60×80; 3-layer stride-2 CNN | 128 |
| depth (optional) | 4-layer stride-2 CNN, 120×160 → 15×20 | 64 |
| previous action | 9 bits + `USE` pressed last step flag | 16 |

Fusion: concatenate (~464–528) → Linear 320 → GRU(320–384) → policy/value/aux heads.

**Auxiliary heads:**

- exit visible (binary);
- exit bearing bin (12 or 24 bins) and exit distance;
- damage expected over the next 8 steps;
- key colours visible (6-way multi-label), if RGB labels are available;
- keys held (6), enemy visible, door ahead (existing heads).

**Action heads:** move (3) × strafe (3) × turn (3) × attack (2) × use (2) ×
next-weapon (2) = 216 joint actions, emitted as a valid `MultiBinary(9)`. This is
action masking by construction and is allowed.

**Target size:** ~2–3M parameters with RGB. Target latency: p99 under 8 ms on one CPU
thread; benchmark CUDA as well, but CPU is the safe default.

### 4.2 B+ lite — deterministic memory + event tokens

B+ lite adds an explicit spatial memory to A. It is deliberately **not** a learned
ConvGRU over the whole map; that is harder to train, harder to debug, and harder to
explain under time pressure.

#### Pose without global coordinates

1. **Heading:** integrate the measured turn table using `turn_held` and the action
   history. Reset `turn_held` on release and at `reset()`. Handle both-turn-buttons
   and no-turn as zero. The first episode turn has special cases; calibrate the full
   table (Appendix A).
2. **Translation:** integrate ego-frame velocity from `vitals[14:16]`. Refit the
   displacement coefficients on generated and Freedoom maps; do not hardcode another
   project’s coefficients. Report drift curves.
3. **Correction (optional but recommended):** heading is exact, so scan matching only
   searches translation. Score the 64 ray endpoints against the map’s wall channel
   over a small window (e.g. ±16 units, ~81 candidates, vectorised, <0.5 ms). Accept
   a correction only when the match is sharply peaked.
4. **Discontinuities:** a large scan mismatch plus a velocity/height jump triggers a
   new map segment (teleporter/lift). Archive the old segment as event tokens; do not
   corrupt the live map.

#### Map

- **Resolution:** 64 units/cell is the compromise. A 96×96 or 128×128 grid covers
  6,144–8,192 units. The review’s 32 units/cell is acceptable only if the map is
  deterministic and read through two crops; do not run a full-map ConvGRU at that
  size.
- **Channels (~11):** free space, wall, door, locked door, visited count, recency,
  frontier, item/key seen, enemy last seen (decaying), exit likelihood (from the RGB
  auxiliary head), opened door.
- **Writing:** project rays and entity positions into the map at the estimated pose.
  No search, no planning.
- **Reading:** two egocentric crops rotated to the current heading:
  - local 32×32 at 64 units/cell (≈2,048-unit span);
  - global 32×32 at 128 units/cell (≈4,096-unit span).
  Pass both through a small CNN → 256 dims.

#### Event tokens

Up to 64 tokens, one per locked door, key, exit cue, or large frontier cluster.
Features (24): relative x/y, distance, sin/cos relative to heading, type one-hot,
age, visit count, keys held at sighting (6), resolved flag. Recompute the relative
fields from pose every step. Merge tokens within 64 units; evict the oldest resolved
tokens first. A query from the GRU state attends over the tokens → 128 dims.

This is the long-horizon fix: a locked door seen 700 steps ago still points the right
way, and the attention weights give a strong live-demo visual.

#### Optional learned write

Add 4 channels whose values come from a linear projection of the GRU state, written at
the agent’s cell (Neural Map style). Keep only if an ablation shows it helps. This is
the “learned memory” component; the rest of the writer is deterministic and should be
described honestly as such.

### 4.3 Why this is legal

`docs/RULES.md` allows preprocessing and encoding observations, action masking, frame
stacking, and learned memory modules. A deterministic writer that projects
observations into a map is perception; the policy decides what to do. There is no
frontier pathfinding, no goal search, and no classical planner at inference.

**Action item:** ask the organisers for a written yes on the deterministic map +
frontier-channel design before submission. Do not put any BFS/A*/Dijkstra code in
`submission/`; keep it under `training/` only.

### 4.4 Inference runtime

- Weights: one fp32 `state_dict` (~10–25 MB), Git LFS.
- `__init__`: load weights, `torch.set_num_threads(1)`, `eval()`, run 3 dummy forward
  passes, preallocate map/token buffers. No network, no disk reads after loading.
- `reset()`: zero GRU state, map, token bank, pose, `turn_held`, `prev_action`, and
  key tracker.
- `act()`: sanitise NaN/inf, clamp inputs, forward under `torch.inference_mode()`,
  argmax the six heads, map to `int8[9]`. Wrap in `try/except`; on failure return a
  valid zero action and log once.
- Handle all-zero entity masks with a learned null token or a safe zero embedding.
- No writes outside `/tmp`; no imports from `training/`, `vizdoom`, or the engine.
- Determinism: argmax by default. If sampling scores better on Val-ID, use a seeded
  generator and document it.
- Benchmark CPU and CUDA. Choose CPU if p99 is comfortably under 10 ms; avoid CUDA
  warm-up and VRAM risk unless the speedup is material.

---

## 5. Training plan

### 5.1 Map supply

The map supply decides generalisation more than model size. Keep the existing binary
generator; do not switch to UDMF for the main pool (UDMF loads, but things do not
spawn unless written as binary `THINGS`).

**Generated maps (bulk of training):**

- Tiers 1–4: arena, rooms, keycard → locked door → exit, two-key gauntlet.
- 64-unit tile grid, deterministic from `(seed, tier)`.
- Keep maps under 2,048 sectors.
- **P0 fixes:**
  - make exits visually distinguishable: use a visible exit switch/sign
    (`SW1EXIT`/`EXITSIGN`) or a clearly marked walk-over exit, not only an invisible
    line;
  - use coloured locked-door textures (`DOORBLU`, `DOORRED`, `DOORYEL`) instead of
    one generic door texture;
  - randomise exit appearance, door textures, wall/floor textures, light levels,
    start angle, start offset, monster mix, and item density.
- **Held-out split:** reserve seed blocks and generator configurations for validation
  and OOD tests; never train on them. Example: seeds 0–9,999 train, 10,000–10,999
  validation, larger/deeper OOD configurations held out entirely.

**Freedoom 2:**

- 32 maps × 3 variants (raw, doors-open, doors-open + no-monsters).
- **Split now:** use MAP01–MAP22 for training/demos, MAP23–MAP32 for held-out
  evaluation. Write the split to a file and make the collector read it. Never train on
  the held-out maps.
- Freedoom-Nav demos may use only the training maps.
- Optional Freedoom 1/community WADs only after verifying textures/IWAD compatibility
  and only if time remains. Not core.

### 5.2 Expert seeding and behaviour cloning

**Goal:** a policy that can navigate rooms, follow corridors, approach doors, pick up
keys, and reach exits; a strong initialisation for PPO.

1. **Navigation demos:** generated maps with doors removed and monsters removed, plus
   Freedoom training maps with door ceilings raised. Expert succeeds ~96% on
   door-free maps. Target 60–100k transitions total across the mix.
2. **Door/key drills:** short corridors with isolated doors, optional key alcove, and
   locked door. The multi-room expert fails doors; the drill expert works in
   isolation. Fix the teacher’s door state machine before collecting.
3. **Combat drills:** arenas with 1–3 monsters; staged from infinite ammo → real ammo
   → mixed monster types → ambushes from behind.
4. **BC loss:** cross-entropy on the six heads + auxiliary exit/key-colour/damage/key/
   enemy/door heads + a Monte-Carlo value head. Episodes as padded sequences; the GRU
   learns 32–300 step memory.
5. **Gate:** ≥70% exit on held-out door-free maps; ≥40% on door drills; no NaNs; BC
   runs end-to-end on CPU/MPS.

### 5.3 PPO fine-tuning

- **Algorithm:** PPO with GAE(0.95), γ=0.997, clipping 0.1–0.2, entropy bonus
  (0.01 → 0.003), advantage normalisation, value clipping, gradient norm 0.5,
  recurrent sequences with burn-in 32–64.
- **Vectorisation:** 8–10 worker processes, each with its own Doom env and CPU
  inference; periodic weight broadcast. Batch the model forward where possible.
  Benchmark before promising a step count.
- **Expected budget:** 50–150M environment steps on this hardware. Do not promise
  300–500M; the review’s estimate assumes a larger machine.
- **BC prior:** initialise from the best BC checkpoint; keep a BC auxiliary loss or
  KL anchor early in PPO to avoid catastrophic forgetting.

### 5.4 Reward (training only; uses `privileged_state()`)

Official weights are hidden. Use the following as a **configurable proxy**, not as
the scorer’s exact formula:

| Term | Value | Rationale |
|---|---|---|
| Exit | +10, plus up to +2 for remaining health/time | Mirrors the largest scorer term |
| Key picked up | +3, once per key | Progression |
| Locked door opened | +3, once per door | Progression |
| Ordinary door opened | +0.5, once per door | Stops door-bounce farming |
| New sector entered | +0.3, episodic | Matches the scorer’s sector coverage |
| New 64-unit cell | +0.01, episodic, total capped at +3 | Dense exploration signal that cannot outweigh the exit |
| Kill | +0.3 to +1 by monster type | Finite enemy count |
| Damage taken | −0.01 per HP | Small survival pressure |
| Death | `−(3·(1 − t/T) + 1)` | Early death costs more, mirroring the scorer |
| Wasted shot | −0.003 | Fired with no enemy near the crosshair |
| Useless USE | −0.005 | USE with no door/switch nearby or during door cooldown |
| Time | −0.001 per step | Tie-breaker; too small to drive suicide |
| Objective potential | `γΦ(s′) − Φ(s)`, `Φ = −geodesic distance to objective / 1000` | Potential-based shaping preserves the optimal policy; anneal to 0 by the end |

- Objective = nearest unheld key if a lock blocks the exit, else exit.
- Geodesic distance comes from a training-only BFS on privileged geometry; this is
  explicitly allowed as scaffolding and must not appear in `submission/`.
- **Do not** use the old raw `−0.0002 × BFS cells` penalty; it is not potential-based
  and can change the optimal policy.
- **Do not** clip per-step reward to `[−5, +10]`; normalise advantages and clamp only
  for numerical safety.

### 5.5 Curriculum

| Stage | Maps | Modifications | Advance when |
|---|---|---|---|
| 0. Validation | `two_rooms` + 20 generated | None | Wrapper, odometry, exit labels, and BC pipeline pass tests |
| 1. Combat | Arenas | Infinite ammo → real ammo → mixed types → ambushes | ≥80% arena clears; damage falling |
| 2. Navigation | 1–3 rooms, exit visible | Monsters off, doors auto-open | ≥90% exits in under 300 steps |
| 3. Exploration | 4–10 rooms, exit hidden | Monsters sparse, real doors | ≥70% exits; coverage plateaued |
| 4. Key progression | 1 lock, then 2–3 | Monsters sparse | ≥60% key → door → exit |
| 5. Combined | Full generator + Freedoom train split | Vanilla rules | Held-out generator seeds within 10% of train |
| 6. Generalisation | Held-out generator configs + Freedoom held-out | Vanilla rules | Held-out score stops improving |

Keep 20–30% of each batch from earlier stages to avoid forgetting. Once the pool is
large, optionally replace the hand schedule with Prioritized Level Replay; do not make
PLR a dependency for the first run.

### 5.6 PPO settings to start from

- 8–10 workers × 128-step rollouts.
- 4 epochs, 4 minibatches.
- Clip 0.1–0.2, lr 2.5e-4 with linear decay.
- Entropy 0.01 → 0.003.
- Value clip, grad-norm 0.5.
- Auxiliary loss weight 0.1 each.
- Burn-in 32–64 for recurrent sequences.

### 5.7 Gates

- **A baseline:** ≥50% exit on held-out tier 2; ≥25% on tier 3; positive combat K/D
  on tier 1; p99 latency <10 ms; validator passes.
- **B+ lite:** must beat A on the held-out proxy score, not just on training maps.
  If it does not, ship A.
- **Final:** best held-out proxy score, validator clean, repo clean, brief/demo ready.

---

## 6. Evaluation protocol

### 6.1 Splits

| Split | Content | Purpose |
|---|---|---|
| Train | Generated seeds A + Freedoom MAP01–MAP22 | Learning |
| Val-ID | Generated seeds B, same configuration | Checkpoint selection |
| Test-OOD-gen | Larger maps, deeper key chains, new monster mixes, unseen generator configs | Compositional generalisation |
| Test-Freedoom | Freedoom MAP23–MAP32 raw/open, never trained | Hand-made realism |
| Test-community | Optional small vanilla-compatible WADs | Final sanity check |

Run at least 3–5 engine seeds per map. One training seed is acceptable for the 24-hour
run; report it honestly. Add multi-seed training only if time remains.

### 6.2 Proxy scorer

Build `training/eval/proxy_score.py` (or extend `training/evaluate.py`) to mirror the
four stated categories using `privileged_state()` and `sector_polygons()`:

- exit (+ health/time bonus);
- keys, doors, locked doors;
- unique sectors visited;
- kills, damage taken, early death.

Weights are **configurable assumptions**. Report every component separately and never
claim the official weights.

### 6.3 Metrics

Exit completion rate, proxy score, coverage (% of sectors), keys collected, locked
doors unlocked, kills, damage taken, survival rate, steps to exit, redundant revisits
(cells visited more than 3 times / unique cells), odometry drift, and inference
latency (mean/p99/per-map total against budget).

### 6.4 Ablations

Run the subset that fits the time budget:

| Variant | Tests the claim |
|---|---|
| A baseline | recurrent multimodal PPO with RGB |
| B+ lite full | map + tokens help beyond the GRU |
| − map | tokens alone vs map alone |
| − event tokens | tokens fix long-horizon key → door return |
| − scan matching | correction matters on long maps |
| − RGB/exit aux | RGB is necessary for reliable exits |
| Attention → mean/max pool | attention helps multi-enemy combat |
| − episodic novelty | intrinsic reward drives coverage |

Do not block shipping on the full table. Report what was run and what was not.

### 6.5 Causal evidence that memory is used

- **Locked-door recall:** generated maps where the locked door is seen first and the
  key is far down a side branch. Measure return rate and steps-to-return with and
  without tokens.
- **Memory wipe:** zero the map and tokens immediately after key pickup. If success
  collapses, the agent was using memory.
- **Attention visualisation:** replay overlay showing token attention spike onto the
  remembered door when the key flag flips. This is the demo moment.
- **Linear probe (optional):** recover the true relative door position from token
  embeddings vs the GRU state.

---

## 7. 24-hour execution schedule

Assume the deadline is 24 hours from the hackathon start. Adjust relative times if the
actual deadline differs.

| Hours | P1 — RL / model | P2 — env / eval | P3 — memory / submission | Gate |
|---|---|---|---|---|
| 0–1 | Repo, validator, torch repair | Docker if available; local validator | Git LFS, `.gitignore`, weights dir | Random agent passes validator |
| 1–2 | Add RGB encoder, previous action, exit/key-colour/damage aux heads | Add RGB/exit labels to obs/collect; freeze splits | Add visible exits/coloured locks to generator; draft `agent.py` | A model runs end-to-end |
| 2–3 | BC first pass | Collect 20–30k transitions | Odometry table + unit test | BC loss falls; no NaNs |
| 3–5 | Train A BC; tune | Collect remaining demos; proxy scorer | Map + event tokens; pose drift test | A baseline checkpoint |
| 5–6 | Integrate B+ lite | Smoke test full pool | Latency/RAM check | Training starts by hour 6 |
| 6–12 | Main PPO run; monitor crashes only | Eval every 2h; held-out replays | Brief draft; attention overlay; validator on A | **Hour 12: A checkpoint pushed via LFS + validator passes** |
| 12–18 | Continue PPO/DAgger; lower entropy/lr if plateaued | Memory-wipe + recall tests; run-2 comparison | Brief/figure/demo | A second checkpoint beats A on held-out |
| 18–20 | Stop training; pick by held-out proxy | Final eval table | Attention-overlay video | Numbers for brief |
| 20–22 | Final weights; fresh clone | Final validator 3× | Brief PDF final | **Hour 22: freeze** |
| 22–24 | Q&A rehearsal | Final checks | Submit URL; demo rehearsal | Submitted |

### Roles

- **P1 — RL/model:** RGB encoder, exit/aux heads, model, PPO, training run,
  monitoring.
- **P2 — environment/evaluation:** RGB/exit labels, drills, map pool, splits, reward,
  vector env, proxy scorer, held-out evaluation.
- **P3 — memory/submission:** odometry, map, tokens, `agent.py`, validator, brief,
  demo, Q&A.

If solo, run the same order and cut: no PLR, one training seed, fewer ablations.

---

## 8. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Exit still not learned | High | High | RGB exit head + visible exits in generated maps + held-out Freedoom; coverage fallback |
| BC policy can’t open doors | High | High | Door drills + PPO + USE edge/reversal handling; teacher door fix |
| Odometry drift | Medium | High | Measure on 5 maps; scan matching if drift > 1 cell/200 steps; tokens |
| RGB overfits to generated textures | Medium | Medium | Randomise exit/door/wall appearance; held-out Freedoom |
| PPO sample inefficiency | Medium | High | BC prior, dense shaping, small drill episodes, 8–10 workers |
| Compute/time | High | High | A-first; phase gates; stop training at hour 20 |
| Torch/environment broken | Medium | High | P0 repair; CPU fallback; pin versions |
| Repo/LFS > 1 GB | Medium | High | `.gitignore` data/runs; one checkpoint; LFS from day one |
| Freedoom leakage | Medium | High | Split file; collector reads it; never train on MAP23–MAP32 |
| Rule ambiguity (map/frontier) | Medium | Disqualification | Written organiser check; no planner-like inference code |
| Determinism failure | Low | Medium | Argmax default; no dropout; seeded sampling only if needed |
| Maps > 2,048 sectors | Low | Medium | Filter Freedoom maps; keep generated maps small |
| Docker unavailable | Medium | Low | Local validator; run `--docker` only if available |
| NaN in entity masks | Medium | Medium | Null token; sanitise inputs; unit test all-zero masks |
| CUDA warm-up on first act | Low | Medium | 3 dummy forwards in `__init__`; CPU path tested |

---

## 9. Compliance checklist

- [x] Repo built from the official template; `env/` untouched.
- [x] Official template Dockerfile left as shipped; no new Dockerfile; no CI/CD.
- [ ] Final agent decisions come only from network parameters.
- [ ] Teacher/privileged code exists only under `training/`, never imported by `submission/`.
- [ ] No A*/Dijkstra/BFS or rule tree at inference; map writer is perception only.
- [ ] Weights in `submission/weights/` via Git LFS; total repo ≤ 1 GB.
- [ ] `submission/requirements.txt` empty (torch/numpy are in the base image).
- [ ] `training/data/` and `training/runs/` are gitignored; no `.npz` demo shards in git.
- [ ] `python validate.py` passes on the final commit.
- [ ] Repo public at submission; last commit tagged.
- [ ] 2-page PDF brief submitted.
- [ ] Live demo + Q&A rehearsed; attention-overlay video ready.
- [ ] No network, no engine access, no writes outside `/tmp` at inference.
- [ ] Deterministic argmax or seeded sampling documented.

---

## 10. Repo map

```
submission/           # what the scorer runs
  agent.py            # DoomAgent: init / reset / act (currently random baseline)
  model.py            # network definition (shared with training)
  weights/            # Git LFS checkpoint
training/
  maps/               # procedural generator, drills, Freedoom prep
  teacher/            # privileged expert (TRAINING ONLY)
  collect.py          # demonstration shards
  bc.py               # behaviour cloning
  ppo.py              # RL fine-tuning (to create)
  evaluate.py         # proxy leaderboard / replays
  obs.py              # observation flattening / RGB + labels
  data/               # generated datasets (gitignored)
  runs/               # checkpoints, tensorboard, curves (gitignored)
docs/
  ARCHITECTURE.md     # diagrams + module reference (to create)
  brief/              # 2-page PDF source and output (to create)
  CONTRACT/EVALUATION/RULES/SUBMISSION (official)
PLAN.md               # this document
progress.md           # live lab log
README.md             # public-facing story, visuals, results
validate.py           # official validator (untouched)
```

---

## 11. Immediate next actions (P0)

1. **Repair torch.** Reinstall torch 2.15.0 into `.venv` with `uv`; verify
   `import torch`, a GRU forward pass on CPU, and MPS if available. This blocks
   everything.
2. **Fix `.gitignore`.** Add `training/data/` and `training/runs/`; confirm
   `submission/weights/*.pt` is not ignored and is LFS-tracked.
3. **Freeze the A baseline.** Add RGB encoder, previous action + `USE-last`, exit
   visible/bearing, key-colour, and damage heads to `submission/model.py`; keep the
   six action heads.
4. **Add RGB/exit labels to `training/obs.py` and `training/collect.py`.** Store
   downsampled RGB (e.g. 60×80 grayscale or 3-channel uint8) and exit visibility/
   bearing/distance labels. Derive labels from `training/teacher/mapdata.py` exit
   lines and privileged pose.
5. **Fix generated maps.** Add visible exit switches/signs, coloured locked-door
   textures (`DOORBLU`/`DOORRED`/`DOORYEL`), and domain randomisation.
6. **Add the turn table and `turn_held` state** to the model/agent; reset at
   `reset()`. Add `prev_action` and `USE-last` inputs.
7. **Write `submission/agent.py`** with the inference contract: load weights,
   `torch.set_num_threads(1)`, `inference_mode`, argmax, try/except, NaN sanitisation,
   warm-up, buffer reset.
8. **Freeze the Freedoom split:** MAP01–MAP22 train, MAP23–MAP32 held out. Write it
   to a file and make collection/evaluation read it.

Only after these pass should the team start the long training run.

---

## Appendix A — Measured engine facts

These were measured on the host with ViZDoom 1.3.3 and `FRAME_SKIP=4`. Re-measure on
the final maps and hardware.

### A.1 Turn acceleration

| Situation | Turn deltas per decision |
|---|---|
| First turn from reset | `0.000°, 3.516°, 14.063°, 14.063°...` |
| After 1 warm-up step | `1.758°, 12.305°, 14.063°...` |
| After 2+ warm steps / re-hold after release | `7.031°, 12.305°, 14.063°...` |
| Continuous hold after ramp | `14.063°` per decision |
| Both turn buttons or neither | `0°` |

Use a calibrated lookup keyed by `turn_held` and the episode warm-up state. Do not use
a single `14.06°` constant.

### A.2 Other engine facts

| Quantity | Value |
|---|---|
| Walk speed | ~7.3 units/tic → ~31–32 units/decision after acceleration |
| First forward decision from rest | No displacement (engine warm-up tic) |
| Strafe speed | ~2.4 units/tic during acceleration; lower than forward |
| Pistol refire | ~14 tics (~3.5 decisions) |
| Door open time | ~9–12 decisions to clear 56 units of headroom |
| Episode caps | 2,100–4,200 tics = 525–1,050 decisions |
| Step rate, template map | ~5,700 steps/s without pixels on this host |
| Step rate, Freedoom MAP01 | ~1,000 steps/s without pixels on this host |
| Step rate, 200+ sector generated map | ~800–1,100 steps/s |
| Pixels on this host | ~2% overhead; model compute is the real cost |
| Exit observability | Not in rays or entities; RGB/linedef labels required |
| Key/lock colour | Not in structured observations until pickup |
| USE | Fires on button-down; reverses a moving door |
| Coverage | Scorer uses sector polygons, not 64-unit cells |

---

## Appendix B — Why the validator is not the score

`validate.py` checks size, interface, one dry-run episode, latency, and RAM. It does
**not** measure skill. Passing it with margin is necessary but not sufficient. The
proxy suite in §6 is the real optimisation target; the validator is a regression test
run after every promoted checkpoint.

---

## Appendix C — Rejected or deferred ideas

| Idea | Decision | Reason |
|---|---|---|
| From-scratch PPO, no BC/expert | Rejected | The expert and BC pipeline already exist and save samples; the review assumes a team starting from zero |
| Cut the procedural map generator | Rejected | Already built and validated 120/120; Freedoom alone cannot provide tier-controlled key/door depth |
| Cut scan matching outright | Deferred | Measure drift first; add a cheap translation match if drift exceeds ~1 cell per 200 steps |
| Switch to UDMF for the main generator | Rejected | Binary is required for things to spawn; UDMF loads but things do not |
| Hierarchical goal selection (C) | Rejected for 24h | Two-level credit assignment and non-stationarity are too risky for the time budget |
| RND / two value heads | Deferred | Not the main signal; episodic novelty + sector reward is enough for the first run |
| Three training seeds | Deferred | Too expensive; use one training seed and 3–5 evaluation seeds |
| 300–500M PPO steps | Not promised | Hardware-dependent; target 50–150M and measure throughput |
| “Never train on Freedoom” | Rejected | Use a held-out split: MAP01–MAP22 train, MAP23–MAP32 held out |
| 68 Freedoom maps | Deferred | Wrapper IWAD is `freedoom2.wad`; verify Freedoom 1 textures before relying on it |
| Docker validate as an hour-1 gate | Deferred | Docker daemon unavailable and official image private; local validator is the gate |
| Fully learned memory claim | Rejected | The writer is deterministic; describe it honestly as perception + learned readout |
