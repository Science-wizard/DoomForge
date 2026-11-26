# DoomForge — Progress Log

**Track:** Yugaantar DevForge · Track 1 “Can It Play DOOM?”
**Repo:** `Science-wizard/DoomForge` (private during development; must be public at submission)
**Plan of record:** [`PLAN.md`](PLAN.md) — **Revision 2, 24-hour cut**
**Contract:** `1.0.0-draft`, official template `Reinforce-SST/devforge-doom-template`

This is the live engineering log: what was tried, the numbers, and the decision each
result triggered. Revision 2 supersedes the old multi-phase plan; the old log is kept
below as evidence, but the status table here is authoritative.

---

## Revision 2 re-baseline (2026-11-26)

### What the review changed

| Area | Revision 1 | Revision 2 | Action taken |
|---|---|---|---|
| Exit perception | structured rays/entities; pixels optional | **RGB + privileged exit head is P0** | model, collector, generator, agent rewritten |
| Key/lock colour | flags only | **RGB key-colour head / try-all-doors** | key-colour aux head + coloured door textures |
| Memory | learned 48×48 ConvGRU | **deterministic coarse map + event tokens (B+ lite)** | deferred until A ships |
| Sequencing | BC → PPO → memory | **ship A first; promote B+ only if it wins** | A is the only training target now |
| Turn model | one 14.06° constant | **measured table + `turn_held` + `prev_action`** | added inputs and state |
| Evaluation | proxy suite | **train/val/OOD/Freedoom splits + proxy scorer** | splits file + sector coverage + proxy score |
| Schedule | multi-phase | **24-hour schedule, hard gates** | see PLAN.md §7 |

### P0 checklist (Revision 2 §11)

| # | Item | Status | Evidence / notes |
|---|---|---|---|
| 1 | Repair torch, verify CPU/MPS GRU | **in progress** | partial wheel repaired by ranged HTTP; 101 MB `libtorch_cpu.dylib` still downloading at ~58 KB/s |
| 2 | Fix `.gitignore`; LFS for weights | **done** | `training/data/`, `training/runs/`, `maps/generated/` ignored; `.pt` already LFS-tracked |
| 3 | Freeze A baseline (RGB, prev action, exit/key/damage heads) | **done** | `submission/model.py`: 128 ray + 128 entity + 64 vitals + 128 RGB + 32 prev → GRU(384); 35-dim aux head |
| 4 | RGB/exit labels in obs + collector | **done** | `training/obs.py`, `training/collect.py`; 30×40×3 RGB, exit visible/bearing/distance, key colours, damage-next-8 |
| 5 | Visible exits + coloured locks + randomisation | **done** | `EXITSIGN`/`SW1EXIT`/`SW1BRN1` wall behind every exit; `DOORBLU`/`DOORRED`/`DOORYEL`; randomised wall/floor/light/door/exit textures |
| 6 | Turn table + `turn_held` + `prev_action` | **done (input + state)** | `prev_action` 12-vector; turn-held counter; odometry integration itself is B+ |
| 7 | `submission/agent.py` inference contract | **done** | load, `set_num_threads(1)`, warm-up, `inference_mode`, argmax, sanitise, try/except fallback |
| 8 | Freeze Freedoom split | **done** | `training/splits.py` + `splits.json`: MAP01–22 train, MAP23–32 held out |
| 9 | Proxy scorer with components | **done** | `training/evaluate.py`: exit, keys, locks, doors, sectors, kills, damage, death; configurable weights |

### Additional Revision 2 requirements landed

- **Colour-aware locks:** locked doors now use the blue/red/yellow door textures; the
  key-colour auxiliary head learns the correspondence from RGB.
- **Exit is visible:** every generated walk-over exit has a bright sign wall behind it;
  Freedoom exits are already switches/signs. The exit head is trained from privileged
  exit-line geometry.
- **Never train on held-out maps:** the collector reads `training/splits.py`; the
  Freedoom source pool is MAP01–22 only.
- **No `.npz` in git:** all demonstration shards live under `training/data/`, ignored.
- **No planner at inference:** all A*/BFS/door-state code is under `training/teacher/`;
  `submission/agent.py` only flattens observations and runs the network.

### Re-evaluation of old results under Revision 2

| Old result | Still valid? | Revision 2 interpretation |
|---|---|---|
| Validator PASSED with 2 host warnings | ✅ | Still the gate; rerun after every promoted checkpoint |
| 6,100 steps/s small map | ✅ | Confirms RGB is affordable (~2% overhead); latency risk is the model, not the engine |
| 120/120 generated maps valid | ✅ | Generator survives the visible-exit/coloured-door changes; revalidated 4/4 tiers |
| Expert 23/24 exits door-free | ✅ | Still the navigation data source |
| Expert fails multi-room doors | ✅ | Door drills + PPO remain the plan; A-first sequencing means doors are learned, not taught perfectly |
| UDMF loads but things do not spawn | ✅ | Binary generator remains mandatory |
| Torch install “broken” | ⚠️ | Not a code issue: one 101 MB compressed dylib was truncated by a killed download; being repaired |
| Old model “1.2 M, no RGB” | ❌ superseded | New A model is ~2.0 M with RGB, prev action and 35 aux outputs |

### Blockers / risks now

1. **Torch download** is the only hard blocker to training. Fallback: pure-NumPy
   reactive MLP baseline (not started; only if the download fails).
2. **Exit positives are rare** in collected episodes (~0.5–2% of steps on door-free
   maps). Mitigation: `pos_weight=12` on the exit-visible BCE and oversampling
   near-exit frames; monitor the exit head's precision/recall, not just loss.
3. **Teacher door skill** still fails on multi-room maps (6 drill episodes → 3 exits).
   A-first plan says: collect reliable navigation + drills, then PPO learns doors from
   the reward and the USE edge.
4. **Compute** is a 10-core M1/16 GB host with no CUDA; PPO throughput must be
   benchmarked before promising step counts.

---

## Phase 1 close (hand-off to Phase 2)

**Shipped checkpoint:** `training/runs/bc_arena/ckpt.pt` (copied to
`submission/weights/doomforge.pt`). Map-aware multimodal BC, 1.807 M parameters,
low-LR arena-specialized fine-tune.

| Check | Result |
|---|---|
| Official validator (2 runs) | **PASSED**, init 0.4–0.5 s, p50 0.8–1.0 ms, p99 0.9–2.0 ms, max 4 ms |
| Broad bench tier 1 (8 maps) | **5/8 exits**, 100% coverage, proxy +5.42 |
| Broad bench tier 2 | 0/8, 6.1% coverage, proxy −0.18 |
| Broad bench tier 3 | 0/8, 4.2% coverage, proxy −0.38 |
| Broad bench tier 4 | 0/8, 2.0% coverage, proxy −0.73 |
| Freedoom 23–32 (10 maps) | 0/10 exits, 3.6% coverage, proxy +1.00 |
| Door research branch (`bc_use`) | **6/8** door drills, but 1/8 tier 1 (catastrophic forgetting) |
| Repo | private `Science-wizard/DoomForge`, LFS weight, tracked size 7.7 MB |

**What was learned:** exits and key colours are invisible to the structured contract and
must be learned from RGB; a global-path teacher is not imitable from a single
observation (turn accuracy stuck at 43–49%); a deterministic pose+map is necessary and
cuts drift 4×; door USE is learnable in two epochs from drills but overwrites arena
behaviour without replay; the validator is not the score.

**Failed iterations:** the focused arena-anchored door merge (`bc_combined`, drills +
room-doors, KL anchor to `bc_arena`, 1 epoch) measured **tier 1 1/8, drills 0/8**. A
success-only mixed replay (`bc_success`, 652 successful episodes, LR 5e-5) measured
**tier 1 4/8, proxy +4.95; tier 2 −0.54; drills 0/8** — slightly worse than the arena
specialist. Neither is shipped. Doors still do not generalise to rooms.

**Phase 2 entry point:** replay-balanced BC/PPO (mix tiers in every batch), event
tokens + scan matching, a dedicated door-approach objective, Freedoom MAP23–32
held-out evaluation, then freeze at the deadline.

---

## Phase 2 — exploration RL and mixture of experts (in progress)

### PPO

A 25-update PPO run (25.6k env steps, 8 envs, dense new-cell/door/exit reward) from the
arena checkpoint:

| Suite | `bc_arena` | PPO-25 |
|---|---:|---:|
| tier 1 exits | 5/8 | **6/8** |
| tier 1 proxy | +5.42 | **+8.20** |
| tier 1 kills | 4 | **19** |
| tier 2 proxy | −0.18 | **+0.48** |
| tier 3 proxy | −0.38 | **−0.03** |
| tier 4 proxy | −0.73 | **−0.32** |
| door drills | 0/8 | 0/8 |

Tier-weighted proxy **+8.09** (from +3.17). PPO fixed the t1 regression and improved
every tier proxy, but still cannot use doors. The 20-update resume is running.

### Mixture of experts

Two frozen specialists:
- **arena/PPO** — tier-1 exits, 19 kills, t2 proxy +0.48;
- **door** — `bc_use`, 6/8 door drills, 1 key.

Two frozen specialists: PPO-25 (tier-1 6/8) and `bc_use` (drills 6/8). A 2k gate over
30 observation-only features routes per step. Results:

| Gate | t1 | drills | t2 proxy | t3/t4 | verdict |
|---|---:|---:|---:|---:|---|
| distilled | 2/8 soft, 6/8 hard | 6/8 soft, 0/8 hard | +0.23 / +0.48 | — | fails |
| supervised near-door, hard 0.1 | 5/8 | **6/8** | +0.32 | −0.22 / −0.93 | doors work, tiers regress |
| supervised near-door, hard 0.5 | 6/8 | 0/8 | +0.48 | −0.03 / −0.32 | identical to PPO |

The gate can learn “door in view → door expert”, but the door expert is worse on arena
and rooms; the mixture is not shipped. PPO-25 remains the best hidden-tier agent.

### Teacher work

A local ray-based door reflex was added to the teacher (press once, 30-step cooldown,
locked doors only with a key). It still cannot align with arbitrary room doors; the
frontier teacher explores 160–305 cells on door-free maps but never exits within 1050
steps. Room-door drills are 200/200 with the A* teacher; the door expert trained on them
scores 4/8 drills and 0/8 room doors, so straight drills do not transfer.

---

## Phase 1 result: Baseline A is not Markovian enough — pivot to B+ minimal

### What happened

Baseline A (rays + entities + vitals + RGB + previous action, no map) was trained with
behaviour cloning on 100,921 transitions (400 door-free navigation episodes, 150 arena
episodes, 2 Freedoom episodes).

| Metric | Result |
|---|---|
| BC loss (epoch 0 → 3) | 6.21 → 1.89 |
| Teacher-forced move accuracy | 85.6% |
| Teacher-forced strafe accuracy | 76.0% |
| Teacher-forced **turn** accuracy | **43.2%** (chance 33%) |
| Teacher-forced attack accuracy | 92.4% |
| Exact six-head match | 20.8% |
| Held-out exit rate | **0/4** |
| Typical behaviour | forward + turn-left; circles in the start room |

### Diagnosis

The teacher's turn action is computed from a **global A* waypoint**. From a single
observation the agent cannot know where that waypoint is; the mapping from observation
to turn is not a function. Move/strafe/attack are learnable because they are local;
turn is not. More BC epochs did not help (CE plateaued at ~1.9, turn accuracy stayed at
chance). This is the partial-observability failure the review predicted, but it bites
Baseline A even on door-free maps, not only on keys/doors.

### Decision (Revision 2 §5.7 promotion rule)

Promote a **minimal B+** now instead of waiting for A to clear a gate it cannot clear:

- deterministic `MapBuilder` (`submission/mapmem.py`): heading from the measured turn
  table, translation from ego velocity, rays and entity sightings projected into a
  64×64 allocentric grid at 96 units/cell; 5 channels (visited, wall, door, locked,
  item/key);
- a 32×32 crop + a 6-value pose vector fed to a small CNN/MLP;
- no planning, no privileged state, no engine access. The map is perception.

The map makes the teacher's global waypoint **observable** (relative to the visited
layout), so turn becomes learnable. It also directly serves the key→locked-door
return requirement.

### Updated model

| Item | Value |
|---|---|
| Parameters | 1.807 M |
| Inputs | rays, entities, vitals, RGB, prev action, 5×32×32 map crop, 6-value pose |
| Validator latency | p50 1.0 ms, p99 1.9 ms (limit 5 s) |
| Init | 0.4–0.6 s (limit 60 s) |

### Door demonstrations (the missing skill)

The first map-aware BC model was trained on **zero closed doors** (navigation maps
were door-free by design; arenas have no doors). Its USE head therefore learned
"never use", and every tier-2 map failed at the first door. Fixed by collecting
isolated drills:

| Drill | Episodes | Exits | Wall time |
|---|---:|---:|---:|
| 1 door, no key | 120 | 120 | 24 s |
| 1 door + key + locked door | 120 | 120 | 24 s |
| 2 doors, no key | 120 | 120 | 28 s |

These are cheap, reliable USE/key/lock demonstrations. They are mixed into the BC
dataset with a 3× repeat so the USE and key-colour heads see enough positives.

### First held-out evaluation (map-aware BC, epoch 0, no door data)

| Split | Exit | Kills | Sector coverage | Proxy |
|---|---:|---:|---:|---:|
| Generated tier 1 (arena) | **2/2** | 4 | 100% | **11.4** |
| Generated tier 2 | 0/2 | 1 | 5.9% | −1.9 |
| Generated tier 3 | 0/2 | 0 | 3.6% | 0.0 |
| Generated tier 4 | 0/2 | 0 | 2.1% | −2.2 |

Tier 1 exits in 87–112 steps; tiers 2–4 stall at the first door/room. This is the
concrete evidence for adding door demonstrations and PPO recovery.

### Door fine-tune and catastrophic forgetting

Fine-tuning the base checkpoint on the 360 drills (weighted USE loss) learned the door
skill in two epochs:

| Eval | Exit | Steps | Notes |
|---|---:|---:|---|
| 1 door | 2/2 | 65 | perfect |
| 1 door + key + locked door | 2/2 | 65 | key pickup + lock |
| 2 doors | 0/2 | 1050 | gets through the first, stalls at the second |

But the drill-only run **forgot the arena skill**: tier 1 dropped from 2/2 to 0/2 and
the tier-1 proxy went from +11.4 to −1.2. This is textbook catastrophic forgetting; the
plan's “keep 20–30% of earlier stages in every batch” rule was violated.

**Fix in progress:** a balanced fine-tune from the tier-1 checkpoint on
`demos_all + demos_drills` (drill repeat 1, weighted USE loss, 2 epochs). The final
checkpoint is selected by held-out proxy score, not by training loss.

### Broad bench (no cherry-picking)

The two-map eval was optimistic. A fixed bench (8 maps per generated tier, seeds
30,000+, one engine seed; Freedoom MAP23–32 held out) gives the real picture:

| Suite | Shipped tier-1 model | Door branch | Head-only door fine-tune |
|---|---:|---:|---:|
| t1 arena | **2/8**, +0.35, 10 kills | 1/8, +0.89, 19 kills | 1/8, −1.53 |
| t2 rooms | 0/8, −0.99, 6.1% cov | 0/8, −0.26 | 0/8, −0.22 |
| t3 key/lock | 0/8, −0.89 | 0/8, −0.70 | 0/8, −0.87 |
| t4 gauntlet | 0/8, −1.61 | 0/8, −1.41 | 0/8, −2.17 |
| door drills | 0/8 | **6/8**, +9.12 | 0/8 |
| Freedoom 23–32 | 0/10, −0.77, 1.1% cov | not run | not run |

**Findings:** the tier-1 exit rate is 25%, not 100%; doors are learnable (6/8 drills)
but do not transfer to room doors; freezing the trunk and training only the actor cannot
learn door approach (0/8 drills). The bench is the selection metric from here on.

### Arena specialization (shipped)

A low-LR fine-tune on 300 extra arena episodes (53% teacher exits) produced the Phase-1
winner:

| Suite | Before | `bc_arena` |
|---|---:|---:|
| tier 1 exits | 2/8 | **5/8** |
| tier 1 proxy | +0.35 | **+5.42** |
| tier 2 proxy | −0.99 | **−0.18** |
| tier 3 proxy | −0.89 | **−0.38** |
| tier 4 proxy | −1.61 | **−0.73** |
| Freedoom 23–32 proxy | −0.77 | **+1.00** |
| Freedoom coverage | 1.1% | **3.6%** |

Tier-weighted proxy over t1–t4: **+3.17** (from −5.40). The checkpoint is copied to
`submission/weights/doomforge.pt` and is the shipped Phase-1 agent.

### Room-door demonstrations and KL anchoring

To bridge “straight drill” to “door on a room wall”, a new generator places one or two
doors on a room wall with a random start position and heading. The teacher solves
**200/200** in 34 s, so the demonstrations are reliable.

The next iteration trains the full network on the mixed pool with:
- a frozen KL anchor to the tier-1 checkpoint on **non-door frames**, so arena/nav
  behaviour is preserved;
- door-visible frames free to adapt to drills and room-door data;
- LR 1e-4, 2 epochs.

### Next experiment

1. BC with maps on the same 100,921 transitions. Measure teacher-forced turn accuracy
   and held-out exit rate. Success = turn accuracy > 70% and exit rate > 50% on
   door-free validation maps.
2. If the exit rate is still low, run DAgger (`training/dagger.py`) with the map-aware
   student and A* teacher labels; this is the planned covariate-shift fix.
3. If the map-aware BC clears the gate, run the full evaluation (`gen_val`,
   `freedoom_eval`) and promote the checkpoint; then PPO for doors/combat.

### Honest risk

The map is coarse (96 units/cell) and the pose integration uses a constant 14.06° turn
instead of the full table; drift may limit long episodes. If so, the next fix is the
measured table + translation-only scan matching (already specified in PLAN.md §4.2).

---

## Historical Phase 0/1 log (pre-Revision 2)

### Environment

| Item | Value |
|---|---|
| Host | macOS 27.2.1, Apple Silicon arm64, 10 cores, 16 GB RAM |
| Python | 3.13.5 (uv-managed) |
| ViZDoom | 1.3.3 (arm64 wheel) |
| Gymnasium | 1.4.0 |
| NumPy | 2.5.5 |
| Docker | installed, daemon not running; official image is private |

### Template integrity

- `env/` untouched (official wrapper + contract).
- `validate.py` untouched.
- `submission/agent.py` keeps the required `DoomAgent` signature.
- Work lives in `training/`, `submission/model.py`, `submission/weights/`, `docs/`.
- No Dockerfile added; no CI/CD.

### Validator baseline

```
✓ Repository size (tracked files): 216.9 KB of 1.0 GB
✓ requirements.txt: no extra packages
✓ DoomAgent interface: __init__, reset, act
! CPU limit: could not be pinned on macOS (Linux-only check)
✓ Initialization: 0.1s (limit 60s) on device 'cpu'
✓ Episode on 'two_rooms.wad:MAP01': 525 steps, tick cap, agent time 0.1s of 21.0s
✓ Step latency: first 0 ms, p50 0.2 ms, p99 0.2 ms, max 0 ms
! Peak RAM: could not be measured (needs Linux /proc)
PASSED with 2 warnings
```

The two warnings are host limitations, not submission failures. `--docker` cannot run
here (daemon off, image private); compensated by explicit latency/RAM budgets.

### Engine facts (measured)

| Fact | Measurement |
|---|---|
| Instance creation | 0.10–0.15 s per map |
| Step rate, template map | ~6,100 steps/s (`render_buffers=False`) |
| Step rate, Freedoom MAP01 | ~1,090 steps/s |
| Step rate, 200-sector generated map | ~800–1,000 steps/s |
| Turn rate | 14.06°/decision steady state; first decision 3.52°; full table in PLAN.md Appendix A |
| Walk speed | ~31–32 units/decision after acceleration; ~7.3 units/tic |
| First forward decision | no displacement (warm-up tic) |
| Pistol refire | ~1 shot per 14 tics |
| USE | button-down edge; reverses a moving door |
| Pixels | ~2% step overhead on this host |

### Map sourcing

- Freedoom 2 ships 32 Doom-format maps.
- ZDoom builds BSP nodes at load when `SEGS/SSECTORS/NODES/BLOCKMAP/REJECT` are absent;
  dropping only `NODES` crashes. Verified.
- UDMF loads node-less but **things do not spawn** unless written as binary `THINGS`.
  Generator uses Doom binary.
- 120/120 generated maps loaded and matched `mapinfo` door metadata.

### Expert results (pre-Revision 2)

| Suite | Maps | Exit | Deaths |
|---|---:|---:|---:|
| Generated, no doors, no monsters, tier 2 | 8 | 8 | 0 |
| Generated, no doors, no monsters, tier 3 | 8 | 8 | 0 |
| Generated, no doors, no monsters, tier 4 | 8 | 7 | 0 |
| Generated, doors, no monsters, tier 2 | 8 | 2 | 0 |
| Generated, doors+monsters, tier 1 | 8 | 6 | 2 |
| Generated, doors+monsters, tier 2 | 8 | 1 | 4 |
| Handcrafted 2-room + door | 1 | 1 | 0 |
| Drills (6 episodes, key/no-key) | 6 | 3 | 0 |

### Phase log

| Time | Event |
|---|---|
| 13:42 | Workspace created, template cloned |
| 13:50 | `uv sync`, baseline validator PASSED |
| 14:00 | Engine physics measured, Freedoom maps extracted |
| 14:10 | Node-less binary/UDMF loading verified |
| 14:20 | Procedural generator written; 120/120 valid |
| 14:35 | Expert navigation: 23/24 door-free |
| 14:45 | Binary THINGS fix for monsters; door failure isolated |
| 14:55 | Revision 1 plan + architecture written |
| 15:10 | Revision 2 received; re-baseline started |
| 15:25 | RGB/exit/prev-action model + collector + agent + proxy scorer rewritten |
| — | Torch repair; then collect ≥60k transitions and train A |

---

## Next actions

1. **Torch**: finish the ranged repair, verify `import torch`, CPU forward, MPS GRU.
2. **Smoke test A**: build a random-weight checkpoint, run `validate.py`, check
   latency/RAM; then train BC on 20–30k transitions as the first real checkpoint.
3. **Collect** 300 nav demos (generated door-free + Freedoom MAP01–22 nav), 300 door
   drills, 200 arena demos, 100 key drills.
4. **BC** 4–8 epochs; run `training/evaluate.py --suite gen_val,freedoom_eval`.
5. **PPO** from the best BC checkpoint; 8–10 workers; curriculum per PLAN.md §5.5.
6. **Promote A** by the hour-12 gate with LFS weights and a validator run.
7. **B+ lite** only if it beats A on held-out proxy score.
8. Brief PDF, visual README, private repo push, public flip at submission.

---

## Phase 3 — suite + Atlas-G (2026-11-26 evening)

### Measured compute decision

`mlx` was installed (0.32.5, Metal available) and benchmarked on the actual
workloads rather than assumed:

| workload | MLX GPU | PyTorch CPU (10 threads) | PyTorch MPS |
|---|---:|---:|---:|
| 2 M-param MLP, 256×512 fwd+bwd | **4.2 ms** | 11.2 ms | 11.3 ms |
| DoomNet recurrent batch (16×128) | — | 3.50 s | **0.22 s** |

Decision: **PyTorch MPS for training, all 10 CPU threads for environments and
suite workers**. MLX wins only for stateless MLPs; the recurrent policy is 16×
faster on MPS, so porting the training stack to MLX would slow it down.

### Suite (suite-1)

- 14 categories C0–C13, 4 splits, leakage-checked by WAD hash.
- Category-specific generator seeds, acceptance filters (C6 exactly 2 keys/2
  locks, C7 8–13 rooms, C8 detour geometry), facing starts for C0/C3/C1.
- Disjoint Freedoom maps: train 1–16, val 17–22, test 23–32.
- Parallel episode runner, tier-weighted proxy + stratified bootstrap CI, paired
  promotion gate, probes (door angle, exit sign, colour, corruption,
  determinism, memory wipe).
- The suite drives training: PPO `--suite` samples the train split with the
  30/50/20 mastered/current/hardest mix; DAgger `--source suite` labels the
  failure states it exposes.

First val baselines (136 maps, 1 seed):

| checkpoint | weighted proxy | C0 | C1 | C2 | doors C3/C4 | C5/C6 | C12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO-25 (shipped) | +0.466 | 0% | 10% | 80% | 0/0% | 0/0% | 0% |
| BC v2 (goal head + frontier) | +0.793 (fast, 70 maps) | 100% | 40% | 20% | 0/0% | 0/0% | 0% |

PPO-25's failure mode is now measurable: it circles the spawn room (path 40k
units, coverage 3–4%) on generated maps. BC v2 fixes C0 and improves C1, but is
still stuck on door/key categories and loses on the old tier-1 bench (2/12 vs
PPO-25's 6/8), so it is not shippable. DAgger on suite maps is the next step.

### Dead end: unbalanced DAgger poisons the policy

191 DAgger episodes on suite maps were collected with `beta=0` (student rollout,
teacher label). The student got stuck early on most maps, so the teacher's label
for the rest of the episode was "turn", producing a 73% turn-label rate. BC v3
(resumed from BC v2) collapsed into spinning: 0% exit on every category, spin
fraction 99.8% on C0/C1/C5/C6/C8. Removed from the training set; DAgger needs
teacher mixing and a cap on repeated stuck frames, or PPO's reward signal.

### Teacher fixes that mattered

1. `NavGrid` treated *all* lines touching a door sector as passable, including
   the door's solid side tracks. That carved false paths through walls; A* led
   the teacher into corridors that do not exist. Only door *faces* between two
   real sectors are passable now.
2. The teacher required a path *to the door centre* before fetching a key; the
   centre is often on the far side of the locked door, so it never went for the
   key. Removed that precondition; key reachability is checked directly.
3. Frontier fallback rescanned 40 A* goals every replan and hung on tier-4
   maps. It reuses one fallback goal for 240 steps and scans at most 8.
4. The room generator's locked doors are usually optional: only 5/30 tier-4
   maps required a key, 0/30 required two. C5/C6/C8 were rebuilt as drill
   layouts with guaranteed cut-edge locks (teacher 3/3, 5/5, 3/3).

### Atlas-G v1 + hybrid router (measured)

The goal head + frontier channel + key/backtrack demos produced a model that is
strong where PPO-25 is weak and vice versa:

| val fast (70 maps) | PPO-25 | Atlas-G BC v5 | Hybrid (no reflex) |
|---|---:|---:|---:|
| weighted proxy | +1.18 | +3.07 | **+3.17** |
| C0 sanity | 0% | 100% | 100% |
| C1 duel | 10% | 0% | 20% |
| C2 arena | 80% | 60% | 60% |
| C3 door | 0% | 0% | 0% |
| C4 room door | 0% | 20% | 20% |
| C5 key/lock | 0% | 100% | 100% |
| C6 two-key | 0% | 0% | 0% |
| C8 backtrack | 0% | 20% | 20% |

The router selects PPO-25 when an enemy is visible (16-decision hysteresis) and
Atlas-G otherwise; both nets run every step so their memory stays current. The
official validator passes at p50 1.7 ms / p99 2.0 ms.

A second failure mode was "freeze": the policy stops one tile short of a door
(C6: at x=944 facing the red door, all-zero actions forever) or presses forward
into a wall (C3). An observation-only reflex — 8 forward steps with vf≈0 ⇒ turn
4 steps; 8 pure no-ops ⇒ forward+USE — fixes C6 5/5, C3 3/5, C8 3/5 in targeted
tests. Combat is excluded whenever an enemy is visible.

PPO v2 (suite reward, key potential, 40 updates) raised mean return 1.4→8.5 but
*forgot* C5 (0/5) and did not beat BC v5 on C6/C8, so it is not the nav expert.

### Final shipped checkpoint (measured)

- **Val fast** (70 episodes): weighted proxy **+5.696** (95% CI +4.862…+6.551),
  paired delta vs PPO-25 **+4.521** (CI +2.589…+6.728), gate **PROMOTE**.
  C0 100, C1 20, C2 80, C3 0, C4 20, C5 100, C6 100, C8 40; C7/C9–C13 0.
- **Test fast** (untouched, logged): **+5.790** (CI +5.115…+6.509), paired
  +4.907 (CI +3.002…+7.073), gate **PROMOTE**.
- Pre-existing checkpoints on the same val split: bc_arena +0.924, bc_use
  +1.168, PPO-25 +1.175. Atlas-G alone +3.066.
- Official validator: PASS; p50 1.7 ms, p99 2.0 ms, max 3 ms; init 0.5 s.

The last change before shipping was per-expert previous-action state in the
router (each expert sees its own last action, not the other's). That restored
C2 from 0% to 80% on val; a map-pose fix (`prev_exec` for the deterministic map)
then traded C3 back to 0% for correct pose integration. We kept the correct
version: C3 is one category, the map state is used by every navigation decision.

### Next phase: C3 closed by a gated door specialist

A third expert (BC from BC v5 on 150 C3 + 150 C4 suite teacher demos, anchor
0.5) solves C3 5/5 but forgets C5/C6 (0/5), so it is only used behind an
observation-only gate: no enemy, no locked door, no key seen/held, a door known,
and median side ray <0.23 at first activation. The decision latches. C4 and the
validator two-room map are wider than 0.23 and stay on the generalist.

Final fast suite (70 episodes):

| split | proxy | 95% CI | C3 | C4 | C5 | C6 | C0 | C2 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| val | **+6.196** | +5.46…+6.98 | 100% | 0% | 100% | 100% | 100% | 80% |
| test | **+6.397** | +5.72…+7.08 | 100% | 60% | 100% | 100% | 100% | 40% |

Paired delta vs PPO-25: **+5.03 val** (CI +3.08…+7.25), **+5.51 test**
(CI +3.59…+7.69), gate PROMOTE. Validator dry run exits in 195 steps again.

Dead ends this phase: ungated BC v6/v7 (C0/C5 collapsed to 0–1/5), visible-door
controller (C5 5/5 → 0/5), 10/16-step blind reflex (C3 stayed 0), logit blending
(C2 5/5 → 3/5, C5 3/3 → 0/3), pure frontier controller on C7 (0/5), and teacher
demos on Freedoom/ambush/resource (0/3, no usable labels).

Remaining: C4 room doors (0% val / 60% test), C7 exploration, C9–C13 combat/
resource/transfer/OOD. The teacher fails these 0/3, so the next step is a
stronger combat/exploration teacher or PPO on the existing arena data, not more
of the same labels.

### Remaining-gap push (post-Phase 4)

- **C4**: collected 400 more room-door demos (392 exits) and trained a 550-episode
  specialist (BC v9). Val C4 1/5 — no better than the generalist. The room-door
  skill is not the bottleneck; finding the door from a random pose is.
- **C7**: fixed the teacher's door reflex to inspect the center rays and only
  fire once a path exists. Teacher C7 success rose from 2/6 to 6/10. Collected
  300 demos (165 exits) and trained BC v10; val C7 0/5. A KL-anchored PPO
  fine-tune on C7 (40 updates, β=0.1) reached training return 8.1 but 0/5
  argmax exits. A pure full-grid frontier planner (BFS to nearest frontier,
  door USE) also 0/5. C7 needs a stronger exploration policy, not more labels.
- **C9/C10/C12**: the teacher is still 0/3 after the door fix; no usable demos.
- **Reflex experiments**: a ray-distance wall detector fixed C7's stuck state
  but broke C5 (5/5 → 2/5); reverted. 48-step combat hold did not improve C9.
- **Ship state unchanged**: the Phase-4 hybrid (+6.20 val / +6.40 test) remains
  the best candidate; experimental checkpoints are not promoted.

The teacher's remaining C7 failure is the door task: at (1376,224) it plans a
127-cell path but the door task stays at stage 1 and never moves. Fixing that
state machine is the next concrete step for exploration labels.
