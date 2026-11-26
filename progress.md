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

**Shipped checkpoint:** `training/runs/bc_map2/ckpt_e0.pt` (copied to
`submission/weights/doomforge.pt`). Map-aware multimodal BC, 1.807 M parameters.

| Check | Result |
|---|---|
| Official validator (2 runs) | **PASSED**, init 0.4 s, p50 1.0–1.2 ms, p99 1.7–1.9 ms, max 3 ms |
| Held-out tier 1 (generated) | **2/2 exits**, 4 kills in one, 100% sector coverage, proxy +11.4 |
| Held-out tier 2 | 0/2, 5.9% coverage, proxy −1.9 |
| Held-out tier 3 | 0/2, 3.6% coverage, proxy 0.0 |
| Held-out tier 4 | 0/2, 2.1% coverage, proxy −2.2 |
| Door research branch (`bc_use`) | **4/4** on 1-door and key drills, but **0/2 tier 1** (catastrophic forgetting) |
| Repo | private `Science-wizard/DoomForge`, LFS weight, tracked size 7.7 MB |

**What was learned:** exits and key colours are invisible to the structured contract and
must be learned from RGB; a global-path teacher is not imitable from a single
observation (turn accuracy stuck at 43–49%); a deterministic pose+map is necessary and
cuts drift 4×; door USE is learnable in two epochs from drills but overwrites arena
behaviour without replay; the validator is not the score.

**Phase 2 entry point:** replay-balanced BC/PPO (mix tiers in every batch), event
tokens + scan matching, Freedoom MAP23–32 held-out evaluation, then freeze at the
deadline.

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
| Validator latency | p50 0.8 ms, p99 1.0 ms (limit 5 s) |
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
