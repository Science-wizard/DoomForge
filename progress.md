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
