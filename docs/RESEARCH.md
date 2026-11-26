# Research Notes — Track 1 “Can It Play DOOM?”

This document records the research that shaped Revision 2 of the plan: what the
environment actually exposes, which algorithms fit the 1-CPU/40 ms budget, and which
claims are measured versus assumed. It is deliberately separate from the lab log
(`progress.md`) and the plan (`PLAN.md`).

---

## 1. Problem framing

The task is a **partially observable, long-horizon, sparse-reward control problem**
with a hard inference budget:

- 64 raycasts + up to 32 entities + 17 vitals + a 120×160 RGB/depth frame.
- 9 binary buttons, held 4 tics per decision, 525–1,050 decisions per map.
- 2 GB VRAM, 8 GB RAM, **1 CPU thread**, 40 ms average per decision.
- Hidden maps, four tiers, non-binary scoring where exit ≫ progression > exploration >
  combat.

That combination rules out the usual “big pixel CNN + huge PPO batch” recipe and
favours a small multimodal encoder with a recurrent core and explicit memory.

---

## 2. Environment findings (measured on this host)

| Finding | Evidence | Consequence |
|---|---|---|
| Raycasts are a 360° metric sensor | `sensors.py`: 64 rays over 2π, 1024-unit range, 6 hit types; rays stop at walls, doors, enemies and items | Rays are the primary spatial input; pixels are for colour semantics |
| Exit is invisible to structured data | `RAY_HIT_TYPES` has no exit; `ENTITY_TYPES` has no exit; `mapinfo.py` parses doors only | Exit must be learned from RGB or depth, with privileged linedef labels |
| Key/lock colour is hidden | `classify()` maps all keys to `key`; `hit_type=3` means any lock | Colour must come from RGB, or the agent must try remembered doors |
| Turn rate is stateful | Measured 0°, 3.516°, 14.063°, 14.063° on the first hold; 1.758°, 12.305° after a warm-up step | Track `turn_held` and `prev_action`; don’t use one constant |
| USE is an edge | `CONTRACT.md`: button-down only; reverses a moving door | Penalise useless USE; never spam |
| Walk speed | ~7.3 units/tic, ~31 units/decision after acceleration | A 1-tile door (64 units) is ~2 decisions wide |
| Pistol refire | ~14 tics (~3.5 decisions) | Firing is not every-frame; the policy must position |
| Step rate | 6,100/s template; 800–1,100/s 200+ sector maps; RGB adds ~2% | CPU training is feasible; RGB is cheap |
| ZDoom builds BSP nodes | Node-less binary maps load; dropping only NODES crashes | Unlimited procedural maps, no external node builder |
| UDMF things do not spawn | UDMF loads geometry but `THINGS` must be binary | Use Doom binary for the main pool |
| Freedoom 2 ships 32 maps | Extracted and validated | Real-geometry diversity with a held-out split |
| Coverage is sector-based | `sector_polygons()` docstring | Reward unique sectors, not only cells |

---

## 3. Map generation research

### 3.1 WAD formats

- **Doom binary:** `THINGS`, `LINEDEFS`, `SIDEDEFS`, `VERTEXES`, `SECTORS`, plus
  derived `SEGS/SSECTORS/NODES/BLOCKMAP/REJECT`. ZDoom rebuilds the derived set if it
  is absent — as long as *all* derived lumps are absent. Keeping stale `SEGS` but
  dropping `NODES` crashes.
- **UDMF:** `TEXTMAP`/`ENDMAP`; loads node-less, but thing spawning requires a binary
  `THINGS` lump in practice. We therefore emit binary.
- **Node builders** (ZDBSP, BSP) are unnecessary for this engine; avoiding them removes
  a build dependency and keeps generation deterministic and fast.

### 3.2 Layout algorithm

BSP partition of a 64-unit tile grid into rooms, with one-tile wall bands. Doors are
placed **after** all leaves exist, at a band position whose two neighbours are floor,
so a deeper wall band cannot swallow a door. Additional loops are added only between
isolated wall tiles. This produced 120/120 valid maps across four tiers and exact
agreement with `doom_env.mapinfo.door_sectors`.

### 3.3 Visual semantics

Because exit and colour are not structured:

- locked doors use `DOORBLU` / `DOORRED` / `DOORYEL`;
- every walk-over exit gets a bright `EXITSIGN` / `SW1EXIT` / `SW1BRN1` wall behind it;
- walls, floors, ceilings, light levels and door/exit textures are randomised per seed.

This makes the RGB auxiliary heads learn a *semantic* cue rather than a single
texture, and the held-out Freedoom maps test transfer.

---

## 4. Algorithm choices

### 4.1 Why PPO, not DQN or D4PG

- Discrete 9-bit action space with a factorised structure → policy gradient is natural.
- PPO’s clipped surrogate is stable with a small replay-free batch and works with
  recurrent state; DQN’s off-policy replay is awkward with recurrent state and
  non-stationary map curricula.
- D4PG/continuous control is unnecessary: the action space is discrete.
- We keep PPO on-policy and add a BC prior to compensate for sample cost.

### 4.2 Why behaviour cloning first

The teacher solves navigation (23/24 door-free exits) but is unreliable on multi-room
doors and combat. BC transfers the reliable part in ~100k transitions instead of tens
of millions of RL steps. The final policy is still trained parameters; the teacher is
training-only scaffolding, explicitly allowed.

### 4.3 Recurrent core

A GRU(384) is the best cost/benefit on one CPU thread: ~1.1 M parameters, ~0.5 ms per
step, and it handles door timing and short-term enemy motion. Transformers over long
histories would blow the budget; a pure feed-forward policy cannot remember a door.

### 4.4 Memory: deterministic map + event tokens

The literature offers three families:

1. **Learned recurrent maps** (Neural Map, differentiable occupancy grids): expressive
   but hard to train and debug in 24 hours.
2. **Learned SLAM / topological graphs** (Active Neural SLAM, Neural Topological SLAM):
   strong but require a planning stack, which the rules discourage at inference.
3. **Deterministic projection + learned readout**: project rays/entities into a grid at
   an estimated pose, then learn to read it. Cheap, debuggable, and legal as perception.

Revision 2 chooses (3) plus **event tokens** for long-horizon key/door memory. Heading
integration is exact from the measured turn table; translation comes from ego velocity
and is validated by drift measurement before adding scan matching.

### 4.5 Exploration and reward shaping

- **Sector novelty** matches the scorer’s coverage unit; a cell-novelty term adds a
  dense signal with a cap so it cannot outweigh the exit.
- **Potential-based shaping** (Ng et al., 1999) with a geodesic distance field preserves
  the optimal policy; a raw distance penalty does not.
- **Curriculum** (arenas → rooms → key chains → full pool) is cheaper than RND/NGU and
  sufficient with a BC prior. RND/PLR are deferred, not rejected.

### 4.6 Action factorisation

Nine independent Bernoulli outputs allow 512 combinations, most contradictory. Six
heads (move/strafe/turn/attack/use/next) reduce this to 216 meaningful joint actions
and act as action masking, which the rules explicitly allow. The emitted vector is
still a valid `MultiBinary(9)`.

### 4.7 Perception

- Rays: circular 1-D CNN preserves the angular topology (ray 0 is ahead, 16 left,
  32 behind).
- Entities: type embedding + one multi-head self-attention layer + masked mean/max.
  This is permutation-invariant and handles zero to 32 entities; a flattened vector
  would not.
- RGB: 4× pooled to 30×40 to preserve colour at 3.6 KB/frame. The exit/key-colour heads
  force the encoder to use it.
- Depth: deferred; rays already encode metric free space.

---

## 5. Verified vs assumed

| Claim | Status |
|---|---|
| Exit is not in rays/entities | **Verified** in `spaces.py`, `sensors.py` |
| Key/lock colour hidden until pickup | **Verified** in `sensors.py` |
| Turn table is stateful | **Verified** by measurement |
| Node-less binary maps load | **Verified** |
| UDMF things do not spawn | **Verified** (player only until binary THINGS) |
| 120/120 generated maps valid | **Verified** |
| Expert 23/24 door-free exits | **Verified** |
| Expert fails multi-room doors | **Verified** |
| 1.56 M model at 0.45 ms CPU | **Verified** |
| Validator p50 0.6 ms | **Verified** |
| PPO will reach 50–150M steps on this host | **Assumed** — must be benchmarked |
| RGB exit head will generalise to Freedoom | **Assumed** — held-out test |
| Memory tokens improve tier 3 | **Assumed** — causal ablation planned |
| Official scoring weights | **Unknown** — proxy only |

---

## 6. References

- Kempka, Wydmuch, Rzepecki, Jaśkowski, Wąsik. *ViZDoom: A Doom-based AI Research
  Platform for Visual Reinforcement Learning.* 2016. arXiv:1605.02059.
- Wydmuch, Kempka, Jaśkowski. *ViZDoom Competitions: Playing Doom from Pixels.* 2018.
- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization
  Algorithms.* 2017. arXiv:1707.06347.
- Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.* 2015. arXiv:1506.02438.
- Ng, Harada, Russell. *Policy Invariance Under Reward Transformations: Theory and
  Application to Reward Shaping.* ICML 1999.
- Mnih et al. *Human-level control through deep reinforcement learning.* Nature 2015.
- Parisotto, Salakhutdinov. *Neural Map: Structured Memory for Deep Reinforcement
  Learning.* ICLR 2018. arXiv:1702.08360.
- Chaplot, Gandhi, Gupta, Salakhutdinov. *Learning to Explore using Active Neural
  SLAM.* ICLR 2020. arXiv:2004.05155.
- Chaplot, Salakhutdinov, Gupta, Gupta. *Neural Topological SLAM for Visual
  Navigation.* NeurIPS 2020.
- Burda, Edwards, Storkey, Klimov. *Exploration by Random Network Distillation.* 2018.
  arXiv:1810.12894.
- Badia et al. *Never Give Up: Learning Directed Exploration Strategies.* ICLR 2020.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction
  to No-Regret Online Learning.* AISTATS 2011 (DAgger).
- id Software. *Doom* (1993) and the Doom Wiki: player movement, linedef types, thing
  types, WAD format.
- ZDoom Wiki: UDMF, node building, line specials.
- Freedoom project (BSD-licensed IWAD used by the official wrapper).
- Farama Foundation. *Gymnasium* API.
- Official template: `Reinforce-SST/devforge-doom-template`, contract `1.0.0-draft`.

---

## 7. Open research questions for Phase 2

1. Does an RGB exit head trained on generated signs transfer to Freedoom switches?
2. Do event tokens improve key→locked-door return rate beyond a GRU, measured by a
   memory-wipe test?
3. How much does translation drift accumulate per 200 steps, and does a cheap
   translation-only scan match recover it?
4. Is sector novelty enough, or does the policy need frontier clusters to explore
   large Freedoom maps?
5. What is the real PPO throughput with 8–10 workers on this host, and how many steps
   fit before the hour-20 freeze?
6. Does colour randomisation prevent overfitting to generated textures, or does it
   slow learning enough that held-out Freedoom suffers?
