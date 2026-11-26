# Research Review — Track 1 “Can It Play DOOM?”

**Author’s note.** This document is intentionally long. It is the research and
engineering reference for the 24-hour DoomForge submission to Yugaantar DevForge Track 1.
It complements `PLAN.md` (the plan of record), `progress.md` (the lab log),
`docs/RESULTS.md` (the measured benchmark), and `docs/ARCHITECTURE.md` (the module
reference). It does not replace any of them; it explains the reasoning behind them and
what a strong implementer should do next.

**Evidence tags.** **M** = measured in this repository (`progress.md`,
`docs/RESULTS.md`, `docs/brief/`). **A** = derived from the official contract
(`docs/CONTRACT.md`, `docs/EVALUATION.md`, `docs/RULES.md`). **S** = speculative or
extrapolated from the literature. The official scoring weights and tier multipliers are
hidden; every proxy number in this document is an assumption, not the organiser’s
formula. Reported results from papers are described qualitatively unless a number is
well established; uncertain claims are marked “reported” or “roughly.”

---

## 0. Executive summary

The task is a partially observable, long-horizon control problem in ViZDoom with a hard
inference budget and a non-binary score dominated by reaching the exit. The central
difficulty is not visual control. It is that the information needed to act — where the
exit is, which key opens which door, where the agent has been — is either absent from the
structured observations or hidden behind a colour flag that only appears after pickup.
The policy must build and use an explicit memory of the map, and it must be trained from a
teacher whose actions are not a function of a single observation.

The measured evidence is unambiguous:

- A recurrent multimodal policy trained by behaviour cloning on 100,921 transitions
  learned movement, strafing and shooting, but **turn accuracy stuck at 43.2%** against a
  33.3% chance rate, exact six-head match was 20.8%, and held-out exit rate was **0/4**.
  The teacher’s turn was computed from a global A* waypoint that the observation did not
  contain. **M**
- The best shipped checkpoint (`bc_arena`) reached **5/8** tier-1 arena exits but **0/8**
  on tier-2 rooms, 0/8 tier-3 key/lock, 0/8 tier-4 gauntlet, and 0/10 Freedoom MAP23–32;
  coverage was 6.1%, 4.2%, 2.0% and 3.6%. **M**
- A 25-update PPO run on 25.6k environment steps improved tier-1 to **6/8** and the
  tier-weighted proxy to **+8.09**, but left door drills at 0/8 and Freedoom coverage at
  1.6%. PPO learned to fight, not to explore. **M**
- Door USE is learnable in isolation: 360 drill episodes took door drills from 0/8 to
  **6/8** in two epochs. The same checkpoint opened **0/8** room doors. A room-door
  generator with random start pose produced **200/200** teacher successes; the student
  scored 4/8 on drills and **0/8** on room doors. **M**
- Sequential specialist fine-tuning causes catastrophic forgetting: drill-only training
  dropped tier-1 exits from 2/2 to 0/2 and the tier-1 proxy from +11.4 to −1.2. A
  KL-anchored merge measured tier-1 1/8 and drills 0/8. A success-only mixed replay
  measured tier-1 4/8, tier-2 −0.54, drills 0/8. The MoE gate could route doors (6/8
  drills at threshold 0.1) but cost tier-1 exits and lowered tier-2/3/4 proxies. **M**
- Dead reckoning drifts: after fixing the turn table and velocity timing, heading error
  over 250 steps fell from 63° to **16°**, position error from 907 to **361 units** — still
  3.8 cells at 96 units/cell, enough to smear a 64-unit door out of a coarse map. **M**

The recommendation is **B+ lite v2**: keep the existing multimodal encoder and GRU, but
add a scan-matched 8-channel allocentric map, heading-aligned local/global crops, a bank of
≤32 event tokens (locked doors, keys, exit cues, frontiers), a goal vector, an asymmetric
privileged critic, and a **single skill-balanced policy** trained with BC → DAgger → PPO.
The manager over tokens is a stretch goal, not a dependency. Inference remains ~3–6 ms on
one CPU thread, 7–13× inside the 40 ms budget. The benchmark suite is a first-class
deliverable: 14 diagnostic categories, 5 engine seeds, stratified bootstrap confidence
intervals, leakage controls, and a proxy score that mirrors the official non-binary
scoring.

The blunt version: with 12 hours, do not train a world model, do not add a transformer, do
not run three seeds, and do not fine-tune another specialist. Fix the observation function,
fix the data distribution, and add an asymmetric critic. That is the highest expected value
per hour.

---

## 1. Problem statement

### 1.1 The contract

The official wrapper exposes a `gymnasium.spaces.Dict` with five keys: `rgb`
(`uint8[120,160,3]`), `depth` (`uint8[120,160]`), `vitals` (`float32[17]`), `rays`
(64 distances in `[0,1]` plus 64 hit types in `{0..5}`), and `visible_entities` (up to 32
entities with 5 features, a type id and a mask). The action is `MultiBinary(9)`, held for
4 game tics per decision. Episodes run 2,100–4,200 tics, i.e. 525–1,050 decisions.

| Dimension | Specification | Consequence |
|---|---|---|
| Observations | 64 ego-centric rays (`distance`, 6 hit types), ≤32 entities (5 features + type + mask, 90° FOV), 17 vitals, 120×160×3 RGB, optional depth | No global coordinates, no map, no god view; entities are only in a 90° cone and line-of-sight; rays stop at walls, doors, enemies and items |
| Action | `MultiBinary(9)`: forward, backward, strafe L/R, turn L/R, attack, use, next weapon; held 4 tics | 525–1,050 decisions/map; USE is a button-down edge; USE on a moving door reverses it; `SELECT_NEXT_WEAPON` is also edge-triggered |
| Budget | 1 CPU thread, 40 ms average per decision, 2 GB VRAM, 8 GB RAM, 60 s init, 10 s reset, 10 s first `act`, 5 s any later `act` | A 1–3 M-parameter model is comfortable; training throughput, not inference, is the binding constraint |
| Maps | Unseen; tier 1 combat and survival, tier 2 multi-room exploration, tier 3 keycard → locked door → exit, tier 4 full gauntlet | Generalisation across geometry, texture, monster mix and key ordering is the test |
| Score | Tier-weighted, non-binary: exit (largest) > keys/doors/locks > unique sectors > kills, minus damage and early death | Partial progress counts; a perfect shooter that never exits loses the largest term |
| Rules | Actions must come from trained parameters; no rule trees, A*/BFS/Dijkstra or `atan2` aimbots at inference; training scaffolding and privileged state allowed | A deterministic map writer is legal perception; a planner at inference is not |

The contract’s **verified gaps** matter more than its shape:

1. **The exit is not in the structured observations.** `RAY_HIT_TYPES` has no exit class;
   `ENTITY_TYPES` has no exit class; `mapinfo.py` parses door and lock specials only; the
   wrapper sets `_exit_reached` only when the episode ends. The agent must learn exits from
   RGB or depth, with privileged linedef labels during training.
2. **Key and lock colour is hidden until pickup.** Every key is classified as `key`; every
   lock as `locked_door`; the six colour flags in `vitals[8:14]` turn on only after pickup.
   The agent must read colour from RGB or try remembered doors.
3. **USE is a button-down edge.** Holding USE across steps presses it once. Pressing USE on
   a door that is already moving reverses it, so spamming USE keeps a door bouncing.
4. **Turn acceleration is stateful.** Measured per-decision turn deltas are
   `0.000°, 3.516°, 14.063°, 14.063°...` on the first hold, `1.758°, 12.305°, 14.063°...`
   after one warm-up step, and `7.031°, 12.305°, 14.063°...` after two or more. A single
   14.06° constant is wrong.
5. **Coverage is sector-based.** The scorer’s `sector_polygons()` docstring says coverage
   is counted in unique sectors, not 64-unit cells. A cell-novelty reward is a proxy, not
   the metric.

### 1.2 Formal framing

Let the true state be

```
s_t = (pose_t, map_geometry, door_states, key_states, monster_states, projectile_states,
       health_t, armour_t, ammo_t, weapon_t, time_t)
```

The agent does not observe `s_t`. It observes

```
o_t = (rays_t, entities_t, vitals_t, rgb_t)
```

where `rays_t` is a 64-beam range image with hit types, `entities_t` is a variable-length
set of at most 32 visible objects with relative positions and velocities, `vitals_t`
contains health, armour, weapon, four ammo pools, six key flags and ego velocity, and
`rgb_t` is a 120×160 colour image. The exit and the colour of keys/locks are **not** in
`o_t`. The policy is a recurrent function

```
a_t ~ π_θ(a_t | o_{1:t}, a_{1:t-1}),   a_t ∈ {0,1}^9
```

and the environment is a POMDP with a long horizon, sparse top-level reward, and hidden
task-relevant variables. The belief state must be approximated by the recurrent hidden
state plus an explicit memory (map and tokens). This is the formal reason a pure GRU
policy fails: the same observation can correspond to different global positions, goals and
key inventories, so the optimal action is not a function of `o_t`.

The score is a weighted sum over maps and tiers. Our proxy is

```
S_map = 10·exit + 3·keys + 3·locks + 0.5·doors
      + min(0.3·sectors, 10) + 0.6·kills
      − 0.01·damage − (3·(1 − t/T) + 1)·death
S = Σ_tier w_tier · mean_{maps in tier} S_map
```

with assumed tier weights `[1.0, 1.2, 1.5, 2.0]`. The official weights are hidden; the
ordering (exit > progression > exploration > combat, minus damage/death) is stated in
`docs/EVALUATION.md`.

### 1.3 Why it is hard

**1. Partial observability is structural, not incidental.** The entity list covers only a
90° horizontal field of view and line-of-sight; rays cover 360° but with 64 samples, a
1,024-unit range, and a stop at the first blocker. A door behind the agent, a key in a side
room, or a locked door seen 300 steps ago is not in the current observation. The agent must
integrate observations over time and resolve aliasing: many different rooms look similar
from the inside.

The measured consequence is severe. Behaviour cloning on 100,921 transitions reached
teacher-forced move accuracy 85.6%, strafe 76.0%, attack 92.4%, but **turn accuracy 43.2%**
against a 33.3% chance rate; exact six-head match was 20.8%; held-out exit rate was 0/4.
More BC epochs did not help: cross-entropy plateaued at ~1.9 and turn accuracy stayed near
chance. The policy learned “forward + turn-left” and circled the start room. The missing
variable was the teacher’s global A* waypoint, which the observation did not contain.

**2. The horizon is long relative to the memory.** Episodes run 525–1,050 decisions. A
key→locked-door→exit chain can require 300–700 steps, including backtracking through
rooms the agent has already seen. A GRU’s effective memory is short and decays; dead
reckoning drifts. Measured: after fixing the turn table and velocity timing, heading error
over 250 steps fell from 63° to 16° and position error from 907 to 361 units. At 96
units/cell, 361 units is 3.8 cells; a 64-unit door is smaller than one cell. The map
smears, the frontier channel is noisy, and a door token points to the wrong place. Memory
must be explicit, and pose must be corrected, not just integrated.

**3. Reward is sparse at the top and dense only if we build the density.** The exit is the
largest term. In our proxy it is +10, a key or lock +3, an ordinary door +0.5, a new sector
+0.3 (capped at +10), a kill +0.6, damage −0.01/HP, and death `−(3(1−t/T)+1)`. A policy
that farms kills or spins for sectors can look busy while losing the dominant term. Dense
shaping must be potential-based or tightly capped, or it changes the optimal policy. The
old raw distance penalty (`−0.0002 × BFS cells`) was not potential-based and was removed
for exactly this reason.

**4. The exit is invisible to the structured senses.** There is no ray hit type or entity
class for it. The wrapper only learns that the exit was reached when the episode ends. The
exit must be learned from RGB or depth, with privileged exit-linedef labels during
training. Measured: exit-visible positives are only 0.5–2% of steps on door-free maps; we
used `pos_weight=12` and oversampled near-exit frames, but the actor still got 0/10 exits
on Freedoom MAP23–32. Detection is necessary but not sufficient: the agent also needs to
*go* to the remembered exit cue.

**5. Key and lock colour is a hidden variable.** Every key is reported as `key`; every lock
as `locked_door`; colour is only exposed after pickup via the six vitals flags. A
colour-blind policy cannot reliably order a two-key chain. Trying remembered doors is
expensive and dangerous: USE is an edge, and pressing USE on a moving door reverses it. A
policy that spams USE can keep a door closed forever.

**6. Resources and combat are coupled to progress.** Health, armour, four ammo pools, six
key flags and the selected weapon are all in the 17-value vitals vector. The pistol
refires about every 14 tics (~3.5 decisions), so firing every step is wasteful; the damage
penalty and early-death penalty make suicide-for-kills irrational. Tier 4 adds scarce ammo
and multiple locks. Combat is not a separate mini-game; it is a tax on the path to the
exit.

**7. The inference budget is generous; the training budget is not.** The shipped 1.807 M
model runs at 0.45–1.9 ms per decision on one CPU thread, roughly 20–40× inside the 40 ms
budget. That is not the bottleneck. The bottleneck is sample throughput on a 10-core
M1/16 GB host with no CUDA: a 25-update PPO run used only 25.6k environment steps.
Realistic remaining compute is 10^5–10^6 decisions, not the 10^8 of the ViZDoom
competition literature. This single fact rules out from-scratch pixel RL, large
distributed RL, and most model-based recipes.

**8. Generalisation is the actual test.** The hidden suite is unseen maps. The shipped
`bc_arena` scored 5/8 on tier-1 arenas but 0/8 on tier-2 rooms, 0/8 tier-3 key/lock, 0/8
tier-4 gauntlet, and 0/10 Freedoom MAP23–32. Coverage was 6.1%, 4.2%, 2.0% and 3.6%. A
two-map eval had shown 2/2 tier-1 exits; the eight-map bench showed 2/8. The hidden suite
is larger and more varied than any local eval we have run.

### 1.4 What “good” looks like

The tier-weighted sum rewards exit first, progression second, coverage third, kills
fourth. Exiting 30% of tier-2 maps and 10% of tier-3 maps beats a better shooter that
never leaves the start room. The research question is therefore not “how do we aim
better?” but “how do we make the exit, the key and the locked door observable and
memorable within a 10^5–10^6 decision budget?” Every architectural choice in this document
is judged against that question.

---

## 2. Prior art

The literature splits into seven families. For each, this section covers the mechanism,
where it works, sample and compute cost, inference cost, fit to our exact constraints, and
failure modes. Sample costs are order-of-magnitude from the original papers; they are not
promises for ViZDoom. Inference costs are per decision on one CPU thread.

### 2.1 ViZDoom and the pixel-RL canon

**The platform.** ViZDoom (Kempka, Wydmuch, Rzepecki, Jaśkowski and Wąsik, 2016,
arXiv:1605.02097) is a 3D first-person-shooter research platform built on a Doom engine.
It exposes RGB, depth, object labels, game variables and configurable scenarios; it runs
faster than most 3D simulators because the engine is lightweight, and it supports
frame-skip, resolution changes and headless execution. The original paper established
DQN and DRQN baselines on deathmatch and health-gathering scenarios and showed that
pixel-only agents could learn aiming, dodging and resource collection with millions of
frames. The ViZDoom competitions (Wydmuch, Kempka and Jaśkowski, 2018, arXiv:1809.03470)
ran deathmatch and health-gathering tracks on fixed maps and reported that the strongest
entries combined recurrent networks with game features, frame stacking and
scenario-specific reward shaping. The competition protocol is a useful reference for
evaluation discipline, but its maps are short, fixed and combat-centric; our hidden suite
is longer and tests key/lock progression.

**DQN.** Deep Q-Networks (Mnih et al., 2015, arXiv:1312.5602) learn `Q(s,a)` with a
convolutional network, a replay buffer and a periodically copied target network. The loss
is

```
L(θ) = E_{(s,a,r,s')~D} [ ( r + γ max_{a'} Q_{θ−}(s',a') − Q_θ(s,a) )^2 ]
```

DQN reached human-level performance on 29 of 49 Atari games, but it needs 10^7–10^8
frames, a large replay buffer, and a stationary observation distribution. In ViZDoom
deathmatch, DQN can learn to strafe and shoot, but it overfits to weapon and monster
layouts and has no memory for keys or backtracking. It is a poor fit for our 10^5–10^6
decision budget and for a partially observable task with hidden state.

**DRQN.** Deep Recurrent Q-Learning (Hausknecht and Stone, 2015, arXiv:1507.06527)
replaces the first fully connected layer with an LSTM/GRU and trains on sampled sequences
from replay. The recurrent state carries information across frames, which helps with
flickering observations and occlusions. DRQN improved performance on partially observable
Atari variants and became the backbone of the ViZDoom deathmatch entries. The engineering
difficulty is recurrent replay: the hidden state must be stored at the sequence start and
burned in before the loss; off-policy corrections are approximate; and long sequences are
expensive. DRQN is conceptually the right core for our task, but it is not enough on its
own: a GRU cannot remember a door seen 300 steps ago with the precision needed for
backtracking.

**Lample and Chaplot (2017, arXiv:1609.05521).** “Playing FPS Games with Deep
Reinforcement Learning” used a DRQN with game features (health, ammo, weapon, etc.), a
curriculum over increasing difficulty, and a shaped reward to learn ViZDoom deathmatch.
The paper reported strong deathmatch performance and popularised the combination of
recurrent Q-learning, game features and curriculum that the ViZDoom competition entries
used. Its limitation for us is that game features are privileged relative to our contract:
we get vitals, but not the enemy/geometry features the agent used. The curriculum idea
transfers; the feature engineering does not.

**Papoudakis et al. (2018, arXiv:1807.01960).** “Deep Reinforcement Learning for Doom using
Unsupervised Auxiliary Tasks” proposed a divide-and-conquer architecture with auxiliary
prediction tasks (e.g., predicting game variables or future frames) to improve sample
efficiency in Doom. Auxiliary tasks are a good fit for our exit/colour heads: they force
the shared encoder to represent task-relevant variables. The paper’s sample cost is still
10^7 frames, and its auxiliary tasks are scenario-specific.

**D4PG and DDPG.** Deep Deterministic Policy Gradient (Lillicrap et al., 2016,
arXiv:1509.02971) learns a deterministic actor `μ_θ(s)` and a critic `Q_φ(s,a)` with
off-policy replay and target networks. D4PG (Barth-Maron et al., 2018, arXiv:1804.08617)
adds a distributional critic, N-step returns, prioritized replay and distributed actors.
It reported strong results on a suite of continuous-control tasks. D4PG is strong on
continuous control but assumes a continuous action space. Our action is a 9-bit discrete vector with strong combinatorial structure;
a deterministic continuous actor would need a thresholding layer and would lose the
factorised action semantics. D4PG is not a fit.

**A3C, IMPALA and APPO.** Asynchronous Advantage Actor-Critic (Mnih et al., 2016,
arXiv:1602.01783) runs multiple actor threads and updates a shared policy asynchronously.
It reported better Atari performance than DQN with less wall-clock time on a multi-core
CPU. IMPALA (Espeholt et al., 2018, arXiv:1802.01561) decouples actors from a learner and
uses V-trace to correct off-policy lag:

```
v_s = V(s) + Σ_{t=s}^{s+n−1} γ^{t−s} (Π_{i=s}^{t−1} c_i) δ_t V
δ_t V = ρ_t ( r_t + γ V(s_{t+1}) − V(s_t) )
c_i = min(c̄, ρ_i),  ρ_i = min(ρ̄, π(a_i|s_i) / μ(a_i|s_i))
```

IMPALA reported large gains on DMLab-30 and Atari with stable off-policy correction. APPO
(Asynchronous Proximal Policy Optimization, the algorithm used in Sample Factory;
Petrenko et al., 2020, arXiv:2006.11751) combines PPO’s clipped objective with
asynchronous actors and a decoupled inference server. Sample Factory reported throughput
around 100k frames per second on a single machine for 3D environments. These systems are
excellent when you have many cores and a GPU; we have one CPU thread at inference and a
single 10-core host at training. Their algorithmic lessons (clipped updates, decoupled
inference, V-trace) matter; their scale does not.

**PPO.** Proximal Policy Optimization (Schulman et al., 2017, arXiv:1707.06347) maximises
a clipped surrogate objective:

```
L^CLIP(θ) = E_t [ min( ρ_t(θ) Â_t, clip(ρ_t(θ), 1−ε, 1+ε) Â_t ) ]
ρ_t(θ) = π_θ(a_t|o_t) / π_{θ_old}(a_t|o_t)
L(θ) = L^CLIP − c_1 L^VF + c_2 E_t[ H(π_θ(·|o_t)) ]
```

with advantages from Generalized Advantage Estimation (Schulman et al., 2015,
arXiv:1506.02438):

```
δ_t = r_t + γ V(o_{t+1}) − V(o_t)
Â_t = Σ_{l=0}^{T−t} (γλ)^l δ_{t+l}
```

PPO is on-policy, replay-free and compatible with recurrent state, which is why it is the
right algorithm for our BC-initialised fine-tuning. It reported better sample efficiency
than TRPO on continuous control and Atari. Its failure modes are well known:
sensitivity to advantage normalisation, clipping, learning rate and entropy; recurrent PPO
needs stored hidden states and Burn-In; and it is sample-inefficient relative to
off-policy methods. At 10^5–10^6 decisions, the BC prior and dense shaping matter more
than the PPO variant.

**R2D2 and distributed RL.** R2D2 (Kapturowski et al., ICLR 2019, OpenReview
`r1lyTjAqYX`) made recurrent off-policy RL work at scale with stored recurrent states,
burn-in, overlap between stored and burn-in sequences, and value-function rescaling. It
reported state-of-the-art Atari-57 results at publication. Ape-X (Horgan et al., 2018,
arXiv:1803.00933) distributed prioritized replay across many actors; SEED (Espeholt et
al., 2020, arXiv:1910.06591) moved inference to a central server and accelerated RL;
DD-PPO (Wijmans et al., 2019, arXiv:1911.00357) trained near-perfect PointGoal navigators
with 2.5 billion frames. These are existence proofs for recurrence and distribution, not
recipes for a 24-hour CPU-only run.

| Method | Mechanism / where it works | Sample cost | Inference | Fit / failure |
|---|---|---|---|---|
| ViZDoom (arXiv:1605.02097) | 3D FPS platform; RGB, depth, game vars, scenarios | — | — | High fit; pixel RL is sample-hungry |
| ViZDoom competitions (arXiv:1809.03470) | Deathmatch/health tracks, fixed maps | 10^7–10^8 frames | Cheap | Protocol reference; narrow and short-horizon |
| DQN (arXiv:1312.5602) | Off-policy Q-learning, replay, target net | 10^7–10^8 | Cheap | Sample cost; no memory |
| DRQN (arXiv:1507.06527) | Recurrent Q for POMDPs | 10^7 | GRU, cheap | Recurrent replay is awkward |
| Lample & Chaplot (arXiv:1609.05521) | DRQN + game features + curriculum | 10^6–10^7 | Cheap | Privileged features; task-specific |
| Papoudakis et al. (arXiv:1807.01960) | Divide-and-conquer + auxiliary tasks | 10^7 | Cheap | Sample cost; auxiliary heads transfer |
| D4PG/DDPG (arXiv:1804.08617, 1509.02971) | Distributional deterministic PG | 10^7 | Cheap | Continuous actions; ours are discrete |
| A3C/IMPALA/APPO (arXiv:1602.01783, 1802.01561, 2006.11751) | Async/distributed PG; V-trace; decoupled inference | 10^8–10^9 | Cheap/actor | Needs many cores/GPUs |
| PPO (arXiv:1707.06347) | Clipped surrogate PG | 10^6–10^8 | Cheap | Recurrent PPO is delicate |
| R2D2/Ape-X/SEED/DD-PPO (OpenReview `r1lyTjAqYX`; arXiv:1803.00933, 1910.06591, 1911.00357) | Distributed/recurrent RL at scale | 10^8–10^9 | Cheap | Compute; engineering |
| Sample Factory/Podracer (arXiv:2006.11751, 2104.06272) | High-throughput async; inference-aware architecture | 10^8+ | Cheap | Many cores/GPUs |

The practical takeaway: the canon assumes 10^7–10^9 frames. We have 10^5–10^6 decisions.
Imitation and privileged labels are not a shortcut; they are the only route to competence
inside the budget.

### 2.2 Navigation, SLAM and spatial memory

**The problem.** Navigation methods decompose into mapping (estimate the layout),
localisation (estimate pose in the map), and planning (choose a path). Our contract forbids
the third at inference, but the first two are legal perception. The literature is useful
because it shows which memory representations actually survive long horizons and which
ones collapse under pose noise.

**Neural Map.** Parisotto and Salakhutdinov (2018, arXiv:1702.08360) proposed a 2D matrix
memory written by a convolutional network at the agent’s location and read by attention.
The write is a learned function of the observation and the previous memory; the read
attends over memory locations. It reported improvements over an LSTM on 3D maze
navigation, especially when the goal was far away. The appeal is a learned, differentiable
map; the cost is training instability, memory collapse (the writer learns to ignore the
map), and a large sample budget (10^7 steps). Our deterministic writer avoids the training
instability but gives up the ability to learn what to write; the plan keeps a learned
readout and defers a learned write.

An occupancy grid is the classical alternative. Each cell stores a log-odds belief, and
each observation updates it:

```
l_t(i) = l_{t−1}(i) + log( p(m_i = occupied | z_t) / (1 − p(m_i = occupied | z_t)) ) − l_0
l_0 = log( p_0 / (1 − p_0) )
```

The update is cheap and incremental; the difficulty is inverse-sensor-model calibration
and pose noise. Our map writer is a variant of this idea with semantic channels (door,
lock, exit, frontier) instead of a single occupancy probability.

**Active Neural SLAM.** Chaplot, Gandhi and Gupta (2020, arXiv:2004.05155) combine a
learned projective map, a global policy that selects a long-term goal, a local policy that
navigates to it, and a classical planner for the goal. The map is built by projecting
egocentric depth into a top-down grid; the global policy runs every K steps; the local
policy runs every step. It reported strong Habitat navigation results and much better
generalisation than reactive baselines. The architecture is close to what we need, except
that the planner at inference is disallowed. The legal subset is: deterministic map writer,
learned global goal selector (manager), learned local policy. The failure modes are map
drift, planner dependence, and sim-to-real gap.

**Neural Topological SLAM.** Chaplot et al. (2020, arXiv:2005.12256) build a topological
graph of visited locations, store node embeddings, and localise the agent by matching the
current observation to nodes. It reported improvements over metric-map baselines on
ObjectNav. The graph is attractive for our event tokens: a node can represent a door, a
key, or a room, and edges encode reachability. The engineering cost is higher than a grid
and the localisation is sensitive to perceptual aliasing. For 24 hours, a token bank with
relative positions is a cheaper approximation.

**Occupancy Anticipation.** Ramakrishnan, Al-Halah and Grauman (2020, arXiv:2008.09285)
train a model to predict occupancy beyond the current field of view, then use the predicted
map for exploration and navigation. It reported improved exploration and navigation in
Habitat. The idea maps directly to our exit-likelihood channel: predict what is not
visible and write it into the map. The failure mode is prediction blur and false positives;
calibration and a precision/recall probe are essential.

**Frontier exploration.** Yamauchi (1997) introduced the classical frontier approach: find
the boundary between known free space and unknown space, choose the nearest frontier, and
navigate to it. It is cheap, interpretable and surprisingly strong in unknown indoor
environments. The catch is that it is a planner/heuristic; using it at inference would
violate the spirit and likely the letter of the rules. It is excellent as a **teacher**:
the frontier teacher explores 160–305 cells on door-free maps, and its goal (the nearest
unvisited cell) is observable in the student’s own map. That makes it imitable in a way the
global A* teacher is not.

**ObjectNav, Habitat and ProcTHOR.** Habitat (Savva et al., 2019, arXiv:1904.01201) is a
photorealistic embodied-AI simulator with PointNav, ObjectNav and instruction-following
tasks; the ObjectNav evaluation protocol (Anderson et al., 2018, arXiv:1807.06757) defines
success rate and SPL. ProcTHOR (Deitke et al., 2022, arXiv:2206.06994) generates thousands
of procedural houses for training and evaluates generalisation to unseen houses. These
benchmarks contribute two things: a **procedural train/test split** discipline and
**path-efficiency metrics** (SPL). They do not solve our key/lock/exit problem, but they
tell us how to measure whether a policy is exploring efficiently rather than wandering.

**Semantic mapping.** Semantic mapping projects detected objects or labels into the map
and uses the resulting semantic layer for goal selection. In our setting the semantic
layers are exactly the missing observations: exit likelihood, door colour, key colour, and
frontier. A semantic map is legal because it is built from observations; the policy still
chooses the action. The failure mode is label noise: a false exit detection can send the
agent to a wall. Calibration and a decay/confidence channel mitigate it.

**Cognitive Mapping and Planning.** Gupta et al. (2017, arXiv:1702.03920) learn a spatial
map and a planner jointly for visual navigation. It is a strong existence proof for
map-based navigation, but the planner is the part we cannot use. The map encoder and the
auxiliary losses are the transferable pieces.

**Semi-parametric topological memory.** Savinov et al. (2018, arXiv:1803.00653) store
observations in a non-parametric memory and retrieve the nearest neighbour to decide where
to go. It is conceptually close to event tokens and episodic memory (Neural Episodic
Control, Pritzel et al., 2017, arXiv:1703.01988). The failure modes are memory growth,
perceptual aliasing, and the need for a good distance metric.

| Method | Mechanism / fit | Sample cost | Inference | Failure |
|---|---|---|---|---|
| Neural Map (arXiv:1702.08360) | Learned 2D memory written by a CNN | 10^7 | Cheap | Hard to train; memory collapse |
| Active Neural SLAM / Cognitive Mapping (arXiv:2004.05155, 1702.03920) | Projective map + global/local policy + planner | 10^6–10^7 | Planner cost | Planner at inference is disallowed |
| Neural Topological SLAM (arXiv:2005.12256) | Topological graph + localisation | 10^6 | Cheap | Pose drift; engineering |
| Occupancy Anticipation (arXiv:2008.09285) | Predict occupancy beyond FOV | 10^6 | Cheap | Labels; prediction blur |
| Frontier exploration (Yamauchi, 1997) | Nearest unknown boundary; robotics | 0 | Cheap | Heuristic/planner-like; thrashes |
| ObjectNav/Habitat/ProcTHOR (arXiv:1904.01201, 1807.06757, 2206.06994) | Embodied navigation; SPL; procedural houses | 10^6 | Cheap | Sim-only; semantic categories differ |
| Semantic mapping | Project labels (exit, colour, key) into the map | 10^6 | Cheap | Label noise; calibration |
| Semi-parametric topological memory (arXiv:1803.00653) | Retrieval from remembered observations | 10^6 | Cheap | Memory growth; aliasing |

The consistent winner is an **explicit map plus a learned readout**, not an end-to-end
recurrent policy. The map makes the teacher’s goal observable and supports backtracking;
the planner is what the rules forbid. Our design uses a deterministic writer and a learned
reader, with semantic channels for exit and colour.

### 2.3 Exploration

**The problem.** A sparse-reward agent has no gradient toward an unvisited room unless
something makes novelty valuable. Exploration methods add an intrinsic reward, a
structured archive, or a curriculum. The right choice depends on the sample budget: RND
and NGU are designed for 10^8–10^9 frames; count-based and potential-based methods can work
at 10^5–10^6.

**ε-greedy.** With probability ε choose a random action, otherwise act greedily. It is the
cheapest baseline and is surprisingly hard to beat in small action spaces, but in a
64×64 map with doors, random actions mostly bounce off walls. It is a sanity baseline, not
a solution.

**Count-based exploration.** Bellemare et al. (2016, arXiv:1606.01868) use a density model
to assign a pseudo-count `N(s)` and reward

```
r_t^+ = β / sqrt(N(s_t))
```

or a similar decreasing function. The method is theoretically grounded (it approximates
visit counts) and works well in small state spaces. In our setting, cell or sector counts
are a natural fit: the scorer already rewards unique sectors, and a count bonus gives a
dense signal for visiting new areas. The failure mode is saturation: once every cell has
been visited once, the bonus vanishes; and in a large map, counts are sparse.

**Random Network Distillation.** Burda et al. (2018, arXiv:1810.12894) train a predictor
`f_θ` to match a fixed random network `f̄` on visited states and use the prediction error
as novelty:

```
r_t^int = || f_θ(s_t) − f̄(s_t) ||^2
L_RND = || f_θ(s_t) − f̄(s_t) ||^2
```

RND reported solving Montezuma’s Revenge without human demonstrations. It is robust to
some stochasticity but vulnerable to the “noisy TV” problem: a source of irreducible
randomness (a flickering screen, a random texture) dominates the error. ViZDoom has
stochastic monster behaviour and texture variation; RND would need careful observation
normalisation and maybe a learned embedding. At 10^7–10^8 frames it is plausible; at
10^5–10^6 it is not the best use of samples.

**NGU and Agent57.** Never Give Up (Badia et al., 2020, arXiv:2002.06038) combines episodic
novelty (a memory of states within the episode) with lifelong novelty (a global
state-conditional exploration bonus) and a family of policies with different discount
factors. Agent57 (Badia et al., 2020, arXiv:2003.13350) added a meta-controller over those
policies and was the first agent to outperform the human baseline on all 57 Atari games.
The sample cost is 10^9 frames and the engineering is substantial. The idea worth
borrowing is **episodic novelty**: a per-episode count of sectors/cells is cheap and
directly aligned with the scorer.

**Curiosity and ICM.** Pathak et al. (2017, arXiv:1705.05363) train a forward model to
predict the next latent state and an inverse model to predict the action, and use the
forward error as intrinsic reward:

```
φ_t = encoder(o_t)
φ̂_{t+1} = g(φ_t, a_t)
â_t = h(φ_t, φ_{t+1})
L = || φ̂_{t+1} − φ_{t+1} ||^2 + CE(â_t, a_t)
r_t^int = η || φ̂_{t+1} − φ_{t+1} ||^2
```

ICM improves exploration in sparse-reward games but is notoriously vulnerable to
stochastic dynamics and the noisy-TV problem. Our environment has stochastic monsters and
projectiles; a forward model would spend capacity predicting them. Not a priority.

**Potential-based reward shaping.** Ng, Harada and Russell (1999) proved that a shaping
reward of the form

```
F(s, a, s') = γ Φ(s') − Φ(s)
```

preserves the optimal policy for any potential `Φ`. This is the correct way to add dense
guidance without changing what the agent is trying to do. In our setting, `Φ` can be the
negative geodesic distance to the current objective (nearest unheld key if a lock blocks
the exit, else exit), or the negative distance to the nearest frontier. The old raw
distance penalty was not potential-based and could reward standing still; the plan
explicitly removed it.

**Go-Explore.** Ecoffet et al. (2019, arXiv:1901.10995) separate exploration from
exploitation: archive promising states, return to them (by restoring the simulator state),
explore from there, and only then train a policy. It solved Montezuma’s Revenge and
Pitfall and demonstrated that hard exploration is often a memory/return problem, not a
policy-gradient problem. The requirement is state restoration and deterministic resets.
ViZDoom supports save/load in some configurations, but relying on it for training is an
engineering risk; for 24 hours, a simpler episodic novelty bonus is safer.

**Quality diversity and MAP-Elites.** Mouret and Clune (2015, arXiv:1504.04909) maintain an
archive of solutions indexed by behaviour descriptors and fill it with diverse, high-
performing policies. In our setting, QD is more useful for **map/level diversity** than for
a single policy: generate a diverse pool of layouts (loops, dead ends, key orders) and
train on the archive. The failure mode is that QD optimises diversity, not the scorer’s
objective.

**Prioritized Level Replay.** Jiang et al. (2021, arXiv:2010.03934) prioritise training
levels by a score such as TD error or learning progress:

```
p_i ∝ (|TD_i| + ε)^β   or   (LP_i + ε)^β
P(i) = p_i / Σ_j p_j
```

and replay levels with high learning potential, with staleness caps. It reported improved
generalisation on ProcGen and other procedural suites. PLR is cheap, compatible with our
map pool, and directly addresses the “which map should I train on?” question. It is a
scheduler, not a new algorithm; add it after the baseline is stable.

| Method | Mechanism / fit | Sample cost | Inference | Failure |
|---|---|---|---|---|
| ε-greedy / count-based (arXiv:1606.01868) | Random exploration; pseudo-count novelty | 10^5–10^7 | Cheap | Too slow; counts saturate |
| RND (arXiv:1810.12894) | Random-network prediction error | 10^7–10^8 | Cheap | Noisy TV; reward hacking |
| NGU/Agent57/ICM (arXiv:2002.06038, 2003.13350, 1705.05363) | Episodic + lifelong novelty; forward-model curiosity | 10^7–10^9 | Cheap | Compute; noisy TV; complexity |
| Potential shaping (Ng et al., 1999) | `γΦ(s′)−Φ(s)` preserves the optimal policy | 0 | Cheap | Needs a good potential |
| Go-Explore (arXiv:1901.10995) | Archive, return, explore | 10^6–10^8 | Cheap | State restore; many rollouts |
| MAP-Elites / PLR (arXiv:1504.04909, 2010.03934) | Quality-diversity archive; prioritise levels by learning potential | 10^6–10^7 | Cheap | QD is not a policy learner; PLR needs a pool |

For 10^5–10^6, use potential shaping plus episodic sector/cell novelty plus PLR, not
RND/NGU. Exploration is an objective problem, not a memory problem.

### 2.4 Imitation and hybrid methods

**The problem.** RL from scratch is too expensive. A teacher can provide a strong prior,
but only if its actions are a function of the student’s observations. The literature on
imitation is therefore as much about **observability** as about loss functions.

**Behaviour cloning.** BC minimises

```
L_BC(θ) = −Σ_{h∈H} λ_h Σ_t log π_θ^h(a_t^h | o_t)
```

over demonstrations `(o_t, a_t)`, where `h` indexes the factorised action heads (move,
strafe, turn, attack, use, next). It is cheap, stable and easy to debug. The classical
failure is compounding error: a small policy error changes the state distribution, which
causes a larger error, and so on. The error can grow quadratically in the horizon for
naive BC. In our case there is a second failure: the teacher’s action may not be a function
of `o_t` at all, in which case the irreducible loss is high and no amount of data fixes
it. That is exactly what we measured: turn accuracy 43.2% against 33.3% chance.

**DAgger.** Dataset Aggregation (Ross, Gordon and Bagnell, 2011, arXiv:1011.0686) runs the
student, queries the expert on the **student’s** state distribution, and aggregates:

```
π_0 = BC(D)
for i = 1..N:
    run π_{i−1} to collect states s ~ d_{π_{i−1}}
    label each s with the expert π*(s)
    D ← D ∪ {(s, π*(s))}
    π_i = BC(D)
```

DAgger reduces covariate shift and has a no-regret guarantee under strong assumptions. It
is the right fix for our door and recovery failures, provided the expert is reliable on the
queried states. The frontier teacher is reliable on exploration states; the door teacher is
reliable on room-door states; the A* teacher is reliable on door-free navigation but not on
multi-room doors. DAgger should therefore use a **mixture of teachers**, not one global
expert.

**Expert iteration.** Anthony, Tian and Barber (2017, arXiv:1705.08439) alternate between
policy improvement with a search oracle and supervised training on the improved policy. It
is the basis of AlphaGo/AlphaZero-style systems and is very strong when a fast simulator
and a search oracle are available. It reported superhuman performance in Hex and Go. At
inference, however, the policy is a network; the search is training-only. In our setting, A* is the search oracle, but its cost and the
rules’ no-planner-at-inference constraint mean we can only use it for labels. Expert
iteration is conceptually similar to DAgger with a planner; the engineering cost is higher
than we need.

**Kickstarting.** Schmitt et al. (2018, arXiv:1803.03835) add a teacher imitation loss to
an RL agent to accelerate learning, then anneal it. It reported faster convergence on
Atari. The idea is attractive because we already have a BC checkpoint: keep a BC auxiliary
loss during PPO, decayed over time. The risk is teacher bias: if the teacher is wrong on
doors, the BC loss fights the RL signal. Use a small weight and only on states where the
teacher is reliable.

**Asymmetric actor-critic.** Pinto et al. (2017, arXiv:1710.06542) give the critic
privileged state (true pose, object poses, contacts) during training while the actor sees
only observations. The critic learns a low-variance value function; the actor is deployable
without privileged inputs. It reported improved sample efficiency on image-based robot
manipulation. This is a perfect fit for us: the training-time `privileged_state()` gives
true pose, keys, locks, damage and sector geometry. The critic can use them; the actor
cannot. The risk is accidental leakage (a privileged feature sneaking into the actor
input), which a unit test can prevent.

**Teacher–student distillation.** A privileged teacher (with map access) can be distilled
into a student that sees only observations. This is common in locomotion (Lee et al., 2020,
arXiv:2010.11251) and in navigation. For us, the teacher’s privileged labels are most
useful for **auxiliary heads** (exit visible, exit bearing, key colour, door ahead) rather
than for the actor’s action targets. The student learns the representation; the actor
learns from RL and the reliable subset of the teacher.

| Method | Mechanism / fit | Sample cost | Inference | Failure |
|---|---|---|---|---|
| Behaviour cloning | Supervised imitation; the only realistic warm start | 10^5 | Cheap | Compounding errors; teacher may be unobservable |
| DAgger (arXiv:1011.0686) | Relabel student-visited states with the expert | 10^5–10^6 | Cheap | Teacher quality; query cost |
| Expert iteration (arXiv:1705.08439) | Policy improvement with search | 10^7 | Search | No search at inference |
| Kickstarting / distillation (arXiv:1803.03835) | RL with a teacher loss; privileged teacher → student | 10^7 | Cheap | Teacher bias; capacity gap |
| Asymmetric actor-critic (arXiv:1710.06542) | Privileged critic, observation-only actor | 10^6 | Zero extra | Leakage if actor sees privileged inputs |

BC + DAgger is the highest-value combination here. The teacher must be
observation-conditioned; privileged data belongs in auxiliary heads and the critic.

### 2.5 Long-horizon and hierarchical methods

**The problem.** A flat policy must choose between “open this door”, “find the key”,
“fight”, and “explore” at every step. Hierarchical methods factor the decision into a
manager that selects a goal or option every K steps and a worker that executes it. The
appeal is credit assignment over long horizons; the cost is non-stationarity and two-level
optimisation.

**Options.** An option is a triple `ω = (I_ω, π_ω, β_ω)`: an initiation set, an intra-option
policy, and a termination condition. The semi-MDP value decomposition is

```
Q_ω(s,a) = r + γ Σ_{s'} P(s'|s,a) [ (1−β_ω(s')) Q_Ω(s',ω) + β_ω(s') V_Ω(s') ]
```

Options let a policy reuse a skill (e.g., “approach and open a door”) across rooms. The
classical difficulty is that options must be discovered or specified; option-critic
(Bacon, Harb and Precup, 2017, arXiv:1609.05140) learns them end-to-end but is unstable at
small data. For us, a **door-servo option** is a well-specified, reusable skill worth
trying; a fully learned option hierarchy is not.

**Feudal RL.** FeUdal Networks (Vezhnevets et al., 2017, arXiv:1703.01161) use a manager
that outputs a goal vector every K steps and a worker that acts to maximise the dot product
between the goal and the feature change. The manager’s reward is the sum of extrinsic
rewards over its horizon:

```
g_t ~ π^M(·|s_t)
a_t ~ π^W(·|s_t, g_t)
r_t^M = Σ_{k=t}^{t+c−1} γ^{k−t} r_k
∇_M = A^M_t ∇ log π^M(g_t|s_t)
```

It reported strong results on DMLab. The two-level credit assignment is the hard part. Our
proposed manager is a lightweight version: it selects an event token or frontier every 8
steps, and the worker is a goal-conditioned PPO policy. Hierarchical PPO is the same
pattern with PPO at both levels; it is easier to implement than a full feudal stack but
still risks non-stationarity.

**Goal-conditioned RL and UVFA.** Universal Value Function Approximators (Schaul et al.,
2015, ICML) learn `V(s,g)` or `Q(s,a,g)` for a goal space `G`:

```
Q^π(s,a,g) = E_π[ Σ_{t=0}^{∞} γ^t r(s_t,a_t,g) | s_0=s, a_0=a ]
V^π(s,g) = max_a Q^π(s,a,g)
```

The goal can be a relative position, a door token, or a frontier. UVFA is attractive
because the same policy can be reused for different goals without a separate manager per
goal. The failure mode is goal sampling: if the goal distribution is wrong, the policy
learns goals it never sees at evaluation.

**Hindsight Experience Replay.** HER (Andrychowicz et al., 2017, arXiv:1707.01495) relabels
failed trajectories with the goal that was actually achieved:

```
g' = φ(s_{t+k})
r' = r(s_t, a_t, g')
```

It turns failure into useful data and is very effective in robotics with sparse rewards.
For us, HER can relabel “go to the locked door” episodes with the key that was picked up, or
“go to the exit” episodes with the room that was reached. It is cheap and compatible with
PPO if the goal is part of the observation and the value function.

**HIRO and HAC.** HIRO (Nachum et al., 2018, arXiv:1805.08296) uses a two-level policy with
off-policy corrections and hindsight goal relabelling; HAC (Levy et al., 2017,
arXiv:1712.00948) stacks multiple levels with hindsight. HIRO reported strong robotic
manipulation results. Both are strong in robotics but add engineering complexity. We do not need multi-level hierarchies; one manager and one
worker are enough.

**Director.** Hafner et al. (2022, arXiv:2206.04114) learn a manager that selects latent
goals every K steps and a worker that achieves them in a learned world model. It reported
strong results on DMLab and Atari. It is the closest published architecture to our manager,
but it depends on a world model and a large training budget. We can borrow the **option-
level PPO** idea without the world model.

| Method | Mechanism / fit | Sample cost | Inference | Failure |
|---|---|---|---|---|
| Options / option-critic (Sutton et al., 1999; arXiv:1609.05140) | Temporally abstract actions; learned options | 10^6–10^7 | Cheap | Credit assignment; option collapse |
| FeUdal (arXiv:1703.01161) | Manager sets goals, worker acts; hierarchical PPO is the same pattern | 10^7 | Cheap | Non-stationarity; two-level credit |
| HIRO/HAC (arXiv:1805.08296, 1712.00948) | Goal-conditioned hierarchy + hindsight | 10^6 | Cheap | Goal drift; complexity |
| UVFA/HER (Schaul et al., 2015; arXiv:1707.01495) | Goal-conditioned value; relabel failed goals | 10^6 | Cheap | Goal sampling; reward hacking |
| Director (arXiv:2206.04114) | Manager selects latent goals every K steps | 10^7 | Cheap | Needs a world model; complexity |

A two-level manager is the right long-term shape but a 24-hour risk. The pragmatic version
is a single policy conditioned on a goal vector, selected by a tiny scorer over event
tokens.

### 2.6 Model-based RL

**The problem.** Model-based RL learns a dynamics model and uses it for planning or
imagination. It is the most sample-efficient family in the literature, but it moves cost
from environment steps to model training and inference-time search. On a CPU-only host
with 12 hours, that trade is usually bad.

**World Models.** Ha and Schmidhuber (2018, arXiv:1803.10122) train a VAE to compress
frames, an RNN to predict the next latent, and a controller in the latent space. It
demonstrated that a policy can be trained inside a hallucinated dream and transferred back
to the real environment. The sample cost is 10^6 frames, but the model training is
significant and the controller is small. For us, the useful piece is the idea of an
auxiliary latent-prediction loss, not the full dream loop.

**DreamerV2/V3.** DreamerV2 (Hafner et al., 2020, arXiv:2010.02193) uses a recurrent
state-space model (RSSM) with discrete latents and trains the policy by imagination.
DreamerV3 (Hafner et al., 2023, arXiv:2301.04104) uses a fixed set of hyperparameters
across 150+ tasks and reported collecting diamonds in Minecraft from scratch without human
data or curricula. The loss is a sum of reconstruction, dynamics, representation, reward
and continuation terms:

```
L = L_pred + L_dyn + L_rep + L_reward + L_cont
```

Dreamer’s inference is cheap (an RSSM forward plus a small policy), but training the world
model on 120×160 RGB on a Mac CPU is slow. The failure modes are model bias, posterior
collapse, and hyperparameter sensitivity. Not a 24-hour deliverable.

**MuZero.** MuZero (Schrittwieser et al., 2019, arXiv:1911.08265) learns a latent dynamics
model, a reward model and a value/policy model, and plans with MCTS in the latent space. It
matched or exceeded AlphaZero on Go, chess and shogi without being given the rules, and
reached superhuman Atari performance. The loss is

```
L = Σ_k l^r(u_{t+k}, r_t^k) + l^v(z_{t+k}, v_t^k) + l^p(π_{t+k}, p_t^k)
```

The inference cost includes a search tree; even a small tree would be risky inside 40 ms
on one CPU thread, and the training cost is far beyond the budget. The conceptual lesson
is that a learned model plus search can substitute for a lot of data; we do not have the
compute to exploit it.

**EfficientZero, TD-MPC2 and IRIS.** EfficientZero (Ye et al., 2021, arXiv:2111.00210)
reported strong Atari results with 100k frames by combining a learned model with
self-supervised consistency and MCTS. TD-MPC2 (Hansen et al., 2023, arXiv:2310.16828) uses
a latent model and model-predictive control and reported strong results across many
continuous-control tasks. IRIS (Micheli et al., 2022, arXiv:2209.00588) uses a transformer
world model and reported strong Atari 100k results. All three are sample-efficient but
GPU-heavy; none fits a CPU-only 24-hour run.

**Plan2Explore.** Sekar et al. (2020, arXiv:2005.05960) use a learned world model and
disagreement-based intrinsic reward to explore. It reported improved exploration in DMLab.
It is a principled exploration method, but it inherits the world-model cost. Not a priority.

| Method | Mechanism / fit | Sample cost | Inference | Failure |
|---|---|---|---|---|
| World Models / DreamerV2/V3 (arXiv:1803.10122, 2010.02193, 2301.04104) | VAE/RSSM + imagination; strong sample efficiency | 10^6–10^7 | Small RSSM | CPU world-model training is too slow |
| MuZero / EfficientZero (arXiv:1911.08265, 2111.00210) | Learned model + MCTS; sample-efficient Atari | 10^5–10^7 | Search | Rules/compute; GPU engineering |
| TD-MPC2 / IRIS (arXiv:2310.16828, 2209.00588) | Latent MPC; transformer world model | 10^5–10^6 | Search/cheap | Continuous control or GPU/data |
| Plan2Explore (arXiv:2005.05960) | Model-based curiosity | 10^6 | Cheap | Complexity; model bias |

DreamerV3 is the strongest sample-efficiency story, but it assumes a GPU and hours of
world-model updates. Steal the auxiliary latent-prediction idea, not the imagination loop.

### 2.7 Benchmarks and evaluation

**ViZDoom competition protocol.** The competitions used fixed scenarios, a fixed frame
skip, a public leaderboard, and a hidden test set of maps. Agents were ranked by average
score across maps. The protocol is a good example of **fixed seeds and no cherry-picking**;
its weakness for us is that the maps are short and combat-centric, so it does not test
key/lock progression or long-horizon backtracking.

**ProcGen.** ProcGen (Cobbe et al., 2019, arXiv:1912.01588) has 16 games with procedural
levels and a train/test split over level seeds. It exposed a large generalisation gap:
agents that overfit the training levels do much worse on test levels. The lesson is to
treat the level generator as part of the data distribution and to hold out generator
configurations, not just seeds. Our C13 OOD split and texture holdout are the equivalent.

**MiniHack.** MiniHack (Samvelyan et al., 2021, arXiv:2109.13202) is a sandbox of small,
composable tasks built on NetHack. It is designed as a skill ladder: each task isolates a
capability (navigation, doors, keys, combat, inventory), and the agent can be tested on
combinations. The suite is designed to expose composition failures; our own drill-to-room-
door gap is the same phenomenon. The lesson is to make each category diagnostic and to
report per-category results, not just an aggregate.

**NetHack and Crafter.** The NetHack Learning Environment (Küttler et al., 2020,
arXiv:2006.13760) is a long-horizon, sparse-reward dungeon crawl; Crafter (Hafner, 2021,
arXiv:2109.06780) is a 2D open-world survival benchmark with 22 achievements. NetHack
reports score and game progress; Crafter reports 22 achievement metrics. Both argue for
**non-binary progress reporting** rather than a single win rate, which is exactly the right
style for our scorer.

**DMLab and Habitat.** DMLab (Beattie et al., 2016, arXiv:1612.03801) is a suite of 3D
partially observable tasks; DMLab-30 was used by IMPALA. Habitat (Savva et al., 2019,
arXiv:1904.01201) is a photorealistic embodied navigation simulator; the ObjectNav protocol
(Anderson et al., 2018, arXiv:1807.06757) defines success rate and SPL. These benchmarks
provide task shapes and metrics, not algorithms. SPL is useful for us: it distinguishes an
agent that exits quickly from one that wanders.

**Generalisation and OOD splits.** The standard recipe is to train on one distribution of
levels and test on another: unseen seeds, unseen sizes, unseen textures, unseen monster
mixes. ProcGen popularised this; ProcTHOR scaled it to houses. Our splits must be
**generator-configuration holdouts**, not just seed holdouts, because a seed holdout with
the same generator still shares the layout grammar. We need both: seed holdout (easy) and
config holdout (hard).

**Statistical reporting.** Agarwal et al. (2021, arXiv:2108.13264) argue that single-run
point estimates and mean±std over a few seeds are misleading. They recommend stratified
bootstrap confidence intervals, the interquartile mean (IQM), and performance profiles.
Henderson et al. (2017, arXiv:1709.06560) show that RL results are highly sensitive to
seeds, hyperparameters and implementation details; “Deep RL that Matters” is a warning, not
a recipe. “Show Your Work” (Agarwal et al., 2019, arXiv:1909.03004) argues for reporting
the full distribution of results and the compute budget. Our benchmark should follow these
practices: 5 engine seeds, 30 maps per category, stratified bootstrap over maps, IQM, and a
test-touch log.

| Benchmark / paper | What it gives us | Fit |
|---|---|---|
| ViZDoom competitions (arXiv:1809.03470) | Fixed scenarios, protocol, baselines | Medium: maps here are hidden and longer |
| ProcGen / MiniHack (arXiv:1912.01588, 2109.13202) | Procedural train/test splits; skill ladder | High: split discipline and categories |
| NetHack / Crafter (arXiv:2006.13760, 2109.06780) | Long-horizon sparse reward; achievement metrics | High: non-binary reporting |
| DMLab / Habitat/ObjectNav (arXiv:1612.03801, 1904.01201, 1807.06757) | 3D POMDP navigation; SPL, success rate | High: task shape and path efficiency |
| Statistical precipice / Deep RL that Matters (arXiv:2108.13264, 1709.06560) | CIs, IQM, performance profiles, reporting | High: mandatory for honest claims |

Borrow ProcGen’s splits, MiniHack’s ladder, Crafter’s reporting and the statistical-
precipice toolkit; do not borrow the compute assumptions.

### 2.8 Synthesis: what transfers

| Transfers to this track | Does not transfer |
|---|---|
| Small recurrent multimodal encoder | 10^8-step pixel PPO from scratch |
| BC + DAgger + privileged labels | A global A* teacher as an imitation target |
| Explicit map + learned readout | Planner at inference |
| Potential-based shaping + sector novelty | Raw distance penalties |
| Asymmetric critic | Privileged actor inputs |
| Event tokens / goal vector | Full feudal hierarchy in 24 hours |
| PLR / balanced curriculum | Population-based training |
| Exit/colour auxiliary heads | End-to-end RGB-only RL at this budget |
| 5-seed evaluation with CIs | Single-map or two-map evals |
| Option-level PPO for a door servo | Dreamer/MuZero on a CPU-only host |

---

## 3. What the standard recipes get wrong for this track

This section is deliberately blunt. Each subsection pairs a standard assumption with the
measurement that falsified it and the fix that follows.

### 3.1 A global A* teacher is not a function of the observation

The standard imitation recipe assumes a teacher mapping `a_t = π*(o_t)`. Our teacher
computes a turn action from a **global A\* waypoint**:

```
a_t^turn = sign( wrap180( atan2(y_goal − y_t, x_goal − x_t) − θ_t ) )
```

The student sees `o_t`, which contains rays, entities, vitals and RGB, but not `(x_t, y_t)`,
`θ_t`, the global map, or the goal. Many different global states map to the same
observation, and the optimal turn differs. The mapping is not a function; BC minimises an
irreducible error.

Measurement: 100,921 transitions, BC loss 6.21 → 1.89, move 85.6%, strafe 76.0%, attack
92.4%, turn **43.2%** (chance 33.3%), exact six-head match 20.8%, held-out exit 0/4. More
epochs did not help; CE plateaued at ~1.9. The policy learned “forward + turn-left” and
circled the start room.

Fix: make the goal observable. A map with a visited/frontier channel plus a
frontier-conditioned goal turns the teacher’s action into a function of the student’s
input. Alternatively, use local-reflex teachers (door servo, combat) whose actions depend
only on nearby observations. The map-aware BC was the first response; it improved tier 1
but not tiers 2–4 because the map was coarse and the pose drifted.

### 3.2 No exploration mechanism means tiers 2–4 stall

The standard PPO recipe assumes that a dense reward or a random exploration bonus will
eventually find the exit. In a 30×30-room map with doors, the probability that random
actions reach the exit before the tick cap is tiny. The policy needs a gradient toward
unvisited areas.

Measurement: `bc_arena` reached 100% coverage on tier-1 arenas but **6.1%** on tier-2
rooms, 4.2% on tier-3, 2.0% on tier-4, and 3.6% on Freedoom. PPO-25 improved the proxy but
Freedoom coverage *fell* from 3.6% to **1.6%**: it learned to fight, not explore. The
new-cell reward (+0.01) is too weak and delayed to overcome the BC prior’s circling. There
is no frontier term, no episodic novelty bonus, and no goal-conditioned “go to the
unknown” signal.

Fix: add a frontier channel and a frontier-conditioned goal; add an episodic sector bonus
and a potential-based frontier potential. Memory alone does not create exploration; the
objective must reward it.

### 3.3 Door USE is rare, and straight drills do not transfer to room doors

The standard curriculum assumes that a skill learned in isolation transfers to the full
task. Door USE is a counterexample.

Measurement: 360 reliable drill episodes (1 door, 1 door + key + lock, 2 doors) taught door
USE in two epochs: **6/8** door drills. The same checkpoint opened **0/8** room doors on
tier-2 maps. A room-door generator with random start position and heading produced
**200/200** teacher successes; the student trained on it scored 4/8 on drills and **0/8**
on room doors. Freezing the trunk and training only the actor scored 0/8 drills: the
features, not just the head, must adapt.

Why: a straight drill is a 1-D alignment problem with a fixed approach angle. A room door
is a 2-D approach from an arbitrary angle, with a 64-unit opening, a 16-unit player radius,
door-frame collisions, and a USE edge that reverses a moving door. The policy needs a
closed-loop servo (turn, approach, press once, wait, cross), not a single
observation-to-action mapping learned from one approach angle. The door also has a
**button-down edge**: holding USE across steps presses once, and pressing again while the
door is moving reverses it.

Fix: collect room-door demonstrations from many angles and positions; train a door-servo
option or a goal-conditioned policy with a door token; evaluate with a door-angle probe
(8 headings × 8 offsets), not a straight drill.

### 3.4 Colour and exit perception require RGB and privileged auxiliary labels

The standard “use the structured observations” recipe fails because the structured
observations omit the two most important semantics: the exit and the colour of keys/locks.

Measurement: exit is absent from `RAY_HIT_TYPES` and `ENTITY_TYPES`; key/lock colour is
hidden until pickup. The model has RGB and 35 auxiliary outputs, but the exit head only
shapes the encoder; the actor has no explicit exit goal. Exit-visible positives are 0.5–2%
of steps; `pos_weight=12` helps the loss but the actor still got 0/10 exits on Freedoom
MAP23–32.

Fix: keep RGB, but make the exit head **spatial**: write its probability into the map at
the ray endpoint (exit-likelihood channel). Add a key-colour head and store colour in event
tokens. Privileged labels are training-only; at inference the heads are just network
outputs. The RGB encoder can stay small (30×40 pooled, 3.6 KB/frame); the point is colour
and high-contrast signs, not photorealistic detail.

### 3.5 Two-map evals were misleading; the eight-map bench changed decisions

The standard “run a few episodes” eval is not enough. Measurement: the first map-aware BC
looked like 2/2 tier-1 exits; the fixed 8-map bench showed **2/8**. After arena fine-tuning,
the same bench showed **5/8**. The two-map eval would have promoted a weaker checkpoint and
hidden the tier-2/3/4 failure.

Fix: never select on fewer than 8 maps per tier and 1 engine seed; for final claims, 5
seeds × 30 maps per category with confidence intervals. Report every map, not the best.

### 3.6 Odometry drift breaks the map

The standard “integrate velocity” pose estimate is not accurate enough for a 64-unit door.
Measurement: the naive odometry used a single 14.06°/decision turn constant and ignored the
ramp. After the measured turn table and velocity timing, heading drift over 250 steps fell
from **63° to 16°**, and position error from **907 to 361 units**. At 96 units/cell, 361
units is 3.8 cells; a 64-unit door is smaller than one cell. The map smears, the frontier
channel is noisy, and door tokens point to the wrong place.

Fix: exact heading from the turn table; translation-only scan matching against the wall
channel (±8 cells, 17×17 candidates, vectorised, <1 ms). Accept only a sharply peaked
match; otherwise fall back and set a drift flag. Re-measure after every map-generator
change. The turn table itself is cheap and should be a unit-tested lookup, not a constant.

### 3.7 Catastrophic forgetting: KL anchors and mixtures failed

The standard “fine-tune a specialist, then add a KL anchor” recipe assumes that the anchor
covers the new policy’s state distribution. Measurement says otherwise.

- Drill-only fine-tuning learned doors (6/8) but dropped tier-1 exits from 2/2 to 0/2 and
  the tier-1 proxy from +11.4 to −1.2.
- A KL-anchored merge (`bc_combined`: drills + room doors, anchor to `bc_arena` on non-door
  frames) measured **tier-1 1/8, drills 0/8**.
- Success-only mixed replay (`bc_success`: 652 successful episodes, LR 5e-5) measured
  **tier-1 4/8 (+4.95), tier-2 −0.54, drills 0/8** — slightly worse than the arena
  specialist.
- The MoE gate routed doors (6/8 drills at threshold 0.1) but cost one tier-1 exit and
  lowered tier-2/3/4 proxies (t3 −0.22, t4 −0.93); at threshold 0.5 it collapsed to PPO
  (0/8 drills).

Why: KL on a fixed “non-door” set does not cover the new policy’s states; the trunk
changes and the actor drifts. Sequential fine-tuning on a small imbalanced pool is the
classic forgetting setup. A gate over frozen specialists cannot fix a specialist that is
worse on the target distribution.

Fix: stop sequential specialist fine-tuning. Train **one multi-task policy on a
skill-balanced pool** with 30% rehearsal in every batch; if a fine-tune is unavoidable,
distil from the old policy on *student-visited states*, not on a fixed dataset. A gate is a
last resort, not a substitute for a shared representation.

### 3.8 Hype versus likely wins

| Hype for this track | Why it fails here | Likely win |
|---|---|---|
| From-scratch PPO at 10^8 steps | 25.6k steps moved tier-1 5/8→6/8 and left tier-2/3 near zero | BC + DAgger + small PPO |
| Distributed RL (IMPALA/SEED/Sample Factory) | Needs many cores/GPUs; one CPU thread at inference | 8 sequential envs; profile first |
| Dreamer/MuZero/world models | World-model training and search cost; CPU-only host | Auxiliary latent prediction |
| Transformers / Decision Transformer | Data-hungry; quadratic cost; no advantage at 10^5–10^6 | GRU + token bank |
| MoE / learned router | Measured brittle; frozen specialists worse on rooms | Single balanced policy + goal vector |
| Population-based training | Compute multiplier | PLR / curriculum |
| End-to-end RGB-only RL | Exit/key colour are rare and sparse | RGB + privileged auxiliary heads |

**Blunt conclusion.** With 12 hours, do not train a world model, add a transformer, run
three seeds, or fine-tune another specialist. Fix the observation function (map + scan
match + exit/colour channels), fix the data distribution (balanced multi-task BC +
DAgger), and add an asymmetric critic. That is the highest expected value per hour.

---

## 4. Architecture analysis

### 4.1 Comparison table

Ratings are 1–5; for “risk”, 5 is highest. “Expected gain” is relative to the shipped
PPO-25 agent on the hidden-tier proxy, based on our measurements and the literature. It is
a judgment, not a guarantee.

| # | Candidate | Sample efficiency | CPU inference | Engineering risk | Expected gain | 24 h feasibility | Recommendation |
|---|---|---:|---:|---:|---|---|---|
| 1 | Recurrent PPO from BC (current) | 3 | 5 | 2 | Baseline (+8.09 weighted proxy) | Done | Keep as baseline |
| 2 | **B+ lite v2: map + scan match + tokens + goal (single policy)** | 4 | 5 | 3 | High: +2–4 tier-2/3 exits if pose fixed | High | **Core** |
| 3 | Goal-conditioned hierarchical PPO (manager over tokens/frontiers) | 3 | 4 | 5 | High: +2–4 tier-2/3 exits if it trains | Medium | Stretch after #2 |
| 4 | Explicit SLAM/occupancy + frontier policy | 3 | 4 | 3 | Medium: +1–3 exits | Medium | Map + scan match, not planner |
| 5 | Asymmetric actor-critic with privileged critic | 4 | 5 | 2 | Medium: better credit assignment, free at inference | High | **Do it** |
| 6 | Model-based Dreamer-style | 5 | 3 | 5 | Uncertain; likely negative in 24 h | Low | No |
| 7 | Transformer memory / Decision Transformer | 2 | 3 | 4 | Low at our data scale | Low | No |
| 8 | Mixture of experts / learned router | 2 | 4 | 4 | Measured negative on tiers 2–4 | Low | No |
| 9 | Population-based / curriculum / PLR | 3 | 5 | 2 | Medium: sample efficiency | Medium | PLR only |
| 10 | Imitation + DAgger + PPO hybrid | 5 | 5 | 3 | High: fixes covariate shift and forgetting | High | **Do it** |
| 11 | Options / feudal | 2 | 4 | 5 | Medium: door skill if it works | Low | Door-servo option only |

### 4.2 Candidate notes

**#1 Recurrent PPO from BC.** The current model is a strong baseline: 1.807 M parameters,
0.45–1.9 ms per decision, multimodal encoder, GRU(384), factorised heads. Its failures are
not encoder capacity; they are data distribution (teacher unobservable), pose drift and
exploration. Rewriting the encoder first would waste the remaining time.

**#2 B+ lite v2.** The pragmatic core: keep the encoder, add a better map (8 channels,
96×96 at 64 u/cell), translation-only scan matching, local/global crops, ≤32 event tokens,
and a goal vector. The map makes the frontier teacher imitable; the tokens make the
key→door return explicit. Inference stays ~3–6 ms. Engineering risk is moderate: the map
writer, scan matcher and token bank must be unit-tested, and the training data must be
regenerated with the new inputs. Expected gain is high because it directly addresses the
measured turn and memory failures.

**#3 Goal-conditioned hierarchical PPO.** A manager selects a token or frontier every 8
steps; a worker is a goal-conditioned PPO policy. This is the most intellectually
satisfying option and the most likely to produce a strong demo (attention over remembered
doors). It is also the highest risk: two-level credit assignment on 10^5–10^6 samples is
fragile, and the manager can collapse to “continue” or oscillate between tokens. Build it
only after #2 clears the tier-2 gate; freeze the manager to a fixed selector for the first
PPO run.

**#4 Explicit SLAM/occupancy + frontier policy.** The map and scan match are worth it; a
classical planner at inference is not allowed. A learned frontier policy is a compromise,
but it is still a policy; a deterministic “nearest frontier” selector is a heuristic that
should be documented and, ideally, replaced by a learned scorer. The risk is rule ambiguity
and engineering time; the gain is better coverage.

**#5 Asymmetric actor-critic.** The cheapest real win: the critic sees true pose, keys,
locks, geodesic distance and monster counts during training; the actor does not. The
critic is discarded at inference, so the cost is zero. The risk is privileged leakage; a
unit test that asserts the actor’s input keys are observation-only prevents it. Expected
gain is better value estimates and sample efficiency, especially on sparse exit reward.

**#6 Model-based Dreamer-style.** DreamerV3 is the strongest sample-efficiency story in the
literature, but it assumes a GPU and hours of world-model updates. On a Mac CPU, training
an RSSM on 120×160 RGB and then imagining rollouts is a research project. The expected gain
in 24 hours is negative relative to BC/DAgger. Do not.

**#7 Transformer memory / Decision Transformer.** At 10^5–10^6 decisions, a GRU plus a
token bank is a better cost/benefit. Transformers are data-hungry, quadratic in sequence
length, and need careful stabilization (GTrXL). Decision Transformer is an offline RL
method that needs a large, high-quality trajectory dataset; we have ~100k transitions and a
weak teacher. Not a fit.

**#8 Mixture of experts / learned router.** Measured: the gate routes doors but the door
expert is worse on rooms; hard thresholds collapse. A gate over frozen specialists cannot
fix a specialist that is worse on the target distribution. The engineering cost is low, but
the expected gain is negative. Do not spend more time here.

**#9 Population-based / curriculum / PLR.** Population-based training is a compute
multiplier; we do not have the compute. Curriculum is already in the plan; PLR is a cheap
scheduler that prioritises levels by learning potential. Add PLR after the baseline is
stable; do not make it a dependency.

**#10 Imitation + DAgger + PPO hybrid.** The single best fix for the BC covariate shift.
The teacher is already in the repo; DAgger labels student-visited states; PPO fine-tunes
with a BC auxiliary loss and rehearsal. The risk is teacher quality on doors; use a mixture
of teachers (frontier, door, A*) and gate by state. Expected gain is high.

**#11 Options / feudal.** Only the door-servo option is worth it: a reusable closed-loop
skill for approach/press/wait/cross. A full feudal stack is too much for 24 hours. The
option can be trained with BC from room-door demos, then option-level PPO.

### 4.3 Recommendation summary

The core is **#2 + #5 + #10**: B+ lite v2, asymmetric critic, BC/DAgger/PPO. The stretch is
**#3**: a learned manager over tokens, built only after the core clears the tier-2 gate.
The door-servo option (#11) is a focused sub-skill worth adding if time permits. The
non-recommendations are #6–#8 (model-based, transformer, MoE) and #9 beyond PLR.

---

## 5. Recommended architecture for the remaining time

### 5.1 Rationale

The architecture is designed around the measured failure modes, not around novelty:

- The 43.2% turn accuracy says the policy needs an **observable goal**. The map, pose and
  goal vector provide it.
- The 6.1% coverage says the policy needs a **frontier signal**. The map has a frontier
  channel and the goal vector can point at a frontier.
- The 0/8 room doors says the policy needs a **closed-loop door skill** and door tokens,
  not more straight drills.
- The 0/10 Freedoom exits says the policy needs **exit perception** and a spatial exit
  memory, not just a per-frame classifier.
- The catastrophic forgetting says the policy needs **one balanced training run**, not a
  chain of specialists.
- The 40 ms budget says the model can afford a map and tokens, but not a transformer or a
  world model.

### 5.2 Module diagram

```
 rays 64 ─────────► RayEncoder (circular Conv1d) ───────────► 128 ┐
 entities ≤32 ────► EntityEncoder (type emb + MHA + pool) ──► 128 ┤
 vitals 17 ───────► VitalsEncoder (MLP) ────────────────────►  64 ┤
 RGB 30×40×3 ─────► RGBEncoder (3-layer CNN) ───────────────► 128 ┤
 prev action 12 ──► PrevEncoder (MLP) ──────────────────────►  32 ┤
 local map 8×32×32 ► MapEncoder (CNN) ──────────────────────► 128 ┤
 global map 8×32×32► MapEncoder (shared/2nd CNN) ───────────►  96 ┤
 pose 8 ──────────► PoseEncoder (MLP) ──────────────────────►  32 ┤
 goal 8 ──────────► GoalEncoder (MLP) ──────────────────────►  32 ┤
                                                                  ├─ concat 768
                                                                  ▼
                                                         Linear 768 → 384
                                                                  │
                                                              GRU(384)
                                                                  │
       ┌──────────────┬──────────────────┬───────────────┬────────┴──────┐
       ▼              ▼                  ▼               ▼               ▼
   actor 15       critic 1           aux 45         token query      manager
   logits         (train only)       heads          (attention)      (every 8)
```

The manager is optional initially. A fixed “nearest frontier” selector provides the goal
vector; later, a 2-layer MLP over token embeddings, GRU state, global map summary and
vitals outputs a categorical over ≤32 tokens plus “continue.”

### 5.3 Tensor shapes and parameter estimate

| Stream | Input shape | Encoder | Output |
|---|---|---|---|
| Rays / entities | 64×9 / 32×(5+type) | circular Conv1d / type emb + MHA + pool | 128 / 128 |
| Vitals / prev / RGB | 17 / 12 / 3×30×40 | MLP / MLP / 3-layer CNN stride 2 | 64 / 32 / 128 |
| Maps / pose / goal | 8×32×32 ×2 / 8 / 8 | two small CNNs / MLP / MLP | 128+96 / 32 / 32 |
| Fusion / recurrence | 768 → 384 | Linear, GRU(384) | 384 |
| Heads / tokens | 384 / 32×20 | actor 15, critic 1, aux 45 / MLP+MHA | 15/1/45, 128 |

Parameter estimate: rays ~0.1 M, entities ~0.1 M, RGB ~0.3 M, maps ~0.2 M, fusion ~0.3 M,
GRU ~0.9 M, heads ~0.1 M; total ~2.0–2.6 M. At the current model’s 0.45–1.9 ms for 1.8 M,
the proposed model should be ~0.7–3 ms; plus map update and scan match ~1 ms; target p99
<6 ms. That leaves >6× margin inside the 40 ms budget. The model is small enough for 2 GB
VRAM and 8 GB RAM; weights are ~10 MB.

### 5.4 Map and pose

**Grid.** 96×96 cells at 64 units/cell (6,144-unit span), 8 `uint8` channels: visited/free,
wall, door, locked door, item/key, frontier, exit likelihood, enemy last-seen (decay
0.95/step). Memory is 73,728 bytes. The grid is allocentric, with the origin at spawn; the
pose estimate maps ego-centric observations into it.

**Write.** For each ray, compute the endpoint in world coordinates using the estimated
pose and heading, and rasterise the line from the agent to the endpoint. Mark free cells
along the line; mark the endpoint according to hit type (wall, door, locked door, item).
For key/item entities, project their relative position into the map. For the exit head,
write `p_exit` at the endpoint of a ray whose hit type is wall and whose RGB patch is
consistent with an exit cue; decay slowly. The frontier channel is computed as free cells
adjacent to unknown cells. The enemy channel decays by 0.95 per step so old sightings fade.

**Read.** Two heading-aligned crops: a local 32×32 crop at 64 units/cell (2,048-unit span)
and a global 32×32 crop at 128 units/cell (4,096-unit span). Rotate by nearest-neighbour;
this is deterministic and <0.3 ms. The local crop is for door alignment and immediate
geometry; the global crop is for room layout and backtracking.

**Pose.** Heading is integrated from the measured turn table keyed by `turn_held`, the
episode warm-up state and the previous action. Translation is integrated from ego velocity
(`vitals[14:16]`), then corrected by a **translation-only scan match**. The score is

```
score(T) = Σ_{j=1}^{64} w_j · 1[ M(T ⊕ p_j) = wall ]
```

where `p_j` is the endpoint of ray `j` in the predicted pose, `M` is the wall channel, and
`T` ranges over ±8 cells (17×17 = 289 candidates at 64 units/cell). The search is
vectorised NumPy, <1 ms. Accept the correction if the best score beats the predicted-pose
score by ≥2 endpoints and the best/second-best ratio is ≥1.2; otherwise fall back to
velocity integration and set a drift flag. Optionally refine heading over ±2° in 0.5°
steps. This is perception, not planning: it uses the map and rays, not engine state.

**Pose vector (8).** `x/2048`, `y/2048`, `sin h`, `cos h`, `vf/16`, `vr/16`,
`scan_confidence`, `drift_flag`. The network can learn to trust or discount the map based
on confidence.

**Why 64 units/cell.** A 64-unit door is exactly one cell; the local crop at 64 u/cell
represents the door opening at the resolution needed for alignment. The current 96 u/cell
map is too coarse: a door is smaller than a cell. The global crop at 128 u/cell keeps the
span manageable. The two-resolution design is a compromise between precision and memory.

### 5.5 Event tokens and the goal vector

**Token bank.** Up to 32 tokens, one per locked door, key, exit cue, frontier cluster,
health/ammo cache, or unexplored branch. Merge tokens within 64 units; evict resolved
tokens first, then oldest. The bank is the “unfinished business” memory: a locked door
seen 700 steps ago still points the right way.

**Features (20).** Relative `x/512`, `y/512`, `dist/1024`, `sin`, `cos`, type one-hot (6),
age/256, visits/8, key-colour one-hot (3), resolved flag, confidence, sector-novelty flag,
door-open flag. Recompute the relative fields from the current pose every step. The colour
field is predicted by the RGB colour head; it is not privileged at inference.

**Encoder.** MLP 20→64, two transformer encoder layers (4 heads, 64 dim), masked mean plus
an attention query from the GRU state:

```
q = W_q h_t
k_i = W_k e_i,  v_i = W_v e_i
α_i = softmax_i( q^T k_i / sqrt(d) )
z_tok = Σ_i α_i v_i
```

The attention weights are the live-demo visual: when a key is picked up, the weight should
spike on the matching door token.

**Manager.** Every K=8 decisions, score tokens with

```
u_t = MLP([h_t, z_tok, z_map, v_t])
π^M(·|u_t) = Categorical(33)
```

where the 33 classes are 32 tokens plus “continue.” Sample during training, argmax at
inference. The selected token produces a goal vector (8): `dx/512`, `dy/512`, `dist/1024`,
`sin`, `cos`, `type/6`, `colour/3`, `resolved`. The worker sees this vector every step. The
manager is trained with PPO at option boundaries; reward +0.5 on token resolution or within
96 units, +0.1 for a new sector during the option, −0.05 on timeout (64 steps), entropy
0.01.

**Why a goal vector rather than a separate worker per option.** A single goal-conditioned
worker shares data across options and avoids the non-stationarity of many independent
policies. The goal vector is a compact interface; the worker learns to interpret it. This
is the UVFA idea (Schaul et al., 2015) with a discrete token vocabulary.

### 5.6 Asymmetric critic

The critic input is the 384-d GRU state plus 32 privileged features: true `x/y`, `sin/cos`
heading, health, armour, four ammo pools, six key flags, locks remaining (3), geodesic
distance to the current objective / 1024, sector novelty flag, monster count within 1,024
units / 8, time fraction, damage in the last 8 steps, door states (3), exit-visible flag.
The actor never sees these. The critic is an MLP 416→256→1 trained with the PPO value loss;
it is not instantiated at inference. This is legal training-only privileged state and costs
nothing at evaluation. The risk is accidental leakage; a unit test asserts that the actor’s
input dictionary contains only observation keys.

The geodesic distance is computed by a training-only BFS on the navigation grid. The
objective is the nearest unheld key if a lock blocks the exit, else the exit. The potential
is `Φ = −d/1000`, and the shaping reward is `γΦ(s') − Φ(s)`, annealed to 0. This is
potential-based (Ng et al., 1999), so it preserves the optimal policy.

### 5.7 Loss terms

PPO with GAE:

```
δ_t = r_t + γ V(o_{t+1}) − V(o_t)
Â_t = Σ_{l=0}^{T−t} (γλ)^l δ_{t+l}
L^CLIP = E_t[ min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t) ]
L = L^CLIP − c_v L^VF + c_e E[H(π)] + λ_BC L_BC + λ_KL L_KL + λ_aux L_aux + λ_g L_goal
```

The BC loss is cross-entropy on the six factorised heads, with per-head weights (turn and
USE higher). The KL term is to the BC prior on the actor, decayed to 0 over the first 50
updates. The auxiliary loss is a sum of BCE/CE/MSE heads; the goal loss is BCE on token
resolution and MSE on distance. The manager has its own clipped PPO objective at option
boundaries plus a goal-completion bonus and entropy.

| Loss | Weight | Schedule | Purpose |
|---|---:|---|---|
| PPO clipped surrogate (factorised heads) | 1.0 | constant | Policy improvement |
| Value MSE (asymmetric critic) | 0.5 | constant | Credit assignment |
| Entropy | 0.01→0.003 | linear decay | Exploration |
| BC cross-entropy | 0.2→0.05 | decay over PPO | Preserve skills |
| KL to BC prior | 0.05→0 | first 50 updates | Trust region |
| Aux heads (exit, colour, door, damage, frontier, scan, goal) | 0.1; exit/colour 0.2 | constant | Shape encoder |
| Manager PPO + goal + entropy | 1.0 + 0.1 + 0.01 | after unfreeze | Option selection |

Aux layout (45): keys 6, enemy 1, door 1, exit visible 1, exit bearing 13, exit distance
6, key colours 6, damage next 8, frontier bearing 8, scan confidence 1, goal completion 1.
The first 35 match the current model; the last 10 are new (frontier 8, scan 1, goal 1).

### 5.8 Training phases and gates

| Phase | Time | Data / objective | Gate to advance |
|---|---|---:|---|
| 0. Pose and map | 2 h | Scan match, map channels, unit tests; no learning | Drift <0.5 cell/250; match >70%; p99 <2 ms |
| 1. Balanced BC | 3 h | 150–250k: nav 30%, frontier 20%, doors 20%, arena 20%, keys 10%; 3× USE/key/exit | Turn >65%; drills ≥6/8; T1 ≥5/8; T2 coverage ≥15% |
| 2. DAgger | 2 h | 30–50k student states; frontier/A*/door teachers; 50/50 mix | T2 ≥2/8; T3 keys ≥1/8; no T1 regression |
| 3. PPO | 4 h | 8 envs, 128-step rollouts, 4 epochs; 30% rehearsal; asymmetric critic; manager frozen then 1 h unfrozen | T2 ≥3/8; T3 ≥1/8; weighted proxy > +8.09 |
| 4. Evaluate and freeze | 2 h | Full suite, ablations, brief | Ship best held-out checkpoint |

The BC phase is the most important. The data mix must be **skill-balanced**, not
episode-balanced: a door drill is short and a navigation episode is long, so naive sampling
over-represents navigation. The 3× oversampling of USE/key/exit frames compensates for the
rarity of those events. The BC loss weights turn and USE higher because they are the
measured weak heads.

### 5.9 Curriculum

| Stage | Maps | Modifications | Advance when |
|---|---|---|---|
| 1. Navigation/frontier | Door-free, 4–10 rooms | No monsters; visible exit | ≥90% exits in <300 steps; coverage >25% |
| 2. Doors | Corridors + room doors, random poses | No monsters | ≥70% door-angle probe |
| 3. Key/lock | 1 key/1 lock, then 2 keys | Sparse monsters | ≥60% key→door→exit |
| 4. Combined | Full generator + Freedoom train | Vanilla rules | Held-out within 10% of train |
| 5. Generalisation | OOD seeds + Freedoom MAP23–32 | Vanilla rules | Score plateaus |

Keep 30% of every batch from earlier stages. Add PLR once the pool is large; do not make it
a dependency for the first run. The curriculum is a schedule, not a gate: the agent can
see later stages if the earlier ones are mastered.

### 5.10 Exact improvements over the current agent

| Component | Current | Proposed | Why |
|---|---|---|---|
| Map | 5×32×32 at 96 u/cell, no scan match | 8×96×96 at 64 u/cell, local/global crops, scan match | Door resolution, pose accuracy, frontier/exit channels |
| Pose | 6 values, velocity integration | 8 values, turn table + scan match + confidence | 361-unit drift is too large |
| Tokens | None | ≤32 tokens, colour, age, resolved | Key→door return over hundreds of steps |
| Goal | None | 8-d goal vector from token/frontier | Makes the teacher’s goal observable |
| Critic | Symmetric | Asymmetric, 32 privileged features | Lower value variance, free at inference |
| Aux heads | 35 | 45 (frontier 8, scan 1, goal 1) | Shape the encoder for exploration and pose |
| Data | Sequential specialists | Balanced multi-task + DAgger + 30% rehearsal | Stop catastrophic forgetting |
| Inference | 0.45–1.9 ms | ~3–6 ms | Still 7–13× inside 40 ms |

Expected effect on the measured bench: tier-1 stays 5–6/8; tier-2 exits 0/8 → 3/8;
tier-3 exits 0/8 → 1–2/8; tier-2 coverage 6.1% → 20–25%; Freedoom coverage 3.6% → 5–8%.
These are targets, not guarantees; the gates in §5.8 are the decision points.

### 5.11 What to cut if time is short

| Time left | Cut | Keep |
|---|---|---|
| <2 h | Everything except final eval | Ship PPO-25; run the suite; write the brief |
| 2–4 h | Manager, tokens, PLR, depth | Scan match + balanced BC + asymmetric critic |
| 4–6 h | Manager, learned write, PLR | Map + scan match + tokens + BC/DAgger |
| 6–10 h | PLR, multi-seed, tier-4 specialisation | Full B+ lite v2 + DAgger + PPO + asym critic |
| 10–12 h | Only ablations | Full recommendation |

**Never cut:** the held-out split, the full bench, the validator, deterministic inference,
and the fallback checkpoint. The fallback is PPO-25 or the best BC checkpoint; it must
remain shippable at every hour.

### 5.12 Implementation risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Scan match drifts or locks onto the wrong wall | Medium | High | Confidence threshold, sharp-peak test, fallback to velocity |
| Map writer bug corrupts all training data | Medium | High | Unit tests on 5 maps; compare projected walls to privileged geometry |
| Token attention collapses | Medium | Medium | Fixed selector first; entropy/attention probes; token merging |
| Manager oscillates between tokens | Medium | High | Freeze manager for first PPO run; option timeout; reward shaping |
| BC/DAgger data imbalance | High | High | Skill-balanced sampler; 3× rare events; per-head accuracy |
| PPO forgets BC skills | High | High | 30% rehearsal; BC auxiliary loss; KL early; rollback gate |
| Latency exceeds budget | Low | Medium | Profile p99; cut global map or token encoder if >10 ms |
| Rule ambiguity (map/frontier) | Medium | High | No planner in `submission/`; document perception-only; ask organisers |

---

## 6. Benchmark design

The benchmark is a deliverable in its own right. It must **drive** RL improvement, not
just report a number: every category should diagnose a specific failure, feed the
curriculum, and produce an ablation that a judge can understand.

### 6.1 Design goals

1. **Diagnostic.** Each category isolates a skill (combat, doors, keys, backtracking,
   exploration, resource management, ambush, long horizon) so a failure points to a fix.
2. **Procedural.** Maps are generated from a seed and a spec, not hand-authored, so the
   test distribution can be held out by configuration.
3. **Leakage-free.** Train, validation, test and OOD splits are disjoint by seed and by
   generator config; hashes prevent accidental overlap.
4. **Cheap.** A fast subset runs in minutes for gates; the full suite runs at phase
   boundaries.
5. **Honest.** 5 engine seeds, 30 maps per category, stratified bootstrap CIs, no
   cherry-picking, a test-touch log.
6. **Actionable.** Every result row includes the components (exit, keys, locks, doors,
   sectors, kills, damage, death) and the diagnostic probes.

### 6.2 Skill ladder and diagnostic categories

| ID | Category | Setup / skills | Ticks | Primary metric |
|---|---|---|---|---|
| C0 | Sanity | Empty room, exit 256 u ahead, one door; movement, exit recognition | 2100 | Exit |
| C1 | Duel | 512-u arena, 1 monster, pistol; aim, dodge, fire timing | 2100 | Win, damage |
| C2 | Arena | 3–6 monsters, cover, limited ammo; target selection, kiting | 2100 | Kills, damage |
| C3 | Corridor door | Straight corridor, 1–2 doors; USE edge, alignment | 2100 | Door, exit |
| C4 | Room door | Door on a room wall, 16 start poses; arbitrary-angle approach | 2100 | Door-angle success |
| C5 | Key–lock | 1 key, 1 lock, 1–3 rooms; colour, backtracking | 3150 | Key, lock, exit |
| C6 | Two-key chain | 2 keys, 2 locks, loops; ordering, long horizon | 4200 | Locks, exit |
| C7 | Exploration | 8–12 rooms, no keys, hidden exit; coverage, frontier choice | 4200 | Coverage, exit |
| C8 | Backtrack | Key behind start, lock ahead; return-to-door memory | 4200 | Return, exit |
| C9 | Ambush | Monsters spawn after key/door; re-orientation, threat memory | 4200 | Damage, survival |
| C10 | Resource | Scarce ammo, many monsters; fight-vs-flee, ammo | 4200 | Kills/ammo, exit |
| C11 | Gauntlet | All; 2 keys, 16 monsters; full stack | 4200 | Weighted proxy |
| C12 | Freedoom | MAP23–32; hand-made transfer | 4200 | Exit, coverage |
| C13 | OOD | Larger/deeper, unseen textures/configs; compositional generalisation | 4200 | Exit, coverage |

**C0–C2** isolate movement and combat. C0 is a smoke test: if the agent cannot walk to a
visible exit, nothing else matters. C1 and C2 separate aiming/dodging from target
selection and resource use.

**C3–C4** isolate doors. C3 is the straight drill that the current agent can solve (6/8);
C4 is the room-door transfer test that it fails (0/8). The contrast is the diagnosis.

**C5–C6** isolate key/lock progression. C5 is one key and one lock; C6 adds a second key,
a second lock, loops and a longer horizon. The two-key chain is where colour memory and
ordering matter.

**C7–C8** isolate exploration and backtracking. C7 has no keys, only rooms and a hidden
exit; C8 places the key behind the start and the lock ahead, so the agent must remember
where it has been.

**C9–C10** isolate ambush and resource management. C9 spawns monsters behind the agent
after a key or door event; C10 gives scarce ammo and many monsters, so fighting everything
is suboptimal.

**C11–C13** are integration and transfer. C11 is the full gauntlet; C12 is hand-made
Freedoom; C13 is the generator-config OOD split.

### 6.3 Procedural generator requirements

The generator must produce solvable maps with controlled difficulty. The core grammar is a
room graph:

```
rooms = sample_room_graph(n_rooms, degree, loops, dead_ends)
doors = place_doors(rooms, n_doors, width=1 tile)
keys, locks = place_key_chain(rooms, n_locks, colour_order)
start = sample_start(rooms, avoid=exit_room)
exit = choose_exit(rooms, farthest_by_graph_distance)
things = populate(rooms, monsters, ammo, health, cover)
```

| Parameter | Range / constraint | Why |
|---|---|---|
| Room graph | 4–20 rooms; degree 2–4 | Corridors to open maps |
| Doors | 0–8; 1 tile (64 u); height ≥56 u | Door physics; passable when open |
| Keys/locks | 0–3; key reachable without crossing its lock; lock on exit path | Solvability and ordering |
| Decoy keys | 0–1 wrong-colour key | Tests colour, not hoarding |
| Loops/dead ends | 0–4 loops; 0–3 dead ends | Loop closure, recovery |
| Cover | 1–2 tile pillars/crates | Combat and line-of-sight |
| Monsters | Zombie, shotgun guy, imp, demon, cacodemon; 0–16 | Tier difficulty |
| Ammo | Bullets 50–200, shells 0–50; pistol always | Resource management |
| Start | Random room/position; heading 0/90/180/270 ±15° | Prevent orientation overfit |
| Exit | Walk-over or switch; visible sign; 20% adjacent to start | Perception and recognition |
| Textures/light | Random; held-out texture split | Prevent RGB overfit |
| Determinism | `(seed, spec)` → exact WAD; `spec_hash` | Reproducibility and leakage |

**Solvability checks.** The generator must assert that (a) the exit is reachable from the
start using the keys that are obtainable without crossing an unopened lock; (b) every key
is reachable before its lock; (c) the map has at least one path to the exit; (d) no door is
placed in a wall band that cannot be opened; (e) the start is not inside a monster or a
wall. The current generator already validates door metadata against `doom_env.mapinfo`;
extend the validation to the key/lock graph and the exit path.

**Difficulty knobs.** Room count, graph degree, loop count, dead-end count, key count,
monster count, ammo budget, cover density, start distance to the exit, and the number of
doors on the critical path. The benchmark should sample these knobs from a config, not
hard-code a single distribution.

### 6.4 Splits and leakage

| Split | Seeds / maps | Use |
|---|---|---|
| Train | Generated 0–9,999; Freedoom MAP01–22 | BC, DAgger, PPO |
| Val-ID | Generated 10,000–10,999, same configs | Checkpoint selection/gates |
| Test-ID | Generated 30,000–30,999, same configs | Final report; touch once |
| Test-OOD | Generated 40,000+, new sizes/depths/textures/mixes | Generalisation |
| Test-Freedoom | MAP23–32 | Hand-made transfer |

**Leakage rules.** No test map in any training manifest; no test frames in BC/PPO; no
hyperparameter selection on test; keep a `test_touch.log` with timestamp, checkpoint and
metric; `spec_hash` intersection must be empty; Freedoom MAP23–32 never trained; texture
and generator-config holdouts are explicit axes. If a test map is touched during tuning, it
becomes validation and a fresh test block is generated.

**Why seed holdout is not enough.** A generator with the same grammar and the same
parameter ranges will produce maps that share structural biases (e.g., BSP rooms, 1-tile
doors, critical-path key placement). A seed holdout tests memorisation of specific layouts;
a config holdout tests generalisation of the grammar. Use both. The OOD split should vary
room count, graph degree, loop/dead-end counts, texture set, monster mix and key order.

**Texture holdout.** Split the wall/floor/ceiling/door/exit textures into train and test
sets. The agent must not learn “this exact wall texture means exit”; it must learn the sign
or the switch. Freedoom is the natural texture OOD set.

**Monster holdout.** Train on a subset of monster types and test on held-out types to
measure combat generalisation. The hidden suite likely includes all types, so this is a
diagnostic, not the primary metric.

### 6.5 Metrics

| Metric | Definition | Target | Why |
|---|---|---|---|
| Exit rate | Fraction of maps with exit | T1 ≥70%; T2 ≥30%; T3 ≥10% | Dominant term |
| Proxy | Mirror of official categories | Beat +8.09 | Comparable score |
| Coverage | Unique sectors / total | T2 ≥25%; Freedoom ≥8% | Exploration |
| Keys/locks/doors | Counts per episode | All keys on T3; ≥1 lock | Progression |
| Kills/damage/deaths | Combat outcomes | Positive K/D; damage <50 | Survival |
| SPL | `S_i / max(L_i, P_i)` on successes | >0.3 | Path efficiency |
| Success@k | P(at least one of k engine seeds exits) | k=5 | RNG robustness |
| Sample efficiency | Steps to 50% of final exit rate | Lower is better | Algorithm choice |
| Latency | p50/p99; per-map total vs budget | p99 <8 ms; <20% | Validator margin |
| RAM/VRAM | Peak RSS / CUDA | <8 GB / <2 GB | Contract |
| Determinism | Action hash on repeat | 100% identical | Replay validity |
| Calibration | Exit precision/recall; colour; door F1 | Precision >0.7; recall >0.5 | Aux heads useful |
| Degeneracy | Spin, no-op, action entropy | Spin <10%; no-op <20% | Catches circling |

**SPL** is the standard path-efficiency metric: `S_i` is the shortest path length, `L_i` the
agent’s path length, `P_i` the shortest path length from the agent’s start; the score is
`S_i / max(L_i, P_i)` on successful episodes, averaged. It distinguishes an agent that
exits quickly from one that wanders. The shortest path comes from a training-only BFS on
the nav grid; it is not available to the agent.

**Success@k** is the probability that at least one of `k` engine seeds exits. It measures
robustness to ViZDoom’s monster/damage RNG. Report `k=5` for final claims; `k=1` for gates.

**Degeneracy metrics** catch policies that look busy but do not progress: spin fraction
(angular velocity with near-zero displacement), no-op fraction, action entropy, and the
ratio of revisited cells to unique cells. A policy that circles the start room has high
spin and low coverage.

### 6.6 Proxy score

```
proxy_map = 10·exit + 3·keys + 3·locks + 0.5·doors
          + min(0.3·sectors, 10) + 0.6·kills
          − 0.01·damage − (3·(1 − steps/T) + 1)·death
```

Tier weights `[1.0, 1.2, 1.5, 2.0]`; report `Σ_t w_t · mean proxy_t` and every component
separately. Also report a **normalised score**:

```
proxy_norm = proxy_map / (10 + 3·n_keys + 3·n_locks + 0.5·n_doors + 10 + 0.6·n_monsters)
```

clipped to [0,1], to compare maps of different sizes. The official weights are hidden; the
report must say so, and every table must show the components, not just the sum.

**Why the ordering matters.** If the proxy weights are wrong (e.g., kills are weighted
higher), the agent may learn to farm kills. The ordering in `docs/EVALUATION.md` is the only
reliable signal: exit is largest, progression next, exploration next, combat last. The proxy
should enforce that ordering by construction: the maximum possible kill term (0.6 × 16 =
9.6) is still less than the exit term (10), and the sector cap is 10, so no combination of
non-exit terms can dominate a single exit. This is a design choice, not the official
formula.

### 6.7 Diagnostic probes

| Probe | Procedure | Metric | Pass criterion | What it proves |
|---|---|---|---|---|
| Memory wipe | Zero map, tokens and GRU at key pickup | Exit drop; return rate | >30% relative drop | The agent uses memory |
| Locked-door recall | Key far from door; measure return within 300 steps | Return rate; steps-to-return | >50% return | Long-horizon token memory |
| Door angles | 8 headings × 8 offsets around one door | Success rate per pose | >70% overall; >50% worst | Door skill generalises |
| Frontier coverage | C7, 300 steps | New sectors/100 steps; coverage | >20% coverage | Exploration is directed |
| Combat duels | 1v1 at 128/256/512 units | Win rate; TTK; damage | >70% at 256; damage <30 | Combat competence |
| No-op/spin | 200-step episode | Spin fraction; no-op fraction; entropy | Spin <10%; no-op <20% | No degenerate policy |
| Colour swap | Swap door/key textures at test | Colour accuracy; lock success | Accuracy >70%; drop <20% | Colour is semantic, not memorised |
| Exit-sign ablation | Replace exit sign with wall texture | Exit rate | >50% drop | Exit perception is used |
| Observation corruption | 10% ray dropout; RGB blur; entity shuffle | Exit/coverage degradation | <20% relative drop | Robustness |
| Reward ablation | Exit-only vs shaped | Steps to 50% exit | Shaped <50% steps | Shaping helps |
| Determinism | Two runs, same seed | Action hash | Identical | Replay validity |

**Memory wipe** is the causal test. If zeroing the map/tokens after key pickup does not
change the exit rate, the agent was not using memory. **Locked-door recall** is the
targeted version: the key is far from the door, and the agent must return. **Door angles**
is the transfer test that the current agent fails. **Colour swap** distinguishes semantic
colour from memorised texture. **Exit-sign ablation** distinguishes exit perception from
lucky wall-following.

### 6.8 Integration into training

**Curriculum.** `ladder.yaml` defines categories, seeds, ticks and weights. The PPO sampler
draws 30% from mastered categories, 50% from the current stage and 20% from the hardest.
This replaces the hand-written `Sampler` progress bands, which are too coarse and do not
adapt to the agent’s actual failures.

**Prioritized Level Replay.** After the baseline, compute per-map priority from TD error or
learning progress:

```
p_i ∝ (|TD_i| + ε)^β
P(i) = p_i / Σ_j p_j
```

Replay high-priority maps, cap staleness at 20 updates, and keep a minimum probability for
each category to avoid forgetting. PLR is a scheduler, not a new algorithm; it is cheap and
fits the existing level pool.

**Evaluation gates.** Run a fast subset (5 maps/category, 1 seed) every 10 updates; run the
full suite (30 maps/category, 5 seeds) at phase boundaries. Promote only if the tier-
weighted proxy improves with a paired bootstrap CI that excludes 0. A gate that passes on
the fast subset but fails on the full suite is a variance warning, not a promotion.

**Ablations.** Map, scan match, tokens, manager, asymmetric critic, exit channel, colour
head, BC/DAgger, frontier reward. Run each for a fixed 2-hour budget on the same seeds;
report the delta with a CI; stop an ablation early if the CI is clearly negative. The
ablation table is part of the submission; judges reward evidence, not just a score.

### 6.9 Statistical rigor

- **Seeds.** 5 engine seeds per map for final claims; 1 for gates. ViZDoom has monster-AI
  and damage RNG; one seed is a point estimate, not a result.
- **Maps.** ≥30 maps per category for final; ≥8 for gates. Stratified bootstrap over maps
  (10k resamples), not over episodes, to avoid pseudo-replication: episodes on the same map
  are correlated.
- **Estimators.** Mean ±95% CI, interquartile mean (IQM), and performance profiles
  (Agarwal et al., 2021). For ablations, paired bootstrap over maps; report the delta and
  the CI, not just “better.”
- **No cherry-picking.** Report every map and every seed; keep failed runs in the log;
  never report “best of n”; pre-register the primary metric (tier-weighted proxy on test)
  and secondary metrics (exit, coverage).
- **Test discipline.** Test only at phase boundaries; log every touch; a test set touched
  during tuning becomes validation, and a fresh block is generated. This is the only way to
  keep the final number honest.

### 6.10 File and CLI design

| Path | Role |
|---|---|
| `training/suite/spec.py` | Category/MapSpec, seed blocks, parameter ranges |
| `training/suite/generate.py` | WADs + `manifest.jsonl`; `spec_hash`; leakage check |
| `training/suite/run.py` | Checkpoint → episodes; one JSONL row each; resumable |
| `training/suite/metrics.py` | Proxy, SPL, coverage, stratified bootstrap, IQM |
| `training/suite/probes.py` | Memory wipe, door angles, colour swap, corruption, determinism |
| `training/suite/report.py` | Markdown/JSON report; baseline diff; CI flags |
| `training/suite/leakage.py` | Hash intersection; test-touch log |
| `training/suite/configs/ladder.yaml` | Categories, seeds, ticks, weights, gates |
| `training/data/suite/{split}/{category}/manifest.jsonl` | Maps + metadata |

Manifest row:

```
{map_id, wad, map_name, category, tier, seed, ticks, spec_hash, split, generator_version}
```

CLI:

```bash
python -m training.suite.generate --split val  --categories c0,c1,c4,c7 --n 30
python -m training.suite.generate --split test --categories all --n 30
python -m training.suite.run --ckpt runs/ppo/ckpt.pt --split val --categories all \
    --seeds 1 --fast --out runs/suite/val_ppo
python -m training.suite.run --ckpt runs/ppo/ckpt.pt --split test --categories all \
    --seeds 5 --full --out runs/suite/test_ppo
python -m training.suite.report --run runs/suite/test_ppo --baseline runs/suite/test_bc
python -m training.suite.probes --ckpt runs/ppo/ckpt.pt \
    --probes memory_wipe,door_angles,colour_swap,determinism
```

`run.py` reuses `training/evaluate.run_agent_episode` for the episode loop, but attaches
suite metadata and writes one row per episode. `metrics.py` aggregates over maps first,
then episodes. `report.py` emits a Markdown table with per-category means, CIs and
component breakdowns; it also flags any test-touch violation.

### 6.11 Example experiment matrix

| Run | Change | Expected signal | Cost |
|---|---|---|---|
| B0 | PPO-25 baseline | Tier-1 6/8, tier-2 0/8 | Done |
| B1 | + scan match + map | Drift <0.5 cell; tier-2 coverage >15% | 2 h |
| B2 | + balanced BC | Turn >65%; doors ≥6/8; tier-1 ≥5/8 | 3 h |
| B3 | + DAgger | Tier-2 exits ≥2/8 | 2 h |
| B4 | + asymmetric critic + PPO | Tier-2 ≥3/8; proxy > +8.09 | 4 h |
| B5 | + tokens + manager | Tier-3 ≥1/8; recall >50% | 2 h (stretch) |
| B6 | + door-servo option | Door-angle >70%; room doors >30% | 2 h (stretch) |
| B7 | OOD + Freedoom | Coverage ≥5%; exit ≥1/10 | 1 h |

The matrix is ordered by expected value per hour. B0–B4 are the core; B5–B6 are stretch;
B7 is the final report.

---

## 7. Meaningful improvements over standard approaches

The following ideas are not all novel in the literature, but each is a defensible,
well-engineered choice for this track. For each: mechanism, training signal, inference
cost, risk, expected gain, and the ablation that proves it.

### 7.1 Colour-resolving event tokens with a learned manager

**Mechanism.** Store a predicted colour on each door token and re-rank tokens when a key is
picked up. Structured observations hide colour, but RGB shows `DOORBLU`/`DOORRED`/`DOORYEL`
and the key sprite. A token stores `colour` and `keys_held_at_sighting`; when a key flag
flips, the manager selects the matching door without re-exploring.

**Training signal.** A 3-way colour head on the RGB encoder, trained with privileged map
labels (door texture / key thing). The manager is trained with option-level PPO; the
goal-completion reward is +0.5 when the matching door is opened. Use HER-style relabelling:
if a key is picked up, relabel the matching door token as the goal for the K steps before
the pickup. This turns a failed “go to the door without the key” episode into a useful
“go to the key, then the door” episode.

**Inference cost.** Token encoder ~0.2 ms; manager every 8 steps ~0.05 ms. Negligible.

**Risk.** Colour head overfits generated textures; manager non-stationarity; token spam.
Mitigate with texture randomisation, token merging, and a frozen “nearest matching token”
selector for the first PPO run.

**Expected gain.** Tier-3 lock success and return-to-door rate; the two-key tier-4 chain.

**Ablation.** Zero the colour features at test; compare tier-3 lock success and
return-to-door rate. Replace the manager with a fixed selector; compare coverage and exit.

### 7.2 Exit-likelihood map channel from a self-supervised exit head

**Mechanism.** Write the exit head’s probability into the map at the ray endpoint. The exit
is invisible to rays/entities; a per-frame detector only fires when the exit is in view. A
map channel lets the policy return to a remembered cue after a key or fight, and gives the
critic a useful feature without leaking privileged state to the actor.

**Training signal.** Exit head with privileged linedef labels: visible (BCE), bearing (13
classes), distance (6 bins). Temperature-scale the head for calibration. At inference, for
each ray endpoint, write `p_exit` into channel 6 with a slow decay; add an exit-cue token
when `p_exit` exceeds a threshold. Never use the head as a hard rule.

**Inference cost.** <0.1 ms.

**Risk.** False positives; Freedoom switch textures differ; calibration drift. Use
precision/recall monitoring and an exit-sign ablation.

**Expected gain.** Exit rate and time-to-exit on tier 2–4; Freedoom transfer.

**Ablation.** Remove the exit channel vs remove the exit head; measure exit rate,
time-to-exit, and the exit-sign ablation drop.

### 7.3 Asymmetric privileged critic with geodesic potential

**Mechanism.** Give the critic true pose, keys, locks, geodesic distance and monster counts;
keep the actor observation-only. Exit reward is sparse and delayed, and value is the
noisiest part of PPO. True pose and sector-graph distance remove aliasing.

**Training signal.** Critic input = 384-d GRU state + 32 privileged features; same PPO
value loss; actor unchanged; do not instantiate the critic at inference. Geodesic distance
from a training-only BFS; objective = nearest unheld key if a lock blocks the exit, else
exit; potential `Φ = −d/1000`, annealed to 0. The potential is potential-based (Ng et al.,
1999), so it preserves the optimal policy.

**Inference cost.** Zero.

**Risk.** Privileged leakage; objective-selection bugs; distribution shift. Unit-test that
the actor’s input dict contains no privileged key.

**Expected gain.** Better value estimates, sample efficiency, and exit rate.

**Ablation.** Symmetric vs asymmetric critic on the same BC init, 10 PPO updates; compare
value loss, exit rate and sample efficiency. With vs without potential.

### 7.4 Event-gated dual-timescale memory

**Mechanism.** Write to the slow token bank only on salient events: new sector, key pickup,
door open, exit cue, or scan residual above threshold. Writing every frame creates
redundant tokens and attention collapse; the GRU handles local control, tokens handle
keys/locks/exits over hundreds of steps.

**Training signal.** A fixed write rule first, then a small learned gate. Merge tokens
within 64 units; evict resolved/oldest; train the token encoder end-to-end with PPO; add an
auxiliary loss to predict token age/resolved. Use memory-wipe and recall probes.

**Inference cost.** ~0.2 ms.

**Risk.** Gate thresholds; token drift; attention collapse. Start with a fixed rule, then
learn the gate only if the ablation is positive.

**Expected gain.** Tier-3 return rate; interpretable attention for the demo.

**Ablation.** Write every step vs event-gated; remove tokens; measure tier-3 return and
attention entropy.

### 7.5 Door-servo option with option-level PPO

**Mechanism.** Treat door approach as a reusable closed-loop option invoked when a door
token is selected. Straight drills teach one approach angle; room doors need turn,
approach, press once, wait, cross. The option terminates when the door opens or times out.

**Training signal.** Initiate when a door token is selected and distance <256; condition the
worker on a door-servo goal (token + phase); terminate on open or after 40 steps; train with
option-level PPO (open +0.5, cross +0.5, time penalty, USE-edge penalty); shared trunk;
termination head. Start with BC from room-door demos, then PPO.

**Inference cost.** Same forward pass; a few option-state floats.

**Risk.** Option collapse; sparse reward; door reversal; non-stationarity.

**Expected gain.** Door-angle success and tier-2 room-door exits.

**Ablation.** Flat policy vs option on the door-angle probe (8×8 poses) and tier-2 room
doors.

### 7.6 Bonus: map-conditioned auxiliary prediction

**Mechanism.** Add an auxiliary head that predicts the next local map crop or the occupancy
change from the current latent. This forces the GRU to encode geometry and dynamics, not
just current pixels. It is a lightweight world-model loss without imagination.

**Training signal.** MSE between predicted and actual next map crop (or next occupancy
logits), weight 0.05. No inference cost; the head is dropped at inference.

**Risk.** The auxiliary can dominate the representation; keep the weight small and monitor
the main losses.

**Expected gain.** Better map use and short-horizon prediction; sample efficiency.

**Ablation.** With vs without the auxiliary; measure coverage and exit rate.

---

## 8. Reading list and execution plan

### 8.1 Prioritized reading list

**Core platform and algorithms**

1. ViZDoom — Kempka et al., 2016, [arXiv:1605.02097](https://arxiv.org/abs/1605.02097)
2. ViZDoom competitions — Wydmuch et al., 2018, [arXiv:1809.03470](https://arxiv.org/abs/1809.03470)
3. DQN — Mnih et al., 2015, [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
4. DRQN — Hausknecht & Stone, 2015, [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)
5. Playing FPS games with deep RL — Lample & Chaplot, 2017, [arXiv:1609.05521](https://arxiv.org/abs/1609.05521)
6. PPO — Schulman et al., 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
7. GAE — Schulman et al., 2015, [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
8. IMPALA — Espeholt et al., 2018, [arXiv:1802.01561](https://arxiv.org/abs/1802.01561)
9. Sample Factory / APPO — Petrenko et al., 2020, [arXiv:2006.11751](https://arxiv.org/abs/2006.11751)
10. R2D2 — Kapturowski et al., ICLR 2019, [OpenReview](https://openreview.net/forum?id=r1lyTjAqYX)

**Navigation and memory**

11. Neural Map — Parisotto & Salakhutdinov, 2018, [arXiv:1702.08360](https://arxiv.org/abs/1702.08360)
12. Active Neural SLAM — Chaplot et al., 2020, [arXiv:2004.05155](https://arxiv.org/abs/2004.05155)
13. Neural Topological SLAM — Chaplot et al., 2020, [arXiv:2005.12256](https://arxiv.org/abs/2005.12256)
14. Occupancy Anticipation — Ramakrishnan et al., 2020, [arXiv:2008.09285](https://arxiv.org/abs/2008.09285)
15. Cognitive Mapping and Planning — Gupta et al., 2017, [arXiv:1702.03920](https://arxiv.org/abs/1702.03920)
16. Semi-parametric topological memory — Savinov et al., 2018, [arXiv:1803.00653](https://arxiv.org/abs/1803.00653)
17. Habitat — Savva et al., 2019, [arXiv:1904.01201](https://arxiv.org/abs/1904.01201)
18. ObjectNav evaluation — Anderson et al., 2018, [arXiv:1807.06757](https://arxiv.org/abs/1807.06757)
19. ProcTHOR — Deitke et al., 2022, [arXiv:2206.06994](https://arxiv.org/abs/2206.06994)

**Exploration**

20. Count-based exploration — Bellemare et al., 2016, [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
21. RND — Burda et al., 2018, [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
22. NGU — Badia et al., 2020, [arXiv:2002.06038](https://arxiv.org/abs/2002.06038)
23. Agent57 — Badia et al., 2020, [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)
24. ICM — Pathak et al., 2017, [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)
25. Go-Explore — Ecoffet et al., 2019, [arXiv:1901.10995](https://arxiv.org/abs/1901.10995)
26. MAP-Elites — Mouret & Clune, 2015, [arXiv:1504.04909](https://arxiv.org/abs/1504.04909)
27. PLR — Jiang et al., 2021, [arXiv:2010.03934](https://arxiv.org/abs/2010.03934)

**Imitation and hierarchy**

28. DAgger — Ross et al., 2011, [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
29. Expert iteration — Anthony et al., 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439)
30. Kickstarting — Schmitt et al., 2018, [arXiv:1803.03835](https://arxiv.org/abs/1803.03835)
31. Asymmetric actor-critic — Pinto et al., 2017, [arXiv:1710.06542](https://arxiv.org/abs/1710.06542)
32. FeUdal — Vezhnevets et al., 2017, [arXiv:1703.01161](https://arxiv.org/abs/1703.01161)
33. HIRO — Nachum et al., 2018, [arXiv:1805.08296](https://arxiv.org/abs/1805.08296)
34. HER — Andrychowicz et al., 2017, [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)
35. Director — Hafner et al., 2022, [arXiv:2206.04114](https://arxiv.org/abs/2206.04114)

**Model-based and benchmarks**

36. World Models — Ha & Schmidhuber, 2018, [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
37. DreamerV2 — Hafner et al., 2020, [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
38. DreamerV3 — Hafner et al., 2023, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
39. MuZero — Schrittwieser et al., 2019, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
40. EfficientZero — Ye et al., 2021, [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)
41. ProcGen — Cobbe et al., 2019, [arXiv:1912.01588](https://arxiv.org/abs/1912.01588)
42. MiniHack — Samvelyan et al., 2021, [arXiv:2109.13202](https://arxiv.org/abs/2109.13202)
43. NetHack — Küttler et al., 2020, [arXiv:2006.13760](https://arxiv.org/abs/2006.13760)
44. Crafter — Hafner, 2021, [arXiv:2109.06780](https://arxiv.org/abs/2109.06780)
45. DMLab — Beattie et al., 2016, [arXiv:1612.03801](https://arxiv.org/abs/1612.03801)
46. Statistical precipice — Agarwal et al., 2021, [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
47. Deep RL that Matters — Henderson et al., 2017, [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
48. Show Your Work — Agarwal et al., 2019, [arXiv:1909.03004](https://arxiv.org/abs/1909.03004)

**Further citations used above:** DDPG (1509.02971); Papoudakis et al. (1807.01960);
Ape-X (1803.00933); SEED (1910.06591); DD-PPO (1911.00357); Option-critic (1609.05140);
HAC (1712.00948); Neural Episodic Control (1703.01988); GTrXL (1910.06764); Decision
Transformer (2106.01345); TD-MPC2 (2310.16828); IRIS (2209.00588); Plan2Explore
(2005.05960). Links follow `https://arxiv.org/abs/<id>`.

### 8.2 Next 6 hours

| Time | Action | Output / gate |
|---|---|---|
| 0:00–0:45 | Freeze PPO-25; run the full suite on the current checkpoint; record baseline with CIs | `runs/suite/baseline`; no code changes |
| 0:45–2:15 | Implement scan match + 8-channel map + pose vector; unit-test drift on 5 maps | Drift <0.5 cell/250; p99 <2 ms |
| 2:15–4:15 | Balanced BC on nav + frontier + doors + arena + keys; 3× oversampling of USE/key/exit; room-door data | Turn >65%; door drills ≥6/8; tier-1 ≥5/8 |
| 4:15–5:15 | DAgger on 30–50k student states (frontier/A*/door teachers); mix 50/50 | Tier-2 exits ≥2/8; no tier-1 regression |
| 5:15–6:00 | Evaluate; decide whether to continue to PPO or ship the best BC | Ship rule: best held-out weighted proxy |

If only 6 hours: stop after DAgger, run the full suite, and write the brief. Do not start
PPO unless the BC gate is clean.

### 8.3 Next 12 hours

| Time | Action | Output / gate |
|---|---|---|
| 0:00–1:00 | Baseline suite + freeze PPO-25; write the test-touch log | Baseline JSON + CIs |
| 1:00–3:00 | Scan match, map channels, pose, unit tests | Drift gate; latency gate |
| 3:00–5:00 | Balanced BC + room-door demonstrations; per-head accuracy | BC gate |
| 5:00–7:00 | DAgger + mixed replay; evaluate | Tier-2 ≥2/8 |
| 7:00–10:00 | PPO + asymmetric critic + 30% rehearsal; manager frozen | Tier-2 ≥3/8; tier-3 ≥1/8; proxy > +8.09 |
| 10:00–11:00 | Unfreeze manager/tokens only if gates pass; otherwise ablate | Ablation table |
| 11:00–11:45 | Full test suite (5 seeds × 30 maps), CIs, report | Final numbers |
| 11:45–12:00 | Freeze weights, validator, brief, demo | Submission |

**Stop rules.** Tier-2 <2/8 after DAgger → cut the manager and tokens; ship the best BC/PPO
checkpoint. Tier-1 <5/8 → roll back to PPO-25. p99 >10 ms → cut the global map or token
encoder. Test touched twice → regenerate a fresh block. Never trade the exit term for kills.

### 8.4 Next 24 hours

| Block | Work | Gate |
|---|---|---|
| 0–2 h | Baseline, validator, splits, test-touch log | Repo clean; baseline JSON |
| 2–5 h | Scan match, map, pose, unit tests | Drift/latency gates |
| 5–9 h | Balanced BC + room-door demos + per-head accuracy | BC gate |
| 9–12 h | DAgger + mixed replay + evaluation | Tier-2 ≥2/8 |
| 12–17 h | PPO + asymmetric critic + 30% rehearsal; manager frozen | Tier-2 ≥3/8; proxy > +8.09 |
| 17–20 h | Tokens/manager or door-servo option (one, not both) | Tier-3 ≥1/8 or door-angle >70% |
| 20–22 h | Full suite, ablations, CIs, report | Final numbers |
| 22–23 h | Freeze weights, validator, brief, demo | Submission-ready |
| 23–24 h | Q&A rehearsal, final checks | Submitted |

### 8.5 Decision tree

1. **Is the validator passing?** If not, stop training and fix the submission path.
2. **Is the BC gate clean (turn >65%, doors ≥6/8, tier-1 ≥5/8)?** If not, do not start
   PPO; fix the data or the map.
3. **Is tier-2 ≥2/8 after DAgger?** If not, cut the manager/tokens; ship the best BC.
4. **Is the weighted proxy > +8.09 after PPO?** If not, roll back to PPO-25.
5. **Is tier-3 ≥1/8?** If not, the tokens/manager did not help; drop them.
6. **Is the full suite within latency/RAM?** If not, cut streams until it is.
7. **Is the test set untouched?** If not, regenerate and re-evaluate once.

### 8.6 What to report

- The full per-category table with mean, 95% CI, and component breakdown.
- The ablation table with paired deltas and CIs.
- The memory-wipe, door-angle, colour-swap and exit-sign probes.
- The latency/RAM/determinism numbers from the validator.
- The test-touch log and the exact commit hash.
- A clear statement that the official weights are hidden and the proxy is an assumption.

---

## Appendix A — Measured results

| Checkpoint | Tier-1 exits | Tier-1 proxy | Tier-2 proxy | Tier-3 proxy | Tier-4 proxy | Drills | Freedoom |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bc_map2_e0` | 2/8 | +0.35 | −0.99 | −0.89 | −1.61 | 0/8 | 0/10, −0.77 |
| `bc_use` | 1/8 | +0.89 | −0.26 | −0.70 | −1.41 | **6/8** | not run |
| `bc_head` | 1/8 | −1.53 | −0.22 | −0.87 | −2.17 | 0/8 | not run |
| `bc_arena` | **5/8** | **+5.42** | −0.18 | −0.38 | −0.73 | 0/8 | 0/10, +1.00 |
| PPO-25 | **6/8** | **+8.20** | **+0.48** | −0.03 | −0.32 | 0/8 | 0/10, +0.36 |

Other measurements: BC 100,921 transitions, loss 6.21→1.89, move 85.6%, strafe 76.0%,
attack 92.4%, turn 43.2%, exact six-head 20.8%, held-out exit 0/4. Doors: 360 drills, 6/8
after two epochs, 0/8 room doors; room-door teacher 200/200; student 4/8 drills, 0/8 room
doors; frozen trunk 0/8 drills. Forgetting: drill-only tier-1 2/2→0/2, proxy +11.4→−1.2;
`bc_combined` tier-1 1/8, drills 0/8; `bc_success` tier-1 4/8, tier-2 −0.54, drills 0/8.
MoE: 6/8 drills at threshold 0.1, tier-1 5/8, tier-2 +0.32, tier-3 −0.22, tier-4 −0.93;
threshold 0.5 collapses to PPO (0/8 drills). Drift: 63°→16° heading, 907→361 units over
250 steps. Validator: PASSED, init 0.4–0.5 s, p50 0.8–1.0 ms, p99 0.9–2.0 ms, max 4 ms.

## Appendix B — Contract gaps

| Gap | Evidence | Consequence |
|---|---|---|
| Exit invisible | `RAY_HIT_TYPES` has no exit; `ENTITY_TYPES` has no exit; wrapper sets exit only on episode end | Must learn from RGB/depth; privileged linedef labels |
| Key/lock colour hidden | `classify()` maps all keys to `key`; `hit_type=3` means any lock | Must read colour from RGB or try doors |
| USE edge | `CONTRACT.md` §3 | Holding USE presses once; pressing on a moving door reverses it |
| Turn stateful | Measured table in `PLAN.md` Appendix A | Track `turn_held`, warm-up, previous action |
| Scoring hidden | `EVALUATION.md` | Proxy only; report components |
| Coverage sector-based | `sector_polygons()` docstring | Reward sectors, not only cells |
| Rays stop at monsters/items | `sensors.py` ray-vs-circle | Rays are a 360° threat/item radar |

## Appendix C — Reproducibility checklist

- [ ] Fixed seeds for maps, engine, Python/NumPy/Torch.
- [ ] `spec_hash` recorded in every manifest; train/test intersection empty.
- [ ] Test-touch log with timestamp, checkpoint, metric.
- [ ] One command per suite (`generate`, `run`, `report`, `probes`).
- [ ] Raw JSONL per episode, not just aggregates.
- [ ] CIs computed over maps, not episodes.
- [ ] Validator run on the final commit, three times.
- [ ] Weights in Git LFS; repo ≤1 GB; `submission/requirements.txt` empty if possible.
- [ ] Deterministic argmax; no dropout/batch norm; seeded sampling if used.
- [ ] No network, no engine access, no writes outside `/tmp` at inference.

---

*Prepared 2026-11-26. Measured numbers are from this repository; literature claims are
cited by arXiv ID or venue. Official scoring weights remain unknown; all proxy scores are
assumptions and should be reported as such.*
---

## Afterword — what the Phase 3 build changed in this review

The literature review above argued for hierarchical goal conditioning, map-centric
memory, and balanced expert data. Building and measuring them produced four
concrete corrections:

1. **Goal conditioning helps navigation but is not a substitute for local
   control.** The learned goal head reaches 5.7° median bearing error on the
   training distribution and drives C0/C5 success from 0% to 100%. On the
   two-key C6 map the policy still parked one tile short of the red door with a
   perfect goal; an observation-only anti-freeze reflex, not a better goal, fixed
   it. Hierarchy solves *what* to pursue, not *how* to recover.
2. **A benchmark can be wrong in the same way a model can.** The room
   generator's locks were usually optional (5/30 tier-4 maps actually required a
   key; 0/30 required two), so the "key" categories measured nothing. Purpose-
   built cut-edge drills were necessary before any architecture comparison was
   meaningful. This is the benchmark analogue of leakage.
3. **DAgger is not automatically safe.** Student rollouts with β=0 produced 73%
   turn labels because the student froze; behaviour cloning collapsed to 99.8%
   spin. DAgger needs teacher mixing, stuck-tail downsampling, or a trust region;
   otherwise it poisons the prior it was meant to correct.
4. **Mixture-of-experts routing works when the cue is perception, not a learned
   black box.** The earlier gate failed on t2–t4. A router that asks "is an enemy
   visible?" and hands combat to a frozen specialist restored 7/12 tier-1 exits
   and 25 kills while keeping the key categories at 100%, with no privileged
   state. The trade-off is interpretability and a hard behavioural boundary, not
   accuracy.

These findings align with the hierarchical-RL and options literature (goal
abstraction reduces horizon; low-level recovery still needs coverage) and with
the procedural-content literature (distribution validity is an empirical
property, not a generator claim). They also reinforce the review's original
conclusion: on a 1-CPU, 8 GB budget, a compact recurrent policy with a
deterministic spatial memory and a small set of learned goals is a stronger use
of parameters than a large end-to-end model.
