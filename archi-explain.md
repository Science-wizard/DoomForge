# DoomForge — Architecture & ML Pipeline, Explained From Scratch

This document assumes **no machine-learning background**. Every term is defined
the first time it appears, and there is a full glossary at the end. Read §1–§3
to understand the vocabulary, then §4–§10 for the pipeline, then §12 for the
questions you are likely to be asked.

If you only have one minute, read **§2 (the pipeline in one picture)** and
**§11 (the numbers)**.

Companion documents (more technical, less explanatory):
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/TRAINING.md`](docs/TRAINING.md),
[`explanation.md`](explanation.md).

---

## 1. The problem, in plain language

A DOOM agent plays a level. It does not see the game's internal state; it gets a
**fixed-size bundle of sensor readings** and must press buttons.

| Term | Plain meaning |
|---|---|
| **Observation** | What the agent is allowed to see at one moment: 64 laser range-finders ("rays"), up to 32 visible objects, 17 numbers about health/ammo/keys/velocity, and a 120×160 colour image. |
| **Action** | Nine on/off buttons: forward, back, strafe left/right, turn left/right, attack, use, next weapon. |
| **Decision step** | Every 4 game ticks (a tick is DOOM's simulation unit) the agent gets one observation and returns one action. |
| **Episode** | One run through one map, from spawn until exit, death, or the tick limit. |
| **Reward** | A number the trainer uses to say "that was good/bad". The *real* scorer is hidden; we use a proxy (§10). |

Two properties make this hard, and they are worth stating in exactly these words:

1. **Partial observability.** The agent has no map coordinates and no exit
   marker. It cannot see behind itself. So the same observation can correspond to
   many different situations — the problem is a **POMDP** (partially observable
   Markov decision process). The agent must *remember*.
2. **Sparse, delayed reward.** The big reward (reach the exit) arrives hundreds
   of decisions after the actions that made it possible. Figuring out *which*
   earlier action deserves credit is the **credit-assignment problem**.

A third constraint shapes every design choice: at evaluation the agent gets
**one CPU thread and 40 ms per decision**. Training can use a GPU; inference
cannot.

---

## 2. The pipeline in one picture

```
  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
  │ 1. MAPS     │──►│ 2. SPLITS    │──►│ 3. TEACHER    │──►│ 4. EGO MAPS  │
  │ generate    │   │ train/val/   │   │ privileged    │   │ replay demos │
  │ WAD levels  │   │ test by seed │   │ expert plays  │   │ through the  │
  └─────────────┘   └──────────────┘   └───────────────┘   │ shipped map  │
                                                            └──────┬───────┘
                                                                   │
                                          ┌────────────────────────▼───────┐
                                          │ 5. BEHAVIOUR CLONING (BC)      │
                                          │ supervised copy of the teacher │
                                          └────────────────┬───────────────┘
                                                           │
                                          ┌────────────────▼───────────────┐
                                          │ 6. PPO × 3 STAGES              │
                                          │ practise, score, improve       │
                                          └────────────────┬───────────────┘
                                                           │
                                          ┌────────────────▼───────────────┐
                                          │ 7. EVALUATION + PROMOTION GATE │
                                          │ paired stats; one test look    │
                                          └────────────────────────────────┘
```

Read it as **two loops**:

- **Data loop (cheap, offline):** maps → teacher → demonstrations → ego maps.
- **Learning loop (expensive):** BC → PPO → evaluate → keep or reject → repeat.

The central idea: **do not ask reinforcement learning to discover exploration,
door timing, key logic and combat all at once from a sparse reward on a laptop.**
Instead, use a *privileged teacher* (an expert that is allowed to cheat during
training) to generate good examples, copy it, then let RL improve the copy using
only what the real agent can see.

---

## 3. Vocabulary you must be able to say out loud

These are the words that make the rest of the document readable.

### Learning paradigms

- **Supervised learning** — learning from labelled examples: "here is an input,
  here is the correct output". Like flashcards.
- **Imitation learning** — supervised learning where the "correct output" is
  what an expert did. Also called **behaviour cloning (BC)** when done directly
  on (observation → expert action) pairs. Like an apprentice copying a master.
- **Reinforcement learning (RL)** — learning by trial, error and reward, with no
  correct answer shown. Like training a dog. The learner is called the **agent**;
  the thing it interacts with is the **environment**.
- **Policy** — the agent's decision rule: given an observation, output a
  probability for each action. Written π(a | s). "Policy" and "network" are
  often used interchangeably here.
- **On-policy vs off-policy** — on-policy methods learn only from data the
  *current* policy just produced (PPO is on-policy). Off-policy methods can reuse
  old data (DQN, SAC). On-policy is simpler and more stable, but throws data away.

### Neural-network pieces

- **Parameter / weight** — a learned number inside the network. "2 M parameters"
  means two million such numbers.
- **Encoder** — a small network that turns one input (rays, image, map) into a
  compact vector of numbers. Several encoders run in parallel, one per input type.
- **MLP (multi-layer perceptron)** — the basic stack of `Linear → activation →
  Linear`. A universal-ish function approximator.
- **CNN (convolutional neural network)** — a network that slides small filters
  over a grid (an image or a map). Good at "is there a door-shaped pattern here?".
- **Attention** — a mechanism that lets each item in a set look at the others and
  decide what is relevant. Used for the variable-length entity list.
- **GRU (gated recurrent unit)** — a recurrent layer with a memory vector. It
  reads one input per step and carries a hidden state forward, so it can remember
  "I already have the red key" or "that door is still opening".
- **Hidden state** — the GRU's memory vector between decisions.
- **Logits** — raw, unnormalised scores the network outputs, one per action.
- **Softmax** — turns logits into probabilities that sum to 1.
- **Cross-entropy** — the standard loss for classification. It is low when the
  network gives high probability to the correct class, high when it does not.
- **Loss** — a number measuring how wrong the network is; training pushes it down.
- **Gradient / backpropagation** — the direction and size in which each weight
  should change to reduce the loss.
- **AdamW** — a standard optimiser (the algorithm that applies those changes).
- **Epoch / batch / window** — one pass over the data / a group of examples
  processed together / a fixed-length slice of an episode used for training.
- **Learning rate** — how big a step the optimiser takes.

### RL-specific pieces

- **Return** — the total (discounted) reward from a point onward. Discounting
  (γ, gamma) means "reward now is worth more than reward later".
- **Value function V(s)** — the network's guess of the return it will get from
  this situation. Trained by regression; used as a baseline.
- **Critic** — the network head that predicts V(s). The **actor** is the part
  that picks actions. Together: actor–critic.
- **Advantage A(s, a)** — "how much better was this action than the average
  action in this situation?" = return − value. Positive → reinforce; negative →
  discourage.
- **GAE (generalised advantage estimation)** — a standard, slightly-smoothed way
  to compute advantages from a run of rewards. Two knobs: γ (gamma, 0.995) and λ
  (lambda, 0.95). Higher λ = lower bias, higher variance.
- **PPO (proximal policy optimisation)** — the RL algorithm used here. It
  improves the policy in small, *clipped* steps so it does not collapse.
- **Clipped surrogate** — PPO's trick: ignore policy changes larger than ±10%
  (clip 0.1) in one update. Prevents destructive updates.
- **Entropy** — a measure of how spread-out the action probabilities are. An
  **entropy bonus** keeps the agent exploring instead of becoming deterministic
  too early.
- **KL divergence** — a measure of how different two probability distributions
  are. A **KL penalty/anchor** keeps the new policy close to a reference policy.
- **Reward shaping** — adding extra, denser reward terms (e.g. "new area seen")
  so learning is not waiting only for the sparse exit reward. Training only.
- **Curriculum** — training more often on the maps/skills the agent is currently
  bad at.
- **Privileged information** — information available during training but not at
  evaluation (true position, true distance to the exit). Using it is legal as
  long as the shipped agent never reads it.
- **Asymmetric actor–critic** — the actor sees only real observations; the critic
  additionally sees privileged information. This reduces the variance of the
  training signal without leaking anything into the policy.

### Evaluation pieces

- **Proxy score** — our stand-in for the hidden official scorer.
- **Tier weighting** — harder map categories count more in the headline number.
- **Bootstrap confidence interval (CI)** — estimate uncertainty by resampling the
  data many times with replacement and looking at the spread of the result. A
  95% CI that excludes zero means "the effect is probably real".
- **Paired comparison** — evaluate both agents on the *same maps* and compare
  per-map differences, which cancels map difficulty and is far more sensitive
  than comparing two independent averages.
- **OOD (out-of-distribution)** — maps deliberately unlike the training maps, to
  test generalisation.
- **Leakage** — accidentally training on test data. We check it by hashing every
  map file.

---

## 4. Stage 1 — Maps

**What.** Levels are generated programmatically (`training/maps/`): rooms carved
by recursively splitting a grid, doors on walls, coloured locks with matching
keys, an exit line with a visible exit-sign texture. Four difficulty tiers:
arenas with monsters, multi-room exploration, key→lock progression, and
gauntlets. There are also hand-built **drills** that isolate one skill (a door on
a room wall, a key behind the spawn, two keys and two locks), and 32 hand-made
Freedoom maps.

**Why not just use the official maps?** They are hidden, and you cannot measure
progress on them. A controllable distribution lets you (a) train, (b) isolate
failures per skill, and (c) measure generalisation on held-out seeds.

**Why "node-less" WADs?** DOOM maps normally need a precomputed BSP tree (a
spatial index) built by an external tool. ZDoom rebuilds it at load time, so the
generator can emit the map directly. That removes an external dependency and
makes generation deterministic in `(seed, tier, flags)`.

**Key idea to say:** *the map distribution is a design choice, not a detail.*
Drills guarantee the lock is on the only path, so the agent cannot "solve" a key
map by luck.

---

## 5. Stage 2 — Splits, and why they matter

Maps are split by generation seed:

| Split | Seeds | Purpose |
|---|---|---|
| train | 0+ | learning |
| validation | 10,000+ | model selection, promotion decisions |
| test | 30,000+ | touched **once** per shipped decision |
| OOD | 40,000+ | deliberately different maps |

Every WAD is hashed, and the pipeline verifies that train ∩ val and train ∩ test
are empty. Test runs require an explicit `--allow-test` flag and are logged in
`test_touch.log`.

**Why this matters.** If you tune against the same maps you report on, your
number is optimistic — the model has effectively seen the answers. Separating
selection (validation) from reporting (test) is the difference between a
measurement and a marketing claim. This is the answer to "how do you know you
didn't overfit the benchmark?"

---

## 6. Stage 3 — The privileged teacher

**What.** `training/teacher/` is a hand-written expert that is allowed to read
things the real agent cannot: true position, which keys it holds, door heights,
and the map geometry. It builds a 16-unit navigation grid from the level's walls,
runs **A\*** (a shortest-path search) with door costs and key gating, and follows
a priority: go to the exit if reachable with current keys → otherwise go to the
nearest needed key → otherwise explore. It has a **door state machine**
(approach, align, press USE once, wait, cross) and circle-strafing combat with
projectile dodging.

**Why.** This is the single biggest simplification in the project. The teacher
solves exploration, credit assignment and door timing *once*, offline, with
perfect information. It is **not** the submitted agent: it never runs at
evaluation, and `submission/` never imports `training/`.

**Analogy.** A driving instructor with a second set of pedals and a map. You
record what the instructor does, then teach a student who can only see out of the
windscreen.

**What it produces.** 13 demonstration datasets, **436k steps**, each step
storing: the observation, the teacher's action, the previous action, auxiliary
labels (is the exit visible? its bearing and distance; key colours; damage over
the next 8 decisions) and the teacher's current goal as an ego-relative vector
(dx, dy, distance, bearing, type).

---

## 7. Stage 4 — Ego maps, regenerated offline

**What.** The agent builds its own map from observations (`EgoMap` in
`submission/mapmem.py`). It estimates its own pose by **dead reckoning** —
integrating its executed actions and reported velocity — then projects rays,
items, keys and its own trail into a 160×160 grid, and reads out two
heading-up crops (fine and coarse) as a 16×32×32 tensor.

**Why regenerate them for the demos?** The demonstrations were recorded by the
teacher. When the student trains, it must see the *same* map it would build at
inference time. So each stored episode is replayed through the **shipped**
`EgoMap`, bit-packed, and used as training input.

This is the fix for **train/serve skew** (also called distribution shift): the
model must not learn on inputs that differ from the ones it will see when it is
deployed. It is a classic silent bug — the model looks fine in training and
fails in production because the preprocessing differed.

**Is the map "planning"?** No. It stores what was observed, at a pose the agent
computed itself. Nothing in it chooses a target or a route. It is perception and
memory; the network decides what to do with the picture.

---

## 8. Stage 5 — Behaviour cloning (BC)

**What.** Train the network by supervised learning on the teacher's
(observation → action) pairs.

- **Input:** 128-step windows of episodes, batches of 16.
- **Loss:** cross-entropy on the six action heads, plus auxiliary losses, plus
  value regression, plus goal prediction:
  `loss = CE + 0.2·aux + 0.5·value + goal`.
- **Details:** 5 epochs, AdamW at 3e-4, on Apple MPS (the Mac GPU).
- **Result:** validation loss **0.63** — i.e. the student's predicted button
  distribution is close to the teacher's.

**Why BC first?** RL from random weights would spend enormous time rediscovering
that doors need USE, or that keys are worth picking up. BC gives a policy that
already works in about an hour. PPO then improves it.

**Important nuance:** BC is *not* guaranteed to work, because the teacher is
omniscient. Some teacher decisions are not reproducible from the student's
limited view. That is exactly why PPO follows: it optimises the real task with
the real observations, and can learn to recover from the student's own mistakes.

**Terms to have ready:**
- *Auxiliary heads* — extra outputs (exit visible/bearing/distance, key colours,
  enemy visible, door ahead, damage next 8 steps) trained with privileged labels.
  They are **never used to act**; they force the shared encoder to represent
  things the raw observation hides. Think of them as side-quests that make the
  main skill better.
- *Goal head* — predicts "where should I be going" as an 8-number vector plus a
  4-way type (exit/key/frontier/…). It is trained on the teacher's goal, then its
  prediction is fed back into the policy as an input. During PPO it is frozen so
  this "intent" signal stays stable.
- *Class imbalance / up-weighting* — USE is rare but critical, so its loss is
  multiplied (×11 when a door is ahead, ×6 on USE labels). Otherwise the network
  would learn "never press USE" and still get low average loss.

---

## 9. Stage 6 — PPO

**What.** Reinforcement learning that improves the BC policy by playing the game
and being scored.

### 9.1 The core loop

1. Run the current policy in many parallel game instances.
2. Store (observation, action, reward, value, done) for each step.
3. Compute **advantages** — was each action better or worse than expected?
4. Update the policy to make good actions more likely and bad ones less likely,
   in a *clipped* step.
5. Update the critic to predict returns better.
6. Repeat.

### 9.2 The pieces, and why each exists

- **Parallel workers (9 processes × 4–8 environments).** DOOM is CPU-bound and
  one episode is slow. Many environments at once produce a batch fast (~500
  steps/s on a 10-core laptop). The learner batches inference on the GPU (MPS)
  and double-buffers: half the workers step while the other half's actions are
  computed.
- **GAE (γ 0.995, λ 0.95).** Turns a sequence of rewards into per-step
  advantages. γ close to 1 means the agent cares about reward far in the future —
  necessary when the exit is hundreds of steps away.
- **Recurrent windows (32 steps).** The GRU needs sequences, so training uses
  32-step slices and resets the hidden state at episode boundaries inside the
  slice.
- **Asymmetric critic.** The value baseline sees 16 privileged features (time
  left, geodesic distance and bearing to the objective, keys held, coverage,
  health, ammo, kills, …). Its features are **detached** — gradients do not flow
  from the critic into the shared encoder. Only the policy loss and regularisers
  shape the acting network.
  *Why:* a critic that cannot see the exit has enormous variance; a critic that
  can see it gives much cleaner advantages. This is standard asymmetric
  actor–critic, not cheating: the critic never influences an action and is not
  even evaluated by the shipped agent.
- **Reward shaping.** Dense terms: exit (+10 + 2·time left), key (+5), lock
  (+3), door (+0.5), new sector (+0.3), new cell (+0.01), newly seen floor
  (+0.004/cell), geodesic progress (+0.003/unit), kill (+0.6); penalties for
  damage, death, stagnation and useless USE. *The seen-floor and stagnation terms
  are what made multi-room maps learnable* — without them the agent parks in a
  corner or presses USE on a locked door forever.
- **Geodesic distance.** True shortest-path distance on the teacher's nav grid,
  toward the exit if reachable with the keys held, otherwise toward the nearest
  needed key. It powers both the shaping and the critic features. Training only.
- **KL anchor.** An exact KL penalty to a reference policy keeps PPO from
  drifting too far from a known-good behaviour. The reference's hidden state is
  tracked along the same trajectories.
- **Teacher imitation term (DAPG-style).** Cross-entropy on random demo windows
  mixed into the PPO loss. It keeps door/key skills grounded while PPO explores.
  This is the safety net against "catastrophic forgetting".
- **Curriculum.** Maps are drawn from the 14 training categories with weight
  `0.6 + (1 − recent success rate)`, plus 15% freshly generated maps. The agent
  spends more time on what it is bad at, and never overfits a fixed map set.

### 9.3 Why three stages?

The shipped weights come from BC, then three PPO runs, each starting from the
previous best checkpoint:

| Stage | Starts from | What changes | Snapshot shipped |
|---|---|---|---|
| A | BC | conservative: lr 1e-4, strong KL to BC (0.3 → 0.02), small entropy; no seen-floor/stagnation terms | — |
| B | A, update 50 | lr 2e-4, KL 0.03, imitation 0.2 → 0.05, adds seen-floor, stagnation and locked-USE terms, key reward 3 → 5 | — |
| C | B, update 100 | as B, KL anchored to A, imitation 0.15 → 0.05, harsher death/damage costs | **update 50 (0.46 M steps)** |

Total: about **1.6 M PPO steps**. Think of it as a curriculum for the *trainer*,
not just the agent: first learn to imitate safely, then explore with dense
guidance, then tighten survival costs without forgetting the skills.

**Why not train PPO from scratch?** Sparse reward + long horizon + laptop compute.
BC supplies the prior; PPO supplies the correction.

---

## 10. Stage 7 — Evaluation and the promotion gate

**What.** Every snapshot is scored on the full validation split through the
**real `DoomAgent`** (`tools/auto_eval.py`), using a 14-category diagnostic suite
(C0–C13) that mirrors the four official tiers.

**Proxy score.** Since the official weights are hidden, we define an explicit
stand-in: exit +10, key +3, lock +3, door +0.5, new sector +0.3 (capped +10),
kill +0.6, damage −0.01/HP, death −(3·time-left + 1). The headline is the
**tier-weighted mean** (weights 1.0 / 1.2 / 1.5 / 2.0), with a stratified
bootstrap 95% CI over maps.

**Promotion gate.** A candidate replaces the current agent only if:

1. the **paired** bootstrap CI of the tier-weighted difference (same maps) is
   entirely above zero, **and**
2. no category regresses by more than 10 points with a CI excluding zero.

Finalists are re-scored on two engine seeds. The winner gets **one** logged look
at the untouched test split.

**Why paired?** Map difficulty varies enormously. Comparing two independent
averages is noisy; comparing per-map differences cancels that noise. It is the
statistical version of "same exam, same questions, different students".

**Why a CI instead of a single number?** A single number cannot tell you whether
a +0.3 improvement is real or luck. A CI says "if we re-ran on a different sample
of maps, the effect would probably still be positive".

---

## 11. The numbers, in one place

| Thing | Value |
|---|---|
| Acting parameters | 2,000,735 (plus a 103k training-only privileged critic) |
| Teacher demonstrations | 13 datasets, 436k steps |
| BC | 128-step windows, 5 epochs, AdamW 3e-4, validation loss 0.63 |
| PPO | 9 workers × 4–8 envs, ~500 steps/s, GAE γ 0.995 λ 0.95, clip 0.1 |
| PPO total | ~1.6 M steps over 3 stages; shipped = stage C, update 50 (0.46 M) |
| Maps | 14 categories; 1,966 train / 136 val / 400 test (seed-disjoint, hash-checked) |
| Inference | 0.69 ms network + ~0.55 ms map; p50 1.6 ms / p99 1.9 ms total |
| Held-out test | tier-weighted proxy +8.48 vs +6.11 previous; paired +2.37 (95% CI +1.47…+3.24) |
| Validation | +8.28 vs +6.11; paired +2.17 (CI +1.23…+3.08) |
| Official validator | PASS, init 0.4 s |

---

## 12. Questions you will be asked (and the plain answer)

**"Why not just train PPO from pixels?"**
Sparse reward, long horizons, and a one-CPU-thread inference budget. A
privileged teacher gives a working policy in under an hour of BC; PPO then fixes
what the omniscient teacher cannot do from the agent's limited view.

**"Isn't the teacher cheating?"**
It never runs at evaluation and `submission/` never imports `training/`. It is a
demonstration generator — the same role a simulator or a human demo plays in any
imitation-learning system.

**"Isn't the map a form of planning?"**
No. It stores observations at a self-estimated pose. There is no search, no
target selection, no rule tree. The network reads the picture and decides.

**"How does it find the exit if the exit isn't in the observations?"**
Three learned routes: the RGB encoder is trained (via auxiliary heads) to
recognise exit signs and estimate bearing/distance; the goal head predicts the
current objective; PPO rewards reaching it. Exploration toward unseen space plus
visual recognition becomes the learned strategy.

**"How does it know which key opens which door?"**
Door and key colours are visible in RGB (trained by the key-colour auxiliary
head); keys and locked doors are remembered in the map's channels; vitals say
which keys are held.

**"Why is the critic privileged? Isn't that cheating?"**
The critic is only a training baseline for variance reduction. It never
influences an action, and the agent does not evaluate it. Standard asymmetric
actor–critic.

**"Why sample actions instead of taking the most likely one?"**
PPO trains a *stochastic* policy. Taking the argmax (most likely button) is a
different, untrained policy that freezes at near-ties — we measured 90%+ no-op
episodes. Sampling is still purely the trained network's output, and the random
generator is re-seeded per map, so replays are reproducible.

**"How do you know you didn't overfit?"**
Seed-disjoint splits, WAD-hash leakage checks, selection on validation only, and
a single logged test look. Test gain (+2.37) is consistent with validation
(+2.17).

**"What is the biggest weakness?"**
Combined-skill maps (ambush, gauntlet, Freedoom, OOD). Exits there need more
samples than ~1.6 M PPO steps on a laptop allow. The agent still earns partial
credit (coverage, kills, doors, keys).

**"What is actually novel?"**
Exact DOOM dead reckoning (turn-run model, per-tick friction, wall-pin clip);
offline ego-map regeneration that removes train/serve skew; a learned goal head
fused back into the policy; an asymmetric critic with geodesic objective
features; seen-floor and stagnation shaping; and a paired statistical promotion
gate over a 14-category suite.

---

## 13. Common confusions (say these correctly)

| Wrong | Right |
|---|---|
| "The teacher is the agent." | The teacher is a training-time data generator; the agent is the 2 M-parameter network. |
| "BC is reinforcement learning." | BC is supervised imitation. PPO is the RL part. |
| "The map is a planner." | The map is perception/memory. It never picks a target. |
| "Sampling means random actions." | Sampling means drawing from the *trained* policy's distribution. |
| "The privileged critic leaks information." | It only shapes training advantages; it is never used to act. |
| "The test score was used to pick the model." | Selection is on validation; test is touched once, logged. |
| "More PPO steps would fix everything." | Compute is the binding constraint (~500 steps/s on a laptop). |

---

## 14. Glossary (alphabetical)

| Term | Definition |
|---|---|
| **A\*** | A shortest-path graph search; used by the teacher, never by the agent. |
| **Action head** | One output distribution for one button group (move, strafe, turn, attack, use, next). Factorised = six separate distributions instead of 2⁹ bit patterns. |
| **Actor** | The part of the network that chooses actions. |
| **Advantage** | Return minus value: how much better an action was than expected. |
| **AdamW** | A standard gradient-based optimiser with weight decay. |
| **Auxiliary head** | Extra training-only prediction task used to shape the encoder. |
| **Batch** | A group of examples processed in one optimiser step. |
| **BC** | Behaviour cloning: supervised imitation of an expert. |
| **Bootstrap CI** | Uncertainty estimate from resampling with replacement. |
| **CNN** | Convolutional network for grid/image inputs. |
| **Critic** | Network head predicting the value function. |
| **Cross-entropy** | Classification loss; low when the correct class gets high probability. |
| **Curriculum** | Training more on what the agent is currently bad at. |
| **Dead reckoning** | Estimating pose by integrating one's own motion. |
| **Discount (γ)** | How much future reward is worth relative to now. |
| **Distribution shift / train-serve skew** | Training inputs differ from deployment inputs; a silent failure mode. |
| **Entropy** | Spread of a probability distribution; bonus encourages exploration. |
| **Episode** | One run through one map. |
| **Epoch** | One full pass over the training data. |
| **GAE** | Generalised advantage estimation; smoothed advantage from a reward run. |
| **GRU** | Recurrent layer with a learned memory vector. |
| **Hidden state** | The GRU's memory between decisions. |
| **Imitation term** | Extra cross-entropy on demos during PPO, to avoid forgetting. |
| **KL divergence** | Measure of difference between two distributions. |
| **Leakage** | Training on test data by accident. |
| **Logits** | Raw scores before softmax. |
| **Loss** | Scalar measuring how wrong the network is. |
| **MPS** | Apple's GPU backend for PyTorch (Metal Performance Shaders). |
| **OOD** | Out-of-distribution: deliberately unfamiliar test maps. |
| **On-policy** | Learning only from data the current policy produced. |
| **Paired comparison** | Compare two agents on the same maps, per-map. |
| **Parameter** | A learned weight in the network. |
| **POMDP** | Partially observable Markov decision process; the formal name for "you cannot see everything". |
| **Policy (π)** | Observation → action-probability rule. |
| **PPO** | Proximal policy optimisation; clipped-step actor–critic RL. |
| **Privileged information** | Training-only information unavailable at evaluation. |
| **Proxy score** | Our explicit stand-in for the hidden official scorer. |
| **Return** | Discounted sum of future reward. |
| **Reward shaping** | Extra dense reward terms to guide learning. |
| **Softmax** | Normalises logits into probabilities. |
| **Supervised learning** | Learning from labelled input→output examples. |
| **Tier weighting** | Harder map categories contribute more to the headline score. |
| **Value function** | Predicted future return from a state; the critic's target. |
| **Window** | Fixed-length slice of an episode used for recurrent training. |

---

## 15. Where to look in the repo

| Path | What it is |
|---|---|
| `submission/agent.py` | The shipped `DoomAgent`: sanitise → EgoMap → network → seeded sampling. |
| `submission/model.py` | Atlas-E network definition (encoders, goal head, GRU, heads). |
| `submission/mapmem.py` | `EgoMap`: dead reckoning + heading-up two-scale map. |
| `training/maps/` | Procedural generator, drills, Freedoom preparation. |
| `training/teacher/` | Privileged expert: nav grid, A\*, door state machine, combat. |
| `training/collect.py` | Teacher demonstrations + auxiliary/goal labels. |
| `training/egomaps.py` | Offline ego-map regeneration, bit-packing. |
| `training/bc.py` | Behaviour cloning. |
| `training/ppo2.py` | Parallel PPO: asymmetric critic, shaping, imitation, KL anchor. |
| `training/evaluate.py` | Episode runner, proxy score, coverage. |
| `training/suite/` | 14-category diagnostic suite and statistics. |
| `tools/auto_eval.py` | Scores snapshots on validation. |
| `validate.py` | Official validator (untouched). |
