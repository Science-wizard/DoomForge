# DoomForge: presentation script

About **10 minutes** of speaking, then Q&A. Each section has a suggested slide, the
spoken script, and the one point the audience must remember. Figures referenced
are in `docs/brief/assets/` and the gameplay clips in `docs/assets/videos/`
(full-length episodes: `docs/assets/videos/full/`).

Pacing: roughly 140 words per minute. Times in brackets are cumulative.

---

## 1. Hook (0:00 – 0:45)

**Slide:** the C7 exploration clip playing full screen (`docs/assets/videos/C7_exploration_10000.mp4`), title "DoomForge" in a corner.

> What you're watching is a level this agent has never seen. It has no map, no
> position and no idea where the exit is. It opens six doors, works out where
> it has already been, and reaches the exit in 915 decisions.
>
> The whole agent is one neural network with two million parameters. It runs on a
> single CPU thread and takes 1.6 milliseconds per decision, against a budget of 40.
>
> We're going to show you how we built it, why each piece is there, and how we
> know it works.

**Remember:** one small network plays unseen levels end to end.

---

## 2. The problem, as we read it (0:45 – 2:00)

**Slide:** the observation contract as four icons: 64 rays, 32 entities, 17 vitals, a 120×160 image. Below: "no position · no exit marker · no key colours".

> We started by reading the observation contract very carefully, because it
> decides everything.
>
> The agent gets 64 range-finder rays, up to 32 visible objects, 17 numbers such as
> health, ammo and key flags, and a small colour image. Three things are
> missing, and they shaped the whole design.
>
> First, there is **no position**. Everything is relative to the player, so to
> explore you have to remember where you've been, and nothing gives you that.
>
> Second, the **exit isn't in the structured data**. No ray type or entity marks
> it. The only way to find it is to see it in the pixels.
>
> Third, **key and lock colours are hidden**. Every key is just "key" and every lock
> just "locked door". Colour only exists in the image.
>
> On top of that: one CPU thread at evaluation, sparse rewards hundreds of steps
> away, and a rule that every action must come from trained parameters. No
> path-finding, no hand-written rules.

**Remember:** the core problem is memory and perception, not reflexes.

---

## 3. How we got here (2:00 – 3:30)

**Slide:** a three-step timeline: "hybrid of experts → measure everything → one network with real memory".

> Our first working system was a hybrid: a navigation network, a separate combat
> network and a door specialist, switched by a router. It worked on simple maps
> and it taught us a lot, but it had a ceiling. Every new skill meant another
> expert and another rule to choose between them.
>
> So we built a measurement system before building more models: a 14-category
> benchmark, from "walk to a visible exit" up to multi-room exploration, ambushes
> and hand-made Freedoom maps, with separate training, validation and test maps.
>
> Measuring carefully turned up two root causes that every model had shared.
>
> The first was the **memory itself**. Our map relies on the agent working out its
> own position from its movements. We compared that against the true position and
> found it drifting badly. Doom keeps accelerating a turn even when you reverse
> direction, and it reports momentum while you're pressed against a wall. We
> modelled both exactly. On the agent's own episodes that cut drift by 2.3×, and
> heading error dropped to almost zero.
>
> The second was in our **RL trainer**. The returns weren't being bootstrapped at
> the end of each rollout, and time-outs were being treated as deaths. Fixing
> those made reinforcement learning actually work.
>
> With correct memory and correct training, we could replace three networks and a
> router with one network that does everything.

**Remember:** we measured before we scaled, and the measurements changed the design.

---

## 4. Architecture decisions (3:30 – 6:00)

**Slide 4a:** the page-1 brief figure: "It plays" next to "It remembers".

> This is the heart of DoomForge. On the left is the real path through the level.
> On the right is the map the agent built for itself in the same episode, using
> only its own observations and movements.
>
> Every step, it updates that map from its rays and the objects it sees. The map
> has eight channels: free floor, walls, doors, locked doors, items, keys, its own
> trail, and the frontier of unexplored space.
>
> **Decision one: the map is drawn heading-up.** We rotate it so the agent always
> faces up, and read it at two scales: a fine 2,000-unit window and a coarse
> 8,000-unit one. That means "unexplored doorway ahead and to the left" looks the
> same no matter which way the agent faces, which is a pattern a small
> convolutional network learns easily.
>
> Importantly, this is perception, not planning. The map stores what was seen.
> It never picks a target or a route. The network decides.

**Slide 4b:** the architecture diagram from page 1 of the brief.

> **Decision two: a dedicated encoder for each input type.**
> - Rays go through a *circular* convolution, because ray 63 sits right next to
>   ray 0. Circular padding keeps that wrap-around.
> - The entity list goes through *self-attention*. It has a variable length and no
>   natural order, so attention handles anywhere from zero to 32 monsters.
> - The image goes through a small colour CNN, shrunk to 30 by 40 pixels. That's
>   enough to read exit signs and lock colours, and it's cheap.
>
> **Decision three: a learned goal.** A small head predicts where the agent should
> be heading (the exit, a key or unexplored space) and feeds that back in as an
> input. During training it learns from the teacher's goal; at evaluation it
> predicts the goal from observations alone.
>
> **Decision four: memory over time.** A 384-unit GRU carries what happened
> earlier: "I already have the red key", "that door is still opening".
>
> **Decision five: factorised actions.** Instead of nine independent buttons, there
> are six heads (move, strafe, turn, attack, use, weapon), giving 216 legal
> combinations. The network can never press forward and back at the same time.

**Slide 4c:** the four gameplay shots with their tags.

> We also train auxiliary heads that predict things the observation hides: is the
> exit visible, which direction, how far, which key colours are in view. They're
> never used to act, but they force the network to *represent* those things. Here
> you can see the network seeing the exit with full confidence, reading a blue key
> at 94%, and tracking five monsters at once.

**Remember:** give a small network memory it can read, and train it to see what the observations hide.

---

## 5. Training (6:00 – 7:45)

**Slide:** the pipeline strip from page 2 of the brief, plus the learning-curve chart.

> Training has two phases: learn from an expert, then learn from the game.
>
> **The expert.** We wrote our own level generator that writes real DOOM map files:
> arenas, multi-room layouts, key-and-lock progressions and gauntlets. A training-
> only teacher with full map access plays them, using path-finding, a door state
> machine and circle-strafing combat. That gave us 436,000 labelled steps. The
> teacher is never part of the submitted agent.
>
> **Behaviour cloning.** The student copies the teacher. One detail matters a lot:
> we replay every demonstration through the *exact* map code the agent ships with,
> so training inputs match what the agent will actually see.
>
> **Then PPO.** Reinforcement learning on 72 game instances at once. Three choices
> made it work:
> - an **asymmetric critic**: the value estimate gets privileged training-only signals
>   such as true distance to the goal and time left, while the policy only ever sees the
>   real observations;
> - **dense rewards** for exits, keys and doors, plus two terms that unlocked
>   exploration: a reward for revealing new floor and a cost for standing still;
> - a **teacher-imitation term** that keeps door and key skills from being forgotten
>   while the agent explores.
>
> The chart shows skills climbing across the three training stages. Multi-room
> exploration only appears once the agent is rewarded for revealing new floor.

**Remember:** imitation gives a strong start; PPO teaches the agent to act on its own view.

---

## 6. Results (7:45 – 9:15)

**Slide:** the results row from page 2: the +8.48 score panel, "Biggest gains", "Explores the hardest maps".

> We evaluate on held-out test maps the agent never trained on, and we touched
> that test set once per decision, logged.
>
> On the tier-weighted score, DoomForge reaches **+8.48 against +6.11** for our
> earlier design. That's 39% higher, and the paired 95% confidence interval runs
> from +1.5 to +3.2, so the gain is real, not noise.
>
> It solves five skill categories on every unseen test map. The biggest jumps come
> on the hard skills: keys behind the spawn, doors on room walls, duels and
> multi-room exploration. On the hardest tiers it explores three to four times more
> of each map.
>
> And it does all this in 1.6 milliseconds per decision on one CPU thread.

**Slide (optional):** the skill replays strip from page 2.

> These are real replays: two keys and two locks in the right order, a door found
> from a random starting angle, a key fetched from behind the spawn, and an arena
> cleared on the way to the exit.

**Remember:** a large, statistically solid gain on maps it has never seen.

---

## 7. Tech stack and close (9:15 – 10:00)

**Slide:** tech stack logos/list.

> Our stack: **Python 3.13** and **PyTorch**, trained on Apple's **MPS** GPU, with
> **ViZDoom** and the official Gymnasium wrapper for the environment. **NumPy**
> for the map, **uv** to manage the environment, **Git LFS** for weights. We wrote
> our own procedural WAD generator, the training teacher, a parallel PPO trainer
> and a statistical evaluation suite. No reinforcement-learning library.
>
> To sum up: DoomForge is one small network that sees, remembers and acts. It
> builds its own map, learns from an expert, then teaches itself, and it was
> measured honestly on levels it has never seen.
>
> Thank you. We'd love to take your questions.

---

## Q&A preparation

**"Isn't the teacher cheating?"**
The teacher only generates training data. It never runs at evaluation, and the
submitted code doesn't import it. It plays the same role a human demonstration
plays in any imitation-learning system.

**"Isn't the map a form of path-finding?"**
No. It stores what the agent has seen, at a position the agent computed itself.
There's no search, no target selection and no route. The network looks at the
picture and decides what to do.

**"How does it find an exit it can't see in the data?"**
Three ways: the colour encoder is trained to recognise exit signs and their
direction; the goal head predicts where to go next; and PPO rewards reaching the
exit, so exploring toward unseen space becomes the learned strategy.

**"How does it match keys to doors?"**
It sees key and door colours in the image (trained by the key-colour auxiliary
head), remembers keys and locked doors in its map, and its vitals tell it which
keys it holds.

**"Why sample actions instead of picking the most likely one?"**
PPO trains a stochastic policy. Always taking the top action is effectively a
different, untrained policy that freezes when two options are almost tied.
Sampling is still purely the network's output, and the random generator is
re-seeded every map, so every run is reproducible. It's worth about +1.4 points.

**"Why one network instead of specialists?"**
We tried specialists first. Routing between them needed hand-written rules,
and the combat specialist actually hurt the arena score once the unified network
had learned to fight (80% → 20% when routed).

**"How do you know you didn't overfit your own benchmark?"**
Training, validation and test maps come from disjoint seeds, and every map file is
hashed to prove there's no overlap. Models were selected on validation only; test
was touched once per decision and logged. The test gain (+2.37) matches validation (+2.17).

**"What would you do with more time?"**
Train longer. We ran about 1.6 million PPO steps on a laptop at around 500 steps
per second, and the hardest maps (ambushes, gauntlets, hand-made Freedoom levels)
need far more experience. The architecture is ready; it's a compute question.

**"Why PPO and not DQN or SAC?"**
The action space is discrete and factorised, the policy is recurrent, and PPO is
stable with recurrent state and changing curricula. Off-policy replay is awkward
with a GRU and a non-stationary map distribution.

**Numbers to have ready**

| | |
|---|---|
| Parameters | 2.0 M (acting) |
| Decision time | 1.6 ms p50, 2.0 ms p99, one CPU thread (budget 40 ms) |
| Test score | +8.48 vs +6.11; paired +2.37, 95% CI +1.47 … +3.24 |
| Validation | +8.28 (two seeds) |
| Demonstrations | 436 k steps, 13 datasets |
| PPO | ~1.6 M steps, 72 parallel environments, 3 stages |
| Maps | 1,966 train / 136 validation / 400 test, 14 categories |
| Odometry | 2.3× less drift than naive integration (99 vs 225 units / 500 decisions) |
