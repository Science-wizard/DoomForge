# Training pipeline

This folder is **training-only**. The submitted agent (`submission/agent.py`) never
imports it; the scorer mounts the repo read-only and runs the agent in a sandbox with
no engine reference.

```
maps/          procedural binary maps, door drills, Freedoom preparation
teacher/       privileged expert (A*/frontier), map parser, navigation grid
collect.py     demonstrations + privileged exit/key/door labels
dagger.py      student-state relabelling with teacher actions
bc.py          behaviour cloning (windowed episodes)
ppo.py         recurrent PPO fine-tuning
evaluate.py    proxy leaderboard, sector coverage, replays
coverage.py    sector lookup from sector_polygons()
splits.py      frozen train/val/OOD/Freedoom splits
obs.py         observation flattening, RGB pooling, map/pose
```

## Map supply

```bash
# 32 Freedoom maps -> raw / doors-open / doors-open+no-monsters
python -m training.maps.prepare_freedoom

# procedural map (tier 1..4), node-less Doom binary
python -m training.maps.gen_wad --tier 3 --seed 42 --out maps/generated

# door/key drill
python -c "from training.maps.drills import generate_drill; generate_drill(0,'maps/generated',n_doors=2,key=True)"
```

Generated maps are deterministic in `(seed, tier, flags)`. Seeds 0–9,999 train,
10,000–10,999 validation, 20,000+ OOD; Freedoom MAP01–22 train, MAP23–32 held out
(`training/splits.py`).

## Demonstrations

```bash
python -m training.collect --episodes 400 --source gen --no-doors --no-monsters
python -m training.collect --episodes 150 --source gen --tier 1
python -m training.collect --episodes 120 --source drill --drill-doors 1
python -m training.collect --episodes 120 --source drill --drill-doors 1 --drill-key
python -m training.collect --episodes 120 --source drill --drill-doors 2
```

Each shard stores rays, entities, vitals, 30×40 RGB, previous action, a 5×32×32 map
crop, a 6-value pose, the teacher's 9-button action, and auxiliary labels (exit
visible/bearing/distance, key colours, keys, enemy, door, damage-next-8).

The map crop is built by `submission/mapmem.py` from observations and actions only:
heading from the measured turn table, translation from ego velocity, rays/items
projected into a 64×64 allocentric grid. It is perception, not planning.

## Behaviour cloning

```bash
python -m training.bc \
  --data training/data/demos_all,training/data/demos_drills \
  --drill-repeat 3 --epochs 4 --batch 12 --device cpu
```

Loss = six-head cross-entropy + 0.2 × auxiliary BCE/CE + 0.5 × value MSE. Episodes
are split into overlapping 128-step windows for memory efficiency; the GRU sees the
episode context through the map and the window.

## DAgger

```bash
python -m training.dagger --ckpt training/runs/bc_door/ckpt.pt \
  --episodes 120 --source gen --no-doors --no-monsters
```

Runs the student, relabels every visited state with the teacher, and writes the same
shard format so it can be mixed into BC.

## PPO (Phase 5: `ppo2.py`)

```bash
python -m training.egomaps --data training/data/demos_v2_nav,...   # ego maps for BC
python -m training.bc --arch ego --data ... --out training/runs/bc_ego1
python -m training.ppo2 --init training/runs/bc_ego1/ckpt.pt --out training/runs/ppo_ego1 \
    --workers 9 --envs-per-worker 6 --minibatch-windows 72
```

Parallel envs, bootstrapped GAE, asymmetric privileged critic, geodesic
progress shaping, learning-progress curriculum, exact KL anchor. Logs per-episode
return and per-category exit rates to `train.jsonl`. The legacy trainer below
is kept for reference; it has known bugs (no bootstrap, cumulative return log).

## PPO (legacy)

```bash
python -m training.ppo --init training/runs/bc_door/ckpt.pt --updates 200
```

Eight sequential envs, 128-step rollouts, GAE(0.95), clipped surrogate, stored GRU
state, 32-step update windows, curriculum by progress. Reward: exit +10, key +3,
locked door +3, door +0.5, new cell +0.01, kill +0.6, damage −0.01/HP, early-death
penalty, objective potential shaping. Scale to worker processes only after profiling.

## Evaluation

```bash
python -m training.evaluate --ckpt training/runs/bc_door/ckpt.pt --suite gen_val
python -m training.evaluate --ckpt ... --suite freedoom_eval
python -m training.evaluate --ckpt ... --suite all
```

Reports exit, keys, locks, doors, unique sectors, coverage, kills, damage, survival,
steps, proxy components and p50/p99 latency. `--record` stores frames for a GIF.

## Notes

- Training never touches the held-out splits.
- The teacher uses `privileged_state()` and static WAD geometry; it is not part of the
  submission.
- Checkpoints and data are gitignored; only `submission/weights/*.pt` is committed (via
  Git LFS).
