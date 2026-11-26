# Training code

Training-only. The submitted agent (`submission/agent.py`) never imports this
folder. The full pipeline with commands and settings is in
[`../docs/TRAINING.md`](../docs/TRAINING.md).

```
maps/
  gen_wad.py           procedural node-less WADs, tiers 1–4 (arenas, rooms, key→lock, gauntlet)
  drills.py            corridor doors, key alcove, key chain, backtrack layouts
  doorroom.py          door on a room wall, random start pose
  prepare_freedoom.py  Freedoom maps: raw / doors-open / no-monster variants
teacher/
  mapdata.py           WAD geometry, doors, locks, keys, exits
  navgrid.py           16-unit nav grid, A* with door costs and key gating, line of sight
  teacher.py           privileged expert: goal choice, door state machine, combat
  frontier.py          frontier helpers for exploration goals
collect.py             teacher demonstrations + auxiliary and goal labels
egomaps.py             replay demos through EgoMap; bit-pack 16-channel maps
bc.py                  behaviour cloning (--arch ego)
ppo2.py                parallel PPO: asymmetric critic, geodesic shaping, imitation, KL anchor, EMA
evaluate.py            episode runner, proxy score, sector coverage
coverage.py            sector lookup from sector_polygons()
obs.py                 observation flattening, RGB pooling, previous-action vector
splits.py              generator seed ranges and Freedoom splits
suite/                 the 14-category diagnostic suite (see suite/README.md)
```

Earlier-design tools kept for reproducibility but not part of the current
pipeline: `ppo.py` (single-process PPO), `dagger.py`, `moe.py`, `bench.py`,
`suite/hybrid.py`.

## Data layout (gitignored)

```
training/data/demos_*/          demo shards (.npz) + manifest.jsonl; ego/ holds regenerated ego maps
training/data/suite/<split>/    suite WADs + manifest.jsonl per split
training/runs/<run>/            checkpoints, train.jsonl, auto_eval.jsonl
training/runs/suite/<run>/      suite episodes.jsonl per evaluation
```

## Most-used commands

```bash
uv run python -m training.egomaps --data training/data/demos_v3_key            # ego maps for demos
uv run python -m training.bc --arch ego --data <dirs> --out training/runs/bc_ego1
uv run python -m training.ppo2 --init training/runs/bc_ego1/ckpt.pt --out training/runs/ppo_run
uv run python tools/auto_eval.py --run training/runs/ppo_run --baseline <val run>
uv run python -m training.suite.run --agent --primary <ckpt> --combat none --split val \
    --per-category 10 --workers 8 --out training/runs/suite/val_cand
```
