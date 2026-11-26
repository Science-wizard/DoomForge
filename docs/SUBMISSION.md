# Submission Guide

You submit the URL of a **public GitHub repository** created from this template.

## Checklist

- [ ] The repository is **public**, and the URL was sent to organizers before the deadline.
- [ ] `submission/agent.py` defines `DoomAgent` with `__init__`, `reset`, `act` ([CONTRACT.md](CONTRACT.md#6-agent-interface)).
- [ ] Weights are committed under `submission/` (e.g. `submission/weights/`) with **Git LFS**.
- [ ] Total size, including LFS objects, is **≤ 1 GB**.
- [ ] Extra Python packages (only ones not already in the base image) are listed in `submission/requirements.txt` with pinned versions.
- [ ] `python validate.py` passes on your final commit. We recommend `python validate.py --docker`.
- [ ] Training code is in the repo (e.g. `training/`).
- [ ] Your agent does not need network access or write access to the repository at runtime.

## Git LFS

```bash
git lfs install
git lfs track "*.pt" "*.pth" "*.safetensors" "*.ckpt" "*.npz" "*.msgpack" "*.onnx"   # already in .gitattributes
git add .gitattributes submission/weights/model.pt
git commit -m "Add weights"
git push
```

Check that LFS picked up the weights: `git lfs ls-files` should list them.

> **Size tip:** the scorer fetches only the evaluated commit (no history) and only the LFS objects
> it references, so old weight versions don't count. Weights committed
> **without** LFS still count, and they also bloat your history. Use LFS from the start.

## Base image and `requirements.txt`

The official image already includes Python 3.13, `vizdoom`, `gymnasium`, `torch`, `jax`, `flax`,
`numpy`, `scipy`, `opencv-python-headless` and `doom_env`. Exact versions of everything are in
[`docker/requirements.lock`](../docker/requirements.lock).

If you need anything else, list it in `submission/requirements.txt`, one `package==version` per line
(no `-r`/`-e`/`--index-url`/`--find-links` options, URLs or paths; at most 50 packages). These
packages are installed **before** evaluation (with network, under a time limit), and evaluation then
runs **without** network. If the file is invalid or the install fails, the agent can't start: you
get 0 on every map.

## Local validator

```bash
python validate.py                 # quick check in your local environment
python validate.py --docker        # same checks inside the official image (recommended)
```

It checks:

1. Repository size (≤ 1 GB, including LFS objects), and that nothing under `submission/` is
   ignored by `.gitignore`, untracked or uncommitted. The scorer evaluates your **last pushed
   commit**, while the validator runs your working tree, so these would differ.
2. `DoomAgent` imports cleanly and can be called as `DoomAgent(observation_space, action_space, device)`,
   `reset()`, `act(observation)`. `requirements.txt` follows the rules above.
3. A full episode on the test map (or your own with `--map file.wad:MAP01`, repeatable) runs without crashing and returns valid actions.
4. Init time, per-step latency, and per-map agent compute time are within the [timeouts](EVALUATION.md#timeouts),
   measured with your agent **pinned to one CPU core** and numeric libraries limited to one thread
   (`OMP_NUM_THREADS=1` etc.), as at evaluation.
5. Peak RAM (≤ 8 GB) and peak GPU memory (≤ 2 GB) are within the [per-agent limits](EVALUATION.md#hardware).

## Final artifact (Phase 3)

- `submission/weights/doomforge.pt` — Atlas-G v1 (1.87 M parameters, goal head +
  frontier map). Navigation, exploration, keys and doors.
- `submission/weights/combat.pt` — frozen Phase-2 PPO-25 (1.81 M parameters).
  Selected whenever an enemy is visible; both nets see every observation.
- `submission/weights/specialist.pt` — corridor-door expert (1.87 M), selected
  only behind the conservative narrow-corridor gate described in
  `docs/ARCHITECTURE.md` §5.5.
- `submission/agent.py` — loads both, routes with a 16-decision enemy-visibility
  hysteresis, and applies an observation-only anti-freeze reflex outside combat.
- `submission/model_v1.py` — the v1 network definition needed to deserialize the
  combat checkpoint.

If only `doomforge.pt` is present the agent runs single-model; if only a v1
checkpoint is present it also runs single-model. The hybrid path is optional and
fails back safely. Validator: PASS, p50 1.7 ms / p99 2.0 ms on one CPU thread.
