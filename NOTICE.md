# Notices and attribution

DoomForge is original work built during Yugaantar DevForge 2026.

## Upstream

- **Starter template:** [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template).
  The `env/` wrapper, `validate.py`, contract and evaluation documentation are used
  unchanged. Our architecture, training pipeline, map generator, teacher and agent are
  new.
- **ViZDoom 1.3.3** — the DOOM AI research platform; the evaluation engine is the
  vanilla bundled binary.
- **Freedoom 2** — BSD-licensed free DOOM assets and maps, bundled with ViZDoom. Used
  as real-map training/evaluation data: MAP01–16 train, MAP17–22 validation, MAP23–32 test.
- **PyTorch, NumPy, Gymnasium** — model and environment plumbing.

## Assets

- `docs/assets/videos/*` are replays recorded from our own trained agent in the
  official wrapper by `tools/record_videos.py`.
- No third-party game assets, textures, music or code are redistributed. The generator
  references Freedoom texture names from the IWAD already bundled with ViZDoom.

## Data

All training data is synthetic (procedurally generated maps) or derived from the
BSD-licensed Freedoom IWAD. No real patient, personal or licensed commercial game data
is used.

## Licence

Our source code is released under the MIT licence. Upstream components retain their own
licences; see the repositories above.
