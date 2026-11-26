# Results — Phase 1 benchmark

All numbers come from `training/bench.py` (fixed bench seeds `30,000+`, never used for
training; 8 maps per generated tier; one engine seed; Freedoom MAP23–32 held out). The
proxy score mirrors the four stated categories with **assumed** weights, because the
official weights are hidden.

```
proxy = 10·exit + 3·key + 3·lock + 0.5·door + 0.3·new_sector (cap 10)
        + 0.6·kill − 0.01·damage − (3·(1−t/T)+1)·death
```

## Checkpoints compared

| Checkpoint | What it is |
|---|---|
| `bc_map2_e0` | Map-aware multimodal BC on nav + arenas; earlier baseline |
| `bc_use` | Drill-only fine-tune from a base checkpoint; learned doors, forgot arenas |
| `bc_head` | Frozen trunk, actor-only fine-tune on drills; cannot learn door approach |
| `bc_arena` | Low-LR fine-tune on 300 extra arena episodes; **shipped** |
| `bc_anchor` | KL-anchored full fine-tune; stopped before checkpoint (inconclusive) |

## Generated tiers

| Suite | `bc_map2_e0` | `bc_use` | `bc_head` | `bc_arena` |
|---|---:|---:|---:|---:|
| t1 arena: exits | 2/8 | 1/8 | 1/8 | **5/8** |
| t1 proxy | +0.35 | +0.89 | −1.53 | **+5.42** |
| t1 kills | 10 | **19** | 5 | 4 |
| t2 rooms: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t2 proxy | −0.99 | −0.26 | −0.22 | **−0.18** |
| t2 coverage | 6.1% | 6.1% | 6.1% | 6.1% |
| t3 key/lock: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t3 proxy | −0.89 | −0.70 | −0.87 | **−0.38** |
| t4 gauntlet: exits | 0/8 | 0/8 | 0/8 | 0/8 |
| t4 proxy | −1.61 | −1.41 | −2.17 | **−0.73** |
| Door drills: exits | 0/8 | **6/8** | 0/8 | 0/8 |
| Freedoom 23–32: exits | 0/10 | not run | not run | 0/10 |
| Freedoom proxy | −0.77 | not run | not run | **+1.00** |
| Freedoom coverage | 1.1% | not run | not run | 3.6% |

Tier-weighted proxy (1, 1.2, 1.5, 2) over t1–t4:
`bc_map2_e0` ≈ **−5.40**, `bc_use` ≈ **−3.29**, `bc_arena` ≈ **+3.17**.
The arena fine-tune is the clear Phase-1 winner; the door branch still wins door drills.

## What the numbers mean

1. **The two-map eval was optimistic.** Tier-1 exits are 2/8 (25%), not 2/2. The hidden
   suite is larger; report the bench, never the small eval.
2. **Doors are learnable in isolation.** 360/360 reliable drill episodes + a weighted
   USE loss took door drills from 0/8 to 6/8 in two epochs.
3. **Door skill does not transfer to rooms.** The door branch still opens 0 doors on
   tier-2 maps; straight-corridor drills do not teach “find a door on a wall”.
4. **Freezing the trunk fails.** Actor-only training cannot map frozen features to door
   approach; 0/8 drills. The trunk must adapt.
5. **Full fine-tuning forgets.** Drill-only training drops tier-1 exits from 2/8 to 1/8
   and loses arena kills/exit behaviour. This is the central Phase-2 problem.
6. **Room-door demonstrations are reliable** (200/200, random start heading). A focused
   arena-anchored merge (`bc_combined`: drills + room-doors, KL anchor to `bc_arena` on
   non-door frames, 1 epoch) was trained and measured: **tier 1 1/8, drills 0/8**. The
   anchor does not preserve the arena policy under this data mix, and the USE head still
   does not generalise to rooms. “Add doors without forgetting” remains the Phase-2
   problem; the KL anchor alone is not the answer.
7. **More arena data helps.** A low-LR fine-tune on 300 extra arena episodes
   (53% teacher exits) raised held-out tier-1 exits from 2/8 to **5/8** and improved every
   proxy.

## Reproduce

```bash
python -m training.bench --ckpt training/runs/bc_map2/ckpt_e0.pt --maps 8
python -m training.bench --ckpt training/runs/bc_use/ckpt.pt --maps 8 --no-freedoom
```

Each run writes `results.json` with every row and the summary. Latency is p99 per map;
the agent is ~1 ms on CPU, well inside the 40 ms/decision budget.

## Caveats

- One training seed and one engine seed. The bench is a point estimate; 8 maps per tier
  has a wide confidence interval.
- Proxy weights are assumptions. The official scorer may weight exit, kills, damage or
  coverage differently.
- Generated maps are one distribution; Freedoom MAP23–32 is the only hand-made
  held-out set and the agent scores near zero there.
- Hidden tiers may include layouts/skills not represented here (teleporters, lifts,
  crushing ceilings, boss monsters).
