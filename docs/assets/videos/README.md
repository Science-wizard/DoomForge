# Gameplay videos

Unedited episodes from the shipped hybrid agent on held-out validation maps.
Each clip is recorded through `submission/agent.py` by
[`tools/record_videos.py`](../../../tools/record_videos.py); the map IDs are
validation seeds (10 000+), never training or test.

| Clip | Category | Decisions | Outcome | MP4 |
|---|---|---:|---|---|
| C0 | sanity / visible exit | 64 | exit | [C0_sanity_10000.mp4](C0_sanity_10000.mp4) |
| C5 | key → locked door | 268 | exit, 1 key | [C5_key_lock_10000.mp4](C5_key_lock_10000.mp4) |
| C6 | two keys, two locks | 393 | exit, 2 keys | [C6_two_key_10000.mp4](C6_two_key_10000.mp4) |
| C2 | arena combat | 218 | exit | [C2_arena_10001.mp4](C2_arena_10001.mp4) |
| C3 | corridor door | 73 | exit | [C3_corridor_door_10000.mp4](C3_corridor_door_10000.mp4) |
| C8 | backtrack / key behind spawn | 400 | exit, 1 key | [C8_backtrack_10000.mp4](C8_backtrack_10000.mp4) |

The GIFs in the repository root README are 4 fps previews; the MP4s are 9 fps
(one decision per engine four-tic step) at 480×360.

## Inline players

These render in browsers that allow local `<video>` elements; GitHub's Markdown
sanitizer strips them, so use the MP4 links above there.

<video src="C0_sanity_10000.mp4" controls width="480"></video>
<video src="C5_key_lock_10000.mp4" controls width="480"></video>
<video src="C6_two_key_10000.mp4" controls width="480"></video>
<video src="C2_arena_10001.mp4" controls width="480"></video>
<video src="C3_corridor_door_10000.mp4" controls width="480"></video>
<video src="C8_backtrack_10000.mp4" controls width="480"></video>

`clips.json` contains the exact metadata and paths.
