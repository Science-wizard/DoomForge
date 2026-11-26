# Gameplay videos

Unedited episodes from the shipped agent on held-out validation maps (seeds
10 000+, never used for training). Each clip is recorded through
`submission/agent.py` by [`tools/record_videos.py`](../../../tools/record_videos.py),
which picks the first validation map of each category that the agent exits.

| Category | Map | Decisions | Outcome | MP4 |
|---|---|---:|---|---|
| C7 multi-room exploration | C7_exploration_10000 | 915 | exit | [MP4](C7_exploration_10000.mp4) |
| C6 two keys, two locks | C6_two_key_10001 | 215 | exit, 2 keys | [MP4](C6_two_key_10001.mp4) |
| C4 door on a room wall | C4_room_door_10000 | 201 | exit | [MP4](C4_room_door_10000.mp4) |
| C5 key → locked door | C5_key_lock_10000 | 338 | exit, 1 key | [MP4](C5_key_lock_10000.mp4) |
| C8 key behind the spawn | C8_backtrack_10000 | 144 | exit, 1 key | [MP4](C8_backtrack_10000.mp4) |
| C2 arena combat | C2_arena_10001 | 125 | exit, 3 kills | [MP4](C2_arena_10001.mp4) |

MP4s play at 9 fps (one frame per decision) at 480×360. The GIFs used in the
README are subsampled 256×192 previews. `clips.json` has the exact metadata.

## Inline players

These render where local `<video>` elements are allowed (GitHub's Markdown
strips them; use the links above there).

<video src="C7_exploration_10000.mp4" controls width="480"></video>
<video src="C6_two_key_10001.mp4" controls width="480"></video>
<video src="C4_room_door_10000.mp4" controls width="480"></video>
<video src="C5_key_lock_10000.mp4" controls width="480"></video>
<video src="C8_backtrack_10000.mp4" controls width="480"></video>
<video src="C2_arena_10001.mp4" controls width="480"></video>
