"""Inference contract stress test (not part of the scorer).

Checks:
  * DoomAgent accepts the official space samples
  * all-zero entity masks, zero RGB, NaN/inf rays and vitals
  * output shape/values and determinism across two identical agents
"""

from __future__ import annotations

import numpy as np

from doom_env.spaces import action_space, observation_space
from submission.agent import DoomAgent

OBS = observation_space()
ACT = action_space()


def blank_obs() -> dict:
    return {
        "rgb": np.zeros((120, 160, 3), np.uint8),
        "depth": np.zeros((120, 160), np.uint8),
        "vitals": np.zeros(17, np.float32),
        "rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)},
        "visible_entities": {"features": np.zeros((32, 5), np.float32),
                             "entity_type": np.zeros(32, np.int64),
                             "mask": np.zeros(32, np.int8)},
    }


def check_action(a, label):
    a = np.asarray(a)
    assert a.shape == (9,), (label, a.shape)
    assert np.all((a == 0) | (a == 1)), (label, a)
    assert a.dtype == np.int8, (label, a.dtype)


def main() -> None:
    agent = DoomAgent(OBS, ACT, "cpu")
    agent.reset()

    # 1. blank observation, all masks zero
    for i in range(5):
        check_action(agent.act(blank_obs()), f"blank/{i}")

    # 2. sampled observations from the official spaces
    rng = np.random.default_rng(0)
    for i in range(20):
        obs = OBS.sample()
        check_action(agent.act(obs), f"sample/{i}")

    # 3. hostile numerics
    obs = blank_obs()
    obs["rays"]["distance"][:] = np.nan
    obs["vitals"][:] = np.inf
    obs["rgb"][:] = 0
    check_action(agent.act(obs), "nan/inf")

    # 4. determinism: two fresh agents, same observation stream
    a1 = DoomAgent(OBS, ACT, "cpu")
    a2 = DoomAgent(OBS, ACT, "cpu")
    a1.reset()
    a2.reset()
    rng = np.random.default_rng(7)
    for i in range(30):
        obs = OBS.sample()
        x = np.asarray(a1.act(obs))
        # rebuild an identical observation (spaces.sample uses its own RNG; feed the same object)
        y = np.asarray(a2.act(obs))
        assert np.array_equal(x, y), (i, x, y)

    # 5. reset clears recurrent state
    a1.reset()
    check_action(a1.act(blank_obs()), "after-reset")
    print("CONTRACT STRESS: PASS")


if __name__ == "__main__":
    main()
