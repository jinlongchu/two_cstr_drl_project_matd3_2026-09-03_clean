"""Smoke-test the distributed PettingZoo two-CSTR environment."""

from __future__ import annotations

import numpy as np

from two_cstr_env import TwoCSTRParallelEnv


def main() -> None:
    env = TwoCSTRParallelEnv(horizon=25)
    observations, infos = env.reset(seed=123)
    assert set(observations) == {"cstr1", "cstr2"}
    assert observations["cstr1"].shape == (13,)
    assert observations["cstr2"].shape == (13,)
    assert infos["cstr2"]["action_mask"].tolist() == [0.0, 0.0]

    c2_handoff = None
    recycle_activation = None
    for _ in range(25):
        actions = {
            agent: np.zeros(2, dtype=np.float32)
            for agent in ("cstr1", "cstr2")
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        base_info = infos["cstr1"]["base_info"]
        if base_info["c2_handoff"]:
            c2_handoff = (base_info["state"][0], base_info["state"][2])
        if base_info["recycle_event"]:
            recycle_activation = base_info["recycle_active"]
        if all(terminations.values()) or all(truncations.values()):
            break

    assert c2_handoff is not None, "CSTR2 handoff at t=10 s was not observed"
    assert np.isclose(c2_handoff[0], c2_handoff[1], atol=1e-10)
    assert recycle_activation is True, "Recycle did not activate at t=20 s"
    assert infos["cstr1"]["global_state"].shape == (13,)
    print("TwoCSTRParallelEnv smoke test passed")


if __name__ == "__main__":
    main()
