#!/usr/bin/env python3

import numpy as np
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from gym.spaces import Discrete, Box

from cpr_reputation.environments import HarvestEnv


def path_from_ini(x: dict) -> str:
    return "".join(f"{key}={value}" for key, value in x.items())


defaults_ini = {
    "num_agents": 4,
    "size": (20, 20),
    "sight_width": 5,
    "sight_dist": 10,
    "num_crosses": 4,
}

if __name__ == "__main__":
    register_env("harvest", lambda config: HarvestEnv(config, **defaults_ini))

    walker1 = (
        None,
        Box(
            0.0,
            1.0,
            (defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1, 3),
            np.float32,
        ),  # obs
        Discrete(8),  # action
        dict(),
    )
    # walkers = {f"Agent{k}": walker1 for k in range(defaults_ini["num_agents"])}
    config = {
        "multiagent": {
            "policies": {"walker1": walker1},
            "policy_mapping_fn": lambda agent_id: "walker1",
        },
        "framework": "torch",
        "model": {
            "dim": 3,
            "conv_filters": [
                [16, [4, 4], 1],
                [
                    32,
                    [defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1],
                    1,
                ],
            ],
        },
    }

    ray.init()
    trainer = ppo.PPOTrainer(
        env="harvest",
        config=config,
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
    )

    results = list()
    while True:
        result_dict = trainer.train()
        results.append(result_dict)
        print(result_dict)
