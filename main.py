#!/usr/bin/env python3

from cpr_reputation.learning import HarvestEnv

import numpy as np
import ray
from ray.rllib.agents import ppo
# from ray.rllib.models.utils import get_filter_config
from ray.tune.registry import register_env
# from ray import tune
from gym.spaces import Discrete, Box

defaults_ini = {
    "num_agents": 4,
    "size": (20, 20),
    "sight_width": 5,
    "sight_dist": 10,
    "num_crosses": 4,
}

if __name__ == "__main__":
    register_env("harvest", lambda config: HarvestEnv(config, **defaults_ini))

    # tune.run("harvest", {"framework": "torch"})
    walker1 = (
        None,
        Box(
            0.,
            1.,
            (defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1, 3),
            np.float32
        ),  # obs
        Discrete(8),  # action
        dict(),
    )
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
                [32, [defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1], 1],
            ]
        },
    }
    ray.init()
    trainer = ppo.PPOTrainer(env="harvest", config=config)

    while True:
        print(trainer.train())
