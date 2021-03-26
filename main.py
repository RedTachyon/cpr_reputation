#!/usr/bin/env python3

from cpr_reputation.learning import HarvestEnv

from ray.rllib.agents import pg
from ray.tune.registry import register_env
# from ray import tune
from gym.spaces import Discrete

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
        Discrete(
            (2 * defaults_ini["sight_width"] + 1) * defaults_ini["sight_dist"] * 3
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
    }

    trainer = pg.PGTrainer(env="harvest", config=config)

    while True:
        print(trainer.train())
