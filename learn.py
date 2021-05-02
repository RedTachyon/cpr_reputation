#!/usr/bin/env python3

from copy import deepcopy
from typing import Optional

import numpy as np
import ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from typarse import BaseParser

from cpr_reputation.environments import HarvestEnv


class Parser(BaseParser):
    heterogeneous: bool = False
    checkpoint: Optional[int]
    # wandb_token: Optional[str]  # TODO: add this somehow at some point?

    _abbrev = {
        "heterogeneous": "het",
        "checkpoint": "c",
        # "wandb_token": "w",
    }

    _help = {
        "heterogeneous": "Flag whether the agents should be heterogeneous (as opposed to default homogeneous)",
        "checkpoint": "From which checkpoint the training should be possibly resumed",
        # "wandb_token": "Path to the wandb token for wandb integration"
    }


env_config = {  # TODO: put configs into a yaml?
    "num_agents": 4,
    "size": (20, 20),
    "sight_width": 5,
    "sight_dist": 10,
    "num_crosses": 4,
}


if __name__ == "__main__":

    ray.init()

    args = Parser()

    walker1 = (
        None,
        Box(
            -1.0,
            1.0,
            (env_config["sight_dist"], 2 * env_config["sight_width"] + 1, 4),
            np.float32,
        ),  # obs
        Discrete(8),  # action
        dict(),
    )

    if args.heterogeneous:
        walkers = {
            f"Agent{k}": deepcopy(walker1) for k in range(env_config["num_agents"])
        }

        multiagent: dict = {
            "policies": walkers,
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    else:
        multiagent: dict = {
            "policies": {"walker": walker1},
            "policy_mapping_fn": lambda agent_id: "walker",
        }

    config = {
        "multiagent": multiagent,
        "framework": "torch",
        "model": {
            "dim": 3,
            "conv_filters": [
                [16, [4, 4], 1],
                [32, [env_config["sight_dist"], 2 * env_config["sight_width"] + 1], 1],
            ],
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 3,
    }

    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(config, **env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=config,
    )

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        checkpoint_dir = f"ckpnts/checkpoint_{checkpoint}/checkpoint-{checkpoint}"
        print(f"Pulling checkpoint from {checkpoint_dir}")
        trainer.restore(checkpoint_dir)
    else:
        print("Not pulling from a checkpoint")
        checkpoint = 1

    print(("Heterogeneous" if args.heterogeneous else "Homogeneous") + " training")
    for iteration in range(checkpoint, 400):
        result_dict = trainer.train()

        print(iteration, result_dict)
        if iteration % 10 == 0:
            trainer.save("ckpnts")
