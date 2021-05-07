#!/usr/bin/env python3

from copy import deepcopy
from typing import Optional

import numpy as np
import ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.utils import get_config, ArgParser

args = ArgParser()

env_config, ray_config = get_config(args.ini)

if __name__ == "__main__":

    ray.init()

    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(config, **env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=ray_config,
    )

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        checkpoint_dir = f"ckpnts/{args.ini}/checkpoint_{checkpoint}/checkpoint-{checkpoint}"
        print(f"Pulling checkpoint from {checkpoint_dir}")
        trainer.restore(checkpoint_dir)
    else:
        print("Not pulling from a checkpoint")
        checkpoint = 1

    for iteration in range(checkpoint, 400):
        result_dict = trainer.train()

        print(iteration, result_dict)
        if iteration % 10 == 0:
            trainer.save(f"ckpnts/{args.ini}")
