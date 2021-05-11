#!/usr/bin/env python3
from copy import deepcopy
from typing import Optional

import torch
import numpy as np

import ray
from gym.spaces import Box, Discrete
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

import yaml

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.metrics import CPRCallbacks

from typarse import BaseParser


class ArgParser(BaseParser):
    name: str
    config: str
    iters: int

    _abbrev = {
        "name": "n",
        "config": "c",
        "iters": "i",
    }

    _help = {
        "name": "Name of the run for checkpoints",
        "config": "Path to the config of the experiment",
        "iters": "Number of training iterations",
    }


if __name__ == "__main__":

    cuda = torch.cuda.is_available()

    ray.init()

    args = ArgParser()

    # Load the configs

    with open(args.config, "r") as f:
        config = yaml.load(f.read())

    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    config["num_gpus"] = 1 if cuda else 0

    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(env_config))

    # Fill out the rest of the ray config
    walker_policy = (
        None,
        Box(
            0.,
            1.,
            (env_config["sight_dist"], 2 * env_config["sight_width"] + 1, 4),
            np.float32,
        ),  # obs
        Discrete(8),  # action
        dict(),
    )

    if run_config["heterogeneous"]:
        multiagent = {
            "policies": {
                f"Agent{k}": deepcopy(walker_policy)
                for k in range(env_config["num_agents"])
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    else:
        multiagent = {
            "policies": {"walker": walker_policy},
            "policy_mapping_fn": lambda agent_id: "walker",
        }

    ray_config["multiagent"] = multiagent
    ray_config["model"] = {
        "dim": 3,
        "conv_filters": [
            [16, [4, 4], 1],
            [32, [env_config["sight_dist"], 2 * env_config["sight_width"] + 1], 1],
        ],
    }

    ray_config["callbacks"] = CPRCallbacks

    trainer = ppo.PPOTrainer(env="CPRHarvestEnv-v0", config=ray_config)

    for iteration in range(args.iters):
        result_dict = trainer.train()

        if run_config["verbose"]:
            print(f"Iteration: {iteration}", end="\t")  # , result_dict)
        if iteration % 10 == 0:
            trainer.save(f"ckpnts/{args.name}")
