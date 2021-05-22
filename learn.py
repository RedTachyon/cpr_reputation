#!/usr/bin/env python3
import os
from copy import deepcopy
from typing import Optional

import numpy as np

import ray
from gym.spaces import Box, Discrete
from ray.rllib.agents import ppo
from ray.tune import tune
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

import yaml

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.metrics import CPRCallbacks

from typarse import BaseParser


class ArgParser(BaseParser):
    config: str
    name: str = "run_name"
    iters: int = 1
    checkpoint_freq: int = 10
    checkpoint_path: str = None

    _abbrev = {
        "name": "n",
        "config": "c",
        "iters": "i",
        "checkpoint_freq": "f",
        "checkpoint_path": "p",
    }

    _help = {
        "name": "Name of the run for checkpoints",
        "config": "Path to the config of the experiment",
        "iters": "Number of training iterations",
        "checkpoint_freq": "How many training iterations between checkpoints. "
        "A value of 0 (default) disables checkpointing.",
        "checkpoint_path": "Which checkpoint to load, if any",
    }


if __name__ == "__main__":

    args = ArgParser()
    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    register_env(
        "CPRHarvestEnv-v0", lambda ray_config: HarvestEnv(env_config, ray_config)
    )

    ray_config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))

    # Fill out the rest of the ray config
    walker_policy = (
        None,
        Box(
            0.0,
            1.0,
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
    ray_config["env"] = "CPRHarvestEnv-v0"
    ray_config["env_config"] = env_config

    name = args.name
    iters = args.iters
    checkpoint_freq = args.checkpoint_freq
    if args.checkpoint_path:
        checkpoint_path = os.path.expanduser(args.checkpoint_path)
        print(f"loading checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    ray.init()
    results = tune.run(
        "PPO",
        name=name,
        config=ray_config,
        stop={"training_iteration": iters},
        checkpoint_freq=checkpoint_freq,
        restore=checkpoint_path,
    )

    ray.shutdown()

    # trainer = ppo.PPOTrainer(env="CPRHarvestEnv-v0", config=ray_config)
    #
    # for iteration in range(args.iters):
    #     result_dict = trainer.train()
    #
    #     if run_config["verbose"]:
    #         print(f"Iteration: {iteration}", end="\t")  # , result_dict)
    #     if iteration % 10 == 0:
    #         trainer.save(f"ckpnts/{args.name}")
