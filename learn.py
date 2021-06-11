#!/usr/bin/env python3
import os
from copy import deepcopy

# from typing import Optional

import numpy as np
import ray
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents import ppo
from gym.spaces import Box, Discrete

# from ray.rllib.agents import ppo
from ray.tune import tune

# from ray.tune.logger import UnifiedLogger
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env
import yaml
from typarse import BaseParser

# import wandb

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.metrics import CPRCallbacks
from cpr_reputation.recording import HarvestRecorder


class ArgParser(BaseParser):
    config: str
    name: str = "run_name"
    iters: int = 1
    checkpoint_freq: int = 10
    checkpoint_path: str = None
    wandb_project: str = "quinn-workspace"

    _abbrev = {
        "name": "n",
        "config": "c",
        "iters": "i",
        "checkpoint_freq": "f",
        "checkpoint_path": "p",
        "wandb_project": "w",
    }

    _help = {
        "name": "Name of the run for checkpoints",
        "config": "Path to the config of the experiment",
        "iters": "Number of training iterations",
        "checkpoint_freq": "How many training iterations between checkpoints. "
        "A value of 0 (default) disables checkpointing.",
        "checkpoint_path": "Which checkpoint to load, if any",
        "wandb_project": "What project name in wandb?",
    }


if __name__ == "__main__":

    args = ArgParser()
    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(env_config))

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

    base_ray_config = {
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "multiagent": multiagent,
        "model": {
            "dim": 3,
            "conv_filters": [
                [16, [4, 4], 1],
                [32, [env_config["sight_dist"], 2 * env_config["sight_width"] + 1], 1],
            ],
        },
        "callbacks": CPRCallbacks,
        "env": "CPRHarvestEnv-v0",
        "evaluation_interval": 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "record_env": "videos",
            "render_env": True,
        }
    }

    full_ray_config = {**ray_config, **base_ray_config}

    name = args.name
    iters = args.iters
    checkpoint_freq = args.checkpoint_freq
    if args.checkpoint_path:
        checkpoint_path = os.path.expanduser(args.checkpoint_path)
        print(f"Will load checkpoint {checkpoint_path}")
    else:
        checkpoint_path = None

    ray.init()
    results = tune.run(
        "PPO",
        name=name,
        config=full_ray_config,
        stop={"training_iteration": iters},
        checkpoint_freq=checkpoint_freq,
        # restore=checkpoint_path,
        callbacks=[
            WandbLoggerCallback(
                project=args.wandb_project,
                api_key_file=run_config["wandb_key_file"],
                monitor_gym=True,
                entity="marl-cpr",
            )
        ],
    )

    # I
    final_checkpoint = results.get_last_checkpoint()

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=full_ray_config,
    )

    recorder = HarvestRecorder(
        trainer, env_config, run_config["heterogeneous"], final_checkpoint
    )

    recorder.record()

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
