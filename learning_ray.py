#!/usr/bin/env python3

import io
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import ray
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from gym.spaces import Discrete, Box

from cpr_reputation.environments import HarvestEnv


def make_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--geneity",
        default="hom",
        help="select `hom` for homogenous training, select `het` for heterogenous training",
    )
    parser.add_argument(
        "--checkpoint-no",
        default=1,
        type=int,
        help="select an integer pointing to the checkpoint you want",
    )
    return parser


def path_from_ini(x: dict) -> str:
    return "".join(f"{key}={value}" for key, value in x.items())


# with io.open("WANDB_TOKEN", "r") as file:
#     WANDB_TOKEN = file.read()

defaults_ini = {
    "num_agents": 4,
    "size": (20, 20),
    "sight_width": 5,
    "sight_dist": 10,
    "num_crosses": 4,
}

walker1 = (
    None,
    Box(
        -1.0,
        1.0,
        (defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1, 4),
        np.float32,
    ),  # obs
    Discrete(8),  # action
    dict(),
)

# build agent policies
parser = make_arguments()
args = parser.parse_args()

if args.geneity == "hom":
    multiagent: dict = {
        "policies": {"walker": walker1},
        "policy_mapping_fn": lambda agent_id: "walker",
    }
elif args.geneity == "het":
    walkers = {
        f"Agent{k}": deepcopy(walker1) for k in range(defaults_ini["num_agents"])
    }
    multiagent: dict = {
        "policies": walkers,
        "policy_mapping_fn": lambda agent_id: agent_id,
    }
else:
    raise ValueError("Invalid argument supplied to --geneity")

config = {
    "multiagent": multiagent,
    "framework": "torch",
    "model": {
        "dim": 3,
        "conv_filters": [
            [16, [4, 4], 1],
            [32, [defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1], 1],
        ],
    },
    "train_batch_size": 4000,
    "sgd_minibatch_size": 1000,
    "num_sgd_iter": 3,
}


with io.open("WANDB_TOKEN", "r") as file:
    WANDB_TOKEN = file.read()

checkpoint_dir = (
    f"ckpnts/checkpoint_{args.checkpoint_no}/checkpoint-{args.checkpoint_no}"
)

ray.init()

register_env("harvest", lambda config: HarvestEnv(config, **defaults_ini))

trainer = ppo.PPOTrainer(
    env="harvest", logger_creator=lambda cfg: UnifiedLogger(cfg, "log"), config=config,
)


if __name__ == "__main__":
    try:
        trainer.restore(checkpoint_dir)
    except FileNotFoundError:
        print(f"NOT pulling checkpoint from {checkpoint_dir}")
    else:
        print(f"Pulling checkpoint from {checkpoint_dir}")
    finally:
        print(f"geneity: `{args.geneity}`")
        for iteration in range(args.checkpoint_no, 400):
            result_dict = trainer.train()

            print(iteration, result_dict)
            if iteration % 10 == 9:
                trainer.save("ckpnts")
