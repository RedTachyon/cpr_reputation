import io
from argparse import ArgumentParser

import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.integration.wandb import WandbLoggerCallback
from gym.spaces import Discrete, Box

# import wandb

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.utils import retrieve_checkpoint


def make_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--geneity",
        default="hom",
        help="select `hom` for homogenous training, select `het` for heterogenous training",
    )
    parser.add_argument(
        "--train-fn-name",
        default=None,
        help="name the training function. this is important for naming directories in ~/ray_results",
    )
    return parser


parser = make_arguments()
args = parser.parse_args()

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
        0.0,
        1.0,
        (defaults_ini["sight_dist"], 2 * defaults_ini["sight_width"] + 1, 3),
        np.float32,
    ),  # obs
    Discrete(8),  # action
    dict(),
)

if args.geneity == "hom":
    multiagent: dict = {
        "policies": {"walker": walker1},
        "policy_mapping_fn": lambda agent_id: "walker",
    }
elif args.geneity == "het":
    walkers = {f"Agent{k}": walker1 for k in range(defaults_ini["num_agents"])}
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
}

with io.open("WANDB_TOKEN", "r") as file:
    WANDB_TOKEN = file.read()


def train_fn(dummy, cnfg=config):

    trainer = ppo.PPOTrainer(
        env="harvest",
        config=cnfg,
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
    )

    results = trainer.train()
    print(results)

    return results


if args.train_fn_name is not None:
    train_fn.__name__ = args.train_fn_name

if __name__ == "__main__":
    register_env("harvest", lambda config: HarvestEnv(config, **defaults_ini))

    ray.init()

    while True:
        tune.run(
            train_fn,
            config=dict(),
            callbacks=[
                WandbLoggerCallback(
                    project="cpr_reputation_gridworld",
                    api_key_file="WANDB_TOKEN",
                    log_config=True,
                )
            ],
            checkpoint_at_end=True,
            restore=retrieve_checkpoint(prefix=train_fn.__name__),
        )
