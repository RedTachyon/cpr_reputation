#!/usr/bin/env python3

import ray
from ray.rllib.agents import ppo
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.utils import get_config, ArgParser

args = ArgParser()

env_config, ray_config, run_config = get_config(args.ini)

if __name__ == "__main__":

    ray.init()
    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(config, **env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        # logger_creator=lambda cfg: UnifiedLogger(cfg, f"log/{args.ini}"),
        config=ray_config,
    )

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        checkpoint_dir = (
            f"ckpnts/{args.ini}/checkpoint_{checkpoint}/checkpoint-{checkpoint}"
        )
        print(f"Pulling checkpoint from {checkpoint_dir}")
        trainer.restore(checkpoint_dir)
    else:
        print("Not pulling from a checkpoint")
        checkpoint = 1

    for iteration in range(checkpoint, run_config["num_iterations"]):
        result_dict = trainer.train()

        if run_config["verbose"]:
            print(f"Iteration: {iteration}", end="\t")  # , result_dict)
        if iteration % 10 == 0:
            trainer.save(f"ckpnts/{args.ini}")
