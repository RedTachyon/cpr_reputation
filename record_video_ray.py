#!/usr/bin/env python3
# import os
from copy import deepcopy
import numpy as np
import yaml

import ray
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents import ppo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from gym.spaces import Box, Discrete

from cpr_reputation.board import Position, SHOOT, in_bounds
from cpr_reputation.environments import HarvestEnv
from learn import ArgParser

cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey", "red"])


class HarvestRecorder(HarvestEnv):
    def __init__(
        self,
        trainer,
        env_config: dict,
        heterogeneous: bool,
        checkpoint_path: str = None,
    ):
        super().__init__(env_config)

        self.trainer = trainer
        self.heterogeneous = heterogeneous

        try:
            trainer.restore(checkpoint_path)
            print(f"Loaded in checkpoint {checkpoint_path}")
        except AssertionError:
            print(
                f"Not loading any checkpoint at all, tried checkpoint {checkpoint_path}"
            )

    def record(self, filename: str = None):
        """Records a video of the loaded checkpoint."""

        if filename is None:
            filename = checkpoint_path + ".mp4"
        fig, ax = plt.subplots()

        images = list()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = dict()
            if self.heterogeneous:
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agents_obs()[agent_id],
                        policy_id=agent_id,
                    )
            else:
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agents_obs()[agent_id],
                        policy_id="walker",
                    )

            obs, rewards, done, info = self.step(actions)

            board = self.game.board.copy()
            for agent_id, agent in self.game.agents.items():
                board[agent.pos] = 2
                if actions[agent_id] == SHOOT:
                    (
                        (bound1_i, bound1_j),
                        (bound2_i, bound2_j),
                    ) = self.game.get_beam_bounds(agent_id)
                    if bound1_i > bound2_i:
                        inc_i = -1
                    else:
                        inc_i = 1
                    if bound1_j > bound2_j:
                        inc_j = -1
                    else:
                        inc_j = 1
                    size = Position(*board.shape)
                    for i in range(bound1_i, bound2_i, inc_i):
                        for j in range(bound1_j, bound2_j, inc_j):
                            if in_bounds(Position(i, j), size) and board[i, j] == 0:
                                board[i, j] = 4
            board[self.game.walls.astype(bool)] = 3
            board[0, 0] = 4
            im = ax.imshow(board, cmap=cmap)
            images.append([im])

        ani = animation.ArtistAnimation(
            fig, images, interval=50, blit=True, repeat_delay=10000
        )
        ani.save(filename)
        print(f"Successfully wrote {filename}")


if __name__ == "__main__":

    args = ArgParser()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    env_config = config["env_config"]
    ray_config = config["ray_config"]
    run_config = config["run_config"]

    # copied from learn.py
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

    checkpoint_path = args.checkpoint_path

    ray.init()
    register_env("CPRHarvestEnv-v0", lambda ray_config: HarvestEnv(env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=ray_config,
    )

    recorder = HarvestRecorder(
        trainer, env_config, run_config["heterogeneous"], checkpoint_path
    )

    recorder.record()
