#!/usr/bin/env python3
from argparse import ArgumentParser

from cpr_reputation.environments import HarvestEnv
from cpr_reputation.utils import get_config, ArgParser

import ray
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents import ppo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey"])

args = ArgParser()
env_config, ray_config, heterogenous = get_config(args.ini)
ini = args.ini
if ini[-4:] == ".ini":
    ini = ini[:-4]

class HarvestRecorder(HarvestEnv):
    def __init__(
            self,
            config: dict,
            trainer,
            checkpoint_no: int = 1,
            checkpoints_superdir: str = "ckpnts",
            ini: str = ini,
            heterogenous: bool = heterogenous,
            **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.checkpoints_superdir = checkpoints_superdir
        self.checkpoint_no = checkpoint_no
        self.ini = ini
        self.heterogenous = heterogenous
        self.trainer = trainer
        try:
            trainer.restore(
                f"{self.checkpoints_superdir}/{self.ini}/checkpoint_{checkpoint_no}/checkpoint-{checkpoint_no}"
            )
            print(f"Loaded in checkpoint {checkpoint_no}")
        except AssertionError:
            print(
                f"Not loading any checkpoint at all, tried checkpoint {checkpoint_no}"
            )

    def record(self, filename: str = None):
        """Records a video of the loaded checkpoint.

        WARNING: is only really compatible with homogeneous training."""
        if filename is None:
            filename = f"harvest-anim-{self.ini}-{self.checkpoint_no}.mp4"
        fig, ax = plt.subplots()

        obs = self.reset()
        images = list()
        done = {"__all__": False}
        while not done["__all__"]:
            if self.heterogenous:
                actions = dict()
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agent_obs(agent_id),
                        policy_id=agent_id,
                    )
            else:
                actions = self.trainer.compute_action(
                    observation=obs, policy_id="walker",
                )

            obs, rewards, done, info = self.step(actions)

            board = self.game.board.copy()
            for _, agent in self.game.agents.items():
                board[agent.pos] = 2
            board[self.game.walls.astype(bool)] = 3
            im = ax.imshow(board, cmap=cmap)
            images.append([im])

        ani = animation.ArtistAnimation(
            fig, images, interval=50, blit=True, repeat_delay=10000
        )
        ani.save(filename)


if __name__ == "__main__":

    ray.init()

    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(config, **env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=ray_config,
    )

    checkpoint_no = args.checkpoint

    recorder = HarvestRecorder(
        ray_config, trainer, checkpoint_no, heterogenous=heterogenous, **env_config
    )

    recorder.record()


"""
TODO

- try `%run record_video_ray.py` with geneity="hom"
    + debug `trainer.compute_action`
- try training again with DQN
- parse args for geneity, checkpoint_no
- msg AK/QD
"""
