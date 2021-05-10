#!/usr/bin/env python3

from cpr_reputation.board import Position, SHOOT, in_bounds
from cpr_reputation.environments import HarvestEnv
from cpr_reputation.utils import get_config, ArgParser

import ray
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents import ppo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey", "red"])

args = ArgParser()
env_config, ray_config, run_config = get_config(args.ini)
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
        heterogenous: bool = run_config["heterogeneous"],
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

        images = list()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = dict()
            if self.heterogenous:
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agent_obs(agent_id),
                        policy_id=agent_id,
                    )
            else:
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agent_obs(agent_id),
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

    ray.init()
    register_env("CPRHarvestEnv-v0", lambda config: HarvestEnv(config, **env_config))

    trainer = ppo.PPOTrainer(
        env="CPRHarvestEnv-v0",
        logger_creator=lambda cfg: UnifiedLogger(cfg, "log"),
        config=ray_config,
    )

    checkpoint_no = args.checkpoint

    recorder = HarvestRecorder(ray_config, trainer, checkpoint_no, **env_config)

    recorder.record()
