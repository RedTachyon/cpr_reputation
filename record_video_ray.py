#!/usr/bin/env python3

from learning_ray import trainer, config, defaults_ini
from cpr_reputation.environments import HarvestEnv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey"])


class HarvestRecorder(HarvestEnv):
    def __init__(
        self,
        config: dict,
        trainer,
        checkpoints_superdir: str = "ckpnts",
        checkpoint_no: int = 1,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.checkpoints_superdir = checkpoints_superdir
        self.trainer = trainer
        try:
            trainer.restore(
                f"{checkpoints_superdir}/checkpoint_{checkpoint_no}/checkpoint-{checkpoint_no}"
            )
            print(f"Loaded in checkpoint {checkpoint_no}")
        except AssertionError:
            print(
                f"Not loading any checkpoint at all, tried checkpoint {checkpoint_no}"
            )
        self.checkpoint_no = checkpoint_no

    def record(self, filename: str = None):
        """Records a video of the loaded checkpoint.

        WARNING: is only really compatible with homogeneous training."""
        if not filename:
            filename = f"harvest-anim-{self.checkpoint_no}.mp4"
        fig, ax = plt.subplots()

        obs = self.reset()
        images = list()
        while True:
            actions = self.trainer.compute_actions(
                obs,
                policy_id="walker",  # This is the part that's only really compatible with homogeneity.
            )
            obs, rewards, done, info = self.step(actions)

            board = self.game.board.copy()
            for _, agent in self.game.agents.items():
                board[agent.pos] = 2
            board[self.game.walls.astype(bool)] = 3
            im = ax.imshow(board, cmap=cmap)
            images.append([im])

            if done["__all__"]:
                break
        ani = animation.ArtistAnimation(
            fig, images, interval=50, blit=True, repeat_delay=10000
        )
        ani.save(filename)


if __name__ == "__main__":
    recorder = HarvestRecorder(config, trainer, **defaults_ini)

    recorder.record()
