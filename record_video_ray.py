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
        checkpoint_no: int = 1,
        checkpoints_superdir: str = "ckpnts",
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

    def record(self, filename: str = None, geneity: str = "hom"):
        """Records a video of the loaded checkpoint.

        WARNING: is only really compatible with homogeneous training."""
        if not filename:
            filename = f"harvest-anim-{self.checkpoint_no}.mp4"
        fig, ax = plt.subplots()

        obs = self.reset()
        images = list()
        while True:
            if geneity == "hom":
                actions = self.trainer.compute_action(
                    observation=obs,
                    policy_id="walker",
                )
            elif geneity == "het":
                actions = {}
                for agent_id, _ in self.game.agents.items():
                    actions[agent_id] = self.trainer.compute_action(
                        observation=self.game.get_agent_obs(agent_id),
                        policy_id="walker"
                    )
            else:
                raise ValueError(f"bad geneity: {geneity}")

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
    checkpoint_no = 170
    recorder = HarvestRecorder(config, trainer, checkpoint_no, **defaults_ini)

    recorder.record(geneity="hom")


"""
TODO

- try `%run record_video_ray.py` with geneity="hom"
    + debug `trainer.compute_action`
- try training again with DQN
- parse args for geneity, checkpoint_no
- msg AK/QD
"""
