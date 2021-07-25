from cpr_reputation.board import Position, in_bounds
from cpr_reputation.environments import HarvestEnv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional

from cpr_reputation.utils import SHOOT

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
        self.checkpoint_path = checkpoint_path

        self.video_path = None

        try:
            trainer.restore(checkpoint_path)
            print(f"Loaded in checkpoint {checkpoint_path}")
        except AssertionError:
            print(
                f"Not loading any checkpoint at all, tried checkpoint {checkpoint_path}"
            )

    def record(self, filename: Optional[str] = None):
        """Records a video of the loaded checkpoint."""

        if filename is None:
            filename = self.checkpoint_path + ".mp4"

        self.video_path = filename
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
