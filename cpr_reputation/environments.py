import numpy as np
from typing import Dict

from ray.rllib.env import MultiAgentEnv as RayMultiAgentEnv

from cpr_reputation.board import HarvestGame


class HarvestEnv(RayMultiAgentEnv):
    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        self.config = config
        self.game = HarvestGame(**kwargs)
        # self.original_board = np.copy(self.game.board)

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        return {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, int]):
        # process actions and rewards
        # if self.game.time < 50:
        #     actions = {
        #         agent_id: NOOP if action == SHOOT else action
        #         for (agent_id, action) in actions.items()
        #     }

        rewards = self.game.step(actions)

        # get observations, done, info
        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        isdone = self.game.time > 1000 or self.game.board.sum() == 0
        done = {agent_id: isdone for agent_id, _ in self.game.agents.items()}
        done["__all__"] = isdone

        info = dict()

        info = {}

        return obs, rewards, done, info
