import numpy as np
from typing import Dict, Any

from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv as RayMultiAgentEnv

from cpr_reputation.board import HarvestGame


class HarvestEnv(RayMultiAgentEnv):
    def __init__(self, env_config: Dict[str, Any], ray_config: Dict[str, Any]):
        super().__init__()
        self.config = ray_config
        self.game = HarvestGame(**env_config)

        self.observation_space = Box(
            0, 1, (self.game.sight_dist, 2 * self.game.sight_width + 1, 4), np.float32,
        )

        self.action_space = Discrete(8)

        self.original_board = np.array([], dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        return {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, int]):

        rewards = self.game.step(actions)

        # get observations, done, info
        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        isdone = self.game.time > 1000 or self.game.board.sum() == 0
        done = {agent_id: isdone for agent_id, _ in self.game.agents.items()}
        done["__all__"] = isdone

        info = {}

        return obs, rewards, done, info
