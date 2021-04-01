import numpy as np
import random
from typing import Dict

from cpr_reputation.board import HarvestGame, SHOOT, regenerate_apples
from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv as RayMultiAgentEnv


class HarvestEnv(RayMultiAgentEnv):
    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        self.config = config
        self.time = 0
        self.game = HarvestGame(**kwargs)
        self.original_board = np.copy(self.game.board)

        self.observation_space = Box(
            0., 1., (self.game.sight_dist * (self.game.sight_width * 2 + 1),)
        )
        self.action_space = Discrete(8)

    def reset(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        self.time = 0
        return {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, int]):
        rewards = {agent_id: 0.0 for agent_id in self.game.agents}
        action_pairs = list(actions.items())
        random.shuffle(action_pairs)
        for (agent_id, action) in action_pairs:
            reward = self.game.process_action(agent_id, action)
            rewards[agent_id] += reward

        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        done = {agent_id: self.time > 1000 for agent_id, _ in
                self.game.agents.items()}
        # done["__all__"] = self.time > 1000  # Required for rllib (I think)

        num_shots = sum(1 for key, action in actions.items() if action == SHOOT)
        info = {"m_shots": num_shots}
        self.game.board = regenerate_apples(self.game.board)
        self.game.board = self.game.board * self.original_board

        self.time += 1

        return obs, rewards, done, info

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
