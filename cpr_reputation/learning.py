from typing import Dict
import numpy as np
from ray.rllib.env import MultiAgentEnv
from cpr_reputation.board import HarvestGame


class HarvestEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        self.config = config
        self.time = 0
        self.game = HarvestGame(**kwargs)

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.time = 0
        return {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, int]):
        rewards = {agent_id: 0.0 for agent_id in self.game.agents}
        for (agent_id, action) in actions.items():
            reward = self.game.process_action(agent_id, action)
            rewards[agent_id] += reward

        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        done = {agent_id: self.time > 1000 for agent_id, _ in self.game.agents.items()}
        done["__all__"] = self.time > 1000  # Required for rllib (I think)
        info = {}
        self.time += 1

        return obs, rewards, done, info
