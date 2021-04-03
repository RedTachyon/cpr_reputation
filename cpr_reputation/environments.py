import numpy as np
import random
from typing import Dict

from ray.rllib.env import MultiAgentEnv as RayMultiAgentEnv

from cpr_reputation.board import HarvestGame, regenerate_apples, SHOOT


class HarvestEnv(RayMultiAgentEnv):
    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        self.config = config
        self.time = 0
        self.game = HarvestGame(**kwargs)
        # self.original_board = np.copy(self.game.board)

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        self.time = 0
        return {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, int]):
        # process actions and rewards
        rewards = {agent_id: 0.0 for agent_id in self.game.agents}

        action_pairs = list(actions.items())
        random.shuffle(action_pairs)
        for (agent_id, action) in action_pairs:
            reward = self.game.process_action(agent_id, action)
            rewards[agent_id] += reward

        # update environment
        self.game.board = regenerate_apples(self.game.board)
        self.game.board = self.game.board * self.original_board
        self.time += 1

        # get observations, done, info
        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        isdone = self.time > 1000 or self.game.board.sum() == 0
        done = {agent_id: isdone for agent_id, _ in
                self.game.agents.items()}
        done["__all__"] = isdone

        num_shots = sum(1 for key, action in actions.items() if action == SHOOT)

        #info = {"m_shots": num_shots}
        info = dict()

        info = {}

        return obs, rewards, done, info
