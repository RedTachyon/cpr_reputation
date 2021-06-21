import gym
import numpy as np
from typing import Dict, Any, Optional

from gym.spaces import Box, Discrete
from ray.rllib.env import MultiAgentEnv as RayMultiAgentEnv

from cpr_reputation.board import HarvestGame, Position, in_bounds
from cpr_reputation.utils import SHOOT

import matplotlib as mpl
import matplotlib.pyplot as plt


class HarvestEnv(RayMultiAgentEnv, gym.Env):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        # self.config = ray_config
        self.game = HarvestGame(**env_config)

        self.observation_space = Box(
            0, 1, (self.game.sight_dist, 2 * self.game.sight_width + 1, 4), np.float32,
        )

        self.action_space = Discrete(8)

        self.original_board = np.array([], dtype=np.float32)

        self.last_actions = {}

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        self.last_actions = {}

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
        self.last_actions = actions

        return obs, rewards, done, info

    def render(self, mode: str = "rgb"):
        # if fig:
        #     ax = fig.gca()
        # else:
        #     fig, ax = plt.subplots()
        # print("RENDERING START")
        cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey", "red"])

        board = self.game.board.copy()
        actions = self.last_actions

        for agent_id, agent in self.game.agents.items():
            # Copied from record_video_ray.py, not sure if it works exactly
            board[agent.pos] = 2
            if actions.get(agent_id, None) == SHOOT:
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
        board = board.repeat(10, axis=0).repeat(
            10, axis=1
        )  # Upsample so that each cell is 10 pixels
        board = cmap(board)
        board = (board * 255).astype(np.uint8)
        # print("RENDERING END")
        # board[0, 0] = 4
        return board


class SingleHarvestEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        # self.config = ray_config
        env_config["num_agents"] = 1
        self.game = HarvestGame(**env_config)

        self.agent_id = list(self.game.agents.keys())[0]

        self.observation_space = Box(
            0, 1, (self.game.sight_dist, 2 * self.game.sight_width + 1, 4), np.float32,
        )

        self.action_space = Discrete(8)

        self.original_board = np.array([], dtype=np.float32)

        self.last_actions = {}

    def reset(self) -> np.ndarray:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        self.last_actions = {}

        obs = {
            agent_id: self.game.get_agent_obs(agent_id)
            for agent_id, _ in self.game.agents.items()
        }

        self.agent_id = list(self.game.agents.keys())[0]

        return obs[self.agent_id]

    def step(self, action: int):

        actions = {self.agent_id: action}
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
        self.last_actions = actions

        return obs[self.agent_id], rewards[self.agent_id], done[self.agent_id], info

    def render(self, mode: str = "rgb"):
        # if fig:
        #     ax = fig.gca()
        # else:
        #     fig, ax = plt.subplots()
        # print("RENDERING START")
        cmap = mpl.colors.ListedColormap(["brown", "green", "blue", "grey", "red"])

        board = self.game.board.copy()
        actions = self.last_actions

        for agent_id, agent in self.game.agents.items():
            # Copied from record_video_ray.py, not sure if it works exactly
            board[agent.pos] = 2
            if actions.get(agent_id, None) == SHOOT:
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
        board = board.repeat(10, axis=0).repeat(
            10, axis=1
        )  # Upsample so that each cell is 10 pixels
        board = cmap(board)
        board = (board * 255).astype(np.uint8)
        # print("RENDERING END")
        # board[0, 0] = 4
        return board
