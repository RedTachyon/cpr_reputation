from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from scipy.ndimage import convolve

import matplotlib as mpl
import matplotlib.pyplot as plt

GO_FORWARD = 0
GO_BACKWARD = 1
GO_LEFT = 2
GO_RIGHT = 3
ROT_LEFT = 4
ROT_RIGHT = 5
SHOOT = 6
NOOP = 7

# Kernel to compute the number of surrounding apples - not the same as in DM paper
NEIGHBOR_KERNEL = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

DIRECTIONS = [
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
]

# noinspection PyUnresolvedReferences
cmap = mpl.colors.ListedColormap(['white', 'red', 'green'])

Board = np.ndarray
Position = Tuple[int, int]


@dataclass
class Walker:
    pos: Position
    rot: int
    frozen: int = 0
    #
    # def __repr__(self) -> str:
    #     return f"Agent(pos={self.pos}, frozen={self.frozen > 0})"


def in_bounds(pos: Position,
              size: Position) -> bool:
    """Checks whether the position fits within the board of a given size"""
    (x, y) = pos
    (w, h) = size
    return 0 <= x < w and 0 <= y < h


def get_neighbors(pos: Position,
                  size: Position,
                  radius=1) -> List[Position]:
    """Get a list of neighboring positions (including self) that are still within the boundaries of the board"""
    increments = [(x, y) for y in range(size[0]) for x in range(size[1]) if abs(x) + abs(y) <= radius]

    (x, y) = pos
    return [(x + dx, y + dy) for (dx, dy) in increments if in_bounds((x + dx, y + dy), size)]


# TODO representing `pos` as (row, col) instead of (x, y) makes array processing easier
def create_board(size: Position,
                 apples: List[Position]) -> Board:
    """Creates an empty board with a number of cross-shaped apple patterns"""
    board = np.zeros(size, dtype=np.int8)

    for pos in apples:
        for (x, y) in get_neighbors(pos, size, radius=1):
            board[x, y] = 1

    return board


def random_board(size: Position,
                 prob: float = 0.1) -> Board:
    """Creates a board with each square having a probability of containing an apple"""
    prob_map = np.random.rand(*size)
    return (prob_map < prob).astype(np.int8)


def initial_position(i: int, total: int) -> Position:
    """
    Finds a starting position for an i-th agent with $total agents overall.
    The idea is that they're aligned in a square in the (0, 0) corner.
    The size of the square is determined by the number of agents.
    For example, with 10 agents, they will be in a 4x4 square, with 6 spots unfilled.
    With 16 agents, they're in a full 4x4 square.
    """
    layout_base = int(np.ceil(np.sqrt(total)))
    idx_map = np.arange(layout_base ** 2).reshape(layout_base, layout_base)
    (x, y) = np.where(idx_map == i)
    x, y = x[0], y[0]
    return x, y


class HarvestGame:
    def __init__(self,
                 num_agents: int,
                 size: Position,
                 init_prob: float = 0.1,
                 regen_prob: float = 0.01,
                 sight_width: Optional[int] = 10,  # default values from DM paper
                 sight_dist: Optional[int] = 20
                 ):

        self.num_agents = num_agents
        self.size = size
        self.init_prob = init_prob
        self.regen_prob = regen_prob
        self.sight_width = sight_width
        self.sight_dist = sight_dist

        self.board = np.array([[]])
        self.agents = {}
        self.reputation = {}

        self.reset()

    def reset(self):
        self.board = random_board(self.size, self.init_prob)
        self.agents = {
            f"Agent{i}": Walker(pos=initial_position(i, self.num_agents), rot=1+np.random.randint(2))
            for i in range(self.num_agents)
        }

        self.reputation = {
            f"Agent{i}": 0
            for i in range(self.num_agents)
        }

        # don't spawn apple in cells which already contain agents
        for name, agent in self.agents.items():
            self.board[agent.pos] = 0

    def render(self, ax: plt.Axes):
        """Writes the image to a pyplot axes"""
        board = self.board[:]
        for name, agent in self.agents.items():
            board[agent.pos] = 2
        ax.cla()
        ax.imshow(board, cmap=cmap)

    def is_free(self, pos: Position) -> bool:
        """Checks whether the position is within bounds, and unoccupied"""
        if not in_bounds(pos, self.size):
            return False

        for agent_id, agent in self.agents.items():
            if agent.pos == pos:
                return False

        return True

    def _move_agent(self, agent_id: str, new_pos: Position):
        """Maybe moves an agent to a new position"""
        agent = self.agents[agent_id]
        if self.is_free(new_pos):
            agent.pos = new_pos

    def _set_rotation(self, agent_id: str, rot: int):
        """Debugging/testing method"""
        agent = self.agents[agent_id]
        assert rot in (0, 1, 2, 3)
        agent.rot = rot

    def _rotate_agent(self, agent_id: str, direction: int):
        """Rotates an agent left or right"""
        agent = self.agents[agent_id]
        assert direction in [-1, 1]
        # -1: rotate left
        #  1: rotate right
        agent.rot = (agent.rot + direction) % 4

    def regenerate_apples(self):
        """
        Performs a single update of stochastically regenerating apples.
        It works by computing a convolution counting neighboring apples, scaling this with a base probability,
        and using that as the probability of an apple being generated there.
        Needs to be adapted to the version in the DM paper. Also might need an agent-based mask.
        """
        kernel = NEIGHBOR_KERNEL
        neighbor_map: np.ndarray = convolve(self.board, kernel, mode='constant')
        prob_map = neighbor_map * self.regen_prob

        rand = np.random.rand(*neighbor_map.shape)
        regen_map = rand < prob_map
        updated_board = np.clip(self.board + regen_map, 0, 1)

        # don't spawn apples in cells which already contain agents
        for name, agent in self.agents.items():
            updated_board[agent.pos] = 0

        self.board = updated_board

    def regenerate_apples_dm(self):
        """
        Stochastically respawn apples based on number of neighbors:

        prob | n_neighbors
        ------------------
        0    | L = 0
        0.01 | L = 1 or 2
        0.05 | L = 3 or 4
        0.1  | L > 4
        """
        prob_table = defaultdict(lambda: 0.1)
        prob_table[0] = 0.0
        prob_table[1] = 0.01
        prob_table[2] = 0.01
        prob_table[3] = 0.05
        prob_table[4] = 0.05

        prob_map = np.zeros(self.size)
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                pos = (x, y)
                neighboring_pos = get_neighbors(pos, self.size, radius=2)
                neighboring_apples = sum([self.board[y][x] for (x, y) in neighboring_pos])
                prob_map[y][x] = prob_table[neighboring_apples]

        rand = np.random.rand(prob_map.shape)
        regen_map = rand < prob_map
        updated_board = np.clip(self.board + regen_map, 0, 1)

        # don't spawn apples in cells which already contain agents
        for name, agent in self.agents.items():
            updated_board[agent.pos] = 0

        self.board = updated_board

    def process_action(self, agent_id: str, action: int) -> float:
        """Processes a single action for a single agent"""
        agent = self.agents[agent_id]
        (x, y), rot = agent.pos, agent.rot
        if agent.frozen > 0:  # if the agent is frozen, should we avoid updating the gradient?
            agent.frozen -= 1
            return 0.
        if action == GO_FORWARD:
            # Go forward
            (dx, dy) = DIRECTIONS[rot]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == GO_BACKWARD:
            # Go backward
            (dx, dy) = DIRECTIONS[rot]
            new_pos = (x - dx, y - dy)
            self._move_agent(agent, new_pos)
        elif action == GO_LEFT:
            # Go left
            (dx, dy) = DIRECTIONS[(rot - 1) % 4]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == GO_RIGHT:
            # Go right
            (dx, dy) = DIRECTIONS[(rot + 1) % 4]
            new_pos = (x + dx, y + dy)
            self._move_agent(agent, new_pos)
        elif action == ROT_LEFT:
            # Rotate left
            self._rotate_agent(agent, -1)
            new_pos = (x, y)
        elif action == ROT_RIGHT:
            # Rotate right
            self._rotate_agent(agent, 1)
            new_pos = (x, y)
        elif action == SHOOT:
            # Shoot a beam
            affected_agents = self.get_affected_agents(agent_id)
            for _agent in affected_agents:
                _agent.frozen = 25
            self.reputation[agent_id] += 1  # why does reputation increase for each opponent zapped?
            new_pos = (x, y)
            # see notebook for weird results
        elif action == 7:
            # No-op
            new_pos = (x, y)

        new_x, new_y = new_pos[0], new_pos[1]
        if self.board[new_y][new_x]:  # apple in new cell
            self.board[new_y][new_x] = 0
            return 1
        else:  # no apple in new cell
            return 0

    def get_affected_agents(self, agent_id: str) -> List[Walker]:
        """Returns a list of agents caught in the ray"""
        (x0, y0), (x1, y1) = self.get_observable_window(agent_id)
        return [ag for _, ag in self.agents.items()
                if x0 <= ag.pos[0] <= x1
                and y0 <= ag.pos[1] <= y1]

    def get_observable_window(self, agent_id: str) -> List[Position]:
        """Returns indices of the (top left, bottom right) (inclusive) boundaries of an agent's vision."""
        x_max = self.size[1] - 1
        y_max = self.size[0] - 1
        agent = self.agents[agent_id]
        x, y = agent.pos
        if agent.rot == 0:  # facing north
            bound_left = max(0, x - self.sight_width)
            bound_right = min(x_max, x + self.sight_width)
            bound_up = max(0, y - self.sight_dist)
            bound_down = y
        elif agent.rot == 1:  # facing east
            bound_left = x
            bound_right = min(x_max, x + self.sight_dist)
            bound_up = max(0, y - self.sight_width)
            bound_down = min(y_max, y + self.sight_width)
        elif agent.rot == 2:  # facing south
            bound_left = max(0, x - self.sight_width)
            bound_right = min(x_max, x + self.sight_width)
            bound_up = y
            bound_down = min(y_max, y + self.sight_dist)
        elif agent.rot == 3:  # facing west
            bound_left = max(0, x - self.sight_dist)
            bound_right = x
            bound_up = max(0, y - self.sight_width)
            bound_down = min(y_max, y + self.sight_width)
        else:
            raise ValueError("agent.rot must be % 4")
        return [(bound_left, bound_up), (bound_right, bound_down)]

    def get_agent_obs(self, agent_id: str) -> np.ndarray:
        """The partial observability of the environment.
        TODO: this should have been TDD, so the TODO is to write some simple test cases and make sure they pass.
        """
        (x0, y0), (x1, y1) = self.get_observable_window(agent_id)
        return self.board[y0:y1+1][x0:x1+1]  # kind of ugly; feel free to change `get_observable_window` to be non-inclusive for the bottom-right boundary


class MultiAgentEnv:  # Placeholder
    pass


class HarvestEnv(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        self.game = HarvestGame(*args, **kwargs)
        self.time = 0

    def reset(self) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.time = 0
        return {
            agent_id: self.game.get_agent_obs(agent_id) for agent_id in self.game.agents
        }

    def step(self, actions: Dict[str, int]):
        rewards = {agent_id: 0. for agent_id in self.game.agents}
        for (agent_id, action) in actions.items():
            reward = self.game.process_action(agent_id, action)
            rewards[agent_id] += reward

        self.game.regenerate_apples()

        obs = {
            agent_id: self.game.get_agent_obs(agent_id) for agent_id in self.game.agents
        }

        done = {
            agent_id: self.time > 1000 for agent_id in self.game.agents
        }
        done["__all__"] = self.time > 1000  # Required for rllib (I think)
        info = {}
        self.time += 1

        return obs, rewards, done, info